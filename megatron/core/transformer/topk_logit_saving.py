# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in saving of the top-k output logits of every token, for later distillation.

Hooks the same two points as ``output_logit_logging`` (the vocab-parallel cross entropy): the
fp32 logits are still unshifted when ``prepare`` runs, and ``record`` runs once the global
log-sum-exp is known. Per valid token the file keeps

* ``q``      uint8 [k]: ``(top1 - logit_j) / step`` rounded, ``step = clip / 255``, so a logit is
             recovered to within ``clip / 510`` (clipped ``clip`` below the top-1 logit);
* ``idx_ef`` uint8 [bytes]: the k token ids, sorted ascending and stored with Elias-Fano coding
             (about 1.44 bytes per id at k = 256 and a 200k vocabulary, against 3 for plain 24-bit
             ids); ``q`` is in the same ascending-id order;
* ``top1``   float32: the largest logit;
* ``lse``    float32: log-sum-exp over the full vocabulary (log-prob = logit - lse);
* ``labels`` int32: the target token (for every token, loss-masked or not).

Every token of the microbatch is saved, including loss-masked ones (goldfish drops, masked ids),
so that a rank's rows form whole, contiguous sequences. Three more per-token fields locate a row:

* ``position``    int32: the position inside its sequence;
* ``sample``      int32: ``microbatch * batch + index in the microbatch``, counted per rank and step;
* ``loss_valid``  uint8: 1 if the token counts in the training loss, 0 if it is loss-masked.

Only the output stage's TP rank 0 stores; every TP rank takes part in the merge. No full logits
survive the forward, and the per-microbatch results are moved to the host immediately.
"""

import atexit
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps
import math
import os
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

_active = False
_in_forward = False
_in_loss = False
_loss_mask: Optional[torch.Tensor] = None
_k = 256
_clip = 16.0
_pending: List[Dict[str, torch.Tensor]] = []
_vocab = 0
_batch = 1
_micro = 0
_writer: Optional[ThreadPoolExecutor] = None
_futures: list = []
# Rows per top-k tile: bounds the sort workspace, not the (already allocated) logits.
_TILE_ELEMENTS = 33554432


def should_save(step: int, interval: int, window_iterations: int) -> bool:
    """True for the first ``window_iterations`` steps of every ``interval`` steps (1-based)."""
    if interval < 0 or window_iterations < 0:
        raise ValueError("save-topk-logits interval and window must be nonnegative")
    return interval > 0 and window_iterations > 0 and step > 0 and (step - 1) % interval < window_iterations


def window_iterations(tokens_per_window: int, tokens_per_iteration: int, interval: int) -> int:
    """Whole iterations needed to cover ``tokens_per_window`` tokens, capped at ``interval``."""
    if tokens_per_window <= 0 or tokens_per_iteration <= 0:
        return 0
    return min(interval, math.ceil(tokens_per_window / tokens_per_iteration))


def reset_attempt() -> None:
    """Drop results of a rejected attempt without disabling this step."""
    global _micro
    _pending.clear()
    _micro = 0


def set_active(active: bool, k: int = 256, clip: float = 16.0) -> None:
    global _active, _in_forward, _in_loss, _loss_mask, _k, _clip
    if active and (k < 1 or clip <= 0):
        raise ValueError("save-topk-logits-k must be >= 1 and save-topk-logits-clip > 0")
    _active = bool(active)
    _in_forward = _in_loss = False
    _loss_mask = None
    _k, _clip = int(k), float(clip)
    reset_attempt()


def is_active() -> bool:
    return _active


@contextmanager
def forward_scope() -> Iterator[None]:
    global _in_forward
    previous = _in_forward
    _in_forward = _active
    try:
        yield
    finally:
        _in_forward = previous


def wrap_forward_step(forward_step: Callable) -> Callable:
    @wraps(forward_step)
    def wrapped(*args, **kwargs):
        with forward_scope():
            return forward_step(*args, **kwargs)

    return wrapped


@contextmanager
def loss_scope(loss_mask: Optional[torch.Tensor], *, enabled: bool = True) -> Iterator[None]:
    """Observe the primary GPT loss only, excluding MTP and unrelated CE calls."""
    global _in_loss, _loss_mask
    previous = (_in_loss, _loss_mask)
    _in_loss = _active and _in_forward and enabled
    _loss_mask = loss_mask if _in_loss else None
    try:
        yield
    finally:
        _in_loss, _loss_mask = previous


def local_topk(logits_2d: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k values (descending) and indices of every row, in bounded-size row tiles."""
    rows, vocab = logits_2d.shape
    if vocab < k:
        raise ValueError(f"vocab shard ({vocab}) is smaller than save-topk-logits-k ({k})")
    tile = max(1, _TILE_ELEMENTS // vocab)
    values = torch.empty((rows, k), dtype=torch.float32, device=logits_2d.device)
    indices = torch.empty((rows, k), dtype=torch.int64, device=logits_2d.device)
    for start in range(0, rows, tile):
        end = min(start + tile, rows)
        top = torch.topk(logits_2d[start:end], k, dim=-1, sorted=True)
        values[start:end] = top.values.float()
        indices[start:end] = top.indices
    return values, indices


def merge_topk(
    values: torch.Tensor, indices: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge concatenated per-shard candidates [rows, shards * k] into the global top-k."""
    top = torch.topk(values, k, dim=-1, sorted=True)
    return top.values, torch.gather(indices, 1, top.indices)


def prepare(logits: torch.Tensor, target: torch.Tensor, tp_group) -> Optional[dict]:
    """Global top-k of the unshifted logits. Every TP rank takes part; only rank 0 returns it."""
    if not (_active and _in_forward and _in_loss):
        return None
    group_size = tp_group.size() if tp_group is not None else 1
    group_rank = tp_group.rank() if tp_group is not None else 0
    with torch.no_grad():
        vocab = logits.shape[-1]
        flat = logits.detach().reshape(-1, vocab)
        values, indices = local_topk(flat, _k)
        indices += group_rank * vocab  # the shard's first global id (equal vocab shards)
        if group_size > 1:
            gathered_values = [torch.empty_like(values) for _ in range(group_size)]
            gathered_indices = [torch.empty_like(indices) for _ in range(group_size)]
            torch.distributed.all_gather(gathered_values, values, group=tp_group)
            torch.distributed.all_gather(gathered_indices, indices, group=tp_group)
            values, indices = merge_topk(
                torch.cat(gathered_values, dim=1), torch.cat(gathered_indices, dim=1), _k
            )
        if group_rank != 0:
            return None
        lead = tuple(logits.shape[:-1])  # [sequence, batch]
        return {
            "values": values,
            "indices": indices,
            "labels": target.reshape(-1),
            "vocab": vocab * group_size,
            "seq": lead[0],
            "batch": lead[1] if len(lead) > 1 else 1,
        }


def quantize(values: torch.Tensor, clip: float) -> torch.Tensor:
    """uint8 distance below the top-1 logit, in steps of ``clip / 255``, clipped at ``clip``."""
    step = clip / 255.0
    gap = (values[:, :1] - values).clamp(0.0, clip)
    return torch.round(gap / step).to(torch.uint8)


def ef_layout(vocab: int, k: int) -> Tuple[int, int, int]:
    """Elias-Fano parameters: (low bits per id, bytes of the low part, bytes of the upper part).

    Each id is split into ``low_bits`` low bits, stored as is, and a high part, stored as a
    unary-coded bit vector of ``k + (vocab - 1 >> low_bits) + 1`` bits.
    """
    low_bits = max(0, (vocab // k).bit_length() - 1)
    upper_bits = k + ((vocab - 1) >> low_bits) + 1
    return low_bits, (k * low_bits + 7) // 8, (upper_bits + 7) // 8


def _pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """[rows, bits] of 0/1 -> uint8 [rows, ceil(bits / 8)], least-significant bit first."""
    rows, width = bits.shape
    pad = (-width) % 8
    if pad:
        bits = torch.nn.functional.pad(bits, (0, pad))
    weights = (1 << torch.arange(8, device=bits.device)).to(torch.int32)
    return (bits.reshape(rows, -1, 8).to(torch.int32) * weights).sum(-1).to(torch.uint8)


def encode_indices_ef(indices: torch.Tensor, vocab: int) -> torch.Tensor:
    """Strictly ascending ids [rows, k] -> Elias-Fano bytes uint8 [rows, low + upper bytes]."""
    rows, k = indices.shape
    ids = indices.to(torch.int64)
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= vocab or bool((ids[:, 1:] <= ids[:, :-1]).any())):
        raise ValueError("Elias-Fano coding needs strictly ascending ids inside the vocabulary")
    low_bits, _, _ = ef_layout(vocab, k)
    if low_bits:
        shifts = torch.arange(low_bits, device=ids.device)
        low = ((ids & ((1 << low_bits) - 1))[..., None] >> shifts) & 1
        low_packed = _pack_bits(low.reshape(rows, k * low_bits))
    else:
        low_packed = torch.zeros((rows, 0), dtype=torch.uint8, device=ids.device)
    upper = torch.zeros((rows, k + ((vocab - 1) >> low_bits) + 1), dtype=torch.uint8, device=ids.device)
    upper.scatter_(1, (ids >> low_bits) + torch.arange(k, device=ids.device), 1)
    return torch.cat((low_packed, _pack_bits(upper)), dim=1)


def decode_indices_ef(packed: np.ndarray, vocab: int, k: int, chunk: int = 16384) -> np.ndarray:
    """Inverse of :func:`encode_indices_ef`: uint8 [rows, bytes] -> int64 ids [rows, k]."""
    low_bits, low_bytes, upper_bytes = ef_layout(vocab, k)
    out = np.empty((packed.shape[0], k), dtype=np.int64)
    for start in range(0, packed.shape[0], chunk):
        block = packed[start : start + chunk]
        rows = block.shape[0]
        if low_bits:
            bits = np.unpackbits(block[:, :low_bytes], axis=1, bitorder="little")[:, : k * low_bits]
            low = (bits.reshape(rows, k, low_bits).astype(np.int64) << np.arange(low_bits)).sum(-1)
        else:
            low = np.zeros((rows, k), dtype=np.int64)
        upper = np.unpackbits(block[:, low_bytes : low_bytes + upper_bytes], axis=1, bitorder="little")
        _, position = np.nonzero(upper)  # exactly k set bits per row, in row-major order
        high = position.reshape(rows, k) - np.arange(k)
        out[start : start + chunk] = (high << low_bits) | low
    return out


def record(
    prepared: Optional[dict], logits_max: torch.Tensor, sum_exp_logits: torch.Tensor
) -> None:
    """Quantize this microbatch's top-k and keep it on the host (all tokens, with a loss flag)."""
    if prepared is None or not (_active and _in_forward and _in_loss):
        return
    with torch.no_grad():
        values, indices, labels = prepared["values"], prepared["indices"], prepared["labels"]
        rows = values.shape[0]
        lse = logits_max.detach().reshape(-1) + sum_exp_logits.detach().reshape(-1).log()
        if lse.numel() != rows:
            raise ValueError("top-k logit rows do not match the cross-entropy rows")
        if _loss_mask is None:
            valid = torch.ones(rows, dtype=torch.bool, device=values.device)
        else:
            valid = (_loss_mask.transpose(0, 1).reshape(-1) > 0).to(values.device)
            if valid.numel() != rows:
                raise ValueError("loss mask must match [sequence, batch] cross-entropy rows")
        global _vocab, _batch, _micro
        _vocab, _batch = int(prepared["vocab"]), int(prepared["batch"])
        if int(prepared["seq"]) * _batch != rows:
            raise ValueError("top-k logit rows do not match [sequence, batch]")
        row = torch.arange(rows, device=values.device)
        position = (row // _batch).to(torch.int32)
        sample = (_micro * _batch + row % _batch).to(torch.int32)
        _micro += 1
        q = quantize(values, _clip)
        indices, order = torch.sort(indices, dim=1)  # ascending ids for the Elias-Fano coding
        _pending.append(
            {
                "q": q.gather(1, order).cpu(),
                "idx_ef": encode_indices_ef(indices, _vocab).cpu(),
                "top1": values[:, 0].float().cpu(),
                "lse": lse.float().cpu(),
                "labels": labels.to(torch.int32).cpu(),
                "position": position.cpu(),
                "sample": sample.cpu(),
                "loss_valid": valid.to(torch.uint8).cpu(),
            }
        )


def _write(path: str, arrays: Dict[str, np.ndarray]) -> None:
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def flush(step: int, out_dir: str, rank: int) -> Optional[str]:
    """Write this step's tokens as one file per rank (in the background) and clear the buffer."""
    if not _pending:
        return None
    arrays = {name: torch.cat([p[name] for p in _pending]).numpy() for name in _pending[0]}
    arrays.update(
        step=np.int64(step),
        k=np.int64(_k),
        vocab=np.int64(_vocab),
        batch=np.int64(_batch),
        clip=np.float32(_clip),
    )
    _pending.clear()
    step_dir = os.path.join(out_dir, f"iter_{step:07d}")
    os.makedirs(step_dir, exist_ok=True)
    path = os.path.join(step_dir, f"rank_{rank:05d}.npz")
    global _writer
    if _writer is None:
        _writer = ThreadPoolExecutor(max_workers=1)
        atexit.register(wait)
    _futures.append(_writer.submit(_write, path, arrays))
    return path


def wait() -> None:
    """Block until queued writes finish; raises the first write error."""
    while _futures:
        _futures.pop(0).result()


def decode(path: str) -> Dict[str, np.ndarray]:
    """Read a saved file back, ids ascending within each token: ``indices`` [T, k], ``logits``
    [T, k], ``lse`` [T], ``labels`` [T], and ``position``, ``sample``, ``loss_valid`` [T] (None for
    files written before those fields existed). A log-probability is ``logits - lse[:, None]``."""
    with np.load(path) as data:
        clip, k, vocab = float(data["clip"]), int(data["k"]), int(data["vocab"])
        logits = data["top1"][:, None] - data["q"].astype(np.float32) * (clip / 255.0)
        names = set(data.files)
        return {
            "indices": decode_indices_ef(data["idx_ef"], vocab, k),
            "logits": logits.astype(np.float32),
            "lse": data["lse"],
            "labels": data["labels"],
            "position": data["position"] if "position" in names else None,
            "sample": data["sample"] if "sample" in names else None,
            "loss_valid": data["loss_valid"].astype(bool) if "loss_valid" in names else None,
            "batch": int(data["batch"]) if "batch" in names else None,
            "step": int(data["step"]),
        }
