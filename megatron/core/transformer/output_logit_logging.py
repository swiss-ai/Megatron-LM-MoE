# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in output diagnostics using the native FP32 cross-entropy intermediates.

No full logits or hidden states survive the forward. Only per-token margins are
retained for an exact distributed percentile, plus scalar sums and maxima.
"""

from contextlib import contextmanager
from functools import wraps
import math
from typing import Callable, Dict, Iterator, Optional

import torch

_active = False
_in_forward = False
_in_loss = False
_loss_mask: Optional[torch.Tensor] = None
_sums: Optional[torch.Tensor] = None
_maxima: Optional[torch.Tensor] = None
_margins: list[torch.Tensor] = []
_CHUNK_ELEMENTS = 262144
# Larger vocabulary tiles avoid thousands of tiny launches per microbatch.
_MARGIN_CHUNK_ELEMENTS = 8388608  # 32 MiB FP32 scratch per tile
FP32_PRECISION_CLIFF = 24 * math.log(2.0)

# sums: token count, LSE sum, zero-loss count, hidden-token count, hidden-norm sum,
#       count of target-versus-rest margins above the theoretical FP32 cliff
# maxima: logit max, LSE max, target margin max, hidden-norm max, margin p99.9


def should_log(step: int, interval: int = 0) -> bool:
    if interval < 0:
        raise ValueError("output-logit-log-interval must be nonnegative")
    return interval > 0 and step > 0 and step % interval == 0


def reset_attempt() -> None:
    """Drop diagnostics from a rejected attempt, without disabling this step."""
    global _sums, _maxima
    _sums = _maxima = None
    _margins.clear()


def set_active(active: bool) -> None:
    global _active, _in_forward, _in_loss, _loss_mask
    _active = bool(active)
    _in_forward = _in_loss = False
    _loss_mask = None
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
    """Observe the primary GPT loss, excluding MTP and unrelated CE calls."""
    global _in_loss, _loss_mask
    previous = (_in_loss, _loss_mask)
    _in_loss = _active and _in_forward and enabled
    _loss_mask = loss_mask if _in_loss else None
    try:
        yield
    finally:
        _in_loss, _loss_mask = previous


def _size(group) -> int:
    return group.size() if group is not None else 1


def _rank(group) -> int:
    return group.rank() if group is not None else 0


def _initialize(device: torch.device) -> None:
    global _sums, _maxima
    if _sums is None:
        _sums = torch.zeros(6, dtype=torch.float64, device=device)
        _maxima = torch.full((5,), float("-inf"), dtype=torch.float64, device=device)


def _valid(shape: tuple, device: torch.device) -> torch.Tensor:
    if _loss_mask is None:
        return torch.ones(shape, device=device, dtype=torch.bool)
    mask = _loss_mask.transpose(0, 1)
    if tuple(mask.shape) != tuple(shape):
        raise ValueError("Output diagnostic loss mask must match [sequence, batch] CE tensors")
    return mask.to(device=device) > 0


def prepare_margin(logits: torch.Tensor, target: torch.Tensor, tp_group) -> Optional[torch.Tensor]:
    """Compute Barron et al.'s m = z_target - LSE(non-target logits).

    Every TP rank participates; only rank zero keeps the resulting token vector.
    Exclude the target BEFORE summation: subtracting its exponential from CE's
    rounded denominator would lose precisely the small values we need to measure.
    Two bounded-workspace passes find the non-target maximum and exponential sum.
    TP communicates token vectors only; model logits are never modified.
    """
    if not (_active and _in_forward and _in_loss):
        return None
    with torch.no_grad():
        vocab = logits.shape[-1]
        flat = logits.detach().reshape(-1, vocab)
        labels = target.reshape(-1)
        offset = _rank(tp_group) * vocab
        local_labels = labels - offset
        owned = (local_labels >= 0) & (local_labels < vocab)
        indices = local_labels.clamp(0, vocab - 1).long()
        rows = max(1, _MARGIN_CHUNK_ELEMENTS // vocab)
        target_logits = torch.empty(flat.shape[0], device=logits.device, dtype=torch.float32)
        rest_max = torch.empty_like(target_logits)

        def excluding_target(start, end):
            chunk = flat[start:end].to(dtype=torch.float32, copy=True)
            index = indices[start:end, None]
            selected = chunk.gather(1, index)
            chunk.scatter_(1, index, torch.where(owned[start:end, None], -float("inf"), selected))
            return chunk, selected.squeeze(-1)

        for start in range(0, flat.shape[0], rows):
            end = min(start + rows, flat.shape[0])
            chunk, selected = excluding_target(start, end)
            target_logits[start:end] = torch.where(owned[start:end], selected, 0.0)
            rest_max[start:end] = chunk.amax(dim=-1)
            del chunk, selected
        if _size(tp_group) > 1:
            torch.distributed.all_reduce(target_logits, group=tp_group)
            torch.distributed.all_reduce(rest_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        # Handle a vocabulary with no finite competitors without -inf - -inf.
        shift = torch.where(rest_max == -float("inf"), 0.0, rest_max)
        rest_sum = torch.empty_like(target_logits)
        for start in range(0, flat.shape[0], rows):
            end = min(start + rows, flat.shape[0])
            chunk, _ = excluding_target(start, end)
            chunk.sub_(shift[start:end, None]).exp_()
            rest_sum[start:end] = chunk.sum(dim=-1)
            del chunk
        if _size(tp_group) > 1:
            torch.distributed.all_reduce(rest_sum, group=tp_group)
        if _rank(tp_group) != 0:
            return None
        return (target_logits - shift - rest_sum.log()).reshape(target.shape)


def record_ce(
    logits_max: torch.Tensor,
    sum_exp_logits: torch.Tensor,
    loss: torch.Tensor,
    margin: Optional[torch.Tensor],
) -> None:
    """Reuse global MAX and exp-sum from CE; loss is BEFORE training masks/scaling."""
    if not (_active and _in_forward and _in_loss) or margin is None:
        return
    if loss.numel() == 0:
        return
    with torch.no_grad():
        _initialize(loss.device)
        valid = _valid(loss.shape, loss.device)
        lse = logits_max.detach() + sum_exp_logits.detach().log()
        _sums[0].add_(valid.sum())
        _sums[1].add_(torch.where(valid, lse, 0.0).sum(dtype=torch.float64))
        _sums[2].add_(((loss.detach() == 0) & valid).sum())
        _sums[5].add_(((margin > FP32_PRECISION_CLIFF) & valid).sum())
        maxima = torch.stack(
            [torch.where(valid, t.detach(), float("-inf")).amax()
             for t in (logits_max, lse, margin)]
        ).double()
        _maxima[:3] = torch.maximum(_maxima[:3], maxima)
        # A new token-sized tensor, never a view into the enormous vocab tensor.
        _margins.append(torch.where(valid, margin, float("-inf")).reshape(-1))


def record_hidden(
    hidden: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    tp_group,
    sequence_parallel: bool,
) -> None:
    """Final head-input token L2 norms; SP shards counted once, TP replicas once."""
    if not (_active and _in_forward):
        return
    if not sequence_parallel and _rank(tp_group) != 0:
        return
    with torch.no_grad():
        _initialize(hidden.device)
        mask = loss_mask
        if mask is not None and sequence_parallel and _size(tp_group) > 1:
            mask = mask.chunk(_size(tp_group), dim=1)[_rank(tp_group)]
        if mask is not None and tuple(mask.shape) != (hidden.shape[1], hidden.shape[0]):
            raise ValueError("Final hidden diagnostic mask must match the local sequence shard")
        positions = max(1, _CHUNK_ELEMENTS // max(1, hidden.shape[1] * hidden.shape[2]))
        for start in range(0, hidden.shape[0], positions):
            end = min(start + positions, hidden.shape[0])
            norms = hidden.detach()[start:end].float().norm(dim=-1)
            valid = torch.ones_like(norms, dtype=torch.bool) if mask is None else (
                mask[:, start:end].transpose(0, 1).to(device=hidden.device) > 0
            )
            _sums[3].add_(valid.sum())
            _sums[4].add_(torch.where(valid, norms, 0.0).sum(dtype=torch.float64))
            _maxima[3] = torch.maximum(
                _maxima[3], torch.where(valid, norms, float("-inf")).amax().double()
            )


def percentile_tail_size(count: int) -> int:
    """Exact tail length for linearly interpolated q=999/1000, without FP rank error."""
    return count - ((count - 1) * 999 // 1000) if count > 0 else 0


def percentile_from_tail(values: torch.Tensor, count: int) -> torch.Tensor:
    """Read the pooled p99.9 from the largest required values, with interpolation."""
    lower, remainder = divmod((count - 1) * 999, 1000)
    upper = lower + (remainder > 0)
    tail = values.topk(count - lower).values.double()
    a, b = tail[count - 1 - lower], tail[count - 1 - upper]
    return a + (b - a) * (remainder / 1000.0)


@torch.no_grad()
def collect(device: torch.device, tp_group, dp_cp_group, is_last_stage: bool) -> Dict[str, float]:
    """Pool scalar diagnostics globally and exact margin tails on output-stage TP0.

    The DP/CP root gathers at most K=ceil(0.001*(N-1))+1 values from each rank.
    Other pipeline stages never receive margin vectors or vocabulary logits.
    """
    global _active, _in_forward
    _active = _in_forward = False
    _initialize(device)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        torch.distributed.all_reduce(_sums, op=torch.distributed.ReduceOp.SUM)
    counts = _sums.cpu().tolist()
    token_count = int(counts[0])
    keep = percentile_tail_size(token_count)
    if keep and is_last_stage and _rank(tp_group) == 0:
        local = torch.cat(_margins) if _margins else torch.empty(0, device=device)
        tail = local.topk(min(keep, local.numel())).values
        if _size(dp_cp_group) > 1:
            padded = torch.full((keep,), float("-inf"), device=device)
            padded[:tail.numel()] = tail
            gathered = [torch.empty_like(padded) for _ in range(_size(dp_cp_group))] if _rank(dp_cp_group) == 0 else None
            root = torch.distributed.get_global_rank(dp_cp_group, 0)
            torch.distributed.gather(padded, gather_list=gathered, dst=root, group=dp_cp_group)
            if gathered is not None:
                _maxima[4] = percentile_from_tail(torch.cat(gathered), token_count)
        else:
            _maxima[4] = percentile_from_tail(tail, token_count)
    _margins.clear()
    if distributed:
        torch.distributed.all_reduce(_maxima, op=torch.distributed.ReduceOp.MAX)
    maxima = _maxima.cpu().tolist()
    stats = {}
    if token_count:
        stats.update({
            "output_logits/max": maxima[0],
            "output_logits/lse_mean": counts[1] / token_count,
            "output_logits/lse_max": maxima[1],
            "output_logits/zero_loss_fraction": counts[2] / token_count,
            "output_logits/target_margin_max": maxima[2],
            "output_logits/target_margin_p999": maxima[4],
            "output_logits/target_margin_gt_fp32_cliff_fraction": counts[5] / token_count,
        })
    if counts[3]:
        stats.update({
            "output_logits/final_hidden_norm_mean": counts[4] / counts[3],
            "output_logits/final_hidden_norm_max": maxima[3],
        })
    reset_attempt()
    return stats
