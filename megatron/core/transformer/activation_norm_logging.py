# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Sampled activation RMS diagnostics without retaining activation tensors.

Only the scheduled forward-step callable enables recording. Backward checkpoint
recomputation therefore cannot add observations, including no-grad initial
checkpoint forwards. Each rerun starts with fresh accumulators.
"""

from contextlib import contextmanager
from functools import wraps
from typing import Callable, Dict, Iterator, Optional

import torch

METRICS = ("residual_input", "attention_output", "mlp_output")
# Bound FP32 workspace to about 1 MiB (or one sequence position if larger).
_CHUNK_ELEMENTS = 262144
# Spike stats: the median of |x| comes from a log10 histogram, 16 bins per decade over
# [1e-12, 1e6) (bin width ~15%); smaller values, including 0, fall in the first bin.
_HIST_LOG10_MIN, _HIST_LOG10_MAX, _HIST_BINS_PER_DECADE = -12, 6, 16
_HIST_BINS = (_HIST_LOG10_MAX - _HIST_LOG10_MIN) * _HIST_BINS_PER_DECADE
_active = False
_spike_stats = False
_in_forward = False
_acc: Dict[tuple[int, str], torch.Tensor] = {}
# Per (layer, metric): [sum x, sum |x|, sum x^3, sum x^4, histogram of |x|...], min |x|, max |x|.
_SPIKE_SUMS = 4
_spike: Dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def should_log(step: int, interval: int = 0) -> bool:
    """Sample completed-step numbers, with zero disabling diagnostics."""
    if interval < 0:
        raise ValueError("activation-norm-log-interval must be nonnegative")
    return interval > 0 and step > 0 and step % interval == 0


def set_active(active: bool, spike_stats: bool = False) -> None:
    """Start a sampled step, or release any stale diagnostics. ``spike_stats`` also
    collects min/mean/median/max of |x| and the kurtosis of x."""
    global _active, _spike_stats, _in_forward
    _active = bool(active)
    _spike_stats = bool(spike_stats)
    _in_forward = False
    _acc.clear()
    _spike.clear()


def is_active() -> bool:
    return _active


def reset_attempt() -> None:
    """Discard observations from a rejected forward/backward attempt."""
    _acc.clear()
    _spike.clear()


@contextmanager
def forward_scope() -> Iterator[None]:
    """Enable observation only while executing a scheduled model forward."""
    global _in_forward
    previous = _in_forward
    _in_forward = _active
    try:
        yield
    finally:
        _in_forward = previous


def wrap_forward_step(forward_step: Callable) -> Callable:
    """Wrap the schedule's callback, preserving optional checkpoint arguments."""
    @wraps(forward_step)
    def wrapped(*args, **kwargs):
        with forward_scope():
            return forward_step(*args, **kwargs)

    return wrapped


def record(
    layer: int,
    metric: str,
    x: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    *,
    training: bool = True,
) -> None:
    """Accumulate squared sum and element count for [sequence, batch, hidden].

    ``padding_mask`` is already CP/SP-local, shaped [batch, sequence], with True
    for padding. Outputs are measured BEFORE sandwich norm/residual scaling/add,
    including the separate linear bias if present. FP32 copies are bounded and
    discarded immediately; neither inputs nor their autograd graphs are stored.
    """
    if not (_active and _in_forward and training):
        return
    if metric not in METRICS:
        raise ValueError(f"Unknown activation norm metric: {metric}")
    if x.ndim != 3:
        raise ValueError(f"Expected [sequence, batch, hidden], got {tuple(x.shape)}")
    if padding_mask is not None and tuple(padding_mask.shape) != (x.shape[1], x.shape[0]):
        raise ValueError("Activation norm padding mask must match the local batch/sequence")
    _accumulate(layer, metric, x, padding_mask, bias)


@torch.no_grad()
def _accumulate(
    layer: int,
    metric: str,
    x: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
) -> None:
    """Perform tensor work only after the cheap Python recording guards pass."""
    key = (layer, metric)
    acc = _acc.get(key)
    if acc is None:
        acc = torch.zeros(2, device=x.device, dtype=torch.float64)
        _acc[key] = acc
    if x.numel() == 0:
        return
    positions = max(1, _CHUNK_ELEMENTS // (x.shape[1] * x.shape[2]))
    for start in range(0, x.shape[0], positions):
        end = min(start + positions, x.shape[0])
        # copy=True is essential for FP32 inputs: in-place squaring must never
        # modify model activations, even when detach() shares their storage.
        values = x.detach()[start:end].to(dtype=torch.float32, copy=True)
        if bias is not None:
            values.add_(bias.detach().float())
        if padding_mask is not None:
            padding = padding_mask[:, start:end].transpose(0, 1).to(
                device=x.device, dtype=torch.bool
            )
            # Mask before squaring so NaNs in padding do not poison the sum.
            values.masked_fill_(padding.unsqueeze(-1), 0.0)
            count = (~padding).sum(dtype=torch.int64) * x.shape[-1]
        else:
            count = values.numel()
        if _spike_stats:
            _accumulate_spike(key, values, padding if padding_mask is not None else None)
        values.square_()
        acc[0].add_(values.sum(dtype=torch.float64))
        acc[1].add_(count)


def _accumulate_spike(key: tuple, values: torch.Tensor, padding: Optional[torch.Tensor]) -> None:
    """Add one chunk to the moment sums, |x| histogram and |x| min/max of ``key``."""
    spike = _spike.get(key)
    if spike is None:
        spike = (
            torch.zeros(_SPIKE_SUMS + _HIST_BINS, device=values.device, dtype=torch.float64),
            torch.full((), float("inf"), device=values.device, dtype=torch.float64),
            torch.zeros((), device=values.device, dtype=torch.float64),
        )
        _spike[key] = spike
    sums, min_abs, max_abs = spike
    if padding is not None:
        values = values[~padding]
    if values.numel() == 0:
        return
    magnitude = values.abs()
    sums[0].add_(values.sum(dtype=torch.float64))
    sums[1].add_(magnitude.sum(dtype=torch.float64))
    sums[2].add_(values.pow(3).sum(dtype=torch.float64))
    sums[3].add_(values.pow(4).sum(dtype=torch.float64))
    min_abs.copy_(torch.minimum(min_abs, magnitude.min().double()))
    max_abs.copy_(torch.maximum(max_abs, magnitude.max().double()))
    bins = ((magnitude.log10() - _HIST_LOG10_MIN) * _HIST_BINS_PER_DECADE).floor()
    bins = bins.nan_to_num_(0.0).clamp_(0, _HIST_BINS - 1).long()
    sums[_SPIKE_SUMS:].add_(torch.bincount(bins.flatten(), minlength=_HIST_BINS).double())


@torch.no_grad()
def collect(num_layers: int, device: torch.device) -> Dict[str, float]:
    """Pool scalar sums/counts across PP/DP/CP/TP ranks; call on every rank.

    Sequence shards contribute disjoint elements. Replicated TP observations
    multiply both sums and counts equally, leaving the RMS unchanged. Absent
    PP layers contribute zero. Only scalar diagnostics cross the network.
    """
    global _active, _in_forward
    _active = _in_forward = False
    totals = torch.zeros((num_layers, len(METRICS), 2), device=device, dtype=torch.float64)
    for (layer, metric), acc in _acc.items():
        if 0 <= layer < num_layers:
            totals[layer, METRICS.index(metric)] = acc
    _acc.clear()
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if distributed:
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    totals = totals.cpu()
    stats = {}
    for layer in range(num_layers):
        for index, metric in enumerate(METRICS):
            square_sum, count = totals[layer, index].tolist()
            if count > 0:
                stats[f"activation_norm/{metric}_rms_layer_{layer}"] = (square_sum / count) ** 0.5
    if _spike_stats:
        stats.update(_collect_spike(num_layers, device, totals, distributed))
    return stats


def _collect_spike(num_layers: int, device: torch.device, totals: torch.Tensor,
                   distributed: bool) -> Dict[str, float]:
    """Pool the spike accumulators like the RMS sums (SUM for moments and histogram, MIN/MAX
    for |x|) and turn them into per-layer stats. Kurtosis is the plain (Pearson) kurtosis
    E[(x - mean)^4] / Var^2 of every logged element: 3 for a Gaussian, larger when a few
    entries are much bigger than the rest. TP replication scales all sums equally."""
    shape = (num_layers, len(METRICS))
    sums = torch.zeros(shape + (_SPIKE_SUMS + _HIST_BINS,), device=device, dtype=torch.float64)
    min_abs = torch.full(shape, float("inf"), device=device, dtype=torch.float64)
    max_abs = torch.zeros(shape, device=device, dtype=torch.float64)
    for (layer, metric), (s, lo, hi) in _spike.items():
        if 0 <= layer < num_layers:
            index = (layer, METRICS.index(metric))
            sums[index], min_abs[index], max_abs[index] = s, lo, hi
    _spike.clear()
    if distributed:
        torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(min_abs, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(max_abs, op=torch.distributed.ReduceOp.MAX)
    sums, min_abs, max_abs = sums.cpu(), min_abs.cpu(), max_abs.cpu()
    stats = {}
    for layer in range(num_layers):
        for index, metric in enumerate(METRICS):
            square_sum, count = totals[layer, index].tolist()
            if count <= 0:
                continue
            s1, abs_mean, s3, s4 = (v / count for v in sums[layer, index, :_SPIKE_SUMS].tolist())
            s2 = square_sum / count
            hist = sums[layer, index, _SPIKE_SUMS:]
            median_bin = int(torch.searchsorted(hist.cumsum(0), hist.sum() / 2).item())
            var = s2 - s1 * s1
            m4 = s4 - 4 * s1 * s3 + 6 * s1 * s1 * s2 - 3 * s1 ** 4
            prefix = f"activation_spike/{metric}"
            stats[f"{prefix}_abs_min_layer_{layer}"] = min_abs[layer, index].item()
            stats[f"{prefix}_abs_mean_layer_{layer}"] = abs_mean
            stats[f"{prefix}_abs_median_layer_{layer}"] = 10 ** (
                _HIST_LOG10_MIN + (median_bin + 0.5) / _HIST_BINS_PER_DECADE)
            stats[f"{prefix}_abs_max_layer_{layer}"] = max_abs[layer, index].item()
            stats[f"{prefix}_kurtosis_layer_{layer}"] = m4 / (var * var) if var > 0 else float("nan")
    return stats
