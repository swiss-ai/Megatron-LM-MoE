# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Sampled activation RMS diagnostics without retaining activation tensors.

Only the scheduled forward-step callable enables recording. Backward checkpoint
recomputation therefore cannot add observations, including no-grad initial
checkpoint forwards. Each rerun starts with fresh accumulators.
"""

from contextlib import contextmanager
from functools import wraps
from typing import Callable, Dict, Iterator, Optional, Union

import torch

METRICS = ("residual_input", "attention_output", "mlp_output")
# Bound FP32 workspace to about 1 MiB (or one sequence position if larger).
_CHUNK_ELEMENTS = 262144
# Spike stats run 40-60 small kernels per chunk, so they use 16 MiB chunks to stay off the
# kernel-launch floor. Transient workspace is about 80 MiB: three FP32 chunk buffers plus the
# float64 temporary of each float64 sum.
_SPIKE_CHUNK_ELEMENTS = 4 * 1024 * 1024
# torch.histc counts in float32, which is exact only below 2**24 per call.
assert _SPIKE_CHUNK_ELEMENTS < 2**24
# Spike stats: the median of |x| is read from a log10 histogram with 16 bins per decade
# over [1e-12, 1e6], so inside that range it is within half a bin (about 7.5%) of the true
# median. |x| below 1e-12, including 0, counts as 1e-12 and |x| above 1e6 as 1e6, and the
# result is clamped to [min, max]; NaN is left out.
_HIST_LOG10_MIN, _HIST_LOG10_MAX, _HIST_BINS_PER_DECADE = -12, 6, 16
_HIST_BINS = (_HIST_LOG10_MAX - _HIST_LOG10_MIN) * _HIST_BINS_PER_DECADE
# Per (layer, metric): [count, sum d, sum d^2, sum d^3, sum d^4, sum |x|, histogram...] with
# d = x - shift, where shift is the mean of the first chunk with a valid element, rounded to
# float32 (the precision d is computed in). Moments about a nearby shift keep the kurtosis
# exact even when the mean is large against the spread.
_SPIKE_SUMS = 6
_active = False
_spike_stats = False
_in_forward = False
_acc: Dict[tuple[int, str], torch.Tensor] = {}
_spike: Dict[tuple[int, str], "_SpikeState"] = {}


class _SpikeState:
    __slots__ = ("sums", "shift", "has_shift", "shift_known", "extrema")

    def __init__(self, device: torch.device):
        self.sums = torch.zeros(_SPIKE_SUMS + _HIST_BINS, device=device, dtype=torch.float64)
        self.shift = torch.zeros((), device=device, dtype=torch.float64)
        # Whether a chunk with a valid element has set the shift: as a tensor (padded chunks
        # are only known on the device), and on the host once an unpadded chunk set it.
        self.has_shift = torch.zeros((), device=device, dtype=torch.bool)
        self.shift_known = False
        # [-min |x|, max |x|], so one MAX all-reduce pools both.
        self.extrema = torch.tensor([-float("inf"), 0.0], device=device, dtype=torch.float64)


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
    """Accumulate squared sum and element count for [sequence, batch, hidden], and the
    spike accumulators when spike stats are on.

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
    chunk_elements = _SPIKE_CHUNK_ELEMENTS if _spike_stats else _CHUNK_ELEMENTS
    positions = max(1, chunk_elements // (x.shape[1] * x.shape[2]))
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
            padding = None
            count = values.numel()
        if _spike_stats:
            _accumulate_spike(key, values, padding, count)
        values.square_()
        acc[0].add_(values.sum(dtype=torch.float64))
        acc[1].add_(count)


def _accumulate_spike(
    key: tuple, values: torch.Tensor, padding: Optional[torch.Tensor], count: Union[int, torch.Tensor]
) -> None:
    """Add one chunk (padding already zeroed, ``count`` valid elements) to the spike
    accumulators of ``key``. No host synchronisation: padding is handled by masking."""
    state = _spike.get(key)
    if state is None:
        state = _spike[key] = _SpikeState(values.device)
    if not state.shift_known:
        # Until a chunk with a valid element is seen, every accumulated sum is 0, so the shift
        # can still change. A chunk that is all padding must not set it to 0.
        valid = count.clamp(min=1) if torch.is_tensor(count) else count
        chunk_mean = (values.sum(dtype=torch.float64) / valid).float().double()
        state.shift = torch.where(state.has_shift, state.shift, chunk_mean)
        if torch.is_tensor(count):
            state.has_shift |= count > 0
        else:  # an unpadded chunk is never empty here
            state.shift_known = True
    pad = None if padding is None else padding.unsqueeze(-1).expand_as(values)
    magnitude = values.abs()
    abs_sum = magnitude.sum(dtype=torch.float64)
    max_abs = magnitude.max()
    if pad is not None:
        magnitude.masked_fill_(pad, float("inf"))
    min_abs = magnitude.min()
    torch.maximum(state.extrema, torch.stack((-min_abs, max_abs)).double(), out=state.extrema)
    # Histogram of log10 |x|: 0 maps to -inf and NaN stays NaN; the clamp keeps 0 and
    # out-of-range values in the end bins, and histc ignores NaN and the -inf padding.
    log_magnitude = magnitude.log10_().clamp_(_HIST_LOG10_MIN, _HIST_LOG10_MAX)
    if pad is not None:
        log_magnitude.masked_fill_(pad, -float("inf"))
    state.sums[_SPIKE_SUMS:].add_(
        torch.histc(log_magnitude, _HIST_BINS, _HIST_LOG10_MIN, _HIST_LOG10_MAX)
    )
    # Moments of d = x - shift; the buffer of |x| is free now and takes d, d^3.
    d = torch.sub(values, state.shift, out=magnitude)
    if pad is not None:
        d.masked_fill_(pad, 0.0)
    d2 = d.square()
    s1, s2 = d.sum(dtype=torch.float64), d2.sum(dtype=torch.float64)
    s3 = d.mul_(d2).sum(dtype=torch.float64)
    s4 = d2.square_().sum(dtype=torch.float64)
    state.sums[0].add_(count)
    state.sums[1:_SPIKE_SUMS].add_(torch.stack((s1, s2, s3, s4, abs_sum)))


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
    spike_stats = _collect_spike(num_layers, device, totals, distributed) if _spike_stats else {}
    totals = totals.cpu()
    stats = {}
    for layer in range(num_layers):
        for index, metric in enumerate(METRICS):
            square_sum, count = totals[layer, index].tolist()
            if count > 0:
                stats[f"activation_norm/{metric}_rms_layer_{layer}"] = (square_sum / count) ** 0.5
    stats.update(spike_stats)
    return stats


def _collect_spike(num_layers: int, device: torch.device, totals: torch.Tensor,
                   distributed: bool) -> Dict[str, float]:
    """Pool the spike accumulators like the RMS sums and turn them into per-layer stats.

    Each rank holds moments about its own shift. The global mean is pooled first (SUM),
    every rank moves its moments to that mean, and the central moments, |x| sum and
    histogram are pooled with SUM; [-min, max] of |x| with MAX. Kurtosis is the plain
    (Pearson) kurtosis E[(x - mean)^4] / Var^2 of every logged element: 3 for a Gaussian,
    larger when a few entries are much bigger than the rest. TP replication scales all
    sums equally. ``totals`` holds the pooled [square sum, count] of the RMS path."""
    shape = (num_layers, len(METRICS))
    sums = torch.zeros(shape + (_SPIKE_SUMS + _HIST_BINS,), device=device, dtype=torch.float64)
    shift = torch.zeros(shape, device=device, dtype=torch.float64)
    extrema = torch.tensor([-float("inf"), 0.0], device=device, dtype=torch.float64).repeat(
        shape + (1,))
    for (layer, metric), state in _spike.items():
        if 0 <= layer < num_layers:
            index = (layer, METRICS.index(metric))
            sums[index], shift[index], extrema[index] = state.sums, state.shift, state.extrema
    _spike.clear()
    n, s1, s2, s3, s4 = (sums[..., i] for i in range(5))
    # Global mean from sum x = sum d + count * shift, then the central moments about it.
    sum_x = s1 + n * shift
    if distributed:
        torch.distributed.all_reduce(sum_x, op=torch.distributed.ReduceOp.SUM)
    d = shift - sum_x / totals[..., 1].clamp(min=1)
    m2 = s2 + 2 * s1 * d + n * d * d
    m4 = s4 + 4 * s3 * d + 6 * s2 * d * d + 4 * s1 * d * d * d + n * d * d * d * d
    pooled = torch.cat((torch.stack((m2, m4), -1), sums[..., 5:]), -1)
    if distributed:
        torch.distributed.all_reduce(pooled, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(extrema, op=torch.distributed.ReduceOp.MAX)
    pooled, extrema, counts = pooled.cpu(), extrema.cpu(), totals[..., 1].cpu()
    stats = {}
    for layer in range(num_layers):
        for index, metric in enumerate(METRICS):
            count = counts[layer, index].item()
            if count <= 0:
                continue
            m2, m4, abs_mean = (v / count for v in pooled[layer, index, :3].tolist())
            abs_min, abs_max = (-extrema[layer, index, 0]).item(), extrema[layer, index, 1].item()
            hist = pooled[layer, index, 3:]
            if hist.sum() > 0:
                # Lower median for an even count, as torch.median; the bin centre, kept
                # inside [min, max] so constant or all-zero inputs read exactly.
                median_bin = int(torch.searchsorted(hist.cumsum(0), hist.sum() / 2).item())
                median = 10 ** (_HIST_LOG10_MIN + (median_bin + 0.5) / _HIST_BINS_PER_DECADE)
                median = min(max(median, abs_min), abs_max)
            else:
                median = float("nan")
            prefix = f"activation_spike/{metric}"
            stats[f"{prefix}_abs_min_layer_{layer}"] = abs_min
            stats[f"{prefix}_abs_mean_layer_{layer}"] = abs_mean
            stats[f"{prefix}_abs_median_layer_{layer}"] = median
            stats[f"{prefix}_abs_max_layer_{layer}"] = abs_max
            stats[f"{prefix}_kurtosis_layer_{layer}"] = m4 / (m2 * m2) if m2 > 0 else float("nan")
    return stats
