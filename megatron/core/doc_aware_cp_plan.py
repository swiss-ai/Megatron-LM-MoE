# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.
"""Document-aware context parallelism inside a fixed CP group: the per-sample plan.

TE's THD context parallelism splits every document of a packed sample into ``2 * cp``
zigzag chunks, so even a 500-token document pays the ring attention and, for KDA, the
relayout and the state exchange. This plan keeps short documents whole on one CP rank
and only splits the long ones:

* every CP rank holds ``seq_len`` tokens: its *local* documents (whole, no CP
  communication), padding up to ``local_len``, then its zigzag share of the *long*
  documents (each padded to a multiple of ``2 * cp * sp``, exactly as today);
* softmax attention runs once without CP on the local part and once with the CP group
  on the long part;
* KDA (KCP) runs on a *virtual* packed buffer ``[rank 0 local][rank 1 local]...[long
  documents]`` cut into equal contiguous slices. A sample without long documents is
  then already in that layout (no relayout) and no document crosses a slice boundary
  (no state exchange).

Local documents are assigned greedily (largest first, to the least-loaded rank). If the
padding needed to equalise the ranks exceeds ``max_padding`` of the sample, the largest
local document is demoted to the long set and the assignment is redone; in the limit
every document is long, which is the per-document zigzag layout of plain THD CP.

This module only uses numpy so the plan can be tested without torch.
"""
import heapq
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class DocAwarePlan:
    cp: int
    sp: int
    sample_len: int  # tokens of the original sample
    local_len: int  # tokens of local documents + local padding on every rank
    long_len: int  # tokens of long documents (incl. their padding) on every rank
    seq_len: int  # local_len + long_len: tokens per CP rank
    sources: np.ndarray  # [cp, seq_len] index into the sample, -1 = padding token
    local_cu: List[np.ndarray]  # per rank: cu_seqlens of [0, local_len) (padding is a segment)
    local_max: List[int]
    long_cu: np.ndarray  # padded cu_seqlens of the long documents (global, TE THD CP)
    long_max: int
    virtual_cu: np.ndarray  # cu_seqlens of the virtual KCP buffer (cp * seq_len tokens)
    virtual_pos: np.ndarray  # [cp, seq_len] position of each held token in the virtual buffer
    spans: bool  # a virtual segment crosses a CP slice boundary (KCP must exchange states)
    moves: bool  # the KCP relayout is not the identity
    num_local_docs: int
    num_long_docs: int


def zigzag_positions(cu, cp, rank):
    """Positions held by ``rank`` under TE's per-document zigzag (each length % 2cp == 0)."""
    cu = np.asarray(cu, dtype=np.int64)
    starts = cu[:-1]
    width = (cu[1:] - starts) // (2 * cp)
    if width.size == 0:
        return np.zeros(0, dtype=np.int64)
    chunk_starts = np.stack(
        (starts + rank * width, starts + (2 * cp - 1 - rank) * width), axis=1
    ).reshape(-1)
    widths = np.repeat(width, 2)
    offsets = np.cumsum(widths) - widths
    return np.arange(int(widths.sum()), dtype=np.int64) + np.repeat(chunk_starts - offsets, widths)


def _round_up(x, m):
    return -(-x // m) * m


def _assign(lengths, local_ids, cp):
    """Largest-first greedy assignment of local documents to the least-loaded rank."""
    heap = [(0, r) for r in range(cp)]
    bins = [[] for _ in range(cp)]
    for i in sorted(local_ids, key=lambda i: (-lengths[i], i)):
        load, r = heapq.heappop(heap)
        bins[r].append(i)
        heapq.heappush(heap, (load + int(lengths[i]), r))
    loads = np.zeros(cp, dtype=np.int64)
    for load, r in heap:
        loads[r] = load
    return [sorted(b) for b in bins], loads


def plan_doc_aware(
    cu_seqlens,
    cp: int,
    sp: int = 1,
    local_multiple: int = 1,
    max_local_len: Optional[int] = None,
    max_padding: float = 0.01,
) -> DocAwarePlan:
    """Plan one packed sample (``cu_seqlens`` over ``[0, sample_len)``) for ``cp`` ranks.

    Args:
        cu_seqlens: document boundaries of the sample, starting at 0.
        cp: context-parallel size.
        sp: sequence-parallel (TP) size; per-rank lengths stay divisible by it.
        local_multiple: every rank's token count is also a multiple of this (FP8: 16).
        max_local_len: documents longer than this are always split (default: sample/cp).
        max_padding: largest acceptable padding, as a fraction of the sample, before the
            largest local document is demoted to the long set.
    """
    cu = np.asarray(cu_seqlens, dtype=np.int64).reshape(-1)
    lengths = cu[1:] - cu[:-1]
    keep = lengths > 0
    starts, lengths = cu[:-1][keep], lengths[keep]
    sample_len = int(cu[-1])
    n = len(lengths)
    align = 2 * cp * sp
    quantum = sp * local_multiple
    if max_local_len is None:
        max_local_len = sample_len // cp
    # Local candidates, largest first: a trial demotes the k largest to the long set.
    candidates = sorted(
        (i for i in range(n) if lengths[i] <= max_local_len), key=lambda i: (-lengths[i], i)
    )
    budget = max_padding * sample_len

    def trial(k):
        local_ids = candidates[k:]
        local_set = set(local_ids)
        long_ids = [i for i in range(n) if i not in local_set]
        long_padded = np.array(
            [_round_up(int(lengths[i]), align) for i in long_ids], dtype=np.int64
        )
        long_len = int(long_padded.sum()) // cp
        bins, loads = _assign(lengths, local_ids, cp)
        seq_len = _round_up(int(loads.max()) + long_len, quantum)
        ok = cp * seq_len - sample_len <= budget or not local_ids
        return ok, (local_ids, long_ids, long_padded, long_len, bins, seq_len)

    # Smallest k (approximately: the padding is not monotonic in k) that fits the budget:
    # exponential search, then bisection.
    ok, result = trial(0)
    if not ok:
        lo, hi = 0, 1
        while True:
            hi = min(hi, len(candidates))
            ok, result = trial(hi)
            if ok:
                break
            lo, hi = hi, 2 * hi
        while hi - lo > 1:
            mid = (lo + hi) // 2
            ok_mid, result_mid = trial(mid)
            if ok_mid:
                hi, result = mid, result_mid
            else:
                lo = mid
    local_ids, long_ids, long_padded, long_len, bins, seq_len = result

    local_len = seq_len - long_len
    long_cu = np.concatenate(([0], np.cumsum(long_padded))).astype(np.int64)
    long_starts = long_cu[:-1]
    # Source index of every token of the padded long buffer (-1 = its padding tail).
    long_src = np.full(int(long_cu[-1]), -1, dtype=np.int64)
    for j, i in enumerate(long_ids):
        long_src[long_starts[j] : long_starts[j] + lengths[i]] = np.arange(
            starts[i], starts[i] + lengths[i]
        )

    sources = np.full((cp, seq_len), -1, dtype=np.int64)
    virtual_pos = np.empty((cp, seq_len), dtype=np.int64)
    local_cu, local_max, virtual_segments = [], [], []
    for r in range(cp):
        seg, pos = [0], 0
        for i in bins[r]:
            sources[r, pos : pos + lengths[i]] = np.arange(starts[i], starts[i] + lengths[i])
            pos += int(lengths[i])
            seg.append(pos)
        if pos < local_len:
            seg.append(local_len)  # the padding is its own segment
        seg = np.asarray(seg, dtype=np.int64)
        local_cu.append(seg)
        local_max.append(int((seg[1:] - seg[:-1]).max()) if len(seg) > 1 else 0)
        virtual_segments.append(seg[1:] + r * local_len)
        zz = zigzag_positions(long_cu, cp, r)
        sources[r, local_len:] = long_src[zz]
        virtual_pos[r, :local_len] = r * local_len + np.arange(local_len)
        virtual_pos[r, local_len:] = cp * local_len + zz
    virtual_cu = np.concatenate([[0]] + virtual_segments + [cp * local_len + long_cu[1:]])
    virtual_cu = np.unique(virtual_cu)  # drops the empty segments of zero-length local parts
    # A segment crosses a slice boundary if a boundary lies strictly inside it.
    boundaries = np.arange(1, cp) * seq_len
    inside = np.searchsorted(virtual_cu, boundaries, side="left")
    spans = bool(np.any(virtual_cu[np.minimum(inside, len(virtual_cu) - 1)] != boundaries))
    moves = bool(np.any(virtual_pos != np.arange(cp * seq_len).reshape(cp, seq_len)))
    return DocAwarePlan(
        cp=cp,
        sp=sp,
        sample_len=sample_len,
        local_len=local_len,
        long_len=long_len,
        seq_len=seq_len,
        sources=sources,
        local_cu=local_cu,
        local_max=local_max,
        long_cu=long_cu,
        long_max=int(long_padded.max()) if len(long_padded) else 0,
        virtual_cu=virtual_cu,
        virtual_pos=virtual_pos,
        spans=spans,
        moves=moves,
        num_local_docs=len(local_ids),
        num_long_docs=len(long_ids),
    )
