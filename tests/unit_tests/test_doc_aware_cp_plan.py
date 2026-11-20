# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.
"""Properties of the document-aware CP plan (megatron/core/doc_aware_cp_plan.py)."""
import numpy as np
import pytest

from megatron.core.doc_aware_cp_plan import plan_doc_aware


def check(lengths, cp, sp, lm=1, **kw):
    cu = np.concatenate(([0], np.cumsum(lengths)))
    S = int(cu[-1])
    p = plan_doc_aware(cu, cp, sp, lm, **kw)
    T = p.seq_len
    assert T % (sp * lm) == 0 and p.long_len % (2 * sp) == 0 and p.local_len + p.long_len == T
    src = p.sources
    real = src[src >= 0]
    assert np.array_equal(np.sort(real), np.arange(S)), "every token exactly once"
    doc_of = np.searchsorted(cu[1:], np.arange(S), side="right")
    # local part: local_cu segments are whole documents (in order) or pure padding
    for r in range(cp):
        lc = p.local_cu[r]
        assert lc[0] == 0 and lc[-1] == p.local_len
        for a, b in zip(lc[:-1], lc[1:]):
            seg = src[r, a:b]
            if (seg < 0).all():
                continue
            assert (seg >= 0).all(), "local segment mixes padding and tokens"
            d = doc_of[seg[0]]
            assert np.array_equal(seg, np.arange(cu[d], cu[d + 1])), "local doc whole and in order"
        assert p.local_max[r] == (np.diff(lc).max() if len(lc) > 1 else 0)
    # long part: TE zigzag of each padded long document
    lcu = p.long_cu
    for j in range(len(lcu) - 1):
        Lp = lcu[j + 1] - lcu[j]; w = Lp // (2 * cp)
        assert Lp % (2 * cp * sp) == 0
        full = np.full(Lp, -1)
        for r in range(cp):
            # rank r's piece of doc j: its offset inside rank r's long part
            off = p.local_len + sum((lcu[k + 1] - lcu[k]) // cp for k in range(j))
            piece = src[r, off: off + 2 * w]
            full[r * w:(r + 1) * w] = piece[:w]
            full[(2 * cp - 1 - r) * w:(2 * cp - r) * w] = piece[w:]
        toks = full[full >= 0]
        d = doc_of[toks[0]]
        assert np.array_equal(toks, np.arange(cu[d], cu[d + 1])), "long doc complete, in order"
        assert (full[len(toks):] < 0).all(), "long doc padding only at its tail"
    # virtual buffer: a permutation; every segment is one document or padding, in order
    vp = p.virtual_pos.reshape(-1)
    assert np.array_equal(np.sort(vp), np.arange(cp * T))
    inv = np.empty(cp * T, dtype=np.int64); inv[vp] = src.reshape(-1)
    vcu = p.virtual_cu
    assert vcu[0] == 0 and vcu[-1] == cp * T and np.all(np.diff(vcu) > 0)
    for a, b in zip(vcu[:-1], vcu[1:]):
        seg = inv[a:b]; toks = seg[seg >= 0]
        if len(toks) == 0:
            continue
        d = doc_of[toks[0]]
        assert np.array_equal(toks, np.arange(cu[d], cu[d + 1])), "virtual segment = one doc"
        assert (seg[:len(toks)] >= 0).all(), "doc tokens first, padding at the tail"
    # spans / moves flags
    bounds = np.arange(1, cp) * T
    assert p.spans == (not np.isin(bounds, vcu).all())
    assert p.moves == (not np.array_equal(vp, np.arange(cp * T)))
    return p


@pytest.mark.parametrize("seed", range(6))
def test_random_plans(seed):
    rng = np.random.default_rng(seed)
    for trial in range(50):
        cp = int(rng.choice([2, 4, 8]))
        sp = int(rng.choice([1, 2]))
        lm = int(rng.choice([1, 1, 16]))
        S = int(rng.choice([4096, 8192, 16384]))
        kind = trial % 3
        lens = []
        while sum(lens) < S:
            if kind == 0:
                l = int(rng.integers(1, 800))
            elif kind == 1:
                l = int(rng.integers(S // 6, S))
            elif rng.random() < 0.8:
                l = int(rng.integers(1, 800))
            else:
                l = int(rng.integers(S // 8, S // 2))
            lens.append(l)
        lens[-1] -= sum(lens) - S
        check([l for l in lens if l > 0], cp, sp, lm, max_padding=float(rng.choice([0.0, 0.01, 0.05])))


def test_short_documents_stay_local():
    rng = np.random.default_rng(0)
    l = np.minimum(rng.lognormal(6.5, 1.0, 400).astype(int) + 1, 8191)
    l = l[np.cumsum(l) < 65536]
    p = check(list(l) + [65536 - int(l.sum())], 4, 2)
    assert not p.moves and not p.spans and p.long_len == 0
    assert p.seq_len * 4 - 65536 < 0.01 * 65536


def test_long_documents_split():
    p = check([9000, 7000, 384], 4, 2)
    assert p.num_local_docs == 0 and p.spans and p.moves
