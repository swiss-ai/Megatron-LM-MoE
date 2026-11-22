# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.
"""CPU checks of the document-aware CP batch slicing on Apertus-SFT-shaped packs (``--ap-sft --cp-doc-aware``).

SFT packs are BFD-packed conversations, each padded to ``lcm(2 * cp, cp * sp)`` tokens, with the pack's tail padding
merged into the last segment (``ApertusSFTDataset.__getitem__``). Across all CP ranks the doc-aware slices must hold
every real token exactly once, keep loss weights, assistant masks, labels and positions with their tokens, flag all
padding, give every rank the same length, and keep local documents whole and in order on one rank.
"""
import numpy as np
import pytest
import torch

from megatron.core.doc_aware_cp import get_doc_aware_batch_on_this_cp_rank
from megatron.core.doc_aware_cp_plan import plan_doc_aware


def _sft_pack(lengths, seq_len, align, rng):
    """Batch dict and cu_seqlens laid out as ApertusSFTDataset builds them (batch dim 1)."""
    tokens = np.zeros(seq_len, dtype=np.int64)
    labels = np.zeros(seq_len, dtype=np.int64)
    loss_mask = np.zeros(seq_len, dtype=np.float32)
    assistant = np.zeros(seq_len, dtype=bool)
    positions = np.zeros(seq_len, dtype=np.int64)
    padding = np.ones(seq_len, dtype=bool)
    boundaries, cursor = [0], 0
    for length in lengths:
        padded = -(-length // align) * align
        assert cursor + padded <= seq_len
        span = np.arange(cursor, cursor + length)
        tokens[span] = span + 1  # unique ids: position + 1
        labels[cursor:cursor + length - 1] = tokens[cursor + 1:cursor + length]
        weights = rng.choice([0.0, 0.5, 1.0], size=length).astype(np.float32)  # fractional loss weights
        loss_mask[cursor:cursor + length - 1] = weights[1:]
        assistant[cursor:cursor + length - 1] = weights[1:] > 0
        positions[span] = np.arange(length)
        padding[span] = False
        cursor += padded
        boundaries.append(cursor)
    boundaries[-1] = seq_len  # tail padding joins the last segment
    batch = {
        'tokens': torch.from_numpy(tokens)[None],
        'labels': torch.from_numpy(labels)[None],
        'loss_mask': torch.from_numpy(loss_mask)[None],
        'assistant_mask': torch.from_numpy(assistant)[None],
        'position_ids': torch.from_numpy(positions)[None],
        'padding_mask': torch.from_numpy(padding)[None],
    }
    return batch, torch.tensor(boundaries, dtype=torch.int32)


def _cases(seq_len, cp, rng):
    local_max = seq_len // cp
    short = lambda n: rng.integers(30, local_max // 3, size=n)
    yield "all short, full", _fill(short(10 * cp), seq_len, 2 * cp)
    yield "mixed long and short", _fill(np.concatenate([[int(1.6 * local_max), int(2.3 * local_max)], short(6 * cp)]), seq_len, 2 * cp)
    yield "one long document", [seq_len - 3 * cp]
    yield "half full (large tail padding)", list(short(2 * cp))[: max(1, cp // 2)]
    yield "many tiny documents", _fill(rng.integers(5, 40, size=40 * cp), seq_len, 2 * cp)


def _fill(lengths, seq_len, align):
    out, used = [], 0
    for length in lengths:
        padded = -(-int(length) // align) * align
        if used + padded > seq_len:
            break
        out.append(int(length))
        used += padded
    return out


@pytest.mark.parametrize("cp,seq_len", [(4, 4096), (16, 131072)])
@pytest.mark.parametrize("max_padding", [0.01, 0.05, 0.10])
def test_doc_aware_sft_slices_conserve_the_pack(cp, seq_len, max_padding):
    rng = np.random.default_rng(cp * 1000 + int(max_padding * 100))
    align = 2 * cp
    for name, lengths in _cases(seq_len, cp, rng):
        batch, cu = _sft_pack(lengths, seq_len, align, rng)
        plan = plan_doc_aware(cu.numpy().astype(np.int64), cp, max_padding=max_padding)
        ref = {k: v[0].numpy() for k, v in batch.items()}
        real = ~ref['padding_mask']
        seen = np.zeros(seq_len, dtype=np.int64)
        sums = dict(loss=0.0, loss_tokens=0, assistant=0)
        rank_lengths = set()
        for rank in range(cp):
            out, params = get_doc_aware_batch_on_this_cp_rank(
                {k: v.clone() for k, v in batch.items()}, cu, cp, rank, max_padding=max_padding
            )
            got = {k: v[0].numpy() for k, v in out.items()}
            rank_lengths.add(len(got['tokens']))
            keep = ~got['padding_mask']
            pos = got['tokens'][keep] - 1  # original positions of the real tokens held by this rank
            seen[pos] += 1
            for key in ('labels', 'loss_mask', 'assistant_mask', 'position_ids'):
                np.testing.assert_array_equal(got[key][keep], ref[key][pos], err_msg=f"{name}: {key} moved")
            assert not got['loss_mask'][~keep].any(), f"{name}: loss on padding"
            assert not got['assistant_mask'][~keep].any(), f"{name}: assistant mask on padding"
            sums['loss'] += float(got['loss_mask'].sum())
            sums['loss_tokens'] += int((got['loss_mask'] > 0).sum())
            sums['assistant'] += int(got['assistant_mask'].sum())
            layout = params.doc_aware
            assert layout.seq_len == plan.seq_len and int(layout.virtual_cu[-1]) == cp * plan.seq_len
            # Local documents: whole, contiguous and in order on this rank (start of the slice).
            local = got['tokens'][:layout.local_len]
            local_cu = plan.local_cu[rank]
            for a, b in zip(local_cu[:-1], local_cu[1:]):
                seg = local[a:b]
                seg = seg[seg > 0]
                if len(seg):
                    assert np.all(np.diff(seg) == 1), f"{name}: local document split or reordered on rank {rank}"
        assert len(rank_lengths) == 1, f"{name}: ranks differ in length {rank_lengths}"
        assert rank_lengths.pop() == plan.seq_len
        np.testing.assert_array_equal(seen[real], 1, err_msg=f"{name}: real tokens lost or duplicated")
        assert not seen[~real].any(), f"{name}: dataset padding counted as real"
        assert sums['loss'] == pytest.approx(float(ref['loss_mask'].sum()))
        assert sums['loss_tokens'] == int((ref['loss_mask'] > 0).sum())
        assert sums['assistant'] == int(ref['assistant_mask'].sum())
        assert plan.seq_len - seq_len // cp <= max_padding * seq_len / cp + align or plan.num_local_docs == 0
