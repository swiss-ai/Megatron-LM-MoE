# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the goldfish-loss dropping mechanism (megatron.core.datasets.goldfish)."""

import torch

_MOCK_VOCAB_SIZE = 8192


def test_apply_goldfish():
    from megatron.core.datasets.goldfish import (
        GOLDFISH_TOKEN_ID,
        _create_exemption_lut,
        _create_hash_table,
        apply_goldfish,
    )

    generator = torch.Generator().manual_seed(0)
    seq_length, k, h = 8192, 50, 50
    labels = torch.randint(
        0, _MOCK_VOCAB_SIZE, (seq_length,), dtype=torch.long, generator=generator
    )
    table = _create_hash_table(device=labels.device)

    original = labels.clone()
    out = apply_goldfish(labels, GOLDFISH_TOKEN_ID, k, table, goldfish_context_width=h)

    # apply_goldfish works on a clone: the input labels are not mutated.
    assert torch.equal(labels, original)

    # Deterministic: identical inputs -> identical drops.
    out2 = apply_goldfish(labels, GOLDFISH_TOKEN_ID, k, table, goldfish_context_width=h)
    assert torch.equal(out, out2)

    # The first h-1 positions are never eligible for dropping (unfold alignment).
    assert not torch.any(out[: h - 1] == GOLDFISH_TOKEN_ID)

    # Drop rate over the eligible tail is approximately 1/k.
    drop_rate = (out[h - 1 :] == GOLDFISH_TOKEN_ID).float().mean().item()
    assert abs(drop_rate - 1.0 / k) < 0.01, drop_rate

    # Exemption: no dropped position may carry a label in the exempt band, while drops
    # still occur outside it. `out` above is the positive control (drops with no exemption).
    assert torch.any(out == GOLDFISH_TOKEN_ID)
    lo, hi = 2000, 6000
    exemption_lut = _create_exemption_lut(list(range(lo, hi)), _MOCK_VOCAB_SIZE, labels.device)
    out_exempt = apply_goldfish(
        labels, GOLDFISH_TOKEN_ID, k, table, goldfish_context_width=h, exemption_lut=exemption_lut
    )
    dropped = out_exempt == GOLDFISH_TOKEN_ID
    assert not torch.any((labels >= lo) & (labels < hi) & dropped), "exempt-band token was dropped"
    assert torch.any(dropped), "expected drops outside the exempt band"

    # Pinned position: exempting exactly the label id of a known dropped position
    # cancels that drop (deterministic alignment of window hash vs. exempt mask).
    pinned = int((out == GOLDFISH_TOKEN_ID).nonzero(as_tuple=True)[0][0])
    pin_lut = _create_exemption_lut([int(labels[pinned])], _MOCK_VOCAB_SIZE, labels.device)
    out_pinned = apply_goldfish(
        labels, GOLDFISH_TOKEN_ID, k, table, goldfish_context_width=h, exemption_lut=pin_lut
    )
    assert out_pinned[pinned] != GOLDFISH_TOKEN_ID

    # Arbitrary scattered special-token ids are represented by the same LUT.
    scattered_lut = _create_exemption_lut([0, 18, 2000, 6000], _MOCK_VOCAB_SIZE, labels.device)
    assert torch.all(scattered_lut[torch.tensor([0, 18, 2000, 6000])])

    # Windows containing id-0 labels (pad/unk) must hash like any other window: with a
    # zero every 10 positions every window contains one, yet the rate stays ~1/k (the
    # old product hash collapsed all such windows onto a single drop decision).
    zero_heavy = labels.clone()
    zero_heavy[::10] = 0
    out_zero = apply_goldfish(zero_heavy, GOLDFISH_TOKEN_ID, k, table, goldfish_context_width=h)
    zero_rate = (out_zero[h - 1 :] == GOLDFISH_TOKEN_ID).float().mean().item()
    assert abs(zero_rate - 1.0 / k) < 0.01, zero_rate


if __name__ == "__main__":
    test_apply_goldfish()
