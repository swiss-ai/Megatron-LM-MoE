# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

##
# Compile megatron.core.datasets.helpers_cpp dependencies before BlendedDataset import
##

import random

import numpy
import pytest
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.utils import compile_helpers
from megatron.core.tokenizers import MegatronTokenizer
from tests.unit_tests.test_utilities import Utils

_MOCK_VOCAB_SIZE = 8192


def sample_N(dataset, N, randomize):
    if randomize:
        indices = [random.randint(0, len(dataset) - 1) for _ in range(N)]
    else:
        indices = list(range(N))
    samples = [dataset[index]["tokens"].numpy() for index in indices]
    return samples


def test_mock_gpt_dataset():
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=_MOCK_VOCAB_SIZE
    )

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
        split="990,9,1",
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [100, 100, 100], lambda: True, config
    ).build()

    N = 10

    # Check iso-index variance by split
    subsets = [sample_N(dataset, N, randomize=False) for dataset in datasets]
    assert not numpy.allclose(subsets[0], subsets[1])
    assert not numpy.allclose(subsets[0], subsets[2])
    assert not numpy.allclose(subsets[1], subsets[2])

    # Check iso-split / iso-index identity
    subset_1A = sample_N(datasets[0], N, randomize=False)
    subset_1B = sample_N(datasets[0], N, randomize=False)
    assert numpy.allclose(subset_1A, subset_1B)

    # Check iso-split variance by index
    subset_1A = sample_N(datasets[0], N, randomize=True)
    subset_1B = sample_N(datasets[0], N, randomize=True)
    assert not numpy.allclose(subset_1A, subset_1B)

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
        split="990,10,0",
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        drop_last_partial_validation_sequence=False,
        add_extra_token_to_sequence=False,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [0, None, 0], lambda: True, config
    ).build()

    sample = datasets[1][datasets[1].shuffle_index.argmax()]
    argmax = sample['labels'].shape[0] - torch.flip(sample['labels'], [0]).argmax() - 1

    # Test add_extra_token_to_sequence
    assert sample['tokens'][argmax] != tokenizer.eod
    assert sample['labels'][argmax] == tokenizer.eod

    # Test eod_mask_loss, drop_last_partial_validation_sequence
    assert argmax < sample['labels'].shape[0] - 1
    assert torch.all(sample['labels'][argmax + 1 :] == 0)
    assert not torch.any(
        sample['loss_mask'][
            torch.logical_and(sample['labels'] == tokenizer.eod, sample['labels'] == 0)
        ]
    )

    sample = datasets[1][None]

    # Check handling of None index
    assert not torch.any(sample['loss_mask'])


def test_apply_goldfish():
    from megatron.core.datasets.gpt_dataset import (
        _GOLDFISH_TOKEN_ID,
        _create_exemption_lut,
        _create_hash_table,
        apply_goldfish,
    )

    torch.manual_seed(0)
    seq_length, k, h = 8192, 50, 50
    labels = torch.randint(0, _MOCK_VOCAB_SIZE, (seq_length,), dtype=torch.long)
    table = _create_hash_table(device=labels.device)

    original = labels.clone()
    out = apply_goldfish(labels, _GOLDFISH_TOKEN_ID, k, table, goldfish_context_width=h)

    # apply_goldfish works on a clone: the input labels are not mutated.
    assert torch.equal(labels, original)

    # Deterministic: identical inputs -> identical drops.
    out2 = apply_goldfish(labels, _GOLDFISH_TOKEN_ID, k, table, goldfish_context_width=h)
    assert torch.equal(out, out2)

    # The first h-1 positions are never eligible for dropping (unfold alignment).
    assert not torch.any(out[: h - 1] == _GOLDFISH_TOKEN_ID)

    # Drop rate over the eligible tail is approximately 1/k.
    drop_rate = (out[h - 1 :] == _GOLDFISH_TOKEN_ID).float().mean().item()
    assert abs(drop_rate - 1.0 / k) < 0.01, drop_rate

    # Exemption: no dropped position may carry a label in the exempt band, while drops
    # still occur outside it. `out` above is the positive control (drops with no exemption).
    assert torch.any(out == _GOLDFISH_TOKEN_ID)
    lo, hi = 2000, 6000
    exemption_lut = _create_exemption_lut(list(range(lo, hi)), _MOCK_VOCAB_SIZE, labels.device)
    out_exempt = apply_goldfish(
        labels, _GOLDFISH_TOKEN_ID, k, table, goldfish_context_width=h, exemption_lut=exemption_lut
    )
    dropped = out_exempt == _GOLDFISH_TOKEN_ID
    assert not torch.any((labels >= lo) & (labels < hi) & dropped), "exempt-band token was dropped"
    assert torch.any(dropped), "expected drops outside the exempt band"

    # Pinned position: exempting exactly the label id of a known dropped position
    # cancels that drop (deterministic alignment of window hash vs. exempt mask).
    pinned = int((out == _GOLDFISH_TOKEN_ID).nonzero(as_tuple=True)[0][0])
    pin_lut = _create_exemption_lut([int(labels[pinned])], _MOCK_VOCAB_SIZE, labels.device)
    out_pinned = apply_goldfish(
        labels, _GOLDFISH_TOKEN_ID, k, table, goldfish_context_width=h, exemption_lut=pin_lut
    )
    assert out_pinned[pinned] != _GOLDFISH_TOKEN_ID

    # Windows containing id-0 labels (pad/unk) must hash like any other window: with a
    # zero every 10 positions every window contains one, yet the rate stays ~1/k (the
    # old product hash collapsed all such windows onto a single drop decision).
    zero_heavy = labels.clone()
    zero_heavy[::10] = 0
    out_zero = apply_goldfish(zero_heavy, _GOLDFISH_TOKEN_ID, k, table, goldfish_context_width=h)
    zero_rate = (out_zero[h - 1 :] == _GOLDFISH_TOKEN_ID).float().mean().item()
    assert abs(zero_rate - 1.0 / k) < 0.01, zero_rate


def test_mock_gpt_dataset_goldfish():
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=_MOCK_VOCAB_SIZE
    )
    # Cache-friendly flags on purpose: goldfish alone must not disable the mask cache,
    # and interleaved access must not leak one sample's drops into another (the cache
    # hands out clones).
    base = dict(
        random_seed=1234,
        sequence_length=1024,
        split="990,9,1",
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
    )

    def build(**overrides):
        config = GPTDatasetConfig(**base, **overrides)
        return BlendedMegatronDatasetBuilder(
            MockGPTDataset, [100, 100, 100], lambda: True, config
        ).build()

    goldfish_sets = build(goldfish_loss=True, goldfish_k=4, goldfish_h=13)
    plain_sets = build()
    ds, plain = goldfish_sets[0], plain_sets[0]

    assert ds.masks_and_position_ids_are_cacheable

    # Goldfish zeroes a strict superset of the plain loss mask (~1/k of the tail).
    gf_mask, plain_mask = ds[0]["loss_mask"].clone(), plain[0]["loss_mask"]
    assert not torch.any((gf_mask == 1) & (plain_mask == 0))
    n_extra = int(((gf_mask == 0) & (plain_mask == 1)).sum())
    assert n_extra > 0, "goldfish produced no drops"

    # Same index -> identical mask; interleaving other samples must not accumulate
    # zeros through the cache (fresh dataset accessed in a different order agrees).
    _ = ds[1]
    assert torch.equal(ds[0]["loss_mask"], gf_mask)
    fresh_sets = build(goldfish_loss=True, goldfish_k=4, goldfish_h=13)
    assert torch.equal(fresh_sets[0][1]["loss_mask"], ds[1]["loss_mask"])

    # The idx-None batch-padding sample stays fully masked.
    assert not torch.any(goldfish_sets[1][None]["loss_mask"])


def test_goldfish_config_validation():
    tokenizer = MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=_MOCK_VOCAB_SIZE
    )
    base = dict(
        random_seed=1234,
        sequence_length=1024,
        split="990,9,1",
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
    )

    # A valid goldfish config constructs without error.
    GPTDatasetConfig(goldfish_loss=True, goldfish_k=50, goldfish_h=50, **base)

    # k must be >= 2 (k=1 drops ~100% of tokens).
    with pytest.raises(AssertionError):
        GPTDatasetConfig(goldfish_loss=True, goldfish_k=1, goldfish_h=50, **base)

    # h must be < sequence_length (else the unfold has no valid window).
    with pytest.raises(AssertionError):
        GPTDatasetConfig(goldfish_loss=True, goldfish_k=50, goldfish_h=1024, **base)


if __name__ == "__main__":
    test_mock_gpt_dataset()
    test_apply_goldfish()
    test_mock_gpt_dataset_goldfish()
    test_goldfish_config_validation()
