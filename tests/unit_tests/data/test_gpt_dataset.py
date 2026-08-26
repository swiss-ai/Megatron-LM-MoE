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
from megatron.core.tokenizers.utils.tokenizer_extra_metadata import (
    ModelSpecialTokens,
    TokenizerExtraMetadata,
)
from megatron.core.utils import _merge_cu_seqlens_across_micro_batch
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


def test_inter_document_masking():
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

    sequence_length = 1024

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=sequence_length,
        split="990,9,1",
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
        inter_document_masking=True,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [100, 100, 100], lambda: True, config
    ).build()

    N = 20
    for idx in range(N):
        sample = datasets[0][idx]

        assert "cu_seqlens" in sample
        assert "max_seqlen" in sample
        assert "attention_mask" not in sample

        # Strip collation padding before validation.
        cu_seqlens = _merge_cu_seqlens_across_micro_batch(
            sample["cu_seqlens"].unsqueeze(0), sequence_length
        )
        max_seqlen = sample["max_seqlen"]
        tokens = sample["tokens"]
        position_ids = sample["position_ids"]

        assert tokens.shape[0] == sequence_length
        assert position_ids.shape[0] == sequence_length

        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens[0] == 0
        assert cu_seqlens[-1] == sequence_length

        # cu_seqlens must be strictly increasing.
        diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        assert torch.all(diffs > 0), f"cu_seqlens not strictly increasing: {cu_seqlens}"

        assert max_seqlen == diffs.max()

        # Position IDs must reset to 0 at each document boundary.
        for i in range(cu_seqlens.numel() - 1):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            expected = torch.arange(end - start, dtype=torch.long)
            assert torch.equal(
                position_ids[start:end], expected
            ), f"position_ids mismatch in segment {i} [{start}:{end}]"

    # Verify that None index zeros out loss_mask.
    sample = datasets[0][None]
    assert not torch.any(sample["loss_mask"])
    assert "cu_seqlens" in sample

    # The token stream itself must be untouched.
    baseline_config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=sequence_length,
        split="990,9,1",
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
        inter_document_masking=False,
    )
    baseline = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [100, 100, 100], lambda: True, baseline_config
    ).build()

    for idx in range(N):
        packed = datasets[0][idx]
        plain = baseline[0][idx]
        assert torch.equal(packed["tokens"], plain["tokens"])
        assert torch.equal(packed["labels"], plain["labels"])
        assert torch.equal(packed["loss_mask"], plain["loss_mask"])
        assert "cu_seqlens" not in plain

    # Samples with different document counts must collate into one batch.
    from torch.utils.data._utils.collate import default_collate

    batch = default_collate([datasets[0][i] for i in range(4)])
    assert batch["cu_seqlens"].shape == (4, sequence_length + 1)
    assert batch["max_seqlen"].shape == (4,)
    merged = _merge_cu_seqlens_across_micro_batch(batch["cu_seqlens"], sequence_length)
    assert merged[0].item() == 0
    assert merged[-1].item() == 4 * sequence_length
    assert torch.all(merged[1:] - merged[:-1] > 0)


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
    text_tokenizer_extra_metadata = TokenizerExtraMetadata(
        special_tokens=ModelSpecialTokens(full_ids=[1, 2, 3])
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
        tokenizer_extra_metadata=text_tokenizer_extra_metadata,
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
    assert ds._goldfish_exemption_ids == (1, 2, 3)

    # Goldfish zeroes a strict superset of the plain loss mask (~1/k of the tail);
    # the returned labels are untouched.
    gf_mask, plain_mask = ds[0]["loss_mask"].clone(), plain[0]["loss_mask"]
    assert not torch.any((gf_mask == 1) & (plain_mask == 0))
    n_extra = int(((gf_mask == 0) & (plain_mask == 1)).sum())
    assert n_extra > 0, "goldfish produced no drops"
    assert torch.equal(ds[0]["labels"], plain[0]["labels"])

    # Exempt ids are never dropped.
    labels = ds[0]["labels"]
    assert not torch.any((gf_mask == 0) & (plain_mask == 1) & (labels <= 3) & (labels >= 1))

    # Train split only: the validation and test splits get no drops.
    for split_index in (1, 2):
        assert torch.equal(
            goldfish_sets[split_index][0]["loss_mask"], plain_sets[split_index][0]["loss_mask"]
        )
        assert torch.equal(
            goldfish_sets[split_index][0]["labels"], plain_sets[split_index][0]["labels"]
        )

    # Same index -> identical mask; interleaving other samples must not accumulate
    # zeros through the cache (fresh dataset accessed in a different order agrees).
    _ = ds[1]
    assert torch.equal(ds[0]["loss_mask"], gf_mask)
    fresh_sets = build(goldfish_loss=True, goldfish_k=4, goldfish_h=13)
    assert torch.equal(fresh_sets[0][1]["loss_mask"], ds[1]["loss_mask"])

    # The idx-None batch-padding sample stays fully masked.
    assert not torch.any(goldfish_sets[1][None]["loss_mask"])

    # Composes with inter-document masking: the goldfish mask is applied before the
    # cu_seqlens return branch, so the same drops appear there.
    idm_sets = build(
        goldfish_loss=True, goldfish_k=4, goldfish_h=13, inter_document_masking=True
    )
    idm_sample = idm_sets[0][0]
    assert "cu_seqlens" in idm_sample
    assert torch.equal(idm_sample["loss_mask"], gf_mask)

    # Exemption ids must index the vocab-sized LUT: fail at dataset build, not lazily
    # in a dataloader worker. (The null tokenizer reports vocab_size + 1 for EOD, so
    # derive the out-of-range id from the tokenizer, not from _MOCK_VOCAB_SIZE.)
    base_bad = dict(base)
    base_bad["tokenizer_extra_metadata"] = TokenizerExtraMetadata(
        special_tokens=ModelSpecialTokens(full_ids=[1, tokenizer.vocab_size])
    )
    # The mock builder rewraps construction errors; the assertion is the chained cause.
    with pytest.raises(Exception, match="failed to build") as excinfo:
        BlendedMegatronDatasetBuilder(
            MockGPTDataset,
            [100, 100, 100],
            lambda: True,
            GPTDatasetConfig(**base_bad, goldfish_loss=True, goldfish_k=4, goldfish_h=13),
        ).build()
    assert isinstance(excinfo.value.__cause__, AssertionError)
    assert "outside the tokenizer vocab" in str(excinfo.value.__cause__)


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

    # h must be a positive context width.
    with pytest.raises(AssertionError):
        GPTDatasetConfig(goldfish_loss=True, goldfish_k=50, goldfish_h=0, **base)


if __name__ == "__main__":
    test_mock_gpt_dataset()
    test_inter_document_masking()
    test_mock_gpt_dataset_goldfish()
    test_goldfish_config_validation()
