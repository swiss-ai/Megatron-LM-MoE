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
from megatron.core.datasets.utils import Split, compile_helpers
from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.tokenizers.utils.tokenizer_extra_metadata import (
    ModelSpecialTokens,
    TokenizerExtraMetadata,
)
from megatron.core.utils import (
    _merge_cu_seqlens_across_micro_batch,
    flatten_batch_for_packed_sequences,
)
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


def test_bfd_dataset_emits_compact_padding_boundary():
    """The compact BFD metadata must not classify the last real token as padding."""
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
        sequence_length=4,
        split="990,9,1",
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
        pretraining_packing_strategy="bfd",
        inter_document_masking=True,
    )
    dataset = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [10, 0, 0], lambda: True, config
    ).build()[0]

    pad = dataset._pad_token_id
    dataset._query_bfd_packed_sample = lambda _idx: (
        numpy.array([11, 12, 13, pad, pad], dtype=numpy.int64),
        numpy.array([0], dtype=numpy.int64),
        [3],
    )
    sample = dataset[0]

    assert sample["num_valid_tokens"].dtype == torch.int32
    assert sample["num_valid_tokens"].item() == 3
    assert torch.equal(sample["tokens"], torch.tensor([11, 12, 13, 0]))
    assert "cu_seqlens" in sample
    # This real token predicts the first pad, so it is absent from LM loss but
    # remains inside the routing-valid prefix [0:num_valid_tokens].
    assert sample["loss_mask"][2] == 0


def test_bfd_num_valid_tokens_array_matches_samples(tmp_path):
    """The per-sample num_valid_tokens array that middle pipeline stages read equals what
    __getitem__ emits for every sample, and NumValidTokensDataset serves it."""
    from megatron.core.datasets import gpt_dataset

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
        sequence_length=64,
        split="990,9,1",
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
        pretraining_packing_strategy="bfd",
        path_to_cache=str(tmp_path),
    )
    dataset = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [300, 0, 0], lambda: True, config
    ).build()[0]

    gpt_dataset.save_num_valid_tokens(dataset, "train")
    stand_in = gpt_dataset.NumValidTokensDataset(
        *gpt_dataset.NUM_VALID_TOKENS_PATHS["train"], Split.train
    )
    assert len(stand_in) == len(dataset) and stand_in.index_split == Split.train
    expected = [int(dataset[i]["num_valid_tokens"]) for i in range(len(dataset))]
    assert [int(stand_in[i]["num_valid_tokens"]) for i in range(len(dataset))] == expected
    assert min(expected) > 0 and max(expected) <= 64
    assert stand_in[0]["num_valid_tokens"].dtype == torch.int32


def test_bfd_num_valid_tokens_array_matches_blend(tmp_path):
    """Same for a blend of real indexed datasets, with documents longer than a sample, a changed
    max_docs_per_bin, rebuilt BFD indices, and lazily loaded blend indices."""
    import os

    from megatron.core.datasets import gpt_dataset
    from megatron.core.datasets.gpt_dataset import GPTDataset
    from megatron.core.datasets.indexed_dataset import DType, IndexedDatasetBuilder

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
    # 200 documents of 20-150 tokens per file with 64-token samples: whole documents and split chunks.
    rng = numpy.random.default_rng(1234)
    os.makedirs(tmp_path / "data")
    prefixes = []
    for i in range(3):
        prefix = str(tmp_path / "data" / f"file_{i}")
        builder = IndexedDatasetBuilder(prefix + ".bin", dtype=DType.optimal_dtype(_MOCK_VOCAB_SIZE))
        for _ in range(200):
            document = rng.integers(0, _MOCK_VOCAB_SIZE - 1, rng.integers(20, 150))
            builder.add_document(torch.from_numpy(document), [len(document)])
        builder.finalize(prefix + ".idx")
        prefixes.append(prefix)

    def build(max_docs_per_bin, defer=False):
        config = GPTDatasetConfig(
            random_seed=1234,
            sequence_length=64,
            blend=(prefixes, [0.5, 0.3, 0.2]),
            split="100,0,0",
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            tokenizer=tokenizer,
            mid_level_dataset_surplus=0.005,
            pretraining_packing_strategy="bfd",
            max_docs_per_bin=max_docs_per_bin,
            path_to_cache=str(tmp_path / "cache"),
            defer_npy_index_mmap=defer,
        )
        return BlendedMegatronDatasetBuilder(GPTDataset, [100, 0, 0], lambda: True, config).build()[0]

    def check(dataset):
        gpt_dataset.save_num_valid_tokens(dataset, "train")
        path, length = gpt_dataset.NUM_VALID_TOKENS_PATHS["train"]
        stand_in = gpt_dataset.NumValidTokensDataset(path, length, Split.train)
        assert len(stand_in) == len(dataset)
        expected = [int(dataset[i]["num_valid_tokens"]) for i in range(len(dataset))]
        assert [int(stand_in[i]["num_valid_tokens"]) for i in range(len(dataset))] == expected
        return path

    path_uncapped = check(build(0))
    path_capped = check(build(1))  # one document per bin: other bins, so another array
    assert path_capped != path_uncapped

    # BFD indices newer than the array (rebuilt since) -> the array is rebuilt.
    dataset = build(1)
    mtime = os.path.getmtime(path_capped)
    for ds in dataset.datasets:
        os.utime(ds.path_to_bfd_docs, (mtime + 10, mtime + 10))
    assert check(dataset) == path_capped and os.path.getmtime(path_capped) > mtime

    # Lazily loaded blend indices (defer_npy_index_mmap), from the cache written above.
    before = numpy.load(path_capped)
    os.remove(path_capped)
    deferred = build(1, defer=True)
    assert deferred.dataset_index is None
    gpt_dataset.save_num_valid_tokens(deferred, "train")
    assert gpt_dataset.NUM_VALID_TOKENS_PATHS["train"] == (path_capped, len(dataset))
    assert numpy.array_equal(numpy.load(path_capped), before)


def _build_masked_blend(tmp_path, packing, add_extra_token, defer_npy_index_mmap=False, single=False):
    """Two indexed datasets with documents of 1 to 150 tokens (plus 1-token documents and documents
    longer than a 64-token sample), blended 0.6 / 0.4 (or the first one alone with `single`), packed with `packing`,
    inter-document masking on."""
    import os

    from megatron.core.datasets.gpt_dataset import GPTDataset
    from megatron.core.datasets.indexed_dataset import DType, IndexedDatasetBuilder

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
    prefixes = [str(tmp_path / "data" / f"file_{i}") for i in range(2)]
    if not os.path.isdir(tmp_path / "data"):
        rng = numpy.random.default_rng(1234)
        os.makedirs(tmp_path / "data")
        for prefix in prefixes:
            builder = IndexedDatasetBuilder(prefix + ".bin", dtype=DType.optimal_dtype(_MOCK_VOCAB_SIZE))
            lengths = list(rng.integers(1, 150, 150)) + [1, 1, 2, 64, 65, 129]
            for n in lengths:
                document = rng.integers(0, _MOCK_VOCAB_SIZE - 1, int(n))
                builder.add_document(torch.from_numpy(document), [len(document)])
            builder.finalize(prefix + ".idx")
    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=64,
        blend=(prefixes[:1], None) if single else (prefixes, [0.6, 0.4]),
        split="80,10,10",
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
        pretraining_packing_strategy=packing,
        inter_document_masking=True,
        add_extra_token_to_sequence=add_extra_token,
        path_to_cache=str(tmp_path / "cache"),
        defer_npy_index_mmap=defer_npy_index_mmap,
    )
    # within one BFD epoch of the blend (or of the single file)
    sizes = [60, 8, 8] if single else [150, 20, 20]
    return BlendedMegatronDatasetBuilder(GPTDataset, sizes, lambda: True, config).build()


def _check_stand_in(dataset, key):
    """The stand-in of a middle pipeline stage serves what __getitem__ serves, per sample and merged
    across a micro-batch of two as the model receives it. Returns the document-length array paths."""
    from torch.utils.data._utils.collate import default_collate

    from megatron.core.datasets import gpt_dataset

    gpt_dataset.NUM_VALID_TOKENS_PATHS.clear()
    gpt_dataset.DOC_LENGTHS_PATHS.clear()
    if dataset.config.pretraining_packing_strategy == "bfd":
        gpt_dataset.save_num_valid_tokens(dataset, key)
    else:
        gpt_dataset.save_greedy_document_lengths(dataset, key)
    path, length = gpt_dataset.NUM_VALID_TOKENS_PATHS[key]
    doc_paths = gpt_dataset.DOC_LENGTHS_PATHS[key]
    split = getattr(dataset, "split", None) or dataset.index_split  # BlendedDataset / GPTDataset
    stand_in = gpt_dataset.NumValidTokensDataset(path, length, split, doc_paths=doc_paths, seq_length=64)
    assert len(stand_in) == len(dataset) > 0
    for i in range(len(dataset)):
        real, got = dataset[i], stand_in[i]
        assert torch.equal(real["cu_seqlens"], got["cu_seqlens"]) and got["cu_seqlens"].dtype == torch.int32
        assert int(real["max_seqlen"]) == int(got["max_seqlen"]) and got["max_seqlen"].dtype == torch.int32
        if path is None:  # greedy packing: no padding, no num_valid_tokens
            assert "num_valid_tokens" not in real and "num_valid_tokens" not in got
        else:
            assert int(real["num_valid_tokens"]) == int(got["num_valid_tokens"])
    for i in range(0, min(len(dataset), 20) - 1, 2):
        real = flatten_batch_for_packed_sequences(default_collate([dataset[i], dataset[i + 1]]))
        got = flatten_batch_for_packed_sequences(default_collate([stand_in[i], stand_in[i + 1]]))
        assert torch.equal(real["cu_seqlens"], got["cu_seqlens"])
        assert torch.equal(real["max_seqlen"], got["max_seqlen"])
    return doc_paths


@pytest.mark.parametrize("packing", ["bfd", "greedy"])
@pytest.mark.parametrize("add_extra_token", [True, False])
def test_document_lengths_arrays_match_samples_with_masking(tmp_path, packing, add_extra_token):
    """With inter-document masking, the stand-in that middle pipeline stages use serves the same
    cu_seqlens, max_seqlen and (BFD) num_valid_tokens as __getitem__, for every sample of every split
    of a packed blend, including documents longer than a sample and 1-token documents, whose adjusted
    length can be zero."""
    import os

    datasets = _build_masked_blend(tmp_path, packing, add_extra_token)
    for key, dataset in zip(("train", "valid", "test"), datasets):
        offsets_path, lengths_path = _check_stand_in(dataset, key)
        offsets, lengths = numpy.load(offsets_path), numpy.load(lengths_path)
        assert f"-{key}-{packing}-" in os.path.basename(offsets_path)
        assert offsets.dtype == numpy.int64 and lengths.dtype == numpy.int32
        assert len(offsets) >= len(dataset) + 1 and offsets[-1] == len(lengths)
        assert numpy.all(numpy.diff(offsets) >= 1) and numpy.all(lengths >= 0)
    if packing == "greedy":
        assert not any(f.endswith("num_valid_tokens.npy") for f in os.listdir(tmp_path / "cache"))


@pytest.mark.parametrize("packing", ["bfd", "greedy"])
def test_document_lengths_arrays_single_dataset_layout(tmp_path, packing):
    """A single indexed dataset (no blend) takes the per-component arrays as they are: same equality with
    __getitem__ and the same layout (int64 offsets, one row per sample, int32 lengths) as a blend."""
    import os

    from megatron.core.datasets import gpt_dataset

    datasets = _build_masked_blend(tmp_path, packing, True, single=True)
    for key, dataset in zip(("train", "valid", "test"), datasets):
        assert not hasattr(dataset, "datasets")
        offsets_path, lengths_path = _check_stand_in(dataset, key)
        offsets, lengths = numpy.load(offsets_path), numpy.load(lengths_path)
        assert offsets.dtype == numpy.int64 and lengths.dtype == numpy.int32
        assert len(offsets) == len(dataset) + 1 and offsets[-1] == len(lengths)
        assert f"-{key}-{packing}-" in os.path.basename(offsets_path)
        if packing == "bfd":
            nvt = numpy.load(gpt_dataset.NUM_VALID_TOKENS_PATHS[key][0])
            assert nvt.dtype == numpy.int32 and len(nvt) == len(dataset)


@pytest.mark.parametrize("packing", ["bfd", "greedy"])
def test_document_lengths_arrays_rebuilt_and_deferred(tmp_path, packing):
    """The arrays are reused from the cache, rebuilt with identical content when a component's index
    is newer, and built from lazily memory-mapped indices (--dataloader-defer-npy-index-mmap)."""
    import os
    import time

    train = _build_masked_blend(tmp_path, packing, True)[0]
    offsets_path, lengths_path = _check_stand_in(train, "train")
    before = numpy.load(lengths_path)
    mtime = os.path.getmtime(lengths_path)
    time.sleep(1.1)
    assert _check_stand_in(train, "train") == (offsets_path, lengths_path)
    assert os.path.getmtime(lengths_path) == mtime
    component = train.datasets[0]
    source = component.path_to_bfd_docs if packing == "bfd" else component.path_to_sample_index
    os.utime(source, (mtime + 10, mtime + 10))
    assert _check_stand_in(train, "train") == (offsets_path, lengths_path)
    assert os.path.getmtime(lengths_path) > mtime
    assert numpy.array_equal(numpy.load(lengths_path), before)

    os.remove(lengths_path)
    deferred = _build_masked_blend(tmp_path, packing, True, defer_npy_index_mmap=True)[0]
    assert deferred.dataset_index is None
    if packing == "greedy":
        assert deferred.datasets[0].sample_index is None
    assert _check_stand_in(deferred, "train") == (offsets_path, lengths_path)
    assert numpy.array_equal(numpy.load(lengths_path), before)


def test_packed_batch_flattens_padding_mask():
    batch = {
        "tokens": torch.arange(8).reshape(2, 4),
        "padding_mask": torch.tensor(
            [[False, False, False, True], [False, False, True, True]]
        ),
        "cu_seqlens": torch.tensor([[0, 3, 4, 4, 4], [0, 2, 4, 4, 4]], dtype=torch.int32),
        "max_seqlen": torch.tensor([3, 2], dtype=torch.int32),
    }

    flattened = flatten_batch_for_packed_sequences(batch)

    assert flattened["padding_mask"].shape == (1, 8)
    assert torch.equal(
        flattened["padding_mask"],
        torch.tensor([[False, False, False, True, False, False, True, True]]),
    )


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


def test_mask_loss_token_ids_masks_targets_only(monkeypatch):
    import pretrain_gpt

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
    masked_dataset = BlendedMegatronDatasetBuilder(
        MockGPTDataset,
        [100, 0, 0],
        lambda: True,
        GPTDatasetConfig(**base, loss_mask_token_ids=(10, 11)),
    ).build()[0]
    baseline_dataset = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [100, 0, 0], lambda: True, GPTDatasetConfig(**base)
    ).build()[0]

    args = type(
        "Args",
        (),
        {"modelopt_enabled": False, "check_for_nan_in_loss_and_grad": False,
         "check_for_spiky_loss": False},
    )()
    monkeypatch.setattr(pretrain_gpt, "get_args", lambda: args)
    monkeypatch.setattr(pretrain_gpt, "has_nvidia_modelopt", False)

    masked_target_count = 0
    for index in range(100):
        sample, baseline = masked_dataset[index], baseline_dataset[index]
        assert torch.equal(sample["labels"], baseline["labels"])
        masked_targets = (sample["labels"] == 10) | (sample["labels"] == 11)
        expected_loss_mask = baseline["loss_mask"].clone()
        expected_loss_mask[masked_targets] = 0.0
        assert torch.equal(sample["loss_mask"], expected_loss_mask)

        masked_target_count += int(masked_targets.sum())
    assert masked_target_count > 0

    # Verify the exact pretraining loss function excludes these targets.
    losses = torch.arange(sample["loss_mask"].numel(), dtype=torch.float)
    loss, num_tokens, _ = pretrain_gpt.loss_func(sample["loss_mask"], losses)
    assert loss == torch.sum(losses * expected_loss_mask)
    assert num_tokens == expected_loss_mask.sum()


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
