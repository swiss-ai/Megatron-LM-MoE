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
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.tokenizers.test_modality_lut import vision_audio_modalities

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
    assert torch.all(sample['labels'][argmax + 1 :] == -100)
    assert not torch.any(sample['loss_mask'][argmax + 1 :])

    sample = datasets[1][None]

    # Check handling of None index
    assert not torch.any(sample['loss_mask'])
    assert torch.all(sample['labels'] == -100)


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

    # Both in-base and separate-range structure IDs are represented by the same
    # scattered LUT.
    scattered_lut = _create_exemption_lut([0, 18, 2000, 6000], _MOCK_VOCAB_SIZE, labels.device)
    assert torch.all(scattered_lut[torch.tensor([0, 18, 2000, 6000])])

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


def test_mock_gpt_dataset_goldfish_with_modality_weights():
    """Composition through __getitem__: combined mask == goldfish mask * weight LUT.

    In particular goldfish drops survive weighting (0 * w == 0) and weighting applies
    to positions goldfish left untouched.
    """
    from megatron.core.datasets.gpt_dataset import _create_modality_weight_lut
    from megatron.core.tokenizers.utils.tokenizer_extra_metadata import parse_omni_metadata

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
    # Omni layout inside the mock vocab: text ids [0, 100), vision content [1000, 1500).
    # The mock data cycles through low ids, so samples contain structure ids (18/27/28,
    # goldfish-exempt) and vision content ids (goldfish-eligible, weighted).
    omni = parse_omni_metadata(
        100,
        {
            "omni_special_token_offset": 100,
            "modalities": [
                {
                    "name": "vision",
                    "offset": 1000,
                    "vocab_size": 500,
                    "start_token": 27,
                    "end_token": 28,
                    "structure_token_ids": {
                        "<|image|>": 18,
                        "<|img_start|>": 27,
                        "<|img_end|>": 28,
                    },
                }
            ],
        },
    )
    metadata = TokenizerExtraMetadata(
        special_tokens=ModelSpecialTokens(full_ids=[18, 27, 28]), omni=omni
    )
    base = dict(
        random_seed=1234,
        sequence_length=1024,
        split="990,9,1",
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=tokenizer,
        tokenizer_extra_metadata=metadata,
        mid_level_dataset_surplus=0.005,
        goldfish_loss=True,
        goldfish_k=4,
        goldfish_h=13,
    )

    def build(**overrides):
        config = GPTDatasetConfig(**base, **overrides)
        return BlendedMegatronDatasetBuilder(
            MockGPTDataset, [100, 100, 100], lambda: True, config
        ).build()

    weighted = build(modality_weights={"vision": 0.25})[0]
    goldfish_only = build()[0]
    lut = _create_modality_weight_lut(
        [(omni.modality("vision"), 0.25)], _MOCK_VOCAB_SIZE, torch.device("cpu")
    )

    saw_drop = saw_weight = False
    for index in range(16):
        sample_w, sample_g = weighted[index], goldfish_only[index]
        assert torch.equal(sample_w["labels"], sample_g["labels"])
        # Exact composition: weighting scales the goldfish-processed mask elementwise,
        # so goldfish zeros stay zero under any weight.
        assert torch.equal(sample_w["loss_mask"], sample_g["loss_mask"] * lut[sample_w["labels"]])
        saw_drop = saw_drop or bool(torch.any(sample_g["loss_mask"] == 0))
        saw_weight = saw_weight or bool(torch.any(lut[sample_w["labels"]] != 1.0))
    # The checked samples exercise both branches: real drops and real down-weighted ids.
    assert saw_drop and saw_weight


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


def test_modality_weight_lut():
    from megatron.core.datasets.gpt_dataset import (
        _create_modality_weight_lut,
        _get_modality_weight_lut,
    )
    from megatron.core.tokenizers.utils.tokenizer_extra_metadata import ModalityInfo

    vision = ModalityInfo(
        name="vision",
        offset=1000,
        vocab_size=100,
        start_token=5,
        end_token=7,
        structure_token_ids={"<|img_start|>": 5, "<|img_end|>": 7},
    )
    audio = ModalityInfo(
        name="audio",
        offset=1100,
        vocab_size=50,
        start_token=9,
        end_token=9,
        structure_token_ids={"<|audio|>": 9},
    )
    lut = _create_modality_weight_lut(
        [(vision, 0.25), (audio, 0.0)], vocab_size=2000, device=torch.device("cpu")
    )

    # Text ids stay 1.0; content ranges AND per-modality structure ids get the weight.
    labels = torch.tensor([0, 5, 7, 9, 999, 1000, 1099, 1100, 1149, 1150])
    expected = torch.tensor([1.0, 0.25, 0.25, 0.0, 1.0, 0.25, 0.25, 0.0, 0.0, 1.0])
    assert torch.equal(lut[labels], expected)

    # Application semantics: multiplicative on loss_mask, existing zeros stay zero.
    loss_mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert torch.equal(
        loss_mask * lut[labels], torch.tensor([1.0, 0.25, 0.0, 0.0, 1.0, 0.25, 0.25, 0.0, 0.0, 1.0])
    )

    # The module memo hands out one shared table per (weights, vocab, device).
    pairs = ((vision, 0.25), (audio, 0.0))
    memoized = _get_modality_weight_lut(pairs, 2000, torch.device("cpu"))
    assert _get_modality_weight_lut(pairs, 2000, torch.device("cpu")) is memoized


def test_modality_weights_config_validation():
    from megatron.core.tokenizers.utils.tokenizer_extra_metadata import parse_omni_metadata

    omni = parse_omni_metadata(
        100,
        {
            "omni_special_token_offset": 100,
            "modalities": [
                {
                    "name": "vision",
                    "offset": 100,
                    "vocab_size": 1024,
                    "start_token": 27,
                    "end_token": 28,
                    "structure_token_ids": {
                        "<|image|>": 18,
                        "<|img_start|>": 27,
                        "<|img_end|>": 28,
                    },
                }
            ],
        },
    )
    metadata = TokenizerExtraMetadata(
        special_tokens=ModelSpecialTokens(full_ids=[18, 27, 28]), omni=omni
    )
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

    GPTDatasetConfig(modality_weights={"vision": 0.25}, tokenizer_extra_metadata=metadata, **base)

    # Omni metadata is required.
    with pytest.raises(AssertionError, match="requires tokenizer_extra_metadata"):
        GPTDatasetConfig(modality_weights={"vision": 0.25}, tokenizer_extra_metadata=None, **base)
    text_only = TokenizerExtraMetadata(special_tokens=ModelSpecialTokens(full_ids=[1]))
    with pytest.raises(AssertionError, match="requires tokenizer_extra_metadata"):
        GPTDatasetConfig(
            modality_weights={"vision": 0.25}, tokenizer_extra_metadata=text_only, **base
        )

    with pytest.raises(AssertionError, match="does not describe"):
        GPTDatasetConfig(
            modality_weights={"video": 0.25}, tokenizer_extra_metadata=metadata, **base
        )

    with pytest.raises(AssertionError, match=">= 0"):
        GPTDatasetConfig(
            modality_weights={"vision": -1.0}, tokenizer_extra_metadata=metadata, **base
        )


def test_modality_loss_report():
    """Per-group metrics partition lm loss."""
    from pretrain_gpt import modality_loss_report

    from megatron.core.tokenizers.utils.tokenizer_extra_metadata import ModalityInfo

    vision = ModalityInfo(
        name="vision",
        offset=100,
        vocab_size=10,
        start_token=5,
        end_token=6,
        structure_token_ids={"<|img_start|>": 5, "<|img_end|>": 6},
    )

    # Two text and four vision targets.
    labels = torch.tensor([1, 2, 100, 105, 109, 5])
    losses = torch.tensor([2.0, 4.0, 6.0, 6.0, 6.0, 6.0])
    loss_mask = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.5, 0.5])

    report = modality_loss_report(losses, loss_mask, labels, [vision], 200)
    assert set(report) == {"vision loss", "vision error", "text loss", "text error"}

    assert torch.equal(report["vision loss"], torch.tensor([12.0, 2.0]))
    assert torch.equal(report["vision error"], torch.tensor([24.0, 4.0]))
    assert report["vision loss"][0] / report["vision loss"][1] == 6.0
    assert report["vision error"][0] / report["vision error"][1] == 6.0

    assert torch.equal(report["text loss"], torch.tensor([6.0, 2.0]))
    assert torch.equal(report["text error"], torch.tensor([6.0, 2.0]))

    # Token groups partition lm loss.
    lm_sum = torch.sum(losses * loss_mask)
    lm_count = loss_mask.sum()
    assert report["vision loss"][0] + report["text loss"][0] == lm_sum
    assert report["vision loss"][1] + report["text loss"][1] == lm_count

    # Raw error survives a zero weight.
    zeroed = modality_loss_report(losses, torch.zeros_like(loss_mask), labels, [vision], 200)
    assert torch.equal(zeroed["vision loss"], torch.tensor([0.0, 0.0]))
    assert torch.equal(zeroed["vision error"], torch.tensor([24.0, 4.0]))

    # Ignored targets do not become text.
    sft_labels = torch.tensor([1, 2, 100, 105, 109, -100])
    sft_loss_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    sft = modality_loss_report(losses, sft_loss_mask, sft_labels, [vision], 200)
    assert torch.equal(sft["text error"], torch.tensor([6.0, 2.0]))
    assert torch.equal(sft["text loss"], torch.tensor([6.0, 2.0]))
    sft_lm_sum = torch.sum(losses * sft_loss_mask)
    sft_lm_count = sft_loss_mask.sum()
    assert sft["vision loss"][0] + sft["text loss"][0] == sft_lm_sum
    assert sft["vision loss"][1] + sft["text loss"][1] == sft_lm_count

    # Supervised counts exclude zero-mask targets.
    sup_loss_mask = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.5, 0.0])
    sup = modality_loss_report(
        losses, sup_loss_mask, labels, [vision], 200, normalize_by_num_supervised_tokens=True
    )
    assert torch.equal(sup["vision loss"], torch.tensor([9.0, 3.0]))
    assert torch.equal(sup["vision error"], torch.tensor([24.0, 4.0]))
    assert sup["vision loss"][0] + sup["text loss"][0] == torch.sum(losses * sup_loss_mask)
    assert sup["vision loss"][1] + sup["text loss"][1] == (sup_loss_mask > 0).sum()


def test_loss_func_normalize_by_num_supervised_tokens():
    """num_tokens semantics of loss_func with and without the normalization flag."""
    from types import SimpleNamespace

    from megatron.training import global_vars
    from megatron.training.global_vars import set_args

    def run_loss_func(
        normalize, loss_mask, output_tensor, *, labels=None, metadata=None, log_modalities=False
    ):
        from pretrain_gpt import loss_func

        from megatron.core.rerun_state_machine import destroy_rerun_state_machine

        saved_args = global_vars._GLOBAL_ARGS
        set_args(
            SimpleNamespace(
                normalize_by_num_supervised_tokens=normalize,
                check_for_nan_in_loss_and_grad=False,
                check_for_spiky_loss=False,
                tokenizer_extra_metadata=metadata,
                modelopt_enabled=False,
                log_per_modality_loss=log_modalities,
                padded_vocab_size=2000,
            )
        )
        try:
            return loss_func(loss_mask, output_tensor, labels=labels)
        finally:
            set_args(saved_args)
            # Reset implicit test state.
            destroy_rerun_state_machine()

    losses = torch.tensor([2.0, 4.0, 6.0, 8.0])
    loss_mask = torch.tensor([1.0, 0.5, 0.0, 1.0])

    loss, num_tokens, report = run_loss_func(False, loss_mask, losses)
    assert loss == 12.0
    assert num_tokens.dtype == torch.int and num_tokens == 2
    assert torch.equal(report["lm loss"], torch.tensor([12.0, 2.0]))

    loss, num_tokens, report = run_loss_func(True, loss_mask, losses)
    assert loss == 12.0
    assert num_tokens.dtype == torch.int and num_tokens == 3
    assert torch.equal(report["lm loss"], torch.tensor([12.0, 3.0]))

    # Binary masks are unchanged.
    binary_mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
    loss_off, num_off, report_off = run_loss_func(False, binary_mask, losses)
    loss_on, num_on, report_on = run_loss_func(True, binary_mask, losses)
    assert loss_off == loss_on == 14.0
    assert num_off == num_on == 3
    assert torch.equal(report_off["lm loss"], report_on["lm loss"])

    zero_weight_mask = torch.tensor([1.0, 0.0, 0.0, 1.0])
    _, num_off, _ = run_loss_func(False, zero_weight_mask, losses)
    _, num_on, _ = run_loss_func(True, zero_weight_mask, losses)
    assert num_off == num_on == 2

    # Reporting is opt-in.
    omni = SimpleNamespace(modalities=vision_audio_modalities())
    metadata = SimpleNamespace(omni=omni)
    labels = torch.tensor([1, 1000, 1100, 2])
    _, _, report_off = run_loss_func(False, binary_mask, losses, labels=labels, metadata=metadata)
    _, _, report_on = run_loss_func(
        False, binary_mask, losses, labels=labels, metadata=metadata, log_modalities=True
    )
    assert set(report_off) == {"lm loss"}
    assert set(report_on) == {
        "lm loss",
        "vision loss",
        "vision error",
        "audio loss",
        "audio error",
        "text loss",
        "text error",
    }
    assert torch.equal(report_on["vision loss"], torch.tensor([4.0, 1.0]))
    assert torch.equal(report_on["vision error"], torch.tensor([4.0, 1.0]))
    assert torch.equal(report_on["audio loss"], torch.tensor([0.0, 0.0]))
    assert torch.equal(report_on["audio error"], torch.tensor([6.0, 1.0]))
    assert torch.equal(report_on["text loss"], torch.tensor([10.0, 2.0]))
    assert torch.equal(report_on["text error"], torch.tensor([10.0, 2.0]))
    assert torch.equal(
        report_on["vision loss"] + report_on["audio loss"] + report_on["text loss"],
        report_on["lm loss"],
    )

    _, _, normalized_report = run_loss_func(
        True, loss_mask, losses, labels=labels, metadata=metadata, log_modalities=True
    )
    assert set(normalized_report) == set(report_on)
    assert torch.equal(normalized_report["vision loss"], torch.tensor([2.0, 1.0]))
    assert torch.equal(normalized_report["vision error"], torch.tensor([4.0, 1.0]))
    assert torch.equal(normalized_report["audio loss"], torch.tensor([0.0, 0.0]))
    assert torch.equal(normalized_report["audio error"], torch.tensor([6.0, 1.0]))
    assert torch.equal(normalized_report["text loss"], torch.tensor([10.0, 2.0]))
    assert torch.equal(normalized_report["text error"], torch.tensor([10.0, 2.0]))
    assert torch.equal(
        normalized_report["vision loss"]
        + normalized_report["audio loss"]
        + normalized_report["text loss"],
        normalized_report["lm loss"],
    )


def test_modality_weights_require_safe_normalization():
    from types import SimpleNamespace

    from megatron.training.arguments import _validate_modality_loss_args

    def args(weight, normalize=False, per_token=False):
        return SimpleNamespace(
            vision_weight=weight,
            audio_weight=1.0,
            normalize_by_num_supervised_tokens=normalize,
            calculate_per_token_loss=per_token,
        )

    _validate_modality_loss_args(args(1.0))
    # Weight 0.0 needs per-token loss only: the mask stays binary, so the
    # supervised-count and mask-sum denominators coincide.
    _validate_modality_loss_args(args(0.0, per_token=True))
    _validate_modality_loss_args(args(0.5, normalize=True, per_token=True))

    # Any non-default weight without per-token loss is rejected (local per-microbatch
    # normalization would redistribute the masked-out weight).
    with pytest.raises(AssertionError, match="calculate-per-token-loss"):
        _validate_modality_loss_args(args(0.0))
    with pytest.raises(AssertionError, match="calculate-per-token-loss"):
        _validate_modality_loss_args(args(0.5, normalize=True))
    # Fractional weights additionally need the supervised-count denominator.
    with pytest.raises(AssertionError, match="normalize-by-num-supervised-tokens"):
        _validate_modality_loss_args(args(0.5, per_token=True))
    with pytest.raises(AssertionError, match="must be finite"):
        _validate_modality_loss_args(args(float("inf"), normalize=True, per_token=True))


if __name__ == "__main__":
    test_mock_gpt_dataset()
    test_apply_goldfish()
    test_mock_gpt_dataset_goldfish()
    test_mock_gpt_dataset_goldfish_with_modality_weights()
    test_goldfish_config_validation()
    test_modality_weight_lut()
    test_modality_weights_config_validation()
    test_modality_loss_report()
    test_loss_func_normalize_by_num_supervised_tokens()
    test_modality_weights_require_safe_normalization()
