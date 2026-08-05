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

    # Valid: the weighted modality is described by the omni metadata.
    GPTDatasetConfig(modality_weights={"vision": 0.25}, tokenizer_extra_metadata=metadata, **base)

    # Weights without omni metadata are rejected (text-only metadata or none at all).
    with pytest.raises(AssertionError, match="requires tokenizer_extra_metadata"):
        GPTDatasetConfig(modality_weights={"vision": 0.25}, tokenizer_extra_metadata=None, **base)
    text_only = TokenizerExtraMetadata(special_tokens=ModelSpecialTokens(full_ids=[1]))
    with pytest.raises(AssertionError, match="requires tokenizer_extra_metadata"):
        GPTDatasetConfig(
            modality_weights={"vision": 0.25}, tokenizer_extra_metadata=text_only, **base
        )

    # An unknown modality name is rejected.
    with pytest.raises(AssertionError, match="does not describe"):
        GPTDatasetConfig(
            modality_weights={"video": 0.25}, tokenizer_extra_metadata=metadata, **base
        )

    # Negative weights are rejected.
    with pytest.raises(AssertionError, match=">= 0"):
        GPTDatasetConfig(
            modality_weights={"vision": -1.0}, tokenizer_extra_metadata=metadata, **base
        )


def test_modality_loss_report():
    """The three per-category metrics, and the partition property against 'lm loss'."""
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

    # 6 positions: 2 text, 3 vision content, 1 vision structure token.
    labels = torch.tensor([1, 2, 100, 105, 109, 5])
    losses = torch.tensor([2.0, 4.0, 6.0, 6.0, 6.0, 6.0])
    # Base mask is all-supervised; vision carries weight 0.5 (as the dataset LUT applies).
    loss_mask = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.5, 0.5])

    report = modality_loss_report(losses, loss_mask, labels, [vision])
    assert set(report) == {
        "vision loss",
        "vision weighted loss",
        "vision error",
        "text loss",
        "text weighted loss",
        "text error",
    }

    # vision: 4 tokens at weight 0.5 -> weighted count 2.0, weighted sum 4*0.5*6.0 = 12.0
    assert torch.equal(report["vision loss"], torch.tensor([12.0, 2.0]))
    assert torch.equal(report["vision weighted loss"], torch.tensor([12.0, 4.0]))
    assert torch.equal(report["vision error"], torch.tensor([24.0, 4.0]))
    # The weight cancels in 'loss' (true mean CE) but not in 'weighted loss'.
    assert report["vision loss"][0] / report["vision loss"][1] == 6.0
    assert report["vision weighted loss"][0] / report["vision weighted loss"][1] == 3.0
    # 'error' survives independently of mask and weight.
    assert report["vision error"][0] / report["vision error"][1] == 6.0

    assert torch.equal(report["text loss"], torch.tensor([6.0, 2.0]))
    assert torch.equal(report["text error"], torch.tensor([6.0, 2.0]))

    # '<name> loss' entries partition 'lm loss' exactly.
    lm_sum = torch.sum(losses * loss_mask)
    lm_count = loss_mask.sum()
    assert report["vision loss"][0] + report["text loss"][0] == lm_sum
    assert report["vision loss"][1] + report["text loss"][1] == lm_count

    # A fully masked modality: 'loss' collapses to 0/0 but 'error' stays informative.
    zeroed = modality_loss_report(losses, torch.zeros_like(loss_mask), labels, [vision])
    assert torch.equal(zeroed["vision loss"], torch.tensor([0.0, 0.0]))
    assert torch.equal(zeroed["vision error"], torch.tensor([24.0, 4.0]))

    # IGNORE_INDEX labels (SFTDataset leaves -100 in the labels for prompt positions)
    # must not be swept into the text category by the complement.
    sft_labels = torch.tensor([1, 2, 100, 105, 109, -100])
    sft_loss_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    sft = modality_loss_report(losses, sft_loss_mask, sft_labels, [vision])
    # Text sees only the 2 real text tokens, not the ignored position.
    assert torch.equal(sft["text error"], torch.tensor([6.0, 2.0]))
    assert torch.equal(sft["text weighted loss"], torch.tensor([6.0, 2.0]))
    # The partition against 'lm loss' still holds: ignored positions carry zero mask.
    sft_lm_sum = torch.sum(losses * sft_loss_mask)
    sft_lm_count = sft_loss_mask.sum()
    assert sft["vision loss"][0] + sft["text loss"][0] == sft_lm_sum
    assert sft["vision loss"][1] + sft["text loss"][1] == sft_lm_count

    # supervised_denominator=True: 'weighted loss' normalizes by the category's
    # supervised-token count, matching the 'lm loss' convention under
    # --normalize-by-num-supervised-tokens. Mask out one vision position (e.g. eod
    # masking) so the supervised count differs from the raw count.
    sup_loss_mask = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.5, 0.0])
    sup = modality_loss_report(losses, sup_loss_mask, labels, [vision], supervised_denominator=True)
    # vision: 3 supervised of 4 total; weighted sum 3*0.5*6.0 = 9.0.
    assert torch.equal(sup["vision weighted loss"], torch.tensor([9.0, 3.0]))
    # 'loss' and 'error' keep their denominators.
    assert torch.equal(sup["vision loss"], torch.tensor([9.0, 1.5]))
    assert torch.equal(sup["vision error"], torch.tensor([24.0, 4.0]))
    # The 'weighted loss' pairs now partition the supervised-count 'lm loss' exactly.
    assert sup["vision weighted loss"][0] + sup["text weighted loss"][0] == torch.sum(
        losses * sup_loss_mask
    )
    assert (
        sup["vision weighted loss"][1] + sup["text weighted loss"][1] == (sup_loss_mask > 0).sum()
    )


def test_loss_func_normalize_by_num_supervised_tokens():
    """num_tokens semantics of loss_func with and without the normalization flag."""
    from types import SimpleNamespace

    from megatron.training import global_vars
    from megatron.training.global_vars import set_args

    def run_loss_func(normalize, loss_mask, output_tensor):
        from pretrain_gpt import loss_func

        from megatron.core.rerun_state_machine import destroy_rerun_state_machine

        saved_args = global_vars._GLOBAL_ARGS
        set_args(
            SimpleNamespace(
                normalize_by_num_supervised_tokens=normalize,
                check_for_nan_in_loss_and_grad=False,
                check_for_spiky_loss=False,
                tokenizer_extra_metadata=None,
                modelopt_enabled=False,
            )
        )
        try:
            return loss_func(loss_mask, output_tensor)
        finally:
            set_args(saved_args)
            # loss_func implicitly initializes the rerun state machine; drop it so a
            # later test doing explicit initialization doesn't hit 'already initialized'.
            destroy_rerun_state_machine()

    losses = torch.tensor([2.0, 4.0, 6.0, 8.0])
    # Fractional mask: weighted sum 2.5 vs 3 supervised positions.
    loss_mask = torch.tensor([1.0, 0.5, 0.0, 1.0])

    loss, num_tokens, report = run_loss_func(False, loss_mask, losses)
    # Default: weighted mask sum, truncated to int (2.5 -> 2).
    assert loss == 12.0
    assert num_tokens.dtype == torch.int and num_tokens == 2
    assert torch.equal(report["lm loss"], torch.tensor([12.0, 2.0]))

    loss, num_tokens, report = run_loss_func(True, loss_mask, losses)
    # Flag: count of supervised (mask > 0) positions; the numerator is unchanged.
    assert loss == 12.0
    assert num_tokens.dtype == torch.int and num_tokens == 3
    assert torch.equal(report["lm loss"], torch.tensor([12.0, 3.0]))

    # Binary masks (every existing text-only run): the flag is a strict no-op.
    binary_mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
    loss_off, num_off, report_off = run_loss_func(False, binary_mask, losses)
    loss_on, num_on, report_on = run_loss_func(True, binary_mask, losses)
    assert loss_off == loss_on == 14.0
    assert num_off == num_on == 3
    assert torch.equal(report_off["lm loss"], report_on["lm loss"])

    # Weight 0.0 (fully masked modality) drops from the count either way, like padding.
    zero_weight_mask = torch.tensor([1.0, 0.0, 0.0, 1.0])
    _, num_off, _ = run_loss_func(False, zero_weight_mask, losses)
    _, num_on, _ = run_loss_func(True, zero_weight_mask, losses)
    assert num_off == num_on == 2


if __name__ == "__main__":
    test_mock_gpt_dataset()
    test_modality_weight_lut()
    test_modality_weights_config_validation()
    test_modality_loss_report()
    test_loss_func_normalize_by_num_supervised_tokens()
