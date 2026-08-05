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


if __name__ == "__main__":
    test_mock_gpt_dataset()
    test_modality_weight_lut()
    test_modality_weights_config_validation()
    test_modality_loss_report()
