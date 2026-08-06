# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the modality-classification LUTs shared by dataset weighting and loss reporting."""

import torch


def vision_audio_modalities():
    """Two-modality layout shared by the LUT tests."""
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
    return (vision, audio)


def test_modality_weight_lut():
    from megatron.core.tokenizers.utils.modality_lut import (
        _create_modality_weight_lut,
        get_modality_weight_lut,
    )

    modalities = vision_audio_modalities()
    lut = _create_modality_weight_lut(
        modalities, {"vision": 0.25, "audio": 0.0}, vocab_size=2000, device=torch.device("cpu")
    )

    # Content and structure IDs receive the modality weight.
    labels = torch.tensor([0, 5, 7, 9, 999, 1000, 1099, 1100, 1149, 1150])
    expected = torch.tensor([1.0, 0.25, 0.25, 0.0, 1.0, 0.25, 0.25, 0.0, 0.0, 1.0])
    assert torch.equal(lut[labels], expected)

    # Existing mask zeros remain zero.
    loss_mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert torch.equal(
        loss_mask * lut[labels], torch.tensor([1.0, 0.25, 0.0, 0.0, 1.0, 0.25, 0.25, 0.0, 0.0, 1.0])
    )

    # Unweighted groups remain 1.0.
    partial = _create_modality_weight_lut(
        modalities, {"audio": 0.5}, vocab_size=2000, device=torch.device("cpu")
    )
    assert torch.equal(
        partial[labels], torch.tensor([1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0])
    )

    weights = {"vision": 0.25, "audio": 0.0}
    memoized = get_modality_weight_lut(modalities, weights, 2000, torch.device("cpu"))
    assert get_modality_weight_lut(modalities, weights, 2000, torch.device("cpu")) is memoized


def test_modality_index_lut():
    """The shared token ID to modality-index table."""
    from megatron.core.tokenizers.utils.modality_lut import (
        _create_modality_index_lut,
        get_modality_index_lut,
    )

    modalities = vision_audio_modalities()
    lut = _create_modality_index_lut(modalities, vocab_size=2000, device=torch.device("cpu"))

    labels = torch.tensor([0, 5, 7, 9, 999, 1000, 1099, 1100, 1149, 1150])
    expected = torch.tensor([0, 1, 1, 2, 0, 1, 1, 2, 2, 0], dtype=torch.int8)
    assert torch.equal(lut[labels], expected)

    # Cross-check direct ID membership.
    generator = torch.Generator().manual_seed(1234)
    ids = torch.randint(0, 2000, (512,), generator=generator)
    for modality_index, modality in enumerate(modalities, start=1):
        reference = (ids >= modality.offset) & (ids < modality.offset + modality.vocab_size)
        reference |= torch.isin(ids, torch.tensor(modality.structure_ids))
        assert torch.equal(lut[ids] == modality_index, reference)

    memoized = get_modality_index_lut(modalities, 2000, torch.device("cpu"))
    assert get_modality_index_lut(modalities, 2000, torch.device("cpu")) is memoized


if __name__ == "__main__":
    test_modality_weight_lut()
    test_modality_index_lut()
