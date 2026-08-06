# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Modality-classification lookup tables derived from tokenizer omni metadata.

This module centralizes the single membership definition (a modality is its content
range plus its structure tokens, see ``ModalityInfo``) shared by the dataset-side loss
weighting in ``megatron.core.datasets.gpt_dataset`` and the per-modality loss report
in ``pretrain_gpt.py``. It is kept out of ``tokenizer_extra_metadata`` so that the
metadata module stays torch-free.
"""

import torch

# Share immutable Lookup Tables (LUTs) between dataset instances in a process.
_LUT_CACHE = {}


def _modality_layout_key(modalities):
    """The memo-key fragment identifying a modality layout; shared by both LUT caches."""
    return tuple(
        (modality.name, modality.offset, modality.vocab_size, modality.structure_ids)
        for modality in modalities
    )


def get_modality_index_lut(modalities, vocab_size: int, device) -> torch.Tensor:
    """Module-memoized :func:`_create_modality_index_lut`."""
    key = ("modality-index-lut", _modality_layout_key(modalities), int(vocab_size), str(device))
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = _create_modality_index_lut(modalities, vocab_size, device)
    return _LUT_CACHE[key]


def _create_modality_index_lut(modalities, vocab_size: int, device) -> torch.Tensor:
    """Map token IDs to 0 for text or a one-based modality index."""
    vocab_size = int(vocab_size)
    assert len(modalities) < 128, "int8 modality-index LUT supports at most 127 modalities"
    lut = torch.zeros(vocab_size, dtype=torch.int8, device=device)
    for modality_index, modality in enumerate(modalities, start=1):
        lut[modality.offset : modality.offset + modality.vocab_size] = modality_index
        lut[torch.as_tensor(modality.structure_ids, dtype=torch.long, device=device)] = (
            modality_index
        )
    return lut


def get_modality_weight_lut(modalities, weights, vocab_size: int, device) -> torch.Tensor:
    """Module-memoized :func:`_create_modality_weight_lut`."""
    key = (
        "wlut",
        _modality_layout_key(modalities),
        tuple(sorted(weights.items())),
        int(vocab_size),
        str(device),
    )
    if key not in _LUT_CACHE:
        _LUT_CACHE[key] = _create_modality_weight_lut(modalities, weights, vocab_size, device)
    return _LUT_CACHE[key]


def _create_modality_weight_lut(modalities, weights, vocab_size: int, device) -> torch.Tensor:
    """Map token IDs to their configured loss weight."""
    weight_vector = torch.ones(len(modalities) + 1, dtype=torch.float, device=device)
    for modality_index, modality in enumerate(modalities, start=1):
        weight_vector[modality_index] = weights.get(modality.name, 1.0)
    modality_index_lut = _create_modality_index_lut(modalities, vocab_size, device)
    return weight_vector[modality_index_lut.long()]
