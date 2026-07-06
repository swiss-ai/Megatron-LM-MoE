# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the V2 omnimodal_config metadata extraction."""

from types import SimpleNamespace

import pytest

from megatron.core.tokenizers.utils.tokenizer_omni_metadata import (
    extract_model_special_tokens,
    extract_omni_metadata,
    populate_omni_metadata_from_tokenizer,
)

# Literal copy of the omnimodal_config written by apertus-omni-tokenizer for the v2
# in_place 200k base with vision + audio (omni_mul200k_vision_audio_335232_recipe).
V2_IN_PLACE_CONFIG = {
    "allocation": "in_place",
    "base_vocab_size": 200064,
    "special_token_offset": 0,
    "special_token_count": 124,
    "modalities": [
        {
            "name": "vision",
            "offset": 200064,
            "vocab_size": 131072,
            "start_token": 27,
            "end_token": 28,
        },
        {"name": "audio", "offset": 331136, "vocab_size": 4096, "start_token": 33, "end_token": 34},
    ],
}

V2_APPEND_CONFIG = {
    "omni_special_token_offset": 131072,
    "modalities": [
        {
            "name": "vision",
            "offset": 131272,
            "vocab_size": 131072,
            "start_token": 131073,
            "end_token": 131074,
        },
        {
            "name": "audio",
            "offset": 262344,
            "vocab_size": 4096,
            "start_token": 131080,
            "end_token": 131081,
        },
    ],
}


def test_in_place_special_block():
    """in_place: special_token_offset/count give the base's full contiguous block."""
    omni = extract_omni_metadata(200064, V2_IN_PLACE_CONFIG)

    assert omni.base_vocab_size == 200064
    assert omni.special_tokens.allocation == "in_place"
    assert omni.special_tokens.full_ids == list(range(0, 124))
    assert omni.special_tokens.id_range == (0, 124)

    assert [m.name for m in omni.modalities] == ["vision", "audio"]
    vision, audio = omni.modalities
    assert (vision.offset, vision.vocab_size) == (200064, 131072)
    assert (vision.start_token, vision.end_token) == (27, 28)
    assert (audio.offset, audio.vocab_size) == (331136, 4096)
    assert (audio.start_token, audio.end_token) == (33, 34)

    assert omni.raw_config is V2_IN_PLACE_CONFIG


def test_in_place_base_vocab_size_fallback():
    """base_vocab_size falls back to the copy inside omnimodal_config when absent."""
    omni = extract_omni_metadata(None, V2_IN_PLACE_CONFIG)
    assert omni.base_vocab_size == 200064
    assert omni.special_tokens.full_ids == list(range(0, 124))


def test_explicit_special_token_ids():
    """An explicit special_token_ids list is authoritative in both allocation modes."""
    scattered = list(range(0, 124)) + list(range(200064, 200071))
    config = dict(V2_IN_PLACE_CONFIG, special_token_ids=scattered)
    special = extract_model_special_tokens(config)
    assert special.full_ids == scattered
    assert special.id_range is None  # two disjoint ranges -> not contiguous
    assert special.allocation == "in_place"

    # Append mode (no allocation key), e.g. base text specials + the appended omni block.
    config = dict(V2_APPEND_CONFIG, special_token_ids=[0, 1, 2] + list(range(131072, 131272)))
    special = extract_model_special_tokens(config)
    assert special.allocation == "append"
    assert special.full_ids == [0, 1, 2] + list(range(131072, 131272))
    assert special.id_range is None


def test_append_without_list_raises():
    """append without special_token_ids cannot describe the model's FULL special set.

    The derived [base_vocab_size, min modality offset) omni block would miss the base
    text specials, and with omni metadata present the full set is required: raise.
    """
    with pytest.raises(ValueError, match="special_token_ids"):
        extract_omni_metadata(131072, V2_APPEND_CONFIG)


def test_degenerate_configs():
    """No omnimodal_config yields no special tokens (training proceeds as before);
    a present-but-insufficient config raises instead of guessing."""
    assert extract_omni_metadata(None, None) is None

    # No omnimodal_config at all: metadata exists but no special tokens, no raise.
    omni = extract_omni_metadata(131072, None)
    assert omni.special_tokens is None and omni.modalities == ()

    # in_place without offset/count (and no explicit list) is malformed: raise loudly
    # rather than silently disabling downstream exemptions.
    broken = {"allocation": "in_place", "base_vocab_size": 200064, "modalities": []}
    with pytest.raises(ValueError, match="special_token_offset"):
        extract_model_special_tokens(broken)


class _FakeHFTokenizer:
    """Innermost object of the wrapper chain, carrying init_kwargs."""

    def __init__(self, init_kwargs):
        self.init_kwargs = init_kwargs


class _FakeWrapper:
    """Megatron-style wrapper: reaches the HF tokenizer via ``_tokenizer``."""

    def __init__(self, inner):
        self._tokenizer = inner


def test_populate_omni_metadata_from_tokenizer():
    """populate writes ONLY args.omni_metadata; all values are accessed through it."""
    tokenizer = _FakeWrapper(
        _FakeHFTokenizer({"base_vocab_size": 200064, "omnimodal_config": V2_IN_PLACE_CONFIG})
    )
    args = SimpleNamespace()

    omni = populate_omni_metadata_from_tokenizer(args, tokenizer)

    # args.omni_metadata is the single access path: no flat args.* mirrors are written.
    assert vars(args) == {"omni_metadata": omni}
    assert omni.base_vocab_size == 200064
    assert omni.raw_config is V2_IN_PLACE_CONFIG
    assert omni.modality("vision").offset == 200064
    assert omni.modality("vision").vocab_size == 131072
    assert omni.modality("audio").offset == 331136
    assert omni.modality("audio").vocab_size == 4096
    assert omni.modality("nonexistent") is None
    assert omni.special_tokens.full_ids == list(range(0, 124))


def test_populate_returns_none_without_metadata():
    tokenizer = _FakeWrapper(_FakeHFTokenizer({}))
    args = SimpleNamespace()
    assert populate_omni_metadata_from_tokenizer(args, tokenizer) is None
    assert not hasattr(args, "omni_metadata")
