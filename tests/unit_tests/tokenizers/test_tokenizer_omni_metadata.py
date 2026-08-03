# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the omnimodal_config metadata extraction.

Fixtures are literal copies of the configs shipped by
https://github.com/swiss-ai/apertus-omni-tokenizer (``tokenizers/Apertus_2`` and
``tokenizers/Apertus_1p5``), abbreviated only in the added-token tables.
"""

from types import SimpleNamespace

import pytest

from megatron.core.tokenizers.utils.tokenizer_omni_metadata import (
    extract_added_special_token_ids,
    extract_model_special_tokens,
    extract_omni_metadata,
    populate_omni_metadata_from_tokenizer,
)

# Apertus 2: structure tokens at low ids INSIDE the 200064-token base vocab; each
# modality publishes its structure_token_ids map. tokenizer_config.json's
# added_tokens_decoder is EMPTY -- the added-token table lives in tokenizer.json only.
APERTUS_2_CONFIG = {
    "omni_special_token_offset": 200064,
    "modalities": [
        {
            "name": "vision",
            "offset": 200064,
            "vocab_size": 131072,
            "start_token": 27,
            "end_token": 28,
            "structure_token_ids": {
                "<|image|>": 18,
                "<|img_start|>": 27,
                "<|img_end|>": 28,
                "<|img_token_start|>": 29,
                "<|img_end_of_row|>": 30,
                "<|img_end_of_frame|>": 31,
                "<|img_generation_start|>": 32,
            },
        },
        {
            "name": "audio",
            "offset": 331136,
            "vocab_size": 4096,
            "start_token": 33,
            "end_token": 34,
            "structure_token_ids": {
                "<|audio|>": 19,
                "<|audio_start|>": 33,
                "<|audio_end|>": 34,
                "<|stt_transcribe|>": 35,
                "<|stt_continue|>": 36,
                "<|tts_continue|>": 37,
                "<|stt_translate|>": 38,
                "<|audio_annotate|>": 39,
            },
        },
    ],
}

# The artifact's real special block: ids 0-123 inside the base vocab (text control
# tokens, the structure tokens above, and the <SPECIAL_40..123> reserve pool). ALL
# omni content tokens are ALSO flagged special=True -- representative ids included
# here to prove the derivation excludes them.
APERTUS_2_ADDED_SPECIAL_IDS = list(range(0, 124)) + [200064, 265599, 331135, 331136, 335231]

# Apertus 1.5: append geometry -- structure tokens in the gap [131072, 131272) above
# the base vocab, content blocks after them; no structure_token_ids maps.
APERTUS_1P5_CONFIG = {
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

# 1.5-era builds flag text specials, the appended structure block, AND content as
# special added tokens.
APERTUS_1P5_ADDED_SPECIAL_IDS = (
    [0, 1, 2, 3] + list(range(131072, 131272)) + [131272, 262343, 262344, 266439]
)


def test_apertus_2_derivation():
    """Apertus 2: derived set is exactly the in-base 0-123 block; content ids excluded."""
    omni = extract_omni_metadata(200064, APERTUS_2_CONFIG, APERTUS_2_ADDED_SPECIAL_IDS)

    assert omni.base_vocab_size == 200064
    assert omni.special_tokens.source == "derived"
    assert omni.special_tokens.full_ids == list(range(0, 124))
    assert omni.special_tokens.id_range == (0, 124)

    assert [m.name for m in omni.modalities] == ["vision", "audio"]
    vision, audio = omni.modalities
    assert (vision.offset, vision.vocab_size) == (200064, 131072)
    assert (vision.start_token, vision.end_token) == (27, 28)
    assert (audio.offset, audio.vocab_size) == (331136, 4096)
    assert (audio.start_token, audio.end_token) == (33, 34)

    assert omni.raw_config is APERTUS_2_CONFIG


def test_apertus_1p5_derivation():
    """Apertus 1.5: text specials + appended structure gap survive; content ids excluded."""
    omni = extract_omni_metadata(131072, APERTUS_1P5_CONFIG, APERTUS_1P5_ADDED_SPECIAL_IDS)

    assert omni.special_tokens.source == "derived"
    assert omni.special_tokens.full_ids == [0, 1, 2, 3] + list(range(131072, 131272))
    assert omni.special_tokens.id_range is None  # two disjoint runs

    # Boundary tokens sit inside the structure gap, not in content.
    assert 131073 in omni.special_tokens.full_ids
    assert 131272 not in omni.special_tokens.full_ids  # first vision content id
    assert 262344 not in omni.special_tokens.full_ids  # first audio content id


def test_structure_token_ids_parsed():
    """structure_token_ids lands on ModalityInfo (Apertus 2) and is None for 1.5."""
    omni = extract_omni_metadata(200064, APERTUS_2_CONFIG, APERTUS_2_ADDED_SPECIAL_IDS)
    assert omni.modality("vision").structure_token_ids["<|image|>"] == 18
    assert omni.modality("audio").structure_token_ids["<|stt_transcribe|>"] == 35

    omni = extract_omni_metadata(131072, APERTUS_1P5_CONFIG, APERTUS_1P5_ADDED_SPECIAL_IDS)
    assert omni.modality("vision").structure_token_ids is None


def test_structure_ids_union_even_without_added_coverage():
    """Config-declared structure/boundary ids are unioned in even when the added-token
    table misses them (defensive completeness for the goldfish exemption)."""
    special = extract_model_special_tokens(APERTUS_2_CONFIG, added_special_ids=[0, 1, 2])
    # 18/19, 27-39 come from structure_token_ids + start/end despite absent added flags.
    assert special.full_ids == [0, 1, 2, 18, 19] + list(range(27, 40))


def test_explicit_special_token_ids_authoritative():
    """An explicit special_token_ids list wins over the added-token derivation."""
    scattered = list(range(0, 124)) + [200070]
    config = dict(APERTUS_2_CONFIG, special_token_ids=scattered)
    special = extract_model_special_tokens(config, added_special_ids=[5000])
    assert special.source == "explicit"
    assert special.full_ids == scattered
    assert special.id_range is None  # disjoint -> not contiguous


def test_no_added_token_source_raises():
    """omnimodal_config present but no special_token_ids and no added-token table: raise
    loudly rather than silently disabling the goldfish exemption."""
    with pytest.raises(ValueError, match="added_tokens_decoder"):
        extract_model_special_tokens(APERTUS_2_CONFIG, added_special_ids=None)
    with pytest.raises(ValueError, match="added_tokens_decoder"):
        extract_omni_metadata(200064, APERTUS_2_CONFIG)


def test_structure_id_inside_content_range_raises():
    """A config-declared structure id inside a content range is malformed: exempting it
    would shield real content tokens from goldfish drops."""
    broken = {
        "modalities": [
            {
                "name": "vision",
                "offset": 200064,
                "vocab_size": 131072,
                "start_token": 200064,  # collides with the first content id
                "end_token": 28,
            }
        ]
    }
    with pytest.raises(ValueError, match="content range"):
        extract_model_special_tokens(broken, added_special_ids=[0, 1])


def test_degenerate_configs():
    """No omnimodal_config yields no special tokens (training proceeds as before)."""
    assert extract_omni_metadata(None, None) is None

    # No omnimodal_config at all: metadata exists but no special tokens, no raise.
    omni = extract_omni_metadata(131072, None)
    assert omni.special_tokens is None and omni.modalities == ()


class _FakeHFTokenizer:
    """Innermost object of the wrapper chain, carrying init_kwargs and the added-token
    table -- like the real Apertus 2 artifact, ``init_kwargs``'s added_tokens_decoder
    copy is empty and only the live property has the tokens."""

    def __init__(self, init_kwargs, added_tokens_decoder=None):
        self.init_kwargs = init_kwargs
        self.added_tokens_decoder = added_tokens_decoder or {}


class _FakeWrapper:
    """Megatron-style wrapper: reaches the HF tokenizer via ``_tokenizer``."""

    def __init__(self, inner):
        self._tokenizer = inner


def _added_tokens_decoder(special_ids, non_special_ids=()):
    """AddedToken-like objects with int keys, as the live HF property yields them."""
    decoder = {i: SimpleNamespace(special=True, content=f"<tok{i}>") for i in special_ids}
    decoder.update(
        {i: SimpleNamespace(special=False, content=f"<tok{i}>") for i in non_special_ids}
    )
    return decoder


def test_populate_omni_metadata_from_tokenizer():
    """populate writes ONLY args.omni_metadata; all values are accessed through it."""
    tokenizer = _FakeWrapper(
        _FakeHFTokenizer(
            # base_vocab_size deliberately absent from init_kwargs: it must fall back to
            # omni_special_token_offset inside the config.
            {"omnimodal_config": APERTUS_2_CONFIG, "added_tokens_decoder": {}},
            added_tokens_decoder=_added_tokens_decoder(
                APERTUS_2_ADDED_SPECIAL_IDS, non_special_ids=[150000]
            ),
        )
    )
    args = SimpleNamespace()

    omni = populate_omni_metadata_from_tokenizer(args, tokenizer)

    # args.omni_metadata is the single access path: no flat args.* mirrors are written.
    assert vars(args) == {"omni_metadata": omni}
    assert omni.base_vocab_size == 200064
    assert omni.raw_config is APERTUS_2_CONFIG
    assert omni.modality("vision").offset == 200064
    assert omni.modality("vision").vocab_size == 131072
    assert omni.modality("audio").offset == 331136
    assert omni.modality("audio").vocab_size == 4096
    assert omni.modality("nonexistent") is None
    assert omni.special_tokens.full_ids == list(range(0, 124))
    assert omni.special_tokens.id_range == (0, 124)


def test_populate_returns_none_without_metadata():
    tokenizer = _FakeWrapper(_FakeHFTokenizer({}))
    args = SimpleNamespace()
    assert populate_omni_metadata_from_tokenizer(args, tokenizer) is None
    assert not hasattr(args, "omni_metadata")


def test_added_ids_from_init_kwargs_dict_shape():
    """Serialized added_tokens_decoder copies (plain dicts, str keys) parse too."""
    tokenizer = _FakeWrapper(
        _FakeHFTokenizer(
            {
                "added_tokens_decoder": {
                    "0": {"special": True, "content": "<unk>"},
                    "1": {"special": True, "content": "<s>"},
                    "7": {"special": False, "content": "<plain>"},
                }
            }
        )
    )
    assert extract_added_special_token_ids(tokenizer) == [0, 1]
