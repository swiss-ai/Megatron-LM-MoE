# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for general tokenizer metadata and its optional omni extension."""

import pickle
from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from megatron.core.tokenizers.utils.tokenizer_extra_metadata import (
    _extract_special_token_ids,
    _find_hf_tokenizer,
    extract_tokenizer_extra_metadata,
    parse_omni_metadata,
    populate_tokenizer_extra_metadata_from_tokenizer,
)

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

# Content IDs are deliberately included: the omni path must filter them even though
# the tokenizer marks them special=True.
APERTUS_2_ADDED_SPECIAL_IDS = list(range(0, 124)) + [200064, 265599, 331135, 331136, 335231]

SEPARATE_RANGE_CONFIG = {
    "omni_special_token_offset": 100,
    "modalities": [
        {
            "name": "vision",
            "offset": 110,
            "vocab_size": 4,
            "start_token": 101,
            "end_token": 102,
            "structure_token_ids": {"<|image|>": 100, "<|img_start|>": 101, "<|img_end|>": 102},
        },
        {
            "name": "audio",
            "offset": 114,
            "vocab_size": 2,
            "start_token": 105,
            "end_token": 106,
            "structure_token_ids": {"<|audio|>": 104, "<|audio_start|>": 105, "<|audio_end|>": 106},
        },
    ],
}

CONFIG_WITHOUT_STRUCTURE_TOKEN_IDS = {
    "omni_special_token_offset": 100,
    "modalities": [
        {"name": "vision", "offset": 110, "vocab_size": 4, "start_token": 101, "end_token": 102}
    ],
}


def test_text_only_metadata_contains_general_special_tokens():
    metadata = extract_tokenizer_extra_metadata(
        base_vocab_size=None, omnimodal_config=None, special_ids=[0, 1, 7, 1, 2, 8]
    )

    assert metadata.omni is None
    assert metadata.special_tokens.full_ids == [0, 1, 2, 7, 8]


def test_empty_text_only_metadata_is_valid():
    metadata = extract_tokenizer_extra_metadata(None, None, [])
    assert metadata.omni is None
    assert metadata.special_tokens.full_ids == []


def test_apertus_2_in_place_metadata():
    metadata = extract_tokenizer_extra_metadata(
        200064, APERTUS_2_CONFIG, APERTUS_2_ADDED_SPECIAL_IDS
    )

    assert metadata.special_tokens.full_ids == list(range(0, 124))
    assert metadata.omni.base_vocab_size == 200064
    assert [modality.name for modality in metadata.omni.modalities] == ["vision", "audio"]
    assert metadata.omni.modality("vision").offset == 200064
    assert metadata.omni.modality("audio").offset == 331136


def test_omni_schema_supports_separate_structure_range():
    metadata = extract_tokenizer_extra_metadata(
        100, SEPARATE_RANGE_CONFIG, special_ids=[0, 1, 2, 3] + list(range(100, 116))
    )

    assert metadata.special_tokens.full_ids == [0, 1, 2, 3, 100, 101, 102, 104, 105, 106]
    # Unnamed reserve slots and modality content remain eligible for Goldfish.
    assert not ({103, 107, 108, 109} | set(range(110, 116))) & set(metadata.special_tokens.full_ids)


def test_omni_structure_ids_do_not_depend_on_added_token_flags():
    metadata = extract_tokenizer_extra_metadata(100, SEPARATE_RANGE_CONFIG, special_ids=[0, 1])

    assert metadata.special_tokens.full_ids == [0, 1, 100, 101, 102, 104, 105, 106]


def test_missing_structure_token_ids_is_rejected_generically():
    with pytest.raises(ValueError, match="missing required fields: structure_token_ids"):
        parse_omni_metadata(100, CONFIG_WITHOUT_STRUCTURE_TOKEN_IDS)


def test_omni_schema_rejects_missing_and_unexpected_fields():
    for field in ("name", "offset", "vocab_size", "start_token", "end_token"):
        broken = deepcopy(APERTUS_2_CONFIG)
        del broken["modalities"][0][field]
        with pytest.raises(ValueError, match=f"missing required fields: {field}"):
            parse_omni_metadata(200064, broken)

    broken = dict(APERTUS_2_CONFIG, typo_field=True)
    with pytest.raises(ValueError, match="contains unexpected fields: typo_field"):
        parse_omni_metadata(200064, broken)

    broken = deepcopy(APERTUS_2_CONFIG)
    broken["modalities"][0]["typo_field"] = True
    with pytest.raises(ValueError, match="contains unexpected fields: typo_field"):
        parse_omni_metadata(200064, broken)


def test_omni_schema_rejects_malformed_types():
    for malformed in ([], "not-an-object", 123):
        with pytest.raises(ValueError, match="omnimodal_config must be an object"):
            parse_omni_metadata(200064, malformed)

    for invalid_base in (True, 200064.0, "200064"):
        with pytest.raises(ValueError, match="base_vocab_size must be an integer"):
            parse_omni_metadata(invalid_base, APERTUS_2_CONFIG)

    for field, invalid_value in (
        ("offset", 200064.5),
        ("vocab_size", True),
        ("start_token", "27"),
        ("end_token", 28.0),
    ):
        broken = deepcopy(APERTUS_2_CONFIG)
        broken["modalities"][0][field] = invalid_value
        with pytest.raises(ValueError, match=rf"{field} must be an integer"):
            parse_omni_metadata(200064, broken)


def test_omni_schema_validates_ranges_and_structure_maps():
    broken = deepcopy(APERTUS_2_CONFIG)
    broken["modalities"][0]["structure_token_ids"] = {}
    with pytest.raises(ValueError, match="must be a non-empty object"):
        parse_omni_metadata(200064, broken)

    broken = deepcopy(APERTUS_2_CONFIG)
    broken["modalities"][0]["structure_token_ids"]["<|image|>"] = 200064
    with pytest.raises(ValueError, match="inside a modality's content range"):
        parse_omni_metadata(200064, broken)

    broken = deepcopy(APERTUS_2_CONFIG)
    broken["modalities"][1]["offset"] = 300000
    with pytest.raises(ValueError, match="content ranges overlap"):
        parse_omni_metadata(200064, broken)


@pytest.mark.parametrize(
    "mutate, match",
    [
        pytest.param(
            lambda c: c.pop("omni_special_token_offset"),
            "missing required omni_special_token_offset",
            id="missing-omni-offset",
        ),
        pytest.param(
            lambda c: c.update(omni_special_token_offset=1),
            "must equal top-level base_vocab_size",
            id="omni-offset-mismatch",
        ),
        pytest.param(
            lambda c: c.update(modalities=[]),
            "modalities must be a non-empty list",
            id="empty-modalities",
        ),
        pytest.param(
            lambda c: c.update(modalities="vision"),
            "modalities must be a non-empty list",
            id="non-list-modalities",
        ),
        pytest.param(
            lambda c: c["modalities"][1].update(name="vision"),
            "duplicate modality name",
            id="duplicate-name",
        ),
        pytest.param(
            lambda c: c["modalities"][0].update(offset=100000),
            "offset >= base_vocab_size",
            id="offset-below-base",
        ),
        pytest.param(
            lambda c: c["modalities"][0].update(vocab_size=0), "vocab_size > 0", id="zero-vocab"
        ),
        pytest.param(
            lambda c: c["modalities"][0]["structure_token_ids"].update({"<|neg|>": -1}),
            "negative token id",
            id="negative-structure-id",
        ),
        pytest.param(
            lambda c: c["modalities"][0].update(start_token=20),
            "start_token/end_token must appear",
            id="start-not-in-structure-map",
        ),
        pytest.param(
            lambda c: c["modalities"][1]["structure_token_ids"].update({"<|audio|>": 400000}),
            "outside the declared tokenizer layout",
            id="structure-id-outside-layout",
        ),
        pytest.param(
            lambda c: c["modalities"][1]["structure_token_ids"].update({"<|audio|>": 18}),
            "in more than one modality",
            id="structure-id-shared-across-modalities",
        ),
        pytest.param(
            lambda c: c["modalities"][0]["structure_token_ids"].update({"<|image|>": "18"}),
            "must be an integer",
            id="non-int-structure-id",
        ),
    ],
)
def test_omni_schema_rejects_invalid_layouts(mutate, match):
    broken = deepcopy(APERTUS_2_CONFIG)
    mutate(broken)
    with pytest.raises(ValueError, match=match):
        parse_omni_metadata(200064, broken)


def test_omni_schema_rejects_nonpositive_base_vocab_size():
    with pytest.raises(ValueError, match="base_vocab_size must be positive"):
        parse_omni_metadata(0, APERTUS_2_CONFIG)


def test_omni_schema_canonicalizes_modalities_by_offset():
    unsorted = deepcopy(APERTUS_2_CONFIG)
    unsorted["modalities"].reverse()

    metadata = parse_omni_metadata(200064, unsorted)

    assert [modality.name for modality in metadata.modalities] == ["vision", "audio"]
    assert [modality.offset for modality in metadata.modalities] == [200064, 331136]


def test_absent_omni_extension_returns_none():
    assert parse_omni_metadata(None, None) is None
    assert parse_omni_metadata(131072, None) is None


class _FakeHFTokenizer:
    """HuggingFace-shaped innermost object: ``init_kwargs`` plus special-token surface."""

    def __init__(self, init_kwargs, added_tokens_decoder=None, all_special_ids=None):
        self.init_kwargs = init_kwargs
        self.added_tokens_decoder = added_tokens_decoder or {}
        self.all_special_ids = all_special_ids or []


class _FakeLibraryWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _FakeMegatronWrapper:
    def __init__(self, inner):
        self._tokenizer = inner


def _wrapped_tokenizer(init_kwargs, added_tokens_decoder=None, all_special_ids=None):
    return _FakeMegatronWrapper(
        _FakeLibraryWrapper(_FakeHFTokenizer(init_kwargs, added_tokens_decoder, all_special_ids))
    )


def _added_tokens_decoder(special_ids, non_special_ids=()):
    decoder = {token_id: SimpleNamespace(special=True) for token_id in special_ids}
    decoder.update({token_id: SimpleNamespace(special=False) for token_id in non_special_ids})
    return decoder


class _FakeSentencePieceWrapper:
    """Non-HF library wrapper: exposes Megatron mappings but no ``init_kwargs``."""

    special_token_to_id = {"<pad>": 0, "<custom>": 17}
    eos_id = 1


def test_find_hf_tokenizer_unwraps_wrapper_chain():
    tokenizer = _wrapped_tokenizer({})
    hf_tokenizer = tokenizer._tokenizer.tokenizer
    assert _find_hf_tokenizer(tokenizer) is hf_tokenizer
    assert _find_hf_tokenizer(hf_tokenizer) is hf_tokenizer


def test_find_hf_tokenizer_returns_none_for_non_hf_chains():
    assert _find_hf_tokenizer(_FakeMegatronWrapper(_FakeSentencePieceWrapper())) is None


def test_special_id_extraction_unions_declared_and_added_ids():
    hf_tokenizer = _FakeHFTokenizer(
        {},
        added_tokens_decoder=_added_tokens_decoder([0, 1, 3], non_special_ids=[7]),
        all_special_ids=[2, 3],
    )
    assert _extract_special_token_ids(hf_tokenizer) == [0, 1, 2, 3]


def test_metadata_dataclasses_are_frozen_and_pickle_safe():
    metadata = extract_tokenizer_extra_metadata(100, SEPARATE_RANGE_CONFIG, special_ids=[0, 1])

    with pytest.raises(FrozenInstanceError):
        metadata.special_tokens.full_ids = [7]
    with pytest.raises(FrozenInstanceError):
        metadata.omni.modalities = ()

    # Freezing prevents field rebinding without replacing ordinary nested containers.
    metadata.special_tokens.full_ids.append(7)
    metadata.omni.modalities[0].structure_token_ids["<new>"] = 7

    restored = pickle.loads(pickle.dumps(metadata))
    assert restored == metadata


def test_populate_omni_tokenizer_metadata():
    tokenizer = _wrapped_tokenizer(
        {"base_vocab_size": 200064, "omnimodal_config": APERTUS_2_CONFIG},
        _added_tokens_decoder(APERTUS_2_ADDED_SPECIAL_IDS, non_special_ids=[150000]),
        all_special_ids=[0, 1, 2, 3, 200064],
    )
    args = SimpleNamespace()

    metadata = populate_tokenizer_extra_metadata_from_tokenizer(args, tokenizer)

    assert vars(args) == {"tokenizer_extra_metadata": metadata}
    assert metadata.omni.modality("vision").offset == 200064
    assert metadata.special_tokens.full_ids == list(range(0, 124))


def test_populate_text_only_metadata_replaces_stale_metadata():
    tokenizer = _wrapped_tokenizer(
        {}, _added_tokens_decoder([10, 11], non_special_ids=[12]), all_special_ids=[1, 2, 3]
    )
    args = SimpleNamespace(
        tokenizer_metadata="/tokenizer/build/metadata.json", tokenizer_extra_metadata="stale"
    )

    metadata = populate_tokenizer_extra_metadata_from_tokenizer(args, tokenizer)

    assert metadata.omni is None
    assert metadata.special_tokens.full_ids == [1, 2, 3, 10, 11]
    assert args.tokenizer_extra_metadata is metadata
    assert args.tokenizer_metadata == "/tokenizer/build/metadata.json"


def test_populate_hf_tokenizer_without_specials_yields_empty_exemptions():
    args = SimpleNamespace()

    metadata = populate_tokenizer_extra_metadata_from_tokenizer(args, _wrapped_tokenizer({}))

    assert metadata.omni is None
    assert metadata.special_tokens.full_ids == []
    assert args.tokenizer_extra_metadata is metadata


def test_populate_rejects_omni_config_without_declared_special_tokens():
    args = SimpleNamespace(tokenizer_extra_metadata="stale")
    tokenizer = _wrapped_tokenizer(
        {"base_vocab_size": 200064, "omnimodal_config": APERTUS_2_CONFIG}
    )

    with pytest.raises(ValueError, match="declares no special tokens"):
        populate_tokenizer_extra_metadata_from_tokenizer(args, tokenizer)
    assert args.tokenizer_extra_metadata is None


def test_populate_non_hf_tokenizer_yields_no_metadata():
    args = SimpleNamespace(tokenizer_extra_metadata="stale")
    tokenizer = _FakeMegatronWrapper(_FakeSentencePieceWrapper())

    assert populate_tokenizer_extra_metadata_from_tokenizer(args, tokenizer) is None
    assert args.tokenizer_extra_metadata is None


def test_populate_non_hf_tokenizer_fails_fast_when_goldfish_enabled():
    args = SimpleNamespace(goldfish_loss=True, tokenizer_extra_metadata="stale")
    tokenizer = _FakeMegatronWrapper(_FakeSentencePieceWrapper())

    with pytest.raises(ValueError, match="Goldfish loss requires a HuggingFace-backed tokenizer"):
        populate_tokenizer_extra_metadata_from_tokenizer(args, tokenizer)
    assert args.tokenizer_extra_metadata is None


def test_populate_clears_stale_metadata_before_omni_validation_failure():
    args = SimpleNamespace(tokenizer_extra_metadata="stale")
    tokenizer = _wrapped_tokenizer(
        {"base_vocab_size": 100, "omnimodal_config": CONFIG_WITHOUT_STRUCTURE_TOKEN_IDS},
        _added_tokens_decoder([0, 1, 2, 3]),
    )
    with pytest.raises(ValueError, match="missing required fields: structure_token_ids"):
        populate_tokenizer_extra_metadata_from_tokenizer(args, tokenizer)
    assert args.tokenizer_extra_metadata is None
