# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for tokenizer special-token metadata extraction (text-only)."""

import pickle
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from megatron.core.tokenizers.utils.tokenizer_extra_metadata import (
    _extract_special_token_ids,
    _find_hf_tokenizer,
    extract_tokenizer_extra_metadata,
    populate_tokenizer_extra_metadata_from_tokenizer,
)


def test_text_only_metadata_contains_general_special_tokens():
    metadata = extract_tokenizer_extra_metadata(special_ids=[0, 1, 7, 1, 2, 8])
    assert metadata.special_tokens.full_ids == [0, 1, 2, 7, 8]


def test_empty_text_only_metadata_is_valid():
    metadata = extract_tokenizer_extra_metadata([])
    assert metadata.special_tokens.full_ids == []


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


def test_apertus_2_plain_tokenizer_exempts_all_declared_specials():
    """A plain (non-omni) Apertus 2 artifact declares ids 0..123 as special added tokens."""
    hf_tokenizer = _FakeHFTokenizer(
        {}, added_tokens_decoder=_added_tokens_decoder(range(124)), all_special_ids=[0, 1, 2, 3]
    )
    args = SimpleNamespace(goldfish_loss=True)
    metadata = populate_tokenizer_extra_metadata_from_tokenizer(args, hf_tokenizer)
    assert metadata.special_tokens.full_ids == list(range(124))


def test_metadata_dataclasses_are_frozen_and_pickle_safe():
    metadata = extract_tokenizer_extra_metadata(special_ids=[0, 1])

    with pytest.raises(FrozenInstanceError):
        metadata.special_tokens.full_ids = [7]

    # Nested containers remain mutable.
    metadata.special_tokens.full_ids.append(7)

    restored = pickle.loads(pickle.dumps(metadata))
    assert restored == metadata


def test_populate_text_only_metadata_replaces_stale_metadata():
    tokenizer = _wrapped_tokenizer(
        {}, _added_tokens_decoder([10, 11], non_special_ids=[12]), all_special_ids=[1, 2, 3]
    )
    args = SimpleNamespace(tokenizer_extra_metadata="stale")

    metadata = populate_tokenizer_extra_metadata_from_tokenizer(args, tokenizer)

    assert metadata.special_tokens.full_ids == [1, 2, 3, 10, 11]
    assert args.tokenizer_extra_metadata is metadata


def test_populate_hf_tokenizer_without_specials_yields_empty_exemptions():
    args = SimpleNamespace()

    metadata = populate_tokenizer_extra_metadata_from_tokenizer(args, _wrapped_tokenizer({}))

    assert metadata.special_tokens.full_ids == []
    assert args.tokenizer_extra_metadata is metadata


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
