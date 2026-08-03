# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Helpers for tokenizer-provided omni metadata propagation.

Expected metadata is read from the HuggingFace tokenizer's ``init_kwargs`` (i.e. the
keys stored in ``tokenizer_config.json``) plus its added-token table, reached by
walking the wrapper chain:
- legacy path:  wrapper -> ``_tokenizer`` -> HF tokenizer (has ``init_kwargs``)
- core  path:  wrapper -> ``_tokenizer`` -> library wrapper -> ``tokenizer`` -> HF tokenizer

The ``omnimodal_config`` payload (contract: ``docs/omnimodal_config.md`` in
https://github.com/swiss-ai/apertus-omni-tokenizer) carries ``omni_special_token_offset``
(== ``base_vocab_size``, which is a top-level ``tokenizer_config.json`` key) and one
entry per modality; content ids are contiguous (``id = offset + index``). Two shipped
geometries::

    # Apertus 1.5 -- append: structure tokens sit in the gap
    # [base_vocab_size, min modality offset), content blocks after them.
    {
      "omni_special_token_offset": 131072,
      "modalities": [
        {"name": "vision", "offset": 131272, "vocab_size": 131072,
         "start_token": 131073, "end_token": 131074},
        {"name": "audio",  "offset": 262344, "vocab_size": 4096,
         "start_token": 131080, "end_token": 131081}
      ]
    }

    # Apertus 2 -- structure tokens at low ids INSIDE the base vocab; each modality
    # additionally publishes its full name -> id map as structure_token_ids.
    {
      "omni_special_token_offset": 200064,
      "modalities": [
        {"name": "vision", "offset": 200064, "vocab_size": 131072,
         "start_token": 27, "end_token": 28,
         "structure_token_ids": {"<|image|>": 18, "<|img_start|>": 27, ...}},
        {"name": "audio",  "offset": 331136, "vocab_size": 4096,
         "start_token": 33, "end_token": 34,
         "structure_token_ids": {"<|audio|>": 19, "<|audio_start|>": 33, ...}}
      ]
    }

The model's FULL special-token id set (text control tokens AND omni structure tokens;
consumers like the goldfish exemption rely on that completeness) is not enumerable from
the config alone, and ``all_special_ids`` is version-dependent (the contract forbids
it). It is derived instead as::

    {added tokens with special=True}          # from the tokenizer's added_tokens_decoder
      MINUS  content ranges [offset, offset + vocab_size) of every modality
      UNION  every modality's structure_token_ids values + start_token/end_token

The MINUS matters: omni CONTENT tokens are also flagged ``special=True`` (all 135k of
them in Apertus 2), and exempting them would shield real content from goldfish drops.
An explicit ``special_token_ids`` list in the config, when present, is authoritative
and skips the derivation. Whenever an ``omnimodal_config`` is present the set is
required -- if neither source is available the extraction raises.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpecialTokens:
    """ALL of the model's special-token ids, extracted for a tokenizer with an ``omnimodal_config``.

    This is the complete special-token set of the vocabulary (text control tokens such
    as bos/eos/pad/chat tokens AND omni structure tokens), not just the omni subset --
    consumers like the goldfish exemption rely on that completeness. Omni CONTENT
    tokens are deliberately excluded even though the tokenizer flags them special.

    Attributes:
        source: ``"explicit"`` (config ``special_token_ids`` list) or ``"derived"``
            (added-token derivation described in the module docstring).
        full_ids: Sorted list of every special-token id (always populated).
        id_range: Contiguous ``(start, end)`` span when ``full_ids`` is one run,
            ``None`` when the ids are scattered.
    """

    source: str
    full_ids: List[int]
    id_range: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class ModalityInfo:
    """One modality's token layout from ``omnimodal_config.modalities``.

    Content ids are contiguous, so ``token_id = offset + codebook_index``.

    Attributes:
        name: Modality name (e.g. ``"vision"``, ``"audio"``).
        offset: Content-token block start id, or ``None`` if absent.
        vocab_size: Number of content tokens, or ``None``.
        start_token / end_token: Boundary special-token ids, or ``None``.
        structure_token_ids: Full structure-token name -> id map (Apertus 2+); ``None``
            for 1.5-era configs that don't publish it. Structure tokens resolve by
            name -- do not assume geometry (1.5 puts them above the base vocab,
            2 at low ids inside it).
    """

    name: str
    offset: Optional[int] = None
    vocab_size: Optional[int] = None
    start_token: Optional[int] = None
    end_token: Optional[int] = None
    structure_token_ids: Optional[Dict[str, int]] = None


@dataclass(frozen=True)
class OmniMetadata:
    """All omni metadata read from a tokenizer, as a single typed value.

    This is the single access path for omni metadata: it lives on ``args.omni_metadata``
    (and on ``GPTDatasetConfig.omni_metadata`` for dataloader workers); no flat ``args.*``
    mirrors exist. Access modality layouts via :meth:`modality`, e.g.
    ``args.omni_metadata.modality("vision").offset``.

    Attributes:
        base_vocab_size: Text vocab size (end of text ids), or ``None``.
        modalities: Per-modality layout, one :class:`ModalityInfo` each.
        special_tokens: The model's full special-token id set
            (:class:`ModelSpecialTokens`); ``None`` only when no ``omnimodal_config``
            was present.
        raw_config: The original ``omnimodal_config`` dict, for anything not modeled here.
    """

    base_vocab_size: Optional[int]
    modalities: Tuple[ModalityInfo, ...] = ()
    special_tokens: Optional[ModelSpecialTokens] = None
    raw_config: Optional[Dict[str, Any]] = None

    def modality(self, name: str) -> Optional[ModalityInfo]:
        """Return the :class:`ModalityInfo` named ``name``, or ``None`` if absent."""
        return next((m for m in self.modalities if m.name == name), None)


def _int_or_none(value: Any) -> Optional[int]:
    """Coerce to ``int``, passing ``None`` through unchanged."""
    return int(value) if value is not None else None


def _contiguous_range(sorted_ids: List[int]) -> Optional[Tuple[int, int]]:
    """``(start, end)`` when ``sorted_ids`` is one contiguous run, else ``None``."""
    if not sorted_ids:
        return None
    start, end = sorted_ids[0], sorted_ids[-1] + 1
    return (start, end) if end - start == len(sorted_ids) else None


def _content_ranges(omnimodal_config: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Half-open content-token id ranges ``[offset, offset + vocab_size)`` per modality."""
    ranges = []
    for modality in omnimodal_config.get("modalities", []):
        offset, vocab_size = modality.get("offset"), modality.get("vocab_size")
        if offset is not None and vocab_size is not None:
            ranges.append((int(offset), int(offset) + int(vocab_size)))
    return ranges


def _iter_tokenizer_wrappers(tokenizer: Any):
    """Yield tokenizer and nested wrapper objects linked by ``_tokenizer``/``tokenizer`` attrs.

    Concrete wrapper chains seen across repos include:
    - legacy path: wrapper -> ``_tokenizer`` -> HF tokenizer (has ``init_kwargs``)
    - core path: wrapper -> ``_tokenizer`` -> library wrapper -> ``tokenizer`` -> HF tokenizer
    """
    queue = [tokenizer]
    seen = set()
    while queue:
        obj = queue.pop(0)
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        yield obj
        for attr in ("_tokenizer", "tokenizer"):
            child = getattr(obj, attr, None)
            if child is not None:
                queue.append(child)


def extract_tokenizer_init_kwargs(tokenizer: Any) -> Dict[str, Any]:
    """Find first ``init_kwargs`` dict across the tokenizer wrapper chain.

    Returns ``{}`` when no ``init_kwargs`` dictionary is available.
    """
    # init_kwargs holds tokenizer_config.json (incl. custom keys like omnimodal_config) on
    # transformers 4.x and still on 5.x main
    for obj in _iter_tokenizer_wrappers(tokenizer):
        init_kwargs = getattr(obj, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            return init_kwargs
    return {}


def extract_added_special_token_ids(tokenizer: Any) -> Optional[List[int]]:
    """Collect the ids of added tokens flagged ``special=True`` from the wrapper chain.

    Reads the first non-empty ``added_tokens_decoder`` found -- preferably the live HF
    tokenizer property (the Apertus 2 artifact keeps ``tokenizer_config.json``'s copy
    empty; the table lives in ``tokenizer.json``), falling back to a serialized copy in
    ``init_kwargs``. Handles both value shapes: ``AddedToken`` objects (``.special``
    attribute, int keys) and plain dicts (``{"special": true}``, str keys).

    Returns ``None`` when no added-token table is reachable at all.
    """
    for obj in _iter_tokenizer_wrappers(tokenizer):
        decoder = getattr(obj, "added_tokens_decoder", None)
        if not (isinstance(decoder, dict) and decoder):
            init_kwargs = getattr(obj, "init_kwargs", None)
            decoder = init_kwargs.get("added_tokens_decoder") if isinstance(init_kwargs, dict) else None
        if not (isinstance(decoder, dict) and decoder):
            continue
        ids = []
        for token_id, token in decoder.items():
            special = (
                token.get("special") if isinstance(token, dict) else getattr(token, "special", False)
            )
            if special:
                ids.append(int(token_id))
        return sorted(ids)
    return None


def extract_model_special_tokens(
    omnimodal_config: Optional[Dict[str, Any]],
    added_special_ids: Optional[List[int]] = None,
) -> Optional[ModelSpecialTokens]:
    """Extract the model's FULL special-token id set for an ``omnimodal_config``.

    Resolution order:

    1. Explicit ``special_token_ids`` list in the config, authoritative when present.
    2. Derivation from ``added_special_ids`` (the tokenizer's ``special=True`` added
       tokens): drop every modality's content range ``[offset, offset + vocab_size)``
       (content tokens are also flagged special and must stay goldfish-droppable),
       then union every modality's ``structure_token_ids`` values and
       ``start_token``/``end_token``. Handles both shipped geometries (Apertus 1.5's
       appended structure block and Apertus 2's in-base-vocab structure ids) without
       version switches.

    Whenever omni metadata is present, the full special-token set is REQUIRED: failing
    loudly beats silently disabling downstream consumers such as the goldfish exemption.

    Returns a :class:`ModelSpecialTokens`, or ``None`` only when ``omnimodal_config``
    is ``None`` (non-omni tokenizer).

    Raises:
        ValueError: When the config has no ``special_token_ids`` and no added-token
            table was reachable, or when a config-declared structure/boundary id falls
            inside a content range (malformed config).
    """
    if omnimodal_config is None:
        return None

    # 1. Explicit id list: authoritative when present.
    explicit_ids = omnimodal_config.get("special_token_ids")
    if explicit_ids:
        full_ids = sorted({int(i) for i in explicit_ids})
        return ModelSpecialTokens(
            source="explicit", full_ids=full_ids, id_range=_contiguous_range(full_ids)
        )

    # 2. Derive from the tokenizer's special-flagged added tokens.
    if not added_special_ids:
        raise ValueError(
            "omnimodal_config is present but the model's FULL special-token set cannot "
            "be determined: the config has no special_token_ids list and no added-token "
            "table (added_tokens_decoder) was reachable on the tokenizer. Without it the "
            "goldfish exemption would silently vanish."
        )

    content_ranges = _content_ranges(omnimodal_config)

    def in_content(token_id: int) -> bool:
        return any(start <= token_id < end for start, end in content_ranges)

    special_ids = {int(i) for i in added_special_ids if not in_content(int(i))}

    structure_ids = set()
    for modality in omnimodal_config.get("modalities", []):
        for token_id in (modality.get("structure_token_ids") or {}).values():
            structure_ids.add(int(token_id))
        for key in ("start_token", "end_token"):
            if modality.get(key) is not None:
                structure_ids.add(int(modality[key]))

    misplaced = sorted(i for i in structure_ids if in_content(i))
    if misplaced:
        raise ValueError(
            f"omnimodal_config declares structure/boundary token ids {misplaced} inside a "
            "modality's content range [offset, offset + vocab_size); exempting them would "
            "shield real content tokens from goldfish drops (malformed config)."
        )

    full_ids = sorted(special_ids | structure_ids)
    return ModelSpecialTokens(
        source="derived", full_ids=full_ids, id_range=_contiguous_range(full_ids)
    )


def extract_omni_metadata(
    base_vocab_size: Optional[int],
    omnimodal_config: Optional[Dict[str, Any]],
    added_special_ids: Optional[List[int]] = None,
) -> Optional[OmniMetadata]:
    """Read all omni metadata into an :class:`OmniMetadata` dataclass.

    Pure function of the resolved ``base_vocab_size`` + ``omnimodal_config`` + the
    tokenizer's special-flagged added-token ids; it does not touch ``args`` or the
    tokenizer, so the read model is decoupled from how the values are later
    distributed. Returns ``None`` only when both config inputs are ``None``.
    """
    if base_vocab_size is None and omnimodal_config is None:
        return None

    # base_vocab_size is a top-level tokenizer_config.json key; fall back to copies
    # inside the omnimodal_config (omni_special_token_offset is defined as equal to it).
    if base_vocab_size is None and omnimodal_config is not None:
        base_vocab_size = omnimodal_config.get("base_vocab_size")
        if base_vocab_size is None:
            base_vocab_size = omnimodal_config.get("omni_special_token_offset")

    modalities: List[ModalityInfo] = []
    if omnimodal_config is not None:
        for modality in omnimodal_config.get("modalities", []):
            name = modality.get("name")
            if not name:
                continue
            structure_token_ids = modality.get("structure_token_ids")
            modalities.append(
                ModalityInfo(
                    name=name,
                    offset=_int_or_none(modality.get("offset")),
                    vocab_size=_int_or_none(modality.get("vocab_size")),
                    start_token=_int_or_none(modality.get("start_token")),
                    end_token=_int_or_none(modality.get("end_token")),
                    structure_token_ids=(
                        {str(k): int(v) for k, v in structure_token_ids.items()}
                        if structure_token_ids is not None
                        else None
                    ),
                )
            )

    return OmniMetadata(
        base_vocab_size=_int_or_none(base_vocab_size),
        modalities=tuple(modalities),
        special_tokens=extract_model_special_tokens(omnimodal_config, added_special_ids),
        raw_config=omnimodal_config,
    )


def populate_omni_metadata_from_tokenizer(args, tokenizer: Any) -> Optional[OmniMetadata]:
    """Read omni metadata from ``tokenizer`` onto ``args.omni_metadata``.

    Called during initialization (``set_global_variables``), right after the global
    tokenizer is built -- before the wandb writer snapshots ``vars(args)`` and before
    model/dataset construction -- and again from ``rebuild_tokenizer`` to keep ``args``
    in sync with a rebuilt tokenizer. For a tokenizer without omnimodal information this
    is a no-op: it returns ``None`` and leaves ``args`` untouched.

    ``args.omni_metadata`` (an :class:`OmniMetadata`, via :func:`extract_omni_metadata`)
    is the SINGLE access path -- no flat ``args.*`` mirrors are written. Consumers read
    e.g. ``args.omni_metadata.base_vocab_size``,
    ``args.omni_metadata.modality("vision").offset``, or
    ``args.omni_metadata.special_tokens.full_ids`` (goldfish exemption; carried onto
    ``GPTDatasetConfig`` for dataloader workers).

    Metadata sources: the tokenizer ``init_kwargs`` keys ``base_vocab_size`` and
    ``omnimodal_config`` (i.e. ``tokenizer_config.json``), plus the added-token table
    (``added_tokens_decoder``) for the special-token derivation.

    Returns:
        The :class:`OmniMetadata`, or ``None`` when the tokenizer has no omni metadata.
    """
    init_kwargs = extract_tokenizer_init_kwargs(tokenizer)
    base_vocab_size = init_kwargs.get("base_vocab_size")
    omnimodal_config = init_kwargs.get("omnimodal_config")
    added_special_ids = extract_added_special_token_ids(tokenizer)

    omni = extract_omni_metadata(base_vocab_size, omnimodal_config, added_special_ids)
    if omni is None:
        return None

    names = [modality.name for modality in omni.modalities]
    special = omni.special_tokens
    special_summary = (
        f"{len(special.full_ids)} ids ({special.source}, range={special.id_range})"
        if special is not None
        else "none"
    )
    log_single_rank(
        logger,
        logging.INFO,
        f" > loaded omnimodal_config with modalities: {names}; special tokens: {special_summary}",
    )

    args.omni_metadata = omni
    return omni
