# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Helpers for tokenizer-provided omni metadata propagation.

Expected metadata is read from the HuggingFace tokenizer's ``init_kwargs`` (i.e. the
keys stored in ``tokenizer_config.json``), reached by walking the wrapper chain:
- legacy path:  wrapper -> ``_tokenizer`` -> HF tokenizer (has ``init_kwargs``)
- core  path:  wrapper -> ``_tokenizer`` -> library wrapper -> ``tokenizer`` -> HF tokenizer

The ``omnimodal_config`` payload comes in two shapes (produced by
https://github.com/swiss-ai/apertus-omni-tokenizer):

- append mode -- omni special tokens are a contiguous block right after the text
  vocab; each modality records its content block plus start/end boundary ids::

    {
      "omni_special_token_offset": 131072,  # in artifacts (== base_vocab_size); not parsed here
      "modalities": [
        {"name": "vision", "offset": 131272, "vocab_size": 131072,
         "start_token": 131073, "end_token": 131074},
        {"name": "audio",  "offset": 262344, "vocab_size": 4096,
         "start_token": 131080, "end_token": 131081}
      ]
    }

- in_place mode -- the base tokenizer pre-bakes ALL of its special tokens (text
  bos/eos/pad/chat tokens, omni structure tokens, reserve pool) as one contiguous
  block inside the base vocab, recorded as ``special_token_offset`` +
  ``special_token_count``; only content tokens are appended::

    {
      "allocation": "in_place",
      "base_vocab_size": 200064,
      "special_token_offset": 0,
      "special_token_count": 124,
      "modalities": [
        {"name": "vision", "offset": 200064, "vocab_size": 131072,
         "start_token": 27, "end_token": 28},
        {"name": "audio",  "offset": 331136, "vocab_size": 4096,
         "start_token": 33, "end_token": 34}
      ]
    }

Either shape may additionally carry an explicit ``special_token_ids`` list; when
present it is authoritative for the special-token id set. The extracted set always
means ALL of the model's special tokens (text control tokens AND omni structure
tokens), so it is only derivable from that list or from the in_place
``special_token_offset``/``count`` block. Whenever an ``omnimodal_config`` is present
the set is required -- append configs without the list are rejected (the appended omni
block alone would miss the base text specials).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpecialTokens:
    """ALL of the model's special-token ids, extracted from a tokenizer's ``omnimodal_config``.

    This is the complete special-token set of the vocabulary (text control tokens such
    as bos/eos/pad/chat tokens AND omni structure tokens), not just the omni subset --
    consumers like the goldfish exemption rely on that completeness. It is only
    constructed from config shapes that can actually describe the full set: an explicit
    ``special_token_ids`` list, or the in_place ``special_token_offset``/``count`` block.

    Attributes:
        allocation: ``"append"`` or ``"in_place"`` -- the tokenizer's allocation mode.
        full_ids: Sorted list of every special-token id (always populated).
        id_range: Contiguous ``(start, end)`` span when ``full_ids`` is one run,
            ``None`` for scattered explicit lists.
    """

    allocation: str
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
    """

    name: str
    offset: Optional[int] = None
    vocab_size: Optional[int] = None
    start_token: Optional[int] = None
    end_token: Optional[int] = None


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


def extract_model_special_tokens(
    omnimodal_config: Optional[Dict[str, Any]]
) -> Optional[ModelSpecialTokens]:
    """Extract the model's FULL special-token id set from an ``omnimodal_config``.

    Resolution order:

    1. Explicit ``special_token_ids`` list (either allocation), authoritative when
       present; handles ids scattered across disjoint ranges (how "append" builds of
       https://github.com/swiss-ai/apertus-omni-tokenizer expose base text specials
       plus the appended omni block).
    2. ``in_place``: the contiguous ``special_token_offset`` + ``special_token_count``
       block (the base's FULL special-token block -- text control tokens, omni
       structure tokens, and the reserve pool).

    Whenever omni metadata is present, the full special-token set is REQUIRED: a config
    that cannot describe it raises. There is deliberately no derived fallback for
    ``append`` configs without an explicit list -- the omni block
    ``[base_vocab_size, min modality offset)`` covers only the omni structure tokens,
    not the base text specials, so it cannot stand in for the full set.

    Returns a :class:`ModelSpecialTokens`, or ``None`` only when ``omnimodal_config``
    is ``None`` (non-omni tokenizer).

    Raises:
        ValueError: When the config cannot describe the full set -- ``in_place``
            without ``special_token_offset``/``special_token_count``, or ``append``
            without ``special_token_ids``. Failing loudly beats silently disabling
            downstream consumers such as the goldfish exemption.
    """
    if omnimodal_config is None:
        return None

    allocation = omnimodal_config.get("allocation", "append")

    # 1. Explicit id list: authoritative for both allocations.
    explicit_ids = omnimodal_config.get("special_token_ids")
    if explicit_ids:
        full_ids = sorted({int(i) for i in explicit_ids})
        return ModelSpecialTokens(
            allocation=allocation, full_ids=full_ids, id_range=_contiguous_range(full_ids)
        )

    # 2. in_place: the base's full special block as offset + count.
    if allocation == "in_place":
        offset = omnimodal_config.get("special_token_offset")
        count = omnimodal_config.get("special_token_count")
        if offset is None or count is None:
            raise ValueError(
                "omnimodal_config has allocation=in_place but no special_token_offset/"
                "special_token_count (and no special_token_ids); cannot derive the "
                "special-token id set (malformed or unsupported config)."
            )
        start, end = int(offset), int(offset) + int(count)
        return ModelSpecialTokens(
            allocation="in_place", full_ids=list(range(start, end)), id_range=(start, end)
        )

    # append without an explicit list: the config cannot describe the full set.
    raise ValueError(
        "append-mode omnimodal_config carries no special_token_ids list; when omni "
        "metadata is present the model's FULL special-token set is required, and the "
        "derivable [base_vocab_size, min modality offset) omni block would miss the "
        "base text specials. Rebuild the tokenizer with a special_token_ids list."
    )


def extract_omni_metadata(
    base_vocab_size: Optional[int], omnimodal_config: Optional[Dict[str, Any]]
) -> Optional[OmniMetadata]:
    """Read all omni metadata into an :class:`OmniMetadata` dataclass.

    Pure function of the resolved ``base_vocab_size`` + ``omnimodal_config``; it does not
    touch ``args`` or the tokenizer, so the read model is decoupled from how the values
    are later distributed. Returns ``None`` only when both inputs are ``None``.
    """
    if base_vocab_size is None and omnimodal_config is None:
        return None

    # in_place configs duplicate base_vocab_size inside the omnimodal_config; fall back
    # to it when the top-level/init_kwargs value is absent.
    if base_vocab_size is None and omnimodal_config is not None:
        base_vocab_size = omnimodal_config.get("base_vocab_size")

    modalities: List[ModalityInfo] = []
    if omnimodal_config is not None:
        for modality in omnimodal_config.get("modalities", []):
            name = modality.get("name")
            if not name:
                continue
            modalities.append(
                ModalityInfo(
                    name=name,
                    offset=_int_or_none(modality.get("offset")),
                    vocab_size=_int_or_none(modality.get("vocab_size")),
                    start_token=_int_or_none(modality.get("start_token")),
                    end_token=_int_or_none(modality.get("end_token")),
                )
            )

    return OmniMetadata(
        base_vocab_size=_int_or_none(base_vocab_size),
        modalities=tuple(modalities),
        special_tokens=extract_model_special_tokens(omnimodal_config),
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

    Metadata source: the tokenizer ``init_kwargs`` keys ``base_vocab_size`` and
    ``omnimodal_config`` (i.e. ``tokenizer_config.json``).

    Returns:
        The :class:`OmniMetadata`, or ``None`` when the tokenizer has no omni metadata.
    """
    init_kwargs = extract_tokenizer_init_kwargs(tokenizer)
    base_vocab_size = init_kwargs.get("base_vocab_size")
    omnimodal_config = init_kwargs.get("omnimodal_config")

    omni = extract_omni_metadata(base_vocab_size, omnimodal_config)
    if omni is None:
        return None

    names = [modality.name for modality in omni.modalities]
    log_single_rank(logger, logging.INFO, f" > loaded omnimodal_config with modalities: {names}")

    args.omni_metadata = omni
    return omni
