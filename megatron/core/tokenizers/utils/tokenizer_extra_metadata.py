# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Define and extract runtime tokenizer metadata with two components.

This module defines a common tokenizer metadata structure and helpers that populate it
from a HuggingFace tokenizer. Only HuggingFace-backed tokenizers are supported: the
extraction walks Megatron's wrapper chain to the underlying
``transformers.PreTrainedTokenizerBase`` and reads its declared surface directly.
Non-HuggingFace tokenizers (sentencepiece, tiktoken, byte-level, null) yield no
metadata, which is an error when Goldfish loss is enabled because its special-token
exemptions would be silently empty.

It stores two classes of information:

1. Special-token IDs
- Goldfish loss uses these IDs to avoid dropping control tokens.
- The set is the union of ``all_special_ids`` (named roles plus registered additional
  special tokens) and every ``added_tokens_decoder`` entry flagged special, which
  covers reserved and chat-template tokens not registered as additional specials.
- For omni tokenizers, the set is filtered to IDs below ``base_vocab_size`` plus every
  named modality structure-token ID. The base-vocabulary filter keeps modality content
  tokens Goldfish-eligible because omni artifacts may also mark those tokens special.

2. Omnimodal layout information
- Optional: ``omni=None`` if ``omnimodal_config`` is absent.
- Parsed from the top-level ``base_vocab_size`` and ``omnimodal_config`` keys of the
  tokenizer's ``init_kwargs`` (the extra entries of ``tokenizer_config.json``).
- Validation: Modality content ranges are normalized into offset order and must not overlap.
  Structure-token IDs may be in the base vocabulary or a separate OMNI special range,
  but never inside a modality content range. Modalities must declare
  ``structure_token_ids``; legacy Apertus 1.5 configs that omit them are rejected by
  design (correct multimodal masking cannot be derived from ranges alone).

Example of expected structure:

  {
      "base_vocab_size": 200064,
      "omnimodal_config": {
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
                  },
              }
          ],
      },
  }

Tokenizer metadata is validated against this expected structure. See the
`Apertus omni-tokenizer repository
<https://github.com/swiss-ai/apertus-omni-tokenizer>`_ for its definition and details
about creating extended tokenizers.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpecialTokens:
    """Goldfish exemption IDs derived from a tokenizer.

    For plain-text tokenizers, this contains every discovered special-token
    ID: ``all_special_ids`` plus added tokens flagged special. For omni tokenizers,
    it contains discovered IDs below ``base_vocab_size`` plus every named modality
    structure-token ID. Omni content-token IDs are excluded even when marked special so
    they remain eligible for Goldfish drops.

    Attributes:
        full_ids: Sorted, unique Goldfish exemption IDs; empty when none are discovered.
    """

    full_ids: List[int]


@dataclass(frozen=True)
class ModalityInfo:
    """Container for metadata of one modality.

    Attributes:
        name: Modality name, such as ``"vision"`` or ``"audio"``.
        offset: Content-token block start id.
        vocab_size: Number of content tokens.
        start_token: Modality start-token ID.
        end_token: Modality end-token ID.
        structure_token_ids: Complete structure-token name-to-ID map.
    """

    name: str
    offset: int
    vocab_size: int
    start_token: int
    end_token: int
    structure_token_ids: Dict[str, int]

    @property
    def structure_ids(self) -> Tuple[int, ...]:
        """Sorted unique structure-token ids (the values of ``structure_token_ids``)."""
        return tuple(sorted(set(self.structure_token_ids.values())))


@dataclass(frozen=True)
class OmniMetadata:
    """Container holding all parsed omnimodal information of a tokenizer.

    Attributes:
        base_vocab_size: Exclusive upper bound of the text vocabulary.
        modalities: tuple of modality info containers, one for each modality
    """

    base_vocab_size: int
    modalities: Tuple[ModalityInfo, ...]

    def modality(self, name: str) -> Optional[ModalityInfo]:
        """Return the :class:`ModalityInfo` named ``name``, or ``None`` if absent."""
        return next((m for m in self.modalities if m.name == name), None)


@dataclass(frozen=True)
class TokenizerExtraMetadata:
    """Container for all tokenizer metadata.

    Contains model special tokens in all cases and omni-metadata if it exists.

    Attributes:
        special_tokens: Goldfish exemption IDs for any tokenizer type.
        omni: Validated omni layout, or ``None`` for text-only tokenizers.
    """

    special_tokens: ModelSpecialTokens
    omni: Optional[OmniMetadata] = None


def _validate_object_fields(value: Any, path: str, allowed_fields: set) -> Dict[str, Any]:
    """Validate that ``value`` is an object containing only allowed string keys."""
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    invalid_keys = [key for key in value if not isinstance(key, str)]
    if invalid_keys:
        raise ValueError(f"{path} field names must be strings, got {invalid_keys!r}")
    unexpected = sorted(value.keys() - allowed_fields)
    if unexpected:
        raise ValueError(f"{path} contains unexpected fields: {', '.join(unexpected)}")
    return value


def _require_int(value: Any, path: str) -> int:
    """Validate a JSON integer without coercing booleans or strings."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _parse_modalities(
    base_vocab_size: int, omnimodal_config: Dict[str, Any]
) -> Tuple[ModalityInfo, ...]:
    """Validate and parse the supported ``omnimodal_config`` contract.

    Checks field presence and types, overlapping modality content ranges, and
    misplaced structure tokens.
    """
    omnimodal_config = _validate_object_fields(
        omnimodal_config, "omnimodal_config", {"omni_special_token_offset", "modalities"}
    )

    omni_offset = omnimodal_config.get("omni_special_token_offset")
    if omni_offset is None:
        raise ValueError("omnimodal_config is missing required omni_special_token_offset")
    omni_offset = _require_int(omni_offset, "omnimodal_config.omni_special_token_offset")
    if omni_offset != base_vocab_size:
        raise ValueError(
            "omnimodal_config.omni_special_token_offset must equal top-level "
            f"base_vocab_size ({omni_offset} != {base_vocab_size})"
        )

    raw_modalities = omnimodal_config.get("modalities")
    if not isinstance(raw_modalities, list) or not raw_modalities:
        raise ValueError("omnimodal_config.modalities must be a non-empty list")

    modality_fields = {
        "name",
        "offset",
        "vocab_size",
        "start_token",
        "end_token",
        "structure_token_ids",
    }
    parsed: List[ModalityInfo] = []
    names = set()
    for index, modality in enumerate(raw_modalities):
        path = f"omnimodal_config.modalities[{index}]"
        modality = _validate_object_fields(modality, path, modality_fields)
        missing = sorted(modality_fields - modality.keys())
        if missing:
            raise ValueError(f"{path} is missing required fields: {', '.join(missing)}")

        name = modality["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}.name must be a non-empty string")
        if name in names:
            raise ValueError(f"omnimodal_config contains duplicate modality name {name!r}")
        names.add(name)

        offset = _require_int(modality["offset"], f"{path}.offset")
        vocab_size = _require_int(modality["vocab_size"], f"{path}.vocab_size")
        start_token = _require_int(modality["start_token"], f"{path}.start_token")
        end_token = _require_int(modality["end_token"], f"{path}.end_token")
        raw_structure_ids = modality["structure_token_ids"]
        if not isinstance(raw_structure_ids, dict) or not raw_structure_ids:
            raise ValueError(f"{path}.structure_token_ids must be a non-empty object")
        structure_token_ids = {}
        for token_name, token_id in raw_structure_ids.items():
            if not isinstance(token_name, str) or not token_name:
                raise ValueError(f"{path}.structure_token_ids keys must be non-empty strings")
            structure_token_ids[token_name] = _require_int(
                token_id, f"{path}.structure_token_ids[{token_name!r}]"
            )

        if offset < base_vocab_size or vocab_size <= 0:
            raise ValueError(
                f"omnimodal_config modality {name!r} must have offset >= base_vocab_size "
                "and vocab_size > 0"
            )
        if min(start_token, end_token, *structure_token_ids.values()) < 0:
            raise ValueError(f"omnimodal_config modality {name!r} contains a negative token id")
        structure_values = set(structure_token_ids.values())
        if start_token not in structure_values or end_token not in structure_values:
            raise ValueError(
                f"omnimodal_config modality {name!r} start_token/end_token must appear "
                "in structure_token_ids"
            )

        parsed.append(
            ModalityInfo(
                name=name,
                offset=offset,
                vocab_size=vocab_size,
                start_token=start_token,
                end_token=end_token,
                structure_token_ids=structure_token_ids,
            )
        )

    # Validate Non Overlapping modality ranges
    parsed.sort(key=lambda modality: modality.offset)
    content_ranges = [
        (modality.offset, modality.offset + modality.vocab_size, modality.name)
        for modality in parsed
    ]
    for (_, previous_end, previous_name), (start, _, name) in zip(
        content_ranges, content_ranges[1:]
    ):
        if start < previous_end:
            raise ValueError(
                f"omnimodal_config content ranges overlap for {previous_name!r} and {name!r}"
            )

    # Validate misplaced structure tokens
    structure_ids = {
        token_id for modality in parsed for token_id in modality.structure_token_ids.values()
    }
    misplaced = sorted(
        token_id
        for token_id in structure_ids
        if any(start <= token_id < end for start, end, _ in content_ranges)
    )
    if misplaced:
        raise ValueError(
            f"omnimodal_config declares structure token ids {misplaced} inside a modality's "
            "content range [offset, offset + vocab_size)"
        )

    # Structure ids must be disjoint across modalities: consumers rely on it (the weight
    # LUT would resolve a shared id by iteration order, and the per-modality loss report
    # would double-count it, breaking its partition of 'lm loss').
    seen_ids = set()
    shared = set()
    for modality in parsed:
        modality_ids = set(modality.structure_token_ids.values())
        shared |= modality_ids & seen_ids
        seen_ids |= modality_ids
    if shared:
        raise ValueError(
            f"omnimodal_config declares structure token ids {sorted(shared)} in more than "
            "one modality; structure tokens must be disjoint across modalities"
        )

    layout_size = max(base_vocab_size, *(end for _, end, _ in content_ranges))
    out_of_range = sorted(token_id for token_id in structure_ids if token_id >= layout_size)
    if out_of_range:
        raise ValueError(
            f"omnimodal_config declares structure token ids {out_of_range} outside the "
            f"declared tokenizer layout [0, {layout_size})"
        )

    return tuple(parsed)


def _find_hf_tokenizer(tokenizer: Any) -> Optional[Any]:
    """Return the underlying HuggingFace tokenizer, or ``None`` if the chain has none.

    Follows Megatron's ``_tokenizer``/``tokenizer`` wrapper links (for example
    ``MegatronTokenizerText._tokenizer`` -> ``HuggingFaceTokenizer.tokenizer``) until an
    object carrying an ``init_kwargs`` dictionary is found, the
    ``PreTrainedTokenizerBase`` signature on transformers 4.x and 5.x.
    """
    queue = [tokenizer]
    seen = set()
    while queue:
        obj = queue.pop(0)
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(getattr(obj, "init_kwargs", None), dict):
            return obj
        queue.extend(getattr(obj, attr, None) for attr in ("_tokenizer", "tokenizer"))
    return None


def _extract_special_token_ids(hf_tokenizer: Any) -> List[int]:
    """Return sorted special-token IDs declared by a HuggingFace tokenizer.

    Unions ``all_special_ids`` (named roles plus registered additional special tokens)
    with ``added_tokens_decoder`` entries flagged special, which covers reserved and
    chat-template tokens that are not registered as additional specials.
    """
    # The getattr defaults stay: _find_hf_tokenizer duck-types on init_kwargs alone.
    ids = set(getattr(hf_tokenizer, "all_special_ids", None) or [])
    decoder = getattr(hf_tokenizer, "added_tokens_decoder", None) or {}
    ids.update(token_id for token_id, token in decoder.items() if getattr(token, "special", False))
    return sorted(ids)


def parse_model_special_tokens(
    special_ids: List[int], omni: Optional[OmniMetadata] = None
) -> ModelSpecialTokens:
    """Build Goldfish exemption IDs for text-only or omni tokenizers.

    Text-only tokenizers retain all discovered special IDs. Omni tokenizers retain
    discovered IDs below the base vocabulary and all named structure-token IDs, keeping
    modality content tokens Goldfish-eligible.
    """
    general_ids = set(special_ids)
    if omni is None:
        full_ids = sorted(general_ids)
    else:
        text_special_ids = {token_id for token_id in general_ids if token_id < omni.base_vocab_size}
        structure_ids = {
            token_id
            for modality in omni.modalities
            for token_id in modality.structure_token_ids.values()
        }
        full_ids = sorted(text_special_ids | structure_ids)
    return ModelSpecialTokens(full_ids=full_ids)


def parse_omni_metadata(
    base_vocab_size: Optional[int], omnimodal_config: Optional[Dict[str, Any]]
) -> Optional[OmniMetadata]:
    """Validate omni configuration, returning ``None`` when it is absent.

    Given omnimodal config values from tokenizer, parse into data-classes and validate.
    """
    if omnimodal_config is None:
        return None

    if base_vocab_size is None:
        raise ValueError(
            "omnimodal_config requires top-level base_vocab_size; fallback to "
            "omni_special_token_offset is unsupported"
        )
    base_vocab_size = _require_int(base_vocab_size, "base_vocab_size")
    if base_vocab_size <= 0:
        raise ValueError(f"base_vocab_size must be positive, got {base_vocab_size}")

    modalities = _parse_modalities(base_vocab_size, omnimodal_config)

    return OmniMetadata(base_vocab_size=base_vocab_size, modalities=modalities)


def extract_tokenizer_extra_metadata(
    base_vocab_size: Optional[int],
    omnimodal_config: Optional[Dict[str, Any]],
    special_ids: List[int],
) -> TokenizerExtraMetadata:
    """Orchestrator: build special-token metadata and an optional omni container.

    Parse Omnimodal configuration if given and build model special tokens.
    """
    omni = parse_omni_metadata(base_vocab_size, omnimodal_config)
    return TokenizerExtraMetadata(
        special_tokens=parse_model_special_tokens(special_ids, omni=omni), omni=omni
    )


def populate_tokenizer_extra_metadata_from_tokenizer(
    args, tokenizer: Any
) -> Optional[TokenizerExtraMetadata]:
    """Populate ``args.tokenizer_extra_metadata`` from a tokenizer wrapper chain.

    Clears stale extra metadata before extraction, validates omni configuration when
    present, and logs whether the tokenizer is text-only or omni. Non-HuggingFace
    tokenizers carry no extra metadata and leave ``None``; this is an error when
    Goldfish loss is enabled because its special-token exemptions would be silently
    empty.
    """
    args.tokenizer_extra_metadata = None

    hf_tokenizer = _find_hf_tokenizer(tokenizer)
    if hf_tokenizer is None:
        if getattr(args, "goldfish_loss", False):
            raise ValueError(
                "Goldfish loss requires a HuggingFace-backed tokenizer to derive its "
                "special-token exemptions, but none was found in the wrapper chain of "
                f"{type(tokenizer).__qualname__}."
            )
        log_single_rank(
            logger,
            logging.INFO,
            " > tokenizer is not HuggingFace-backed; no extra metadata extracted",
        )
        return None

    # extract
    init_kwargs = hf_tokenizer.init_kwargs
    base_vocab_size = init_kwargs.get("base_vocab_size")
    omnimodal_config = init_kwargs.get("omnimodal_config")
    special_ids = _extract_special_token_ids(hf_tokenizer)
    if omnimodal_config is not None and not special_ids:
        raise ValueError(
            "omnimodal_config is present but the tokenizer declares no special tokens; "
            "refusing to build Goldfish exemptions from structure tokens alone"
        )

    # parse
    metadata = extract_tokenizer_extra_metadata(base_vocab_size, omnimodal_config, special_ids)

    log_single_rank(
        logger,
        logging.INFO,
        f" > loaded tokenizer metadata with {len(metadata.special_tokens.full_ids)} special ids",
    )
    if metadata.omni is None:
        log_single_rank(
            logger, logging.INFO, " > no omnimodal_config found; tokenizer is treated as text-only"
        )
    else:
        names = [modality.name for modality in metadata.omni.modalities]
        log_single_rank(
            logger, logging.INFO, f" > validated omnimodal_config with modalities: {names}"
        )

    args.tokenizer_extra_metadata = metadata
    return metadata
