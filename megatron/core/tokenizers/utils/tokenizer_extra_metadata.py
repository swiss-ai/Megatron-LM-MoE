# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Extract validated tokenizer metadata; see ``docs/omnimodal_metadata.md``."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpecialTokens:
    """Special-token IDs exempt from Goldfish masking."""

    full_ids: List[int]


@dataclass(frozen=True)
class ModalityInfo:
    """Validated token layout for one modality."""

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
    """Validated omnimodal tokenizer layout."""

    base_vocab_size: int
    modalities: Tuple[ModalityInfo, ...]

    def modality(self, name: str) -> Optional[ModalityInfo]:
        """Return the :class:`ModalityInfo` named ``name``, or ``None`` if absent."""
        return next((m for m in self.modalities if m.name == name), None)


@dataclass(frozen=True)
class TokenizerExtraMetadata:
    """Special-token metadata with an optional omni layout."""

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
    """Validate and parse ``omnimodal_config.modalities``."""
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

    # Content ranges must not overlap.
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

    # Structure tokens cannot be content tokens.
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

    # Structure-token ownership must be unique.
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
    """Find the HuggingFace tokenizer in a Megatron wrapper chain."""
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
    """Collect all special-token IDs from a HuggingFace tokenizer."""
    ids = set(getattr(hf_tokenizer, "all_special_ids", None) or [])
    decoder = getattr(hf_tokenizer, "added_tokens_decoder", None) or {}
    ids.update(token_id for token_id, token in decoder.items() if getattr(token, "special", False))
    return sorted(ids)


def parse_model_special_tokens(
    special_ids: List[int], omni: Optional[OmniMetadata] = None
) -> ModelSpecialTokens:
    """Build Goldfish exemptions from parsed tokenizer metadata."""
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
    """Validate omni configuration, returning ``None`` when absent."""
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
    """Build special-token and optional omni metadata."""
    omni = parse_omni_metadata(base_vocab_size, omnimodal_config)
    return TokenizerExtraMetadata(
        special_tokens=parse_model_special_tokens(special_ids, omni=omni), omni=omni
    )


def populate_tokenizer_extra_metadata_from_tokenizer(
    args, tokenizer: Any
) -> Optional[TokenizerExtraMetadata]:
    """Populate ``args.tokenizer_extra_metadata`` from a tokenizer."""
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

    init_kwargs = hf_tokenizer.init_kwargs
    base_vocab_size = init_kwargs.get("base_vocab_size")
    omnimodal_config = init_kwargs.get("omnimodal_config")
    special_ids = _extract_special_token_ids(hf_tokenizer)
    if omnimodal_config is not None and not special_ids:
        raise ValueError(
            "omnimodal_config is present but the tokenizer declares no special tokens; "
            "refusing to build Goldfish exemptions from structure tokens alone"
        )

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
