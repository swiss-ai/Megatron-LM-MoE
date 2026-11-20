# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Extract the tokenizer's special-token ids for Goldfish exemption; see ``docs/goldfish_loss.md``.

Text-only variant: reads the special-token surface of the underlying HuggingFace
tokenizer (``all_special_ids`` plus added tokens flagged special). No omnimodal layout
parsing is performed here.
"""

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpecialTokens:
    """Special-token IDs exempt from Goldfish masking."""

    full_ids: List[int]


@dataclass(frozen=True)
class TokenizerExtraMetadata:
    """Special-token metadata extracted from the tokenizer."""

    special_tokens: ModelSpecialTokens


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


def parse_model_special_tokens(special_ids: List[int]) -> ModelSpecialTokens:
    """Build Goldfish exemptions from the tokenizer's special-token ids."""
    return ModelSpecialTokens(full_ids=sorted(set(special_ids)))


def extract_tokenizer_extra_metadata(special_ids: List[int]) -> TokenizerExtraMetadata:
    """Build special-token metadata."""
    return TokenizerExtraMetadata(special_tokens=parse_model_special_tokens(special_ids))


def populate_tokenizer_extra_metadata_from_tokenizer(
    args, tokenizer: Any
) -> Optional[TokenizerExtraMetadata]:
    """Populate ``args.tokenizer_extra_metadata`` from a tokenizer.

    Returns ``None`` (and stores ``None``) when the tokenizer is not HuggingFace-backed,
    unless goldfish loss is enabled, in which case a ``ValueError`` is raised: the
    exemption set would otherwise be silently empty.
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

    special_ids = _extract_special_token_ids(hf_tokenizer)
    metadata = extract_tokenizer_extra_metadata(special_ids)

    log_single_rank(
        logger,
        logging.INFO,
        f" > loaded tokenizer metadata with {len(metadata.special_tokens.full_ids)} special ids",
    )

    args.tokenizer_extra_metadata = metadata
    return metadata
