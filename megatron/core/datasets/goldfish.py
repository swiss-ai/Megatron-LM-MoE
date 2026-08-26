# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Goldfish loss: deterministic ~1/k token dropping with special-token exemption.

Implements Hans et al., NeurIPS 2024 (https://arxiv.org/abs/2406.10209; reference
implementation https://github.com/ahans30/goldfish-loss): for each
position, a hash of the window of h labels ending at it decides whether the position
is dropped from the loss. All state (hash table, coefficients, exemption LUT) is
fixed-seed or config-derived and memoized per process. Consumed by
``GPTDataset.__getitem__`` in ``megatron.core.datasets.gpt_dataset``; see
``docs/goldfish_loss.md``.
"""

import logging
import os
from typing import Optional

import torch

from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

# Sentinel written into a cloned labels tensor at dropped positions, and the size
# (a prime) of the precomputed random hash table. The sentinel is negative so it can
# never collide with a real token id at the point of comparison; it never reaches the
# model (GPTDataset.__getitem__ only uses it to zero loss_mask on a cloned tensor).
GOLDFISH_TOKEN_ID = -2
_HASH_TABLE_SIZE = 1_000_003


def _create_hash_table(device):
    """Goldfish loss pre-computed random hash table.

    A fixed-seed random float table indexed by the hashed token context. The
    deterministic seed ensures the same token context is always dropped, both
    within and across runs.
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(2971215073)
    return torch.rand(_HASH_TABLE_SIZE, device=device, generator=rng)


def _create_hash_coefficients(width: int, device) -> torch.Tensor:
    """Fixed-seed odd int64 coefficients for the goldfish window hash.

    Hashing each window as a coefficient dot product (mod table size, int64 wrap-around
    is deterministic) is order-sensitive and free of the plain product hash's
    degeneracies: with a product, an id-0 label collapses the whole window to
    ``hash_table[0]`` (one shared drop decision for every window containing a pad/unk),
    and id-1 labels are invisible. Odd coefficients keep every label contributing under
    wrap-around.
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(2971215073 + width)
    coef = torch.randint(0, 2**62, (width,), dtype=torch.int64, device=device, generator=rng)
    return coef | 1


# Goldfish state (hash table, coefficients, exemption LUT) is identical for every
# dataset instance (fixed seeds, config-derived ids), so it is memoized at module
# level under distinct key prefixes: blended runs construct one GPTDataset per blend
# component per split, and each private hash-table copy would cost 4 MB per instance
# per dataloader worker.
_GOLDFISH_CACHE = {}


def get_hash_table(device) -> torch.Tensor:
    """Module-memoized :func:`_create_hash_table` (one shared table per device)."""
    key = ("table", str(device))
    if key not in _GOLDFISH_CACHE:
        _GOLDFISH_CACHE[key] = _create_hash_table(device)
    return _GOLDFISH_CACHE[key]


def _get_hash_coefficients(width: int, device) -> torch.Tensor:
    """Module-memoized :func:`_create_hash_coefficients`."""
    key = ("coef", int(width), str(device))
    if key not in _GOLDFISH_CACHE:
        _GOLDFISH_CACHE[key] = _create_hash_coefficients(width, device)
    return _GOLDFISH_CACHE[key]


def get_exemption_lut(exemption_ids: tuple, vocab_size: int, device) -> torch.Tensor:
    """Module-memoized :func:`_create_exemption_lut`."""
    key = ("exempt", exemption_ids, int(vocab_size), str(device))
    if key not in _GOLDFISH_CACHE:
        _GOLDFISH_CACHE[key] = _create_exemption_lut(exemption_ids, vocab_size, device)
    return _GOLDFISH_CACHE[key]


def _create_exemption_lut(exemption_ids, vocab_size: int, device) -> torch.Tensor:
    """Boolean lookup table (length ``vocab_size``) marking Goldfish-exempt token ids.

    ``table[labels]`` is an O(1)-per-token membership test that handles arbitrary
    contiguous or scattered exemption id sets.
    """
    vocab_size = int(vocab_size)
    ids = [int(token_id) for token_id in exemption_ids]

    lut = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    lut[torch.as_tensor(ids, dtype=torch.long, device=device)] = True
    return lut


def apply_goldfish(
    labels: torch.Tensor,
    goldfish_token_id: int,
    k: int,
    goldfish_hash_table: torch.Tensor,
    goldfish_context_width: int = 4,
    exemption_lut: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Deterministically drop ~1/k of tokens from the loss (Goldfish loss).

    For each position, hash the window of ``goldfish_context_width`` labels ending at
    it (dot product with fixed random coefficients, mod the hash-table size; see
    :func:`_create_hash_coefficients`) into ``goldfish_hash_table``; where the hashed
    value is ``< 1/k`` the predicted token is replaced by ``goldfish_token_id``.
    Because the decision is a pure function of the local context, the same token in
    the same context is always dropped, mitigating verbatim memorization while
    staying fully reproducible. In packed samples a window may span document
    boundaries: a duplicated document's drop mask is identical wherever the window
    lies fully inside it -- only its first ``goldfish_context_width - 1`` positions
    can vary with the packing neighbor.

    ``exemption_lut`` (a boolean lookup table over token ids) cancels drops for exempt
    tokens (e.g. BOS/EOS/PAD and chat-control tokens); see :func:`_create_exemption_lut`.

    Original implementation: https://github.com/ahans30/goldfish-loss

    Args:
        labels (torch.Tensor): 1D label tensor (as used in GPTDataset.__getitem__).
        goldfish_token_id (int): Sentinel id written at dropped positions.
        k (int): Drop probability is 1 / k.
        goldfish_hash_table (torch.Tensor): Precomputed random table.
        goldfish_context_width (int): Window width; the h labels ending at (and
            including) a position are hashed to decide its drop.
        exemption_lut (Optional[torch.Tensor]): Bool table where ``lut[id]`` marks an
            exempt token id; drops at those positions are cancelled.

    Returns:
        torch.Tensor: A clone of ``labels`` with dropped positions set to
        ``goldfish_token_id`` (the original ``labels`` are not mutated).
    """
    assert labels.ndim == 1, "Expected 1D tensor as used within GPTDataset.__getitem__"
    masked_labels = labels.clone()

    # Order-sensitive dot-product hash of each h-token window; int64 overflow is
    # intentional (deterministic), and % prime spreads it across the table.
    coefficients = _get_hash_coefficients(goldfish_context_width, labels.device)
    window_keys = (labels.unfold(0, goldfish_context_width, 1) * coefficients).sum(dim=-1)
    hashed_keys = goldfish_hash_table[window_keys % _HASH_TABLE_SIZE]

    dropped_token_indices = hashed_keys < 1 / k

    if exemption_lut is not None:
        # Slice by (context_width - 1) so the exempt mask aligns with the unfold
        # window / output positions.
        exempt_mask = exemption_lut[labels]
        exempt_tail = exempt_mask[goldfish_context_width - 1 :]
        if os.getenv("GOLDFISH_EXEMPT_LOG") == "1":
            exempt_dropped = (dropped_token_indices & exempt_tail).nonzero(as_tuple=True)[0]
            if exempt_dropped.numel() > 0:
                label_idx = exempt_dropped + (goldfish_context_width - 1)
                sample = torch.stack((label_idx[:5], labels[label_idx[:5]]), dim=1).cpu().tolist()
                log_single_rank(
                    logger,
                    logging.INFO,
                    f"Goldfish exemption cancels {exempt_dropped.numel()} drops; "
                    f"sample (idx, token_id){sample}",
                )
        dropped_token_indices &= ~exempt_tail

    masked_labels[goldfish_context_width - 1 :][dropped_token_indices] = goldfish_token_id

    return masked_labels
