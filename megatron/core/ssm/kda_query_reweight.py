# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.

"""Sliding-window mass normalisation of the KDA read-out (query reweighting).

Row j of a KDA state accumulates a decayed token count ``W_t = alpha_t W_{t-1} + 1``; rows whose
memory is longer than the training length keep growing beyond it, and their size then encodes the
absolute position. Scaling key channel j of the (unit-norm) query by

    c_{t,j} = W^(w)_{t,j} / W_{t,j}

reads every row as if it had only accumulated the last ``w`` tokens: ``c = 1`` for ``t < w`` and
for rows that forget within ``w`` tokens, ``c ~ w / t`` for a row that has held since the start.
The state, the recurrence, the keys and the values are untouched; only how strongly each row is
read changes. ``c`` is treated as a constant (no gradient).
"""

import torch


def mass_ratio(log_alpha: torch.Tensor, window: int) -> torch.Tensor:
    """``W^(window) / W`` per token and key channel, same shape as ``log_alpha`` ([b, s, h, d]).

    Computed in log space in fp64: with ``G_t = sum_{s<=t} log alpha_s``,
    ``log W_t = G_t + logcumsumexp(-G)_t`` and the mass older than the window is
    ``exp(G_t - G_{t-w}) W_{t-w}``, so ``W^(w)/W = 1 - exp(lse_{t-w} - lse_t)``.
    """
    seq_len = log_alpha.shape[1]
    if seq_len <= window:
        return torch.ones_like(log_alpha)
    with torch.no_grad():
        G = torch.cumsum(log_alpha.double(), dim=1)
        lse = torch.logcumsumexp(-G, dim=1)
        lse_old = torch.full_like(lse, float("-inf"))
        lse_old[:, window:] = lse[:, :-window]
        frac_old = torch.exp((lse_old - lse).clamp(max=0.0)).clamp(max=1 - 1e-12)
        return (1.0 - frac_old).to(log_alpha.dtype)
