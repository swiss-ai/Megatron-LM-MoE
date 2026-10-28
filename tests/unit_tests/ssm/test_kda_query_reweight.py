# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.

"""The sliding-window mass ratio against a direct sum (CPU only)."""

import torch

from megatron.core.ssm.kda_query_reweight import mass_ratio


def _reference(log_alpha, window):
    """W^(w)_t / W_t by the definition W_t = sum_{i<=t} prod_{i<s<=t} alpha_s."""
    alpha = log_alpha.double().exp()
    b, s, h, d = alpha.shape
    out = torch.empty_like(alpha)
    for t in range(s):
        w_full = torch.zeros(b, h, d, dtype=torch.float64)
        w_win = torch.zeros(b, h, d, dtype=torch.float64)
        for i in range(t + 1):
            weight = alpha[:, i + 1 : t + 1].prod(dim=1) if i < t else torch.ones(b, h, d, dtype=torch.float64)
            w_full += weight
            if i > t - window:
                w_win += weight
        out[:, t] = w_win / w_full
    return out


def test_matches_direct_sum():
    torch.manual_seed(0)
    log_alpha = -torch.rand(2, 24, 2, 3, dtype=torch.float32) * 0.5     # alpha in (0.6, 1)
    c = mass_ratio(log_alpha, 8)
    torch.testing.assert_close(c.double(), _reference(log_alpha, 8), rtol=1e-6, atol=1e-6)
    assert c.dtype == log_alpha.dtype


def test_window_and_limits():
    s, w = 40, 8
    # alpha = 1 (no forgetting): W_t = t + 1, W^(w) = min(t + 1, w) -> c = w / (t + 1) beyond the window
    log_alpha = torch.zeros(1, s, 1, 1)
    c = mass_ratio(log_alpha, w)[0, :, 0, 0]
    t = torch.arange(s, dtype=torch.float32)
    torch.testing.assert_close(c, torch.minimum(torch.ones(s), w / (t + 1)), rtol=1e-6, atol=1e-6)
    assert torch.all(c[:w] == 1)
    # a fast-forgetting row (alpha = 0.5) holds 2^-w of its mass from beyond the window: c ~ 1 - 2^-w
    c_fast = mass_ratio(torch.full((1, s, 1, 1), -0.6931), w)[0, :, 0, 0]
    assert torch.all(c_fast > 1 - 2.0 ** -w - 1e-3)
    # sequences no longer than the window are untouched
    assert torch.all(mass_ratio(torch.zeros(1, w, 1, 1), w) == 1)


def test_packed_documents_count_from_their_own_start():
    # cross-document masking: the ratio of a packed sequence is the per-document ratios side by side
    torch.manual_seed(1)
    docs, w = [30, 5, 17, 12], 8
    log_alpha = -torch.rand(1, sum(docs), 2, 3) * 0.3
    cu = torch.tensor([0] + torch.tensor(docs).cumsum(0).tolist(), dtype=torch.int32)
    packed = mass_ratio(log_alpha, w, cu)
    parts = [mass_ratio(log_alpha[:, s:e], w) for s, e in zip(cu[:-1].tolist(), cu[1:].tolist())]
    torch.testing.assert_close(packed, torch.cat(parts, dim=1), rtol=0, atol=0)
    torch.testing.assert_close(packed.double(), _reference_packed(log_alpha, w, docs), rtol=1e-6, atol=1e-6)
    assert torch.all(packed[:, 30:35] == 1)                 # a 5-token document is inside the window
    assert not torch.allclose(packed, mass_ratio(log_alpha, w))


def _reference_packed(log_alpha, window, docs):
    out, start = [], 0
    for n in docs:
        out.append(_reference(log_alpha[:, start : start + n], window))
        start += n
    return torch.cat(out, dim=1)
