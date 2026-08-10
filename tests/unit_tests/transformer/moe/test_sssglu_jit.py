"""Tests for the Triton SSSGLU kernels (``megatron/core/transformer/moe/sssglu_jit.py``).

These are the fp32-internal Triton kernels used on the FP8 (offloading-)experts path (the sibling
of ``rlglu_jit.py`` / ``swiglu_jit.py``). SSSGLU is a GLU whose gate is a shifted, scaled softsign
applied directly: ``gate(a) = softsign(a - (sqrt(2)-1)) + (1 - 1/sqrt(2))``. Forward and backward (including the per-row
``probs`` scaling and its gradient) are checked against a pure-torch autograd reference. CUDA-gated
like the other kernel tests; run on the cluster GPU.
"""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

if torch.cuda.is_available():
    from megatron.core.transformer.moe.sssglu_jit import sssglu_forward, sssglu_backward


def _ref_forward(x, probs=None):
    """Reference ``gate(a) * b [* probs]`` with the [a|b] halves split along the last dim."""
    a, b = torch.chunk(x, 2, -1)
    y = (F.softsign(a - 0.41421356237309515) + 0.2928932188134524) * b
    if probs is not None:
        y = y * probs
    return y


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_probs", [False, True])
@pytest.mark.parametrize("shape", [(16, 64), (128, 512)])
def test_sssglu_jit_matches_autograd_reference(dtype, use_probs, shape):
    M, two_d = shape
    D = two_d // 2
    tols = dict(rtol=2.0e-2, atol=1.0e-3) if dtype == torch.bfloat16 else dict(rtol=1.0e-4, atol=1.0e-5)

    x = torch.randn(M, two_d, dtype=dtype, device="cuda")
    probs = torch.rand(M, 1, dtype=dtype, device="cuda") if use_probs else None
    g = torch.randn(M, D, dtype=dtype, device="cuda")

    # Reference forward + backward via autograd (fp32 math to match the kernel's internal fp32).
    x_ref = x.detach().float().requires_grad_(True)
    probs_ref = probs.detach().float().requires_grad_(True) if use_probs else None
    y_ref = _ref_forward(x_ref, probs_ref)
    y_ref.backward(g.float())

    # Kernel forward.
    y = sssglu_forward(x, probs)
    assert y.shape == (M, D)
    assert y.dtype == dtype
    assert torch.allclose(y.float(), y_ref.detach(), **tols)

    # Kernel backward.
    if use_probs:
        grad_x, grad_probs = sssglu_backward(g, x, probs)
        assert torch.allclose(
            grad_probs.float(), probs_ref.grad.view(-1), **tols
        )
    else:
        grad_x = sssglu_backward(g, x, probs)
    assert grad_x.shape == (M, two_d)
    assert torch.allclose(grad_x.float(), x_ref.grad, **tols)


def test_sssglu_jit_differs_from_swiglu():
    """Sanity check that the SSSGLU kernel is not accidentally computing SwiGLU."""
    from megatron.core.transformer.moe.swiglu_jit import swiglu_forward

    x = torch.randn(32, 128, dtype=torch.float32, device="cuda")
    assert not torch.allclose(sssglu_forward(x), swiglu_forward(x), rtol=1e-3, atol=1e-3)
