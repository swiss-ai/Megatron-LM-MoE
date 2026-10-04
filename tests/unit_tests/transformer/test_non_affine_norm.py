# Copyright (c) 2026, Swiss AI Initiative.
"""Numerics tests that can also run without Megatron's distributed dependencies.

CPU-only invocation: pytest --noconftest tests/unit_tests/transformer/test_non_affine_norm.py
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

# Load this self-contained kernel module without initializing Megatron/TE.
_path = Path(__file__).resolve().parents[3] / "megatron/core/transformer/non_affine_norm.py"
_spec = importlib.util.spec_from_file_location("non_affine_norm_under_test", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
NonAffineNorm = _module.NonAffineNorm


def _reference(x, normalization, eps):
    value = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    if normalization == "LayerNorm":
        out = F.layer_norm(value, (x.shape[-1],), eps=eps)
    else:
        out = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return out.to(x.dtype)


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden", [37, 256, 4096])
def test_forward_backward(normalization, device, dtype, hidden):
    if device == "cuda" and (not torch.cuda.is_available() or _module.triton is None):
        pytest.skip("CUDA and Triton required")
    config = SimpleNamespace(normalization=normalization, layernorm_epsilon=1e-6)
    norm = NonAffineNorm(config, hidden)
    assert list(norm.parameters()) == []
    assert norm.state_dict() == {}
    # Exercise non-contiguous input and random incoming gradients, not just sum loss.
    x = torch.randn(2, 3, hidden, device=device, dtype=dtype).transpose(0, 1).requires_grad_()
    ref = x.detach().clone().requires_grad_()
    dy = torch.randn_like(x)
    actual = norm(x)
    expected = _reference(ref, normalization, norm.eps)
    actual.backward(dy)
    expected.backward(dy)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(x.grad, ref.grad, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
def test_gradcheck(normalization):
    norm = NonAffineNorm(SimpleNamespace(normalization=normalization, layernorm_epsilon=1e-5), 7)
    x = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(norm, (x,))


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
def test_empty_and_zero_input(normalization):
    norm = NonAffineNorm(SimpleNamespace(normalization=normalization, layernorm_epsilon=1e-5), 8)
    for shape in [(0, 8), (2, 8)]:
        x = torch.zeros(shape, requires_grad=True)
        out = norm(x)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert torch.isfinite(x.grad).all()


def test_wrong_hidden_size():
    norm = NonAffineNorm(SimpleNamespace(normalization="RMSNorm", layernorm_epsilon=1e-5), 8)
    with pytest.raises(ValueError, match="Expected hidden size"):
        norm(torch.randn(2, 7))
