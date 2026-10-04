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


@pytest.mark.parametrize("enabled", [False, True])
def test_kda_fused_norm_construction(enabled):
    # Exercise the factory with a CPU constructor spy; actual FLA numerics below
    # require CUDA. In particular, weight=None must not receive TP metadata.
    class RecordingNorm(torch.nn.RMSNorm):
        def __init__(self, hidden_size, activation, **kwargs):
            self.activation = activation
            super().__init__(hidden_size, **kwargs)

    config = SimpleNamespace(
        non_affine_kda_output_norm=enabled, normalization="RMSNorm",
        layernorm_epsilon=1e-6, params_dtype=torch.float32, sequence_parallel=True,
    )
    norm = _module.build_kda_output_norm(config, 8, None, RecordingNorm, device="cpu")
    assert norm.activation == "sigmoid"
    assert norm.elementwise_affine is not enabled
    if enabled:
        assert norm.weight is None
        assert norm.state_dict() == {}
    else:
        assert norm.weight.requires_grad and norm.weight.sequence_parallel


@pytest.mark.parametrize("enabled", [False, True])
def test_kda_unfused_norm_construction(enabled):
    config = SimpleNamespace(
        non_affine_kda_output_norm=enabled, normalization="RMSNorm", layernorm_epsilon=1e-6,
    )
    original = torch.nn.RMSNorm(8, eps=config.layernorm_epsilon)
    norm = _module.build_kda_output_norm(config, 8, original)
    if enabled:
        assert isinstance(norm, NonAffineNorm)
        assert list(norm.parameters()) == []
        x = torch.randn(3, 8, requires_grad=True)
        torch.testing.assert_close(norm(x), original(x))
    else:
        assert norm is original


def test_kda_per_head_gain_absorption():
    # KDA shares one gain vector across all value heads; projection columns do
    # not share gains, so repeat the norm gain across heads before absorbing it.
    heads, dim = 3, 7
    x = torch.randn(2, heads, dim, dtype=torch.float64, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    gain = torch.rand(dim, dtype=torch.float64) + 0.5
    matrix = torch.randn(11, heads * dim, dtype=torch.float64)
    config = SimpleNamespace(
        normalization="RMSNorm", layernorm_epsilon=1e-6, non_affine_kda_output_norm=True,
    )
    norm = _module.build_kda_output_norm(config, dim, None)
    normalized = norm(x)
    actual = (normalized * gate.sigmoid()).flatten(1) @ (matrix * gain.repeat(heads)).T
    expected = (normalized * gain * gate.sigmoid()).flatten(1) @ matrix.T
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    dy = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (x, gate), dy, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (x, gate), dy)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fla_non_affine_gated_forward_backward(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    fused_norm = pytest.importorskip("fla.modules.fused_norm_gate").FusedRMSNormGated
    config = SimpleNamespace(
        non_affine_kda_output_norm=True, normalization="RMSNorm",
        layernorm_epsilon=1e-6, params_dtype=dtype, sequence_parallel=False,
    )
    norm = _module.build_kda_output_norm(config, 64, None, fused_norm, device="cuda")
    assert norm.weight is None and list(norm.parameters()) == []
    x = torch.randn(2, 5, 3, 64, device="cuda", dtype=dtype, requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_gate = gate.detach().clone().requires_grad_()
    expected = (_reference(ref_x.float(), "RMSNorm", norm.eps)
                * ref_gate.float().sigmoid()).to(dtype)
    actual = norm(x, gate)
    dy = torch.randn_like(actual)
    actual.backward(dy)
    expected.backward(dy)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
    for result, reference in [(actual, expected), (x.grad, ref_x.grad), (gate.grad, ref_gate.grad)]:
        torch.testing.assert_close(result, reference, rtol=tolerance, atol=tolerance)
