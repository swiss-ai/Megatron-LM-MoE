"""Tests for --residual-gate: the gate / pre-affine modules and the exact conversion of a plain residual stream."""

import argparse
import math

import pytest
import torch

from megatron.core.transformer.residual_gate import PreAffine, ResidualGate, residual_gate_plan
from megatron.core.transformer.transformer_config import TransformerConfig

H, K, C = 24, 8, 1.0 / math.sqrt(8)


def _rms(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt()


def _toy(seed=0):
    g = torch.Generator().manual_seed(seed)
    gammas = [torch.randn(H, generator=g, dtype=torch.float64) for _ in range(K)]
    gammas[0][:6] = torch.tensor([-0.7, 0.0, 1e-6, -1e-4, 3.0, 0.02], dtype=torch.float64)   # signs, dead and big gains
    mats = [torch.randn(H, H, generator=g, dtype=torch.float64) for _ in range(K)]
    pre = [torch.randn(H, generator=g, dtype=torch.float64) for _ in range(K)]
    x0 = torch.randn(5, H, generator=g, dtype=torch.float64)
    block = lambda k, stream_in: _rms(torch.tanh((pre[k] * _rms(stream_in)) @ mats[k].T))   # weightless post-norm output
    return gammas, block, x0


def _plain(gammas, block, x0):
    y, reads = x0.clone(), []
    for k in range(K):
        reads.append(_rms(y)); y = y + C * gammas[k] * block(k, y)
    return _rms(y), reads


def _gated(gammas, block, x0, dtype):
    gates = [ResidualGate(H) for _ in range(K)]
    values, signs, affines, final_affine, s = residual_gate_plan([C * g for g in gammas], gates, dtype)
    x, reads = x0.clone(), []
    for k in range(K):
        lam = PreAffine(H).double(); lam.weight.data.copy_(affines[k].double())
        gates[k].gate.data = values[k].clone()
        alpha, beta = gates[k].coefficients(gates[k].gate)
        reads.append(_rms(affines[k].double() * x))
        h = signs[k].double() * block(k, affines[k].double() * x)
        x = alpha.double() * x + beta.double() * h
    return _rms(final_affine.double() * x), reads, s


def test_conversion_reproduces_the_plain_stream_exactly_in_fp64():
    gammas, block, x0 = _toy()
    out_p, reads_p = _plain(gammas, block, x0)
    out_g, reads_g, s = _gated(gammas, block, x0, torch.float64)
    for a, b in zip(reads_p, reads_g):
        torch.testing.assert_close(b, a, rtol=0, atol=2e-5)      # fp32 sigmoid in the gate: ~1e-7 per sublayer
    torch.testing.assert_close(out_g, out_p, rtol=0, atol=2e-5)
    assert (s > 0).all() and (s <= 1).all()                       # the gated stream only shrinks


def test_conversion_with_bf16_parameters_does_not_accumulate_rounding():
    """bf16 rounds every stored value by up to 0.4%; the plan is built from the rounded gates, so every
    sublayer still reads the plain stream to within that rounding, at the last sublayer as at the first."""
    gammas, block, x0 = _toy(1)
    out_p, reads_p = _plain(gammas, block, x0)
    out_g, reads_g, _ = _gated(gammas, block, x0, torch.bfloat16)
    err = [float((b - a).abs().max() / a.abs().max()) for a, b in zip(reads_p, reads_g)]
    assert max(err) < 0.03 and err[-1] < 0.03
    assert float((out_g - out_p).abs().max() / out_p.abs().max()) < 0.03


def test_gate_matches_the_reference_formula_and_is_variance_preserving():
    torch.manual_seed(0)
    gate = ResidualGate(H, temperature=0.3)
    gate.gate.data = torch.randn(H)
    residual, hidden = torch.randn(3, H), torch.randn(3, H)
    beta = torch.sigmoid(gate.gate / 0.3)
    torch.testing.assert_close(gate(residual, hidden), torch.sqrt(1 - beta ** 2) * residual + beta * hidden, rtol=1e-5, atol=1e-6)
    alpha, beta = gate.coefficients(gate.gate)
    torch.testing.assert_close(alpha ** 2 + beta ** 2, torch.ones(H), rtol=1e-5, atol=0)
    out = gate(residual.bfloat16(), hidden.bfloat16())
    assert out.dtype == torch.bfloat16
    affine = PreAffine(H)
    assert torch.equal(affine(residual), residual) and affine(residual.bfloat16()).dtype == torch.bfloat16


def test_plan_signs_and_dead_gains():
    gates = [ResidualGate(4)]
    values, signs, affines, final_affine, s = residual_gate_plan([torch.tensor([-0.5, 0.0, 0.5, 2.0], dtype=torch.float64)], gates)
    assert signs[0].tolist() == [-1.0, 1.0, 1.0, 1.0] and affines[0].tolist() == [1.0] * 4
    beta = gates[0].coefficients(values[0])[1].double()
    torch.testing.assert_close(beta[[0, 2, 3]], torch.tensor([0.5 / math.sqrt(1.25), 0.5 / math.sqrt(1.25), 2 / math.sqrt(5)], dtype=torch.float64), rtol=1e-5, atol=0)
    assert 0 < beta[1] < 1e-6 and torch.isfinite(values[0]).all()                 # a zero gain becomes a (nearly) closed gate
    torch.testing.assert_close(final_affine.double(), 1 / s, rtol=1e-6, atol=0)
    torch.testing.assert_close(s[[0, 2, 3]], torch.tensor([1 / math.sqrt(1.25), 1 / math.sqrt(1.25), 1 / math.sqrt(5)], dtype=torch.float64), rtol=1e-5, atol=0)


def test_arguments_and_defaults():
    from megatron.training.arguments import add_megatron_arguments

    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    a = parser.parse_args([])
    assert a.residual_gate is False and a.residual_gate_temperature == 0.3 and a.residual_gate_convert_iteration is None
    a = parser.parse_args(["--residual-gate", "--residual-gate-convert-iteration", "4000"])
    assert a.residual_gate is True and a.residual_gate_convert_iteration == 4000
    assert TransformerConfig.__dataclass_fields__["residual_gate"].default is False
