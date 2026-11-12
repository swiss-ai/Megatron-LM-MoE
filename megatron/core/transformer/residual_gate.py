"""Gated residual stream (--residual-gate).

Every sublayer (attention, MLP/MoE) updates the stream as

    x <- sqrt(1 - beta^2) * x + beta * h,        beta = sigmoid(gate / temperature)   per channel

with h the sandwich post-norm output (its gain frozen at +-1), instead of x <- x + c * gamma * h. A
per-channel pre-affine lambda scales the stream before every norm that reads it (the sublayer
pre-norms and the final norm): the reader sees Norm(lambda * x).

A model trained with the plain residual converts exactly (convert_to_residual_gates): the gated
stream is the plain stream times a per-channel factor s_k, with

    s_0 = 1,    beta_k = s_k w_k / sqrt(1 + (s_k w_k)^2),    s_{k+1} = sqrt(1 - beta_k^2) s_k,
    lambda_k = 1 / s_k,                                       w_k = c * |gamma_k|

so that every sublayer reads and writes the same values as before (the sign of gamma_k stays in the
frozen post-norm gain).
"""

import torch

_BETA_MIN = 1e-7          # a zero post-norm gain cannot be a sigmoid gate; this is the smallest write kept
_BETA_MAX = 1.0 - 1e-6


class ResidualGate(torch.nn.Module):
    """x <- alpha * residual + beta * hidden, beta = sigmoid(gate / T), alpha = sqrt(1 - beta^2)."""

    def __init__(self, hidden_size: int, temperature: float = 0.3, eps: float = 1e-3):
        super().__init__()
        self.gate = torch.nn.Parameter(torch.zeros(hidden_size))
        self.temperature = float(temperature)
        self.eps = float(eps)

    def coefficients(self, gate: torch.Tensor):
        """(alpha, beta) in fp32 for a gate tensor, as used by forward."""
        w = gate.float() / self.temperature
        beta = torch.sigmoid(w)
        # sigmoid(-w) * (1 + beta) = (1 - beta) * (1 + beta) = 1 - beta^2, without cancellation
        alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=self.eps))
        return alpha, beta

    def forward(self, residual: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha, beta = self.coefficients(self.gate)
        return (alpha * residual + beta * hidden_states).to(residual.dtype)

    def extra_repr(self):
        return f"{tuple(self.gate.shape)}, temperature={self.temperature}"


class PreAffine(torch.nn.Module):
    """Per-channel scale applied to the stream before a norm reads it: Norm(weight * x)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() * self.weight.float()).to(x.dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}"


@torch.no_grad()
def residual_gate_plan(write_gains, gates, dtype=torch.float32):
    """Gate and pre-affine values that reproduce a plain residual stream.

    Args:
        write_gains: per sublayer, in stream order, the signed per-channel coefficient of the
            normalized sublayer output in the plain update x <- x + write_gain * h.
        gates: the ResidualGate of each sublayer (its temperature / eps define beta and alpha).
        dtype: dtype the parameters are stored in; the plan is built from the rounded values, so
            rounding does not accumulate along the stream.

    Returns (gate_values, signs, affines, final_affine, scale): lists per sublayer of the gate
    parameter, the sign kept in the post-norm gain and the pre-affine of the sublayer's reader; the
    pre-affine of the final norm; and the final stream scale s_K (fp64).
    """
    s = torch.ones_like(write_gains[0], dtype=torch.float64)
    gate_values, signs, affines = [], [], []
    for gain, gate in zip(write_gains, gates):
        gain = gain.double()
        affines.append((1.0 / s).to(dtype))
        signs.append(torch.where(gain < 0, -1.0, 1.0).to(dtype))
        sw = s * gain.abs()
        beta = (sw / torch.sqrt(1.0 + sw * sw)).clamp(_BETA_MIN, _BETA_MAX)
        value = (gate.temperature * torch.logit(beta)).to(dtype)
        gate_values.append(value)
        alpha, _ = gate.coefficients(value)          # what the forward pass will really use
        s = alpha.double() * s
    return gate_values, signs, affines, (1.0 / s).to(dtype), s


def _set(param: torch.nn.Parameter, value: torch.Tensor) -> None:
    """Write a value into a model parameter and, on the rank that owns it, its fp32 main copy."""
    param.data.copy_(value.to(device=param.device, dtype=param.dtype))
    main = getattr(param, "main_param", None)
    if main is not None:
        main.data.copy_(param.data.to(device=main.device, dtype=main.dtype))


def _decoder(model):
    module = model
    while not hasattr(module, "decoder") and hasattr(module, "module"):
        module = module.module
    return module.decoder


@torch.no_grad()
def convert_to_residual_gates(model_chunks) -> dict:
    """One-shot, function-preserving switch of a plain-residual checkpoint to the gated residual,
    run right after the checkpoint is loaded (--residual-gate-convert-iteration).

    Reads every sandwich post-norm gain (as stored in the model, the values the forward pass uses),
    sets the residual gates and pre-affines from residual_gate_plan and leaves +-1 in the (frozen)
    post-norm gains. Every rank holds the full model, so each rank computes the same values; the
    rank that owns a new parameter in the optimizer also gets its fp32 main copy set.
    """
    summary = {"sublayers": 0, "negative_gains": 0, "gates_below_1e-3": 0, "channels": 0}
    for model in model_chunks if isinstance(model_chunks, (list, tuple)) else [model_chunks]:
        decoder = _decoder(model)
        norms, gates, affines = [], [], []
        for layer in decoder.layers:
            assert getattr(layer, "residual_gate", False), "convert_to_residual_gates needs --residual-gate"
            for affine, norm, gate in (
                (layer.attn_pre_affine, layer.post_self_attn_layernorm, layer.post_attn_residual_gate),
                (layer.mlp_pre_affine, layer.post_mlp_layernorm, layer.post_mlp_residual_gate),
            ):
                assert not norm.weight.requires_grad, "post-norm gains must be frozen under --residual-gate"
                norms.append((norm, layer.residual_gate_plain_scale)); gates.append(gate); affines.append(affine)
        write_gains = [norm.weight.detach().double().cpu() * scale for norm, scale in norms]
        dtype = gates[0].gate.dtype
        gate_values, signs, affine_values, final_affine, s = residual_gate_plan(write_gains, gates, dtype)
        for (norm, _), gate, affine, g, sign, lam in zip(norms, gates, affines, gate_values, signs, affine_values):
            _set(gate.gate, g)
            _set(affine.weight, lam)
            norm.weight.data.copy_(sign.to(device=norm.weight.device, dtype=norm.weight.dtype))
            summary["sublayers"] += 1
            summary["negative_gains"] += int((sign < 0).sum())
            summary["gates_below_1e-3"] += int((gate.coefficients(g)[1] < 1e-3).sum())
            summary["channels"] += g.numel()
        _set(decoder.final_pre_affine.weight, final_affine)
        summary["final_stream_scale_min"] = float(s.min())
        summary["final_stream_scale_median"] = float(s.median())
        summary["final_stream_scale_max"] = float(s.max())
    return summary
