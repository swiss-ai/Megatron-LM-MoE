"""Triton kernels for SiTU activation with optional per-row probability scaling.

SiTU is a gated linear unit whose gate is a sigmoid-times-tanh:
    gate(a) = sigmoid(a) * tanh(a)   (== situ_act in activations.py)
    SiTU    = gate(a) * b
Structured exactly like ``sssglu_jit.py`` / ``rlglu_jit.py`` / ``swiglu_jit.py`` (its GLU siblings)
-- the only difference is the gate; all gate math runs in fp32. Like RLGLU/SSSGLU (and unlike
SwiGLU/SSGLU, ``a * squash(a) * b``) the forward has no extra factor of ``a`` -- the gate is applied
directly. Its derivative is (with s = sigmoid(a), t = tanh(a))
    gate'(a) = s*(1-s)*t + s*(1-t**2).
``tanh`` is evaluated as ``2*sigmoid(2a) - 1`` so the kernel only needs ``tl.sigmoid`` (portable
across Triton versions). See also the (torch) fusion in ``megatron.core.fusions.fused_bias_situ``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _situ_fwd_kernel(
    x_ptr, p_ptr, y_ptr,
    M, D,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    HAS_PROBS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask = (rm[:, None] < M) & (rd[None, :] < D)
    a_off = rm[:, None] * stride_xm + rd[None, :] * stride_xn
    b_off = a_off + D * stride_xn

    a = tl.load(x_ptr + a_off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + b_off, mask=mask, other=0.0).to(tl.float32)

    # gate(a) = sigmoid(a) * tanh(a); tanh(a) = 2*sigmoid(2a) - 1; y = gate(a) * b
    s = tl.sigmoid(a)
    t = 2.0 * tl.sigmoid(2.0 * a) - 1.0
    gate_a = s * t
    y = gate_a * b

    if HAS_PROBS:
        p = tl.load(p_ptr + rm, mask=rm < M, other=0.0).to(tl.float32)
        y = y * p[:, None]

    y_off = rm[:, None] * stride_ym + rd[None, :] * stride_yn
    tl.store(y_ptr + y_off, y, mask=mask)


@triton.jit
def _situ_bwd_kernel(
    x_ptr, p_ptr, gy_ptr, gx_ptr, gp_ptr,
    M, D,
    stride_xm, stride_xn,
    stride_gym, stride_gyn,
    stride_gxm, stride_gxn,
    HAS_PROBS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask = (rm[:, None] < M) & (rd[None, :] < D)

    a_off = rm[:, None] * stride_xm + rd[None, :] * stride_xn
    b_off = a_off + D * stride_xn
    gy_off = rm[:, None] * stride_gym + rd[None, :] * stride_gyn

    a = tl.load(x_ptr + a_off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + b_off, mask=mask, other=0.0).to(tl.float32)
    gy = tl.load(gy_ptr + gy_off, mask=mask, other=0.0).to(tl.float32)

    # gate(a)  = sigmoid(a) * tanh(a)   (s = sigmoid(a), t = tanh(a))
    # gate'(a) = s*(1-s)*t + s*(1-t**2)
    s = tl.sigmoid(a)
    t = 2.0 * tl.sigmoid(2.0 * a) - 1.0
    gate_a = s * t
    d_gate = s * (1.0 - s) * t + s * (1.0 - t * t)

    if HAS_PROBS:
        p = tl.load(p_ptr + rm, mask=rm < M, other=0.0).to(tl.float32)
        gy_scaled = gy * p[:, None]
        # partial grad_probs over this D-block: sum_d (gate(a) * b * gy)
        partial = tl.sum(tl.where(mask, gate_a * b * gy, 0.0), axis=1)
        tl.atomic_add(gp_ptr + rm, partial, mask=rm < M)
    else:
        gy_scaled = gy

    grad_a = gy_scaled * d_gate * b
    grad_b = gy_scaled * gate_a

    ga_off = rm[:, None] * stride_gxm + rd[None, :] * stride_gxn
    gb_off = ga_off + D * stride_gxn
    tl.store(gx_ptr + ga_off, grad_a, mask=mask)
    tl.store(gx_ptr + gb_off, grad_b, mask=mask)


def _block_d(D: int) -> int:
    if D >= 256:
        return 256
    return triton.next_power_of_2(max(D, 16))


def situ_forward(
    input_tensor: torch.Tensor,
    probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton SiTU forward.

    y = gate(a) * b, gate(a) = sigmoid(a) * tanh(a), optionally scaled by per-row `probs`.
    """
    assert input_tensor.shape[-1] % 2 == 0, "last dim must be 2*D"
    orig_shape = input_tensor.shape
    x = input_tensor.contiguous().view(-1, orig_shape[-1])
    M, two_d = x.shape
    D = two_d // 2

    y = torch.empty((M, D), device=x.device, dtype=x.dtype)

    p_flat = None
    if probs is not None:
        p_flat = probs.contiguous().view(-1).to(torch.float32)
        assert p_flat.numel() == M, (
            f"probs has {p_flat.numel()} rows, expected {M}"
        )

    BLOCK_M, BLOCK_D = 16, _block_d(D)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_D))
    _situ_fwd_kernel[grid](
        x,
        p_flat if p_flat is not None else x,  # placeholder ptr when unused
        y,
        M, D,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        HAS_PROBS=(p_flat is not None),
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
    )
    return y.view(orig_shape[:-1] + (D,))


def situ_backward(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    probs: torch.Tensor | None = None,
):
    """Triton SiTU backward.

    Returns grad_input matching input_tensor.shape. If probs is given, also
    returns grad_probs with shape probs.shape[:-1] (last dim already reduced).
    """
    orig_shape = input_tensor.shape
    x = input_tensor.contiguous().view(-1, orig_shape[-1])
    M, two_d = x.shape
    D = two_d // 2
    gy = grad_output.contiguous().view(M, D)
    gx = torch.empty_like(x)

    p_flat = None
    gp = None
    if probs is not None:
        p_flat = probs.contiguous().view(-1).to(torch.float32)
        assert p_flat.numel() == M
        gp = torch.zeros(M, device=x.device, dtype=torch.float32)

    BLOCK_M, BLOCK_D = 16, _block_d(D)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_D))
    _situ_bwd_kernel[grid](
        x,
        p_flat if p_flat is not None else x,
        gy,
        gx,
        gp if gp is not None else x,
        M, D,
        x.stride(0), x.stride(1),
        gy.stride(0), gy.stride(1),
        gx.stride(0), gx.stride(1),
        HAS_PROBS=(p_flat is not None),
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
    )

    grad_input = gx.view(orig_shape)
    if probs is None:
        return grad_input
    grad_probs = gp.to(probs.dtype).view(probs.shape[:-1])
    return grad_input, grad_probs
