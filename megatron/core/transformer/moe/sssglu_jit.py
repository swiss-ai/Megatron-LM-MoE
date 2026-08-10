"""Triton kernels for SSSGLU activation with optional per-row probability scaling.

SSSGLU is a gated linear unit whose gate is a shifted, scaled softsign. The shift is chosen so the
gate derivative has a y-intercept of 0.5 (``gate'(0) = 1 / (1 + |shift|)**2 = 0.5`` gives
``shift = sqrt(2) - 1``); the additive offset ``-softsign(-shift) = 1 - 1/sqrt(2)`` keeps
``gate(0) = 0``:
    shift    = sqrt(2) - 1 ~= 0.41421356
    gate(a)  = softsign(a - shift) + (1 - 1/sqrt(2))
             = (a - shift) / (1 + |a - shift|) + 0.29289322        (== sssglu_act in activations.py)
    SSSGLU   = gate(a) * b
Structured exactly like ``rlglu_jit.py`` / ``ssglu_jit.py`` / ``swiglu_jit.py`` (its GLU siblings)
-- the only difference is the gate; all gate math runs in fp32. Like RLGLU (and unlike SwiGLU/SSGLU,
``a * squash(a) * b``) the forward has no extra factor of ``a`` -- the gate is applied directly. Its
derivative is ``gate'(a) = 1 / (1 + |a - shift|)**2``. See also the (torch) fusion in
``megatron.core.fusions.fused_bias_sssglu``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sssglu_fwd_kernel(
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

    # gate(a) = softsign(a - shift) + (1 - 1/sqrt(2)); shift = sqrt(2) - 1; y = gate(a) * b
    u = a - 0.41421356237309515
    gate_a = 0.2928932188134524 + u / (1.0 + tl.abs(u))
    y = gate_a * b

    if HAS_PROBS:
        p = tl.load(p_ptr + rm, mask=rm < M, other=0.0).to(tl.float32)
        y = y * p[:, None]

    y_off = rm[:, None] * stride_ym + rd[None, :] * stride_yn
    tl.store(y_ptr + y_off, y, mask=mask)


@triton.jit
def _sssglu_bwd_kernel(
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

    # gate(a)  = (1 - 1/sqrt(2)) + (a - shift) / (1 + |a - shift|); shift = sqrt(2) - 1
    # gate'(a) = 1 / (1 + |a - shift|)**2
    u = a - 0.41421356237309515
    denom = 1.0 + tl.abs(u)
    gate_a = 0.2928932188134524 + u / denom
    d_gate = 1.0 / (denom * denom)

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


def sssglu_forward(
    input_tensor: torch.Tensor,
    probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton SSSGLU forward.

    y = gate(a) * b, gate(a) = softsign(a - (sqrt(2)-1)) + (1 - 1/sqrt(2)), optionally scaled by per-row `probs`.
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
    _sssglu_fwd_kernel[grid](
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


def sssglu_backward(
    grad_output: torch.Tensor,
    input_tensor: torch.Tensor,
    probs: torch.Tensor | None = None,
):
    """Triton SSSGLU backward.

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
    _sssglu_bwd_kernel[grid](
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
