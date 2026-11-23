# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.
"""KDA recurrence on the cuDNN sm90 engine, recomputing the L2-norm and decay gate in the backward."""
import math
from typing import Optional

import torch

try:
    import cudnn.linear_attention.ops  # noqa: F401
    from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
    from fla.ops.kda.gate import kda_gate_bwd, kda_gate_fwd

    HAVE_CUDNN_KDA = True
except Exception:
    HAVE_CUDNN_KDA = False


def _activations(q, k, alpha, A_log, dt_bias, l2norm, lower_bound):
    qn, rstd_q = l2norm_fwd(q) if l2norm else (q, None)
    kn, rstd_k = l2norm_fwd(k) if l2norm else (k, None)
    g = kda_gate_fwd(alpha, A_log, dt_bias=dt_bias, lower_bound=lower_bound)
    return qn, kn, rstd_q, rstd_k, g


class _CudnnKDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, alpha, beta, A_log, dt_bias, cu_seqlens, initial_state,
        output_final_state, l2norm, lower_bound,
    ):
        qn, kn, _, _, g = _activations(q, k, alpha, A_log, dt_bias, l2norm, lower_bound)
        scale = 1.0 / math.sqrt(q.shape[-1])
        o, final_state, _ = torch.ops.cudnn.kimi_delta_attention_fwd(
            qn, kn, v, g, beta, cu_seqlens, scale,
            initial_state=initial_state, output_final_state=output_final_state,
        )
        ctx.save_for_backward(q, k, v, alpha, beta, A_log, dt_bias, cu_seqlens, initial_state)
        ctx.l2norm, ctx.lower_bound, ctx.scale = l2norm, lower_bound, scale
        ctx.set_materialize_grads(False)
        return o, final_state

    @staticmethod
    def backward(ctx, do, d_final_state):
        q, k, v, alpha, beta, A_log, dt_bias, cu_seqlens, initial_state = ctx.saved_tensors
        qn, kn, rstd_q, rstd_k, g = _activations(
            q, k, alpha, A_log, dt_bias, ctx.l2norm, ctx.lower_bound
        )
        if do is None:
            do = torch.zeros_like(v)
        if d_final_state is not None and d_final_state.numel() == 0:
            d_final_state = None
        dq, dk, dv, dg, dbeta, d_initial_state, _, _ = torch.ops.cudnn.kimi_delta_attention_bwd(
            do.contiguous(), qn, kn, v, g, beta, cu_seqlens, ctx.scale,
            initial_state=initial_state, d_final_state=d_final_state,
        )
        if ctx.l2norm:
            dq = l2norm_bwd(qn, rstd_q, dq)
            dk = l2norm_bwd(kn, rstd_k, dk)
        dalpha, dA_log, ddt_bias = kda_gate_bwd(
            alpha, A_log, dt_bias=dt_bias, dyg=dg, lower_bound=ctx.lower_bound
        )
        if initial_state is None:
            d_initial_state = None
        return (
            dq, dk, dv, dalpha, dbeta.to(beta.dtype), dA_log, ddt_bias, None, d_initial_state,
            None, None, None,
        )


def cudnn_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    l2norm: bool = True,
    lower_bound: float = -5.0,
    token_multiple: int = 512,
    min_num_seqs: int = 32,
):
    """KDA over packed sequences on cuDNN.

    q, k, v, alpha: [T, H, 128]; beta: [T, H] after the sigmoid; cu_seqlens: [N + 1] int32;
    initial_state: [N, H, V, K]; lower_bound: g_min of the safe decay gate. The token count is padded to a multiple of `token_multiple`
    and the sequence count to a power of two >= `min_num_seqs` with empty sequences, so that
    cuDNN reuses a few compiled plans instead of building one per batch.
    """
    num_tokens, num_seqs = q.shape[0], cu_seqlens.numel() - 1
    pad = -num_tokens % token_multiple
    if pad:
        q, k, v, alpha, beta = (
            torch.cat([x, x.new_zeros((pad,) + tuple(x.shape[1:]))]) for x in (q, k, v, alpha, beta)
        )
        cu_seqlens = torch.cat([cu_seqlens, cu_seqlens.new_full((1,), num_tokens + pad)])
    padded_seqs = cu_seqlens.numel() - 1
    target = max(min_num_seqs, 1 << (padded_seqs - 1).bit_length())
    if target > padded_seqs:
        cu_seqlens = torch.cat([cu_seqlens, cu_seqlens[-1].expand(target - padded_seqs)])
    if initial_state is not None and target > num_seqs:
        initial_state = torch.cat(
            [initial_state, initial_state.new_zeros((target - num_seqs,) + tuple(initial_state.shape[1:]))]
        )
    o, final_state = _CudnnKDAFunction.apply(
        q.contiguous(), k.contiguous(), v.contiguous(), alpha.contiguous(), beta.contiguous(),
        A_log, dt_bias, cu_seqlens, initial_state, output_final_state, l2norm, lower_bound,
    )
    final_state = final_state[:num_seqs] if output_final_state else None
    return o[:num_tokens], final_state
