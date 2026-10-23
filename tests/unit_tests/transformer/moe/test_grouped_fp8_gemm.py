"""Tests for grouped_fp8_gemm_nt_contiguous following the MoE layer computation flow.

This function is a single-kernel-launch replacement for multi_stream_fp8_gemm_nt.
Instead of looping over experts with separate GEMM launches, it dispatches one
grouped GEMM kernel that computes Y[i] = A[i] @ B[expert_of(i)].T for all experts.

Key differences from multi_stream_fp8_gemm_nt:
  - Weights are stacked into a single [G, N, K] tensor (not a Python list)
  - Weight scales are stacked into [G, N//128, K//128] (not a Python list)
  - tokens_per_expert is passed as cumulative-sum (psum) int32 [G]
  - Each M_i must be a multiple of get_m_alignment_for_contiguous_layout() (128 on SM90)
  - Uses a single CUDA stream (no multi-stream parallelism)

MoE computation flow:
  Forward:
    fc1:  hidden_states [M, K] @ w1.T [G, 2H, K]  ->  fc1_out [M, 2H]
    SwiGLU activation on fc1_out
    fc2:  swiglu_out [M, H] @ w2.T [G, K, H]      ->  output  [M, K]
  Backward (data gradients):
    grad_s:  grad_y [M, K] @ w2 [G, K, H]      (transposed stacked weights)  -> grad_s  [M, H]
             then SwiGLU backward -> grad_a [M, 2H]
    grad_x:  grad_a [M, 2H] @ w1 [G, 2H, K]    (transposed stacked weights)  -> grad_x  [M, K]
  Backward (weight gradients — per-expert fp8_gemm_nt, same as multi_stream tests):
    grad_w2[i]:  grad_y[i].T @ s[i]             [K, M_i] @ [M_i, H] -> [K, H]
    grad_w1[i]:  grad_a[i].T @ hs[i]            [2H, M_i] @ [M_i, K] -> [2H, K]

Shape notation:
  M       = total tokens across all experts
  M_i     = tokens for expert i  (must be multiple of 128 for grouped GEMM)
  K       = hidden_size  (multiple of 128)
  H       = ffn_hidden_size  (multiple of 128)
  2H      = fc1 output dim (gated linear unit)
  G       = number of experts
  fp8     = torch.float8_e4m3fn
  bf16    = torch.bfloat16
  f32     = torch.float32
"""

import itertools
import os

import pytest
import torch

try:
    import deep_gemm

    HAVE_DEEP_GEMM = deep_gemm is not None
except ImportError:
    HAVE_DEEP_GEMM = False

from megatron.core.transformer.moe.fp8_utils import (
    m_grouped_fp8_gemm_nt_contiguous,
    k_grouped_fp8_gemm_nt_contiguous,
)
from megatron.core.transformer.moe.fp8_jit import (
    per_token_cast_to_fp8,
    per_token_dequant_from_fp8,
    per_channel_cast_to_fp8,
    pack_to_kmajor,
)
from megatron.core.transformer.moe.experts_util import MergedSwiGLU


def make_kgrouped_prefixes(tokens_per_expert, device="cuda"):
    """Exclusive-prefix-sum int32 tensor expected by pack_to_kmajor.

    Args:  tokens_per_expert  list[int] | [G] int64
    Returns:
           prefixes  [G] int32  (cumulative starts: [0, t0, t0+t1, ...])
    """
    if torch.is_tensor(tokens_per_expert):
        tpe_list = tokens_per_expert.tolist()
    else:
        tpe_list = list(tokens_per_expert)
    return torch.tensor(
        [0, *itertools.accumulate(tpe_list[:-1])],
        device=device, dtype=torch.int32,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_no_deep_gemm():
    if not HAVE_DEEP_GEMM:
        pytest.skip("deep_gemm not available")


def _skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count())


def quantize_activation(x_bf16):
    """Quantize a 2-D activation to FP8 with per-token scales (recipe_a=(1,128)).

    Args:  x_bf16  [M, K]   bf16
    Returns:
           x_fp8   [M, K]               fp8_e4m3fn
           scales  [M, K // 128]         f32
    """
    return per_token_cast_to_fp8(
        x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
    )


def quantize_weight(w_bf16):
    """Quantize a 2-D weight to FP8 with per-block scales (recipe_b=(128,128)).

    Args:  w_bf16  [N, K]   bf16   (N must be multiple of 128)
    Returns:
           w_fp8   [N, K]               fp8_e4m3fn
           scales  [N // 128, K // 128]  f32
    """
    return deep_gemm.per_block_cast_to_fp8(w_bf16, use_ue8m0=False, gran_k=128)


def quantize_and_stack_weights(w_bf16_list):
    """Quantize per-expert weights and stack into grouped-GEMM layout.

    Args:  w_bf16_list[i]  [N, K]  bf16   (N must be multiple of 128)
    Returns:
           stacked_fp8     [G, N, K]            fp8
           stacked_scales  [G, N // 128, K // 128]  f32
    """
    fp8_list, scales_list = [], []
    for w in w_bf16_list:
        fw, fs = quantize_weight(w)
        fp8_list.append(fw)
        scales_list.append(fs)
    return torch.stack(fp8_list), torch.stack(scales_list)


def make_psum(tokens_per_expert, device="cuda"):
    """Build cumulative-sum layout tensor (int32) for grouped GEMM.

    Args:  tokens_per_expert  [G]   int64  (raw counts)
    Returns:
           psum               [G]   int32  (cumulative sum on device)
    """
    return torch.cumsum(tokens_per_expert, dim=0).to(torch.int32).to(device)


def bf16_ref_gemm_nt(a_bf16, b_bf16):
    """Reference: D = A @ B.T  (bf16 -> fp32 -> bf16).

    Args:  a_bf16  [M, K]   bf16
           b_bf16  [N, K]   bf16
    Returns:
           result  [M, N]   bf16
    """
    return (
        a_bf16.to(torch.float32) @ b_bf16.to(torch.float32).t()
    ).to(torch.bfloat16)


def rel_diff(actual, reference):
    """Relative L2 difference: ||actual - ref|| / ||ref||."""
    return (
        torch.norm(actual.float() - reference.float()).item()
        / torch.norm(reference.float()).item()
    )


def bf16_ref_grouped_gemm(a_bf16, b_bf16_list, tokens_per_expert):
    """bf16 reference for grouped GEMM: D[i] = A[i] @ B[i].T per expert.

    Args:  a_bf16           [M, K]       bf16  (concatenated across experts)
           b_bf16_list[i]   [N, K]       bf16  (weight for expert i)
           tokens_per_expert  [G]        int64
    Returns:
           result           [M, N]       bf16
    """
    parts = []
    for i, t in enumerate(tokens_per_expert.tolist()):
        start = sum(tokens_per_expert[:i].tolist())
        end = start + t
        parts.append(bf16_ref_gemm_nt(a_bf16[start:end], b_bf16_list[i]))
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Unit tests for grouped_fp8_gemm_nt_contiguous
# ---------------------------------------------------------------------------


class TestGroupedFP8GemmNTContiguous:
    """Direct unit tests for grouped_fp8_gemm_nt_contiguous."""

    def setup_method(self):
        _skip_if_no_deep_gemm()
        _skip_if_no_gpu()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    @pytest.mark.parametrize("num_experts", [1, 4, 8])
    def test_forward_correctness(self, num_experts):
        """Compare grouped FP8 GEMM against bf16 reference."""
        K = 256   # multiple of 128
        N = 256   # multiple of 128

        # tokens_per_expert:  [G]            int64   (each M_i = 128, satisfies alignment)
        tokens_per_expert = torch.tensor([128] * num_experts, dtype=torch.int64)
        # M:                  scalar
        M = tokens_per_expert.sum().item()

        # a_bf16:             [M, K]          bf16   (activation)
        a_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        # b_bf16_list[i]:     [N, K]          bf16   (weight for expert i)
        b_bf16_list = [
            torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            for _ in range(num_experts)
        ]

        # Quantize activation (recipe_a = (1, 128))
        # fp8_a:              [M, K]          fp8
        # fp8_a_scales:       [M, K // 128]   f32
        fp8_a, fp8_a_scales = quantize_activation(a_bf16)

        # Quantize + stack weights (recipe_b = (128, 128))
        # stacked_b:          [G, N, K]            fp8
        # stacked_b_scales:   [G, N // 128, K // 128]  f32
        stacked_b, stacked_b_scales = quantize_and_stack_weights(b_bf16_list)

        # psum:               [G]             int32   (cumulative sum)
        psum = make_psum(tokens_per_expert)
        stream = torch.cuda.Stream()

        # grouped FP8 GEMM — single kernel launch for all experts
        # output:             [M, N]          bf16
        output = m_grouped_fp8_gemm_nt_contiguous(
            psum,
            (fp8_a, fp8_a_scales),
            (stacked_b, stacked_b_scales),
            stream,
        )
        torch.cuda.synchronize()

        # bf16 reference:  per-expert A[i] @ B[i].T, then concat
        # ref:                 [M, N]          bf16
        ref = bf16_ref_grouped_gemm(a_bf16, b_bf16_list, tokens_per_expert)

        diff = rel_diff(output, ref)
        assert diff < 0.05, f"rel_diff={diff:.4f}"

    def test_forward_preallocated_output(self):
        """Verify pre-allocated output tensor is written in-place."""
        num_experts = 4
        K, N = 256, 256
        # tokens_per_expert:  [4]             int64
        tokens_per_expert = torch.tensor([128, 128, 256, 128], dtype=torch.int64)
        # M:                  scalar = 640
        M = tokens_per_expert.sum().item()

        # a_bf16:             [M, K]          bf16
        a_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        # b_bf16_list[i]:     [N, K]          bf16
        b_bf16_list = [
            torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            for _ in range(num_experts)
        ]

        # fp8_a:              [M, K]          fp8
        # fp8_a_scales:       [M, K // 128]   f32
        fp8_a, fp8_a_scales = quantize_activation(a_bf16)
        # stacked_b:          [G, N, K]       fp8
        # stacked_b_scales:   [G, N//128, K//128]  f32
        stacked_b, stacked_b_scales = quantize_and_stack_weights(b_bf16_list)
        # psum:               [G]             int32
        psum = make_psum(tokens_per_expert)

        # output_buf:         [M, N]          bf16   (pre-allocated)
        output_buf = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()

        # result:             [M, N]          bf16   (same storage as output_buf)
        result = m_grouped_fp8_gemm_nt_contiguous(
            psum,
            (fp8_a, fp8_a_scales),
            (stacked_b, stacked_b_scales),
            stream,
            output=output_buf,
        )
        torch.cuda.synchronize()

        assert result.data_ptr() == output_buf.data_ptr(), "output was not written in-place"

        ref = bf16_ref_grouped_gemm(a_bf16, b_bf16_list, tokens_per_expert)
        diff = rel_diff(result, ref)
        assert diff < 0.05, f"rel_diff={diff:.4f}"

    def test_forward_uneven_tokens(self):
        """Test with varying token counts per expert (all multiples of 128)."""
        num_experts = 4
        K, N = 256, 256
        # tokens_per_expert:  [4]             int64   (M_i = 256, 128, 128, 256 — all multiples of 128)
        tokens_per_expert = torch.tensor([256, 128, 128, 256], dtype=torch.int64)
        # M:                  scalar = 768
        M = tokens_per_expert.sum().item()

        # a_bf16:             [M, K]          bf16
        a_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        # b_bf16_list[i]:     [N, K]          bf16
        b_bf16_list = [
            torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            for _ in range(num_experts)
        ]

        # fp8_a:              [M, K]          fp8
        # fp8_a_scales:       [M, K // 128]   f32
        fp8_a, fp8_a_scales = quantize_activation(a_bf16)
        # stacked_b:          [G, N, K]       fp8
        # stacked_b_scales:   [G, N//128, K//128]  f32
        stacked_b, stacked_b_scales = quantize_and_stack_weights(b_bf16_list)
        # psum:               [G]             int32
        psum = make_psum(tokens_per_expert)
        stream = torch.cuda.Stream()

        # output:             [M, N]          bf16
        output = m_grouped_fp8_gemm_nt_contiguous(
            psum,
            (fp8_a, fp8_a_scales),
            (stacked_b, stacked_b_scales),
            stream,
        )
        torch.cuda.synchronize()

        ref = bf16_ref_grouped_gemm(a_bf16, b_bf16_list, tokens_per_expert)
        diff = rel_diff(output, ref)
        assert diff < 0.05, f"rel_diff={diff:.4f}"

    def test_matches_multi_stream(self):
        """Verify grouped and multi-stream produce numerically identical results."""
        num_experts = 4
        K, N = 256, 256
        # tokens_per_expert:  [4]             int64   (all multiples of 128)
        tokens_per_expert = torch.tensor([128, 256, 128, 128], dtype=torch.int64)
        # M:                  scalar = 640
        M = tokens_per_expert.sum().item()

        # a_bf16:             [M, K]          bf16
        a_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        # b_bf16_list[i]:     [N, K]          bf16
        b_bf16_list = [
            torch.randn(N, K, device="cuda", dtype=torch.bfloat16) for _ in range(num_experts)
        ]

        # Shared quantization
        # fp8_a:              [M, K]          fp8
        # fp8_a_scales:       [M, K // 128]   f32
        fp8_a, fp8_a_scales = quantize_activation(a_bf16)
        fp8_b_list, fp8_b_scales_list = [], []
        for w in b_bf16_list:
            fw, fs = quantize_weight(w)
            fp8_b_list.append(fw)
            fp8_b_scales_list.append(fs)

        # Multi-stream reference using the current per-expert DeepGEMM API.
        streams = [torch.cuda.Stream() for _ in range(2)]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        sizes = tokens_per_expert.tolist()
        a_parts = torch.split(fp8_a, sizes, dim=0)
        scale_parts = torch.split(fp8_a_scales, sizes, dim=0)
        outputs = []
        for i, (a_part, scale_part) in enumerate(zip(a_parts, scale_parts)):
            with torch.cuda.stream(streams[i % len(streams)]):
                expert_output = torch.empty(sizes[i], N, device="cuda", dtype=torch.bfloat16)
                deep_gemm.fp8_gemm_nt(
                    (a_part.contiguous(), scale_part.contiguous()),
                    (fp8_b_list[i].contiguous(), fp8_b_scales_list[i].contiguous()),
                    expert_output,
                    c=None,
                    recipe=None,
                    recipe_a=(1, 128),
                    recipe_b=(128, 128),
                    disable_ue8m0_cast=True,
                )
                outputs.append(expert_output)
        torch.cuda.synchronize()
        # ms_output:          [M, N]          bf16, in the original expert order
        ms_output = torch.cat(outputs, dim=0)

        # Grouped result
        # stacked_b:          [G, N, K]       fp8
        # stacked_b_scales:   [G, N//128, K//128]  f32
        stacked_b = torch.stack(fp8_b_list)
        stacked_b_scales = torch.stack(fp8_b_scales_list)
        # psum:               [G]             int32
        psum = make_psum(tokens_per_expert)
        stream = torch.cuda.Stream()

        # grp_output:         [M, N]          bf16
        grp_output = m_grouped_fp8_gemm_nt_contiguous(
            psum, (fp8_a, fp8_a_scales), (stacked_b, stacked_b_scales), stream
        )
        torch.cuda.synchronize()

        diff = rel_diff(grp_output, ms_output)
        assert diff < 1e-6, f"grouped vs multi-stream rel_diff={diff:.6f}"


# ---------------------------------------------------------------------------
# MoE forward flow tests
# ---------------------------------------------------------------------------


class TestMoEForwardFlowGrouped:
    """Test grouped_fp8_gemm_nt_contiguous in the MoE forward path."""

    def setup_method(self):
        _skip_if_no_deep_gemm()
        _skip_if_no_gpu()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    @pytest.mark.parametrize("num_experts", [4, 8])
    def test_moe_fc1_forward(self, num_experts):
        """Forward fc1: hidden_states [M,K] @ w1.T [G,2H,K] -> fc1_out [M,2H].

        Mirrors call_forward_a where the grouped kernel is used for fc1.
        """
        K = 256
        H = 256
        fc1_out_dim = 2 * H

        # tokens_per_expert:  [G]             int64   (each 128, multiple of 128)
        tokens_per_expert = torch.tensor([128] * num_experts, dtype=torch.int64)
        # M:                  scalar
        M = tokens_per_expert.sum().item()

        # hidden_states:      [M, K]          bf16
        hidden_states = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        # w1_list[i]:         [2H, K]         bf16
        w1_list = [
            torch.randn(fc1_out_dim, K, device="cuda", dtype=torch.bfloat16)
            for _ in range(num_experts)
        ]

        # Quantize hidden states  (recipe_a = (1, 128))
        # fp8_hs:             [M, K]          fp8
        # fp8_hs_scales:      [M, K // 128]   f32
        fp8_hs, fp8_hs_scales = quantize_activation(hidden_states)

        # Quantize + stack fc1 weights  (recipe_b = (128, 128))
        # stacked_w1:         [G, 2H, K]           fp8
        # stacked_w1_scales:  [G, 2H//128, K//128] f32
        stacked_w1, stacked_w1_scales = quantize_and_stack_weights(w1_list)

        # psum:               [G]             int32
        psum = make_psum(tokens_per_expert)
        stream = torch.cuda.Stream()

        # Grouped FP8 GEMM:  hidden_states[i][M_i, K] @ w1[i].T[2H, K] => fc1_out[i][M_i, 2H]
        # fc1_out:            [M, 2H]         bf16
        fc1_out = m_grouped_fp8_gemm_nt_contiguous(
            psum,
            (fp8_hs, fp8_hs_scales),
            (stacked_w1, stacked_w1_scales),
            stream,
        )
        torch.cuda.synchronize()

        # bf16 reference
        ref = bf16_ref_grouped_gemm(hidden_states, w1_list, tokens_per_expert)

        assert fc1_out.shape == (M, fc1_out_dim)
        diff = rel_diff(fc1_out, ref)
        assert diff < 0.05, f"fc1 forward rel_diff={diff:.4f}"

    @pytest.mark.parametrize("num_experts", [4, 8])
    def test_moe_fc2_forward(self, num_experts):
        """Forward fc2: swiglu_out [M,H] @ w2.T [G,K,H] -> output [M,K]."""
        K = 256
        H = 256

        # tokens_per_expert:  [G]             int64
        tokens_per_expert = torch.tensor([128] * num_experts, dtype=torch.int64)
        # M:                  scalar
        M = tokens_per_expert.sum().item()

        # swiglu_out:         [M, H]          bf16    (simulated SwiGLU output)
        swiglu_out = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
        # w2_list[i]:         [K, H]          bf16
        w2_list = [
            torch.randn(K, H, device="cuda", dtype=torch.bfloat16)
            for _ in range(num_experts)
        ]

        # Quantize SwiGLU output  (recipe_a = (1, 128))
        # fp8_s:              [M, H]          fp8
        # fp8_s_scales:       [M, H // 128]   f32
        fp8_s, fp8_s_scales = quantize_activation(swiglu_out)

        # Quantize + stack fc2 weights  (recipe_b = (128, 128))
        # stacked_w2:         [G, K, H]              fp8
        # stacked_w2_scales:  [G, K//128, H//128]    f32
        stacked_w2, stacked_w2_scales = quantize_and_stack_weights(w2_list)

        # psum:               [G]             int32
        psum = make_psum(tokens_per_expert)
        stream = torch.cuda.Stream()

        # Grouped FP8 GEMM:  swiglu_out[i][M_i, H] @ w2[i].T[K, H] => output[i][M_i, K]
        # output:             [M, K]          bf16
        output = m_grouped_fp8_gemm_nt_contiguous(
            psum,
            (fp8_s, fp8_s_scales),
            (stacked_w2, stacked_w2_scales),
            stream,
        )
        torch.cuda.synchronize()

        ref = bf16_ref_grouped_gemm(swiglu_out, w2_list, tokens_per_expert)

        assert output.shape == (M, K)
        diff = rel_diff(output, ref)
        assert diff < 0.05, f"fc2 forward rel_diff={diff:.4f}"

    @pytest.mark.parametrize("num_experts", [4])
    def test_moe_end_to_end_forward(self, num_experts):
        """End-to-end MoE forward: fc1 -> SwiGLU -> fc2, both using grouped GEMM."""
        K = 256
        H = 256
        fc1_out_dim = 2 * H

        # tokens_per_expert:  [G]             int64
        tokens_per_expert = torch.tensor([128] * num_experts, dtype=torch.int64)
        # M:                  scalar
        M = tokens_per_expert.sum().item()
        # permuted_probs:     [M, 1]          bf16
        permuted_probs = torch.rand(M, 1, device="cuda", dtype=torch.bfloat16)

        # hidden_states:      [M, K]          bf16
        hidden_states = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        # w1_list[i]:         [2H, K]         bf16
        w1_list = [
            torch.randn(fc1_out_dim, K, device="cuda", dtype=torch.bfloat16)
            for _ in range(num_experts)
        ]
        # w2_list[i]:         [K, H]          bf16
        w2_list = [
            torch.randn(K, H, device="cuda", dtype=torch.bfloat16)
            for _ in range(num_experts)
        ]

        stream = torch.cuda.Stream()
        # psum:               [G]             int32
        psum = make_psum(tokens_per_expert)

        # ---- FP8 path ----

        # fc1: quantize hidden states
        # fp8_hs:             [M, K]          fp8
        # fp8_hs_scales:      [M, K // 128]   f32
        fp8_hs, fp8_hs_scales = quantize_activation(hidden_states)
        # stacked_w1:         [G, 2H, K]      fp8
        # stacked_w1_scales:  [G, 2H//128, K//128]  f32
        stacked_w1, stacked_w1_scales = quantize_and_stack_weights(w1_list)

        # fc1 grouped GEMM:  hidden_states[i] @ w1[i].T => fc1_out[i]
        # fc1_out:            [M, 2H]         bf16
        fc1_out = m_grouped_fp8_gemm_nt_contiguous(
            psum,
            (fp8_hs, fp8_hs_scales),
            (stacked_w1, stacked_w1_scales),
            stream,
        )

        # SwiGLU:  splits [M, 2H] -> [M, H] + [M, H], applies silu(a)*b*probs
        # swiglu_out:         [M, H]          bf16
        swiglu_out = MergedSwiGLU.call_forward(fc1_out, permuted_probs)

        # fc2: quantize SwiGLU output
        # fp8_s:              [M, H]          fp8
        # fp8_s_scales:       [M, H // 128]   f32
        fp8_s, fp8_s_scales = quantize_activation(swiglu_out)
        # stacked_w2:         [G, K, H]       fp8
        # stacked_w2_scales:  [G, K//128, H//128]  f32
        stacked_w2, stacked_w2_scales = quantize_and_stack_weights(w2_list)

        # fc2 grouped GEMM:  swiglu_out[i] @ w2[i].T => output[i]
        # fp8_output:         [M, K]          bf16
        fp8_output = m_grouped_fp8_gemm_nt_contiguous(
            psum,
            (fp8_s, fp8_s_scales),
            (stacked_w2, stacked_w2_scales),
            stream,
        )
        torch.cuda.synchronize()

        # ---- bf16 reference path ----
        # ref_fc1:            [M, 2H]         bf16
        ref_fc1 = bf16_ref_grouped_gemm(hidden_states, w1_list, tokens_per_expert)
        # ref_swiglu:         [M, H]          bf16
        ref_swiglu = MergedSwiGLU.call_forward(ref_fc1, permuted_probs)
        # ref_output:         [M, K]          bf16
        ref_output = bf16_ref_grouped_gemm(ref_swiglu, w2_list, tokens_per_expert)

        diff = rel_diff(fp8_output, ref_output)
        assert diff < 0.10, f"e2e forward rel_diff={diff:.4f}"


# ---------------------------------------------------------------------------
# MoE backward flow tests
# ---------------------------------------------------------------------------


class TestMoEBackwardFlowGrouped:
    """Test grouped_fp8_gemm_nt_contiguous in the MoE backward path.

    For backward data gradients, the grouped kernel is used with
    transposed stacked weights.  For weight gradients, the per-expert
    fp8_gemm_nt with per_channel quantization is used (same approach as
    the multi_stream tests).
    """

    def setup_method(self):
        _skip_if_no_deep_gemm()
        _skip_if_no_gpu()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    @pytest.mark.parametrize("num_experts", [4, 8])
    def test_backward_grad_a(self, num_experts):
        """Backward through fc2: grad_y @ w2 -> grad_s.

        Uses transposed stacked weights so that the grouped kernel computes
        A @ B.T = grad_y @ (w2.T).T = grad_y @ w2.

        Per-expert GEMM shape:
          A = grad_y[i]     [M_i, K]    fp8
          B = w2_T[i]       [H, K]      fp8    (transposed from w2 [K, H], stacked as [G, H, K])
          D = A @ B.T       [M_i, H]    bf16   = grad_y[i] @ w2[i]
        """
        K = 256
        H = 256

        # tokens_per_expert:  [G]             int64   (each 128)
        tokens_per_expert = torch.tensor([128] * num_experts, dtype=torch.int64)
        # M:                  scalar
        M = tokens_per_expert.sum().item()

        # grad_y:             [M, K]          bf16
        grad_y = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        # w2_list[i]:         [K, H]          bf16
        w2_list = [
            torch.randn(K, H, device="cuda", dtype=torch.bfloat16)
            for _ in range(num_experts)
        ]

        # Quantize grad_y as activation  (recipe_a = (1, 128))
        # fp8_grad_y:         [M, K]          fp8
        # fp8_grad_y_scales:  [M, K // 128]   f32
        fp8_grad_y, fp8_grad_y_scales = quantize_activation(grad_y)

        # Quantize weights, then transpose each and stack
        # fw:                 [K, H]          fp8     => fw_t: [H, K]
        # fs:                 [K//128, H//128] f32    => fs_t: [H//128, K//128]
        # stacked_w2t:        [G, H, K]       fp8
        # stacked_w2t_scales: [G, H//128, K//128]  f32
        w2t_fp8_list, w2t_scales_list = [], []
        for w in w2_list:
            fw, fs = quantize_weight(w)
            w2t_fp8_list.append(fw.t().contiguous())
            w2t_scales_list.append(fs.t().contiguous())
        stacked_w2t = torch.stack(w2t_fp8_list)
        stacked_w2t_scales = torch.stack(w2t_scales_list)

        # psum:               [G]             int32
        psum = make_psum(tokens_per_expert)
        stream = torch.cuda.Stream()

        # Grouped FP8 GEMM:  grad_y[i] @ w2_T[i].T = grad_y[i] @ w2[i]
        #   => [M_i, K] @ [K, H] = [M_i, H]
        # grad_s:             [M, H]          bf16
        grad_s = m_grouped_fp8_gemm_nt_contiguous(
            psum,
            (fp8_grad_y, fp8_grad_y_scales),
            (stacked_w2t, stacked_w2t_scales),
            stream,
        )
        torch.cuda.synchronize()

        # bf16 reference: grad_s[i] = grad_y[i] @ w2[i]
        ref = torch.cat([
            (grad_y[s:e].float() @ w2_list[i].float()).bfloat16()
            for i, (s, e) in enumerate(zip(
                [0] + tokens_per_expert.cumsum(0).tolist()[:-1],
                tokens_per_expert.cumsum(0).tolist(),
            ))
        ], dim=0)

        assert grad_s.shape == (M, H)
        diff = rel_diff(grad_s, ref)
        print(f"backward grad_a rel_diff={diff:.4f}")
        assert diff < 0.05, f"backward grad_a rel_diff={diff:.4f}"

    @pytest.mark.parametrize("num_experts", [4, 8])
    def test_backward_grad_x(self, num_experts):
        """Backward through fc1: grad_a @ w1 -> grad_x.

        Uses transposed stacked weights so that the grouped kernel computes
        A @ B.T = grad_a @ (w1.T).T = grad_a @ w1.

        Per-expert GEMM shape:
          A = grad_a[i]     [M_i, 2H]   fp8
          B = w1_T[i]       [K, 2H]     fp8    (transposed from w1 [2H, K], stacked as [G, K, 2H])
          D = A @ B.T       [M_i, K]    bf16   = grad_a[i] @ w1[i]
        """
        K = 256
        H = 256
        fc1_out_dim = 2 * H

        # tokens_per_expert:  [G]             int64
        tokens_per_expert = torch.tensor([128] * num_experts, dtype=torch.int64)
        # M:                  scalar
        M = tokens_per_expert.sum().item()

        # grad_a:             [M, 2H]         bf16
        grad_a = torch.randn(M, fc1_out_dim, device="cuda", dtype=torch.bfloat16)
        # w1_list[i]:         [2H, K]         bf16
        w1_list = [
            torch.randn(fc1_out_dim, K, device="cuda", dtype=torch.bfloat16)
            for _ in range(num_experts)
        ]

        # Quantize grad_a as activation  (recipe_a = (1, 128))
        # fp8_grad_a:         [M, 2H]         fp8
        # fp8_grad_a_scales:  [M, 2H // 128]  f32
        fp8_grad_a, fp8_grad_a_scales = quantize_activation(grad_a)

        # Quantize weights, transpose each, then stack
        # fw:                 [2H, K]          fp8     => fw_t: [K, 2H]
        # fs:                 [2H//128, K//128] f32    => fs_t: [K//128, 2H//128]
        # stacked_w1t:        [G, K, 2H]       fp8
        # stacked_w1t_scales: [G, K//128, 2H//128]  f32
        w1t_fp8_list, w1t_scales_list = [], []
        for w in w1_list:
            fw, fs = quantize_weight(w)
            w1t_fp8_list.append(fw.t().contiguous())
            w1t_scales_list.append(fs.t().contiguous())
        stacked_w1t = torch.stack(w1t_fp8_list)
        stacked_w1t_scales = torch.stack(w1t_scales_list)

        # psum:               [G]             int32
        psum = make_psum(tokens_per_expert)
        stream = torch.cuda.Stream()

        # Grouped FP8 GEMM:  grad_a[i] @ w1_T[i].T = grad_a[i] @ w1[i]
        #   => [M_i, 2H] @ [2H, K] = [M_i, K]
        # grad_x:             [M, K]          bf16
        grad_x = m_grouped_fp8_gemm_nt_contiguous(
            psum,
            (fp8_grad_a, fp8_grad_a_scales),
            (stacked_w1t, stacked_w1t_scales),
            stream,
        )
        torch.cuda.synchronize()

        # bf16 reference: grad_x[i] = grad_a[i] @ w1[i]
        ref = torch.cat([
            (grad_a[s:e].float() @ w1_list[i].float()).bfloat16()
            for i, (s, e) in enumerate(zip(
                [0] + tokens_per_expert.cumsum(0).tolist()[:-1],
                tokens_per_expert.cumsum(0).tolist(),
            ))
        ], dim=0)

        assert grad_x.shape == (M, K)
        diff = rel_diff(grad_x, ref)
        print(f"backward grad_x rel_diff={diff:.4f}")
        assert diff < 0.05, f"backward grad_x rel_diff={diff:.4f}"

    # -- weight gradients (using k_grouped_fp8_gemm_nt_contiguous) -----------
    #
    # Mirrors call_backward_grad_w{1,2} in experts_offloading_fp8_util.py:
    #   per-channel FP8 quant (transpose=False) + pack_to_kmajor on data
    #   + .T on scales, then a single K-grouped GEMM over the token axis.
    #
    # Per expert i, dw[i] [N1, N2] = A[i].T [N1, M_i] @ B[i] [M_i, N2],
    # which the kernel computes from K-grouped layouts:
    #   A: [M_total, N1] data, [M_total//128, N1] scales
    #   B: [M_total, N2] data, [M_total//128, N2] scales
    # Output is stacked [G, N1, N2].

    @pytest.mark.parametrize("num_experts", [4, 8])
    def test_backward_grad_w2(self, num_experts):
        """grad_w2[i] [K, H] = grad_y[i].T @ s[i] via k_grouped_fp8_gemm.

          A = grad_y  [M, K]   per_channel(transpose=False) -> ([M, K] fp8, [M//128, K] f32)
          B = s       [M, H]   per_channel(transpose=False) -> ([M, H] fp8, [M//128, H] f32)
        """
        K = 256
        H = 256
        fc1_out_dim = 2 * H

        # tokens_per_expert:  [G]             int64   (each M_i = 128)
        tokens_per_expert = torch.tensor([128] * num_experts, dtype=torch.int64)
        # M:                  scalar
        M = tokens_per_expert.sum().item()
        # permuted_probs:     [M, 1]          bf16
        permuted_probs = torch.rand(M, 1, device="cuda", dtype=torch.bfloat16)

        # grad_y:             [M, K]          bf16    (upstream gradient)
        grad_y = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        # fc1_out:            [M, 2H]         bf16
        fc1_out = torch.randn(M, fc1_out_dim, device="cuda", dtype=torch.bfloat16)
        # s:                  [M, H]          bf16    (SwiGLU activation output)
        s = MergedSwiGLU.call_forward(fc1_out, permuted_probs)

        # ---- bf16 reference (computed BEFORE pack_to_kmajor frees fp8 storage) ----
        # ref_grad_w2:        [G, K, H]       bf16
        grad_y_pe = torch.split(grad_y, tokens_per_expert.tolist(), dim=0)
        s_pe = torch.split(s, tokens_per_expert.tolist(), dim=0)
        ref_grad_w2 = torch.stack([
            (grad_y_pe[i].float().t() @ s_pe[i].float()).bfloat16()
            for i in range(num_experts)
        ])

        # ---- per-channel FP8 quant (recipe (1, 1, 128) along K = token axis) ----
        # a_fp8:              [M, K]          fp8
        # a_scales:           [M//128, K]     f32
        a_fp8, a_scales = per_channel_cast_to_fp8(
            grad_y, use_ue8m0=False, gran_k=128, transpose=False
        )
        # b_fp8:              [M, H]          fp8
        # b_scales:           [M//128, H]     f32
        b_fp8, b_scales = per_channel_cast_to_fp8(
            s, use_ue8m0=False, gran_k=128, transpose=False
        )

        # ---- K-major pack on data, transpose on scales ----
        tpe_list = tokens_per_expert.tolist()
        tpe_cuda = tokens_per_expert.to(torch.int32).to("cuda")
        prefixes = make_kgrouped_prefixes(tokens_per_expert)
        # a_packed:           flat [M*K]      fp8     (per-expert K-major tiles)
        # a_scales_t:         [K, M//128]     f32
        a_packed = pack_to_kmajor(a_fp8, tpe_list, tpe_cuda, prefixes)
        a_scales_t = a_scales.T.contiguous()
        # b_packed:           flat [M*H]      fp8
        # b_scales_t:         [H, M//128]     f32
        b_packed = pack_to_kmajor(b_fp8, tpe_list, tpe_cuda, prefixes)
        b_scales_t = b_scales.T.contiguous()

        # output:             [G, K, H]       bf16   (matches main_grad layout)
        output = torch.zeros(num_experts, K, H, device="cuda", dtype=torch.float32)
        stream = torch.cuda.Stream()

        # K-grouped FP8 GEMM — single launch for all experts
        k_grouped_fp8_gemm_nt_contiguous(
            tpe_list,
            tpe_cuda,
            (a_packed, a_scales_t),
            (b_packed, b_scales_t),
            num_experts,
            stream,
            output=output,
        )
        torch.cuda.synchronize()

        for i in range(num_experts):
            diff = rel_diff(output[i], ref_grad_w2[i])
            print(f"grad_w2[{i}] rel_diff={diff:.4f}")
            assert diff < 0.05, f"grad_w2[{i}] rel_diff={diff:.4f}"

    @pytest.mark.parametrize("num_experts", [4, 8])
    def test_backward_grad_w1(self, num_experts):
        """grad_w1[i] [2H, K] = grad_a[i].T @ hidden_states[i] via k_grouped_fp8_gemm.

          A = grad_a         [M, 2H]  per_channel(transpose=False)
                                     -> ([M, 2H] fp8, [M//128, 2H] f32)
          B = hidden_states  [M, K]   per_channel(transpose=False)
                                     -> ([M, K] fp8, [M//128, K] f32)
        """
        K = 256
        H = 256
        fc1_out_dim = 2 * H

        # tokens_per_expert:  [G]             int64   (each M_i = 128)
        tokens_per_expert = torch.tensor([128] * num_experts, dtype=torch.int64)
        # M:                  scalar
        M = tokens_per_expert.sum().item()

        # grad_a:             [M, 2H]         bf16
        grad_a = torch.randn(M, fc1_out_dim, device="cuda", dtype=torch.bfloat16)
        # hidden_states:      [M, K]          bf16
        hidden_states = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        # ---- bf16 reference (computed before pack_to_kmajor) ----
        # ref_grad_w1:        [G, 2H, K]      bf16
        grad_a_pe = torch.split(grad_a, tokens_per_expert.tolist(), dim=0)
        hs_pe = torch.split(hidden_states, tokens_per_expert.tolist(), dim=0)
        ref_grad_w1 = torch.stack([
            (grad_a_pe[i].float().t() @ hs_pe[i].float()).bfloat16()
            for i in range(num_experts)
        ])

        # ---- per-channel FP8 quant ----
        # a_fp8:              [M, 2H]         fp8
        # a_scales:           [M//128, 2H]    f32
        a_fp8, a_scales = per_channel_cast_to_fp8(
            grad_a, use_ue8m0=False, gran_k=128, transpose=False
        )
        # b_fp8:              [M, K]          fp8
        # b_scales:           [M//128, K]     f32
        b_fp8, b_scales = per_channel_cast_to_fp8(
            hidden_states, use_ue8m0=False, gran_k=128, transpose=False
        )

        # ---- K-major pack on data, transpose on scales ----
        tpe_list = tokens_per_expert.tolist()
        tpe_cuda = tokens_per_expert.to(torch.int32).to("cuda")
        prefixes = make_kgrouped_prefixes(tokens_per_expert)
        # a_packed:           flat [M*2H]     fp8
        # a_scales_t:         [2H, M//128]    f32
        a_packed = pack_to_kmajor(a_fp8, tpe_list, tpe_cuda, prefixes)
        a_scales_t = a_scales.T.contiguous()
        # b_packed:           flat [M*K]      fp8
        # b_scales_t:         [K, M//128]     f32
        b_packed = pack_to_kmajor(b_fp8, tpe_list, tpe_cuda, prefixes)
        b_scales_t = b_scales.T.contiguous()

        # output:             [G, 2H, K]      bf16
        output = torch.zeros(
            num_experts, fc1_out_dim, K, device="cuda", dtype=torch.float32
        )
        stream = torch.cuda.Stream()

        k_grouped_fp8_gemm_nt_contiguous(
            tpe_list,
            tpe_cuda,
            (a_packed, a_scales_t),
            (b_packed, b_scales_t),
            num_experts,
            stream,
            output=output,
        )
        torch.cuda.synchronize()

        for i in range(num_experts):
            diff = rel_diff(output[i], ref_grad_w1[i])
            print(f"grad_w1[{i}] rel_diff={diff:.4f}")
            assert diff < 0.05, f"grad_w1[{i}] rel_diff={diff:.4f}"

    @pytest.mark.parametrize("num_experts", [4])
    def test_backward_full_with_wgrad(self, num_experts):
        """Full MoE backward: data gradients + weight gradients.

        Both data gradients and weight gradients use grouped_fp8_gemm_nt_contiguous.

        Forward:
          hidden_states [M, K]  --fc1-->  fc1_out [M, 2H]  --SwiGLU-->  swiglu_out [M, H]  --fc2-->  output [M, K]

        Backward:
          Step 1: grad_s  = grad_y @ w2               (grouped GEMM, transposed stacked w2)
          Step 2: grad_a  = SwiGLU.backward(grad_s, fc1_out, probs)
          Step 3: grad_x  = grad_a @ w1               (grouped GEMM, transposed stacked w1)
          Step 4: grad_w2[i] = grad_y[i].T @ s[i]     (grouped GEMM, per_token+per_block quant)
          Step 5: grad_w1[i] = grad_a[i].T @ hs[i]    (grouped GEMM, per_token+per_block quant)
        """
        K = 256
        H = 256
        fc1_out_dim = 2 * H

        # tokens_per_expert:  [G]             int64
        tokens_per_expert = torch.tensor([128] * num_experts, dtype=torch.int64)
        # M:                  scalar
        M = tokens_per_expert.sum().item()
        # permuted_probs:     [M, 1]          bf16
        permuted_probs = torch.rand(M, 1, device="cuda", dtype=torch.bfloat16)

        # hidden_states:      [M, K]          bf16
        hidden_states = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 64).detach().requires_grad_(True)
        # w1_list[i]:         [2H, K]         bf16
        w1_list = [
            torch.randn(fc1_out_dim, K, device="cuda", dtype=torch.bfloat16)
            for _ in range(num_experts)
        ]
        # w2_list[i]:         [K, H]          bf16
        w2_list = [
            torch.randn(K, H, device="cuda", dtype=torch.bfloat16)
            for _ in range(num_experts)
        ]
        # target:             [M, K]          bf16
        # Shared upstream "label" used to derive grad_y from each forward
        # output, so the fp8 and bf16 paths see different grad_y values
        # (matching what happens in real training where grad_y depends on
        # the forward output).
        target = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

        stream = torch.cuda.Stream()
        # psum:               [G]             int32
        psum = make_psum(tokens_per_expert)

        # ==== FP8 FORWARD SIMULATION (fc1 -> SwiGLU -> fc2) ====

        # fc1: quantize hidden states
        # fp8_hs:             [M, K]          fp8
        # fp8_hs_scales:      [M, K // 128]   f32
        fp8_hs, fp8_hs_scales = quantize_activation(hidden_states)
        # stacked_w1:         [G, 2H, K]      fp8
        # stacked_w1_scales:  [G, 2H//128, K//128]  f32
        stacked_w1, stacked_w1_scales = quantize_and_stack_weights(w1_list)
        # fp8_fc1_out:        [M, 2H]         bf16
        fp8_fc1_out = m_grouped_fp8_gemm_nt_contiguous(
            psum,
            (fp8_hs, fp8_hs_scales),
            (stacked_w1, stacked_w1_scales),
            stream,
        )
        # fp8_swiglu_out:     [M, H]          bf16
        fp8_swiglu_out = MergedSwiGLU.call_forward(fp8_fc1_out, permuted_probs)

        # fc2: quantize SwiGLU output
        # fp8_s:              [M, H]          fp8
        # fp8_s_scales:       [M, H // 128]   f32
        fp8_s, fp8_s_scales = quantize_activation(fp8_swiglu_out)
        # stacked_w2:         [G, K, H]       fp8
        # stacked_w2_scales:  [G, K//128, H//128]  f32
        stacked_w2, stacked_w2_scales = quantize_and_stack_weights(w2_list)
        # fp8_output:         [M, K]          bf16
        fp8_output = m_grouped_fp8_gemm_nt_contiguous(
            psum,
            (fp8_s, fp8_s_scales),
            (stacked_w2, stacked_w2_scales),
            stream,
        )
        torch.cuda.synchronize()

        # ---- bf16 reference forward (for accuracy checks + SwiGLU backward) ----
        # ref_fc1:            [M, 2H]         bf16
        ref_fc1 = bf16_ref_grouped_gemm(hidden_states, w1_list, tokens_per_expert)
        # ref_s:              [M, H]          bf16
        ref_s = MergedSwiGLU.call_forward(ref_fc1, permuted_probs)
        # ref_output:         [M, K]          bf16
        ref_output = bf16_ref_grouped_gemm(ref_s, w2_list, tokens_per_expert)

        diff_fc1 = rel_diff(fp8_fc1_out, ref_fc1)
        diff_swiglu = rel_diff(fp8_swiglu_out, ref_s)
        diff_fwd = rel_diff(fp8_output, ref_output)
        print(f"Forward rel_diffs: fc1={diff_fc1:.4f}, swiglu={diff_swiglu:.4f}, output={diff_fwd:.4f}")
        assert diff_fwd < 0.10, f"e2e forward rel_diff={diff_fwd:.4f}"

        # Derive grad_y from each forward output (MSE-style upstream gradient).
        # fp8 and bf16 paths see different grad_y because their forward outputs
        # differ.
        # grad_y_fp8:         [M, K]          bf16
        grad_y_fp8 = (fp8_output - target).to(torch.bfloat16)
        # grad_y_bf16:        [M, K]          bf16
        grad_y_bf16 = (ref_output - target).to(torch.bfloat16)

        print(f"Forward rel_diff={diff_fwd:.4f}, grad_y rel_diff={rel_diff(grad_y_fp8, grad_y_bf16):.4f}")

        # ==== DATA GRADIENTS (grouped GEMM) ====

        # Step 1: grad_s = grad_y @ w2  (transposed stacked w2)
        # fp8_grad_y:         [M, K]          fp8
        # fp8_grad_y_scales:  [M, K // 128]   f32
        fp8_grad_y, fp8_grad_y_scales = quantize_activation(grad_y_fp8)
        # stacked_w2t:        [G, H, K]       fp8
        # stacked_w2t_scales: [G, H//128, K//128]  f32
        w2t_fp8, w2t_scales = [], []
        for w in w2_list:
            fw, fs = quantize_weight(w)
            w2t_fp8.append(fw.t().contiguous())
            w2t_scales.append(fs.t().contiguous())
        stacked_w2t = torch.stack(w2t_fp8)
        stacked_w2t_scales = torch.stack(w2t_scales)
        # fp8_grad_s:         [M, H]          bf16
        fp8_grad_s = m_grouped_fp8_gemm_nt_contiguous(
            psum,
            (fp8_grad_y, fp8_grad_y_scales),
            (stacked_w2t, stacked_w2t_scales),
            stream,
        )

        # Step 2: SwiGLU backward
        # Use fp8_fc1_out (bf16 output of the fp8 fc1 GEMM), matching what
        # the real backward sees as saved fc1_output.
        # fp8_grad_a:         [M, 2H]         bf16
        fp8_grad_a, _ = MergedSwiGLU.call_backward(fp8_grad_s, fp8_fc1_out, permuted_probs)
        fp8_grad_a = fp8_grad_a * (2.4e-18) ** 0.5 + 1e-12

        # Step 3: grad_x = grad_a @ w1  (transposed stacked w1)
        # fp8_ga:             [M, 2H]         fp8
        # fp8_ga_scales:      [M, 2H // 128]  f32
        fp8_ga, fp8_ga_scales = quantize_activation(fp8_grad_a)
        # stacked_w1t:        [G, K, 2H]      fp8
        # stacked_w1t_scales: [G, K//128, 2H//128]  f32
        w1t_fp8, w1t_scales = [], []
        for w in w1_list:
            fw, fs = quantize_weight(w)
            w1t_fp8.append(fw.t().contiguous())
            w1t_scales.append(fs.t().contiguous())
        stacked_w1t = torch.stack(w1t_fp8)
        stacked_w1t_scales = torch.stack(w1t_scales)
        # fp8_grad_x:         [M, K]          bf16
        fp8_grad_x = m_grouped_fp8_gemm_nt_contiguous(
            psum,
            (fp8_ga, fp8_ga_scales),
            (stacked_w1t, stacked_w1t_scales),
            stream,
        )

        # ==== WEIGHT GRADIENTS (k_grouped GEMM, per_channel quant) ====
        #
        # Mirrors call_backward_grad_w{1,2} in experts_offloading_fp8_util.py.
        # Single K-grouped kernel launch per wgrad — no per-expert transposes.

        tpe_list = tokens_per_expert.tolist()
        tpe_cuda = tokens_per_expert.to(torch.int32).to("cuda")
        prefixes = make_kgrouped_prefixes(tokens_per_expert)

        # Step 4: grad_w2[i] [K, H] = grad_y[i].T @ s[i]
        # Mirrors call_backward_grad_w2: s is recomputed via swiglu_forward
        # on the saved fc1_output (here fp8_fc1_out), NOT on the bf16
        # reference ref_fc1.
        # fp8_s_recompute:    [M, H]          bf16
        fp8_s_recompute = MergedSwiGLU.call_forward(fp8_fc1_out, permuted_probs)
        # A = grad_y_fp8      [M, K] per_channel -> ([M, K] fp8, [M//128, K] f32)
        # B = fp8_s_recompute [M, H] per_channel -> ([M, H] fp8, [M//128, H] f32)
        w2_a_fp8, w2_a_scales = per_channel_cast_to_fp8(
            grad_y_fp8, use_ue8m0=False, gran_k=128, transpose=False
        )
        w2_b_fp8, w2_b_scales = per_channel_cast_to_fp8(
            fp8_s_recompute, use_ue8m0=False, gran_k=128, transpose=False
        )
        w2_a_packed = pack_to_kmajor(w2_a_fp8, tpe_list, tpe_cuda, prefixes)
        w2_a_scales_t = w2_a_scales.T.contiguous()
        w2_b_packed = pack_to_kmajor(w2_b_fp8, tpe_list, tpe_cuda, prefixes)
        w2_b_scales_t = w2_b_scales.T.contiguous()
        # w2_output:         [G, K, H]       bf16
        w2_output = torch.zeros(num_experts, K, H, device="cuda", dtype=torch.float32)
        k_grouped_fp8_gemm_nt_contiguous(
            tpe_list, tpe_cuda,
            (w2_a_packed, w2_a_scales_t),
            (w2_b_packed, w2_b_scales_t),
            num_experts, stream, output=w2_output,
        )
        fp8_grad_w2_list = list(torch.unbind(w2_output, dim=0))

        # Step 5: grad_w1[i] [2H, K] = grad_a[i].T @ hidden_states[i]
        # Mirrors call_backward_grad_w1: the real backward only saved the
        # per-token-FP8 hidden states from the forward; it recovers bf16 via
        # per_token_dequant_from_fp8, then casts per-channel. So x goes
        # through TWO casts (per_token + per_channel), which is the main
        # source of the extra grad_w1 error vs. grad_w2.
        # fp8_hs_pt:          [M, K]          fp8  (per-token, saved from fwd)
        # fp8_hs_pt_scales:   [M, K//128]     f32
        fp8_hs_pt, fp8_hs_pt_scales = quantize_activation(hidden_states)
        # bf16_hs_roundtrip:  [M, K]          bf16 (dequantized in backward)
        bf16_hs_roundtrip = per_token_dequant_from_fp8(fp8_hs_pt, fp8_hs_pt_scales)
        # A = fp8_grad_a (bf16 grad_a) [M, 2H] per_channel -> ([M, 2H] fp8, [M//128, 2H] f32)
        # B = bf16_hs_roundtrip        [M, K]  per_channel -> ([M, K]  fp8, [M//128, K]  f32)
        w1_a_fp8, w1_a_scales = per_channel_cast_to_fp8(
            fp8_grad_a, use_ue8m0=False, gran_k=128, transpose=False
        )
        w1_b_fp8, w1_b_scales = per_channel_cast_to_fp8(
            bf16_hs_roundtrip, use_ue8m0=False, gran_k=128, transpose=False
        )
        w1_a_packed = pack_to_kmajor(w1_a_fp8, tpe_list, tpe_cuda, prefixes)
        w1_a_scales_t = w1_a_scales.T.contiguous()
        w1_b_packed = pack_to_kmajor(w1_b_fp8, tpe_list, tpe_cuda, prefixes)
        w1_b_scales_t = w1_b_scales.T.contiguous()
        # w1_output:         [G, 2H, K]      bf16
        w1_output = torch.zeros(
            num_experts, fc1_out_dim, K, device="cuda", dtype=torch.float32
        )
        k_grouped_fp8_gemm_nt_contiguous(
            tpe_list, tpe_cuda,
            (w1_a_packed, w1_a_scales_t),
            (w1_b_packed, w1_b_scales_t),
            num_experts, stream, output=w1_output,
        )
        fp8_grad_w1_list = list(torch.unbind(w1_output, dim=0))
        torch.cuda.synchronize()

        # ---- bf16 reference backward ----
        # ref_grad_s:         [M, H]          bf16
        ref_grad_s = torch.cat([
            (grad_y_bf16[s:e].float() @ w2_list[i].float()).bfloat16()
            for i, (s, e) in enumerate(zip(
                [0] + tokens_per_expert.cumsum(0).tolist()[:-1],
                tokens_per_expert.cumsum(0).tolist(),
            ))
        ], dim=0)
        ref_grad_a, _ = MergedSwiGLU.call_backward(ref_grad_s, ref_fc1, permuted_probs)
        ref_grad_a = ref_grad_a * (2.4e-18) ** 0.5 + 1e-12

        # ref_grad_x:         [M, K]          bf16
        ref_grad_x = torch.cat([
            (ga.float() @ w1.float()).bfloat16()
            for ga, w1 in zip(
                torch.split(ref_grad_a, tokens_per_expert.tolist(), dim=0),
                w1_list,
            )
        ], dim=0)

        # ref_grad_w2_list[i]: [K, H]         bf16
        ref_grad_w2_list = [
            (gy.float().t() @ si.float()).bfloat16()
            for gy, si in zip(
                torch.split(grad_y_bf16, tokens_per_expert.tolist(), dim=0),
                torch.split(ref_s, tokens_per_expert.tolist(), dim=0),
            )
        ]

        # ref_grad_w1_list[i]: [2H, K]        bf16
        ref_grad_w1_list = [
            (ga.float().t() @ hi.float()).bfloat16()
            for ga, hi in zip(
                torch.split(ref_grad_a, tokens_per_expert.tolist(), dim=0),
                torch.split(hidden_states, tokens_per_expert.tolist(), dim=0),
            )
        ]

        # ---- Assertions ----
        diff_gs = rel_diff(fp8_grad_s, ref_grad_s)
        assert diff_gs < 0.07, f"grad_s rel_diff={diff_gs:.4f}"

        diff_ga = rel_diff(fp8_grad_a, ref_grad_a)
        assert diff_ga < 0.07, f"grad_a rel_diff={diff_ga:.4f}"

        diff_gx = rel_diff(fp8_grad_x, ref_grad_x)
        assert diff_gx < 0.08, f"grad_x rel_diff={diff_gx:.4f}"

        print(f"grad_s rel_diff={diff_gs:.4f}, grad_a rel_diff={diff_ga:.4f}, grad_x rel_diff={diff_gx:.4f}")

        for i in range(num_experts):
            diff_w2 = rel_diff(fp8_grad_w2_list[i].bfloat16(), ref_grad_w2_list[i])
            print(f"grad_w2[{i}] rel_diff={diff_w2:.4f}")
            assert diff_w2 < 0.09, f"grad_w2[{i}] rel_diff={diff_w2:.4f}"

            diff_w1 = rel_diff(fp8_grad_w1_list[i].bfloat16(), ref_grad_w1_list[i])
            print(f"grad_w1[{i}] rel_diff={diff_w1:.4f}")
            # Looser bound than grad_w2: x goes through per_token + per_channel
            # casts (matches OffloadingExpertsFP8GroupedSwiMLP.backward).
            assert diff_w1 < 0.15, f"grad_w1[{i}] rel_diff={diff_w1:.4f}"


if __name__ == "__main__":
    tbwd = TestMoEBackwardFlowGrouped()
    # tbwd.test_backward_grad_a(32)
    # tbwd.test_backward_grad_x(32)
    # tbwd.test_backward_grad_w2(32)
    # tbwd.test_backward_grad_w1(32)
    tbwd.test_backward_full_with_wgrad(32)
