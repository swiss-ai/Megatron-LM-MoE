# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import gc

import pytest
import torch

from megatron.core.transformer.moe.experts_util import MergedSwiGLU
from megatron.core.transformer.moe.moe_offload import StreamManager
from megatron.core.transformer.moe.experts_offloading_fp8_util import (
    FP8ExpertsParameterManager,
    OffloadingFP8Config,
    offloading_fp8_grouped_swiglu_mlp,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class TestOffloadingMoELayerFP8:
    """Test MoE layer with FP8 precision and CPU weight offloading."""

    def setup_method(self, method):
        FP8ExpertsParameterManager.reset_instance()
        gc.collect()
        torch.cuda.empty_cache()

    def teardown_method(self, method):
        FP8ExpertsParameterManager.reset_instance()
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("num_moe_experts", [64])
    def test_offloading_moe_forward_backward(
        self, num_moe_experts, profile=False, num_repeats=10
    ):
        """Test MoE layer forward and backward pass with fp16 params and inputs."""
        hidden_size = 7168
        moe_ffn_hidden_size = 2048
        moe_offloading_num_chunks = 4
        moe_offloading_num_stages = 2
        moe_offloading_chunk_size = num_moe_experts // moe_offloading_num_chunks

        tokens_per_expert = torch.randint(
            1024, 1025, (num_moe_experts,), device="cpu"
        )
        tokens_per_expert_ref = tokens_per_expert.detach().clone()

        hidden_states = (
            torch.randn(
                tokens_per_expert.sum().item(),
                hidden_size,
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16,
                requires_grad=False,
            )
        ).detach().requires_grad_(True)
        hidden_states_ref = hidden_states.detach().clone()

        permuted_probs = torch.rand(
            tokens_per_expert.sum().item(),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        permuted_probs_ref = permuted_probs.detach().clone()

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_moe_experts=num_moe_experts,
            num_attention_heads=16,
            use_cpu_initialization=True,
            perform_initialization=False,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            add_bias_linear=False,
            fp16=False,
            params_dtype=torch.bfloat16,
            gated_linear_unit=True,
            moe_offloading_num_chunks=moe_offloading_num_chunks,
            moe_offloading_num_stages=moe_offloading_num_stages,
            moe_offloading_chunk_size=moe_offloading_chunk_size,
            gradient_accumulation_fusion=True,
        )
        offloading_config = OffloadingFP8Config.from_transformer_config(transformer_config)

        # Draw weights on CUDA so the CUDA RNG stream advances identically to
        # the non-offloading test, then mirror to pinned CPU for the offload path.
        w1_init = torch.randn(
            num_moe_experts, moe_ffn_hidden_size * 2, hidden_size,
            device=torch.cuda.current_device(), dtype=torch.bfloat16,
        )
        w2_init = torch.randn(
            num_moe_experts, hidden_size, moe_ffn_hidden_size,
            device=torch.cuda.current_device(), dtype=torch.bfloat16,
        )
        cpu_w1 = torch.nn.Parameter(
            torch.empty_like(w1_init, device="cpu", pin_memory=True).copy_(w1_init),
            requires_grad=True,
        )
        cpu_w2 = torch.nn.Parameter(
            torch.empty_like(w2_init, device="cpu", pin_memory=True).copy_(w2_init),
            requires_grad=True,
        )

        cpu_w1_list = list(torch.unbind(cpu_w1, dim=0))
        cpu_w2_list = list(torch.unbind(cpu_w2, dim=0))

        cpu_w1.main_grad = torch.zeros_like(cpu_w1, device="cuda", dtype=torch.float32)
        cpu_w2.main_grad = torch.zeros_like(cpu_w2, device="cuda", dtype=torch.float32)

        # Allocate contiguous GPU buffers for FP8 weights: [num_stages, chunk_size, ...]
        experts1_gpu_buffers_storage = torch.empty(
            moe_offloading_num_stages * moe_offloading_chunk_size * moe_ffn_hidden_size * 2 * hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.float8_e4m3fn,
        ).view(
            moe_offloading_num_stages, moe_offloading_chunk_size, moe_ffn_hidden_size * 2, hidden_size
        )
        experts2_gpu_buffers_storage = torch.empty(
            moe_offloading_num_stages * moe_offloading_chunk_size * moe_ffn_hidden_size * hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.float8_e4m3fn,
        ).view(
            moe_offloading_num_stages, moe_offloading_chunk_size, hidden_size, moe_ffn_hidden_size
        )

        # Per-chunk views: [num_stages, chunk_size, ...]
        gpu_w1_buffer = [
            [experts1_gpu_buffers_storage[s, c] for c in range(moe_offloading_chunk_size)]
            for s in range(moe_offloading_num_stages)
        ]
        gpu_w2_buffer = [
            [experts2_gpu_buffers_storage[s, c] for c in range(moe_offloading_chunk_size)]
            for s in range(moe_offloading_num_stages)
        ]

        # Per-stage views: [num_stages, chunk_size, ...]
        gpu_w1_chunks = [
            experts1_gpu_buffers_storage[s] for s in range(moe_offloading_num_stages)
        ]
        gpu_w2_chunks = [
            experts2_gpu_buffers_storage[s] for s in range(moe_offloading_num_stages)
        ]

        # Reference weights on GPU (bf16, no offloading)
        gpu_w1 = torch.nn.Parameter(
            cpu_w1.detach().clone().cuda(), requires_grad=True,
        )
        gpu_w2 = torch.nn.Parameter(
            cpu_w2.detach().clone().cuda(), requires_grad=True,
        )
        gpu_w1.main_grad = torch.zeros_like(gpu_w1, device="cuda", dtype=torch.float32)
        gpu_w2.main_grad = torch.zeros_like(gpu_w2, device="cuda", dtype=torch.float32)

        torch.cuda.synchronize()

        stream_manager = StreamManager(moe_offloading_num_stages, 4)
        FP8ExpertsParameterManager.create_instance(config=offloading_config)

        # Realistic upstream gradient: simulate an MSE loss against a random
        # target, so grad_y has the same per-token magnitude structure as in
        # training.  Using .sum().backward() (grad_y = 1) makes grad_w1/grad_w2
        # errors depend on bf16-vs-fp32 accumulation order rather than on the
        # quantization paths we care about.
        target = torch.randn(
            tokens_per_expert.sum().item(), hidden_size,
            device=torch.cuda.current_device(), dtype=torch.bfloat16,
        )

        # --- Profiling path (manual use only) ---
        if profile:
            wait, warmup, active = 1, 5, 2
            num_steps = wait + warmup + active
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=wait, warmup=warmup, active=active, repeat=1, skip_first=1
                ),
            ) as prof:
                for _ in range(num_steps):
                    output, _ = offloading_fp8_grouped_swiglu_mlp(
                        cpu_w1, cpu_w2, cpu_w1_list, cpu_w2_list,
                        gpu_w1_buffer, gpu_w2_buffer,
                        gpu_w1_chunks, gpu_w2_chunks,
                        hidden_states, tokens_per_expert, num_moe_experts,
                        permuted_probs, None, stream_manager,
                        offloading_config, [],
                    )
                    torch.cuda.synchronize()
                    output.sum().backward()
                    torch.cuda.synchronize()
                    prof.step()
            prof.export_chrome_trace("fp8_e2e.json")
            return

        # --- Reference path (bf16 matmuls on GPU) ---
        hidden_states_ref_list = list(
            torch.split(hidden_states_ref, tokens_per_expert_ref.tolist(), dim=0)
        )
        fc1_outputs = [
            torch.mm(hidden_states_ref_list[i], gpu_w1[i].t())
            for i in range(num_moe_experts)
        ]
        act_output = MergedSwiGLU.apply(
            torch.cat(fc1_outputs, dim=0), permuted_probs_ref.unsqueeze(-1)
        )
        act_list = list(torch.split(act_output, tokens_per_expert_ref.tolist(), dim=0))
        output_ref = torch.cat([
            torch.mm(act_list[i], gpu_w2[i].t())
            for i in range(num_moe_experts)
        ], dim=0)
        ((output_ref - target).float() ** 2).sum().backward()
        torch.cuda.synchronize()

        # --- Offloading path (FP8 with CPU offload) ---
        outputs = []
        for _ in range(num_repeats):
            output, _ = offloading_fp8_grouped_swiglu_mlp(
                cpu_w1, cpu_w2, cpu_w1_list, cpu_w2_list,
                gpu_w1_buffer, gpu_w2_buffer,
                gpu_w1_chunks, gpu_w2_chunks,
                hidden_states, tokens_per_expert, num_moe_experts,
                permuted_probs, None, stream_manager,
                offloading_config, [],
            )
            cpu_w1.main_grad.zero_()
            cpu_w2.main_grad.zero_()
            ((output - target).float() ** 2).sum().backward()
            outputs.append(output.detach())
            torch.cuda.synchronize()

        # --- Comparison ---
        def diff_tensor_norm(tensor1, tensor2):
            return (
                torch.norm(tensor1.to(torch.float) - tensor2.to(torch.float)).item()
                / torch.norm(tensor2.to(torch.float)).item()
            )

        for i, output in enumerate(outputs):
            assert output.shape == output_ref.shape, (
                f"Output shape mismatch: {output.shape} vs {output_ref.shape}"
            )
            diff = diff_tensor_norm(output, output_ref)
            if diff > 0.07:
                half = tokens_per_expert.sum().item() // 2
                diff_half_1 = diff_tensor_norm(output[:half], output_ref[:half])
                diff_half_2 = diff_tensor_norm(output[half:], output_ref[half:])
                raise AssertionError(
                    f"[repeat {i}] Output norm diff {diff:.4f} exceeds threshold "
                    f"(half1={diff_half_1:.4f}, half2={diff_half_2:.4f})"
                )

        w1_grad_diff = diff_tensor_norm(cpu_w1.main_grad, gpu_w1.grad)
        w2_grad_diff = diff_tensor_norm(cpu_w2.main_grad, gpu_w2.grad)
        assert w1_grad_diff < 0.10, (
            f"W1 grad norm diff {w1_grad_diff:.4f} exceeds threshold"
        )
        assert w2_grad_diff < 0.10, (
            f"W2 grad norm diff {w2_grad_diff:.4f} exceeds threshold"
        )


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    TestOffloadingMoELayerFP8().test_offloading_moe_forward_backward(
        num_moe_experts=64, profile=False, num_repeats=10
    )
