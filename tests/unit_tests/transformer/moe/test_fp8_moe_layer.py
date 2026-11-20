# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import gc

import pytest
import torch

from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.core.transformer.moe.experts_util import MergedSwiGLU

from megatron.core.transformer.moe.experts_fp8_util import (
    ExpertsFP8GroupedSwiMLP,
    FP8GPUExpertsParameterManager,
    fp8_grouped_swiglu_mlp,
)


class TestMoELayerFP8:
    """Test the non-offloading FP8 grouped SwiGLU MLP against a bf16 reference."""

    def setup_method(self, method):
        FP8GPUExpertsParameterManager.reset_instance()

    def teardown_method(self, method):
        FP8GPUExpertsParameterManager.reset_instance()
        gc.collect()
        torch.cuda.empty_cache()

    def test_fp8_moe_forward_backward(
        self, num_moe_experts=64, profile=False, num_repeats=10,
    ):
        hidden_size = 7168
        moe_ffn_hidden_size = 2048

        tokens_per_expert = torch.randint(1024, 1025, (num_moe_experts,), device="cpu")
        tokens_per_expert[num_moe_experts//2] = 0
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
            gradient_accumulation_fusion=True,
        )

        # GPU-resident stacked expert weights.
        # Shape mirrors the offloading test: w1 is (E, 2*ffn, hidden), w2 is (E, hidden, ffn).
        w1 = torch.nn.Parameter(
            torch.randn(
                num_moe_experts, moe_ffn_hidden_size * 2, hidden_size,
                device=torch.cuda.current_device(), dtype=torch.bfloat16, requires_grad=True,
            )
        )
        w2 = torch.nn.Parameter(
            torch.randn(
                num_moe_experts, hidden_size, moe_ffn_hidden_size,
                device=torch.cuda.current_device(), dtype=torch.bfloat16, requires_grad=True,
            )
        )

        # main_grad buffers expected by gradient_accumulation_fusion path
        w1.main_grad = torch.zeros_like(w1, device="cuda", dtype=torch.float32)
        w2.main_grad = torch.zeros_like(w2, device="cuda", dtype=torch.float32)

        # bf16 reference parameters (identical values)
        w1_ref = torch.nn.Parameter(w1.detach().clone(), requires_grad=True)
        w2_ref = torch.nn.Parameter(w2.detach().clone(), requires_grad=True)
        w1_ref.main_grad = torch.zeros_like(w1_ref, device="cuda", dtype=torch.float32)
        w2_ref.main_grad = torch.zeros_like(w2_ref, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()

        FP8GPUExpertsParameterManager.create_instance(config=transformer_config)

        # MSE-style upstream gradient (same rationale as the offloading test)
        target = torch.randn(
            tokens_per_expert.sum().item(), hidden_size,
            device=torch.cuda.current_device(), dtype=torch.bfloat16,
        )

        if not profile:
            # bf16 reference: per-expert split, mm, SwiGLU, mm, MSE backward.
            hs_ref_split = list(torch.split(hidden_states_ref, tokens_per_expert_ref.tolist(), dim=0))
            fc1_parts = [torch.mm(hs_ref_split[i], w1_ref[i].t()) for i in range(num_moe_experts)]
            outputs_fc1 = torch.cat(fc1_parts, dim=0)
            act = MergedSwiGLU.apply(outputs_fc1, permuted_probs_ref.unsqueeze(-1))
            act_split = list(torch.split(act, tokens_per_expert_ref.tolist(), dim=0))
            fc2_parts = [torch.mm(act_split[i], w2_ref[i].t()) for i in range(num_moe_experts)]
            output_ref = torch.cat(fc2_parts, dim=0)
            ((output_ref - target).float() ** 2).sum().backward()
            torch.cuda.synchronize()

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
                    output = fp8_grouped_swiglu_mlp(
                        w1,
                        w2,
                        hidden_states,
                        tokens_per_expert,
                        num_moe_experts,
                        permuted_probs,
                        None,
                        transformer_config,
                    )
                    torch.cuda.synchronize()
                    ((output - target).float() ** 2).sum().backward()
                    torch.cuda.synchronize()
                    prof.step()
            prof.export_chrome_trace("fp8_e2e_no_offload.json")
        else:
            outputs = []
            for _ in range(num_repeats):
                output = fp8_grouped_swiglu_mlp(
                    w1,
                    w2,
                    hidden_states,
                    tokens_per_expert,
                    num_moe_experts,
                    permuted_probs,
                    None,
                    transformer_config,
                )
                w1.main_grad.zero_()
                w2.main_grad.zero_()
                # Mark a first microbatch so cached FP8 weights re-quantize from the
                # (unchanged) bf16 params; exercises the refresh path.
                FP8GPUExpertsParameterManager.mark_first_microbatch()
                ((output - target).float() ** 2).sum().backward()
                outputs.append(output)
                torch.cuda.synchronize()

        if not profile:
            def diff_tensor_norm(t1, t2):
                t1f = t1.to(torch.float)
                t2f = t2.to(torch.float)
                return torch.norm(t1f - t2f).item() / torch.norm(t2f).item()

            for i, output in enumerate(outputs):
                assert output.shape == output_ref.shape, \
                    f"Output shape mismatch: {output.shape} vs {output_ref.shape}"
                n_half = tokens_per_expert.sum().item() // 2
                diff_half_1 = diff_tensor_norm(output.cuda()[:n_half], output_ref.cuda()[:n_half])
                diff_half_2 = diff_tensor_norm(output.cuda()[n_half:], output_ref.cuda()[n_half:])
                diff = diff_tensor_norm(output.cuda(), output_ref.cuda())
                if diff > 0.07:
                    print(f"Output norm half 1 difference: {diff_half_1:.4f}")
                    print(f"Output norm half 2 difference: {diff_half_2:.4f}")
                    print(f"Output norm difference: {diff:.4f}")
                    print(output.cuda())
                    print(output_ref.cuda())
                    assert False, f"{i} norm difference {diff:.4f} exceeds threshold"

            print("grad_w1 rel-norm:", diff_tensor_norm(w1.main_grad, w1_ref.grad))
            print("grad_w2 rel-norm:", diff_tensor_norm(w2.main_grad, w2_ref.grad))


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    TestMoELayerFP8().test_fp8_moe_forward_backward(
        num_moe_experts=64, profile=False, num_repeats=10,
    )
