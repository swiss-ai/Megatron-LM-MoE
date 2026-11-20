# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.

import os

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.kimi_delta_attention import (
    HAVE_FUSED_RMSNORM_GATED,
    HAVE_KDA,
    FusedRMSNormGated,
    KimiDeltaAttention,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not HAVE_KDA, reason="The installed FLA does not provide KDA kernels.")
@pytest.mark.internal
class TestKimiDeltaAttentionReferenceParameterization:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            hidden_size=128,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            num_layers=1,
            normalization="RMSNorm",
            num_attention_heads=2,
            activation_func=F.silu,
            bf16=True,
            deterministic_mode=True,
            experimental_attention_variant="kda",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        pg_collection = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )
        submodules = get_experimental_attention_variant_module_spec(config=config).submodules
        cls.kda = KimiDeltaAttention(
            config,
            submodules=submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=pg_collection,
        ).cuda()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_projection_shapes_and_decay_initialization_match_reference(self):
        kda = self.kda
        assert kda.in_proj.weight.shape == (
            2 * kda.qk_dim
            + kda.v_dim
            + 2 * kda.value_head_dim
            + kda.num_value_heads,
            kda.hidden_size,
        )
        assert kda.in_proj.weight.kda_split_shapes == (
            kda.qk_dim,
            kda.qk_dim,
            kda.v_dim,
            kda.value_head_dim,
            kda.value_head_dim,
            kda.num_value_heads,
        )
        assert kda.decay_out_proj.weight.shape == (kda.alpha_dim, kda.value_head_dim)
        assert kda.gate_out_proj.weight.shape == (kda.v_dim, kda.value_head_dim)
        assert HAVE_FUSED_RMSNORM_GATED
        assert isinstance(kda.out_norm, FusedRMSNormGated)
        assert kda.out_norm.weight.shape == (kda.value_head_dim,)
        assert kda.out_norm.activation == "sigmoid"
        assert kda.out_norm.eps == kda.config.layernorm_epsilon

        assert kda.A_log.shape == (kda.num_value_heads,)
        assert kda.dt_bias.shape == (kda.num_value_heads * kda.key_head_dim,)
        assert kda.A_log.dtype == torch.float32
        assert kda.dt_bias.dtype == torch.float32
        dt = F.softplus(kda.dt_bias)
        assert torch.all(dt >= 0.001)
        assert torch.all(dt <= 0.1)

    def test_beta_and_output_gate_match_reference_equations(self):
        kda = self.kda
        batch, seq = 2, 3
        beta_logits = torch.randn(
            batch, seq, kda.num_value_heads, device="cuda", dtype=torch.bfloat16
        )
        beta = kda._activate_beta(beta_logits)
        torch.testing.assert_close(beta, beta_logits.float().sigmoid())

        core_output = torch.randn(
            batch,
            seq,
            kda.num_value_heads,
            kda.value_head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        gate = torch.randn_like(core_output)
        actual = kda._apply_gated_norm(core_output, gate)
        core_output_f32 = core_output.float()
        normalized = core_output_f32 * torch.rsqrt(
            core_output_f32.square().mean(dim=-1, keepdim=True)
            + kda.config.layernorm_epsilon
        )
        normalized = normalized * kda.out_norm.weight.float()
        expected = normalized * torch.sigmoid(gate.float())
        torch.testing.assert_close(actual, expected.to(actual.dtype))

    def test_forward_backward(self):
        hidden_states = torch.randn(
            8, 2, self.kda.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        original_gated_delta_rule = self.kda.gated_delta_rule
        kernel_kwargs = {}

        def recording_gated_delta_rule(*args, **kwargs):
            kernel_kwargs.update(kwargs)
            return original_gated_delta_rule(*args, **kwargs)

        self.kda.gated_delta_rule = recording_gated_delta_rule
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output, bias = self.kda(hidden_states, None)
        finally:
            self.kda.gated_delta_rule = original_gated_delta_rule

        assert (
            kernel_kwargs["use_qk_l2norm_in_kernel"]
            == self.kda._qk_l2norm_in_kernel
        )
        assert (
            kernel_kwargs.get("use_gate_in_kernel", False)
            == self.kda._use_fused_decay_gate
        )
        assert kernel_kwargs["g"].shape[-2:] == (
            self.kda.num_value_heads,
            self.kda.key_head_dim,
        )
        if self.kda._use_fused_decay_gate:
            assert kernel_kwargs["A_log"] is self.kda.A_log
            assert kernel_kwargs["dt_bias"].shape == self.kda.dt_bias.shape
        else:
            # Fallback path: chunk_kda doesn't take A_log/dt_bias at all, since
            # the decay activation was already applied to `g` in Python.
            assert "A_log" not in kernel_kwargs
            assert "dt_bias" not in kernel_kwargs
        assert (
            kernel_kwargs.get("use_beta_sigmoid_in_kernel", False)
            == self.kda._use_fused_beta_sigmoid
        )
        assert output.shape == hidden_states.shape
        assert bias is None
        assert torch.isfinite(output).all()
        output.float().square().mean().backward()
        assert self.kda.in_proj.weight.grad is not None
        assert self.kda.decay_out_proj.weight.grad is not None
        assert self.kda.gate_out_proj.weight.grad is not None
        assert self.kda.out_norm.weight.grad is not None
        assert self.kda.A_log.grad is not None
        assert self.kda.dt_bias.grad is not None

    def test_backward_dw_covers_all_kda_linear_projections(self, monkeypatch):
        calls = []
        projection_names = (
            "in_proj",
            "decay_out_proj",
            "gate_out_proj",
            "out_proj",
        )
        for name in projection_names:
            monkeypatch.setattr(
                getattr(self.kda, name),
                "backward_dw",
                lambda name=name: calls.append(name),
            )

        self.kda.backward_dw()

        assert calls == list(projection_names)


@pytest.mark.skipif(
    not HAVE_KDA or int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="This test requires FLA and a two-rank distributed run.",
)
@pytest.mark.parametrize(("tp_size", "cp_size"), [(2, 1), (1, 2)])
def test_kda_low_rank_projections_distributed(tp_size, cp_size):
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
    )
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            hidden_size=128,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            num_layers=1,
            normalization="RMSNorm",
            num_attention_heads=2,
            activation_func=F.silu,
            bf16=True,
            deterministic_mode=True,
            experimental_attention_variant="kda",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        pg_collection = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )
        submodules = get_experimental_attention_variant_module_spec(config=config).submodules
        kda = KimiDeltaAttention(
            config,
            submodules=submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=pg_collection,
        ).cuda()

        assert kda.in_proj.weight.shape[0] == kda.in_proj_dim // tp_size
        assert kda.decay_out_proj.weight.shape == (
            kda.alpha_dim // tp_size,
            kda.value_head_dim,
        )
        assert kda.gate_out_proj.weight.shape == (
            kda.v_dim // tp_size,
            kda.value_head_dim,
        )
        assert kda.A_log.shape == (kda.num_value_heads // tp_size,)
        assert kda.dt_bias.shape == (kda.alpha_dim // tp_size,)

        hidden_states = torch.randn(
            8, 1, kda.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output, _ = kda(hidden_states, None)
        assert output.shape == hidden_states.shape
        output.float().square().mean().backward()
        assert kda.decay_out_proj.weight.grad is not None
        assert kda.gate_out_proj.weight.grad is not None
        assert kda.A_log.grad is not None
        assert kda.dt_bias.grad is not None
    finally:
        Utils.destroy_model_parallel()
