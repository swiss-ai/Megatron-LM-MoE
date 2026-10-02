# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    get_transformer_layer_offset,
)
from tests.unit_tests.test_utilities import Utils


class TestParallelTransformerLayer:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.parallel_transformer_layer = TransformerLayer(
            transformer_config, get_gpt_layer_with_transformer_engine_submodules()
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        parallel_transformer_layer = self.parallel_transformer_layer
        assert isinstance(parallel_transformer_layer, TransformerLayer)
        assert parallel_transformer_layer.layer_number == 1

        num_weights = sum([p.numel() for p in parallel_transformer_layer.parameters()])
        assert num_weights == 1884

    def test_gpu_forward(self):
        parallel_transformer_layer = self.parallel_transformer_layer
        config: TransformerConfig = parallel_transformer_layer.config
        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_layer.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states, context = parallel_transformer_layer(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def test_chunked_mlp(self):
        with torch.no_grad():

            def test(
                num_layers,
                hidden_size,
                num_attention_heads,
                mlp_chunks_for_prefill,
                hidden_states,
                inference_context,
            ):

                transformer_config = TransformerConfig(
                    num_layers=2,
                    hidden_size=12,
                    num_attention_heads=4,
                    mlp_chunks_for_prefill=4,
                    add_bias_linear=True,
                    use_cpu_initialization=True,
                )
                parallel_transformer_layer = TransformerLayer(
                    transformer_config, get_gpt_layer_with_transformer_engine_submodules()
                )

                parallel_transformer_layer.cuda()

                hidden_states, context = parallel_transformer_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    inference_context=inference_context,
                )

                return hidden_states, context

            num_layers = 2
            hidden_size = 12
            num_attention_heads = 4

            sequence_length = 32
            micro_batch_size = 2

            # [sequence length, batch size, hidden size]
            input_hidden_states = torch.ones((sequence_length, micro_batch_size, hidden_size))
            input_hidden_states = input_hidden_states.cuda()

            attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

            inference_context = StaticInferenceContext(
                max_batch_size=micro_batch_size, max_sequence_length=sequence_length
            )

            outputs = {}

            for mlp_chunks_for_prefill in [1, 4]:
                hidden_states, context = test(
                    num_layers,
                    hidden_size,
                    num_attention_heads,
                    mlp_chunks_for_prefill,
                    input_hidden_states,
                    inference_context,
                )
                assert hidden_states.shape[0] == sequence_length
                assert hidden_states.shape[1] == micro_batch_size
                assert hidden_states.shape[2] == hidden_size

                outputs[mlp_chunks_for_prefill] = (hidden_states, context)

        assert torch.equal(outputs[1][0], outputs[4][0])

    def test_get_layer_offset(self):
        config = self.parallel_transformer_layer.config
        assert get_transformer_layer_offset(config) == 0

    @pytest.mark.parametrize(
        "config_params,expected_offsets",
        [
            # Test case 1: Both first and last stages set (30 layers: 8+6+6+10)
            (
                {
                    "num_layers": 30,
                    "pipeline_model_parallel_size": 4,
                    "virtual_pipeline_model_parallel_size": 2,
                    "num_layers_in_first_pipeline_stage": 8,
                    "num_layers_in_last_pipeline_stage": 10,
                    "pipeline_dtype": torch.bfloat16,
                },
                {
                    (0, 0): 0,  # Stage 0, VP 0: layers 0-3
                    (0, 1): 15,  # Stage 0, VP 1: layers 15-18
                    (1, 0): 4,  # Stage 1, VP 0: layers 4-6
                    (1, 1): 19,  # Stage 1, VP 1: layers 19-21
                    (2, 0): 7,  # Stage 2, VP 0: layers 7-9
                    (2, 1): 22,  # Stage 2, VP 1: layers 22-24
                    (3, 0): 10,  # Stage 3, VP 0: layers 10-14
                    (3, 1): 25,  # Stage 3, VP 1: layers 25-29
                },
            ),
            # Test case 2: Only first stage set (26 layers: 8+6+6+6)
            (
                {
                    "num_layers": 26,
                    "pipeline_model_parallel_size": 4,
                    "virtual_pipeline_model_parallel_size": 2,
                    "num_layers_in_first_pipeline_stage": 8,
                    "num_layers_in_last_pipeline_stage": None,
                    "pipeline_dtype": torch.bfloat16,
                },
                {
                    (0, 0): 0,  # Stage 0, VP 0: layers 0-3
                    (0, 1): 13,  # Stage 0, VP 1: layers 13-16
                    (1, 0): 4,  # Stage 1, VP 0: layers 4-6
                    (1, 1): 17,  # Stage 1, VP 1: layers 17-19
                    (2, 0): 7,  # Stage 2, VP 0: layers 7-9
                    (2, 1): 20,  # Stage 2, VP 1: layers 20-22
                    (3, 0): 10,  # Stage 3, VP 0: layers 10-12
                    (3, 1): 23,  # Stage 3, VP 1: layers 23-25
                },
            ),
            # Test case 3: Only last stage set (26 layers: 6+6+6+8)
            (
                {
                    "num_layers": 26,
                    "pipeline_model_parallel_size": 4,
                    "virtual_pipeline_model_parallel_size": 2,
                    "num_layers_in_first_pipeline_stage": None,
                    "num_layers_in_last_pipeline_stage": 8,
                    "pipeline_dtype": torch.bfloat16,
                },
                {
                    (0, 0): 0,  # Stage 0, VP 0: layers 0-2
                    (0, 1): 13,  # Stage 0, VP 1: layers 13-15
                    (1, 0): 3,  # Stage 1, VP 0: layers 3-5
                    (1, 1): 16,  # Stage 1, VP 1: layers 16-18
                    (2, 0): 6,  # Stage 2, VP 0: layers 6-8
                    (2, 1): 19,  # Stage 2, VP 1: layers 19-21
                    (3, 0): 9,  # Stage 3, VP 0: layers 9-12
                    (3, 1): 22,  # Stage 3, VP 1: layers 22-25
                },
            ),
            # Test case 4: Even distribution (24 layers: 6+6+6+6)
            (
                {
                    "num_layers": 24,
                    "pipeline_model_parallel_size": 4,
                    "virtual_pipeline_model_parallel_size": 2,
                    "num_layers_in_first_pipeline_stage": None,
                    "num_layers_in_last_pipeline_stage": None,
                    "pipeline_dtype": torch.bfloat16,
                },
                {
                    (0, 0): 0,  # Stage 0, VP 0: layers 0-2
                    (0, 1): 12,  # Stage 0, VP 1: layers 12-14
                    (1, 0): 3,  # Stage 1, VP 0: layers 3-5
                    (1, 1): 15,  # Stage 1, VP 1: layers 15-17
                    (2, 0): 6,  # Stage 2, VP 0: layers 6-8
                    (2, 1): 18,  # Stage 2, VP 1: layers 18-20
                    (3, 0): 9,  # Stage 3, VP 0: layers 9-11
                    (3, 1): 21,  # Stage 3, VP 1: layers 21-23
                },
            ),
        ],
    )
    def test_get_layer_offset_parametrized(self, config_params, expected_offsets):
        """
        Parametrized test for get_transformer_layer_offset with different configurations.
        Tests various combinations of first/last stage settings and virtual pipeline sizes.

        This test verifies that the layer offset calculation correctly handles:
        - Asymmetric pipeline stages (different layer counts per stage)
        - Virtual pipeline parallelism (splitting physical stages into virtual stages)
        - Various combinations of first/last stage configurations

        The expected_offsets dictionary maps (pipeline_rank, vp_stage) tuples to
        the expected starting layer index for that stage combination.
        """

        config = TransformerConfig(
            hidden_size=512, num_attention_heads=8, use_cpu_initialization=True, **config_params
        )

        for (pipeline_rank, vp_stage), expected_offset in expected_offsets.items():
            original_get_pipeline_rank = parallel_state.get_pipeline_model_parallel_rank
            parallel_state.set_pipeline_model_parallel_rank(pipeline_rank)

            try:
                actual_offset = get_transformer_layer_offset(config, vp_stage)
                assert actual_offset == expected_offset, (
                    f"Expected offset {expected_offset} for pipeline rank {pipeline_rank}, "
                    f"VP stage {vp_stage}, but got {actual_offset}"
                )
            finally:
                parallel_state.set_pipeline_model_parallel_rank(original_get_pipeline_rank)

    @pytest.mark.parametrize('order', ['tp-pp-dp', 'tp-dp-pp'])
    @pytest.mark.parametrize('tp_pp', [(4, 2), (1, 1), (8, 1), (2, 2)])
    def test_sharded_state_dict(self, tp_pp, order):
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(*tp_pp, order=order)

        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=128, num_attention_heads=8, use_cpu_initialization=True
        )
        parallel_transformer_layer = TransformerLayer(
            transformer_config, get_gpt_layer_with_transformer_engine_submodules()
        )

        sharded_state_dict = parallel_transformer_layer.sharded_state_dict()

        extra_states = {k: v for k, v in sharded_state_dict.items() if k.endswith('extra_state')}
        sharded_tensors = {
            k: v for k, v in sharded_state_dict.items() if not k.endswith('extra_state')
        }
        assert all(isinstance(t, ShardedObject) for t in extra_states.values())
        assert all(isinstance(t, ShardedTensor) for t in sharded_tensors.values())

        # Test all local shapes
        tensor_local_shapes = {k: v.local_shape for k, v in sharded_tensors.items()}
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        assert tensor_local_shapes == get_tensor_shapes_for_tp(transformer_config, tp_size)

        # Test all global shapes. Prepend num layers in front of expected shapes
        tensor_global_shapes = {k: v.global_shape for k, v in sharded_tensors.items()}
        expected_global_shapes = get_tensor_shapes_for_tp(transformer_config, 1)
        assert tensor_global_shapes == expected_global_shapes

        # Test ShardedTensor keys
        for state_dict_key, sh_ten in sharded_tensors.items():
            assert state_dict_key == sh_ten.key

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(1, 1)


def get_tensor_shapes_for_tp(transformer_config, tp_size):
    hs = transformer_config.hidden_size
    return {
        'mlp.linear_fc1.layer_norm_weight': (hs,),
        'mlp.linear_fc1.layer_norm_bias': (hs,),
        'mlp.linear_fc1.weight': (hs * 4 // tp_size, hs),
        'mlp.linear_fc1.bias': (hs * 4 // tp_size,),
        'mlp.linear_fc2.weight': (hs, hs * 4 // tp_size),
        'mlp.linear_fc2.bias': (hs,),
        'self_attention.linear_proj.weight': (hs, hs // tp_size),
        'self_attention.linear_proj.bias': (hs,),
        'self_attention.linear_qkv.layer_norm_weight': (hs,),
        'self_attention.linear_qkv.layer_norm_bias': (hs,),
        'self_attention.linear_qkv.weight': (hs * 3 // tp_size, hs),
        'self_attention.linear_qkv.bias': (hs * 3 // tp_size,),
    }


def test_apply_post_norm_sandwich_norm_math():
    """Unit-test the core sandwich-norm hook (CPU only, no model parallel / TE required).

    When enabled, the hook must fold the bias into the sublayer output, normalize the result,
    and drop the bias (so the downstream bias-dropout-add only adds the residual). When disabled
    it must return the ``(output, bias)`` pair unchanged so the fused bias-dropout-add is intact.
    """
    torch.manual_seed(0)
    hidden = 8
    output = torch.randn(4, 2, hidden)
    bias = torch.randn(hidden)
    norm = torch.nn.LayerNorm(hidden)
    norm.weight.data.normal_()
    norm.bias.data.normal_()

    # Disabled (IdentityOp): the (output, bias) pair is returned unchanged.
    out_disabled, bias_disabled = TransformerLayer._apply_post_norm((output, bias), IdentityOp())
    assert out_disabled is output
    assert bias_disabled is bias

    # Enabled with a bias: bias is folded in, output is normalized, bias is dropped.
    out_enabled, bias_enabled = TransformerLayer._apply_post_norm((output, bias), norm)
    assert bias_enabled is None
    torch.testing.assert_close(out_enabled, norm(output + bias))

    # Enabled without a bias: just normalize the output.
    out_no_bias, bias_no_bias = TransformerLayer._apply_post_norm((output, None), norm)
    assert bias_no_bias is None
    torch.testing.assert_close(out_no_bias, norm(output))


class TestSandwichNormTransformerLayer:
    """Integration tests for the --sandwich-norm option (requires GPU + Transformer Engine)."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _build_layer(sandwich_norm):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            sandwich_norm=sandwich_norm,
            use_cpu_initialization=True,
        )
        return TransformerLayer(
            config,
            get_gpt_layer_with_transformer_engine_submodules(sandwich_norm=sandwich_norm),
        )

    def test_post_norms_wired(self):
        # Disabled: both post-norms are no-ops.
        off = self._build_layer(sandwich_norm=False)
        assert isinstance(off.post_self_attn_layernorm, IdentityOp)
        assert isinstance(off.post_mlp_layernorm, IdentityOp)

        # Enabled: both post-norms are real normalization modules with their own parameters.
        on = self._build_layer(sandwich_norm=True)
        assert not isinstance(on.post_self_attn_layernorm, IdentityOp)
        assert not isinstance(on.post_mlp_layernorm, IdentityOp)
        assert sum(p.numel() for p in on.parameters()) > sum(p.numel() for p in off.parameters())

    @pytest.mark.parametrize("sandwich_norm", [False, True])
    def test_gpu_forward(self, sandwich_norm):
        layer = self._build_layer(sandwich_norm=sandwich_norm).cuda()
        config = layer.config
        sequence_length, micro_batch_size = 32, 2
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size)).cuda()
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        output, _ = layer(hidden_states=hidden_states, attention_mask=attention_mask)
        assert output.shape == (sequence_length, micro_batch_size, config.hidden_size)

    def test_sandwich_norm_changes_output(self):
        # Sandwich norm normalizes each sublayer's contribution, so the layer output must differ
        # from the standard pre-norm layer given identical inputs and seeded weights.
        sequence_length, micro_batch_size, hidden = 32, 2, 12
        hidden_states = torch.rand((sequence_length, micro_batch_size, hidden)).cuda()
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        model_parallel_cuda_manual_seed(123)
        baseline = self._build_layer(sandwich_norm=False).cuda()
        out_baseline, _ = baseline(hidden_states=hidden_states, attention_mask=attention_mask)

        model_parallel_cuda_manual_seed(123)
        sandwiched = self._build_layer(sandwich_norm=True).cuda()
        out_sandwiched, _ = sandwiched(hidden_states=hidden_states, attention_mask=attention_mask)

        assert not torch.allclose(out_baseline, out_sandwiched)


def test_freeze_norm_gain_at_identity():
    """Unit-test the helper behind --fixed-pre-norm-gain (CPU only, no model parallel / TE)."""
    from megatron.core.transformer.utils import freeze_norm_gain_at_identity

    norm = torch.nn.LayerNorm(8)
    with torch.no_grad():
        norm.weight.fill_(2.0)
        norm.bias.fill_(0.5)
    assert freeze_norm_gain_at_identity(norm, zero_centered_gamma=False)
    assert torch.all(norm.weight == 1) and torch.all(norm.bias == 0)
    assert not norm.weight.requires_grad and not norm.bias.requires_grad

    # The norm is now the gain-free formula.
    x = torch.randn(4, 8)
    torch.testing.assert_close(norm(x), torch.nn.functional.layer_norm(x, (8,), eps=norm.eps))

    # The gain is a constant: a value loaded from a checkpoint is overridden with the identity.
    norm.load_state_dict({"weight": torch.full((8,), 3.0), "bias": torch.full((8,), 0.25)})
    assert torch.all(norm.weight == 1) and torch.all(norm.bias == 0)

    # Fused norm + linear (prefixed attributes) with a zero-centered gain: identity is 0, and the
    # linear weight stays trainable.
    class FusedNormLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm_weight = torch.nn.Parameter(torch.full((8,), 0.7))
            self.weight = torch.nn.Parameter(torch.randn(8, 8))

    fused = FusedNormLinear()
    assert freeze_norm_gain_at_identity(fused, zero_centered_gamma=True, attr_prefix="layer_norm_")
    assert torch.all(fused.layer_norm_weight == 0) and not fused.layer_norm_weight.requires_grad
    assert fused.weight.requires_grad

    # Nothing to pin on a module without a gain.
    assert not freeze_norm_gain_at_identity(IdentityOp(), zero_centered_gamma=False)


class TestFixedPreNormGainTransformerLayer:
    """Integration tests for the --fixed-pre-norm-gain option (requires GPU + Transformer Engine).

    Layers are built with RMSNorm, QK norms and sandwich norms so that every kind of norm is
    present: the pre-norms must lose their gain, all other norms must keep theirs.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _build_layer(fixed_pre_norm_gain, num_experts=None):
        moe_kwargs = {}
        if num_experts:
            moe_kwargs = dict(
                num_moe_experts=num_experts,
                moe_router_topk=2,
                moe_router_violation_metrics=[],
                add_bias_linear=False,
            )
        config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            normalization="RMSNorm",
            qk_layernorm=True,
            sandwich_norm=True,
            fixed_pre_norm_gain=fixed_pre_norm_gain,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
            **moe_kwargs,
        )
        submodules = get_gpt_layer_with_transformer_engine_submodules(
            num_experts=num_experts, qk_layernorm=True, sandwich_norm=True
        )
        return TransformerLayer(config, submodules)

    @staticmethod
    def _pre_norm_gains(layer):
        """The pre-norm gains of a layer: fused into linear_qkv, and fused into the dense
        linear_fc1 or standalone before the MoE."""
        linear_qkv = layer.self_attention.linear_qkv
        gains = {"self_attention.linear_qkv.layer_norm_weight": linear_qkv.layer_norm_weight}
        if isinstance(layer.pre_mlp_layernorm, IdentityOp):
            gains["mlp.linear_fc1.layer_norm_weight"] = layer.mlp.linear_fc1.layer_norm_weight
        else:
            gains["pre_mlp_layernorm.weight"] = layer.pre_mlp_layernorm.weight
        return gains

    @staticmethod
    def _other_norm_gains(layer):
        return {
            "self_attention.q_layernorm.weight": layer.self_attention.q_layernorm.weight,
            "self_attention.k_layernorm.weight": layer.self_attention.k_layernorm.weight,
            "post_self_attn_layernorm.weight": layer.post_self_attn_layernorm.weight,
            "post_mlp_layernorm.weight": layer.post_mlp_layernorm.weight,
        }

    @pytest.mark.parametrize("num_experts", [None, 2])
    def test_default_keeps_gains_trainable(self, num_experts):
        layer = self._build_layer(fixed_pre_norm_gain=False, num_experts=num_experts)
        for gain in {**self._pre_norm_gains(layer), **self._other_norm_gains(layer)}.values():
            assert gain.requires_grad

    @pytest.mark.parametrize("num_experts", [None, 2])
    def test_pre_norm_gains_pinned(self, num_experts):
        layer = self._build_layer(fixed_pre_norm_gain=True, num_experts=num_experts)
        for name, gain in self._pre_norm_gains(layer).items():
            assert not gain.requires_grad, name
            assert torch.all(gain == 1), name
        for name, gain in self._other_norm_gains(layer).items():
            assert gain.requires_grad, name

        # Same parameters, same names: the checkpoint layout is unchanged.
        reference = self._build_layer(fixed_pre_norm_gain=False, num_experts=num_experts)
        assert [n for n, _ in layer.named_parameters()] == [
            n for n, _ in reference.named_parameters()
        ]

    @pytest.mark.parametrize("num_experts", [None, 2])
    def test_matches_unit_gain_layer(self, num_experts):
        # A layer with pinned pre-norm gains computes the same function as the standard layer
        # with the same weights and unit pre-norm gains, and the pinned gains receive no gradient.
        sequence_length, micro_batch_size, hidden = 32, 2, 12
        hidden_states = torch.rand((sequence_length, micro_batch_size, hidden)).cuda()
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        reference = self._build_layer(fixed_pre_norm_gain=False, num_experts=num_experts).cuda()
        with torch.no_grad():
            for gain in self._pre_norm_gains(reference).values():
                gain.fill_(1.0)
        out_reference, _ = reference(hidden_states=hidden_states, attention_mask=attention_mask)

        layer = self._build_layer(fixed_pre_norm_gain=True, num_experts=num_experts).cuda()
        layer.load_state_dict(reference.state_dict())
        output, _ = layer(hidden_states=hidden_states, attention_mask=attention_mask)
        torch.testing.assert_close(output, out_reference)

        output.float().sum().backward()
        for name, gain in self._pre_norm_gains(layer).items():
            assert gain.grad is None, name
        for name, gain in self._other_norm_gains(layer).items():
            assert gain.grad is not None, name

    def test_loaded_gain_is_overridden(self):
        layer = self._build_layer(fixed_pre_norm_gain=True)
        state_dict = layer.state_dict()
        for name in self._pre_norm_gains(layer):
            state_dict[name] = torch.full_like(state_dict[name], 2.0)
        layer.load_state_dict(state_dict)
        for name, gain in self._pre_norm_gains(layer).items():
            assert torch.all(gain == 1), name


def _have_kda_kernels():
    try:
        from megatron.core.ssm.kimi_delta_attention import HAVE_KDA
    except ImportError:
        return False
    return HAVE_KDA


@pytest.mark.skipif(
    not _have_kda_kernels(), reason="The installed FLA does not provide KDA kernels."
)
class TestFixedPreNormGainHybridBlock:
    """--fixed-pre-norm-gain on the hybrid block used by the KDA / latent-MoE runs: KDA layers
    (input norm fused into in_proj) interleaved with global attention layers (input norm fused
    into linear_qkv), a first dense layer (pre-MLP norm fused into linear_fc1), latent MoE layers
    with a shared expert (standalone pre-MLP norm), QK norms and sandwich norms. Construction only;
    requires GPU + Transformer Engine + the KDA kernels."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _build_layers(fixed_pre_norm_gain):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_transformer_block_with_experimental_attention_variant_spec,
        )
        from megatron.core.transformer.spec_utils import build_module

        config = TransformerConfig(
            num_layers=4,
            hidden_size=256,
            num_attention_heads=8,
            experimental_attention_variant="kda",
            linear_attention_freq=[1, 1, 1, 0],
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            attention_output_gate=True,
            qk_layernorm=True,
            num_moe_experts=2,
            moe_layer_freq=[0, 1, 1, 1],
            moe_router_topk=2,
            moe_router_violation_metrics=[],
            moe_latent_size=64,
            moe_shared_expert_intermediate_size=64,
            normalization="RMSNorm",
            sandwich_norm=True,
            fixed_pre_norm_gain=fixed_pre_norm_gain,
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            pipeline_dtype=torch.bfloat16,
            transformer_impl="transformer_engine",
            use_cpu_initialization=True,
        )
        block_spec = get_transformer_block_with_experimental_attention_variant_spec(config)
        return [
            build_module(layer_spec, config=config, layer_number=i + 1)
            for i, layer_spec in enumerate(block_spec.layer_specs)
        ]

    @staticmethod
    def _pre_norm_gains(layers):
        kda0, kda1, _, attn3 = layers
        return {
            "0.in_proj.layer_norm_weight": kda0.self_attention.in_proj.layer_norm_weight,
            "0.linear_fc1.layer_norm_weight": kda0.mlp.linear_fc1.layer_norm_weight,
            "1.in_proj.layer_norm_weight": kda1.self_attention.in_proj.layer_norm_weight,
            "1.pre_mlp_layernorm.weight": kda1.pre_mlp_layernorm.weight,
            "3.linear_qkv.layer_norm_weight": attn3.self_attention.linear_qkv.layer_norm_weight,
            "3.pre_mlp_layernorm.weight": attn3.pre_mlp_layernorm.weight,
        }

    @staticmethod
    def _other_norm_gains(layers):
        kda0, kda1, _, attn3 = layers
        return {
            "0.out_norm.weight": kda0.self_attention.out_norm.weight,
            "0.post_self_attn_layernorm.weight": kda0.post_self_attn_layernorm.weight,
            "0.post_mlp_layernorm.weight": kda0.post_mlp_layernorm.weight,
            "1.post_mlp_layernorm.weight": kda1.post_mlp_layernorm.weight,
            "3.q_layernorm.weight": attn3.self_attention.q_layernorm.weight,
            "3.k_layernorm.weight": attn3.self_attention.k_layernorm.weight,
            "3.post_self_attn_layernorm.weight": attn3.post_self_attn_layernorm.weight,
        }

    def test_layout(self):
        layers = self._build_layers(fixed_pre_norm_gain=True)
        # The patterns put KDA + dense first, KDA + MoE next, global attention + MoE last.
        assert isinstance(layers[0].pre_mlp_layernorm, IdentityOp)
        assert not isinstance(layers[1].pre_mlp_layernorm, IdentityOp)
        assert hasattr(layers[3].self_attention, "linear_qkv")
        # Experts and the shared expert have no fused norm, so nothing is pinned there.
        shared = layers[1].mlp.shared_experts
        assert not hasattr(shared.linear_fc1, "layer_norm_weight")

    def test_default_keeps_gains_trainable(self):
        layers = self._build_layers(fixed_pre_norm_gain=False)
        gains = {**self._pre_norm_gains(layers), **self._other_norm_gains(layers)}
        for name, gain in gains.items():
            assert gain.requires_grad, name

    def test_pre_norm_gains_pinned(self):
        layers = self._build_layers(fixed_pre_norm_gain=True)
        for name, gain in self._pre_norm_gains(layers).items():
            assert not gain.requires_grad, name
            assert torch.all(gain == 1), name
        for name, gain in self._other_norm_gains(layers).items():
            assert gain.requires_grad, name

        reference = self._build_layers(fixed_pre_norm_gain=False)
        for layer, ref in zip(layers, reference):
            assert [n for n, _ in layer.named_parameters()] == [
                n for n, _ in ref.named_parameters()
            ]
