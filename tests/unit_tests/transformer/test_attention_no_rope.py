# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestParallelAttentionWithNoRope:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        # use BF16 and a large enough hidden size to enable FlashAttention
        self.transformer_config = TransformerConfig(
            num_layers=8,  # Using 8 layers to test patterns like [0,0,0,1,0,0,0,1]
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
            flash_decode=False,  # Ensure flash_decode is off to test RoPE skipping
        )
        self.parallel_attention = SelfAttention(
            self.transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_integer_no_rope_freq_pattern(self):
        """Test that integer no_rope value is correctly converted to pattern."""
        config = self.transformer_config
        config.no_rope_freq = 4  # Should convert to [0,0,0,1,0,0,0,1]
        config.__post_init__()

        # Verify the pattern conversion
        assert isinstance(config.no_rope_freq, list)
        assert len(config.no_rope_freq) == config.num_layers
        assert config.no_rope_freq == [0, 0, 0, 1, 0, 0, 0, 1]

    def test_custom_no_rope_pattern(self):
        """Test custom no_rope pattern."""
        config = self.transformer_config
        config.no_rope_freq = [0, 1, 0, 1, 0, 1, 0, 1]  # Custom pattern
        config.__post_init__()

        # Verify the pattern is preserved
        assert isinstance(config.no_rope_freq, list)
        assert len(config.no_rope_freq) == config.num_layers
        assert config.no_rope_freq == [0, 1, 0, 1, 0, 1, 0, 1]

    def test_gpu_forward_with_no_rope(self):
        """Test forward pass with no_rope pattern."""
        config = self.parallel_attention.config
        config.no_rope_freq = 4  # Use pattern [0,0,0,1,0,0,0,1]
        config.__post_init__()  # Ensure pattern is converted

        sequence_length = 32
        micro_batch_size = 1

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None

        # Create rotary position embeddings
        # Shape: [seq_len, 1, 1, kv_channels]
        rotary_pos_emb = torch.randn(
            sequence_length, 1, 1, self.parallel_attention.config.kv_channels
        ).cuda()

        # For self-attention, rotary_pos_emb needs to be a tuple of (q_pos_emb, k_pos_emb)
        rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)

        # Layer numbers are 1-indexed; layer 4 should skip RoPE.
        self.parallel_attention.layer_number = 4
        output_without_rope, _ = self.parallel_attention(
            hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb
        )

        # Layer 3 should retain RoPE.
        self.parallel_attention.layer_number = 3
        output_with_rope, bias = self.parallel_attention(
            hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb
        )

        assert not torch.allclose(
            output_without_rope, output_with_rope
        ), "Outputs are expected to be different."

        # Verify output shapes
        assert config.recompute_granularity is None
        assert output_with_rope.shape[0] == sequence_length
        assert output_with_rope.shape[1] == micro_batch_size
        assert output_with_rope.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    @pytest.mark.parametrize(
        ("no_rope_freq", "layer_number"),
        [
            pytest.param([1] * 8, 1, id="all-nope"),
            pytest.param([0, 1, 0, 1, 0, 1, 0, 1], 2, id="mixed-nope-layer"),
        ],
    )
    def test_no_rope_clears_all_rotary_inputs(self, monkeypatch, no_rope_freq, layer_number):
        """NoPE layers must clear standard and precomputed rotary inputs."""
        config = self.parallel_attention.config
        config.no_rope_freq = no_rope_freq
        config.__post_init__()
        self.parallel_attention.layer_number = layer_number
        self.parallel_attention.cuda()

        sequence_length = 8
        hidden_states = torch.randn(
            sequence_length, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        rotary = torch.randn(sequence_length, 1, 1, config.kv_channels, device="cuda")
        combined = torch.randn(sequence_length, 2 * config.kv_channels, device="cuda")
        adjusted_rotary = {}
        original_adjust = self.parallel_attention._adjust_key_value_for_inference

        def capture_adjusted_rotary(*args, **kwargs):
            adjusted_rotary["values"] = args[4:8]
            return original_adjust(*args, **kwargs)

        monkeypatch.setattr(
            self.parallel_attention, "_adjust_key_value_for_inference", capture_adjusted_rotary
        )

        self.parallel_attention(
            hidden_states,
            None,
            rotary_pos_emb=(rotary, rotary),
            rotary_pos_cos=rotary,
            rotary_pos_sin=rotary,
            rotary_pos_cos_sin=combined,
        )

        assert adjusted_rotary["values"] == (None, None, None, None)

    @pytest.mark.parametrize(
        ("no_rope_freq", "layer_number"),
        [
            pytest.param(None, 1, id="ordinary-rope"),
            pytest.param([0] * 8, 1, id="all-zero-pattern"),
            pytest.param([0, 1, 0, 1, 0, 1, 0, 1], 1, id="mixed-rope-layer"),
        ],
    )
    def test_rope_layer_preserves_fused_rotary_table(self, monkeypatch, no_rope_freq, layer_number):
        """RoPE layers must retain the combined table for fused application."""
        config = self.parallel_attention.config
        config.no_rope_freq = no_rope_freq
        config.__post_init__()
        self.parallel_attention.layer_number = layer_number
        self.parallel_attention.cuda()

        hidden_states = torch.randn(8, 1, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        combined = torch.randn(8, 2 * config.kv_channels, device="cuda")
        adjusted_rotary = {}
        original_adjust = self.parallel_attention._adjust_key_value_for_inference

        def capture_adjusted_rotary(*args, **kwargs):
            adjusted_rotary["combined"] = args[7]
            return original_adjust(*args, **kwargs)

        monkeypatch.setattr(
            self.parallel_attention, "_adjust_key_value_for_inference", capture_adjusted_rotary
        )

        self.parallel_attention(hidden_states, None, rotary_pos_cos_sin=combined)

        assert adjusted_rotary["combined"] is combined

    def test_invalid_no_rope_freq_pattern(self):
        """Test invalid no_rope patterns raise appropriate errors."""
        config = self.transformer_config

        # Test invalid integer pattern
        with pytest.raises(AssertionError):
            config.no_rope_freq = 3  # Not divisible by num_layers=8
            config.__post_init__()

        # Test invalid list pattern
        with pytest.raises(AssertionError):
            config.no_rope_freq = [0, 1, 0, 1]  # Wrong length
            config.__post_init__()

    def test_gpu_forward_no_rope_freq_not_specified(self):
        """Test forward pass with no_rope pattern not provided."""
        config = self.parallel_attention.config
        config.no_rope_freq = None
        config.__post_init__()  # Ensure pattern is converted

        sequence_length = 32
        micro_batch_size = 1

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None

        # Create rotary position embeddings
        # Shape: [seq_len, 1, 1, kv_channels]
        rotary_pos_emb = torch.randn(
            sequence_length, 1, 1, self.parallel_attention.config.kv_channels
        ).cuda()

        # For self-attention, rotary_pos_emb needs to be a tuple of (q_pos_emb, k_pos_emb)
        rotary_pos_emb = (rotary_pos_emb, rotary_pos_emb)
        # Run forward pass
        output, bias = self.parallel_attention(
            hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb
        )
        # Verify output shapes
        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    def test_checkpointed_gpu_forward(self):
        """Test checkpointed forward pass with no_rope pattern."""
        transformer_config = self.transformer_config
        transformer_config.recompute_granularity = 'selective'
        transformer_config.no_rope_freq = 4  # Use pattern [0,0,0,1,0,0,0,1]
        transformer_config.__post_init__()

        checkpointed_parallel_attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        config = checkpointed_parallel_attention.config

        sequence_length = 32
        micro_batch_size = 1

        checkpointed_parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, checkpointed_parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None
        rotary_pos_emb = torch.ones(
            sequence_length, 1, 1, checkpointed_parallel_attention.config.kv_channels
        ).cuda()

        output, bias = checkpointed_parallel_attention(
            hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb
        )

        assert config.recompute_granularity == 'selective'
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    def test_flash_decode_with_no_rope_freq(self):
        """Test that flash_decode cannot be used with no_rope."""
        config = self.transformer_config
        config.flash_decode = True
        config.no_rope_freq = 4  # Use pattern [0,0,0,1,0,0,0,1]

        # Verify that setting both flash_decode and no_rope raises an assertion error
        with pytest.raises(AssertionError, match="flash_decode cannot be used with no_rope"):
            config.__post_init__()
