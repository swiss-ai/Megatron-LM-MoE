# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import sys

import pytest
import torch

from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args


def _set_world_compatible_batch_sizes(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.micro_batch_size = 1
    args.global_batch_size = world_size
    args.eval_micro_batch_size = 1
    args.eval_global_batch_size = world_size


def _validated_args(monkeypatch, pattern):
    monkeypatch.setattr(sys, "argv", ["test_mamba_hybrid_args.py"])
    args = parse_args()
    args.hybrid_layer_pattern = pattern
    args.num_layers = len(pattern)
    args.hidden_size = 256
    args.ffn_hidden_size = 512
    args.num_attention_heads = 16
    args.max_position_embeddings = 64
    args.seq_length = 64
    args.position_embedding_type = "rope"
    args.rope_type = "rope"
    _set_world_compatible_batch_sizes(args)
    args.multi_latent_attention = False
    args.group_query_attention = False
    if "D" in pattern:
        args.q_lora_rank = 64
        args.kv_lora_rank = 64
        args.qk_head_dim = 64
        args.qk_pos_emb_head_dim = 32
        args.v_head_dim = 64
        args.dsa_indexer_n_heads = 8
        args.dsa_indexer_head_dim = 64
        args.dsa_indexer_topk = 32
        args.apply_rope_fusion = False
        args.bf16 = True
        args.params_dtype = torch.bfloat16
    return validate_args(args)


@pytest.mark.internal
def test_mamba_pattern_config_conversion_does_not_require_dsa_symbol(monkeypatch):
    args = _validated_args(monkeypatch, "M")

    config = core_transformer_config_from_args(args)

    assert isinstance(config, TransformerConfig)
    assert not isinstance(config, MLATransformerConfig)
    assert config.is_hybrid_model
    assert config.experimental_attention_variant is None


@pytest.mark.internal
def test_cli_pattern_rejects_attention_and_dsa_mixture(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test_mamba_hybrid_args.py"])
    args = parse_args()
    args.hybrid_layer_pattern = "M*D"
    args.num_layers = 3
    _set_world_compatible_batch_sizes(args)

    with pytest.raises(ValueError, match="both Attention and MLA/DSA"):
        validate_args(args)


@pytest.mark.internal
def test_dsa_pattern_infers_mla_config_before_conversion(monkeypatch):
    args = _validated_args(monkeypatch, "MDM")

    config = core_transformer_config_from_args(args)

    assert args.multi_latent_attention
    assert isinstance(config, MLATransformerConfig)
    assert config.is_hybrid_model
    assert config.experimental_attention_variant == "dsa"
