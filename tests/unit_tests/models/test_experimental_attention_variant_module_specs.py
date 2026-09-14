# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import megatron.core.models.gpt.experimental_attention_variant_module_specs as specs


class _Norm:
    pass


class _Backend:
    @staticmethod
    def layer_norm(**kwargs):
        del kwargs
        return _Norm


def test_hybrid_keel_builds_post_layer_norms(monkeypatch):
    config = SimpleNamespace(
        num_layers=2,
        experimental_attention_variant="kda",
        num_moe_experts=None,
        normalization="RMSNorm",
        sandwich_norm=False,
        keel=True,
        pipeline_model_parallel_layout=None,
    )
    attention = SimpleNamespace(metainfo={"fuse_input_layernorm": False})
    mlp = SimpleNamespace(metainfo={"fuse_pre_mlp_layernorm": False})

    monkeypatch.setattr(specs, "_get_backend_spec_provider", lambda **_: _Backend())
    monkeypatch.setattr(specs, "get_linear_attention_pattern", lambda **_: [1, 1])
    monkeypatch.setattr(
        specs, "get_experimental_attention_variant_module_spec", lambda **_: attention
    )
    monkeypatch.setattr(specs, "_get_dense_mlp_module_spec", lambda **_: mlp)
    monkeypatch.setattr(specs, "get_transformer_layer_offset", lambda *_, **__: 0)
    monkeypatch.setattr(specs, "get_num_layers_to_build", lambda *_, **__: 2)

    block = specs.get_transformer_block_with_experimental_attention_variant_spec(config)

    for layer in block.layer_specs:
        assert layer.submodules.post_self_attn_layernorm is _Norm
        assert layer.submodules.post_mlp_layernorm is _Norm
