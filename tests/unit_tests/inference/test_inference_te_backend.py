# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import sssglu_act
from megatron.core.inference.moe import InferenceGroupedGemmBackend
from megatron.core.transformer.moe import moe_layer as moe_layer_module
from megatron.core.transformer.moe.experts import InferenceGroupedMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig


def inference_moe_config(**kwargs):
    defaults = dict(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=2,
        moe_router_dtype="fp32",
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
    )
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


class TestInferenceBackendConfig:
    def test_defaults_to_te_with_compatibility_warning(self):
        with pytest.warns(UserWarning, match="eager compatibility path"):
            config = inference_moe_config()
        assert config.inference_grouped_gemm_backend is InferenceGroupedGemmBackend.TE

    @pytest.mark.parametrize("backend", list(InferenceGroupedGemmBackend))
    @pytest.mark.parametrize("as_string", [False, True])
    def test_accepts_backend_strings_and_enums(self, backend, as_string):
        value = backend.value if as_string else backend
        if backend is InferenceGroupedGemmBackend.TE:
            with pytest.warns(UserWarning, match="eager compatibility path"):
                config = inference_moe_config(inference_grouped_gemm_backend=value)
        else:
            config = inference_moe_config(inference_grouped_gemm_backend=value)
        assert config.inference_grouped_gemm_backend is backend

    @pytest.mark.parametrize("backend", ["auto", "invalid"])
    def test_rejects_legacy_or_invalid_backend(self, backend):
        with pytest.raises(ValueError, match="inference_grouped_gemm_backend must be"):
            inference_moe_config(inference_grouped_gemm_backend=backend)

    @pytest.mark.parametrize(
        "backend, message",
        [
            ("te", "TE inference grouped GEMM backend.*MXFP8"),
            ("flashinfer", "FlashInfer is not compatible with MXFP8"),
            ("vllm", "vLLM Triton fused MoE only supports BF16"),
        ],
    )
    def test_rejects_unsupported_mxfp8_backend(self, backend, message):
        with pytest.raises(ValueError, match=message):
            inference_moe_config(
                inference_grouped_gemm_backend=backend,
                fp8="e4m3",
                fp8_recipe="mxfp8",
                fp8_param=True,
            )

    def test_te_rejects_local_cuda_graphs(self):
        with pytest.raises(
            ValueError, match="TE inference grouped GEMM backend.*local CUDA graphs"
        ):
            inference_moe_config(cuda_graph_impl="local")

    @pytest.mark.parametrize(
        "transformer_impl, num_moe_experts",
        [("transformer_engine", None), ("transformer_engine", 2), ("inference_optimized", None)],
    )
    def test_warning_is_limited_to_inference_optimized_moe(self, transformer_impl, num_moe_experts):
        with warnings.catch_warnings(record=True) as warning_record:
            warnings.simplefilter("always")
            TransformerConfig(
                num_layers=1,
                hidden_size=8,
                num_attention_heads=2,
                transformer_impl=transformer_impl,
                num_moe_experts=num_moe_experts,
                normalization="RMSNorm",
                add_bias_linear=False,
            )
        assert not any(
            "TE inference grouped GEMM backend" in str(item.message) for item in warning_record
        )


def make_te_mlp(monkeypatch, *, activation_func=F.silu, gated_linear_unit=True):
    config = SimpleNamespace(
        inference_grouped_gemm_backend=InferenceGroupedGemmBackend.TE,
        activation_func=activation_func,
        gated_linear_unit=gated_linear_unit,
        inference_moe_token_dispatcher_type="nccl",
    )

    def initialize_parent(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.training = True

    monkeypatch.setattr(TEGroupedMLP, "__init__", initialize_parent)
    return InferenceGroupedMLP(1, config, submodules=None), config


@pytest.mark.parametrize("activation_func", [F.silu, F.gelu, sssglu_act])
def test_te_constructor_skips_optimized_activation_mapping(monkeypatch, activation_func):
    mlp, _ = make_te_mlp(monkeypatch, activation_func=activation_func)

    assert mlp.inference_grouped_gemm_backend is InferenceGroupedGemmBackend.TE
    assert not hasattr(mlp, "_mcore_activation_type")
    assert not hasattr(mlp, "_flashinfer_activation_type")


def test_te_eval_forward_delegates_to_parent_before_building_weights(monkeypatch):
    mlp, _ = make_te_mlp(monkeypatch)
    expected = (torch.tensor([3]), None)
    calls = []

    def parent_forward(self, hidden_states, tokens_per_expert, probs):
        calls.append((hidden_states, tokens_per_expert, probs))
        return expected

    monkeypatch.setattr(TEGroupedMLP, "forward", parent_forward)
    mlp.eval()
    hidden_states, token_counts, probs = torch.ones(2, 3), torch.tensor([2]), torch.ones(2, 1)

    result = mlp(hidden_states, token_counts, probs, routing_map=torch.tensor([[0], [0]]))

    assert result is expected
    assert calls == [(hidden_states, token_counts, probs)]
    assert not mlp._concatenated_weights_built


def test_te_moe_does_not_create_optimized_dispatcher(monkeypatch):
    """Exercise the constructor branch without initializing distributed process groups."""
    config = SimpleNamespace(
        transformer_impl="inference_optimized",
        inference_grouped_gemm_backend=InferenceGroupedGemmBackend.TE,
        moe_token_dispatcher_type="alltoall",
        moe_shared_expert_intermediate_size=None,
        moe_shared_expert_overlap=False,
        recompute_granularity=None,
        recompute_modules=[],
        cuda_graph_impl="none",
        num_moe_experts=1,
        moe_latent_size=None,
    )
    pg_collection = SimpleNamespace(ep=object(), tp=object())

    def initialize_base(self, config, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.num_local_experts = 1
        self.local_expert_indices = [0]
        self.shared_expert_overlap = False
        self.use_shared_expert = False

    monkeypatch.setattr(moe_layer_module, "get_default_pg_collection", lambda: pg_collection)
    monkeypatch.setattr(moe_layer_module.BaseMoELayer, "__init__", initialize_base)
    monkeypatch.setattr(moe_layer_module.utils, "get_pg_size", lambda group: 1)
    monkeypatch.setattr(moe_layer_module.utils, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(moe_layer_module, "MoEAlltoAllTokenDispatcher", lambda *a, **k: object())
    monkeypatch.setattr(moe_layer_module, "MoECudaGraphTensorStore", lambda: object())
    submodules = SimpleNamespace(
        router=lambda **kwargs: torch.nn.Identity(),
        experts=lambda *args, **kwargs: torch.nn.Identity(),
        shared_experts=None,
    )

    layer = MoELayer(config, submodules=submodules, pg_collection=pg_collection)

    assert not hasattr(layer, "_inference_token_dispatcher")
    assert not hasattr(layer, "_training_token_dispatcher")
    dispatcher = layer.token_dispatcher
    assert layer.eval() is layer
    assert layer.token_dispatcher is dispatcher
    assert layer.train() is layer
    assert layer.token_dispatcher is dispatcher


def test_eval_train_swaps_and_restores_dispatcher_and_overlap():
    layer = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    training_dispatcher, inference_dispatcher = object(), object()
    layer.config = SimpleNamespace(moe_shared_expert_overlap=True)
    layer._training_token_dispatcher = training_dispatcher
    layer._inference_token_dispatcher = inference_dispatcher
    layer.token_dispatcher = training_dispatcher
    layer.shared_expert_overlap = True

    assert layer.eval() is layer
    assert layer.token_dispatcher is inference_dispatcher
    assert layer.shared_expert_overlap is False
    assert layer.train() is layer
    assert layer.token_dispatcher is training_dispatcher
    assert layer.shared_expert_overlap is True


def test_eval_routed_experts_compute_passes_dispatcher_routing_map(monkeypatch):
    layer = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    routing_map = torch.tensor([[1]])
    calls = []

    class Dispatcher:
        def dispatch_postprocess(self, hidden_states, probs):
            return hidden_states, torch.tensor([1]), probs

        def combine_preprocess(self, output):
            return output

    class Experts:
        def __call__(self, hidden_states, counts, probs, **kwargs):
            calls.append(kwargs)
            return hidden_states, None

    layer.eval()
    layer._inference_token_dispatcher = object()
    layer.token_dispatcher = Dispatcher()
    layer.token_dispatcher.routing_map = routing_map
    layer.experts = Experts()
    monkeypatch.setattr(moe_layer_module, "apply_module", lambda module: module)

    layer.routed_experts_compute(torch.ones(1, 2), torch.ones(1, 1))

    assert calls == [{"routing_map": routing_map}]


def test_train_routed_experts_compute_keeps_standard_expert_call(monkeypatch):
    layer = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    calls = []

    class Dispatcher:
        def dispatch_postprocess(self, hidden_states, probs):
            return hidden_states, torch.tensor([1]), probs

        def combine_preprocess(self, output):
            return output

    class Experts:
        def __call__(self, hidden_states, counts, probs, **kwargs):
            calls.append(kwargs)
            return hidden_states, None

    layer.train()
    layer._inference_token_dispatcher = object()
    layer.token_dispatcher = Dispatcher()
    layer.experts = Experts()
    monkeypatch.setattr(moe_layer_module, "apply_module", lambda module: module)

    layer.routed_experts_compute(torch.ones(1, 2), torch.ones(1, 1))

    assert calls == [{}]
