# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.models.mamba.mamba_layer_specs import hybrid_stack_spec, mamba_stack_spec
from megatron.core.ssm.mamba_block import MambaStack
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import InferenceCudaGraphScope
from tests.unit_tests.test_utilities import Utils


class _Context:
    def is_static_batching(self):
        return True

    def is_decode_only(self):
        return True

    def using_cuda_graph_this_step(self):
        return True


class _Manager:
    def __init__(self):
        self.calls = []

    def __call__(self, module, args, kwargs):
        self.calls.append((module, args, kwargs))
        return (kwargs['hidden_states'] + 10,)


class _DummyHybridModel(HybridModel):
    def __init__(self, inference_scope):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(
            cuda_graph_impl='local', inference_cuda_graph_scope=inference_scope
        )
        self.cudagraph_manager = _Manager()
        self.eval()

    def forward(self, *, hidden_states, inference_context=None, **kwargs):
        self.seen_context = inference_context
        return hidden_states + 1


def test_legacy_hybrid_aliases_resolve_to_canonical_objects():
    assert MambaStack is HybridStack
    assert issubclass(MambaModel, HybridModel)
    assert mamba_stack_spec is hybrid_stack_spec


def test_inference_params_alias_is_canonicalized_before_block_graph_dispatch():
    context = _Context()
    model = _DummyHybridModel(InferenceCudaGraphScope.block)
    hidden_states = torch.tensor([2.0])

    output = model(hidden_states=hidden_states, inference_params=context)

    assert torch.equal(output, hidden_states + 10)
    assert len(model.cudagraph_manager.calls) == 1
    kwargs = model.cudagraph_manager.calls[0][2]
    assert kwargs['inference_context'] is context
    assert 'inference_params' not in kwargs


def test_both_inference_names_keep_canonical_context_for_graph_dispatch():
    canonical = _Context()
    deprecated = _Context()
    model = _DummyHybridModel(InferenceCudaGraphScope.block)

    output = model(
        hidden_states=torch.tensor([2.0]),
        inference_context=canonical,
        inference_params=deprecated,
    )

    assert torch.equal(output, torch.tensor([12.0]))
    assert model.cudagraph_manager.calls[0][2]['inference_context'] is canonical


def test_eager_and_layer_scopes_do_not_dispatch_block_manager():
    context = _Context()
    hidden_states = torch.tensor([2.0])
    for scope in (InferenceCudaGraphScope.none, InferenceCudaGraphScope.layer):
        model = _DummyHybridModel(scope)
        output = model(hidden_states=hidden_states, inference_params=context)
        assert torch.equal(output, hidden_states + 1)
        assert model.cudagraph_manager.calls == []
        assert model.seen_context is context

def _small_hybrid_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )


def test_mamba_wrapper_and_hybrid_model_have_identical_state_keys_and_shapes():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    try:
        kwargs = dict(
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M",
        )
        legacy = MambaModel(
            config=_small_hybrid_config(), mamba_stack_spec=mamba_stack_spec, **kwargs
        )
        canonical = HybridModel(
            config=_small_hybrid_config(), hybrid_stack_spec=hybrid_stack_spec, **kwargs
        )

        legacy_state = legacy.state_dict()
        canonical_state = canonical.state_dict()
        assert legacy_state.keys() == canonical_state.keys()
        def shape_or_none(value):
            return None if value is None else tuple(value.shape)

        assert {key: shape_or_none(value) for key, value in legacy_state.items()} == {
            key: shape_or_none(value) for key, value in canonical_state.items()
        }
        assert all(key.startswith(("embedding.", "decoder.", "output_layer.")) for key in legacy_state)
        assert not any("mamba" in key.lower() or "hybrid" in key.lower() for key in legacy_state)

        legacy_mixer = legacy.decoder.layers[0].mixer
        canonical_mixer = canonical.decoder.layers[0].mixer
        for name in ("in_proj", "conv1d"):
            legacy_module = getattr(legacy_mixer, name)
            canonical_module = getattr(canonical_mixer, name)
            assert legacy_module.weight.partition_sizes == canonical_module.weight.partition_sizes
    finally:
        Utils.destroy_model_parallel()


def test_mamba_wrapper_rejects_both_stack_spec_names():
    try:
        MambaModel(object(), object(), mamba_stack_spec=object())
    except ValueError as error:
        assert "both hybrid_stack_spec and mamba_stack_spec" in str(error)
    else:
        raise AssertionError("MambaModel accepted both stack spec names")
