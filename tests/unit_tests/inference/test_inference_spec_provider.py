# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.models.backends import InferenceSpecProvider, LocalSpecProvider
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.moe.experts import InferenceGroupedMLP
from megatron.core.transformer.moe.router import InferenceTopKRouter, TopKRouter


@pytest.mark.parametrize("kwargs", [{}, {"moe_use_offloading_experts": False}])
def test_inference_grouped_mlp_preserves_experts(kwargs):
    factory = InferenceSpecProvider().grouped_mlp_modules(moe_use_grouped_gemm=True, **kwargs)
    assert factory.func is InferenceGroupedMLP


def test_inference_grouped_mlp_rejects_offloading():
    with pytest.raises(NotImplementedError, match="does not support offloading experts"):
        InferenceSpecProvider().grouped_mlp_modules(
            moe_use_grouped_gemm=True, moe_use_offloading_experts=True
        )


@pytest.mark.parametrize(
    "provider,router",
    [(InferenceSpecProvider, InferenceTopKRouter), (LocalSpecProvider, TopKRouter)],
)
def test_moe_spec_selects_the_backend_router(provider, router):
    spec = get_moe_module_spec_for_backend(backend=provider(), num_experts=2)
    assert spec.submodules.router is router
