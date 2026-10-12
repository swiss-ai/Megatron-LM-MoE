# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import Mock

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.moe import InferenceGroupedGemmBackend
from megatron.core.transformer.moe.token_dispatcher_inference import (
    InferenceAllGatherDispatcherBase,
    NCCLAllGatherDispatcher,
    NVLSAllGatherVDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def expert_parallel_group():
    if not torch.cuda.is_available():
        pytest.skip("DynamicInferenceContext requires CUDA allocations")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=Utils.world_size,
    )
    try:
        yield
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("dispatcher_type", ["nccl", "nvls"])
def test_te_context_does_not_allocate_optimized_dispatcher_buffers(
    expert_parallel_group, monkeypatch, dispatcher_type
):
    nccl_allocate = Mock(side_effect=AssertionError("TE must not allocate NCCL buffers"))
    nvls_allocate = Mock(side_effect=AssertionError("TE must not allocate NVLS buffers"))
    scalar_allocate = Mock(
        side_effect=AssertionError("TE must not allocate optimized token counts")
    )
    monkeypatch.setattr(NCCLAllGatherDispatcher, "allocate_buffers", nccl_allocate)
    monkeypatch.setattr(NVLSAllGatherVDispatcher, "allocate_buffers", nvls_allocate)
    monkeypatch.setattr(
        InferenceAllGatherDispatcherBase, "allocate_valid_tokens_tensor", scalar_allocate
    )

    with pytest.warns(UserWarning, match="eager compatibility path"):
        model_config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            num_moe_experts=8,
            moe_router_dtype="fp32",
            transformer_impl="inference_optimized",
            inference_grouped_gemm_backend="te",
            inference_moe_token_dispatcher_type=dispatcher_type,
            cuda_graph_impl="none",
            normalization="RMSNorm",
            add_bias_linear=False,
        )

    context = DynamicInferenceContext(
        model_config=model_config,
        inference_config=InferenceConfig(
            max_sequence_length=128,
            max_requests=2,
            max_tokens=8,
            buffer_size_gb=0.01,
            block_size_tokens=16,
            unified_memory_level=0,
            num_cuda_graphs=0,
            use_cuda_graphs_for_non_decode_steps=False,
        ),
    )

    assert context.inference_grouped_gemm_backend is InferenceGroupedGemmBackend.TE
    nccl_allocate.assert_not_called()
    nvls_allocate.assert_not_called()
    scalar_allocate.assert_not_called()
