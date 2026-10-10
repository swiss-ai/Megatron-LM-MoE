# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.moe import InferenceGroupedGemmBackend
from megatron.core.models.gpt.moe_module_specs import get_inference_optimized_moe_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.token_dispatcher_inference import (
    InferenceAllGatherDispatcherBase,
    NCCLAllGatherDispatcher,
)
from tests.unit_tests.inference.test_moe_inference import _make_base_config
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def expert_parallel_group():
    if not torch.cuda.is_available():
        pytest.skip("real MoE integration requires CUDA")
    if Utils.world_size not in (1, 4):
        pytest.skip("run the integration suite with one or four EP ranks")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=Utils.world_size,
    )
    try:
        yield
    finally:
        NCCLAllGatherDispatcher._use_allgather_v = False
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("pattern", ["equal", "variable"])
@pytest.mark.parametrize("dispatcher_type", ["nccl", "nvls"] if Utils.world_size == 1 else ["nccl"])
def test_vllm_moe_uses_real_dispatcher_metadata_and_matches_te(
    expert_parallel_group, pattern, dispatcher_type
):
    """Exercise the real model/dispatcher contract, including the EP=1 local path."""
    model_parallel_cuda_manual_seed(123, inference_rng_tracker=True, force_reset_rng=True)
    common = dict(
        expert_model_parallel_size=Utils.world_size,
        inference_moe_token_dispatcher_type=dispatcher_type,
        cuda_graph_impl="none",
        inference_cuda_graph_scope=InferenceCudaGraphScope.none,
    )
    config = _make_base_config(inference_grouped_gemm_backend="vllm", **common)
    with pytest.warns(UserWarning, match="eager compatibility path"):
        baseline_config = _make_base_config(inference_grouped_gemm_backend="te", **common)
    context = DynamicInferenceContext(
        model_config=config,
        inference_config=InferenceConfig(
            max_sequence_length=128,
            max_requests=4,
            max_tokens=32,
            buffer_size_gb=0.01,
            block_size_tokens=16,
            unified_memory_level=0,
            num_cuda_graphs=0,
            use_cuda_graphs_for_non_decode_steps=False,
        ),
    )
    assert context.inference_grouped_gemm_backend is InferenceGroupedGemmBackend.VLLM
    spec = get_inference_optimized_moe_spec()
    layer = (
        MoELayer(
            config,
            submodules=spec.submodules,
            layer_number=1,
            pg_collection=get_default_pg_collection(),
        )
        .cuda()
        .eval()
    )
    baseline = (
        MoELayer(
            baseline_config,
            submodules=spec.submodules,
            layer_number=1,
            pg_collection=get_default_pg_collection(),
        )
        .cuda()
        .eval()
    )
    baseline.load_state_dict(layer.state_dict())
    valid_tokens = InferenceAllGatherDispatcherBase._valid_tokens()
    assert valid_tokens is not None
    valid_tokens_pointer = valid_tokens.data_ptr()
    if dispatcher_type == "nvls":
        # The upstream EP=1 path initializes metadata, not symmetric collective buffers.
        assert Utils.world_size == 1
        assert layer.token_dispatcher._get_rsv_tensor() is None
    NCCLAllGatherDispatcher._use_allgather_v = pattern != "equal"

    for step in range(2):
        if pattern == "equal":
            counts = [4 + step] * Utils.world_size
        else:
            counts = [3 + 2 * rank + step for rank in range(Utils.world_size)]
        torch.manual_seed(42 + Utils.rank + step)
        inputs = torch.randn(
            counts[Utils.rank], 1, config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        with torch.inference_mode():
            output, bias = layer(inputs)
            expected, expected_bias = baseline(inputs)
        assert bias is None and expected_bias is None
        assert output.shape == inputs.shape
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()
        assert valid_tokens.data_ptr() == valid_tokens_pointer
        assert InferenceAllGatherDispatcherBase._valid_tokens() is valid_tokens
        assert valid_tokens.item() == sum(counts)
        assert InferenceAllGatherDispatcherBase._get_host_valid_tokens_estimate() == sum(counts)
        torch.testing.assert_close(output.float(), expected.float(), rtol=0.03, atol=0.001)
