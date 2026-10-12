from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.inference.config import InferenceConfig, KDAInferenceStateConfig
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.disaggregation.decode_admission import admit_prefilled_decode
from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
)
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.kimi_delta_attention import HAVE_KDA, fused_recurrent_kda_fwd
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.ssm.test_kimi_delta_attention_inference import (
    TestKimiDeltaAttentionInference as KDAInferenceFixture,
)
from tests.unit_tests.test_utilities import Utils


class _HandoffState(InferenceStateHandoffMixin):
    def __init__(self, context, model):
        self.context = context
        self.controller = SimpleNamespace(inference_wrapped_model=SimpleNamespace(model=model))
        self.pg_collection = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            pp=parallel_state.get_pipeline_model_parallel_group(),
        )
        self._initialize_disaggregation_state()


def _context(kda):
    conv_shape, recurrent_shape = kda.kda_state_shapes_per_request()
    return DynamicInferenceContext(
        model_config=kda.config,
        inference_config=InferenceConfig(
            max_sequence_length=32,
            block_size_tokens=8,
            buffer_size_gb=0.01,
            paused_buffer_size_gb=0.002,
            max_tokens=64,
            max_requests=8,
            num_cuda_graphs=None,
            use_flashinfer_fused_rope=None,
            unified_memory_level=0,
            kda_inference_state_config=KDAInferenceStateConfig(
                kda_layer_map={0: 0},
                attention_layer_map={1: 0},
                conv_states_shape=conv_shape,
                recurrent_states_shape=recurrent_shape,
                conv_states_dtype=torch.bfloat16,
                recurrent_states_dtype=torch.float32,
            ),
        ),
    )


@pytest.mark.internal
@pytest.mark.skipif(
    not HAVE_KDA or fused_recurrent_kda_fwd is None, reason="KDA kernels unavailable"
)
@pytest.mark.parametrize(
    "backend,nixl_backend", [("nccl", None), ("nixl", "UCCL")], ids=["nccl", "nixl-uccl"]
)
def test_kda_handoff_preserves_recurrent_decode_and_slot_ownership(backend, nixl_backend):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        rank = dist.get_rank()
        if dist.get_world_size() % 2 or dist.get_world_size() < 2:
            pytest.skip("Requires paired prefill/decode ranks")
        model_parallel_cuda_manual_seed(731)
        kda = KDAInferenceFixture._build_kda()
        for parameter in kda.parameters():
            dist.broadcast(parameter.data, src=0)
        context = _context(kda)
        engine = _HandoffState(context, kda)
        source = rank % 2 == 0
        peer_rank = rank + 1 if source else rank - 1
        engine.setup_kv_transfer(
            "prefill" if source else "decode", backend=backend, nixl_backend=nixl_backend
        )
        request = DynamicInferenceRequest(
            request_id=peer_rank if not source else rank,
            prompt_tokens=torch.tensor([11, 12, 13, 14, 15], device="cuda"),
            sampling_params=SamplingParams(num_tokens_to_generate=2, do_kv_handoff=True),
        )
        torch.manual_seed(987)
        hidden = torch.randn(6, 1, kda.config.hidden_size, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            reference = kda(hidden, None)[0][-1]
        local_metadata = None
        expected_states = None
        pending = None
        if source:
            context.add_request(request)
            context.initialize_attention_state()
            padded = torch.zeros(
                context.padded_active_token_count,
                1,
                kda.config.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            padded[:5] = hidden[:5]
            with torch.inference_mode():
                kda(padded, None, inference_context=context)
            controller = TextGenerationController.__new__(TextGenerationController)
            controller.inference_wrapped_model = SimpleNamespace(inference_context=context)
            blocks, slots, tokens = controller._collect_finished_handoff_state(
                torch.tensor([0], device="cuda"), torch.tensor([77]), None
            )
            source_slot = slots[rank]
            assert source_slot < context.kda_dummy_state_idx
            expected_states = (
                context.kda_conv_states[:, source_slot].cpu().clone(),
                context.kda_recurrent_states[:, source_slot].cpu().clone(),
            )
            prepared = engine._prepare_handoff_metadata_batch(
                [(request, blocks[rank], source_slot)], tokens
            )
            engine._capture_handoff_meta(request, prepared[rank])
            context.release_memory_blocks_from_request_indexes(torch.tensor([0], device="cuda"))
            assert context.kda_metadata.mamba_state_free_slot_count == context.max_requests - 1
            local_metadata = request.disaggregated_params
        else:
            occupied_slot = context.kda_metadata.allocate_slot()
            assert occupied_slot is not None
            pending = engine._reserve_ssm_handoff_import()
            assert pending.live_slot < context.kda_dummy_state_idx
            local_blocks = context.kv_block_allocator.allocate_memory_blocks(1)
            assert local_blocks is not None
            local_blocks = local_blocks.tolist()
            local_metadata = engine._kv_transfer_agent.export_meta()
            local_metadata["ssm"] = {
                kind: agent.export_meta() for kind, agent in engine._ssm_transfer_agents.items()
            }
        metadata = [None] * dist.get_world_size()
        dist.all_gather_object(metadata, local_metadata)
        state_snapshots: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None
        ] * dist.get_world_size()
        dist.all_gather_object(state_snapshots, expected_states)
        if source:
            if engine._kv_transfer_agent.is_push:
                engine.push_handoff_kv(rank, [metadata[peer_rank]])
                for _, handles in engine._pending_kv_pushes:
                    for handle in handles:
                        handle.wait()
        else:
            peer = metadata[peer_rank]
            kv_handle = engine._kv_transfer_agent.begin_pull_blocks(
                peer["kv_meta"], peer["block_ids"], local_blocks
            )
            engine._start_ssm_handoff_import(peer_rank, peer["kv_meta"]["ssm"], pending)
            kv_handle.wait()
            for handle in pending.handles:
                handle.wait()
            snapshot = state_snapshots[peer_rank]
            assert snapshot is not None
            expected_conv, expected_recurrent = snapshot
            torch.testing.assert_close(
                context.kda_conv_states[:, pending.live_slot].cpu(), expected_conv, atol=0, rtol=0
            )
            torch.testing.assert_close(
                context.kda_recurrent_states[:, pending.live_slot].cpu(),
                expected_recurrent,
                atol=0,
                rtol=0,
            )
            assert pending.live_slot != context.max_requests - 1
            admit_prefilled_decode(context, request, local_blocks, [], [77], pending.live_slot)
            context.initialize_attention_state()
            padded = torch.zeros(
                context.padded_active_token_count,
                1,
                kda.config.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            padded[0] = hidden[-1]
            with torch.inference_mode():
                result = kda(padded, None, inference_context=context)[0][0]
            torch.testing.assert_close(result, reference, atol=3e-2, rtol=3e-2)
            assert context.kda_recurrent_states.dtype == torch.float32
            context.release_memory_blocks_from_request_indexes(torch.tensor([0], device="cuda"))
            context.kda_metadata.free_slot(occupied_slot)
            assert context.kda_metadata.mamba_state_free_slot_count == context.max_requests
        dist.barrier()
        if source:
            engine.release_handoff_blocks(rank)
            assert context.kda_metadata.mamba_state_free_slot_count == context.max_requests
            assert context.kv_block_allocator.pool_avail == context.kv_block_allocator.pool_size - 1
        assert context.kda_conv_states[:, context.kda_dummy_state_idx].count_nonzero() == 0
        assert context.kda_recurrent_states[:, context.kda_dummy_state_idx].count_nonzero() == 0
        if backend == "nixl":
            for agent in engine._ssm_transfer_agents.values():
                agent.close()
            engine._kv_transfer_agent.close()
    finally:
        Utils.destroy_model_parallel()
