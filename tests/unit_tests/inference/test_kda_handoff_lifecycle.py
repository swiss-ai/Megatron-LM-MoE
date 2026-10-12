# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lifecycle coverage for disaggregated KDA state handoff."""

import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator
from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
)
from megatron.core.inference.disaggregation.pending_handoff_imports import (
    PendingKvImport,
    PendingSSMImport,
)
from megatron.core.inference.sampling_params import SamplingParams
from tests.unit_tests.inference.test_disagg_handoff_lifecycle import (
    _HandoffHarness,
    _PendingHandle,
    _drain_loop,
    _meta,
)


class _RegisteredBuffer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)

    def export_meta(self):
        return {"agent": self.kwargs["agent_name"]}

    def begin_pull_blocks(self, peer_meta, src_block_ids, dst_block_ids):
        self.calls.append((peer_meta, list(src_block_ids), list(dst_block_ids)))
        return _PendingHandle()

    def begin_push_blocks(self, peer_meta, src_block_ids):
        self.pushes.append((peer_meta, list(src_block_ids)))
        return _PendingHandle()


class _KdaBackend:
    instances = []
    is_push = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)
        self.calls = []
        self.pushes = []

    def export_meta(self):
        return {"agent": self.kwargs["agent_name"]}

    def new_registered_buffer(self, **kwargs):
        return _RegisteredBuffer(**kwargs)

    def begin_pull_blocks(self, peer_meta, src_block_ids, dst_block_ids):
        self.calls.append((peer_meta, list(src_block_ids), list(dst_block_ids)))
        return _PendingHandle()


@pytest.fixture
def kda_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _kda_setup_engine(role="decode"):
    engine = object.__new__(InferenceStateHandoffMixin)
    engine._initialize_disaggregation_state()
    allocator = SimpleNamespace(
        enable_prefix_caching=False, enable_handoff_pinning=False, pool_size=16
    )
    engine.context = SimpleNamespace(
        has_kda=True,
        is_hybrid_model=False,
        config=SimpleNamespace(enable_prefix_caching=False, nixl_backend="UCX"),
        kv_block_allocator=allocator,
        memory_buffer=torch.empty(2, 1, 16, 4, 1, 1),
        max_requests=5,
        num_attention_layers=2,
        num_kda_layers=2,
        num_mamba_layers=0,
        num_attention_heads_per_partition=2,
        hidden_size_per_attention_head=4,
        block_size_tokens=4,
        kda_layer_map={3: 0, 8: 1},
        kda_conv_states=torch.empty(2, 6, 74, 5),
        kda_recurrent_states=torch.empty(2, 6, 6, 4, 7, dtype=torch.float32),
    )
    engine.controller = SimpleNamespace(
        inference_wrapped_model=SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(
                    num_query_groups=2,
                    num_attention_heads=4,
                    linear_num_key_heads=4,
                    linear_num_value_heads=6,
                    linear_key_head_dim=4,
                    linear_value_head_dim=7,
                    linear_conv_kernel_dim=5,
                )
            )
        )
    )
    engine.pg_collection = SimpleNamespace(tp=None, pp=None, mp=None)
    _KdaBackend.instances.clear()
    _RegisteredBuffer.instances.clear()
    backend_factory = (
        "megatron.core.inference.disaggregation.inference_state_handoff."
        "construct_kv_transfer_backend_class"
    )
    with (
        mock.patch(backend_factory, return_value=_KdaBackend),
        mock.patch(
            "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_size",
            return_value=1,
        ),
        mock.patch(
            "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_rank",
            return_value=0,
        ),
    ):
        engine.setup_kv_transfer(role)
    return engine


def test_kda_transfer_setup_registers_live_state_and_rejects_prefix_reuse():
    engine = _kda_setup_engine()

    assert engine.context.kv_block_allocator.enable_handoff_pinning is False
    assert len(_RegisteredBuffer.instances) == 2
    conv, recurrent = _RegisteredBuffer.instances
    assert conv.kwargs["memory_buffer"] is engine.context.kda_conv_states
    assert recurrent.kwargs["memory_buffer"] is engine.context.kda_recurrent_states
    assert [agent.kwargs["expected_num_blocks"] for agent in _RegisteredBuffer.instances] == [5, 5]
    assert [agent.kwargs["ssm_state_kind"] for agent in _RegisteredBuffer.instances] == [
        "conv",
        "recurrent",
    ]
    assert conv.kwargs["ssm_layout"] is recurrent.kwargs["ssm_layout"]
    layout = conv.kwargs["ssm_layout"]
    assert layout.dims.num_key_heads == 4
    assert layout.dims.num_value_heads == 6
    assert layout.dims.key_head_dim == 4
    assert layout.dims.value_head_dim == 7
    assert layout.dims.conv_kernel_dim == 5
    assert layout.kda_layer_map == {3: 0, 8: 1}

    engine.context.config.enable_prefix_caching = True
    with pytest.raises(RuntimeError, match="KDA handoff does not support prefix reuse"):
        engine.setup_kv_transfer("decode")


def test_kda_import_reserves_live_metadata_slot_and_requires_both_state_families(kda_loop):
    engine = _HandoffHarness(kda_loop)
    engine.context.has_kda = True
    engine.context.is_hybrid_model = False
    engine.context.num_speculative_tokens = 0
    engine.context.kda_metadata = SimpleNamespace(
        mamba_state_free_slot_count=0,
        allocated=[],
        freed=[],
        allocate_slot=lambda: None,
        free_slot=lambda slot: None,
    )
    engine._kv_transfer_agent = _KdaBackend(agent_name="decode-kv")
    engine._ssm_transfer_agents = {"conv": _KdaBackend(agent_name="decode-conv")}
    with pytest.raises(RuntimeError, match="missing state kinds"):
        engine.add_request_with_kv_handoff(
            7,
            [1, 2, 3, 4],
            SamplingParams(num_tokens_to_generate=2),
            {"resume_tokens": [99], "ssm": {"conv": {}}},
            [101],
        )

    engine._ssm_transfer_agents["recurrent"] = _KdaBackend(agent_name="decode-recurrent")
    valid_ssm = {"conv": {}, "recurrent": {}}
    engine._ssm_transfer_agents = {}
    with pytest.raises(RuntimeError, match="before SSM transfer setup"):
        engine.add_request_with_kv_handoff(
            7,
            [1, 2, 3, 4],
            SamplingParams(num_tokens_to_generate=2),
            {"resume_tokens": [99], "ssm": valid_ssm},
            [101],
        )

    engine._ssm_transfer_agents = {
        "conv": _KdaBackend(agent_name="decode-conv"),
        "recurrent": _KdaBackend(agent_name="decode-recurrent"),
    }
    with pytest.raises(RuntimeError, match="missing state kinds"):
        engine.add_request_with_kv_handoff(
            7,
            [1, 2, 3, 4],
            SamplingParams(num_tokens_to_generate=2),
            {"resume_tokens": [99], "ssm": {"conv": {}}},
            [101],
        )


def test_kda_pending_import_capacity_deferral_and_slot_release(kda_loop):
    engine = _HandoffHarness(kda_loop)
    engine.context.prefix_cache_lru_clock = 0
    engine.context.kv_block_allocator = KVBlockAllocator(
        engine.context,
        pool_size=8,
        paused_limit=0,
        enable_prefix_caching=False,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
    )
    engine.context.has_kda = True
    engine.context.is_hybrid_model = False
    engine.context.num_speculative_tokens = 0
    metadata = engine.context.kda_metadata = SimpleNamespace(
        mamba_state_free_slot_count=0, next_slot=17, freed=[]
    )

    def allocate_slot():
        if metadata.mamba_state_free_slot_count == 0:
            return None
        metadata.mamba_state_free_slot_count -= 1
        return metadata.next_slot

    def free_slot(slot):
        metadata.freed.append(slot)
        metadata.mamba_state_free_slot_count += 1

    metadata.allocate_slot = allocate_slot
    metadata.free_slot = free_slot
    engine._kv_transfer_agent = _KdaBackend(agent_name="decode-kv")
    engine._ssm_transfer_agents = {
        "conv": _KdaBackend(agent_name="decode-conv"),
        "recurrent": _KdaBackend(agent_name="decode-recurrent"),
    }
    kv_meta = {
        **_meta(7),
        "ssm": {"conv": {"agent": "source-conv"}, "recurrent": {"agent": "source-recurrent"}},
    }
    future = engine.add_request_with_kv_handoff(
        7, [1, 2, 3, 4], SamplingParams(num_tokens_to_generate=2), kv_meta, [101]
    )
    assert not future.done()
    assert len(engine._deferred_kv_handoffs) == 1
    assert not engine._pending_kv_imports
    assert not engine._kv_transfer_agent.calls

    metadata.mamba_state_free_slot_count = 1
    engine._poll_pending_kv_imports()
    _drain_loop(kda_loop)
    assert not engine._deferred_kv_handoffs
    assert len(engine._pending_kv_imports) == 1
    pending = engine._pending_kv_imports[0]
    assert pending.ssm == PendingSSMImport(handles=pending.ssm.handles, live_slot=17)
    assert engine._kv_transfer_agent.calls == [(kv_meta, [101], [5])]
    for state_kind in ("conv", "recurrent"):
        assert len(engine._ssm_transfer_agents[state_kind].calls) == 1

    engine._release_pending_kv_import(pending)
    assert metadata.freed == [17]
    assert pending.ssm is None


def test_kda_unsafe_pending_import_is_quarantined_without_releasing_live_slot(kda_loop):
    engine = _HandoffHarness(kda_loop)
    engine.context.has_kda = True
    engine.context.kda_metadata = SimpleNamespace(freed=[], free_slot=lambda slot: None)
    engine.context.kda_metadata.free_slot = engine.context.kda_metadata.freed.append
    block_id = int(engine.context.kv_block_allocator.allocate_memory_blocks(1)[0])
    pending = PendingKvImport(
        request_id=7,
        prompt=[1, 2, 3, 4],
        sampling_params=SamplingParams(num_tokens_to_generate=2),
        local_blocks=[block_id],
        hashes=[101],
        cached_prefix_block_count=0,
        handle=_PendingHandle(),
        future=kda_loop.create_future(),
        ssm=PendingSSMImport(handles=[], live_slot=17),
    )
    engine._pending_kv_imports.append(pending)
    engine._record_handoff_completion_notification(7, failed=True)

    engine._poll_pending_kv_imports()
    engine._admit_pending_kv_imports()

    assert pending.future.exception() is not None
    assert engine._quarantined_kv_imports == [pending]
    assert engine.context.kv_block_allocator.releases == []
    assert engine.context.kda_metadata.freed == []


def test_kda_source_slot_detaches_and_releases_with_finished_handoff():
    engine = object.__new__(InferenceStateHandoffMixin)
    engine._initialize_disaggregation_state()
    engine.context = SimpleNamespace(
        has_kda=True,
        kv_block_allocator=SimpleNamespace(release_memory_blocks=mock.Mock()),
        kda_metadata=SimpleNamespace(freed=[], free_slot=lambda slot: None),
        block_size_tokens=4,
        num_speculative_tokens=0,
    )
    engine.context.kda_metadata.free_slot = engine.context.kda_metadata.freed.append
    engine.pg_collection = SimpleNamespace(tp=None, pp=None, mp=None)
    engine._kv_peer_metas = {"global_rank": 0}
    engine._ssm_transfer_agents = {"conv": mock.Mock(), "recurrent": mock.Mock()}
    engine._pp_ssm_peer_metas = [{"conv": {"agent": "conv"}, "recurrent": {"agent": "recurrent"}}]
    request = SimpleNamespace(
        request_id=19,
        prompt_tokens=torch.arange(4),
        sampling_params=SamplingParams(do_kv_handoff=True),
        disaggregated_params=None,
    )
    with mock.patch(
        "megatron.core.inference.disaggregation.inference_state_handoff.get_pg_size", return_value=1
    ):
        prepared = engine._prepare_handoff_metadata_batch([(request, [31], 4)], {19: [99]})
        engine._capture_handoff_meta(request, prepared[19])

    assert engine._pinned_handoff_ssm_slots == {19: 4}
    assert request.disaggregated_params["kv_meta"]["resume_tokens"] == [99]
    assert request.disaggregated_params["kv_meta"]["ssm"]
    engine.release_handoff_blocks(19)
    assert engine.context.kda_metadata.freed == [4]
    engine.context.kv_block_allocator.release_memory_blocks.assert_called_once()
