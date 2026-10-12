# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.inference.config import AsyncScheduleMode
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    DecodeForwardPrimer,
    TextGenerationController,
)


_CONTROLLER_MODULE = (
    "megatron.core.inference.text_generation_controllers.text_generation_controller"
)


def _make_context(total_request_count=2, mode=AsyncScheduleMode.SERIAL):
    size = max(total_request_count, 1)
    return SimpleNamespace(
        config=SimpleNamespace(async_sched_mode=mode, materialize_only_last_token_logits=True),
        is_hybrid_model=False,
        has_kda=False,
        enable_prefix_caching=False,
        paused_request_count=0,
        total_request_count=total_request_count,
        active_token_count=total_request_count,
        chunked_prefill_request_id=-1,
        num_prefill_requests=0,
        padded_active_request_count=8,
        request_ids=torch.arange(10, 10 + size, dtype=torch.int32),
        request_metadata={
            "top_k": torch.ones(size, dtype=torch.int64),
            "top_p": torch.zeros(size),
            "return_log_probs": torch.zeros(size, dtype=torch.bool),
            "top_n_logprobs": torch.zeros(size, dtype=torch.int64),
            "termination_id": torch.full((size,), 99, dtype=torch.int64),
        },
        async_sched_step_count=0,
        async_sched_compaction_step_count=0,
        get_active_sequence_lengths=mock.Mock(
            return_value=torch.full((size,), 3, dtype=torch.int32)
        ),
        get_max_sequence_lengths=mock.Mock(return_value=torch.full((size,), 10, dtype=torch.int32)),
        prepare_requests=mock.Mock(),
        resolve_requests=mock.Mock(return_value=torch.empty(0, dtype=torch.int32)),
        using_cuda_graph_this_step=mock.Mock(return_value=False),
    )


def _make_controller(context=None):
    context = context or _make_context()
    model_config = SimpleNamespace(
        params_dtype=torch.float32,
        expert_model_parallel_size=1,
        num_moe_experts=None,
        moe_enable_routing_replay=False,
    )
    controller = TextGenerationController.__new__(TextGenerationController)
    controller.inference_wrapped_model = SimpleNamespace(
        inference_context=context, model=SimpleNamespace(config=model_config)
    )
    controller.model_config = model_config
    controller.num_speculative_tokens = 0
    controller._enable_cuda_graph = False
    controller._decode_forward_primer = DecodeForwardPrimer()
    controller._all_logits_cuda = torch.zeros(1, max(context.total_request_count, 1), 5)
    controller._dynamic_step_context_init = mock.Mock(
        return_value=(torch.tensor([[10, 11]]), torch.tensor([[0, 1]]))
    )
    controller._dynamic_step_forward_logits = mock.Mock()
    return controller


def test_controller_validates_kda_only_for_async_mode():
    context = _make_context()
    context.has_kda = True
    controller = _make_controller(context)

    with pytest.raises(RuntimeError, match="does not support KDA"):
        controller._validate_async_sched_support_for_step()

    context.config.async_sched_mode = AsyncScheduleMode.LEGACY
    controller._run_legacy_step = mock.AsyncMock(return_value="legacy")
    assert asyncio.run(controller.async_generate_output_tokens_dynamic_batch()) == "legacy"


def test_controller_routes_prefill_and_legacy_steps_to_existing_path():
    context = _make_context(mode=AsyncScheduleMode.SERIAL)
    context.num_prefill_requests = 1
    controller = _make_controller(context)
    controller._run_legacy_step = mock.AsyncMock(return_value="legacy")
    controller._run_async_sched_serial_step = mock.AsyncMock(return_value="serial")

    assert asyncio.run(controller.async_generate_output_tokens_dynamic_batch()) == "legacy"
    controller._run_legacy_step.assert_awaited_once_with(False)
    controller._run_async_sched_serial_step.assert_not_awaited()


def test_controller_requires_bookkeeping_for_serial_mode():
    controller = _make_controller()

    with pytest.raises(AssertionError, match="request bookkeeping"):
        asyncio.run(controller.async_generate_output_tokens_dynamic_batch(skip_bookkeeping=True))


def test_controller_rejects_unsupported_metadata_for_serial_step():
    context = _make_context()
    context.request_metadata["top_k"][0] = 0
    controller = _make_controller(context)

    with pytest.raises(RuntimeError, match="greedy sampling"):
        controller._validate_async_sched_support_for_step()


def test_controller_compacts_cached_logits_in_survivor_order():
    controller = _make_controller()
    controller._all_logits_cuda = torch.arange(12).reshape(1, 4, 3)
    controller._decode_forward_primer.mark_primed(4)

    controller._compact_async_sched_logits(torch.tensor([0, 2], dtype=torch.int64))

    assert torch.equal(controller._all_logits_cuda, torch.tensor([[[0, 1, 2], [6, 7, 8]]]))
    assert controller._decode_forward_primer.is_primed
    assert controller._decode_forward_primer.cuda_graph_request_count == 4


def test_controller_serial_step_samples_prepares_resolves_and_compacts():
    context = _make_context(total_request_count=3)
    context.request_metadata["termination_id"] = torch.tensor([99, 2, 99])
    context.resolve_requests = mock.Mock(return_value=torch.tensor([11], dtype=torch.int32))
    controller = _make_controller(context)
    controller._all_logits_cuda = torch.zeros(1, 3, 5)
    controller._all_logits_cuda[0, 0, 1] = 10.0
    controller._all_logits_cuda[0, 1, 2] = 10.0
    controller._all_logits_cuda[0, 2, 3] = 10.0

    with (
        mock.patch(f"{_CONTROLLER_MODULE}.range_push"),
        mock.patch(f"{_CONTROLLER_MODULE}.range_pop"),
    ):
        result = asyncio.run(controller._run_async_sched_serial_step())

    assert result["active_request_ids"].tolist() == [10, 11, 12]
    assert result["finished_request_ids"].tolist() == [11]
    assert result["sample"].tolist() == [1, 2, 3]
    assert result["newly_paused_request_ids"] is None
    assert result["evict_request_ids"] is None
    assert result["cuda_graph_request_count"] is None
    context.prepare_requests.assert_called_once()
    prepared_tokens = context.prepare_requests.call_args.args[0]
    assert torch.equal(prepared_tokens, torch.tensor([1, 2, 3]))
    context.resolve_requests.assert_called_once()
    resolved_mask = context.resolve_requests.call_args.args[0]
    assert resolved_mask.tolist() == [1, 0, 1]
    expected_compacted_logits = torch.tensor([[[0, 10, 0, 0, 0], [0, 0, 0, 10, 0]]])
    assert torch.equal(controller._all_logits_cuda, expected_compacted_logits)
    assert context.async_sched_step_count == 1
    assert context.async_sched_compaction_step_count == 1
    assert controller._decode_forward_primer.is_primed


def test_controller_serial_step_clears_primer_when_no_requests_remain():
    context = _make_context(total_request_count=0)
    context.active_token_count = 0
    controller = _make_controller(context)
    controller._decode_forward_primer.mark_primed(8)
    controller._validate_async_sched_support_for_step = mock.Mock()

    assert asyncio.run(controller._run_async_sched_serial_step()) is None
    assert not controller._decode_forward_primer.is_primed
    controller._validate_async_sched_support_for_step.assert_not_called()


@pytest.mark.parametrize(
    "source, attribute, value, expected_message",
    [
        (
            "config",
            "materialize_only_last_token_logits",
            False,
            "materialize_only_last_token_logits",
        ),
        ("controller", "num_speculative_tokens", 1, "speculative"),
        ("context", "is_hybrid_model", True, "hybrid/Mamba"),
        ("context", "enable_prefix_caching", True, "prefix caching"),
        ("context", "paused_request_count", 1, "paused"),
        ("context", "chunked_prefill_request_id", 0, "chunked prefill"),
        ("model_config", "expert_model_parallel_size", 2, "expert parallelism"),
        ("model_config", "num_moe_experts", 4, "MoE"),
        ("model_config", "moe_enable_routing_replay", True, "routing replay"),
    ],
)
def test_controller_serial_guard_rejects_unsupported_modes(
    source, attribute, value, expected_message
):
    context = _make_context()
    controller = _make_controller(context)
    target = {
        "config": context.config,
        "controller": controller,
        "context": context,
        "model_config": controller.model_config,
    }[source]
    setattr(target, attribute, value)

    with pytest.raises(RuntimeError, match=expected_message):
        controller._validate_async_sched_support_for_step()


def test_controller_compacts_logits_in_static_cuda_graph_buffer():
    controller = _make_controller()
    controller._enable_cuda_graph = True
    controller._decode_forward_primer.mark_primed(8)
    logits = torch.arange(12).reshape(1, 4, 3)
    controller._all_logits_cuda = logits.clone()

    controller._compact_async_sched_logits(torch.tensor([0, 2], dtype=torch.int64))

    assert controller._all_logits_cuda.shape == logits.shape
    assert torch.equal(controller._all_logits_cuda[:, :2], logits[:, [0, 2]])
    assert controller._decode_forward_primer.is_primed
    assert controller._decode_forward_primer.cuda_graph_request_count == 8


def test_controller_async_forward_records_cuda_graph_primer():
    context = _make_context()
    context.using_cuda_graph_this_step.return_value = True
    controller = _make_controller(context)
    controller._dynamic_step_forward_logits = mock.Mock()

    with (
        mock.patch(f"{_CONTROLLER_MODULE}.range_push"),
        mock.patch(f"{_CONTROLLER_MODULE}.range_pop"),
    ):
        graph_request_count = controller._run_async_sched_forward(
            torch.tensor([[10, 11]]), torch.tensor([[0, 1]])
        )

    assert graph_request_count == context.padded_active_request_count
    assert controller._decode_forward_primer.is_primed
    assert (
        controller._decode_forward_primer.cuda_graph_request_count
        == context.padded_active_request_count
    )
    controller._dynamic_step_forward_logits.assert_called_once()


def test_controller_rejects_unknown_async_schedule_mode():
    context = _make_context(mode="unexpected")
    controller = _make_controller(context)
    controller._run_legacy_step = mock.AsyncMock()
    controller._run_async_sched_serial_step = mock.AsyncMock()

    with pytest.raises(AssertionError, match="Unexpected async scheduling mode"):
        asyncio.run(controller.async_generate_output_tokens_dynamic_batch())


@pytest.mark.parametrize(
    "field, value, expected_message",
    [
        ("top_p", 0.5, "greedy sampling"),
        ("return_log_probs", True, "log probabilities"),
        ("top_n_logprobs", 1, "top-n log probabilities"),
    ],
)
def test_controller_serial_guard_rejects_unsupported_sampling_metadata(
    field, value, expected_message
):
    context = _make_context()
    context.request_metadata[field][0] = value
    controller = _make_controller(context)

    with pytest.raises(RuntimeError, match=expected_message):
        controller._validate_async_sched_support_for_step()
