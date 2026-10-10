# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_logging import (
    destroy_moe_metrics_tracker,
    get_moe_metrics_tracker,
)
from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker,
    get_moe_layer_wise_logging_tracker,
    reduce_aux_losses_tracker_across_ranks,
    save_to_aux_losses_tracker,
    track_moe_metrics,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def reset_moe_metrics_tracker():
    destroy_moe_metrics_tracker()
    yield
    destroy_moe_metrics_tracker()


def test_legacy_record_forwarder_and_layer_wise_view_share_tracker():
    tracker = get_moe_metrics_tracker()
    reduce_group = object()
    avg_group = object()

    with pytest.warns(DeprecationWarning):
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            torch.tensor(2.0),
            layer_number=2,
            num_layers=4,
            reduce_group=reduce_group,
            avg_group=avg_group,
            reduce_group_has_dp=True,
        )

    with pytest.warns(DeprecationWarning):
        save_to_aux_losses_tracker("default_dp_average", torch.tensor(1.0), 1, 4)
    assert tracker.metrics["default_dp_average"].needs_dp_avg is True
    assert tracker.metrics["default_dp_average"].reduce_group is None

    tracker.record("direct_metric", torch.tensor(3.0), 1, 4)
    assert get_moe_metrics_tracker() is tracker
    assert tracker.metrics["load_balancing_loss"].values[1].item() == 2.0
    assert tracker.metrics["load_balancing_loss"].reduce_group is reduce_group
    assert tracker.metrics["load_balancing_loss"].avg_group is avg_group
    assert tracker.metrics["load_balancing_loss"].needs_dp_avg is False
    assert tracker.metrics["direct_metric"].needs_dp_avg is True

    with pytest.warns(DeprecationWarning):
        legacy_view = get_moe_layer_wise_logging_tracker()

    assert legacy_view is not tracker.metrics
    assert (
        legacy_view["load_balancing_loss"]["values"]
        is tracker.metrics["load_balancing_loss"].values
    )
    assert legacy_view["load_balancing_loss"]["needs_dp_avg"] is False
    assert "reduce_group_has_dp" not in legacy_view["load_balancing_loss"]
    legacy_view["discarded_write"] = {"values": torch.ones(1)}
    assert "discarded_write" not in tracker.metrics


def test_legacy_clear_zeros_values_without_replacing_storage():
    tracker = get_moe_metrics_tracker()
    tracker.record("load_balancing_loss", torch.tensor(2.0), 1, 2)
    entry = tracker.metrics["load_balancing_loss"]
    values = entry.values
    data_ptr = values.data_ptr()

    with pytest.warns(DeprecationWarning):
        clear_aux_losses_tracker()

    assert tracker.metrics["load_balancing_loss"] is entry
    assert entry.values is values
    assert entry.values.data_ptr() == data_ptr
    torch.testing.assert_close(entry.values, torch.zeros(2))


def test_legacy_reducer_syncs_preinitialized_entry_in_expected_order(monkeypatch):
    tracker = get_moe_metrics_tracker()
    tracker.ensure_initialized("metric", 2, device="cpu")
    pp_group, reduce_group, avg_group, dp_group = (object() for _ in range(4))
    tracker.record(
        "metric",
        torch.tensor(5.0),
        layer_number=1,
        num_layers=2,
        reduce_group=reduce_group,
        avg_group=avg_group,
        needs_dp_avg=True,
    )
    entry = tracker.metrics["metric"]
    pg_collection = ProcessGroupCollection()
    pg_collection.pp = pp_group
    pg_collection.dp = dp_group
    calls = []

    def record_all_reduce(tensor, group=None, op=None):
        calls.append((tensor, group, op))

    monkeypatch.setattr(torch.distributed, "all_reduce", record_all_reduce)

    with pytest.warns(DeprecationWarning):
        reduce_aux_losses_tracker_across_ranks(track_names=["metric"], pg_collection=pg_collection)

    assert tracker.metrics["metric"] is entry
    assert [group for _, group, _ in calls] == [pp_group, reduce_group, avg_group, dp_group]
    assert [op for _, _, op in calls] == [
        None,
        None,
        torch.distributed.ReduceOp.AVG,
        torch.distributed.ReduceOp.AVG,
    ]
    assert all(tensor is entry.values for tensor, _, _ in calls)


def test_report_force_initializes_full_and_mtp_slots_and_accumulates_loss(monkeypatch):
    tracker = get_moe_metrics_tracker()
    tracker.ensure_initialized("load_balancing_loss", 6, device="cpu")
    entry = tracker.metrics["load_balancing_loss"]
    values = entry.values
    data_ptr = values.data_ptr()
    tracker.record("load_balancing_loss", torch.tensor(1.0), 1, 6)
    tracker.record("load_balancing_loss", torch.tensor(3.0), 4, 6)
    monkeypatch.setattr(tracker, "_sync_metrics", lambda *args: None)
    total_loss_dict = {}

    tracker.report(
        loss_scale=0.5,
        iteration=11,
        force_initialize=True,
        track_names=["load_balancing_loss", "new_metric"],
        num_layers=4,
        mtp_num_layers=2,
        total_loss_dict=total_loss_dict,
    )

    assert entry.values is values
    assert entry.values.data_ptr() == data_ptr
    assert entry.values.numel() == 4 + 2
    assert tracker.metrics["new_metric"].values.numel() == 4 + 2
    torch.testing.assert_close(
        tracker.metrics["new_metric"].values, torch.zeros_like(tracker.metrics["new_metric"].values)
    )
    torch.testing.assert_close(total_loss_dict["load_balancing_loss"], torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(entry.values, torch.zeros(6))


def test_deprecated_training_forwarder_preserves_report_arguments(monkeypatch):
    tracker = get_moe_metrics_tracker()
    pg_collection = SimpleNamespace(pp=object(), dp=object())
    received = {}

    def report(**kwargs):
        received.update(kwargs)
        return "logged"

    monkeypatch.setattr(tracker, "report", report)
    total_loss_dict = {}

    with pytest.warns(DeprecationWarning):
        result = track_moe_metrics(
            loss_scale=0.25,
            iteration=7,
            per_layer_logging=True,
            force_initialize=True,
            track_names=["load_balancing_loss"],
            num_layers=4,
            mtp_num_layers=2,
            pg_collection=pg_collection,
            total_loss_dict=total_loss_dict,
        )

    assert result == "logged"
    assert received == {
        "loss_scale": 0.25,
        "iteration": 7,
        "writer": None,
        "wandb_writer": None,
        "per_layer_logging": True,
        "force_initialize": True,
        "track_names": ["load_balancing_loss"],
        "num_layers": 4,
        "moe_layer_freq": None,
        "mtp_num_layers": 2,
        "pg_collection": pg_collection,
        "total_loss_dict": total_loss_dict,
    }


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sync_metrics_reduces_native_pp_tp_dp_groups_numerically():
    if Utils.world_size != 8:
        pytest.skip("Requires the native 8-rank MoE test fixture")

    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)
    try:
        tracker = get_moe_metrics_tracker()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        dp_group = parallel_state.get_data_parallel_group()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        dp_rank = parallel_state.get_data_parallel_rank()
        pg_collection = ProcessGroupCollection()
        pg_collection.pp = pp_group
        pg_collection.dp = dp_group

        local_value = torch.tensor(1.0 + 100.0 * pp_rank + 10.0 * tp_rank + dp_rank, device="cuda")
        tracker.record(
            "sum_then_dp_average", local_value, layer_number=1, num_layers=1, reduce_group=tp_group
        )
        tracker.record(
            "tp_average_then_dp_average",
            local_value,
            layer_number=1,
            num_layers=1,
            avg_group=tp_group,
        )

        tracker._sync_metrics(["sum_then_dp_average", "tp_average_then_dp_average"], pg_collection)

        torch.testing.assert_close(
            tracker.metrics["sum_then_dp_average"].values, torch.tensor([226.0], device="cuda")
        )
        torch.testing.assert_close(
            tracker.metrics["tp_average_then_dp_average"].values,
            torch.tensor([113.0], device="cuda"),
        )
    finally:
        Utils.destroy_model_parallel()
