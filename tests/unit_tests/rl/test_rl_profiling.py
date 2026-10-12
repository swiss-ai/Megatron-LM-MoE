# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest
import torch
import torch.distributed as dist

from megatron.core.timers import Timers
from megatron.rl import rl_profiling
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def distributed_timers():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    yield Timers(log_level=2, log_option="minmax")
    Utils.destroy_model_parallel()


def test_native_timer_classification():
    assert len(rl_profiling.RL_TIMER_NAMES) == len(set(rl_profiling.RL_TIMER_NAMES))
    assert {
        "rl/sync-rollout-state",
        "rl/sync-request-ledger",
        "rl/synchronize-cuda-and-collect-garbage",
        "rl/run-evaluation",
    } <= set(rl_profiling.RL_LOGGABLE_TIMER_NAMES)
    assert "rl/pack-logprobs" in rl_profiling.RL_NONLOGGABLE_TIMER_NAMES
    assert not set(rl_profiling.RL_LOGGABLE_TIMER_NAMES) & set(
        rl_profiling.RL_NONLOGGABLE_TIMER_NAMES
    )


def test_real_timer_collection_jsonl_csv_and_shutdown(distributed_timers, tmp_path):
    """Exercise the native all-rank collector on real CUDA work, not supplied timings."""
    name = "rl/collect-rollouts"
    profiler = rl_profiling.initialize_rl_profiler(
        output_dir=str(tmp_path),
        run_id="real-timers",
        timer_names=[name],
        log_to_wandb=False,
        log_to_tensorboard=False,
    )
    timer = distributed_timers(name, log_level=1)
    tensor = torch.randn(128, 128, device="cuda")
    try:
        for iteration in (1, 2):
            timer.start()
            torch.mm(tensor, tensor)
            timer.stop()
            elapsed_before = timer.elapsed(reset=False)
            rl_profiling.log_iteration_profile(
                iteration, distributed_timers, elapsed_time_ms=100.0, global_batch_size=4
            )
            profile = profiler.iteration_profiles[-1]
            minimum, maximum = profile.timers[name]
            assert 0 < minimum <= maximum
            assert profile.load_imbalance[name] == pytest.approx(maximum / minimum)
            assert profile.rank0_timers[name] > 0
            assert timer.elapsed(reset=False) == pytest.approx(elapsed_before)
        rl_profiling.shutdown_rl_profiler()
        assert rl_profiling.get_rl_profiler() is None
        if dist.get_rank() == 0:
            profiles = rl_profiling.load_profile_jsonl(str(tmp_path / "profile_real-timers.jsonl"))
            assert [profile["iteration"] for profile in profiles] == [1, 2]
            assert profiles[-1]["timer_rl_collect_rollouts_max_ms"] > 0
            summary = rl_profiling.load_summary_csv(str(tmp_path / "summary_real-timers.csv"))
            assert summary[name]["count"] == 2
            assert summary[name]["max_ms"] >= summary[name]["min_ms"] > 0
            assert name in rl_profiling.analyze_bottlenecks(
                str(tmp_path / "summary_real-timers.csv"), top_n=1
            )
        else:
            assert not (tmp_path / "profile_real-timers.jsonl").exists()
            assert not (tmp_path / "summary_real-timers.csv").exists()
    finally:
        rl_profiling.shutdown_rl_profiler()


def test_disabled_profiler_does_not_collect_or_create_files(tmp_path):
    profiler = rl_profiling.RLProfiler(output_dir=str(tmp_path / "disabled"), enabled=False)
    # None is intentionally not a Timers object: disabled profiling must not touch it.
    profiler.log_iteration(1, None, elapsed_time_ms=1.0)
    profiler.close()
    assert profiler.iteration_profiles == []
    assert not profiler.output_dir.exists()


def test_iteration_profile_preserves_ledger_and_throughput_metrics():
    profile = rl_profiling.IterationProfile(
        iteration=3,
        timestamp="test",
        elapsed_time_ms=10.0,
        timers={"rl/sync-request-ledger": (1.0, 2.0)},
        tokens_per_sec=100.0,
        tokens_per_sec_per_gpu=25.0,
        actual_tokens_per_sec=80.0,
        compute_tokens_per_sec=120.0,
        packing_efficiency=2 / 3,
        rank0_timers={"rl/sync-request-ledger": 1.5},
        load_imbalance={"rl/sync-request-ledger": 2.0},
    )
    payload = json.loads(json.dumps(profile.to_dict()))
    assert payload["timer_rl_sync_request_ledger_rank0_ms"] == 1.5
    assert payload["timer_rl_sync_request_ledger_max_ms"] == 2.0
    assert payload["imbalance_rl_sync_request_ledger"] == 2.0
    assert payload["actual_tokens_per_sec"] == 80.0
    assert payload["compute_tokens_per_sec"] == 120.0
    assert payload["packing_efficiency"] == 2 / 3
