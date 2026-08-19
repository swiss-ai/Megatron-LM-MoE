# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
import megatron.core.optimizer.muon_logging as md_logging_module
from megatron.core.optimizer import HAVE_EMERGING_OPTIMIZERS
from megatron.core.optimizer.md_decoupling import MDDecoupling
from megatron.core.optimizer.muon_logging import _gain_log_family
from megatron.core.optimizer.muon_logging import _include_gain_in_global_stats
from megatron.core.optimizer.muon_logging import collect_md_gain_stats
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


def test_md_gain_log_family_classifies_matrix_types():
    cases = [
        ("decoder.layers.0.mlp.router.weight", "is_router", "router"),
        ("embedding.word_embeddings.weight", "is_md_embedding_parameter", "embedding"),
        ("output_layer.weight", "is_md_output_parameter", "output"),
        ("decoder.layers.0.self_attention.linear_qkv.weight", None, "attention-in"),
        (
            "decoder.layers.0.self_attention.linear_proj.weight",
            "is_out_proj",
            "attention-out",
        ),
        ("decoder.layers.0.mlp.experts.linear_fc1.weight", None, "expert-in"),
        (
            "decoder.layers.0.mlp.experts.linear_fc2.weight",
            "is_out_proj",
            "expert-out",
        ),
        ("decoder.layers.0.mlp.fc1_latent_proj.weight", None, "moe-latent-in"),
        ("decoder.layers.0.mlp.fc2_latent_proj.weight", None, "moe-latent-out"),
        ("decoder.layers.0.mlp.linear_fc1.weight", None, "dense-mlp-in"),
        (
            "decoder.layers.0.mlp.linear_fc2.weight",
            "is_out_proj",
            "dense-mlp-out",
        ),
        ("some_unclassified_matrix.weight", None, "unclassified"),
    ]
    for name, attribute, expected in cases:
        param = torch.nn.Parameter(torch.ones(2, 2))
        if attribute:
            setattr(param, attribute, True)
        assert _gain_log_family(name, param) == expected


def test_collect_md_gain_stats_logs_layernorm_effective_gains():
    matrix = torch.nn.Parameter(torch.ones(2, 2))
    matrix.md_gain_log_family = "attention-in"
    md_optimizer = MDDecoupling(
        [matrix], hypersphere_gains_mode="row", gain_parametrization="direct"
    )
    md_optimizer.state[matrix]["row_gain"] = torch.ones(2)

    layernorm = torch.nn.Parameter(torch.tensor([0.0, 0.5]))
    layernorm.md_gain_log_family = _gain_log_family(
        "decoder.layers.2.input_layernorm.weight", layernorm
    )
    layernorm.md_layernorm_gain_offset = 1.0
    layernorm.md_gain_log_layer = 2
    adam = torch.optim.Adam([layernorm])
    optimizer = SimpleNamespace(
        chained_optimizers=(
            SimpleNamespace(optimizer=md_optimizer),
            SimpleNamespace(optimizer=adam),
        )
    )

    stats = collect_md_gain_stats(
        optimizer, per_layer=True, log_sparsity=False, log_param_rms=False
    )

    expected_rms = ((1.0**2 + 1.5**2) / 2) ** 0.5
    for prefix in ("muon-md", "muon-md/layers/2"):
        gain_prefix = f"{prefix}/gains/layernorm/flat"
        assert stats[f"{gain_prefix}/mean"] == pytest.approx(1.25)
        assert stats[f"{gain_prefix}/rms"] == pytest.approx(expected_rms)
        assert stats[f"{gain_prefix}/effective-rms"] == pytest.approx(expected_rms)
        assert stats[f"{gain_prefix}/min"] == pytest.approx(1.0)
        assert stats[f"{gain_prefix}/max"] == pytest.approx(1.5)
        assert f"{gain_prefix}/saturated-fraction" not in stats


def test_collect_md_gain_stats_logs_effective_gains_by_family_and_axis():
    router = torch.nn.Parameter(torch.ones(2, 2))
    router.md_gain_log_family = "router"
    attention = torch.nn.Parameter(torch.ones(2, 2))
    attention.md_gain_log_family = "attention-in"
    optimizer = MDDecoupling(
        [router, attention],
        hypersphere_gains_mode="rowcol",
        gain_parametrization="softplus",
    )
    optimizer.state[router]["row_gain"] = optimizer._phi_inv(torch.tensor([1.0, 3.0]))
    optimizer.state[router]["col_gain"] = optimizer._phi_inv(torch.tensor([2.0, 4.0]))
    optimizer.state[attention]["flat_gain"] = optimizer._phi_inv(torch.tensor(5.0))

    stats = collect_md_gain_stats(SimpleNamespace(optimizer=optimizer))

    assert stats["muon-md/gains/router/row/mean"] == pytest.approx(2.0)
    assert stats["muon-md/gains/router/row/rms"] == pytest.approx(5.0**0.5)
    assert stats["muon-md/gains/router/row/effective-rms"] == pytest.approx(5.0**0.5)
    assert stats["muon-md/gains/router/row/min"] == pytest.approx(1.0)
    assert stats["muon-md/gains/router/row/max"] == pytest.approx(3.0)
    assert stats["muon-md/gains/router/row/saturated-fraction"] == pytest.approx(0.0)
    assert stats["muon-md/gains/router/col/mean"] == pytest.approx(3.0)
    assert stats["muon-md/gains/router/col/rms"] == pytest.approx(10.0**0.5)
    assert stats["muon-md/gains/router/col/effective-rms"] == pytest.approx(10.0**0.5)
    assert stats["muon-md/gain-field/router/rms"] == pytest.approx(50.0**0.5)
    assert stats["muon-md/gauge/router/combined-log-scale"] == pytest.approx(
        (
            torch.tensor([1.0, 3.0]).log().mean()
            + torch.tensor([2.0, 4.0]).log().mean()
        ).item()
    )
    assert stats["muon-md/gauge/router/row-col-imbalance"] == pytest.approx(
        (
            torch.tensor([1.0, 3.0]).log().mean()
            - torch.tensor([2.0, 4.0]).log().mean()
        ).item()
    )
    assert stats["muon-md/gains/attention-in/flat/mean"] == pytest.approx(5.0)
    assert "muon-md/gain-field/attention-in/rms" not in stats
    assert not any("muon-md/layers/" in name for name in stats)
    assert not any("/unclassified/" in name for name in stats)


def test_collect_md_gain_stats_logs_softplus_saturation_only():
    softplus_param = torch.nn.Parameter(torch.ones(2, 2))
    softplus_param.md_gain_log_family = "router"
    softplus_optimizer = MDDecoupling(
        [softplus_param],
        hypersphere_gains_mode="row",
        gain_parametrization="softplus",
    )
    softplus_optimizer.state[softplus_param]["row_gain"] = torch.tensor([-10.0, 0.0])

    softplus_stats = collect_md_gain_stats(
        SimpleNamespace(optimizer=softplus_optimizer)
    )

    assert softplus_stats[
        "muon-md/gains/router/row/saturated-fraction"
    ] == pytest.approx(0.5)

    direct_param = torch.nn.Parameter(torch.ones(2, 2))
    direct_param.md_gain_log_family = "router"
    direct_optimizer = MDDecoupling(
        [direct_param],
        hypersphere_gains_mode="rowcol",
        gain_parametrization="direct",
    )
    direct_optimizer.state[direct_param]["row_gain"] = torch.tensor([1.0, 2.0])
    direct_optimizer.state[direct_param]["col_gain"] = torch.tensor([3.0, 4.0])

    direct_stats = collect_md_gain_stats(SimpleNamespace(optimizer=direct_optimizer))

    assert direct_stats["muon-md/gain-field/router/rms"] == pytest.approx(
        ((5.0 / 2.0) * (25.0 / 2.0)) ** 0.5
    )
    assert not any("saturated-fraction" in name for name in direct_stats)
    assert not any("muon-md/gauge/" in name for name in direct_stats)


def test_collect_md_gain_stats_logs_global_layer_buckets():
    layer_zero = torch.nn.Parameter(torch.ones(2, 2))
    layer_zero.md_gain_log_family = "attention-in"
    layer_zero.md_gain_log_layer = 0
    layer_three = torch.nn.Parameter(torch.ones(2, 2))
    layer_three.md_gain_log_family = "attention-in"
    layer_three.md_gain_log_layer = 3
    optimizer = MDDecoupling(
        [layer_zero, layer_three],
        hypersphere_gains_mode="rowcol",
        gain_parametrization="softplus",
    )
    optimizer.state[layer_zero]["row_gain"] = optimizer._phi_inv(
        torch.tensor([1.0, 1.0])
    )
    optimizer.state[layer_zero]["col_gain"] = optimizer._phi_inv(
        torch.tensor([2.0, 2.0])
    )
    optimizer.state[layer_three]["row_gain"] = optimizer._phi_inv(
        torch.tensor([3.0, 3.0])
    )
    optimizer.state[layer_three]["col_gain"] = optimizer._phi_inv(
        torch.tensor([4.0, 4.0])
    )

    stats = collect_md_gain_stats(SimpleNamespace(optimizer=optimizer), per_layer=True)

    assert stats["muon-md/gains/attention-in/row/effective-rms"] == pytest.approx(2.0)
    assert stats[
        "muon-md/layers/0/gains/attention-in/row/effective-rms"
    ] == pytest.approx(1.0)
    assert stats["muon-md/layers/0/gain-field/attention-in/rms"] == pytest.approx(2.0)
    assert stats[
        "muon-md/layers/3/gains/attention-in/col/effective-rms"
    ] == pytest.approx(4.0)
    assert stats["muon-md/layers/3/gain-field/attention-in/rms"] == pytest.approx(12.0)
    assert not any("muon-md/layers/1/" in name for name in stats)


def test_collect_md_gain_stats_logs_effective_weight_sparsity_per_matrix():
    layer_zero = torch.nn.Parameter(torch.tensor([[0.0, 1e-21], [1e-20, -1.0]]))
    layer_zero.md_gain_log_family = "expert-in"
    layer_zero.md_gain_log_layer = 0
    layer_one = torch.nn.Parameter(torch.tensor([[2.0]]))
    layer_one.md_gain_log_family = "expert-in"
    layer_one.md_gain_log_layer = 1
    optimizer = MDDecoupling(
        [layer_zero, layer_one],
        hypersphere_gains_mode="row",
        gain_parametrization="direct",
    )
    # Parameters contain the gain-baked effective weights when logging runs. Large gain state here
    # ensures collection does not incorrectly apply the gain a second time.
    optimizer.state[layer_zero]["row_gain"] = torch.tensor([100.0, 100.0])
    optimizer.state[layer_one]["row_gain"] = torch.tensor([100.0])
    wrapped_optimizer = SimpleNamespace(optimizer=optimizer)

    stats = collect_md_gain_stats(wrapped_optimizer, per_layer=True)

    assert stats["muon-md/sparsity/expert-in/fraction-below-1e-20"] == pytest.approx(
        0.25
    )
    assert stats["muon-md/sparsity/expert-in/fraction-below-1e-10"] == pytest.approx(
        0.375
    )
    assert stats["muon-md/sparsity/expert-in/fraction-below-1e-30"] == pytest.approx(
        0.125
    )
    assert stats[
        "muon-md/layers/0/sparsity/expert-in/fraction-below-1e-20"
    ] == pytest.approx(0.5)
    assert stats[
        "muon-md/layers/1/sparsity/expert-in/fraction-below-1e-20"
    ] == pytest.approx(0.0)
    assert stats["muon-md/params/expert-in/rms"] == pytest.approx(1.25)
    assert stats["muon-md/layers/0/params/expert-in/rms"] == pytest.approx(0.5)
    assert stats["muon-md/layers/1/params/expert-in/rms"] == pytest.approx(2.0)

    custom_stats = collect_md_gain_stats(wrapped_optimizer, sparsity_thresholds=[0.5])
    assert custom_stats[
        "muon-md/sparsity/expert-in/fraction-below-0.5"
    ] == pytest.approx(0.375)
    assert not any("fraction-below-1e-20" in name for name in custom_stats)

    sparsity_only_stats = collect_md_gain_stats(
        wrapped_optimizer, log_gains=False, log_param_rms=False
    )
    assert any("/sparsity/" in name for name in sparsity_only_stats)
    assert not any("/gains/" in name for name in sparsity_only_stats)
    assert not any("/gain-field/" in name for name in sparsity_only_stats)
    assert not any("/params/" in name for name in sparsity_only_stats)

    gains_only_stats = collect_md_gain_stats(
        wrapped_optimizer, log_sparsity=False, log_param_rms=False
    )
    assert any("/gains/" in name for name in gains_only_stats)
    assert not any("/sparsity/" in name for name in gains_only_stats)
    assert not any("/params/" in name for name in gains_only_stats)

    param_only_stats = collect_md_gain_stats(
        wrapped_optimizer, log_gains=False, log_sparsity=False
    )
    assert param_only_stats["muon-md/params/expert-in/rms"] == pytest.approx(1.25)
    assert all("/params/" in name for name in param_only_stats)


def test_collect_md_gain_stats_pairs_row_and_col_within_each_matrix():
    first = torch.nn.Parameter(torch.ones(2, 2))
    first.md_gain_log_family = "attention-in"
    first.md_gain_log_layer = 0
    second = torch.nn.Parameter(torch.ones(4, 1))
    second.md_gain_log_family = "attention-in"
    second.md_gain_log_layer = 0
    optimizer = MDDecoupling(
        [first, second],
        hypersphere_gains_mode="rowcol",
        gain_parametrization="softplus",
    )
    optimizer.state[first]["row_gain"] = optimizer._phi_inv(torch.tensor([1.0, 1.0]))
    optimizer.state[first]["col_gain"] = optimizer._phi_inv(torch.tensor([10.0, 10.0]))
    optimizer.state[second]["row_gain"] = optimizer._phi_inv(torch.tensor([3.0] * 4))
    optimizer.state[second]["col_gain"] = optimizer._phi_inv(torch.tensor([2.0]))

    stats = collect_md_gain_stats(SimpleNamespace(optimizer=optimizer), per_layer=True)

    assert stats["muon-md/gains/attention-in/row/effective-rms"] == pytest.approx(2.0)
    assert stats["muon-md/gains/attention-in/row/rms"] == pytest.approx(
        (38.0 / 6.0) ** 0.5
    )
    assert stats["muon-md/gain-field/attention-in/rms"] == pytest.approx(8.0)
    assert stats["muon-md/layers/0/gain-field/attention-in/rms"] == pytest.approx(8.0)
    expected_combined = (
        math.log(1.0) + math.log(10.0) + math.log(3.0) + math.log(2.0)
    ) / 2.0
    expected_imbalance = (
        math.log(1.0) - math.log(10.0) + math.log(3.0) - math.log(2.0)
    ) / 2.0
    assert stats["muon-md/gauge/attention-in/combined-log-scale"] == pytest.approx(
        expected_combined
    )
    assert stats["muon-md/gauge/attention-in/row-col-imbalance"] == pytest.approx(
        expected_imbalance
    )

    row_only = torch.nn.Parameter(torch.ones(2, 2))
    row_only.md_gain_log_family = "router"
    col_only = torch.nn.Parameter(torch.ones(2, 2))
    col_only.md_gain_log_family = "router"
    unpaired_optimizer = MDDecoupling(
        [row_only, col_only],
        hypersphere_gains_mode="rowcol",
        gain_parametrization="softplus",
    )
    unpaired_optimizer.state[row_only]["row_gain"] = unpaired_optimizer._phi_inv(
        torch.tensor([1.0, 2.0])
    )
    unpaired_optimizer.state[col_only]["col_gain"] = unpaired_optimizer._phi_inv(
        torch.tensor([3.0, 4.0])
    )

    unpaired_stats = collect_md_gain_stats(
        SimpleNamespace(optimizer=unpaired_optimizer)
    )

    assert "muon-md/gain-field/router/rms" not in unpaired_stats
    assert not any("muon-md/gauge/router/" in name for name in unpaired_stats)


def test_collect_md_gain_stats_treats_merged_experts_as_separate_matrices():
    experts = torch.nn.Parameter(torch.ones(2, 2, 2))
    experts.md_gain_log_family = "expert-in"
    experts.md_gain_log_layer = 1
    optimizer = MDDecoupling(
        [experts],
        hypersphere_gains_mode="rowcol",
        gain_parametrization="softplus",
    )
    optimizer.state[experts]["row_gain"] = optimizer._phi_inv(
        torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    )
    optimizer.state[experts]["col_gain"] = optimizer._phi_inv(
        torch.tensor([[10.0, 10.0], [2.0, 2.0]])
    )

    stats = collect_md_gain_stats(SimpleNamespace(optimizer=optimizer), per_layer=True)

    assert stats["muon-md/gains/expert-in/row/effective-rms"] == pytest.approx(2.0)
    assert stats["muon-md/gain-field/expert-in/rms"] == pytest.approx(8.0)
    assert stats["muon-md/layers/1/gain-field/expert-in/rms"] == pytest.approx(8.0)


def test_collect_md_gain_stats_launches_reductions_before_waiting(monkeypatch):
    tp_group = object()
    expert_tp_group = object()
    normal = torch.nn.Parameter(torch.ones(2, 2))
    normal.partition_dim = 0
    normal.md_gain_log_family = "attention-in"
    expert = torch.nn.Parameter(torch.ones(2, 2))
    expert.partition_dim = 0
    expert.expert_tp = True
    expert.md_gain_log_family = "expert-in"
    optimizer = MDDecoupling(
        [normal, expert],
        hypersphere_gains_mode="row",
        gain_parametrization="softplus",
        pg_collection=SimpleNamespace(
            tp=tp_group,
            expt_tp=expert_tp_group,
            dp_cp=None,
            expt_dp=None,
        ),
    )
    optimizer.state[normal]["row_gain"] = optimizer._phi_inv(torch.tensor([1.0, 2.0]))
    optimizer.state[expert]["row_gain"] = optimizer._phi_inv(torch.tensor([3.0, 4.0]))
    events = []

    def fake_all_reduce(tensor, op=None, group=None, async_op=False):
        del tensor, op
        assert async_op
        events.append(("launch", group))
        return SimpleNamespace(wait=lambda: events.append(("wait", group)))

    monkeypatch.setattr(md_logging_module, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(md_logging_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        md_logging_module.torch.distributed, "is_available", lambda: True
    )
    monkeypatch.setattr(
        md_logging_module.torch.distributed, "is_initialized", lambda: True
    )
    monkeypatch.setattr(
        md_logging_module.torch.distributed, "all_reduce", fake_all_reduce
    )

    collect_md_gain_stats(SimpleNamespace(optimizer=optimizer))

    assert events == [
        ("launch", tp_group),
        ("launch", expert_tp_group),
        ("wait", tp_group),
        ("wait", expert_tp_group),
        *(("launch", None),) * 5,
        *(("wait", None),) * 5,
    ]


def test_md_gain_stats_tp_and_dp_ownership(monkeypatch):
    tp_group = object()
    dp_group = object()
    expert_tp_group = object()
    expert_dp_group = object()
    ranks = {
        tp_group: 1,
        dp_group: 0,
        expert_tp_group: 0,
        expert_dp_group: 0,
    }
    monkeypatch.setattr(md_logging_module, "get_pg_rank", lambda group: ranks[group])
    optimizer = SimpleNamespace(
        pg_collection=SimpleNamespace(
            tp=tp_group,
            dp_cp=dp_group,
            expt_tp=expert_tp_group,
            expt_dp=expert_dp_group,
        )
    )

    cases = [
        (0, "row", 1, 0, False, False, True),
        (0, "col", 1, 0, False, False, False),
        (0, "flat", 1, 0, False, False, False),
        (1, "row", 1, 0, False, False, False),
        (1, "col", 1, 0, False, False, True),
        (0, "row", 0, 1, False, False, False),
        (0, "row", 0, 1, True, False, True),
        (None, "flat", 0, 1, False, True, False),
    ]
    for partition_dim, axis, tp_rank, dp_rank, sharded, expert, expected in cases:
        ranks[expert_tp_group if expert else tp_group] = tp_rank
        ranks[expert_dp_group if expert else dp_group] = dp_rank
        param = torch.nn.Parameter(torch.ones(3, 4))
        param.partition_dim = partition_dim
        param.expert_tp = expert
        assert (
            _include_gain_in_global_stats(optimizer, param, axis, sharded) is expected
        )


requires_distributed_cuda = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not HAVE_EMERGING_OPTIMIZERS
    or int(os.getenv("WORLD_SIZE", "1")) == 1,
    reason="CUDA, emerging_optimizers, and multiple ranks are required",
)


@requires_distributed_cuda
def test_gain_stats_count_tp_shards_once_and_deduplicate_replicas():
    world = int(os.getenv("WORLD_SIZE", "1"))
    Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
    try:
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        param = torch.nn.Parameter(
            torch.zeros(2, 3, device="cuda")
            if tp_rank == 0
            else torch.ones(2, 3, device="cuda")
        )
        param.partition_dim = 0
        param.md_gain_log_family = "attention-in"
        optimizer = MDDecoupling(
            [param],
            hypersphere_gains_mode="rowcol",
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        optimizer.state[param]["row_gain"] = torch.tensor(
            [1.0, 2.0] if tp_rank == 0 else [3.0, 4.0], device="cuda"
        )
        optimizer.state[param]["col_gain"] = torch.tensor(
            [5.0, 7.0, 9.0], device="cuda"
        )

        stats = collect_md_gain_stats(SimpleNamespace(optimizer=optimizer))

        assert stats["muon-md/gains/attention-in/row/mean"] == pytest.approx(2.5)
        assert stats["muon-md/gains/attention-in/row/rms"] == pytest.approx(7.5**0.5)
        assert stats["muon-md/gains/attention-in/col/mean"] == pytest.approx(7.0)
        assert stats["muon-md/gains/attention-in/col/rms"] == pytest.approx(
            (155.0 / 3.0) ** 0.5
        )
        assert stats[
            "muon-md/sparsity/attention-in/fraction-below-1e-20"
        ] == pytest.approx(0.5)
        assert stats["muon-md/params/attention-in/rms"] == pytest.approx(0.5**0.5)
    finally:
        Utils.destroy_model_parallel()


@requires_distributed_cuda
def test_gain_stats_deduplicate_context_parallel_replicas():
    world = int(os.getenv("WORLD_SIZE", "1"))
    Utils.initialize_model_parallel(context_parallel_size=world)
    try:
        param = torch.nn.Parameter(torch.ones(2, 2, device="cuda"))
        param.md_gain_log_family = "attention-in"
        optimizer = MDDecoupling(
            [param],
            hypersphere_gains_mode="rowcol",
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        optimizer.state[param]["row_gain"] = torch.tensor([1.0, 3.0], device="cuda")
        optimizer.state[param]["col_gain"] = torch.tensor([2.0, 4.0], device="cuda")

        stats = collect_md_gain_stats(SimpleNamespace(optimizer=optimizer))

        assert stats["muon-md/gains/attention-in/row/mean"] == pytest.approx(2.0)
        assert stats["muon-md/gain-field/attention-in/rms"] == pytest.approx(50.0**0.5)
    finally:
        Utils.destroy_model_parallel()


@requires_distributed_cuda
def test_gain_stats_aggregate_expert_parallel_matrices():
    world = int(os.getenv("WORLD_SIZE", "1"))
    Utils.initialize_model_parallel(
        expert_model_parallel_size=world,
        expert_tensor_parallel_size=1,
    )
    try:
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        param = torch.nn.Parameter(torch.ones(2, 2, device="cuda"))
        param.expert_tp = True
        param.md_gain_log_family = "expert-in"
        optimizer = MDDecoupling(
            [param],
            hypersphere_gains_mode="rowcol",
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        row_value = float(ep_rank + 1)
        optimizer.state[param]["row_gain"] = torch.full((2,), row_value, device="cuda")
        optimizer.state[param]["col_gain"] = torch.full((2,), 2.0, device="cuda")

        stats = collect_md_gain_stats(SimpleNamespace(optimizer=optimizer))

        expected_row_rms = sum(range(1, world + 1)) / world
        expected_gain_field_rms = 2.0 * expected_row_rms
        assert stats["muon-md/gains/expert-in/row/effective-rms"] == pytest.approx(
            expected_row_rms
        )
        assert stats["muon-md/gain-field/expert-in/rms"] == pytest.approx(
            expected_gain_field_rms
        )
    finally:
        Utils.destroy_model_parallel()


@requires_distributed_cuda
def test_gain_stats_reconstruct_expert_tensor_parallel_shards():
    world = int(os.getenv("WORLD_SIZE", "1"))
    Utils.initialize_model_parallel(expert_tensor_parallel_size=world)
    try:
        expert_tp_rank = parallel_state.get_expert_tensor_parallel_rank()
        param = torch.nn.Parameter(torch.ones(1, 2, device="cuda"))
        param.partition_dim = 0
        param.expert_tp = True
        param.md_gain_log_family = "expert-in"
        optimizer = MDDecoupling(
            [param],
            hypersphere_gains_mode="rowcol",
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        optimizer.state[param]["row_gain"] = torch.tensor(
            [float(expert_tp_rank + 1)], device="cuda"
        )
        optimizer.state[param]["col_gain"] = torch.tensor([5.0, 7.0], device="cuda")

        stats = collect_md_gain_stats(SimpleNamespace(optimizer=optimizer))

        expected_row_rms = (
            sum(value**2 for value in range(1, world + 1)) / world
        ) ** 0.5
        expected_col_rms = 37.0**0.5
        assert stats["muon-md/gains/expert-in/row/rms"] == pytest.approx(
            expected_row_rms
        )
        assert stats["muon-md/gains/expert-in/col/rms"] == pytest.approx(
            expected_col_rms
        )
        assert stats["muon-md/gain-field/expert-in/rms"] == pytest.approx(
            expected_row_rms * expected_col_rms
        )
    finally:
        Utils.destroy_model_parallel()
