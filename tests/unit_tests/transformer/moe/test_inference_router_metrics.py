# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.transformer.moe.moe_utils import (
    consume_inference_router_violation_metrics,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _SyntheticRouter(torch.nn.Module):
    def __init__(
        self,
        *,
        topk=2,
        mbs_samples=None,
        seq_samples=None,
        config=None,
        layer_number=None,
        is_mtp_layer=False,
    ):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        self.topk = topk
        self.tp_cp_group = (
            parallel_state.get_tensor_and_context_parallel_group()
            if parallel_state.model_parallel_is_initialized()
            else None
        )
        self.config = config or SimpleNamespace(
            moe_per_layer_logging=False, num_layers=0, mtp_num_layers=None
        )
        self.layer_number = layer_number
        self.is_mtp_layer = is_mtp_layer
        self.inference_mbs_expert_load_samples = list(mbs_samples or [])
        self.inference_seq_expert_load_samples = list(seq_samples or [])


@pytest.fixture
def router_metric_device():
    # NOTE: Real router reductions need initialized groups and matching CUDA samples.
    # Mocked CPU reductions do not request this fixture.
    if torch.cuda.is_available():
        Utils.initialize_model_parallel()
        try:
            yield torch.device("cuda", torch.cuda.current_device())
        finally:
            Utils.destroy_model_parallel()
    else:
        yield torch.device("cpu")


def test_inference_violation_metrics_are_computed_and_consumed(router_metric_device):
    # With top-k 2 and four experts, four valid tokens imply an ideal load of two.
    # Violations are [1, 0, -0.5, -0.5].
    mbs_sample = torch.tensor([4.0, 2.0, 1.0, 1.0, 4.0], device=router_metric_device)
    # The second sequence contains no valid tokens and is excluded from the statistics.
    seq_sample = torch.tensor(
        [[2.0, 1.0, 1.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0]], device=router_metric_device
    )
    router = _SyntheticRouter(mbs_samples=[mbs_sample], seq_samples=[seq_sample]).to(
        router_metric_device
    )
    model = router

    metrics = consume_inference_router_violation_metrics(model)

    assert metrics == pytest.approx(
        {
            "moe_router_mbs_max_violation": 1.0,
            "moe_router_mbs_min_violation": -0.5,
            "moe_router_mbs_median_violation": -0.5,
            "moe_router_mbs_std_violation": math.sqrt(0.375),
            "moe_router_seq_max_violation": 1.0,
            "moe_router_seq_min_violation": -1.0,
            "moe_router_seq_median_violation": 0.0,
            "moe_router_seq_std_violation": math.sqrt(0.5),
        }
    )
    assert not router.inference_mbs_expert_load_samples
    assert not router.inference_seq_expert_load_samples
    assert consume_inference_router_violation_metrics(model) == {}


def test_inference_metrics_weight_all_observations_and_layers(router_metric_device):
    router_a = _SyntheticRouter(
        mbs_samples=[
            torch.tensor([4.0, 0.0, 0.0, 0.0, 2.0], device=router_metric_device),
            torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0], device=router_metric_device),
        ]
    ).to(router_metric_device)
    router_b = _SyntheticRouter(
        mbs_samples=[torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0], device=router_metric_device)]
    ).to(router_metric_device)
    model = [router_a, router_b]

    metrics = consume_inference_router_violation_metrics(model)

    # First observation has max violation 3; the other two are perfectly balanced.
    assert metrics["moe_router_mbs_max_violation"] == pytest.approx(1.0)
    assert set(metrics) == {
        "moe_router_mbs_max_violation",
        "moe_router_mbs_min_violation",
        "moe_router_mbs_median_violation",
        "moe_router_mbs_std_violation",
    }


def test_per_layer_metrics_use_global_zero_based_layer_suffixes(router_metric_device):
    config = SimpleNamespace(
        moe_per_layer_logging=True, num_layers=2, mtp_num_layers=None
    )
    router_a = _SyntheticRouter(
        mbs_samples=[torch.tensor([4.0, 0.0, 0.0, 0.0, 2.0], device=router_metric_device)],
        config=config,
        layer_number=1,
    ).to(router_metric_device)
    router_b = _SyntheticRouter(
        mbs_samples=[torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0], device=router_metric_device)],
        config=config,
        layer_number=2,
    ).to(router_metric_device)

    metrics = consume_inference_router_violation_metrics([router_a, router_b])

    assert metrics["moe_router_mbs_max_violation"] == pytest.approx(1.5)
    assert metrics["moe_router_mbs_max_violation_layer_0"] == pytest.approx(3.0)
    assert metrics["moe_router_mbs_std_violation_layer_0"] == pytest.approx(
        math.sqrt(3.0)
    )
    assert metrics["moe_router_mbs_max_violation_layer_1"] == pytest.approx(0.0)
    assert metrics["moe_router_mbs_std_violation_layer_1"] == pytest.approx(0.0)
    assert len(metrics) == 12


def test_mtp_per_layer_metrics_follow_transformer_layers(router_metric_device):
    config = SimpleNamespace(moe_per_layer_logging=True, num_layers=2, mtp_num_layers=1)
    router = _SyntheticRouter(
        mbs_samples=[torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0], device=router_metric_device)],
        config=config,
        layer_number=1,
        is_mtp_layer=True,
    ).to(router_metric_device)

    metrics = consume_inference_router_violation_metrics(router)

    assert "moe_router_mbs_max_violation_layer_2" in metrics
    assert not any(key.endswith(("layer_0", "layer_1")) for key in metrics)


def test_counts_are_pooled_before_stats_and_dp_is_observation_weighted(monkeypatch):
    tp_cp_group = object()
    dp_group = object()
    pp_group = object()
    router = _SyntheticRouter(mbs_samples=[torch.tensor([2.0, 0.0, 0.0, 0.0, 1.0])])
    router.tp_cp_group = tp_cp_group
    collective_order = []

    def get_world_size(group):
        return 1 if group is pp_group else 2

    def all_reduce(tensor, op, group):
        collective_order.append(group)
        if group is tp_cp_group:
            # The peer shard routes to expert 1. The pooled violations are [1, 1, -1, -1],
            # unlike either local shard's [3, -1, -1, -1].
            tensor.add_(torch.tensor([[0.0, 2.0, 0.0, 0.0, 1.0]]))
        elif group is dp_group:
            # The peer DP rank contributes one perfectly balanced observation.
            tensor[0, 0, 4] += 1

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", get_world_size)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    metrics = consume_inference_router_violation_metrics(
        router,
        pg_collection=SimpleNamespace(dp=dp_group, pp=pp_group),
    )

    assert collective_order == [tp_cp_group, dp_group]
    assert metrics == pytest.approx(
        {
            "moe_router_mbs_max_violation": 0.5,
            "moe_router_mbs_min_violation": -0.5,
            "moe_router_mbs_median_violation": -0.5,
            "moe_router_mbs_std_violation": 0.5,
        }
    )


def test_samples_are_preserved_when_reduction_fails(monkeypatch):
    router = _SyntheticRouter(mbs_samples=[torch.tensor([2.0, 0.0, 0.0, 0.0, 1.0])])
    router.tp_cp_group = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def fail_reduction(*args, **kwargs):
        raise RuntimeError("collective failed")

    monkeypatch.setattr(torch.distributed, "all_reduce", fail_reduction)

    with pytest.raises(RuntimeError, match="collective failed"):
        consume_inference_router_violation_metrics(router)
    assert len(router.inference_mbs_expert_load_samples) == 1


def test_inference_metrics_config_is_disabled_and_validated():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
    )
    assert config.moe_router_inference_violation_metrics == []

    with pytest.raises(ValueError, match="must be"):
        TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            moe_router_inference_violation_metrics=["ep"],
        )
