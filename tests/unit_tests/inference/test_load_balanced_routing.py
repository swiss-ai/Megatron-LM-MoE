# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core.inference.config import InferenceConfig, PrefixCachingCoordinatorPolicy
from megatron.training.argument_utils import inference_cfg_from_args
from megatron.training.arguments import _add_inference_args
from megatron.training.config.inference_config import InferenceSetupConfig
from tests.unit_tests.inference.coordinator_test_utils import make_coordinator_direct


def test_load_balanced_defaults_and_setup_mapping():
    parser = _add_inference_args(ArgumentParser())
    args = parser.parse_args([])
    assert args.inference_dynamic_batching_prefix_caching_coordinator_policy == "load_balanced"
    setup = inference_cfg_from_args(args)
    assert setup.inference_dynamic_batching_prefix_caching_coordinator_policy == "load_balanced"
    assert InferenceConfig().prefix_caching_coordinator_policy == (
        PrefixCachingCoordinatorPolicy.LOAD_BALANCED
    )
    model = SimpleNamespace(
        position_embedding_type="rotary",
        max_sequence_length=2560,
        pg_collection=None,
        config=SimpleNamespace(params_dtype=torch.float16),
        decoder=SimpleNamespace(layer_type_list=None, layers=[]),
    )
    runtime = InferenceSetupConfig().to_inference_config(model, verbose=False)
    assert runtime.prefix_caching_coordinator_policy == PrefixCachingCoordinatorPolicy.LOAD_BALANCED


@pytest.mark.parametrize("policy", ["load_balanced", "longest_prefix", "first_prefix_block"])
def test_coordinator_cli_accepts_supported_policies(policy):
    parser = _add_inference_args(ArgumentParser())
    args = parser.parse_args(
        ["--inference-dynamic-batching-prefix-caching-coordinator-policy", policy]
    )
    assert args.inference_dynamic_batching_prefix_caching_coordinator_policy == policy


def test_coordinator_cli_rejects_retired_round_robin():
    parser = _add_inference_args(ArgumentParser())
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--inference-dynamic-batching-prefix-caching-coordinator-policy", "round_robin"]
        )
    with pytest.raises(ValueError):
        PrefixCachingCoordinatorPolicy("round_robin")


def test_least_loaded_tie_break_and_updated_counts():
    coordinator = make_coordinator_direct(
        data_parallel_size=3, policy=PrefixCachingCoordinatorPolicy.LOAD_BALANCED
    )
    coordinator._pending_counts[:] = [4, 1, 1]
    assert coordinator.get_best_data_parallel_rank([123]) == b"rank_1"
    # Selection alone does not admit a request or change the load counters.
    np.testing.assert_array_equal(coordinator._pending_counts, [4, 1, 1])
    coordinator._pending_counts[1] += 1
    assert coordinator.get_best_data_parallel_rank([123]) == b"rank_2"
    coordinator._pending_counts[0] = 0
    assert coordinator.get_best_data_parallel_rank([123]) == b"rank_0"


def test_least_loaded_rejects_empty_engine_pool():
    coordinator = make_coordinator_direct(data_parallel_size=0)
    with pytest.raises(RuntimeError, match="No engines connected"):
        coordinator.get_least_loaded_data_parallel_rank()


def test_removed_engine_cleans_load_and_prefix_indices():
    coordinator = make_coordinator_direct(data_parallel_size=3)
    coordinator._pending_counts[:] = [4, 1, 2]
    coordinator._hash_table = {10: {0: 1, 1: 2, 2: 3}, 20: {1: 4}, 30: {2: 5}}
    coordinator._remove_engine(b"rank_1")
    assert coordinator._identities_list == [b"rank_0", b"rank_2"]
    assert coordinator.identity_to_rank_index == {b"rank_0": 0, b"rank_2": 1}
    np.testing.assert_array_equal(coordinator._pending_counts, [4, 2])
    assert coordinator._hash_table == {10: {0: 1, 1: 3}, 30: {1: 5}}
    assert coordinator.get_least_loaded_data_parallel_rank() == b"rank_2"
    coordinator._register_rank_identity(b"rank_new")
    assert coordinator.identity_to_rank_index[b"rank_new"] == 2
    np.testing.assert_array_equal(coordinator._pending_counts, [4, 2, 0])
    assert coordinator.get_least_loaded_data_parallel_rank() == b"rank_new"
