# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared test fixtures and helpers for inference tests."""

import itertools
from collections import deque

import msgpack
import numpy as np

from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.inference.headers import Headers
from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)


def make_coordinator_direct(
    data_parallel_size=2,
    block_size_tokens=4,
    enable_prefix_caching=True,
    deterministic_mode=True,
    prefix_caching_routing_alpha=0.5,
    prefix_cache_ttl_seconds=300.0,
    max_requests=10,
    policy=PrefixCachingCoordinatorPolicy.LONGEST_PREFIX,
    tokenizer=None,
    rank_name_template="rank_{}",
):
    """Create a coordinator with mock ZMQ, for unit testing routing logic.

    Returns the coordinator instance with fake rank identities.

    Args:
        data_parallel_size: Number of DP ranks.
        block_size_tokens: Block size in tokens.
        enable_prefix_caching: Whether prefix caching is enabled.
        deterministic_mode: If True, sort identities for deterministic ordering.
        prefix_caching_routing_alpha: Alpha for prefix-aware scoring.
        prefix_cache_ttl_seconds: How long a routed block is assumed still held.
        max_requests: Max requests per rank (None disables vectorized scoring).
        policy: Prefix caching coordinator routing policy.
        tokenizer: Optional tokenizer instance (set on the coordinator).
        rank_name_template: Format string for rank names, e.g. ``"rank_{}"``
            or ``"rank-{}"``.  The integer rank index is substituted.
    """
    coordinator = object.__new__(DataParallelInferenceCoordinator)
    coordinator.tokenizer = tokenizer
    coordinator.data_parallel_size = data_parallel_size
    coordinator.block_size_tokens = block_size_tokens
    coordinator.enable_prefix_caching = enable_prefix_caching
    coordinator.prefix_caching_coordinator_policy = policy
    coordinator.prefix_caching_routing_alpha = prefix_caching_routing_alpha
    coordinator.prefix_cache_ttl_seconds = prefix_cache_ttl_seconds
    coordinator._hash_expiry = deque()
    coordinator.max_requests = max_requests
    coordinator.known_clients = set()
    coordinator.next_request_id = 0
    coordinator.schedule_records = None
    coordinator.state = coordinator.CoordinatorState.RUNNING
    coordinator.request_id_to_client_id = {}
    coordinator.request_id_to_client_request_id = {}
    coordinator.client_request_to_request_id = {}
    coordinator.request_id_to_rank = {}
    coordinator.removed_engine_identities = set()

    # Create fake rank identities.
    coordinator.identities_of_data_parallel_ranks = deque(
        [rank_name_template.format(i).encode() for i in range(data_parallel_size)]
    )
    coordinator.removed_engine_identities = set()
    if deterministic_mode:
        coordinator.identities_of_data_parallel_ranks = deque(
            sorted(coordinator.identities_of_data_parallel_ranks)
        )
    coordinator.data_parallel_rank_iterator = itertools.cycle(
        coordinator.identities_of_data_parallel_ranks
    )

    n_ranks = data_parallel_size
    coordinator._hash_table = {}
    coordinator._hash_assignment_counter = 0

    sorted_identities = sorted(coordinator.identities_of_data_parallel_ranks)
    coordinator.identity_to_rank_index = {
        identity: idx for idx, identity in enumerate(sorted_identities)
    }

    coordinator._pending_counts = np.zeros(n_ranks, dtype=np.int32)
    coordinator._identities_list = list(sorted_identities)

    return coordinator


def drive_coordinator_message(coordinator, sender, metadata, bodies=()):
    """Dispatch actual multipart input through the monolithic event loop."""
    control_client = b"test-control"
    coordinator.known_clients.add(control_client)
    coordinator.router_socket.recv_multipart.side_effect = [
        [sender, msgpack.packb(metadata, use_bin_type=True), *bodies],
        [control_client, msgpack.packb([Headers.SHUTDOWN.value], use_bin_type=True)],
    ]
    coordinator.start()
