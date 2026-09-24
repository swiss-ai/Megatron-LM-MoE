import pytest
import torch

from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.transformer.moe.expert_checkpoint_utils import (
    make_fused_offloading_experts_canonical_factory,
    make_legacy_offloading_load_factory,
)


@pytest.mark.parametrize("gated", [False, True])
def test_fused_fp8_master_factory_splits_and_merges_experts(gated):
    out_features = 8 if gated else 4
    fused_weight = torch.arange(2 * out_features * 3, dtype=torch.float32).view(
        2, out_features, 3
    )
    factory = make_fused_offloading_experts_canonical_factory(
        fused_weight,
        "experts.",
        "weight1",
        "linear_fc1",
        num_local_experts=2,
        local_expert_indices_offset=4,
        num_global_experts=8,
        sharded_offsets=(),
        replica_id=(0, 0, 0),
        singleton_local_shards=False,
        gated=gated,
    )

    shards = factory.build()
    assert len(shards) == 2 * (2 if gated else 1)
    assert {shard.key for shard in shards} == {"experts.experts.linear_fc1.weight"}
    assert torch.equal(factory.merge_fn([shard.data for shard in shards]), fused_weight)


@pytest.mark.parametrize("gated", [False, True])
def test_legacy_factory_requests_old_key_and_transposes_back(gated):
    canonical = torch.arange(24, dtype=torch.float32).view(8, 3)
    canonical_shard = ShardedTensor.from_rank_offsets(
        "layers.0.mlp.experts.linear_fc1.weight",
        canonical,
        (0, 1, 4),
        replica_id=(0, 0, 0),
        prepend_axis_num=1,
    )
    factory = make_legacy_offloading_load_factory(
        canonical_shard, "linear_fc1", gated=gated
    )

    legacy_shards = factory.build()
    assert len(legacy_shards) == (2 if gated else 1)
    legacy = legacy_shards[0]
    assert legacy.key == "layers.0.mlp.experts.weight1"
    assert legacy.local_shape == ((3, 4) if gated else (3, 8))
    assert legacy.global_shape == (4, 3, 8)
    assert legacy.global_offset == (1, 0, 0)
    if gated:
        assert legacy_shards[1].global_offset == (1, 0, 4)
    else:
        assert torch.equal(legacy.data, canonical.t())
    assert torch.equal(factory.merge_fn([shard.data for shard in legacy_shards]), canonical)
