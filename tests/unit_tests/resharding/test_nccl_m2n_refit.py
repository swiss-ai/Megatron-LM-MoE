# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import traceback
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.resharding.copy_services.nccl_m2n_copy_service import HAVE_NCCL_M2N
from megatron.core.resharding.refit import (
    clear_all_caches,
    prepare_swap_model_weights,
    swap_model_weights,
)
from megatron.core.transformer.module import MegatronModule
from tests.unit_tests.test_utilities import Utils


class _RefitModule(MegatronModule):
    def __init__(self, value, shard_dim, pg_collection, buffer_dtype):
        super().__init__(config=SimpleNamespace(num_moe_experts=None))
        self.pg_collection = pg_collection
        self.weight = torch.nn.Parameter(value, requires_grad=False)
        self.weight.tensor_model_parallel = True
        self.weight.partition_dim = shard_dim
        self.weight.partition_stride = 1
        self.register_buffer("expert_bias", torch.zeros(8, device=value.device, dtype=buffer_dtype))
        self.refresh_count = 0

    def refresh_cache(self):
        self.refresh_count += 1


def _values(rows, columns, row_start=0, column_start=0, offset=0):
    row_ids = torch.arange(row_start, row_start + rows, device="cuda").view(-1, 1)
    column_ids = torch.arange(column_start, column_start + columns, device="cuda").view(1, -1)
    return (offset + row_ids * 1000 + column_ids).float()


@pytest.mark.skipif(not HAVE_NCCL_M2N, reason="requires official nccl.core and nccl.m2n bindings")
@pytest.mark.parametrize("destination_buffer_dtype", [torch.float32, torch.bfloat16])
def test_local_m2n_refit_reuses_plan_and_refreshes_destination(destination_buffer_dtype):
    Utils.initialize_distributed()
    world_size = dist.get_world_size()
    if world_size < 4 or world_size % 2:
        pytest.skip("requires an even distributed world size >= 4")
    rank = dist.get_rank()
    mesh_size = world_size // 2
    source_ranks = list(range(mesh_size))
    destination_ranks = list(range(mesh_size, world_size))
    source_pg = dist.new_group(source_ranks)
    destination_pg = dist.new_group(destination_ranks)
    singleton_groups = [dist.new_group([member]) for member in range(world_size)]
    is_source = rank < mesh_size
    mesh_rank = rank if is_source else rank - mesh_size
    tp_group = source_pg if is_source else destination_pg
    singleton = singleton_groups[rank]
    pg_collection = ProcessGroupCollection(
        tp=tp_group, pp=singleton, ep=singleton, dp=singleton, expt_tp=tp_group
    )
    local_extent = 16
    full_extent = mesh_size * local_extent
    value = (
        _values(full_extent, local_extent, column_start=mesh_rank * local_extent)
        if is_source
        else torch.full((full_extent, local_extent), -1.0, device="cuda")
    )
    module = _RefitModule(
        value, 1, pg_collection, torch.float32 if is_source else destination_buffer_dtype
    )
    source = module if is_source else None
    destination = None if is_source else module
    clear_all_caches()
    try:
        prepare_swap_model_weights(source, destination)
        # Normalize buffer storage before graph capture, then keep it stable on refit.
        assert module.expert_bias.dtype == torch.float32
        weight_ptr = module.weight.data_ptr()
        buffer_ptr = module.expert_bias.data_ptr()
        for iteration in range(2):
            offset = iteration * 100000
            if is_source:
                module.weight.copy_(
                    _values(
                        full_extent,
                        local_extent,
                        column_start=mesh_rank * local_extent,
                        offset=offset,
                    )
                )
                module.expert_bias.fill_(iteration + 3)
            swap_model_weights(source, destination, refit_method="nccl_m2n")
            torch.cuda.synchronize()
            if not is_source:
                expected = _values(
                    full_extent, local_extent, column_start=mesh_rank * local_extent, offset=offset
                )
                torch.testing.assert_close(module.weight, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    module.expert_bias, torch.full_like(module.expert_bias, iteration + 3)
                )
                assert module.refresh_count == iteration + 1
            assert module.weight.data_ptr() == weight_ptr
            assert module.expert_bias.data_ptr() == buffer_ptr
        if is_source:
            module.expert_bias.data = module.expert_bias.data.to(torch.bfloat16)
        with pytest.raises(RuntimeError, match="Persistent-buffer dtypes changed"):
            swap_model_weights(source, destination, refit_method="nccl_m2n")
        if is_source:
            module.expert_bias.data = module.expert_bias.data.to(torch.float32)
            del module.expert_bias
        with pytest.raises(RuntimeError, match="Persistent-buffer dtypes changed"):
            swap_model_weights(source, destination, refit_method="nccl_m2n")
    except BaseException:
        # Report a rank-local failure before collective cleanup can block its traceback.
        traceback.print_exc()
        raise
    finally:
        clear_all_caches()
        dist.barrier()
        for group in singleton_groups:
            if group != dist.GroupMember.NON_GROUP_MEMBER:
                dist.destroy_process_group(group)
        if source_pg != dist.GroupMember.NON_GROUP_MEMBER:
            dist.destroy_process_group(source_pg)
        if destination_pg != dist.GroupMember.NON_GROUP_MEMBER:
            dist.destroy_process_group(destination_pg)
