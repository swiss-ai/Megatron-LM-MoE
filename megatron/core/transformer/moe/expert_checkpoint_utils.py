"""Checkpoint layout adapters shared by MoE expert implementations."""

import gc
import logging
from dataclasses import replace
from typing import Optional

import torch

from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedTensor,
    ShardedTensorFactory,
)

logger = logging.getLogger(__name__)

EXPERT_CKPT_SCHEMA_KEY = 'moe_expert_checkpoint_schema'
EXPERT_CKPT_SCHEMA_LEGACY_OFFLOADING = 'legacy_offloading'
EXPERT_CKPT_HAS_TE_EXTRA_STATE_KEY = 'moe_expert_checkpoint_has_te_extra_state'


def is_legacy_offloading_checkpoint(metadata) -> bool:
    """Whether model sharding should request the pre-canonical offloading schema."""
    return (
        metadata is not None
        and metadata.get(EXPERT_CKPT_SCHEMA_KEY) == EXPERT_CKPT_SCHEMA_LEGACY_OFFLOADING
    )


def apply_swiglu_sharded_factory(
    original_sh_ten, sharded_offsets, singleton_local_shards: bool = False
):
    """Split a gated FC1 tensor for saving and concatenate it again when loading."""
    swiglu_shard_axis = 0
    prepend_axis_num = len(sharded_offsets)
    local_axis_size = original_sh_ten.local_shape[swiglu_shard_axis]
    assert (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] % local_axis_size == 0
    )
    rank_offset = (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] // local_axis_size
    )
    axis_frag = original_sh_ten.axis_fragmentations[swiglu_shard_axis + prepend_axis_num]

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, tensor: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        if singleton_local_shards:
            offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
            offset_v = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
            w_key = f'{key}_w'
            v_key = f'{key}_v'
        else:
            offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag * 2)
            offset_v = (
                swiglu_shard_axis + prepend_axis_num,
                rank_offset + axis_frag,
                axis_frag * 2,
            )
            w_key = key
            v_key = key

        tensor_w, tensor_v = torch.chunk(tensor, 2, dim=swiglu_shard_axis)
        return [
            ShardedTensor.from_rank_offsets(
                w_key,
                tensor_w,
                *sharded_offsets,
                offset_w,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
            ShardedTensor.from_rank_offsets(
                v_key,
                tensor_v,
                *sharded_offsets,
                offset_v,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
        ]

    def sh_ten_merge_fn(sub_state_dict):
        with torch.no_grad():
            try:
                return torch.cat(sub_state_dict)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                logger.warning(
                    "CUDA OOM encountered while merging gated FC1 checkpoint shards; "
                    "switching to a CPU merge. (Error: %s)",
                    error,
                )
                merged_sub_state_dict = torch.cat([tensor.cpu() for tensor in sub_state_dict])
                gc.collect()
                torch.cuda.empty_cache()
                return merged_sub_state_dict

    return ShardedTensorFactory(
        original_sh_ten.key,
        original_sh_ten.data,
        sh_ten_build_fn,
        sh_ten_merge_fn,
        original_sh_ten.replica_id,
        flattened_range=original_sh_ten.flattened_range,
    )


def _canonical_expert_key(runtime_prefix: str, linear_name: str, global_expert_idx=None):
    if global_expert_idx is None:
        return f'{runtime_prefix}experts.{linear_name}.weight'
    return f'{runtime_prefix}experts.{global_expert_idx}.{linear_name}.weight'


def _build_canonical_expert_shards(
    checkpoint_key,
    data,
    offsets,
    replica_id,
    singleton_local_shards,
    gated,
):
    canonical_shard = ShardedTensor.from_rank_offsets(
        checkpoint_key,
        data,
        *offsets,
        replica_id=replica_id,
        prepend_axis_num=len(offsets),
    )
    if not gated:
        return [canonical_shard]
    return apply_swiglu_sharded_factory(
        canonical_shard, offsets, singleton_local_shards
    ).build()


def make_offloading_expert_canonical_factory(
    parameter: torch.Tensor,
    prefix: str,
    parameter_name: str,
    linear_name: str,
    *,
    global_expert_idx: int,
    num_global_experts: int,
    sharded_offsets: tuple,
    replica_id,
    singleton_local_shards: bool,
    gated: bool = False,
):
    """Map one offloading parameter to the canonical Sequential/TE checkpoint schema."""

    @torch.no_grad()
    def build_fn(key, tensor, rep_id, flattened_range):
        assert flattened_range is None, "flattening unsupported for offloaded experts"
        assert key.endswith(parameter_name), (key, parameter_name)
        runtime_prefix = key[: -len(parameter_name)]
        data = tensor.transpose(-2, -1).contiguous()
        if singleton_local_shards:
            checkpoint_key = _canonical_expert_key(
                runtime_prefix, linear_name, global_expert_idx
            )
            offsets = sharded_offsets
        else:
            checkpoint_key = _canonical_expert_key(runtime_prefix, linear_name)
            offsets = (
                *sharded_offsets,
                (len(sharded_offsets), global_expert_idx, num_global_experts),
            )
        return _build_canonical_expert_shards(
            checkpoint_key,
            data,
            offsets,
            rep_id,
            singleton_local_shards,
            gated,
        )

    @torch.no_grad()
    def merge_fn(loaded_shards):
        data = torch.cat(loaded_shards, dim=0) if gated else loaded_shards[0]
        return data.transpose(-2, -1).contiguous()

    return ShardedTensorFactory(
        f'{prefix}{parameter_name}', parameter, build_fn, merge_fn, replica_id
    )


def make_fused_offloading_experts_canonical_factory(
    fused_weight: torch.Tensor,
    prefix: str,
    parameter_name: str,
    linear_name: str,
    *,
    num_local_experts: int,
    local_expert_indices_offset: int,
    num_global_experts: int,
    sharded_offsets: tuple,
    replica_id,
    singleton_local_shards: bool,
    gated: bool = False,
):
    """Map a fused [local_expert, out, in] master to canonical expert shards."""

    @torch.no_grad()
    def build_fn(key, tensor, rep_id, flattened_range):
        assert flattened_range is None, "flattening unsupported for offloaded experts"
        assert key.endswith(parameter_name), (key, parameter_name)
        runtime_prefix = key[: -len(parameter_name)]
        shards = []
        for local_idx in range(num_local_experts):
            global_idx = local_expert_indices_offset + local_idx
            if singleton_local_shards:
                checkpoint_key = _canonical_expert_key(runtime_prefix, linear_name, global_idx)
                offsets = sharded_offsets
            else:
                checkpoint_key = _canonical_expert_key(runtime_prefix, linear_name)
                offsets = (
                    *sharded_offsets,
                    (len(sharded_offsets), global_idx, num_global_experts),
                )
            shards.extend(
                _build_canonical_expert_shards(
                    checkpoint_key,
                    tensor[local_idx],
                    offsets,
                    rep_id,
                    singleton_local_shards,
                    gated,
                )
            )
        return shards

    @torch.no_grad()
    def merge_fn(loaded_shards):
        if gated:
            experts = [
                torch.cat(loaded_shards[i : i + 2], dim=0)
                for i in range(0, len(loaded_shards), 2)
            ]
        else:
            experts = loaded_shards
        return torch.stack(experts, dim=0).contiguous()

    return ShardedTensorFactory(
        f'{prefix}{parameter_name}', fused_weight, build_fn, merge_fn, replica_id
    )


# ------------
# Legacy offloading checkpoint support

# NOTE: This should be removed once we no longer support legacy offloading checkpoints. It is only used to load old checkpoints that were saved in the legacy offloading format.
def make_legacy_offloading_load_factory(
    canonical_sharded_tensor: ShardedTensor, linear_name: str, gated: bool = False
):
    """Request an old [in, out] offloading tensor for a canonical [out, in] parameter."""
    legacy_weight_name = {'linear_fc1': 'weight1', 'linear_fc2': 'weight2'}[linear_name]
    canonical_suffix = f'{linear_name}.weight'

    @torch.no_grad()
    def build_fn(key, tensor, replica_id, flattened_range):
        assert flattened_range is None, "flattening unsupported for legacy offloaded experts"
        assert key.endswith(canonical_suffix), (key, canonical_suffix)
        legacy_key = f'{key[: -len(canonical_suffix)]}{legacy_weight_name}'
        canonical_parts = [(tensor, canonical_sharded_tensor)]
        if gated:
            local_out = canonical_sharded_tensor.local_shape[-2]
            assert local_out % 2 == 0, canonical_sharded_tensor.local_shape
            half_out = local_out // 2
            out_axis = len(canonical_sharded_tensor.global_shape) - 2
            axis_frag = canonical_sharded_tensor.axis_fragmentations[out_axis]
            rank_offset = canonical_sharded_tensor.global_offset[out_axis] // local_out
            tensor_w, tensor_v = torch.chunk(tensor, 2, dim=-2)
            canonical_parts = []
            for part, part_rank in (
                (tensor_w, rank_offset),
                (tensor_v, rank_offset + axis_frag),
            ):
                global_offset = list(canonical_sharded_tensor.global_offset)
                axis_fragmentations = list(canonical_sharded_tensor.axis_fragmentations)
                global_offset[out_axis] = part_rank * half_out
                axis_fragmentations[out_axis] = axis_frag * 2
                canonical_parts.append(
                    (
                        part,
                        replace(
                            canonical_sharded_tensor,
                            data=part,
                            local_shape=tuple(part.shape),
                            global_offset=tuple(global_offset),
                            axis_fragmentations=tuple(axis_fragmentations),
                        ),
                    )
                )

        legacy_parts = []
        for part, part_shard in canonical_parts:
            data = part.transpose(-2, -1).contiguous()
            global_shape = list(part_shard.global_shape)
            global_offset = list(part_shard.global_offset)
            axis_fragmentations = list(part_shard.axis_fragmentations)
            global_shape[-2], global_shape[-1] = global_shape[-1], global_shape[-2]
            global_offset[-2], global_offset[-1] = global_offset[-1], global_offset[-2]
            axis_fragmentations[-2], axis_fragmentations[-1] = (
                axis_fragmentations[-1],
                axis_fragmentations[-2],
            )
            legacy_parts.append(
                replace(
                    part_shard,
                    key=legacy_key,
                    data=data,
                    dtype=data.dtype,
                    local_shape=tuple(data.shape),
                    global_shape=tuple(global_shape),
                    global_offset=tuple(global_offset),
                    axis_fragmentations=tuple(axis_fragmentations),
                    replica_id=replica_id,
                )
            )
        return legacy_parts

    @torch.no_grad()
    def merge_fn(loaded_shards):
        canonical_parts = [
            shard.transpose(-2, -1).contiguous() for shard in loaded_shards
        ]
        return torch.cat(canonical_parts, dim=-2) if gated else canonical_parts[0]

    return ShardedTensorFactory(
        canonical_sharded_tensor.key,
        canonical_sharded_tensor.data,
        build_fn,
        merge_fn,
        canonical_sharded_tensor.replica_id,
    )


def build_offloading_expert_sharded_tensor(
    weight_slice: torch.Tensor,
    prefix: str,
    weight_name: str,
    global_expert_idx: int,
    *,
    sharded_offsets: tuple,
    num_global_experts: int,
    replica_id,
    singleton_local_shards: bool,
    transpose: bool,
):
    """Build one shard in the legacy offloading expert schema."""
    data = weight_slice.transpose(0, 1).contiguous() if transpose else weight_slice
    if singleton_local_shards:
        key = f'{prefix}experts.{global_expert_idx}.{weight_name}'
        offsets = sharded_offsets
    else:
        key = f'{prefix}experts.{weight_name}'
        offsets = (
            *sharded_offsets,
            (len(sharded_offsets), global_expert_idx, num_global_experts),
        )
    return ShardedTensor.from_rank_offsets(
        key,
        data,
        *offsets,
        replica_id=replica_id,
        prepend_axis_num=len(offsets),
    )


def make_fused_experts_sharded_factory(
    fused_weight: torch.Tensor,
    prefix: str,
    weight_name: str,
    *,
    num_local_experts: int,
    local_expert_indices_offset: int,
    num_global_experts: int,
    sharded_offsets: tuple,
    replica_id,
    singleton_local_shards: bool,
):
    """Map a fused inplace-FP8 master to the legacy offloading schema."""

    @torch.no_grad()
    def build_fn(key, tensor, rep_id, flattened_range):
        assert flattened_range is None, "flattening unsupported for offloaded experts"
        assert key.endswith(weight_name), (key, weight_name)
        runtime_prefix = key[: -len(weight_name)]
        return [
            build_offloading_expert_sharded_tensor(
                tensor[local_idx],
                runtime_prefix,
                weight_name,
                local_expert_indices_offset + local_idx,
                sharded_offsets=sharded_offsets,
                num_global_experts=num_global_experts,
                replica_id=rep_id,
                singleton_local_shards=singleton_local_shards,
                transpose=True,
            )
            for local_idx in range(num_local_experts)
        ]

    @torch.no_grad()
    def merge_fn(loaded_shards):
        return torch.stack(
            [shard.transpose(0, 1) for shard in loaded_shards], dim=0
        ).contiguous()

    return ShardedTensorFactory(
        f'{prefix}{weight_name}', fused_weight, build_fn, merge_fn, replica_id
    )
