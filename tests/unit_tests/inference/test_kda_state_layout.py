import json
import os
from dataclasses import asdict
from functools import partial

import msgpack
import pytest
import torch

from megatron.core.inference.disaggregation.ssm_reshard import (
    KDAShardLayout,
    KDAStateDims,
    plan_ssm_reshard,
    state_layout_from_meta,
)
from megatron.core.inference.disaggregation.transfer_backends.base import (
    compute_buffer_geometry,
    export_geometry_meta,
)

DIMS = KDAStateDims(4, 6, 4, 7, 5)
LAYERS = (1, 4, 7)
CONV = torch.arange(3 * 74 * 5, dtype=torch.float32).reshape(3, 74, 5)
RECURRENT = torch.arange(3 * 6 * 4 * 7, dtype=torch.float32).reshape(3, 6, 4, 7)


def _shard(layout):
    indices = [LAYERS.index(layer) for layer in sorted(layout.layer_map, key=layout.layer_map.get)]
    qk = 16 // layout.tp_size
    value = 42 // layout.tp_size
    rank = layout.tp_rank
    conv = torch.cat(
        [
            CONV[indices, rank * qk : (rank + 1) * qk],
            CONV[indices, 16 + rank * qk : 16 + (rank + 1) * qk],
            CONV[indices, 32 + rank * value : 32 + (rank + 1) * value],
        ],
        dim=1,
    )
    heads = 6 // layout.tp_size
    return conv, RECURRENT[indices, rank * heads : (rank + 1) * heads]


def _layouts(tp, pp, start, reverse=False):
    result = []
    for stage in range(pp):
        layers = list(LAYERS[stage::pp])
        if reverse:
            layers.reverse()
        for rank in range(tp):
            result.append(
                KDAShardLayout(
                    start + stage * tp + rank,
                    tp,
                    rank,
                    {layer: index for index, layer in enumerate(layers)},
                    DIMS,
                )
            )
    return result


@pytest.mark.parametrize("source_tp,destination_tp", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("source_pp,destination_pp", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_shared_ssm_planner_reconstructs_kda(source_tp, destination_tp, source_pp, destination_pp):
    source = _layouts(source_tp, source_pp, 0)
    destination = _layouts(destination_tp, destination_pp, 20, reverse=True)
    source_tensors = {layout.global_rank: _shard(layout) for layout in source}
    target_tensors = {
        layout.global_rank: tuple(torch.full_like(tensor, -1) for tensor in _shard(layout))
        for layout in destination
    }
    for transfer in plan_ssm_reshard(source, destination):
        kind = 0 if transfer.is_conv else 1
        target_tensors[transfer.dst_rank][kind][
            transfer.dst_layer, transfer.dst_lo : transfer.dst_hi
        ] = source_tensors[transfer.src_rank][kind][
            transfer.src_layer, transfer.src_lo : transfer.src_hi
        ]
    for layout in destination:
        for actual, expected in zip(target_tensors[layout.global_rank], _shard(layout)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_kda_wire_and_dummy_slot_geometry():
    layout = KDAShardLayout(0, 1, 0, {4: 0, 7: 1}, DIMS)
    buffer = torch.zeros(2, 4, 6, 4, 7)
    geometry = compute_buffer_geometry(
        buffer,
        3,
        backend_name="test",
        ssm_layout=layout,
        ssm_state_kind="recurrent",
        heads_per_partition=6,
        head_dim=28,
        tokens_per_block=1,
    )
    meta = export_geometry_meta(geometry, layout, "recurrent", buffer.dtype)
    for restored in (json.loads(json.dumps(meta)), msgpack.unpackb(msgpack.packb(meta), raw=False)):
        assert state_layout_from_meta(restored) == layout
    assert geometry.num_blocks == 3
    assert geometry.outer_stride_bytes == buffer.stride(0) * buffer.element_size()
    assert meta["state_kind"] == "recurrent"
    assert meta["state_dtype"] == "torch.float32"
    assert "ssm_layout" not in meta


@pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) < 3,
    reason="Requires prefill TP2 and decode TP1 CUDA ranks",
)
@pytest.mark.parametrize("transport", ["nccl", "nixl-uccl"])
@pytest.mark.parametrize("state_kind", ["conv", "recurrent"])
def test_shared_transfer_reshards_kda_tp2_to_tp1(state_kind, transport):
    import torch.distributed as dist
    from megatron.core.inference.disaggregation.transfer_backends.nccl import NcclTransferBackend

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    control = dist.new_group(backend="gloo")
    dist.barrier()
    rank = dist.get_rank()
    kind = 0 if state_kind == "conv" else 1
    backend = None
    layout = None
    if rank < 3:
        layers = LAYERS if rank < 2 else tuple(reversed(LAYERS))
        layout = KDAShardLayout(
            rank,
            2 if rank < 2 else 1,
            rank if rank < 2 else 0,
            {layer: index for index, layer in enumerate(layers)},
            DIMS,
        )
        expected = _shard(layout)[kind].cuda()
        buffer = torch.zeros((3, 4) + tuple(expected.shape[1:]), device="cuda")
        backend_type = NcclTransferBackend
        if transport == "nixl-uccl":
            from megatron.core.inference.disaggregation.transfer_backends.nixl import (
                NixlTransferBackend,
            )

            backend_type = partial(NixlTransferBackend, nixl_backend="UCCL")
        backend = backend_type(
            agent_name=f"kda-{state_kind}-{rank}",
            memory_buffer=buffer,
            expected_num_blocks=3,
            heads_per_partition=layout.conv_dim_local if kind == 0 else layout.nheads_local,
            head_dim=5 if kind == 0 else 28,
            tokens_per_block=1,
            ssm_layout=layout,
            ssm_state_kind=state_kind,
            tp_size=layout.tp_size,
            tp_rank=layout.tp_rank,
        )
        if rank < 2:
            buffer[:, 1] = expected
    metadata = [None] * dist.get_world_size()
    dist.all_gather_object(metadata, backend.export_meta() if backend else None, group=control)
    success = True
    if rank < 2:
        assert backend is not None
        if backend.is_push:
            backend.begin_push_blocks({"tp_metas": [metadata[2]]}, [1]).wait()
    elif rank == 2:
        assert backend is not None
        source_meta = [{**metadata[source], "block_ids": [1]} for source in (0, 1)]
        backend.begin_pull_blocks({"tp_metas": source_meta}, [], [2]).wait()
        success = torch.equal(buffer[:, 2], expected) and buffer[:, 3].count_nonzero().item() == 0
    result = torch.tensor(int(success))
    dist.all_reduce(result, op=dist.ReduceOp.MIN, group=control)
    assert result.item() == 1
    if (
        transport == "nixl-uccl"
        and backend is not None
        and isinstance(backend, NixlTransferBackend)
    ):
        backend.close()


def test_kda_rejects_invalid_layout_and_recurrent_dtype():
    with pytest.raises(ValueError, match="divisible"):
        KDAShardLayout(0, 4, 0, {0: 0}, DIMS)
    with pytest.raises(ValueError, match="contiguous"):
        KDAShardLayout(0, 1, 0, {0: 2}, DIMS)
    layout = KDAShardLayout(**asdict(KDAShardLayout(0, 1, 0, {0: 0}, DIMS)))
    with pytest.raises(ValueError, match="FP32"):
        compute_buffer_geometry(
            torch.zeros(1, 4, 6, 4, 7, dtype=torch.bfloat16),
            3,
            backend_name="test",
            ssm_layout=layout,
            ssm_state_kind="recurrent",
        )
