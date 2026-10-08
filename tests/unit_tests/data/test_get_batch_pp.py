# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.

"""cu_seqlens must reach every pipeline stage intact.

With cross-document masking, each pipeline stage masks its own local attention
layers, so every stage needs the same cu_seqlens. TP rank 0 of each stage reads
it from the dataloader; the other TP ranks receive it by broadcast. A middle
stage (neither first nor last) is the case with no tokens to broadcast and
therefore its own branch in get_batch_on_this_tp_rank, so it needs TP > 1 and
PP >= 3 together to be exercised at all.
"""

import os

import pytest
import torch

from megatron.core import mpu
from megatron.core.utils import flatten_batch_for_packed_sequences
from megatron.training.global_vars import set_args
from megatron.training.utils import get_batch_on_this_tp_rank
from tests.unit_tests.test_utilities import Utils


class _Args:
    """Only the attributes get_batch_on_this_tp_rank actually reads."""

    def __init__(self, pp, mbs, seq_length):
        self.pipeline_model_parallel_size = pp
        self.micro_batch_size = mbs
        self.seq_length = seq_length
        self.sft = False
        self.dataloader_inter_document_masking = True
        self.hybrid_context_parallel = False
        self.create_attention_mask_in_dataloader = False


def _fake_batch(mbs, seq_length, doc_lengths):
    """One collated micro-batch, identical on every rank that builds it."""
    cu = torch.full((mbs, seq_length + 1), seq_length, dtype=torch.int32)
    max_seqlen = torch.empty(mbs, dtype=torch.int32)
    for s in range(mbs):
        acc, row = 0, [0]
        for length in doc_lengths:
            acc += length
            row.append(acc)
        row[-1] = seq_length
        cu[s, : len(row)] = torch.tensor(row, dtype=torch.int32)
        max_seqlen[s] = max(b - a for a, b in zip(row[:-1], row[1:]))
    g = torch.Generator().manual_seed(1234)
    return {
        "tokens": torch.randint(0, 100, (mbs, seq_length), generator=g),
        "labels": torch.randint(0, 100, (mbs, seq_length), generator=g),
        "loss_mask": torch.ones(mbs, seq_length),
        "position_ids": torch.zeros(mbs, seq_length, dtype=torch.long),
        "cu_seqlens": cu,
        "max_seqlen": max_seqlen,
    }


@pytest.mark.internal
@pytest.mark.parametrize("mbs", [1, 2])
@pytest.mark.parametrize('ap_sft,report_assistant', [(False, False), (True, False), (True, True)])
def test_cu_seqlens_reaches_every_pipeline_stage(mbs, ap_sft, report_assistant):
    # Read the launcher's world size, not torch.distributed's: this runs before
    # Utils.initialize_model_parallel has initialized the process group.
    world = Utils.world_size
    tp = 2
    pp = world // tp
    if world < 4 or world % tp != 0:
        pytest.skip(f"needs a world size divisible by {tp} and >= 4, got {world}")

    Utils.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp)
    seq_length, doc_lengths = 64, [17, 20, 27]
    args = _Args(pp, mbs, seq_length)
    args.ap_sft = ap_sft
    args.ap_sft_report_assistant_loss = report_assistant
    args.dataloader_inter_document_masking = not ap_sft
    set_args(args)

    # Only TP rank 0 of each stage has a dataloader, mirroring
    # is_dataset_built_on_rank for a packed batch.
    if mpu.get_tensor_model_parallel_rank() == 0:
        batch = _fake_batch(mbs, seq_length, doc_lengths)
        if ap_sft:
            batch['cu_seqlens_padded'] = batch['cu_seqlens'].clone()
            batch['padding_mask'] = torch.zeros(mbs, seq_length, dtype=torch.bool)
            batch['padding_mask'][:, -5:] = True
            if report_assistant:
                batch['assistant_mask'] = ~batch['padding_mask']
        data_iterator = iter([{k: v.cuda() for k, v in batch.items()}])
    else:
        data_iterator = None

    got = flatten_batch_for_packed_sequences(get_batch_on_this_tp_rank(data_iterator))

    cu = got["cu_seqlens"]
    assert cu is not None, (
        f"cu_seqlens is None on pp_rank={mpu.get_pipeline_model_parallel_rank()} "
        f"tp_rank={mpu.get_tensor_model_parallel_rank()}: that stage would run "
        f"its attention layers unmasked"
    )
    assert got["max_seqlen"] is not None
    if ap_sft:
        assert torch.equal(got['cu_seqlens'], got['cu_seqlens_padded'])
        expected_padding = torch.zeros(mbs, seq_length, dtype=torch.bool, device='cuda')
        expected_padding[:, -5:] = True
        assert torch.equal(got['padding_mask'], expected_padding.reshape(1, -1))
        if report_assistant:
            assert torch.equal(got['assistant_mask'], (~expected_padding).reshape(1, -1))

    # Every rank in the world must hold byte-identical cu_seqlens: across TP
    # (the broadcast) and across PP (every stage masks with the same boundaries).
    expected = flatten_batch_for_packed_sequences(
        {k: v.cuda() for k, v in _fake_batch(mbs, seq_length, doc_lengths).items()}
    )["cu_seqlens"]
    assert torch.equal(cu, expected), (
        f"cu_seqlens mismatch on pp_rank={mpu.get_pipeline_model_parallel_rank()} "
        f"tp_rank={mpu.get_tensor_model_parallel_rank()}: {cu} != {expected}"
    )

    gathered = [torch.zeros_like(cu) for _ in range(world)]
    torch.distributed.all_gather(gathered, cu.contiguous())
    for rank, other in enumerate(gathered):
        assert torch.equal(cu, other), f"rank {rank} disagrees: {other} != {cu}"

    Utils.destroy_model_parallel()


if __name__ == "__main__":
    pytest.main([__file__])
