# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.

"""Bookkeeping of the KDA state recycling (CPU only; the layer itself needs the FLA kernels)."""

import torch

from megatron.core.ssm import state_recycle
from megatron.core.ssm.state_recycle import RecycleBuffer, local_recycled_slots, num_recycled


def _positions(dp_size, mbs, num_microbatches, n):
    """Global positions that are recycled, from every rank's local slots."""
    out = []
    for m in range(num_microbatches):
        for r in range(dp_size):
            for b in local_recycled_slots(m, r, dp_size, mbs, n):
                out.append((m * dp_size + r) * mbs + b)
    return sorted(out)


def test_first_fraction_of_the_global_batch():
    # 32k leg: GBS 96, DP 48, MBS 1 -> 2 micro-batches; 10% = 10 positions = micro-batch 0 of ranks 0-9
    n = num_recycled(0.1, 96)
    assert n == 10
    assert _positions(48, 1, 2, n) == list(range(10))
    assert local_recycled_slots(0, 9, 48, 1, n) == [0]
    assert local_recycled_slots(0, 10, 48, 1, n) == []
    assert local_recycled_slots(1, 0, 48, 1, n) == []
    # 4k ladder geometry: GBS 768, DP 256, MBS 3 -> 1 micro-batch; 77 positions over ranks 0-25
    n = num_recycled(0.1, 768)
    assert n == 77
    assert _positions(256, 3, 1, n) == list(range(77))
    assert local_recycled_slots(0, 25, 256, 3, n) == [0, 1]
    assert num_recycled(0.0, 96) == 0 and local_recycled_slots(0, 0, 48, 1, 0) == []


def test_buffer_is_one_iteration_behind_and_replay_safe():
    buf = RecycleBuffer()
    shape = (2, 3, 4, 4)
    final_a = torch.randn(shape)
    final_b = torch.randn(shape)

    buf.start(0)
    s0 = buf.initial_state(0, [0, 1], shape, "cpu")
    assert torch.equal(s0, torch.zeros(shape))                 # nothing stored yet
    buf.store(0, [0, 1], final_a)

    buf.start(1)
    s1 = buf.initial_state(0, [0, 1], shape, "cpu")
    assert torch.equal(s1, final_a)                             # iteration 0's finals
    buf.store(0, [0, 1], final_b)
    # a replayed micro-batch of the same iteration reads the same states and rewrites the same values
    buf.start(1)
    assert torch.equal(buf.initial_state(0, [0, 1], shape, "cpu"), final_a)
    buf.store(0, [0, 1], final_b)

    buf.start(2)
    s2 = buf.initial_state(0, [0, 1], shape, "cpu")
    assert torch.equal(s2, final_b)
    # a slot that was not recycled keeps a zero state; other micro-batches are separate keys
    assert torch.equal(buf.initial_state(0, [1], shape, "cpu")[0], torch.zeros(shape[1:]))
    assert torch.equal(buf.initial_state(1, [0, 1], shape, "cpu"), torch.zeros(shape))
    assert s2.dtype == torch.float32
    final_b[0].zero_()                                          # stored states are copies
    assert not torch.equal(buf.initial_state(0, [0], shape, "cpu")[0], torch.zeros(shape[1:]))


def test_iteration_context():
    state_recycle.begin_iteration(7, 96)
    assert state_recycle.current() == (7, -1, 96)
    state_recycle.next_microbatch()
    state_recycle.next_microbatch()
    assert state_recycle.current()[1] == 1
    state_recycle.begin_iteration(8, 96)
    assert state_recycle.current() == (8, -1, 96)
    state_recycle.begin_iteration(None, 96)                     # no iteration given: counts on
    assert state_recycle.current()[0] == 9
