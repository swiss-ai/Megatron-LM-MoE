# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.

"""Recycle the KDA recurrent state of a fixed share of every global batch into the next one.

A KDA layer that starts every training sequence from a zero state can read the absolute position
from how far its rows have filled since the start; that count is the position code which breaks
long-context extrapolation. If a share of the sequences instead starts from the final state that
another sequence left behind, the count is no longer a reliable signal, and the model also trains
on carrying a state across unrelated context.

Which samples: the first ``ceil(fraction * global_batch_size)`` positions of every global batch.
Position p is the p-th sample in the order the sampler hands them out; with Megatron's sampler,
micro-batch m on data-parallel rank r holds positions ``(m * dp_size + r) * mbs + [0, mbs)``.
Position p of iteration n starts from the final state that position p of iteration n-1 produced:
one state per recycled position, not per document. Without cross-document masking the packed
documents inside a sequence share the recurrence, and the stored state is the one at the end of
the sequence. With cross-document masking (cu_seqlens) every document restarts the recurrence:
the stored state is the final state of the sequence's first document, and in the next iteration
every document of that position's sequence starts from it (the non-recycled positions start every
document from zero, as usual). States stay on the rank that produced
them, so there is no communication; under context parallelism every CP rank keeps its own heads.
They are detached (no gradient across iterations) and are not checkpointed: after a restart the
recycled positions begin from zero for one iteration.

Hooks in the training loop: ``begin_iteration`` right before each forward/backward (inside the
rerun loop, so a replayed iteration counts its micro-batches again from zero) and
``next_microbatch`` at the start of every forward_step. A replay of the same iteration reads the
same previous states and rewrites the same new ones, so it reproduces the first run. Evaluation
forwards (module in eval mode) do not recycle.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch

_iteration: Optional[int] = None
_global_batch_size: int = 0
_microbatch: int = -1


def begin_iteration(iteration: Optional[int], global_batch_size: int) -> None:
    """Start counting the micro-batches of one optimizer step (``iteration`` None: count internally)."""
    global _iteration, _global_batch_size, _microbatch
    if iteration is None:
        iteration = (_iteration or 0) + 1
    _iteration, _global_batch_size, _microbatch = iteration, global_batch_size, -1


def next_microbatch() -> None:
    global _microbatch
    _microbatch += 1


def current() -> Tuple[Optional[int], int, int]:
    """(iteration, micro-batch index within it, global batch size); iteration is None before the first step."""
    return _iteration, _microbatch, _global_batch_size


def num_recycled(fraction: float, global_batch_size: int) -> int:
    """How many positions of a global batch are recycled."""
    return math.ceil(fraction * global_batch_size) if fraction > 0 else 0


def local_recycled_slots(
    microbatch: int, dp_rank: int, dp_size: int, micro_batch_size: int, n_recycled: int
) -> List[int]:
    """Indices into this rank's micro-batch whose global position is below ``n_recycled``."""
    first = (microbatch * dp_size + dp_rank) * micro_batch_size
    return [b for b in range(micro_batch_size) if first + b < n_recycled]


class RecycleBuffer:
    """Per-layer store of final states keyed by (micro-batch, local index), one iteration behind.

    ``start(iteration)`` must be called at every forward: the first call of a new iteration makes
    the states written during the previous iteration readable and starts a fresh write set.
    """

    def __init__(self) -> None:
        self._iteration: Optional[int] = None
        self._prev: Dict[Tuple[int, int], torch.Tensor] = {}
        self._next: Dict[Tuple[int, int], torch.Tensor] = {}

    def start(self, iteration: int) -> None:
        if iteration != self._iteration:
            self._prev, self._next, self._iteration = self._next, {}, iteration

    def initial_state(
        self, microbatch: int, slots: List[int], shape: Tuple[int, ...], device
    ) -> torch.Tensor:
        """fp32 [batch, heads, k, v]: zeros, except the recycled slots that have a stored state."""
        state = torch.zeros(shape, dtype=torch.float32, device=device)
        for b in slots:
            prev = self._prev.get((microbatch, b))
            if prev is not None:
                state[b] = prev
        return state

    def store(self, microbatch: int, slots: List[int], final_state: torch.Tensor) -> None:
        """Keep ``final_state[b]`` for every slot b (with cu_seqlens: row 0 = the first document)."""
        for b in slots:
            self._next[(microbatch, b)] = final_state[b].detach().float().clone()

    def __len__(self) -> int:
        return len(self._prev)
