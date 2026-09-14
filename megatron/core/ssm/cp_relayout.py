# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.
"""Move context-parallel token shards between the attention layout and contiguous slices.

Softmax attention keeps every document split into ``2 * cp`` zigzag chunks for load
balance, and sequence parallelism then takes a contiguous slice of that. FLA's
context-parallel delta-rule kernels instead want each rank to hold one contiguous
slice of the packed sequence. A plan describes the permutation once per batch and a
single all-to-all applies it in either direction.
"""
from dataclasses import dataclass

import torch


def zigzag_indices(cu_seqlens, cp_size, cp_rank):
    """Global token ids held by ``cp_rank`` after TE's per-document zigzag partition."""
    starts = cu_seqlens[:-1]
    width = (cu_seqlens[1:] - starts) // (2 * cp_size)
    chunk_starts = torch.stack(
        (starts + cp_rank * width, starts + (2 * cp_size - 1 - cp_rank) * width), dim=1
    ).flatten()
    widths = width.repeat_interleave(2)
    offsets = widths.cumsum(0) - widths
    return torch.arange(int(widths.sum())) + (chunk_starts - offsets).repeat_interleave(widths)


def _layout_indices(cu_seqlens, cp_size, sp_size, batch, rank):
    """Global token ids of rank ``rank``'s ``[seq, batch]`` hidden states, flattened."""
    cp_rank, sp_rank = divmod(rank, sp_size)
    ids = zigzag_indices(cu_seqlens, cp_size, cp_rank).view(batch, sp_size, -1)
    return ids[:, sp_rank].t().flatten()


@dataclass
class RelayoutPlan:
    group: torch.distributed.ProcessGroup
    send_order: torch.Tensor
    receive_order: torch.Tensor
    send_back: torch.Tensor
    receive_back: torch.Tensor
    input_splits: list
    output_splits: list


def build_relayout_plan(cu_seqlens_cpu, cp_size, sp_size, batch, group, device):
    """Plan the all-to-all for ``group.rank()``; ranks are ordered ``cp_rank * sp_size + sp_rank``."""
    world = cp_size * sp_size
    rank = group.rank()
    slice_len = int(cu_seqlens_cpu[-1]) // world
    layouts = [_layout_indices(cu_seqlens_cpu, cp_size, sp_size, batch, r) for r in range(world)]
    destinations = layouts[rank] // slice_len
    send_order = torch.argsort(destinations, stable=True)
    received = [ids[ids // slice_len == rank] for ids in layouts]
    receive_order = torch.argsort(torch.cat(received))
    return RelayoutPlan(
        group,
        send_order.to(device),
        receive_order.to(device),
        torch.argsort(receive_order).to(device),
        torch.argsort(send_order).to(device),
        torch.bincount(destinations, minlength=world).tolist(),
        [ids.numel() for ids in received],
    )


def _relayout(x, plan, inverse):
    if inverse:
        send = x.index_select(0, plan.send_back)
        received = torch.empty_like(send)
        torch.distributed.all_to_all_single(
            received, send, plan.input_splits, plan.output_splits, group=plan.group
        )
        return received.index_select(0, plan.receive_back)
    send = x.index_select(0, plan.send_order)
    received = torch.empty_like(send)
    torch.distributed.all_to_all_single(
        received, send, plan.output_splits, plan.input_splits, group=plan.group
    )
    return received.index_select(0, plan.receive_order)


class _Relayout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, plan, inverse):
        ctx.plan, ctx.inverse = plan, inverse
        return _relayout(x, plan, inverse)

    @staticmethod
    def backward(ctx, grad):
        return _relayout(grad, ctx.plan, not ctx.inverse), None, None


def relayout(x, plan, inverse=False):
    """Contiguous slice of the attention-layout tensor ``x`` (or back, with ``inverse``)."""
    return _Relayout.apply(x, plan, inverse)
