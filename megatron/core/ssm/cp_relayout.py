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
    # Every rank already holds its contiguous slice in order: relayouts are no-ops.
    identity: bool = False


def build_relayout_plan(cu_seqlens_cpu, cp_size, sp_size, batch, group, device):
    """Plan the all-to-all for ``group.rank()``; ranks are ordered ``cp_rank * sp_size + sp_rank``."""
    world = cp_size * sp_size
    slice_len = int(cu_seqlens_cpu[-1]) // world
    layouts = [_layout_indices(cu_seqlens_cpu, cp_size, sp_size, batch, r) for r in range(world)]
    return _plan_from_layouts(layouts, slice_len, group, device)


def build_layout_relayout_plan(virtual_pos, seq_len, sp_size, group, device, moves=True):
    """Plan for an arbitrary layout: ``virtual_pos[cp_rank]`` gives, for each of the CP rank's
    ``seq_len`` tokens in order, its position in the contiguous (KCP) buffer. Sequence
    parallelism splits each CP rank's tokens into ``sp_size`` contiguous pieces."""
    if not moves:
        return RelayoutPlan(group, None, None, None, None, None, None, identity=True)
    slice_len = seq_len // sp_size
    pos = torch.as_tensor(virtual_pos).reshape(-1, sp_size, slice_len)
    layouts = [pos[r // sp_size, r % sp_size] for r in range(pos.shape[0] * sp_size)]
    return _plan_from_layouts(layouts, slice_len, group, device)


def _plan_from_layouts(layouts, slice_len, group, device):
    """``layouts[r]``: destination-buffer positions of rank r's tokens, in its local order."""
    world = len(layouts)
    rank = group.rank()
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
    if plan.identity:
        return x
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
    if plan.identity:
        return x
    return _Relayout.apply(x, plan, inverse)


def _kcp_attention(layer):
    """The layer's KCP attention module when it relayouts under CP, else None."""
    attention = getattr(layer, "self_attention", None)
    if getattr(attention, "_kcp", False) and getattr(attention, "cp_size", 1) > 1:
        return attention
    return None


class BlockLayout:
    """Keep a block's residual stream in the contiguous layout across runs of KCP layers.

    A KCP layer relayouts its input to contiguous slices and its output back, two
    all-to-alls of the hidden states per layer. Everything between two KCP layers (norms,
    residual adds, MLP/MoE) is token-wise and does not care which rank holds which token,
    so the residual stream can stay contiguous for a whole run of KCP layers and be
    restored only before a layer that needs the attention layout (softmax attention) and
    at the end of the block. The MoE routing ``padding_mask`` moves with the tokens.
    """

    def __init__(self, layers):
        self.modules = [_kcp_attention(layer) for layer in layers]
        self.plan = None
        self.shape = None
        self.zigzag_padding_mask = None

    @staticmethod
    def applies(layers, packed_seq_params):
        """Only packed (THD, batch 1) inputs: the plan folds the batch into the sequence."""
        return packed_seq_params is not None and any(_kcp_attention(l) for l in layers)

    def before_layer(self, index, hidden_states, packed_seq_params, padding_mask):
        """Hidden states and routing mask in the layout layer ``index`` wants."""
        module = self.modules[index]
        if module is not None:
            module.kcp_external_layout = True
            if self.plan is None:
                seq_len, batch, hidden = hidden_states.shape
                self.plan, _ = module._kcp_layout(hidden_states, packed_seq_params, seq_len, batch)
                self.shape = hidden_states.shape
                hidden_states = relayout(hidden_states.reshape(-1, 1, hidden), self.plan)
                self.zigzag_padding_mask = padding_mask
                if padding_mask is not None:
                    # [b, s] bool -> [s * b, 1, 1] so the token permutation applies.
                    moved = _relayout(
                        padding_mask.transpose(0, 1).reshape(-1, 1, 1).to(torch.uint8),
                        self.plan,
                        inverse=False,
                    )
                    padding_mask = moved.reshape(1, -1).bool()
                self.contiguous_padding_mask = padding_mask
            return hidden_states, self.contiguous_padding_mask
        if self.plan is not None:
            hidden_states = self.finish(hidden_states)
        return hidden_states, padding_mask

    def finish(self, hidden_states):
        """Restore the attention layout if the stream is contiguous."""
        if self.plan is None:
            return hidden_states
        hidden_states = relayout(hidden_states, self.plan, inverse=True).view(self.shape)
        self.plan = None
        return hidden_states
