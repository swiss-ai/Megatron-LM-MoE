# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.
"""Document-aware context parallelism inside a fixed CP group (``--cp-doc-aware``).

Short documents stay whole on one CP rank and pay no CP communication; only long
documents are split over the CP group. The per-sample assignment is
``doc_aware_cp_plan.plan_doc_aware``; this module turns it into this rank's batch
slice and the ``DocAwareLayout`` that the layers read from ``PackedSeqParams.doc_aware``:

* every rank holds ``seq_len`` tokens, ``[local documents | padding | long zigzag]``,
  so pipeline p2p shapes and the MoE dispatch are those of an ordinary CP batch;
* ``TEDotProductAttention`` runs the local part without CP and the long part with
  the CP group (TE's THD ring, per-document zigzag);
* KCP (KDA) runs on the plan's virtual buffer. If no document crosses a CP slice of
  it, KDA runs locally without the state exchange; if the layout already is the
  virtual buffer's slicing, the relayout is skipped.
"""
from typing import Any, Dict, Optional

import torch

from megatron.core.doc_aware_cp_plan import DocAwarePlan, plan_doc_aware
from megatron.core.packed_seq_params import PackedSeqParams


class DocAwareLayout:
    """One micro-batch's document-aware layout, as seen by CP rank ``cp_rank``."""

    def __init__(self, plan: DocAwarePlan, cp_rank: int, device):
        self.plan = plan
        self.cp_rank = cp_rank
        self.device = device
        self.local_len = plan.local_len
        self.seq_len = plan.seq_len
        self.has_local = plan.local_len > 0
        self.has_long = plan.long_len > 0
        local_cu = torch.from_numpy(plan.local_cu[cp_rank]).to(torch.int32)
        self.local_params = _thd_params(local_cu.to(device), plan.local_max[cp_rank])
        long_cu = torch.from_numpy(plan.long_cu).to(torch.int32)
        self.long_params = _thd_params(long_cu.to(device), plan.long_max)
        self.virtual_cu_cpu = torch.from_numpy(plan.virtual_cu)
        self.virtual_cu = self.virtual_cu_cpu.to(device)
        # Filled lazily by the first KDA layer of the micro-batch, shared by the others.
        self.kcp_state = None

    def kcp(self, group, sp_size: int, cp_group, conv1d_kernel_size: int):
        """(relayout plan, FLA cp_context or None, local cu_seqlens or None) for KCP.

        ``cp_context`` is None when no document crosses a CP slice of the virtual
        buffer: every rank then runs KDA on its own slice with ``local_cu``.
        """
        if self.kcp_state is None:
            from megatron.core.ssm.cp_relayout import build_layout_relayout_plan

            plan = self.plan
            relayout_plan = build_layout_relayout_plan(
                plan.virtual_pos, plan.seq_len, sp_size, group, self.device, moves=plan.moves
            )
            if plan.spans:
                from fla.ops.cp import build_cp_context

                cp_context = build_cp_context(
                    self.virtual_cu,
                    group=cp_group,
                    conv1d_kernel_size=conv1d_kernel_size,
                    cu_seqlens_cpu=self.virtual_cu_cpu,
                )
                local = None
            else:
                start, stop = self.cp_rank * plan.seq_len, (self.cp_rank + 1) * plan.seq_len
                cu = self.virtual_cu_cpu
                cu = cu[(cu >= start) & (cu <= stop)] - start
                cp_context, local = None, (cu.to(self.device), cu)
            self.kcp_state = (relayout_plan, cp_context, local)
        return self.kcp_state


def _thd_params(cu_seqlens, max_seqlen):
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=int(max_seqlen),
        max_seqlen_kv=int(max_seqlen),
    )


def get_doc_aware_batch_on_this_cp_rank(
    batch: Dict[str, Any],
    cu_seqlens: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    sp_size: int = 1,
    local_multiple: int = 1,
    max_local_len: Optional[int] = None,
    max_padding: float = 0.01,
):
    """This CP rank's slice of a packed sample under the document-aware plan.

    Every key of ``batch`` with a sequence dimension (dim 1, batch 1) is gathered into
    the rank's layout; padding positions get zeros (``loss_mask`` 0) and are flagged in
    ``padding_mask`` so MoE routing ignores them. Middle pipeline stages may hold no
    token tensors: the plan only needs ``cu_seqlens``.

    Returns the batch and the ``PackedSeqParams`` carrying the ``DocAwareLayout``.
    """
    cu_cpu = cu_seqlens.detach().to("cpu", torch.int64)
    plan = plan_doc_aware(
        cu_cpu.numpy(),
        cp_size,
        sp=sp_size,
        local_multiple=local_multiple,
        max_local_len=max_local_len,
        max_padding=max_padding,
    )
    device = cu_seqlens.device
    layout = DocAwareLayout(plan, cp_rank, device)
    sources = torch.from_numpy(plan.sources[cp_rank]).to(device)
    is_pad = sources < 0
    gather = sources.clamp(min=0)

    padding_mask = is_pad.unsqueeze(0)
    if batch.get('padding_mask') is not None:
        padding_mask = padding_mask | batch['padding_mask'].index_select(1, gather)
    for key, data in batch.items():
        if key in ('attention_mask', 'padding_mask') or data is None:
            continue
        data = data.index_select(1, gather)
        batch[key] = data.masked_fill(is_pad.unsqueeze(0), 0)
    batch['padding_mask'] = padding_mask

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=layout.virtual_cu,
        cu_seqlens_kv=layout.virtual_cu,
        cu_seqlens_q_padded=layout.virtual_cu,
        cu_seqlens_kv_padded=layout.virtual_cu,
        max_seqlen_q=max(plan.long_max, max(plan.local_max)),
        max_seqlen_kv=max(plan.long_max, max(plan.local_max)),
        doc_aware=layout,
    )
    return batch, packed_seq_params
