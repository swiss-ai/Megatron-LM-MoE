# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Attention Residuals (AttnRes) -- Block variant.
#
# Reference: "Attention Residuals" (Kimi Team, 2026, arXiv:2603.15031). Replaces the
# additive residual with a depth-wise softmax attention over previous block
# representations:  h = sum_i softmax(w^T RMSNorm(V_i)) * V_i.
#
# This is a native port that reuses the mHC packed-tensor scaffolding: the residual
# stream is carried as a single [s, b, n*C] tensor with n = num_blocks + 1 slots
# (slot 0 = the embedding block b_0; slots 1..N = the N block representations). Because
# it reuses ``HyperConnectionTransformerLayer`` and the block-level expand/contract
# seams, it inherits mHC's pipeline-parallel comm, checkpoint and optimizer plumbing
# unchanged -- to Megatron the stream is just one wide tensor.
#
# Only the *block* variant is implemented here (the paper's practical choice; Full
# AttnRes -- attend over every sublayer output, O(L*d) -- is intentionally not ported).

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

_ATTNRES_NORM_EPS = 1e-6


def compute_block_sizes(num_sublayers: int, num_blocks: int) -> List[int]:
    """Distribute ``num_sublayers`` sublayers as evenly as possible among ``num_blocks`` blocks.

    Matches the reference ``Transformer._compute_block_sizes`` (leading blocks get the
    remainder). ``num_sublayers`` is ``2 * num_layers`` (attention + MLP counted separately).
    """
    base = num_sublayers // num_blocks
    extra = num_sublayers % num_blocks
    return [base + (1 if i < extra else 0) for i in range(num_blocks)]


def build_block_schedule(num_layers: int, num_blocks: int) -> List[Tuple[int, int]]:
    """Precompute, per global sublayer index j in [0, 2*num_layers), the pair
    ``(num_agg, write_idx)``:

    * ``num_agg``   -- aggregate a softmax over the first ``num_agg`` packed slots
                       (the completed/frozen block slots plus the current partial slot
                       once it has received its first write). Active slots are always a
                       contiguous prefix ``[0, num_agg)``.
    * ``write_idx`` -- the packed slot the sublayer output accumulates into (the block
                       currently being built). Slot 0 is the embedding; block ``c`` maps
                       to slot ``c + 1``.

    This mirrors the reference ``_forward_block_attnres`` bookkeeping exactly, but as a
    static schedule (independent of the runtime tensors), so each module can recover its
    own entry from its global sublayer index regardless of pipeline stage.
    """
    num_sublayers = 2 * num_layers
    sizes = compute_block_sizes(num_sublayers, num_blocks)

    schedule: List[Tuple[int, int]] = []
    completed = 1  # frozen slots 0..completed-1; slot 0 = embedding b_0
    current_block = 0
    sublayer_in_block = 0
    partial_started = False
    for _ in range(num_sublayers):
        # Block boundary: freeze the finished partial slot and advance to the next block.
        if sublayer_in_block >= sizes[current_block]:
            if partial_started:
                completed += 1
                partial_started = False
            current_block += 1
            sublayer_in_block = 0
        num_agg = completed + (1 if partial_started else 0)
        write_idx = completed  # first non-frozen slot = the current partial block
        schedule.append((num_agg, write_idx))
        partial_started = True
        sublayer_in_block += 1
    return schedule


def _attn_res_aggregate(v: Tensor, w: Tensor, norm_weight: Tensor, eps: float) -> Tensor:
    """Depth-wise softmax aggregation over the slot dimension (paper Eq. 2-4).

    Args:
        v: [s, b, k, C] -- the ``k`` active source slots.
        w: [C] -- the zero-init pseudo-query for this aggregation point.
        norm_weight: [C] -- RMSNorm gain used only to compute the attention logits.
        eps: RMSNorm epsilon.

    Returns:
        [s, b, C] -- softmax-weighted sum of the raw (un-normed) source slots.
    """
    dtype = v.dtype
    vf = v.float()
    # RMSNorm over C, applied only to the *keys* used for the logits.
    rms = vf.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    k = vf * rms * norm_weight.float()
    logits = torch.einsum("c, s b k c -> s b k", w.float(), k)  # [s, b, k]
    weights = logits.softmax(dim=-1)
    # Weighted sum of the raw source slots (not the normed keys), matching the reference.
    agg = torch.einsum("s b k, s b k c -> s b c", weights, vf)
    return agg.to(dtype)


class AttnResModule(MegatronModule):
    """One depth-wise attention-residual aggregation point (Block AttnRes).

    Implements the same call contract as :class:`HyperConnectionModule` so it can be
    dropped into ``HyperConnectionTransformerLayer``'s self-attention / MLP slots:

    * ``forward(hidden_states)`` aggregates the active packed slots into the single-stream
      input for the sublayer, and passes the packed residual through unchanged.
    * ``fused_h_res_h_post_bda(...)`` accumulates the sublayer output (after
      bias-dropout) into this point's ``write_idx`` block slot, leaving frozen slots
      untouched.

    Unlike mHC there is no dynamic projection / Sinkhorn / gating: aggregation weights come
    from a single per-point pseudo-query ``w`` (zero-initialized, so training starts from a
    uniform average of active slots == standard residual behaviour) plus an RMSNorm gain.

    Args:
        config: TransformerConfig (``num_residual_streams`` = num_blocks + 1).
        layer_number: 1-indexed global layer number (with pipeline offset applied).
        sublayer_type: ``"attn"`` or ``"mlp"`` -- selects this point's global sublayer index.
    """

    def __init__(self, config: TransformerConfig, layer_number: int, sublayer_type: str):
        super().__init__(config)
        assert sublayer_type in ("attn", "mlp"), sublayer_type
        self.config = config
        self.layer_number = layer_number
        self.sublayer_type = sublayer_type
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.norm_eps = _ATTNRES_NORM_EPS

        # Global 0-indexed sublayer index j = 2*layer_idx + {0 for attn, 1 for mlp}.
        layer_idx = layer_number - 1
        j = 2 * layer_idx + (0 if sublayer_type == "attn" else 1)
        schedule = build_block_schedule(config.num_layers, config.attn_residual_num_blocks)
        self.num_agg, self.write_idx = schedule[j]

        # Per-point pseudo-query -- MUST be zero-initialized (paper Sec 5).
        self.w = nn.Parameter(torch.zeros(self.hidden_size))
        # RMSNorm gain for the attention keys.
        self.norm_weight = nn.Parameter(torch.ones(self.hidden_size))

        # One-hot over slots selecting the write target, for the additive (autograd-safe)
        # accumulation in the merge step. Registered as a buffer so it follows device/dtype.
        onehot = torch.zeros(self.n)
        onehot[self.write_idx] = 1.0
        self.register_buffer("slot_onehot", onehot, persistent=False)

        # Non-TP-aware params (operate on full C of the sequence-sharded stream): their
        # gradients must be all-reduced across TP ranks when sequence parallel is enabled.
        if config.sequence_parallel:
            setattr(self.w, "sequence_parallel", True)
            setattr(self.norm_weight, "sequence_parallel", True)

    def forward(
        self, hidden_states: Tensor, mhc_recompute_manager: Optional[object] = None
    ) -> Tuple[Tensor, None, None, Tensor]:
        """Aggregate active slots -> single-stream sublayer input; pass the residual through.

        Args:
            hidden_states: [s, b, n*C] -- packed residual streams.
            mhc_recompute_manager: unused (native port; kept for signature compatibility).

        Returns:
            aggregated: [s, b, C] -- input for the sublayer (before input_layernorm).
            None, None: placeholders for mHC's (h_res, h_post); AttnRes needs no mixing state.
            residual: [s, b, n*C] -- the packed streams, passed through to the merge.
        """
        s, b, _ = hidden_states.shape
        streams = hidden_states.view(s, b, self.n, self.hidden_size)
        active = streams[:, :, : self.num_agg, :]  # contiguous prefix of active slots
        aggregated = _attn_res_aggregate(active, self.w, self.norm_weight, self.norm_eps)
        return aggregated, None, None, hidden_states

    def fused_h_res_h_post_bda(
        self,
        h_res: None,
        original_residual: Tensor,
        h_post: None,
        layer_output_with_bias: Tuple[Tensor, Optional[Tensor]],
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager: Optional[object] = None,
    ) -> Tensor:
        """Accumulate the sublayer output into this point's block slot.

        AttnRes has no residual-mixing / expansion matrices (``h_res`` / ``h_post`` are
        ``None``): the merge is simply ``streams[write_idx] += dropout(output + bias)``,
        with all other (frozen or empty) slots unchanged. This is the packed-tensor form of
        the reference ``partial = partial + sublayer_out``.

        Returns:
            [s, b, n*C] -- updated packed streams.
        """
        from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add

        x, bias = layer_output_with_bias
        s, b, _ = original_residual.shape

        # delta = dropout(x + bias), reusing Megatron's (optionally fused) bias-dropout-add
        # against a zero residual so dropout RNG semantics match the standard path.
        bda_func = get_bias_dropout_add(training, fused)
        zeros = torch.zeros_like(x)
        delta = bda_func((x, bias), zeros, dropout_prob)  # [s, b, C]

        streams = original_residual.view(s, b, self.n, self.hidden_size)
        onehot = self.slot_onehot.to(delta.dtype).view(1, 1, self.n, 1)
        updated = streams + delta.unsqueeze(2) * onehot  # only write_idx slot changes
        return updated.view(s, b, self.n * self.hidden_size)

    # ==================== Block-level utilities ====================

    @staticmethod
    def input_expand(x: Tensor, n: int) -> Tensor:
        """Expand 1-stream -> n-stream at TransformerBlock entry.

        Unlike mHC (which replicates the input to all streams), AttnRes seeds slot 0 with
        the embedding block ``b_0`` and leaves the N block slots empty (zero); they fill in
        as blocks complete.

        Args:
            x: [s, b, C] -- embedding hidden states.
            n: number of packed slots (num_blocks + 1).

        Returns:
            [s, b, n*C] -- slot 0 = x, slots 1..n-1 = 0.
        """
        s, b, c = x.shape
        zeros = x.new_zeros(s, b, (n - 1) * c)
        return torch.cat([x, zeros], dim=-1)


class AttnResOutputContract(MegatronModule):
    """Final depth-wise attention aggregation over all completed block slots (Block AttnRes).

    The (2L+1)-th aggregation point in the paper: contracts the packed n-stream state to a
    single stream just before the final layer norm. Mirrors :class:`AttnResModule`'s
    aggregation over *all* ``n`` slots. Kept as a module (not bare parameters on
    TransformerBlock) so its ``w`` / ``norm_weight`` are emitted by ``sharded_state_dict``
    and matched by the optimizer's sharded checkpoint, exactly as mHC's
    :class:`HyperConnectionOutputContract`.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.norm_eps = _ATTNRES_NORM_EPS
        self.w = nn.Parameter(torch.zeros(self.hidden_size))
        self.norm_weight = nn.Parameter(torch.ones(self.hidden_size))
        if config.sequence_parallel:
            setattr(self.w, "sequence_parallel", True)
            setattr(self.norm_weight, "sequence_parallel", True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Contract [s, b, n*C] -> [s, b, C] via softmax attention over all slots."""
        s, b, _ = hidden_states.shape
        streams = hidden_states.view(s, b, self.n, self.hidden_size)
        return _attn_res_aggregate(streams, self.w, self.norm_weight, self.norm_eps)
