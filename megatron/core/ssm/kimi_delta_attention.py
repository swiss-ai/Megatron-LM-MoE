# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.
#
# Kimi Delta Attention (KDA) layer for Megatron-LM.
#
# Why we don't drop in MoonshotAI's Kimi-Linear release directly
# --------------------------------------------------------------
# The official Kimi-Linear repo (github.com/MoonshotAI/Kimi-Linear) ships a
# clean reference implementation, but it targets the HuggingFace Transformers
# stack: layers extend nn.Module / PreTrainedModel, projections are plain
# nn.Linear (no Megatron ColumnParallel/RowParallel), tensors flow in
# [b, s, h] format, there is no context-parallel all-to-all, and there is no
# sharded-state-dict logic for distributed checkpointing. Megatron requires
# all of those: pg_collection, TP-aware projections, CP all-to-all, sbhd
# layout, and ShardedTensor-based checkpoints. Plumbing the upstream module
# into Megatron would mean rewriting it anyway, so we wrap the FLA `chunk_kda`
# kernel directly here, mirroring the structure of `gated_delta_net.py`.
#
# This keeps the kernel (including the optional FlashKDA path in FLA 0.5.0+)
# exactly as released by Moonshot, while letting the rest of the layer reuse
# Megatron's distributed conventions.

import copy
import inspect
import logging
import math
import os
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.contexts.attention_context.triton.tensor_ops import (
    tensor_get_slice_after,
    tensor_masked_update,
    tensor_merge,
)
from megatron.core.jit import jit_fuser
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.ops.causal_conv1d_triton import causal_conv1d_update
from megatron.core.ssm.ops.causal_conv1d_varlen import causal_conv1d_varlen_fn
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    get_cuda_rng_tracker,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.ssm.gated_delta_net import (
    GatedDeltaNet,
    GatedDeltaNetSubmodules,
    causal_conv1d,
    get_parameter_local_cp,
    conv1d_input_for_backend,
    nvtx_range_pop,
    nvtx_range_push,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
)
from megatron.core.utils import deprecate_inference_params, is_using_quantization_scales
from megatron.core import tensor_parallel

try:
    from fla.ops.kda import chunk_kda

    HAVE_KDA = True
except ImportError:
    chunk_kda = None
    HAVE_KDA = False

try:
    from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd
except ImportError:
    fused_recurrent_kda_fwd = None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from fla.modules import FusedRMSNormGated

    HAVE_FUSED_RMSNORM_GATED = True
except ImportError:
    FusedRMSNormGated = None
    HAVE_FUSED_RMSNORM_GATED = False


try:
    # Present since fla 0.4.0, but its signature is not stable across versions;
    # _fused_kda_gate_style() resolves the calling convention below.
    from fla.ops.kda.gate import fused_kda_gate

    HAVE_FUSED_KDA_GATE = True
except ImportError:
    fused_kda_gate = None
    HAVE_FUSED_KDA_GATE = False

def _have_causal_conv1d_cuda() -> bool:
    """Whether `causal_conv1d(..., backend="cuda")` will actually dispatch.

    Probe the causal_conv1d package, not FLA's re-export of it: the re-export
    moved between FLA versions, and probing the wrong spelling silently reports
    "unavailable" and leaves the layer on Triton.
    """
    try:
        from causal_conv1d import causal_conv1d_fn  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        from fla.modules.convolution import causal_conv1d_fn

        return causal_conv1d_fn is not None
    except ImportError:
        return False


HAVE_CAUSAL_CONV1D_CUDA = _have_causal_conv1d_cuda()


def _env_flag(name: str, default: bool) -> bool:
    """Read a KDA_* on/off override from the environment."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _chunk_kda_supports(option: str) -> bool:
    """Return whether the installed FLA explicitly exposes an optional KDA feature."""
    if not HAVE_KDA:
        return False
    try:
        return option in inspect.signature(chunk_kda).parameters
    except (TypeError, ValueError):
        return False


def _chunk_kda_accepts(option: str) -> bool:
    """Return whether chunk_kda can receive an argument, named or through **kwargs."""
    if not HAVE_KDA:
        return False
    try:
        params = inspect.signature(chunk_kda).parameters
    except (TypeError, ValueError):
        return False
    return option in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _fused_kda_gate_style() -> Optional[str]:
    """Which `fused_kda_gate` calling convention the installed FLA exposes.

    "0.4" -> fused_kda_gate(g[..., h*d_k], A_log, head_k_dim, g_bias=dt_bias)
    "0.5" -> fused_kda_gate(g[..., h, d_k], A_log, dt_bias=dt_bias)
    None  -> unavailable/unrecognized; fall back to the torch implementation.
    """
    if not HAVE_FUSED_KDA_GATE:
        return None
    try:
        params = inspect.signature(fused_kda_gate).parameters
    except (TypeError, ValueError):
        return None
    if "g_bias" in params:
        return "0.4"
    if "dt_bias" in params:
        return "0.5"
    return None


_KDA_GATE_STYLE = _fused_kda_gate_style()

# NOTE: This is for backwards compatibility, because many of these things were added in the newer version of FLA (like 0.52.0). But we should remove this and just say we use >=0.5.2
_KDA_SUPPORTS_QK_L2NORM_IN_KERNEL = _chunk_kda_supports("use_qk_l2norm_in_kernel")
_KDA_SUPPORTS_FUSED_BETA_SIGMOID = _chunk_kda_supports(
    "use_beta_sigmoid_in_kernel"
)
_KDA_SUPPORTS_FUSED_ALLOW_NEG_EIGVAL = _chunk_kda_supports("allow_neg_eigval")

# Needs both an implementation (only builds with a fused gate name the flag) and
# a way in for A_log, which FLA 0.5.x takes through **kwargs -- so A_log is
# probed with _chunk_kda_accepts, not _chunk_kda_supports.
_KDA_SUPPORTS_FUSED_DECAY_GATE = _chunk_kda_supports(
    "use_gate_in_kernel"
) and _chunk_kda_accepts("A_log")

# Kimi-K3 'safe' decay gate g = g_min * sigmoid(exp(A_log) * (z + dt_bias)):
# supported natively by chunk_kda (safe_gate + lower_bound) and by fused_kda_gate
# (lower_bound), both only in recent FLA. Probe each so older builds fall back to
# the torch reparameterization in _activate_decay_torch.
_KDA_SUPPORTS_SAFE_GATE_IN_KERNEL = _chunk_kda_supports("lower_bound") and _chunk_kda_supports(
    "safe_gate"
)


def _fused_kda_gate_supports_lower_bound() -> bool:
    """Whether the installed fused_kda_gate exposes the lower_bound (safe-gate) arg."""
    if not HAVE_FUSED_KDA_GATE:
        return False
    try:
        return "lower_bound" in inspect.signature(fused_kda_gate).parameters
    except (TypeError, ValueError):
        return False


_KDA_FUSED_GATE_SUPPORTS_LOWER_BOUND = _fused_kda_gate_supports_lower_bound()

logger = logging.getLogger(__name__)


@dataclass
class KimiDeltaAttentionSubmodules(GatedDeltaNetSubmodules):
    """KDA projections in addition to the shared GDN input/output modules."""

    decay_out_proj: Union[ModuleSpec, type] = IdentityOp
    gate_out_proj: Union[ModuleSpec, type] = IdentityOp


class KimiDeltaAttention(GatedDeltaNet):
    """Kimi Delta Attention (KDA) — channel-wise diagonal decay variant of GDN.

    Diff vs GatedDeltaNet:
      - decay g is a vector in R^{key_head_dim} per head (vs scalar per head),
        produced by the reference low-rank projection hidden -> value_head_dim ->
        num_value_heads*key_head_dim.
      - the output gate uses the matching low-rank projection and sigmoid gated
        RMSNorm from the reference KDA architecture. With
        linear_attention_full_rank_output_gate (Kimi-K3) the output gate is instead
        a full-rank hidden -> v_dim projection fused into in_proj and sharded on
        value heads, which drops the output gate's TP/CP all-gather.
      - the chunkwise op is `chunk_kda` (FLA 0.4.0+).
      - A_log is one fp32 scalar per value head, while dt_bias is fp32 and
        channel-wise within each value head.

    Reference: Kimi Team, "Kimi Linear: An Expressive, Efficient Attention
    Architecture", arXiv:2510.26692; FLA op `fla.ops.kda.chunk_kda`.

    The class subclasses `GatedDeltaNet` and replaces four things:
      1. `__init__` builds the reference factorized decay/output-gate projections
         and resizes `self.A_log` / `self.dt_bias`.
      2. `forward` delegates the vector decay activation to FLA's fused KDA gate
         and uses the fused beta sigmoid when the installed kernel exposes it;
         on older chunk_kda builds without the fused decay gate (e.g. flash-
         linear-attention 0.4.0 — no use_gate_in_kernel/A_log/dt_bias params),
         it instead computes g = -exp(A_log) * softplus(alpha + dt_bias) itself
         in Python (see _activate_decay) so A_log/dt_bias stay trainable.
      3. `forward` overrides only the splitting + reshape of `qkvzba` (alpha
         is wider) and the FLA-op call (chunk_kda instead of
         chunk_gated_delta_rule). All other plumbing (conv1d, CP/TP all-to-all,
         output projection, gated norm) is inherited unchanged.
      4. The normal per-channel output-gate path uses FLA's fused sigmoid-gated
         RMSNorm; the optional scalar/disabled gate modes retain an unfused path.

    TP/CP forward execution is supported, including cross-document masking through
    THD `packed_seq_params`. Distributed checkpointing overrides
    `_in_proj_sharded_split` so the fused in_proj is split on KDA's own layout
    rather than GDN's.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: KimiDeltaAttentionSubmodules,
        layer_number: int = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: Optional[float] = None,
        use_qk_l2norm: bool = True,
        A_init_range: Tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
        pp_layer_offset: Optional[int] = None,
        cp_comm_type: Optional[str] = None,
    ):
        if not HAVE_KDA:
            raise ImportError(
                "FLA's chunk_kda op is required for Kimi Delta Attention. "
                "Install a flash-linear-attention build that provides fla.ops.kda.chunk_kda."
            )
        if not _KDA_SUPPORTS_FUSED_DECAY_GATE:
            logger.warning(
                "The installed flash-linear-attention's chunk_kda does not expose "
                "'use_gate_in_kernel'/'A_log' (only accepted as inert **kwargs on older "
                "builds, e.g. flash-linear-attention 0.4.0 — see this repo's own "
                "pyproject.toml pin). Falling back to computing the decay activation "
                "g = -exp(A_log) * softplus(alpha + dt_bias) in Python before calling "
                "chunk_kda, instead of relying on the kernel to do it (loses the fused-"
                "kernel speed benefit, but keeps A_log/dt_bias trainable and correct). "
                "Upgrade flash-linear-attention to a build whose chunk_kda signature "
                "includes 'use_gate_in_kernel' and 'A_log' to use the fused path instead."
            )
        if not config.linear_attention_use_output_gate:
            raise ValueError(
                "Kimi Delta Attention requires linear_attention_use_output_gate=True."
            )

        # pp_layer_offset/cp_comm_type are unused (see GatedDeltaNet.__init__ docstring);
        # accepted only so TransformerLayer's generic self_attention construction can pass them.

        # Initialize via the GatedDeltaNet path. This builds in_proj/A_log/
        # dt_bias with the GDN scalar-alpha layout. We resize them below.
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            bias=bias,
            conv_bias=conv_bias,
            conv_init=conv_init,
            use_qk_l2norm=use_qk_l2norm,
            A_init_range=A_init_range,
            pg_collection=pg_collection,
        )

        # KDA's short depthwise conv1d requires a silu/swish activation (FLA's causal_conv1d
        # kernel and the Kimi-Linear reference). Decouple it from config.activation_func, which
        # may be set to a non-silu MoE/GLU activation (e.g. sslu/xpr) for the MLP path and would
        # otherwise trip `assert self.activation in ["silu", "swish"]` below. In KDA, act_fn /
        # activation are used ONLY for this conv (the output gate uses sigmoid), so forcing them
        # to silu here is safe and leaves the MoE activation untouched.
        self.act_fn = F.silu
        self.activation = "silu"

        # The reference KDA uses low-rank projections for both the decay input
        # and output gate:
        #   f_b(f_a(x)): hidden -> value_head_dim -> num_v_heads * key_head_dim
        #   g_b(g_a(x)): hidden -> value_head_dim -> value_dim
        # Fuse all projections that consume hidden_states directly into one
        # column-parallel GEMM. The second-stage projections remain separate.
        #
        # With linear_attention_full_rank_output_gate (Kimi-K3), the output gate
        # instead is a single full-rank projection hidden -> v_dim, fused straight
        # into in_proj and sharded on value heads like V, so it needs no second
        # stage and no TP/CP all-gather. Only the decay keeps its low-rank
        # bottleneck.
        self._full_rank_output_gate = self.config.linear_attention_full_rank_output_gate
        self.gate_low_rank_dim = self.value_head_dim
        assert self.gate_low_rank_dim % self.tp_size == 0, (
            "KDA gate low-rank dimension must be divisible by tensor parallel size"
        )
        self.alpha_dim = self.num_value_heads * self.key_head_dim
        # in_proj tail after Q/K/V: decay bottleneck (f_a) + output gate + beta.
        # The output gate slot is either the low-rank bottleneck (g_a,
        # gate_low_rank_dim) or the full-rank gate (v_dim).
        output_gate_in_proj_dim = (
            self.v_dim if self._full_rank_output_gate else self.gate_low_rank_dim
        )
        self.in_proj_dim = (
            self.qk_dim * 2
            + self.v_dim
            + self.gate_low_rank_dim
            + output_gate_in_proj_dim
            + self.num_value_heads
        )
        if self.config.fp8:
            # KDA's fused input projection includes per-head beta terms, so its output
            # width is not guaranteed to satisfy the alignment required by FP8 GEMMs (so for now, we disable FP8).
            warnings.warn(
                "KDA does not currently support FP8; running the KDA layer without FP8.",
                stacklevel=2,
            )

        # Rebuild in_proj with the new output dim. Uses the same submodule spec
        # as GDN; the parent's instance is replaced.
        self.in_proj = build_module(
            submodules.in_proj,
            self.hidden_size,
            self.in_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )
        # Keep the first-stage projection fused for the forward GEMM, but expose
        # the six reference matrices so Muon/MuonMD orthogonalize them separately.
        self.in_proj.weight.is_kda_in_proj = True
        self.in_proj.weight.kda_split_shapes = (
            self.qk_dim,                    # Q
            self.qk_dim,                    # K
            self.v_dim,                     # V
            self.gate_low_rank_dim,         # f_a (decay bottleneck)
            output_gate_in_proj_dim,        # g_a bottleneck, or full-rank output gate
            self.num_value_heads,           # beta
        )

        # decay_low_rank/gate_low_rank are already full-sequence by this point
        # (gathered by in_proj) and full-width (gathered explicitly in
        # forward()), so these two must NOT also treat sequence_parallel as on,
        # or TE re-gathers an already-complete sequence -- asserts under FP8
        # blockwise, silently wrong shape otherwise.
        second_stage_config = copy.copy(self.config)
        second_stage_config.sequence_parallel = False

        self.decay_out_proj = build_module(
            submodules.decay_out_proj,
            self.gate_low_rank_dim,
            self.alpha_dim,
            config=second_stage_config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kda_decay_out",
            tp_group=self.pg_collection.tp,
        )
        # Full-rank output gate has no second stage: the gate leaves in_proj
        # already at v_dim, sharded on value heads. Only the low-rank variant
        # needs g_b (gate_low_rank_dim -> v_dim).
        if self._full_rank_output_gate:
            self.gate_out_proj = None
        else:
            self.gate_out_proj = build_module(
                submodules.gate_out_proj,
                self.gate_low_rank_dim,
                self.v_dim,
                config=second_stage_config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=True,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="kda_gate_out",
                tp_group=self.pg_collection.tp,
            )

        # Reference shapes: A_log is scalar per value head; dt_bias is
        # channel-wise. Store dt_bias flattened, as FLA does, so both parameters
        # are one-dimensional fp32 Adam parameters without weight decay.
        self.num_v_heads_local_tp = self.num_value_heads // self.tp_size
        self.dt_bias = nn.Parameter(
            torch.empty(
                self.num_v_heads_local_tp * self.key_head_dim,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.dt_bias, "tensor_model_parallel", True)
        setattr(self.dt_bias, "partition_dim", 0)
        self.dt_bias.is_kda_decay_parameter = True
        self.A_log = nn.Parameter(
            torch.empty(
                self.num_v_heads_local_tp,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
        )
        setattr(self.A_log, "tensor_model_parallel", True)
        setattr(self.A_log, "partition_dim", 0)
        self.A_log.is_kda_decay_parameter = True

        # Bind the FLA op.
        self.gated_delta_rule = chunk_kda
        # Let chunk_kda derive the decay from raw alpha/A_log/dt_bias, so the
        # fp32 [b, s, h, d_k] decay is never materialized.
        self._use_fused_decay_gate = _KDA_SUPPORTS_FUSED_DECAY_GATE and _env_flag(
            "KDA_USE_GATE_IN_KERNEL", True
        )
        # Kimi-K3 safe decay gate g = g_min * sigmoid(exp(A_log) * (z + dt_bias)).
        # FLA computes this natively (chunk_kda safe_gate/lower_bound; fused_kda_gate
        # lower_bound); we only route to the torch reparameterization when the
        # installed FLA cannot. No clamp and no forced un-fusing.
        self._safe_gate = self.config.linear_attention_safe_output_gate
        self._gate_lower_bound = self.config.linear_attention_safe_output_gate_lower_bound
        if self._safe_gate and self._use_fused_decay_gate and not _KDA_SUPPORTS_SAFE_GATE_IN_KERNEL:
            logger.warning(
                "linear_attention_safe_output_gate is on but the installed "
                "flash-linear-attention's chunk_kda has no safe_gate/lower_bound; "
                "falling back to the fused_kda_gate/torch decay gate."
            )
            self._use_fused_decay_gate = False
        # Without the in-kernel gate, prefer FLA's fused_kda_gate over torch.
        self._kda_gate_style = (
            None
            if self._use_fused_decay_gate
            else (_KDA_GATE_STYLE if _env_flag("KDA_FUSED_GATE", True) else None)
        )
        # fused_kda_gate can only do the safe decay in the 0.5 style with a
        # lower_bound arg; otherwise fall through to the torch reparameterization.
        if (
            self._safe_gate
            and self._kda_gate_style is not None
            and not (self._kda_gate_style == "0.5" and _KDA_FUSED_GATE_SUPPORTS_LOWER_BOUND)
        ):
            self._kda_gate_style = None
        if self._kda_gate_style is None and not self._use_fused_decay_gate:
            logger.warning(
                "Neither chunk_kda's use_gate_in_kernel nor a lower_bound-capable "
                "fused_kda_gate is usable in the installed flash-linear-attention; "
                "computing the decay gate in torch instead (correct, but slower)."
            )

        # Dtype of the decay when one is materialized at all (None = fp32, the
        # kernel's native output dtype).
        _decay_dtype_name = os.environ.get("KDA_DECAY_DTYPE", "").strip().lower()
        self._decay_dtype = {
            "": None, "fp32": None, "float32": None,
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16,
        }.get(_decay_dtype_name)
        if _decay_dtype_name and self._decay_dtype is None and _decay_dtype_name not in (
            "fp32", "float32"
        ):
            raise ValueError(
                f"KDA_DECAY_DTYPE={_decay_dtype_name!r} is not one of "
                "fp32/bf16/fp16."
            )

        # Q/K L2-norm inside chunk_kda. Only the plain L2 path has an in-kernel
        # equivalent; the Schlag-style learnable qk_rmsnorm variant must stay
        # external.
        self._qk_l2norm_in_kernel = (
            _KDA_SUPPORTS_QK_L2NORM_IN_KERNEL
            and self.use_qk_l2norm
            and self.qk_rmsnorm is None
            and _env_flag("KDA_QK_L2NORM_IN_KERNEL", True)
        )

        # Prefer the causal_conv1d CUDA package over FLA's Triton conv: same op,
        # faster backend.
        self._conv1d_backend = (
            "cuda"
            if (HAVE_CAUSAL_CONV1D_CUDA and _env_flag("KDA_CONV1D_CUDA", True))
            else "triton"
        )

        # One all-gather for both low-rank bottlenecks instead of two.
        self._fuse_low_rank_gather = _env_flag("KDA_FUSED_LOW_RANK_GATHER", True)

        self._use_fused_beta_sigmoid = _KDA_SUPPORTS_FUSED_BETA_SIGMOID and (
            not self.config.linear_attention_allow_neg_eigval
            or _KDA_SUPPORTS_FUSED_ALLOW_NEG_EIGVAL
        )

        # Re-init A_log + dt_bias using the reference KDA time-scale distribution.
        self._reset_kda_decay_params(A_init_range)

        # Match the reference FLA KDA output path. The norm is per value head,
        # hence its feature dimension is value_head_dim rather than the full
        # (TP-global) v_dim. FLA's fused kernel also applies sigmoid(gate).
        self._use_fused_output_norm_gate = (
            HAVE_FUSED_RMSNORM_GATED
            and self.config.normalization == "RMSNorm"
            and not self.config.layernorm_zero_centered_gamma
            and self.config.linear_attention_use_output_gate
            and self.config.linear_attention_output_gate_form == "per_channel"
        )
        if self._use_fused_output_norm_gate:
            self.out_norm = FusedRMSNormGated(
                self.value_head_dim,
                activation="sigmoid",
                eps=self.config.layernorm_epsilon,
                device=torch.cuda.current_device(),
                dtype=self.config.params_dtype,
            )
            # This scale is replicated across TP ranks, just like Megatron's
            # regular output RMSNorm scale. Preserve its sequence-parallel
            # gradient-reduction marker after replacing the backend norm.
            setattr(
                self.out_norm.weight,
                "sequence_parallel",
                self.config.sequence_parallel,
            )

        # Checkpoint the chunk_kda core (`--recompute-modules linear_attn`).
        # Disjoint from `qkv` above: that one recomputes the PRODUCER of q/k/v
        # (in_proj -> conv1d -> prepare), this one the CONSUMER (chunk_kda ->
        # gated norm), so they compose rather than nesting over conv1d.
        self._recompute_core = (
            self.config.recompute_granularity == "selective"
            and self.config.recompute_modules is not None
            and "linear_attn" in self.config.recompute_modules
        )

        self.recompute_qkv = (
            self.config.recompute_granularity == 'selective'
            and "qkv" in self.config.recompute_modules
        )
        self.qkv_checkpoint = None


    def _in_proj_sharded_split(self):
        """KDA repacks in_proj, so the inherited GDN split does not apply.

        GDN fuses [Q, K, V, z, beta, alpha]; KDA fuses
        [Q, K, V, f_a, g_a, beta], where the two low-rank bottlenecks replace
        GDN's full-width z and per-head alpha. Using GDN's sections here asks
        _split_tensor_factory for 2*qk + 2*v + 2*num_v_heads rows out of a
        tensor that only has 2*qk + v + 2*gate_low_rank + num_v_heads, which
        fails the moment a distributed checkpoint is saved.
        """
        if self._full_rank_output_gate:
            # Output gate is full-rank (v_dim), sharded on value heads like V.
            return (
                [
                    self.qk_dim_local_tp,                     # Q
                    self.qk_dim_local_tp,                     # K
                    self.v_dim_local_tp,                      # V
                    self.gate_low_rank_dim // self.tp_size,   # f_a, decay bottleneck
                    self.v_dim_local_tp,                      # full-rank output gate
                    self.num_value_heads // self.tp_size,     # beta
                ],
                ["query", "key", "value", "decay_low_rank", "gate", "beta"],
            )
        return (
            [
                self.qk_dim_local_tp,                     # Q
                self.qk_dim_local_tp,                     # K
                self.v_dim_local_tp,                      # V
                self.gate_low_rank_dim // self.tp_size,   # f_a, decay bottleneck
                self.gate_low_rank_dim // self.tp_size,   # g_a, output-gate bottleneck
                self.num_value_heads // self.tp_size,     # beta
            ],
            ["query", "key", "value", "decay_low_rank", "gate_low_rank", "beta"],
        )

    def _reset_kda_decay_params(self, A_init_range: Tuple[float, float]) -> None:
        if not self.config.perform_initialization:
            return
        with get_cuda_rng_tracker().fork():
            # use log-uniform distribution for dt_bias to start with a healthy retention span, as in the reference Kimi-Linear code.
            dt = torch.exp(
                torch.rand_like(self.dt_bias.data)
                * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            ).clamp(min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias.data.copy_(inv_dt)
            if self.config.linear_attention_safe_output_gate:
                # Safe (lower-bound) gate g = g_min*sigmoid(exp(A_log)*(z+dt_bias)):
                # A_log is init to 0 (exp(A_log)=1), matching the Kimi-Linear reference.
                # The uniform(1,16) init below is for the softplus form; reusing it here
                # multiplies dt_bias (~[-6.9,-2.3]) by 1..16, saturating the sigmoid near
                # 0 so every head starts at ~no-decay -- killing timescale diversity.
                self.A_log.data.zero_()
            else:
                A = torch.empty(
                    self.num_v_heads_local_tp,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ).uniform_(*A_init_range)
                self.A_log.data.copy_(torch.log(A))

    @jit_fuser
    def _activate_beta(self, beta):
        """Fallback beta activation for FLA versions without fused sigmoid support."""
        beta = beta.float().sigmoid()
        if self.config.linear_attention_allow_neg_eigval:
            beta = beta * 2.0
        return beta

    def _activate_decay(self, alpha, A_log_local_cp, dt_bias_local_cp):
        """Materialize the decay gate g, fp32, [b, s, h, d_k].

        Default: g = -exp(A_log) * softplus(alpha + dt_bias). With
        linear_attention_safe_output_gate (Kimi-K3), the bounded reparameterization
        g = g_min * sigmoid(exp(A_log) * (alpha + dt_bias)) is used instead, passed to
        FLA via fused_kda_gate's `lower_bound` (or done in torch when unsupported).

        `alpha` arrives flat ([b, s, h*d_k]), as it leaves decay_out_proj. The
        reshape is per-branch: 0.4's fused_kda_gate splits per head itself.
        """
        # None => mode 1 (softplus); a value => mode 2 (bounded sigmoid). Only the
        # lower_bound-capable 0.5 fused gate reaches here with _safe_gate set;
        # __init__ routes other cases to the torch reparameterization below.
        lb = self._gate_lower_bound if self._safe_gate else None
        if self._kda_gate_style == "0.4":
            g = fused_kda_gate(
                alpha, A_log_local_cp, self.key_head_dim, g_bias=dt_bias_local_cp
            )
            return g if self._decay_dtype is None else g.to(self._decay_dtype)
        alpha = alpha.reshape(*alpha.shape[:-1], -1, self.key_head_dim)
        if self._kda_gate_style == "0.5":
            # 0.5 can emit the decay in a narrower dtype directly; 0.4 cannot.
            if self._decay_dtype is None:
                return fused_kda_gate(
                    alpha, A_log_local_cp, dt_bias_local_cp, lower_bound=lb
                )
            return fused_kda_gate(
                alpha, A_log_local_cp, dt_bias_local_cp,
                lower_bound=lb, output_dtype=self._decay_dtype,
            )
        g = self._activate_decay_torch(alpha, A_log_local_cp, dt_bias_local_cp)
        return g if self._decay_dtype is None else g.to(self._decay_dtype)

    @jit_fuser
    def _activate_decay_torch(self, alpha, A_log_local_cp, dt_bias_local_cp):
        """Torch fallback for `_activate_decay`; `alpha` already [b, s, h, d_k]."""
        decay_scale = A_log_local_cp.exp().view(1, 1, -1, 1)
        bias = dt_bias_local_cp.view(1, 1, -1, self.key_head_dim)
        if self._safe_gate:
            # Kimi-K3 safe decay: g = g_min * sigmoid(exp(A_log) * (alpha + dt_bias)).
            # Mirrors fla.ops.kda naive_kda_lowerbound_gate (bias added first, then
            # scaled by exp(A_log) inside the sigmoid).
            return self._gate_lower_bound * torch.sigmoid(decay_scale * (alpha.float() + bias))
        return -decay_scale * F.softplus(alpha.float() + bias)

    def _expand_low_rank_inputs(self, projected):
        """Split in_proj output and apply KDA's two second-stage projections."""
        num_v_heads_tp = self.num_value_heads // self.tp_size
        low_rank_local_tp = self.gate_low_rank_dim // self.tp_size
        if self._full_rank_output_gate:
            # Output gate is full-rank and already sharded on value heads exactly
            # like V, so it needs no all-gather. Only the decay bottleneck does.
            qkv, decay_low_rank, gate, beta = torch.split(
                projected,
                [
                    self.conv_dim_local_tp,
                    low_rank_local_tp,
                    self.v_dim_local_tp,
                    num_v_heads_tp,
                ],
                dim=-1,
            )
            if self.tp_size > 1:
                decay_low_rank = gather_from_tensor_model_parallel_region(
                    decay_low_rank, group=self.pg_collection.tp
                )
        elif self._fuse_low_rank_gather and self.tp_size > 1:
            qkv, low_rank_pair, beta = torch.split(
                projected,
                [self.conv_dim_local_tp, 2 * low_rank_local_tp, num_v_heads_tp],
                dim=-1,
            )
            gathered = gather_from_tensor_model_parallel_region(
                low_rank_pair, group=self.pg_collection.tp
            )
            gathered = gathered.view(
                *gathered.shape[:-1], self.tp_size, 2, low_rank_local_tp
            )
            decay_low_rank = gathered[..., 0, :].reshape(
                *gathered.shape[:-3], self.gate_low_rank_dim
            )
            gate_low_rank = gathered[..., 1, :].reshape(
                *gathered.shape[:-3], self.gate_low_rank_dim
            )
        else:
            qkv, decay_low_rank, gate_low_rank, beta = torch.split(
                projected,
                [
                    self.conv_dim_local_tp,
                    low_rank_local_tp,
                    low_rank_local_tp,
                    num_v_heads_tp,
                ],
                dim=-1,
            )
            if self.tp_size > 1:
                decay_low_rank = gather_from_tensor_model_parallel_region(
                    decay_low_rank, group=self.pg_collection.tp
                )
                gate_low_rank = gather_from_tensor_model_parallel_region(
                    gate_low_rank, group=self.pg_collection.tp
                )
        alpha, _ = self.decay_out_proj(decay_low_rank)
        if not self._full_rank_output_gate:
            gate, _ = self.gate_out_proj(gate_low_rank)
        return qkv, gate, beta, alpha

    def _prepare_g_and_beta(self, alpha, beta, A_log, dt_bias):
        """Prepare raw or activated decay and beta inputs for a KDA kernel."""
        g = (
            alpha.reshape(*alpha.shape[:-1], -1, self.key_head_dim)
            if self._use_fused_decay_gate
            else self._activate_decay(alpha, A_log, dt_bias)
        )
        if not self._use_fused_beta_sigmoid:
            beta = self._activate_beta(beta)
        return g, beta.contiguous()

    def _kda_kernel_options(self, A_log, dt_bias):
        """Feature options shared by chunked and recurrent KDA kernels."""
        kwargs = {
            "use_qk_l2norm_in_kernel": self._qk_l2norm_in_kernel,
            "use_gate_in_kernel": self._use_fused_decay_gate,
            "use_beta_sigmoid_in_kernel": self._use_fused_beta_sigmoid,
        }
        if self._use_fused_decay_gate:
            kwargs["A_log"] = A_log
            kwargs["dt_bias"] = dt_bias.reshape(-1)
            if self._safe_gate:
                kwargs["safe_gate"] = True
                kwargs["lower_bound"] = self._gate_lower_bound
        if self._use_fused_beta_sigmoid and _KDA_SUPPORTS_FUSED_ALLOW_NEG_EIGVAL:
            kwargs["allow_neg_eigval"] = self.config.linear_attention_allow_neg_eigval
        return kwargs

    def _in_proj_to_attn_inputs(
        self,
        hidden_states: torch.Tensor,
        batch: int,
        seq_len: int,
        packed_seq_params=None,
        cu_seqlens=None,
    ):
        # Input projection
        nvtx_range_push(suffix="in_proj")
        projected, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        qkv_channels_split_sections = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        ]

        # Q/K/V occupy one contiguous prefix and stay grouped for convolution.
        nvtx_range_push(suffix="low_rank_proj")
        qkv, gate, beta, alpha = self._expand_low_rank_inputs(projected)

        nvtx_range_pop(suffix="low_rank_proj")

        # Keep the logical outputs separate through CP. The CP helper already
        # communicates split sections independently, so packing qkv/gate/beta/
        # alpha into a temporary qkvzba tensor only adds a full-size copy.
        qkv = tensor_a2a_cp2hp(
            qkv,
            seq_dim=0,
            head_dim=-1,
            cp_group=self.pg_collection.cp,
            split_sections=qkv_channels_split_sections,
            packed_seq_params=packed_seq_params,
        )
        gate = tensor_a2a_cp2hp(
            gate,
            seq_dim=0,
            head_dim=-1,
            cp_group=self.pg_collection.cp,
            packed_seq_params=packed_seq_params,
        )
        beta = tensor_a2a_cp2hp(
            beta,
            seq_dim=0,
            head_dim=-1,
            cp_group=self.pg_collection.cp,
            packed_seq_params=packed_seq_params,
        )
        alpha = tensor_a2a_cp2hp(
            alpha,
            seq_dim=0,
            head_dim=-1,
            cp_group=self.pg_collection.cp,
            packed_seq_params=packed_seq_params,
        )

        # Transpose separately: s b x --> b s x.
        qkv = qkv.transpose(0, 1)
        gate = gate.transpose(0, 1)
        beta = beta.transpose(0, 1)
        alpha = alpha.transpose(0, 1)
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, -1)

        # Convolution on qkv (Q, K, V all per-token for KDA — no DeltaProduct).
        nvtx_range_push(suffix="conv1d")
        seq_len = qkv.shape[1]
        conv1d_weight = get_parameter_local_cp(
            self.conv1d.weight, dim=0, cp_group=self.pg_collection.cp,
            split_sections=qkv_channels_split_sections,
        )
        conv1d_bias = (
            get_parameter_local_cp(
                self.conv1d.bias, dim=0, cp_group=self.pg_collection.cp,
                split_sections=qkv_channels_split_sections,
            ) if self.conv_bias else None
        )
        if self.config.deterministic_mode:
            qkv = qkv.transpose(1, 2).contiguous()
            conv_out = F.conv1d(
                input=qkv, weight=conv1d_weight, bias=conv1d_bias,
                stride=self.conv1d.stride, padding=self.conv1d.padding,
                dilation=self.conv1d.dilation,
                groups=self.conv_dim_local_tp // self.cp_size,
            )
            qkv = self.act_fn(conv_out[..., :seq_len])
            qkv = qkv.transpose(1, 2)
        else:
            assert self.activation in ["silu", "swish"]
            qkv, backend = conv1d_input_for_backend(qkv, self._conv1d_backend)
            qkv, _ = causal_conv1d(
                x=qkv, weight=conv1d_weight.squeeze(1), bias=conv1d_bias,
                activation=self.activation, initial_state=None,
                output_final_state=False, backend=backend,
                cu_seqlens=cu_seqlens,
            )
        nvtx_range_pop(suffix="conv1d")

        # Q/K/V split + reshape + alpha reshape (overridden helper handles the wider alpha).
        nvtx_range_push(suffix="prepare_qkv_for_kda")
        query, key, value = self._prepare_qkv_for_kda(qkv, batch, seq_len)
        alpha = alpha.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        nvtx_range_pop(suffix="prepare_qkv_for_kda")
        return query, key, value, gate, beta, alpha

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        **kwargs,
    ):
        """Run KDA without FP8, even when FP8 is enabled for the model."""
        quantization_context = nullcontext()
        if self.config.fp8:
            from transformer_engine.pytorch import fp8_autocast

            quantization_context = fp8_autocast(enabled=False)
        with quantization_context:
            return self._forward_impl(
                hidden_states,
                attention_mask,
                inference_context,
                packed_seq_params,
                sequence_len_offset,
                inference_params=inference_params,
                **kwargs,
            )

    def _forward_impl(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        **kwargs,
    ):
        """KDA forward. Mirrors GDN.forward, but the alpha slot is wider
        (num_v_heads * key_head_dim vs num_v_heads). KDA has no DeltaProduct
        variant, so n_hh is forced to 1 and erase logic is skipped.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size * self.cp_size

        if inference_context is not None:
            if not self.training and inference_context.is_dynamic_batching():
                return self._dynamic_inference(hidden_states, inference_context)
            raise NotImplementedError("KDA inference is supported only by the dynamic engine.")
        assert self.n_hh == 1, "KDA does not have a DeltaProduct (n_householder>1) variant."

        # Cross-document masking. The conv and chunk_kda both run on the full
        # sequence (SP undone by in_proj's all-gather, CP by the cp2hp all-to-all),
        # so the dataloader's global cu_seqlens is what both kernels want.
        cu_seqlens = self._resolve_packed_cu_seqlens(packed_seq_params, seq_len, batch)
        if cu_seqlens is not None:
            if self.config.deterministic_mode:
                raise NotImplementedError(
                    "KDA packed sequence requires the FLA kernels; deterministic_mode's "
                    "F.conv1d fallback ignores cu_seqlens and leaks the convolution "
                    "window across document boundaries."
                )
            if self.initial_state_param is not None or self._carry_enabled:
                raise NotImplementedError(
                    "KDA packed sequence does not support a recurrent initial state: "
                    "cu_seqlens makes chunk_kda expect one state per document, while "
                    "the learnable/carried state is per batch element."
                )

        # 1. in_proj + conv1d + split
        # 2. decay_out_proj (low-rank -> vector decay)
        # 3. gate_out_proj (low-rank -> output gate)
        if self.recompute_qkv and self.training and torch.is_grad_enabled():
            self.qkv_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                fp8=self.config.fp8 or self.config.fp4
            )
            # Only tensors may be passed as checkpoint args (they go through
            # ctx.save_for_backward); bind the shape ints into the callable instead.
            query, key, value, gate, beta, alpha = self.qkv_checkpoint.checkpoint(
                partial(
                    self._in_proj_to_attn_inputs,
                    batch=batch,
                    seq_len=seq_len,
                    packed_seq_params=packed_seq_params,
                    cu_seqlens=cu_seqlens,
                ),
                hidden_states,
            )
        else:
            query, key, value, gate, beta, alpha = self._in_proj_to_attn_inputs(
                hidden_states, batch, seq_len, packed_seq_params, cu_seqlens
            )

        # FLA computes the vector decay from raw alpha, A_log, and dt_bias inside
        # chunk_kda. Newer FLA versions can also fuse beta.float().sigmoid().
        nvtx_range_push(suffix="g_and_beta")
        A_log_local_cp = get_parameter_local_cp(
            self.A_log, dim=0, cp_group=self.pg_collection.cp
        )
        # Stored flat but CP-sharded per head: view as [h, d_k] to slice, then
        # flatten straight back -- every consumer wants it flat, and 0.4's
        # autograd Function rejects a [h, d_k] grad.
        dt_bias_local_cp = get_parameter_local_cp(
            self.dt_bias.view(self.num_v_heads_local_tp, self.key_head_dim),
            dim=0,
            cp_group=self.pg_collection.cp,
        ).reshape(-1)
        nvtx_range_pop(suffix="g_and_beta")

        # Initial state: learnable param > carried full-batch state > None.
        if self.initial_state_param is not None:
            initial_state = self.initial_state_param.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
        elif self._carry_enabled:
            num_v_heads_local = self.num_value_heads // self.tp_size
            need_init = (
                self._carried_state is None
                or self._carried_state.shape[0] != batch
                or self._carried_state.shape[1] != num_v_heads_local
            )
            if need_init:
                self._carried_state = torch.zeros(
                    batch, num_v_heads_local, self.key_head_dim, self.value_head_dim,
                    dtype=hidden_states.dtype, device=hidden_states.device,
                )
            initial_state = self._carried_state.detach()
        else:
            initial_state = None

        log_state_stats = os.environ.get("APERTUS_LOG_STATE_STATS", "0") == "1"
        need_final_state = self._carry_enabled or log_state_stats

        # chunk_kda requires initial_state in float32 (asserted inside FLA).
        initial_state_f32 = initial_state.float() if initial_state is not None else None

        # chunk_kda -> gated norm. Optionally recomputed in the backward
        # (`--recompute-modules linear_attn`); skipped when a final recurrent
        # state must escape, since the bookkeeping below must not run twice.
        core_args = (query, key, value, gate, beta, alpha,
                     A_log_local_cp, dt_bias_local_cp,
                     initial_state_f32, need_final_state, cu_seqlens)
        if self._recompute_core and torch.is_grad_enabled() and not need_final_state:
            norm_out, last_recurrent_state = torch.utils.checkpoint.checkpoint(
                self._kda_core, *core_args, use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            norm_out, last_recurrent_state = self._kda_core(*core_args)

        # Carry-over (matches GDN; only active if linear_attention_carry_state=True).
        if self._carry_enabled and last_recurrent_state is not None:
            with torch.no_grad():
                new_state = last_recurrent_state.detach().to(self._carried_state.dtype)
                cap = self.config.linear_attention_carried_state_max_frob
                if cap > 0.0:
                    flat = new_state.reshape(new_state.shape[0], -1).float()
                    norms = flat.norm(dim=-1, keepdim=True)
                    scale = torch.clamp(cap / norms.clamp_min(1e-12), max=1.0)
                    flat = flat * scale
                    new_state = flat.reshape_as(new_state).to(new_state.dtype)
                self._carried_state = new_state

        if log_state_stats and last_recurrent_state is not None:
            with torch.no_grad():
                s = last_recurrent_state.detach().float()
                stats = {
                    "frob": s.norm(), "amax": s.abs().amax(),
                    "abs_mean": s.abs().mean(), "std": s.std(),
                }
                if initial_state is not None:
                    init_f = initial_state.detach().float().norm()
                    stats["init_frob"] = init_f
                    stats["delta_frob"] = stats["frob"] - init_f
                else:
                    stats["init_frob"] = torch.zeros((), device=s.device)
                    stats["delta_frob"] = stats["frob"].clone()
                self._last_state_stats = stats
        else:
            self._last_state_stats = None

        # checkpointing for recomputation
        if self.qkv_checkpoint is not None:
            self.qkv_checkpoint.discard_output_and_register_recompute(
                norm_out
            )
            self.qkv_checkpoint = None

        # Transpose back to sbhd, CP a2a HP→CP, output projection.
        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()
        norm_out = tensor_a2a_hp2cp(
            norm_out,
            seq_dim=0,
            head_dim=-1,
            cp_group=self.pg_collection.cp,
            packed_seq_params=packed_seq_params,
        )
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")
        return out, out_bias

    def _kda_core(self, query, key, value, gate, beta, alpha,
                  A_log_local_cp, dt_bias_local_cp,
                  initial_state_f32, need_final_state, cu_seqlens=None):
        """chunk_kda -> gated norm: the region that frees what the kernel saves.

        Kept self-contained so it can be handed to `torch.utils.checkpoint`
        (see `self._recompute_core`). Deliberately excludes every TE linear, so
        the recomputed region carries no FP8 GEMM and no `backward_dw`
        bookkeeping.
        """
        nvtx_range_push(suffix="g_and_beta")
        g, beta = self._prepare_g_and_beta(
            alpha, beta, A_log_local_cp, dt_bias_local_cp
        )
        nvtx_range_pop(suffix="g_and_beta")

        nvtx_range_push(suffix="chunk_kda")
        kda_kwargs = self._kda_kernel_options(A_log_local_cp, dt_bias_local_cp)
        kda_kwargs.update(
            g=g,
            beta=beta,
            initial_state=initial_state_f32,
            output_final_state=need_final_state,
        )
        if cu_seqlens is not None:
            kda_kwargs["cu_seqlens"] = cu_seqlens

        core_attn_out, last_recurrent_state = self.gated_delta_rule(
            query, key, value, **kda_kwargs
        )
        nvtx_range_pop(suffix="chunk_kda")

        nvtx_range_push(suffix="gated_norm")
        norm_out = self._apply_gated_norm(core_attn_out, gate.contiguous())
        nvtx_range_pop(suffix="gated_norm")
        return norm_out, last_recurrent_state

    def _split_dynamic_projection(self, projected: Tensor):
        """Expand and reshape the current factorized KDA projection for inference."""
        batch, seq_len, _ = projected.shape
        qkv, gate, beta, alpha = self._expand_low_rank_inputs(projected)
        gate = gate.reshape(batch, seq_len, self.num_v_heads_local_tp, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, self.num_v_heads_local_tp)
        return qkv, gate, beta, alpha

    def _prepare_dynamic_kda(self, qkv, gate, beta, alpha):
        """Prepare convolved projections and raw/fused gates for an inference kernel."""
        batch, seq_len, _ = qkv.shape
        query, key, value = self._prepare_qkv_for_kda(qkv, batch, seq_len)
        g, beta = self._prepare_g_and_beta(alpha, beta, self.A_log, self.dt_bias)
        return query, key, value, gate.contiguous(), g, beta

    def _dynamic_inference_decode(
        self,
        projected: Tensor,
        conv_state: Tensor,
        recurrent_state: Tensor,
        batch_indices: Tensor,
        dummy_state_idx: int,
    ) -> Tensor:
        """Decode one token per request and update indexed KDA states in place."""
        if fused_recurrent_kda_fwd is None:
            raise ImportError(
                "Dynamic KDA decode requires fla.ops.kda.fused_recurrent_kda_fwd."
            )

        qkv, gate, beta, alpha = self._split_dynamic_projection(projected)
        safe_batch_indices = torch.where(
            batch_indices >= 0,
            batch_indices,
            torch.full_like(batch_indices, dummy_state_idx),
        )

        qkv_dtype = qkv.dtype
        conv_weight = self.conv1d.weight.squeeze(1).to(conv_state.dtype)
        conv_bias = self.conv1d.bias if self.conv_bias else None
        if conv_bias is not None:
            conv_bias = conv_bias.to(conv_state.dtype)
        qkv = causal_conv1d_update(
            x=qkv.to(conv_state.dtype),
            conv_state=conv_state,
            weight=conv_weight,
            bias=conv_bias,
            silu_activation=self.activation,
            conv_state_indices=safe_batch_indices,
        ).to(qkv_dtype)

        query, key, value, gate, g, beta = self._prepare_dynamic_kda(
            qkv, gate, beta, alpha
        )
        core_attn_out, _ = fused_recurrent_kda_fwd(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=False,
            inplace_final_state=True,
            ssm_state_indices=safe_batch_indices,
            **self._kda_kernel_options(self.A_log, self.dt_bias),
        )
        return self._apply_gated_norm(core_attn_out, gate).reshape(
            projected.shape[0], projected.shape[1], -1
        )

    def _dynamic_inference_prefill(
        self,
        projected: Tensor,
        context: DynamicInferenceContext,
        conv_state: Tensor,
        recurrent_state: Tensor,
    ) -> Tensor:
        """Run prompt tokens as one packed variable-length KDA prefill."""
        metadata = context.kda_metadata
        token_count = projected.shape[0]
        batch_indices = metadata.batch_indices_prefill
        cu_seqlens = metadata.cu_seqlens

        qkv, gate, beta, alpha = self._split_dynamic_projection(projected.transpose(0, 1))

        initial_conv_states = conv_state[batch_indices]
        qkv_pre_conv = qkv.squeeze(0).contiguous()
        qkv_conv_dtype = qkv_pre_conv.to(conv_state.dtype)
        conv_weight = self.conv1d.weight.squeeze(1).to(conv_state.dtype).contiguous()
        conv_bias = self.conv1d.bias if self.conv_bias else None
        if conv_bias is None:
            conv_bias = qkv_conv_dtype.new_zeros(qkv_conv_dtype.shape[-1])
        else:
            conv_bias = conv_bias.to(conv_state.dtype)

        qkv = causal_conv1d_varlen_fn(
            x=qkv_conv_dtype,
            weight=conv_weight,
            bias=conv_bias,
            cu_seqlens=cu_seqlens,
            initial_states=initial_conv_states[:, :, 1:],
            activation=self.activation,
            precomputed_seq_idx=metadata.conv_seq_idx[:token_count],
            precomputed_seq_start=metadata.conv_seq_start[:token_count],
        ).to(projected.dtype)
        qkv = qkv.unsqueeze(0)

        if causal_conv1d_varlen_states is None:
            final_conv_states = self._extract_final_conv_states(
                qkv_conv_dtype, cu_seqlens, conv_state.shape[-1]
            )
        else:
            final_conv_states = causal_conv1d_varlen_states(
                qkv_conv_dtype, cu_seqlens, state_len=conv_state.shape[-1]
            )
        final_conv_states = self._merge_short_conv_states(
            initial_conv_states, final_conv_states, cu_seqlens
        )
        tensor_masked_update(conv_state, batch_indices, final_conv_states)

        query, key, value, gate, g, beta = self._prepare_dynamic_kda(
            qkv, gate, beta, alpha
        )
        initial_recurrent_state = recurrent_state[batch_indices]
        core_attn_out, final_recurrent_state = self.gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_recurrent_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            **self._kda_kernel_options(self.A_log, self.dt_bias),
        )
        tensor_masked_update(recurrent_state, batch_indices, final_recurrent_state)

        norm_out = self._apply_gated_norm(core_attn_out, gate)
        return norm_out.reshape(1, token_count, -1).transpose(0, 1).contiguous()

    @jit_fuser
    def _extract_final_conv_states(self, inputs, cu_seqlens, state_len):
        """Device-side fallback for packed final-state extraction."""
        starts = cu_seqlens[:-1].long()
        ends = cu_seqlens[1:].long()
        offsets = torch.arange(-state_len, 0, device=inputs.device)
        positions = ends.unsqueeze(1) + offsets.unsqueeze(0)
        from_inputs = positions >= starts.unsqueeze(1)
        input_states = inputs[positions.clamp_min(0)].permute(0, 2, 1)
        return torch.where(from_inputs.unsqueeze(1), input_states, 0.0)

    @jit_fuser
    def _merge_short_conv_states(self, initial_states, final_states, cu_seqlens):
        """Fill a short continuation's missing prefix from its prior state."""
        lengths = cu_seqlens[1:].long() - cu_seqlens[:-1].long()
        state_len = initial_states.shape[-1]
        offsets = torch.arange(state_len, device=initial_states.device)
        old_indices = (offsets.unsqueeze(0) + lengths.unsqueeze(1)).clamp(
            max=state_len - 1
        )
        old_states = torch.gather(
            initial_states,
            2,
            old_indices.unsqueeze(1).expand(-1, initial_states.shape[1], -1),
        )
        from_initial = offsets.unsqueeze(0) < (state_len - lengths).unsqueeze(1)
        return torch.where(from_initial.unsqueeze(1), old_states, final_states)

    def _dynamic_inference(
        self, hidden_states: Tensor, context: DynamicInferenceContext
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Execute mixed dynamic batches with decode rows followed by packed prefill rows."""
        conv_state, recurrent_state = context.kda_states_cache(self.layer_number)
        padded_dims = context.padded_batch_dimensions
        decode_req_count = padded_dims.decode_req_count
        prefill_req_count = padded_dims.prefill_req_count

        projected, _ = self.in_proj(hidden_states)
        y_decode = None
        y_prefill = None

        if decode_req_count > 0:
            y_decode = self._dynamic_inference_decode(
                projected[:decode_req_count],
                conv_state,
                recurrent_state,
                context.kda_metadata.batch_indices_decode,
                context.kda_dummy_state_idx,
            )

        if prefill_req_count > 0:
            if decode_req_count > 0:
                projected_prefill = torch.empty_like(projected)
                tensor_get_slice_after(
                    projected,
                    projected_prefill,
                    context.kda_metadata.device_decode_prefill,
                    check_bounds=False,
                )
            else:
                projected_prefill = projected
            y_prefill = self._dynamic_inference_prefill(
                projected_prefill, context, conv_state, recurrent_state
            )

        if y_decode is not None and y_prefill is not None:
            y = torch.empty(
                (padded_dims.token_count, 1, y_prefill.shape[-1]),
                dtype=y_prefill.dtype,
                device=y_prefill.device,
            )
            tensor_merge(
                y_decode,
                y_prefill,
                context.kda_metadata.device_decode_prefill,
                output_tensor=y,
            )
        elif y_decode is not None:
            y = y_decode
        elif y_prefill is not None:
            y = y_prefill
        else:
            raise RuntimeError("Dynamic KDA inference received an empty batch")

        if is_using_quantization_scales(self.config):
            y[context.padding_slice] = 0.0

        return self.out_proj(y)

    def kda_state_shapes_per_request(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Return convolution and recurrent state shapes for one inference request."""
        return (
            (self.conv_dim_local_tp, self.conv_kernel_dim),
            (self.num_v_heads_local_tp, self.key_head_dim, self.value_head_dim),
        )

    def backward_dw(self):
        """Execute deferred weight-gradient computation for all KDA linear projections."""
        self.in_proj.backward_dw()
        self.decay_out_proj.backward_dw()
        if self.gate_out_proj is not None:
            self.gate_out_proj.backward_dw()
        self.out_proj.backward_dw()

    def _apply_gated_norm(self, x, gate):
        """Apply per-head RMSNorm and the reference KDA sigmoid output gate."""
        if self._use_fused_output_norm_gate:
            return self.out_norm(x, gate)
        return self._apply_gated_norm_unfused(x, gate)

    @jit_fuser
    def _apply_gated_norm_unfused(self, x, gate):
        """Fallback for the optional scalar or disabled output-gate modes."""
        x_dtype = x.dtype
        y = self.out_norm(x.reshape(-1, x.shape[-1]))
        if self.config.linear_attention_use_output_gate:
            if self.config.linear_attention_output_gate_form == "scalar":
                gate = gate.mean(dim=-1, keepdim=True)
            gate = gate.reshape(-1, gate.shape[-1])
            y = y * torch.sigmoid(gate.float())
        return y.to(x_dtype)

    @jit_fuser
    def _prepare_qkv_for_kda(self, qkv, batch, seq_len):
        """Split the post-conv qkv block into Q, K, V in [b, s, h, d] layout.

        The Q/K L2-norm is normally deferred into chunk_kda
        (self._qk_l2norm_in_kernel); only the learnable qk_rmsnorm variant,
        which has no in-kernel equivalent, still normalizes here.
        """
        query_key, value = torch.split(
            qkv,
            [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
            dim=-1,
        )
        query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)

        if self.use_qk_l2norm and not self._qk_l2norm_in_kernel:
            query_key = query_key.contiguous()
            if self.qk_rmsnorm is not None:
                query_key = self.qk_rmsnorm(query_key)
            else:
                from fla.modules.l2norm import l2norm

                query_key = l2norm(query_key)

        split_size = self.qk_dim_local_tp // self.key_head_dim // self.cp_size
        query, key = torch.split(query_key, [split_size, split_size], dim=2)

        if self.num_value_heads // self.num_key_heads > 1:
            repeat_factor = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        return query.contiguous(), key.contiguous(), value.contiguous()


def get_kimi_delta_attention_module_spec(
    config: TransformerConfig, backend=None,
) -> ModuleSpec:
    """Build the KDA spec with separate second-stage gate projections."""
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        _get_backend_spec_provider,
    )

    if backend is None:
        backend = _get_backend_spec_provider(config=config)
    rms_norm = config.normalization == "RMSNorm"
    return ModuleSpec(
        module=KimiDeltaAttention,
        submodules=KimiDeltaAttentionSubmodules(
            in_proj=backend.column_parallel_layer_norm_linear(),
            decay_out_proj=backend.column_parallel_linear(),
            gate_out_proj=backend.column_parallel_linear(),
            out_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
            out_proj=backend.row_parallel_linear(),
        ),
        metainfo={"fuse_input_layernorm": True},
    )
