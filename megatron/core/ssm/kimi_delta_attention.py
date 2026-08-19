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
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor

from megatron.core.jit import jit_fuser
from megatron.core.process_groups_config import ProcessGroupCollection
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
    nvtx_range_pop,
    nvtx_range_push,
    tensor_a2a_cp2hp,
    tensor_a2a_hp2cp,
)
from megatron.core.utils import deprecate_inference_params
from megatron.core import tensor_parallel

try:
    from fla.ops.kda import chunk_kda

    HAVE_KDA = True
except ImportError:
    chunk_kda = None
    HAVE_KDA = False

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
        RMSNorm from the reference KDA architecture.
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
         With config.kda_disable_output_norm the inner RMSNorm is
         dropped entirely (out = core_out * sigmoid(gate)); the fused kernel is
         bypassed and the inherited out_norm is replaced by IdentityOp so its
         weight is never registered. Intended for use with an external norm such
         as sandwich-norm, where the inner normalization is redundant.

    TP/CP forward execution is supported. KDA distributed-checkpoint splitting
    remains a separate TODO because the inherited GDN checkpoint factory assumes
    the GDN projection layout.
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
        self.gate_low_rank_dim = self.value_head_dim
        assert self.gate_low_rank_dim % self.tp_size == 0, (
            "KDA gate low-rank dimension must be divisible by tensor parallel size"
        )
        self.alpha_dim = self.num_value_heads * self.key_head_dim
        self.in_proj_dim = (
            self.qk_dim * 2
            + self.v_dim
            + 2 * self.gate_low_rank_dim
            + self.num_value_heads
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
            self.qk_dim,               # Q
            self.qk_dim,               # K
            self.v_dim,                # V
            self.gate_low_rank_dim,     # f_a (decay bottleneck)
            self.gate_low_rank_dim,     # g_a (output-gate bottleneck)
            self.num_value_heads,       # beta
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
        # Without the in-kernel gate, prefer FLA's fused_kda_gate over torch.
        self._kda_gate_style = (
            None
            if self._use_fused_decay_gate
            else (_KDA_GATE_STYLE if _env_flag("KDA_FUSED_GATE", True) else None)
        )
        if self._kda_gate_style is None and not self._use_fused_decay_gate:
            logger.warning(
                "Neither chunk_kda's use_gate_in_kernel nor fused_kda_gate is "
                "usable in the installed flash-linear-attention; computing the "
                "decay gate in torch instead (correct, but slower)."
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

        # Drop the inner (per-head) RMSNorm from the output path, keeping only the
        # sigmoid output gate — for stacks that normalize this path externally
        # (e.g. sandwich-norm's post-sublayer norm). See _apply_gated_norm.
        self._disable_output_norm = bool(
            getattr(self.config, "kda_disable_output_norm", False)
        )

        # Match the reference FLA KDA output path. The norm is per value head,
        # hence its feature dimension is value_head_dim rather than the full
        # (TP-global) v_dim. FLA's fused kernel also applies sigmoid(gate).
        # The fused kernel always normalizes, so it cannot serve the
        # norm-disabled path — force it off there and fall through to the
        # unfused gate-only branch.
        self._use_fused_output_norm_gate = (
            HAVE_FUSED_RMSNORM_GATED
            and not self._disable_output_norm
            and self.config.normalization == "RMSNorm"
            and not self.config.layernorm_zero_centered_gamma
            and self.config.linear_attention_use_output_gate
            and self.config.linear_attention_output_gate_form == "per_channel"
        )
        if self._disable_output_norm:
            # Replace the RMSNorm the parent GDN built from the submodule spec
            # with a no-op, so its weight is not registered as an unused
            # parameter (would trip DDP / land in the optimizer and checkpoint).
            self.out_norm = IdentityOp()
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
        """g = -exp(A_log) * softplus(alpha + dt_bias), fp32, [b, s, h, d_k].

        `alpha` arrives flat ([b, s, h*d_k]), as it leaves decay_out_proj. The
        reshape is per-branch: 0.4's fused_kda_gate splits per head itself.
        """
        if self._kda_gate_style == "0.4":
            g = fused_kda_gate(
                alpha, A_log_local_cp, self.key_head_dim, g_bias=dt_bias_local_cp
            )
            return g if self._decay_dtype is None else g.to(self._decay_dtype)
        alpha = alpha.reshape(*alpha.shape[:-1], -1, self.key_head_dim)
        if self._kda_gate_style == "0.5":
            # 0.5 can emit the decay in a narrower dtype directly; 0.4 cannot.
            if self._decay_dtype is None:
                return fused_kda_gate(alpha, A_log_local_cp, dt_bias_local_cp)
            return fused_kda_gate(
                alpha, A_log_local_cp, dt_bias_local_cp,
                output_dtype=self._decay_dtype,
            )
        g = self._activate_decay_torch(alpha, A_log_local_cp, dt_bias_local_cp)
        return g if self._decay_dtype is None else g.to(self._decay_dtype)

    @jit_fuser
    def _activate_decay_torch(self, alpha, A_log_local_cp, dt_bias_local_cp):
        """Torch fallback for `_activate_decay`; `alpha` already [b, s, h, d_k]."""
        decay_scale = A_log_local_cp.exp().view(1, 1, -1, 1)
        bias = dt_bias_local_cp.view(1, 1, -1, self.key_head_dim)
        return -decay_scale * F.softplus(alpha.float() + bias)

    def _in_proj_to_attn_inputs(
        self,
        hidden_states: torch.Tensor,
        batch: int,
        seq_len: int,
    ):
        # Input projection
        nvtx_range_push(suffix="in_proj")
        projected, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        num_v_heads_tp = self.num_value_heads // self.tp_size
        low_rank_local_tp = self.gate_low_rank_dim // self.tp_size
        qkv_channels_split_sections = [
            self.qk_dim_local_tp,
            self.qk_dim_local_tp,
            self.v_dim_local_tp,
        ]

        # Q/K/V occupy one contiguous prefix and stay grouped for convolution.
        nvtx_range_push(suffix="low_rank_proj")
        if self._fuse_low_rank_gather and self.tp_size > 1:
            qkv, low_rank_pair, beta = torch.split(
                projected,
                [self.conv_dim_local_tp, 2 * low_rank_local_tp, num_v_heads_tp],
                dim=-1,
            )
            # Both bottlenecks are needed at full width, so gather them in one
            # collective. The all-gather concatenates per-rank blocks, each
            # [decay_shard | gate_shard], giving [d0 g0 d1 g1 ...]; the view
            # below de-interleaves that back into the two full-width tensors.
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
        gate, _ = self.gate_out_proj(gate_low_rank)
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
        )
        gate = tensor_a2a_cp2hp(
            gate,
            seq_dim=0,
            head_dim=-1,
            cp_group=self.pg_collection.cp,
        )
        beta = tensor_a2a_cp2hp(
            beta,
            seq_dim=0,
            head_dim=-1,
            cp_group=self.pg_collection.cp,
        )
        alpha = tensor_a2a_cp2hp(
            alpha,
            seq_dim=0,
            head_dim=-1,
            cp_group=self.pg_collection.cp,
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
            qkv, _ = causal_conv1d(
                x=qkv, weight=conv1d_weight.squeeze(1), bias=conv1d_bias,
                activation=self.activation, initial_state=None,
                output_final_state=False, backend=self._conv1d_backend,
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
        """KDA forward. Mirrors GDN.forward, but the alpha slot is wider
        (num_v_heads * key_head_dim vs num_v_heads). KDA has no DeltaProduct
        variant, so n_hh is forced to 1 and erase logic is skipped.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        seq_len, batch, _ = hidden_states.shape
        seq_len = seq_len * self.sp_size * self.cp_size

        if inference_context is not None:
            raise NotImplementedError("KDA does not support inference for now.")
        if packed_seq_params is not None:
            raise NotImplementedError("KDA does not support packed sequence for now.")
        assert self.n_hh == 1, "KDA does not have a DeltaProduct (n_householder>1) variant."

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
                partial(self._in_proj_to_attn_inputs, batch=batch, seq_len=seq_len),
                hidden_states,
            )
        else:
            query, key, value, gate, beta, alpha = \
                self._in_proj_to_attn_inputs(hidden_states, batch, seq_len)

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
                     initial_state_f32, need_final_state)
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
        norm_out = tensor_a2a_hp2cp(norm_out, seq_dim=0, head_dim=-1, cp_group=self.pg_collection.cp)
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")
        return out, out_bias

    def _kda_core(self, query, key, value, gate, beta, alpha,
                  A_log_local_cp, dt_bias_local_cp,
                  initial_state_f32, need_final_state):
        """chunk_kda -> gated norm: the region that frees what the kernel saves.

        Kept self-contained so it can be handed to `torch.utils.checkpoint`
        (see `self._recompute_core`). Deliberately excludes every TE linear, so
        the recomputed region carries no FP8 GEMM and no `backward_dw`
        bookkeeping.
        """
        nvtx_range_push(suffix="g_and_beta")
        g = (
            alpha.reshape(*alpha.shape[:-1], -1, self.key_head_dim)
            if self._use_fused_decay_gate
            else self._activate_decay(alpha, A_log_local_cp, dt_bias_local_cp)
        )
        if not self._use_fused_beta_sigmoid:
            beta = self._activate_beta(beta)
        beta = beta.contiguous()
        nvtx_range_pop(suffix="g_and_beta")

        nvtx_range_push(suffix="chunk_kda")
        kda_kwargs = {
            "g": g,
            "beta": beta,
            "initial_state": initial_state_f32,
            "output_final_state": need_final_state,
            "use_qk_l2norm_in_kernel": self._qk_l2norm_in_kernel,
        }
        if self._use_fused_decay_gate:
            kda_kwargs["use_gate_in_kernel"] = True
            kda_kwargs["A_log"] = A_log_local_cp
            kda_kwargs["dt_bias"] = dt_bias_local_cp.reshape(-1)
        if self._use_fused_beta_sigmoid:
            kda_kwargs["use_beta_sigmoid_in_kernel"] = True
            if _KDA_SUPPORTS_FUSED_ALLOW_NEG_EIGVAL:
                kda_kwargs["allow_neg_eigval"] = (
                    self.config.linear_attention_allow_neg_eigval
                )
        core_attn_out, last_recurrent_state = self.gated_delta_rule(
            query, key, value, **kda_kwargs
        )
        nvtx_range_pop(suffix="chunk_kda")

        nvtx_range_push(suffix="gated_norm")
        norm_out = self._apply_gated_norm(core_attn_out, gate.contiguous())
        nvtx_range_pop(suffix="gated_norm")
        return norm_out, last_recurrent_state

    def backward_dw(self):
        """Execute deferred weight-gradient computation for all KDA linear projections."""
        self.in_proj.backward_dw()
        self.decay_out_proj.backward_dw()
        self.gate_out_proj.backward_dw()
        self.out_proj.backward_dw()

    def _apply_gated_norm(self, x, gate):
        """Apply per-head RMSNorm and the reference KDA sigmoid output gate."""
        if self._disable_output_norm:
            return self._apply_output_gate_only(x, gate)
        if self._use_fused_output_norm_gate:
            return self.out_norm(x, gate)
        return self._apply_gated_norm_unfused(x, gate)

    @jit_fuser
    def _apply_output_gate_only(self, x, gate):
        """Output gate WITHOUT the inner RMSNorm (kda_disable_output_norm=True).

        Reduces the reference gated RMSNorm to just the sigmoid output gate:
        out = x * sigmoid(gate), on the assumption that an outer norm (e.g.
        sandwich-norm's post-sublayer norm) already normalizes this path. Mirrors
        the gate handling of _apply_gated_norm_unfused, minus self.out_norm.
        """
        x_dtype = x.dtype
        y = x.reshape(-1, x.shape[-1]).float()
        if self.config.linear_attention_output_gate_form == "scalar":
            gate = gate.mean(dim=-1, keepdim=True)
        gate = gate.reshape(-1, gate.shape[-1])
        y = y * torch.sigmoid(gate.float())
        return y.to(x_dtype)

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
