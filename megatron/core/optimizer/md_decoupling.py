# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MDDecoupling optimizer (magnitude-direction decoupling).

Decouples a 2D weight into a *direction* (hypersphere-normalized weight) and a *magnitude*
(learnable per-axis gains), with an optional Muon-style orthogonalized update on the direction:

  - Direction: post-step L2-sphere projection of the weight (modes: row, col, flat, embed),
    optionally per-class for embedding/output and router weights.
  - Magnitude: optional learnable per-axis gains as a fused reparameterization
    (p = normalized_w * phi(gains)); ``gain_parametrization`` selects phi ∈ {direct, softplus}.
  - Update: AdamW by default, or Muon orthogonalized updates (``use_orthogonal_updates``).

1D params (biases / norms) and (optionally) embedding/output params are routed to a chained
external AdamW by :func:`get_megatron_mddecoupling_optimizer` — the ``MDDecoupling`` class itself
only ever steps on 2D params. Setting ``hypersphere_gains_mode=None`` disables the gain
machinery, so a single class covers both the gains and no-gains cases.
"""

import logging
import math
from typing import Any, Callable, Dict, List, Literal, Optional

import torch

from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_size, log_single_rank

# NOTE: these package-level imports are safe because this module is only imported lazily from
# megatron.training.training (never from this package's __init__), so megatron.core.optimizer is
# fully initialized by the time this module loads — same pattern as muon.py.
from . import (
    HAVE_EMERGING_OPTIMIZERS,
    _get_param_groups,
    get_megatron_optimizer,
    get_standard_config_overrides,
)
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig, ParamKey, ParamPredicate

try:
    import emerging_optimizers
    from emerging_optimizers.orthogonalized_optimizers import (
        get_muon_scale_factor as _emerging_get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp
except ImportError:
    emerging_optimizers = None


logger = logging.getLogger(__name__)


def _get_muon_scale_factor(size_out: int, size_in: int, mode: str = "spectral") -> float:
    """Muon orthogonalization scale factor.

    ``shape_up`` (= max(d_out/d_in, d_in/d_out)**0.5) and ``none`` (= 1.0) are handled here;
    all other modes delegate to the emerging_optimizers implementation.
    """
    if mode == "shape_up":
        return max(size_out / size_in, size_in / size_out) ** 0.5
    if mode == "none":
        return 1.0
    return _emerging_get_muon_scale_factor(size_out, size_in, mode=mode)


class _MDDecouplingBase(torch.optim.Optimizer):
    """AdamW + optional Muon orthogonalized updates + L2 hypersphere post-step weight clipping."""

    def __init__(
        self,
        params,
        # Common.
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        # Adam.
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        # Hypersphere (L2, post-step weight projection only).
        hypersphere_mode: Optional[Literal["row", "col", "flat", "embed"]] = None,
        hypersphere_embedding_mode: Optional[Literal["row", "col", "flat", "embed", "none"]] = None,
        hypersphere_router_mode: Optional[Literal["row", "col", "flat", "embed", "none"]] = None,
        hypersphere_eps: float = 1e-8,
        # Tangential-gradient / preserve-init options (off by default).
        hypersphere_tangential_grad: bool = False,
        hypersphere_preserve_init: bool = False,
        # When True, scale the hypersphere target radius for is_out_proj params (linear_proj,
        # linear_fc2) by 1/sqrt(2 * num_layers) — matches scaled_init_method_normal so the
        # constraint preserves Megatron's depth-aware init for residual-out projections.
        hypersphere_scale_out_proj_init: bool = False,
        num_layers: Optional[int] = None,
        # Place each flat-mode matrix's sphere at its init Frobenius norm (init_std=1/sqrt(hidden))
        # instead of the shape-native sqrt(max(d_out,d_in)). Fixes the narrow matrices (MLA lora,
        # MoE fc2, GQA K/V) whose smaller dim != hidden and would otherwise be pulled off init.
        hypersphere_radius_from_init: bool = False,
        hidden_size: Optional[int] = None,
        # Muon (orthogonalized updates).
        use_orthogonal_updates: bool = False,
        momentum_beta: float = 0.95,
        use_nesterov: bool = True,
        split_qkv: bool = True,
        qkv_split_shapes: Optional[tuple[int, int, int]] = None,
        qkv_dim: Optional[int] = None,
        is_qkv_fn: Optional[Callable[[torch.Tensor], bool]] = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ):
        self.fp32_matmul_prec = fp32_matmul_prec
        self.use_nesterov = use_nesterov

        self.hypersphere_mode = hypersphere_mode
        self.hypersphere_embedding_mode = hypersphere_embedding_mode
        self.hypersphere_router_mode = hypersphere_router_mode
        self.hypersphere_eps = hypersphere_eps
        self.hypersphere_tangential_grad = hypersphere_tangential_grad
        self.hypersphere_preserve_init = hypersphere_preserve_init
        if hypersphere_scale_out_proj_init:
            assert num_layers is not None and num_layers > 0, (
                "hypersphere_scale_out_proj_init=True requires num_layers"
            )
            self.out_proj_radius_scale = 1.0 / math.sqrt(2 * num_layers)
        else:
            self.out_proj_radius_scale = 1.0

        self.hypersphere_radius_from_init = hypersphere_radius_from_init
        self.hidden_size = hidden_size
        if hypersphere_radius_from_init:
            assert hidden_size is not None and hidden_size > 0, (
                "hypersphere_radius_from_init=True requires hidden_size"
            )

        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn if is_qkv_fn is not None else (lambda p: False)
        self.qkv_split_shapes = qkv_split_shapes
        self.qkv_dim = qkv_dim

        self.coefficient_type = coefficient_type
        self.num_ns_steps = num_ns_steps
        self.scale_mode = scale_mode
        self.extra_scale_factor = extra_scale_factor

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta1=betas[0],
            beta2=betas[1],
            momentum_beta=momentum_beta,
            step=0,
            eps=eps,
            use_orthogonal_updates=use_orthogonal_updates,
        )
        super().__init__(params, defaults)

        # Ckpt-resume workaround: Megatron's _filter_and_reorder_param_groups (optimizer.py) keys
        # saved groups by (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr). MDDecoupling's
        # overrides differ only in max_lr / min_lr / use_orthogonal_updates, so multiple groups
        # (matrix, embedding, LM-head, router, ...) collide on that tuple. The loader's dict-based
        # map then silently overwrites colliding entries and reassigns the surviving group's
        # `params` repeatedly, which trips torch.optim.Optimizer.load_state_dict's per-group size
        # check. lr_mult is unused at runtime (OptimizerParamScheduler reads max_lr/min_lr), so a
        # unique tag per group disambiguates the loader without touching training math. Save/load
        # round-trip works because both jobs run this same code in the same param-group order.
        for _i, _g in enumerate(self.param_groups):
            _g['lr_mult'] = float(_i + 1)

        # Normalize parameters at init so the first forward sees on-sphere weights. When
        # hypersphere_preserve_init=True we skip this so the model's init magnitude survives into
        # training (gains absorb the magnitude downstream in MDDecoupling._maybe_init_gain_state).
        if (not self.hypersphere_preserve_init
                and (self.hypersphere_mode is not None
                     or self.hypersphere_embedding_mode is not None
                     or self.hypersphere_router_mode is not None)):
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.ndim != 2 and not (len(p.shape) == 3 and getattr(p, "merged_offload_expert", False)):
                            continue
                        is_qkv = self.is_qkv_fn(p)
                        is_out_proj = getattr(p, "is_out_proj", False)
                        is_embedding = getattr(p, "is_embedding_or_output_parameter", False)
                        is_router = getattr(p, "is_router", False)
                        is_merged_offload_expert = getattr(p, "merged_offload_expert", False)
                        self._normalize(p, p, is_qkv=is_qkv, is_out_proj=is_out_proj,
                                        is_embedding=is_embedding, is_router=is_router,
                                        is_merged_offload_expert=is_merged_offload_expert)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] += 1
            if "momentum_beta" not in group:  # Old checkpoint compat.
                group["momentum_beta"] = group["beta1"]
            for p in group["params"]:
                if p.grad is not None:
                    self._param_step(p, group)

        return loss

    def _param_step(self, p, group):
        grad = p.grad
        state = self.state[p]

        if "exp_avg" not in state:  # row_gain may already be present from gains.
            state["exp_avg"] = torch.zeros_like(grad)
            if not group["use_orthogonal_updates"] and group["beta2"] != 0:
                state["exp_avg_sq"] = torch.zeros_like(grad)

        exp_avg = state["exp_avg"]
        beta1 = group["beta1"]
        momentum_beta = group["momentum_beta"]
        is_qkv = self.is_qkv_fn(p)
        is_out_proj = getattr(p, "is_out_proj", False)
        is_embedding = getattr(p, "is_embedding_or_output_parameter", False)
        is_router = getattr(p, "is_router", False)
        is_merged_offload_expert = getattr(p, "merged_offload_expert", False)

        # Strip the radial component of grad before it feeds any momentum buffer or 2nd-moment
        # estimate (applies to both Muon and AdamW).
        if self.hypersphere_tangential_grad:
            self._project_tangent_inplace(
                p, grad, is_qkv=is_qkv, is_out_proj=is_out_proj,
                is_embedding=is_embedding, is_router=is_router,
                is_merged_offload_expert=is_merged_offload_expert,
            )

        if group["use_orthogonal_updates"]:  # Muon branch.
            assert emerging_optimizers is not None, (
                "emerging_optimizers package required for --use-orthogonal-updates"
            )
            self._apply_weight_decay_inplace(p, group)
            exp_avg.lerp_(grad, 1 - momentum_beta)
            if self.use_nesterov:
                grad = grad.lerp(exp_avg, momentum_beta)
            else:
                grad = exp_avg
            flat_mode = self._resolve_mode(is_embedding, is_router) == "flat"
            with emerging_optimizers.utils.fp32_matmul_precision(self.fp32_matmul_prec):
                update = self._orthogonalize_param(p, grad, is_qkv=is_qkv, flat_mode=flat_mode, is_merged_offload_expert=is_merged_offload_expert)
            # Shrink Muon update for is_out_proj params to match the smaller target sphere. Muon's
            # shape_up (and spectral) scale targets the natural RMS of a unit-row/col matrix; with
            # target radius 1/sqrt(2L) the bare update needs the same shrink factor.
            radius_scale = self._resolve_radius_scale(is_out_proj)
            if radius_scale != 1.0:
                update = update * radius_scale
        else:  # AdamW branch.
            beta2 = group["beta2"]
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            bias_correction1 = 1.0 - (beta1 ** group["step"])

            if beta2 == 0:  # plain SGD with momentum (no exp_avg_sq).
                update = exp_avg / bias_correction1
            else:  # Adam.
                exp_avg_sq = state["exp_avg_sq"]
                bias_correction2 = 1.0 - (beta2 ** group["step"])
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                update = exp_avg.div(bias_correction1) / denom

            self._apply_weight_decay_inplace(p, group)

        # Apply update.
        p.add_(update, alpha=-group["lr"])

        # Post-step hypersphere normalization (matrix clipping).
        is_valid_p = (p.ndim == 2 or (len(p.shape) == 3 and is_merged_offload_expert))
        if is_valid_p and self._resolve_mode(is_embedding, is_router) is not None:
            self._normalize(p, p, is_qkv=is_qkv, is_out_proj=is_out_proj,
                            is_embedding=is_embedding, is_router=is_router,
                            is_merged_offload_expert=is_merged_offload_expert)

    def _apply_weight_decay_inplace(self, p, group):
        weight_decay = group["weight_decay"]
        if weight_decay != 0:
            p.add_(p, alpha=-weight_decay * group["lr"])


    def _orthogonalize_param(self, p, grad, is_qkv: bool = False, flat_mode: bool = False, is_merged_offload_expert: bool = False):
        """Newton-Schulz orthogonalization, with optional QKV split. ``flat_mode`` enables the
        init-radius rescale (see _init_radius_scale); it is per-block so Q/K/V get their own factor
        under split_qkv."""
        if self.pg_collection is not None:
            tp_group = (self.pg_collection.expt_tp
                        if getattr(p, "expert_tp", False)
                        else self.pg_collection.tp)
        else:
            tp_group = None
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        if self.split_qkv and is_qkv:
            qs, ks, vs = _split_qkv(grad, self.qkv_split_shapes)
            qs = self._orthogonalize_tensor(qs, tp_group, partition_dim, flat_mode)
            ks = self._orthogonalize_tensor(ks, tp_group, partition_dim, flat_mode)
            vs = self._orthogonalize_tensor(vs, tp_group, partition_dim, flat_mode)
            return _merge_qkv((qs, ks, vs), grad.shape, self.qkv_split_shapes)

        if is_merged_offload_expert:
            experts = _split_experts(grad)
            for i, expert in enumerate(experts):
                experts[i] = self._orthogonalize_tensor(expert, tp_group, partition_dim, flat_mode)
            return _merge_experts(experts)
        return self._orthogonalize_tensor(grad, tp_group, partition_dim, flat_mode)

    def _orthogonalize_tensor(self, grad, tp_group, partition_dim, flat_mode: bool = False):
        assert grad.ndim == 2
        size = [grad.size(-2), grad.size(-1)]
        if partition_dim is not None:
            size[partition_dim] *= get_pg_size(tp_group)
        orth = newton_schulz_tp(
            grad,
            steps=self.num_ns_steps,
            coefficient_type=self.coefficient_type,
            tp_group=tp_group,
            partition_dim=partition_dim,
            tp_mode=("duplicated" if self.tp_mode == "blockwise" else self.tp_mode),
        )
        scale = _get_muon_scale_factor(size[0], size[1], mode=self.scale_mode)
        if flat_mode:
            scale *= self._init_radius_scale(size[0], size[1])
        return orth * scale * self.extra_scale_factor

    def _resolve_mode(self, is_embedding: bool, is_router: bool = False):
        if is_router and self.hypersphere_router_mode is not None:
            mode = self.hypersphere_router_mode
        elif is_embedding and self.hypersphere_embedding_mode is not None:
            mode = self.hypersphere_embedding_mode
        else:
            mode = self.hypersphere_mode
        return None if mode == "none" else mode

    def _resolve_radius_scale(self, is_out_proj: bool) -> float:
        """Effective hypersphere radius scale for this param. Returns 1.0 unless
        --hypersphere-scale-out-proj-init is on AND the param is is_out_proj. When gains +
        preserve_init are both active the gain absorbs the init magnitude
        (_maybe_init_gain_state), so we keep bare scale=1 to avoid double-counting."""
        if not is_out_proj or self.out_proj_radius_scale == 1.0:
            return 1.0
        if (getattr(self, "hypersphere_gains_mode", None) is not None
                and self.hypersphere_preserve_init):
            return 1.0
        return self.out_proj_radius_scale

    def _init_radius_scale(self, size_out: int, size_in: int) -> float:
        """Extra flat-mode radius multiplier that moves the sphere from the shape-native
        sqrt(max(d_out,d_in)) onto the model's init Frobenius norm. With init_std=1/sqrt(hidden),
        init_norm = sqrt(d_out*d_in/hidden), so init_norm/sqrt(max) = sqrt(min(d_out,d_in)/hidden).
        Sizes must be GLOBAL (TP-unsharded), matching the flat target / Muon scale. Returns 1.0 when
        disabled or when the smaller dim == hidden (already on the init sphere). Applied identically
        to the projection target and the Muon update so the two stay consistent; the out_proj depth
        factor cancels in the ratio and composes multiplicatively."""
        if not self.hypersphere_radius_from_init or self.hidden_size is None:
            return 1.0
        return (min(size_out, size_in) / self.hidden_size) ** 0.5

    def _project_tangent_inplace(self, p, grad, is_qkv: bool = False, is_out_proj: bool = False,
                                  is_embedding: bool = False, is_router: bool = False,
                                  is_merged_offload_expert: bool = False):
        """In-place: remove the radial component of `grad` w.r.t. the hypersphere mode at `p`.
        Mirrors _normalize's QKV-split layout so the constraint matches the post-step projection."""
        mode = self._resolve_mode(is_embedding, is_router)
        if mode is None:
            return
        if is_qkv and self.split_qkv:
            ps = _split_qkv(p, self.qkv_split_shapes)
            gs = _split_qkv(grad, self.qkv_split_shapes)
            for pi, gi in zip(ps, gs):
                self._project_tangent_single_(pi, gi, is_out_proj, mode)
            grad.copy_(_merge_qkv(gs, grad.size(), self.qkv_split_shapes))
            return
        if is_merged_offload_expert:
            ps = _split_experts(p)
            gs = _split_experts(grad)
            for pi, gi in zip(ps, gs):
                self._project_tangent_single_(pi, gi, is_out_proj, mode)
            grad.copy_(_merge_experts(gs))
            return
        self._project_tangent_single_(p, grad, is_out_proj, mode)

    def _project_tangent_single_(self, p, grad, is_out_proj: bool, mode: str):
        if mode == "col":
            dim = 0
        elif mode == "row":
            dim = 1
        elif mode == "embed":
            dim = 0 if is_out_proj else 1
        elif mode == "flat":
            dim = None
        else:
            return
        if dim is None:
            p_norm_sq = (p * p).sum().clamp_min(self.hypersphere_eps)
            radial = (p * grad).sum() / p_norm_sq
        else:
            p_norm_sq = (p * p).sum(dim=dim, keepdim=True).clamp_min(self.hypersphere_eps)
            radial = (p * grad).sum(dim=dim, keepdim=True) / p_norm_sq
        grad.sub_(p * radial)

    def _normalize(self, p, x, is_qkv: bool = False, is_out_proj: bool = False,
                   is_embedding: bool = False, is_router: bool = False, 
                   is_merged_offload_expert: bool = False):
        """In-place L2-sphere projection of a 2D tensor `x` (sized like `p`).

        For QKV-merged weights, normalize each of Q/K/V separately. Modes:
            row    → unit-norm each output row     (dim=1)
            col    → unit-norm each input column   (dim=0)
            flat   → unit Frobenius then scale by sqrt(max(d_out, d_in))
            embed  → row for non-out_proj, col for out_proj
        """
        mode = self._resolve_mode(is_embedding, is_router)
        if mode is None:
            return

        partition_dim = getattr(p, "partition_dim", None)
        radius_scale = self._resolve_radius_scale(is_out_proj)

        is_expert_tp = getattr(p, "expert_tp", False)
        if is_qkv and self.split_qkv:
            qs, ks, vs = _split_qkv(x, self.qkv_split_shapes)
            self._normalize_single(qs, is_out_proj, mode, radius_scale, partition_dim, is_expert_tp)
            self._normalize_single(ks, is_out_proj, mode, radius_scale, partition_dim, is_expert_tp)
            self._normalize_single(vs, is_out_proj, mode, radius_scale, partition_dim, is_expert_tp)
            x.copy_(_merge_qkv((qs, ks, vs), x.size(), self.qkv_split_shapes))
            return
        
        # TODO(fuguan): for now merged experts follow the same pipeline of normalization
        # as qkv. But it might introduce high memory cost. 
        if is_merged_offload_expert:
            experts = _split_experts(x)
            for expert in experts:
                self._normalize_single(expert, is_out_proj, mode, radius_scale, partition_dim, is_expert_tp)
            x.copy_(_merge_experts(experts))
            return

        self._normalize_single(x, is_out_proj, mode, radius_scale, partition_dim, is_expert_tp)

    def _normalize_single(self, x, is_out_proj: bool, mode: str, radius_scale: float = 1.0,
                          partition_dim: Optional[int] = None, is_expert_tp: bool = False):
        if mode == "col":
            dim = 0
        elif mode == "row":
            dim = 1
        elif mode == "flat":
            dim = None
        elif mode == "embed":
            dim = 0 if is_out_proj else 1
        else:
            raise ValueError(f"Unsupported hypersphere mode: {mode}")

        # Find norm of x, sync with TP group if needed.
        tp_group = self.pg_collection.expt_tp if is_expert_tp else self.pg_collection.tp
        if partition_dim in {0, 1} and (mode == "flat" or partition_dim == dim):
            norm = torch.norm(x, dim=dim, keepdim=True)
            norm_squared = norm**2
            torch.distributed.all_reduce(norm_squared, group=tp_group)
            norm = torch.sqrt(norm_squared).clamp_min(self.hypersphere_eps)
        else:
            norm = torch.norm(x, dim=dim, keepdim=True).clamp_min(self.hypersphere_eps)

        # Normalize x.
        x.div_(norm)
        if mode == "flat":
            tp_size = get_pg_size(tp_group) if partition_dim in {0, 1} else 1
            global_sizes = [x.size(-2), x.size(-1)]
            if partition_dim in {0, 1}:
                global_sizes[partition_dim] *= tp_size
            x.mul_(max(global_sizes) ** 0.5)
            init_scale = self._init_radius_scale(global_sizes[0], global_sizes[1])
            if init_scale != 1.0:
                x.mul_(init_scale)
        if radius_scale != 1.0:
            x.mul_(radius_scale)


class MDDecoupling(_MDDecouplingBase):
    """MDDecoupling with learnable per-axis gains as a fused reparameterization
    (p = normalized_w * phi(gains)).

    Setting `hypersphere_gains_mode=None` makes this class behave identically to plain
    `_MDDecouplingBase` — no gain tensors are created and the gain hooks in step() are no-ops.
    This lets a single class cover both cases.

    Gain optimizer state (1st/2nd moments) is tracked manually as plain tensors in
    `self.state[p]`. This avoids registering gain `nn.Parameter`s in an inner `AdamW` that lives
    outside `self.param_groups` — torch.optim's state_dict id-mapping doesn't survive that. Gain
    tensors have a different shape than `p`, so use `--ckpt-format torch`.
    """

    def __init__(
        self,
        params,
        hypersphere_gains_mode: Optional[Literal["row", "col", "rowcol", "flat", "embed"]] = None,
        hypersphere_gains_mode_output: Optional[Literal["row", "col", "rowcol", "flat", "none"]] = None,
        hypersphere_gains_mode_embedding: Optional[Literal["row", "col", "rowcol", "flat", "none"]] = None,
        hypersphere_gains_mode_router: Optional[Literal["row", "col", "rowcol", "flat", "none"]] = None,
        gains_lr: Optional[float] = None,
        gains_min_lr: Optional[float] = None,
        gains_betas: tuple[float, float] = (0.9, 0.999),
        gains_eps: float = 1e-8,
        gains_weight_decay: float = 0.0,
        # Reparametrize the per-axis gain: the stored state tensor is `g`, the effective
        # multiplier applied to `p` is `phi(g)`. "direct" is the identity (phi(g)=g);
        # "softplus" uses phi(g)=softplus(g). Applied uniformly to row/col/flat.
        gain_parametrization: Literal["direct", "softplus"] = "direct",
        # Drop the 1e-8 clamp_min on phi(g) in _preprocess_gains so gains can shrink through 1e-8
        # or flip sign (only meaningful for gain_parametrization="direct"; no-op for softplus).
        gains_no_clamp_min: bool = False,
        **kwargs,
    ):
        self.hypersphere_gains_mode = hypersphere_gains_mode
        self.hypersphere_gains_mode_output = hypersphere_gains_mode_output
        self.hypersphere_gains_mode_embedding = hypersphere_gains_mode_embedding
        self.hypersphere_gains_mode_router = hypersphere_gains_mode_router
        self.gains_no_clamp_min = gains_no_clamp_min
        self.gains_lr = gains_lr  # None → follow group["lr"] verbatim
        self.gains_min_lr = gains_min_lr  # gains LR floor (matches the group's min_lr policy)
        self.gains_betas = gains_betas
        self.gains_eps = gains_eps
        self.gains_weight_decay = gains_weight_decay
        self.gain_parametrization = gain_parametrization
        super().__init__(params, **kwargs)
        # Gain state is initialized lazily at first step (see step() → _maybe_init_gain_state).
        # Eager init here would write entries into self.state keyed by the bf16 model param, but
        # the Float16 wrapper later clones bf16 → fp32 main_params and migrates self.state only
        # for params still in param_groups on this rank. Deferring to step() keeps the keys
        # consistent with the post-wrap main_params.

    def _phi(self, g: torch.Tensor) -> torch.Tensor:
        """Forward map from raw gain g to effective multiplier phi(g)."""
        mode = self.gain_parametrization
        if mode == "direct":
            return g
        if mode == "softplus":
            return torch.nn.functional.softplus(g)
        raise ValueError(f"Unknown gain_parametrization {mode}")

    def _phi_prime(self, g: torch.Tensor):
        """Derivative phi'(g). Returns scalar 1.0 for the linear mode so the caller can skip
        a multiply."""
        mode = self.gain_parametrization
        if mode == "direct":
            return 1.0
        if mode == "softplus":
            return torch.sigmoid(g)
        raise ValueError(f"Unknown gain_parametrization {mode}")

    def _phi_inv(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse map: given a target effective multiplier x>0, return g s.t. phi(g) = x. Used
        at init to seed the raw gain so the first step's effective multiplier matches the desired
        target (1.0 in the identity case, ||p[i]|| / cur_frob in the preserve_init absorb case)."""
        mode = self.gain_parametrization
        if mode == "direct":
            return x
        if mode == "softplus":
            # Stable softplus_inv for x > 0: g = x + log1p(-exp(-x)).
            # As x → ∞, g → x. As x → 0+, g → −∞.
            return x + torch.log1p(-torch.exp(-x))
        raise ValueError(f"Unknown gain_parametrization {mode}")

    @torch.no_grad()
    def _tp_reduced_norm(self, x, dim, partition_dim=None, is_expert_tp=False):
        """L2 norm of `x` over `dim` (full Frobenius if dim is None), all-reduced across the TP
        group when the reduced axis is the TP-sharded one — so the preserve_init gains absorb the
        SAME global magnitude on TP=1 and TP>1. Mirrors the distributed-norm logic in
        _normalize_single:
            dim is None (flat) -> reduce iff the param is sharded (partition_dim in {0,1})
            dim in {0, 1}      -> reduce iff partition_dim == dim (reduced axis is sharded)
        """
        norm = torch.norm(x) if dim is None else torch.norm(x, dim=dim)
        should_reduce = self.pg_collection is not None and (
            partition_dim in {0, 1} if dim is None else partition_dim == dim
        )
        if should_reduce:
            tp_group = self.pg_collection.expt_tp if is_expert_tp else self.pg_collection.tp
            norm_squared = norm**2
            torch.distributed.all_reduce(norm_squared, group=tp_group)
            norm = torch.sqrt(norm_squared)
        return norm.to(torch.float32).clamp_min(self.hypersphere_eps)

    def _gain_layout(self, p):
        """Broadcast/sum/norm-axis layout for the per-axis gains of parameter p.

        A 2D weight (out, in) carries row_gain (out,), col_gain (in,) and a scalar
        flat_gain. Row reductions run over the in-axis (dim 1), col reductions over the out-axis (dim 0).

        A 3D merged offload-expert weight (num_local_experts, out, in) is a stack of
        independent (out, in) expert weights: row_gain (E, out), col_gain (E, in),
        flat_gain (E,). The reduced axes shift up by one (in-axis -> dim 2, out-axis -> dim 1)
        and the per-expert flat norm reduces over (1, 2) instead of the whole tensor. 
        The *_bk entries are the slice tuples used to broadcast the effective gains back onto p: 
        2D uses [:, None] / [None, :]; 3D adds a leading expert axis.
        """
        if getattr(p, "merged_offload_expert", False):
            E = p.size(0)
            return dict(
                row_shape=(E, p.size(-2)), col_shape=(E, p.size(-1)), flat_shape=(E,),
                row_norm=2, col_norm=1, row_sum=2, col_sum=1, spatial_dims=(1, 2),
                row_bk=(slice(None), slice(None), None),
                col_bk=(slice(None), None, slice(None)),
                flat_bk=(slice(None), None, None),
            )
        return dict(
            row_shape=(p.size(0),), col_shape=(p.size(1),), flat_shape=(),
            row_norm=1, col_norm=0, row_sum=1, col_sum=0, spatial_dims=None,
            row_bk=(slice(None), None), col_bk=(None, slice(None)), flat_bk=(),
        )

    def _maybe_init_gain_state(self, p):
        if self.hypersphere_gains_mode is None or p.ndim < 2:
            return
        mode = self._resolve_gains_mode(p)
        if mode is None or mode == "none":
            return
        state = self.state[p]
        if any(k in state for k in ("row_gain", "col_gain", "flat_gain")):
            return
        is_out_proj = getattr(p, "is_out_proj", False)
        preserve = self.hypersphere_preserve_init
        # TP sharding info: the absorbed norm must be reduced across the TP group when it reduces
        # over the sharded dimension, else TP=1 and TP>1 absorb different magnitudes.
        partition_dim = getattr(p, "partition_dim", None)
        is_expert_tp = getattr(p, "expert_tp", False)
        layout = self._gain_layout(p)

        wants_row = ("row" in mode) or (mode == "embed" and not is_out_proj)
        wants_col = ("col" in mode) or (mode == "embed" and is_out_proj)
        wants_flat = mode == "flat"

        # For preserve_init, absorb the original magnitude into exactly one axis (row wins when
        # both are configured — the rowcol canonical decomposition leaves col_gain at 1). Other
        # gain axes start at 1. We do NOT touch p.data: the decomposition is implicit at init — p
        # still holds the original weight, and the next step's _preprocess_gains will divide by
        # gain_init to recover bare_p before updating.
        absorb_axis = None
        if preserve:
            if wants_row:
                absorb_axis = "row"
            elif wants_col:
                absorb_axis = "col"
            elif wants_flat:
                absorb_axis = "flat"

        # The stored tensor is the raw gain `g`; the effective multiplier is phi(g). For the
        # identity branch we want phi(g)=1 → seed phi_inv(1). For the preserve_init absorb branch
        # we want phi(g)=||p[i]|| (or cur_frob/target_frob) → seed phi_inv of that.
        # For merged experts the leading axis makes each of these per-expert (shape (E, ...)).
        if wants_row:
            if absorb_axis == "row":
                # norm over the in-axis (dim 1 for 2D, dim 2 for 3D experts) → reduced iff
                # row-parallel (partition_dim == 1); per-expert → (E, out) for 3D.
                target = self._tp_reduced_norm(p.detach(), dim=layout["row_norm"],
                                               partition_dim=partition_dim,
                                               is_expert_tp=is_expert_tp)
            else:
                target = torch.ones(layout["row_shape"], dtype=torch.float32, device=p.device)
            state["row_gain"] = self._phi_inv(target)
            state["row_gain_m"] = torch.zeros_like(state["row_gain"])
            state["row_gain_v"] = torch.zeros_like(state["row_gain"])

        if wants_col:
            if absorb_axis == "col":
                # norm over the out-axis (dim 0 for 2D, dim 1 for 3D experts) → reduced iff
                # col-parallel (partition_dim == 0); per-expert → (E, in) for 3D.
                target = self._tp_reduced_norm(p.detach(), dim=layout["col_norm"],
                                               partition_dim=partition_dim,
                                               is_expert_tp=is_expert_tp)
            else:
                target = torch.ones(layout["col_shape"], dtype=torch.float32, device=p.device)
            state["col_gain"] = self._phi_inv(target)
            state["col_gain_m"] = torch.zeros_like(state["col_gain"])
            state["col_gain_v"] = torch.zeros_like(state["col_gain"])

        if wants_flat:
            if absorb_axis == "flat":
                # full Frobenius norm → reduced iff the param is sharded at all. target_frob uses
                # GLOBAL sizes so TP=1 and TP>1 match.
                # For 3D experts this is per-expert: norm over the spatial axes (1, 2) → (E,), target_frob is a shared scalar.
                gsizes = [p.size(-2), p.size(-1)]
                if self.pg_collection is not None and partition_dim in {0, 1}:
                    tp_group = self.pg_collection.expt_tp if is_expert_tp else self.pg_collection.tp
                    gsizes[partition_dim] *= get_pg_size(tp_group)
                target_frob = max(gsizes) ** 0.5
                cur_frob = self._tp_reduced_norm(p.detach(), dim=layout["spatial_dims"],
                                                 partition_dim=partition_dim,
                                                 is_expert_tp=is_expert_tp)
                target = cur_frob / target_frob
                if layout["spatial_dims"] is None:
                    target = target.reshape(())
            else:
                target = torch.ones(layout["flat_shape"], dtype=torch.float32, device=p.device)
            state["flat_gain"] = self._phi_inv(target)
            state["flat_gain_m"] = torch.zeros_like(state["flat_gain"])
            state["flat_gain_v"] = torch.zeros_like(state["flat_gain"])

    @torch.no_grad()
    def step(self, closure=None):
        if self.hypersphere_gains_mode is None:
            return super().step(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] += 1
            if "momentum_beta" not in group:
                group["momentum_beta"] = group["beta1"]
            for p in group["params"]:
                if p.grad is not None:
                    self._maybe_init_gain_state(p)
                    gain_grads = self._preprocess_gains(p)
                    self._param_step(p, group)
                    self._gains_step(p, group, gain_grads)
                    self._apply_gains(p)

        return loss

    @torch.no_grad()
    def _preprocess_gains(self, p):
        state = self.state[p]
        eps = 1e-8
        flat = state.get("flat_gain")
        row = state.get("row_gain")
        col = state.get("col_gain")
        if flat is None and row is None and col is None:
            return None

        layout = self._gain_layout(p)

        # Effective per-axis multipliers phi(g). For "direct" these are identical to the stored
        # raw gains (no-op overhead).
        flat_eff = self._phi(flat) if flat is not None else None
        row_eff = self._phi(row) if row is not None else None
        col_eff = self._phi(col) if col is not None else None

        # Undo gains to recover bare normalized weight. The clamp_min floors the divisor; under
        # --gains-no-clamp-min we skip it so the divide is the exact inverse of _apply_gains'
        # multiply for any nonzero gain (letting direct gains shrink through eps or flip sign).
        clamp = not self.gains_no_clamp_min
        if flat_eff is not None:
            p.div_(flat_eff[layout["flat_bk"]].clamp_min(eps) if clamp else flat_eff[layout["flat_bk"]])
        if row_eff is not None:
            p.div_(row_eff[layout["row_bk"]].clamp_min(eps) if clamp else row_eff[layout["row_bk"]])
        if col_eff is not None:
            p.div_(col_eff[layout["col_bk"]].clamp_min(eps) if clamp else col_eff[layout["col_bk"]])

        # Compute ∂L/∂phi(g) from bare p and the gain-baked grad still on p.grad.
        # Reduction dims shift up by one for 3D experts (in-axis dim 2, out-axis dim 1) 
        # so the sums land on the per-expert gain axes (E, out) / (E, in).
        p_times_pgrad = p * p.grad
        gain_grads = {}
        if flat is not None:
            assert row is None and col is None
            gain_grads["flat_gain"] = torch.sum(p_times_pgrad, dim=layout["spatial_dims"])
        elif row is not None and col is not None:
            gain_grads["row_gain"] = torch.sum(p_times_pgrad * col_eff[layout["col_bk"]], dim=layout["row_sum"])
            gain_grads["col_gain"] = torch.sum(p_times_pgrad * row_eff[layout["row_bk"]], dim=layout["col_sum"])
        elif row is not None:
            gain_grads["row_gain"] = torch.sum(p_times_pgrad, dim=layout["row_sum"])
        else:
            gain_grads["col_gain"] = torch.sum(p_times_pgrad, dim=layout["col_sum"])

        # Sync gains with TP group.
        # Skipped for merged experts (no partition_dim → not sharded;
        # EP just partitions which experts each rank owns).
        if self.pg_collection is not None:
            # partition_dim=0 -> column parallel sharding.
            # partition_dim=1 -> row parallel sharding.
            partition_dim = getattr(p, "partition_dim", None)
            if partition_dim in {0, 1}:  # Otherwise, p is not sharded so we don't need to sync it.
                tp_group = self.pg_collection.expt_tp if getattr(p, "expert_tp", False) else self.pg_collection.tp
                if flat is not None:  # Flat gains always need to all-reduce to complete the decomposition.
                    torch.distributed.all_reduce(gain_grads["flat_gain"], group=tp_group)
                if row is not None and partition_dim == 1:  # row gains sync only when row-sharded.
                    torch.distributed.all_reduce(gain_grads["row_gain"], group=tp_group)
                if col is not None and partition_dim == 0:  # col gains sync only when col-sharded.
                    torch.distributed.all_reduce(gain_grads["col_gain"], group=tp_group)

        # Chain rule: ∂L/∂g = phi'(g) · ∂L/∂phi(g). For "direct" phi'≡1 and _phi_prime returns
        # the scalar 1.0; skip the multiply.
        if self.gain_parametrization != "direct":
            for name in gain_grads:
                gain_grads[name] = gain_grads[name] * self._phi_prime(state[name])

        # Rescale p.grad so the base step sees ∂L/∂(bare p).
        if flat_eff is not None:
            p.grad.mul_(flat_eff[layout["flat_bk"]])
        if row_eff is not None:
            p.grad.mul_(row_eff[layout["row_bk"]])
        if col_eff is not None:
            p.grad.mul_(col_eff[layout["col_bk"]])

        return gain_grads

    @torch.no_grad()
    def _gains_step(self, p, group, gain_grads):
        if not gain_grads:
            return
        state = self.state[p]
        step = group["step"]
        beta1, beta2 = self.gains_betas
        eps = self.gains_eps
        wd = self.gains_weight_decay

        # Gains follow the param-group LR. If gains_lr is unset, use group["lr"] verbatim (so the
        # param scheduler drives gains too — already floored at the group's min_lr). Otherwise
        # schedule gains on their OWN [gains_min_lr, gains_lr] band using the scheduler's shared
        # decay coefficient, recovered from the group's lr:
        #     group["lr"] = min_lr + coeff * (max_lr - min_lr)   (OptimizerParamScheduler)
        #  => coeff = (group["lr"] - min_lr) / (max_lr - min_lr)
        # so gains decay gains_lr -> gains_min_lr exactly like a real param group (and honour
        # --min-lr-mode absolute, i.e. a fixed gains_min_lr floor — NOT the gains_lr * lr/max_lr
        # rescaling, which would floor at gains_lr * min_lr/max_lr and ignore min_lr).
        if self.gains_lr is None:
            lr = group["lr"]
        else:
            gmax = group.get("max_lr", group["lr"])
            gmin = group.get("min_lr", 0.0)
            denom = gmax - gmin
            coeff = ((group["lr"] - gmin) / denom) if denom > 0 else 1.0
            coeff = min(1.0, max(0.0, coeff))
            gains_floor = self.gains_min_lr if self.gains_min_lr is not None else 0.0
            lr = gains_floor + coeff * (self.gains_lr - gains_floor)

        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step

        # Persistent 0-dim fp32 scalar buffers (gain state is fp32), updated in
        # place each step. Reusing the same tensors keeps the compiled kernel's
        # guards stable across steps; updating via fill_ avoids per-step H2D copies.
        dev = p.device
        if getattr(self, "_gain_lr_buf", None) is None or self._gain_lr_buf.device != dev:
            self._gain_lr_buf = torch.zeros((), dtype=torch.float32, device=dev)
            self._gain_bc1_buf = torch.zeros((), dtype=torch.float32, device=dev)
            self._gain_bc2_buf = torch.zeros((), dtype=torch.float32, device=dev)
        self._gain_lr_buf.fill_(lr)
        self._gain_bc1_buf.fill_(bias_correction1)
        self._gain_bc2_buf.fill_(bias_correction2)

        for name, grad in gain_grads.items():
            gain = state[name]
            m = state[f"{name}_m"]
            v = state[f"{name}_v"]
            _fused_gain_adam(
                gain, m, v, grad,
                self._gain_lr_buf, self._gain_bc1_buf, self._gain_bc2_buf,
                beta1, beta2, eps, wd,
            )

    @torch.no_grad()
    def _apply_gains(self, p):
        state = self.state[p]
        flat = state.get("flat_gain")
        row = state.get("row_gain")
        col = state.get("col_gain")
        if flat is None and row is None and col is None:
            return
        lay = self._gain_layout(p)
        if flat is not None:
            p.mul_(self._phi(flat)[lay["flat_bk"]])
        if row is not None:
            p.mul_(self._phi(row)[lay["row_bk"]])
        if col is not None:
            p.mul_(self._phi(col)[lay["col_bk"]])

    def _resolve_gains_mode(self, p):
        is_output = getattr(p, "is_output_parameter", False)
        is_embedding = getattr(p, "is_embedding_parameter", False)
        is_router = getattr(p, "is_router", False)
        if is_router and self.hypersphere_gains_mode_router is not None:
            return self.hypersphere_gains_mode_router
        if is_output and self.hypersphere_gains_mode_output is not None:
            return self.hypersphere_gains_mode_output
        if is_embedding and self.hypersphere_gains_mode_embedding is not None:
            return self.hypersphere_gains_mode_embedding
        return self.hypersphere_gains_mode


@torch.compile(dynamic=True)
def _fused_gain_adam(gain, m, v, grad, lr, bc1, bc2, beta1, beta2, eps, wd):
    """Fused in-place Adam update for one gain tensor (compiled leaf kernel).

    lr / bc1 / bc2 are 0-dim tensors (they change every step, so passing them as
    tensors instead of Python floats avoids per-step recompilation); beta1, beta2,
    eps and wd are run constants. `dynamic=True` shares one graph across the
    differently shaped gain tensors. Equivalent to the eager update:
        if wd: gain *= 1 - lr*wd
        m  = beta1*m + (1-beta1)*grad
        v  = beta2*v + (1-beta2)*grad^2
        gain -= (lr/bc1) * m / (v.sqrt()/bc2.sqrt() + eps)
    """
    if wd != 0.0:
        gain.mul_(1.0 - lr * wd)
    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    denom = (v.sqrt() / bc2.sqrt()).add_(eps)
    gain.sub_((lr / bc1) * (m / denom))


def _split_qkv(x, shapes: tuple[int, int, int]) -> list[torch.Tensor]:
    """Split grouped attention (Q, K, V / GQA) along the head-group dim."""
    shape = x.shape
    num_query_groups = shape[0] // sum(shapes)
    qkv = torch.split(
        x.view(num_query_groups, sum(shapes), -1),
        shapes,
        dim=1,
    )
    return [g.reshape(-1, shape[-1]) for g in qkv]


def _merge_qkv(qkv, xshape: tuple[int, int], shapes: tuple[int, int, int]) -> torch.Tensor:
    num_query_groups = xshape[0] // sum(shapes)
    qkv = [g.view(num_query_groups, -1, xshape[-1]) for g in qkv]
    return torch.cat(qkv, dim=1).view(xshape)

def _split_experts(x: torch.Tensor) -> list[torch.Tensor]:
    assert len(x.shape) == 3, f"Expected 3D tensor, got {x.shape}"
    return list(torch.unbind(x, dim=0))

def _merge_experts(experts: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(experts, dim=0)


# ===========================================================================
# Optimizer builder
# ===========================================================================


def _group_min_lr(config: OptimizerConfig, max_lr: float) -> float:
    """Per-group min_lr policy for any param group with a custom max_lr.

    The LR scheduler decays every group from its own max_lr to its own min_lr along the same
    schedule shape. To keep the decay curve well-formed, a group's min_lr must be derived from its
    max_lr — not left at the global config.min_lr, which would invert the schedule when
    max_lr < config.min_lr. Controlled by config.min_lr_mode:

    'relative' (default): proportional decay — min_lr = max_lr * (config.min_lr / config.lr).
    'absolute': shared floor — min_lr = config.min_lr (clamped to <= max_lr).
    """
    floor = config.min_lr if config.min_lr is not None else 0.0
    if config.min_lr_mode == 'absolute':
        return min(floor, max_lr)
    ratio = (floor / config.lr) if (config.lr and config.lr > 0) else 0.0
    return max_lr * ratio


def _mddecoupling_config_overrides(
    config: OptimizerConfig, base_overrides: Dict[ParamKey, ParamGroupOverride]
) -> Dict[ParamKey, ParamGroupOverride]:
    """Build the config_overrides for MDDecoupling: matrix / embedding / output / router LR
    groups and per-group use_orthogonal_updates routing.

    Starts from ``base_overrides`` (the standard wd-skip + decoupled-LR overrides) and adds the
    MDDecoupling-specific entries on top. Returned dict is passed to BOTH the MDDecoupling param
    groups and the chained Adam ``get_megatron_optimizer`` call, so each override applies wherever
    its matching params materialize.
    """
    overrides: Dict[ParamKey, ParamGroupOverride] = dict(base_overrides) if base_overrides else {}
    matrix_lr = (config.matrix_lr if config.matrix_lr is not None
                 else config.muon_lr_factor * (config.lr or 0.0))

    # Embedding/output: when a hypersphere mode for embeddings is requested they stay in
    # MDDecoupling on the Adam branch (use_orthogonal_updates=False) + post-step normalization.
    # When it is None they are routed to the external chained Adam (see the param partition in
    # get_megatron_mddecoupling_optimizer), so no MDDecoupling override is needed here.
    if config.hypersphere_embedding_mode is not None:
        overrides[ParamKey(attr='is_embedding_or_output_parameter')] = {
            'use_orthogonal_updates': False
        }

    # MoE router branch override. Effective branch: explicit override > global flag.
    router_uses_adam = (
        config.md_router_use_orthogonal_updates is False
        or (config.md_router_use_orthogonal_updates is None
            and not config.use_orthogonal_updates)
    )
    router_override: ParamGroupOverride = {}
    if config.md_router_use_orthogonal_updates is not None:
        router_override['use_orthogonal_updates'] = config.md_router_use_orthogonal_updates
    if router_uses_adam:
        # Adam routers use base lr (matrix_lr is Muon-tuned; doesn't fit Adam).
        router_override['max_lr'] = config.lr
        router_override['min_lr'] = _group_min_lr(config, config.lr)
    if router_override:
        overrides[ParamKey(attr='is_router')] = router_override

    # Matrix LR for non-embedding/non-output matrix weights. Adam-branch routers are excluded — they
    # get base lr from the router_override above.
    # Offloaded inplace-FP8 expert weights are stored as a merged 3D tensor (E, out, in), 
    # but mathematically they are still per-expert matrices and must stay on the matrix LR schedule.
    non_emb_2d = ParamPredicate(
        name="md_non_embedding_or_output_matrix",
        fn=lambda p: (not getattr(p, "is_embedding_or_output_parameter", False)
                      and (len(p.shape) == 2 or getattr(p, "merged_offload_expert", False))
                      and not (router_uses_adam and getattr(p, "is_router", False))),
    )
    overrides[ParamKey(predicate=non_emb_2d)] = {
        'max_lr': matrix_lr,
        'min_lr': _group_min_lr(config, matrix_lr),
    }

    # Embedding LR (embedding + tied LM head share is_embedding_parameter).
    if config.embedding_lr_multiplier is not None:
        emb_lr = config.embedding_lr_multiplier * config.lr
        overrides[ParamKey(attr='is_embedding_parameter')] = {
            'max_lr': emb_lr,
            'min_lr': _group_min_lr(config, emb_lr),
        }

    # Output (untied LM head) LR: absolute, independent of embedding/scalar knobs.
    if config.output_lr is not None:
        output_only = ParamPredicate(
            name="md_output_not_embedding",
            fn=lambda p: (getattr(p, "is_output_parameter", False)
                          and not getattr(p, "is_embedding_parameter", False)),
        )
        overrides[ParamKey(predicate=output_only)] = {
            'max_lr': config.output_lr,
            'min_lr': _group_min_lr(config, config.output_lr),
        }

    return overrides


def _md_init_state_fn(opt, config=None):
    """Initialize MDDecoupling state for torch_dist checkpoint compatibility (gain state is still
    created lazily at first step; recommend --ckpt-format torch when gains are enabled)."""
    for group in opt.param_groups:
        for p in group['params']:
            if "exp_avg" not in opt.state[p]:
                opt.state[p]["exp_avg"] = torch.zeros_like(p.data)
                if not group.get("use_orthogonal_updates", False) and group.get("beta2", 0) != 0:
                    opt.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)


def _adam_init_state_fn(opt, config=None):
    for group in opt.param_groups:
        for p in group['params']:
            if len(opt.state[p]) == 0:
                if config is None or not config.use_precision_aware_optimizer:
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                    opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                else:
                    opt.initialize_state(p)


def get_megatron_mddecoupling_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Build the MDDecoupling optimizer for the given model chunks.

    Mirrors :func:`megatron.core.optimizer.muon.get_megatron_muon_optimizer`: 2D matrix params
    (plus routers, and embedding/output params when a hypersphere embedding mode is set) are
    optimized by :class:`MDDecoupling`; everything else (1D biases/norms, and embedding/output
    params when no embedding hypersphere mode is set) is delegated to a chained external AdamW via
    :func:`get_megatron_optimizer`. When ``layer_wise_distributed_optimizer`` is True the whole
    chain is wrapped in :class:`LayerWiseDistributedOptimizer` to shard optimizer state over DP.
    """
    assert HAVE_EMERGING_OPTIMIZERS or not config.use_orthogonal_updates, (
        "emerging-optimizers is required for --use-orthogonal-updates under --optimizer md_decoupling."
    )
    # The standard distributed optimizer flattens each param shard to 1-D, which breaks the 2-D
    # hypersphere/Muon math. Shard optimizer state via the layer-wise optimizer instead.
    if config.use_distributed_optimizer:
        raise Exception(
            'md_decoupling with the standard distributed optimizer is not supported; '
            'use --use-layer-wise-distributed-optimizer to shard optimizer state.'
        )
    if config.fp16:
        raise Exception('md_decoupling with fp16 is not supported (use bf16 or fp32).')

    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up MDDecoupling optimizer with config {config}')

    base_overrides = (
        config_overrides if config_overrides is not None else get_standard_config_overrides(config)
    )

    # Tag parameters with optimizer-specific attributes and compute qkv split shapes.
    qkv_split_shapes: Optional[List[int]] = None
    for model_chunk in model_chunks:
        cfg = model_chunk.config
        qkv_split_shapes = [
            cfg.num_attention_heads // cfg.num_query_groups * cfg.kv_channels,
            cfg.kv_channels,
            cfg.kv_channels,
        ]
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # the merged offloading-expert weight is a single 3D (num_local_experts, out, in) tensor
            if ('experts' in name and 'shared' not in name and len(param.shape) == 3
                    and model_chunk.config.moe_use_offloading_experts
                    and model_chunk.config.moe_use_inplace_fp8_param):
                param.merged_offload_expert = True
            # TODO(deyuf): support MLA
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            if name.endswith('router.weight') and len(param.shape) == 2:
                param.is_router = True
            
            is_out_proj = (('linear_fc2' in name or 'linear_proj' in name) and len(param.shape) == 2) or \
                          ('experts.weight2' in name and len(param.shape) == 3)
            if is_out_proj:
                param.is_out_proj = True

    # Partition params: MDDecoupling-managed ("linear") vs external chained Adam ("nonlinear").
    emb_in_md = config.hypersphere_embedding_mode is not None
    linear_params = []
    nonlinear_params = []
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            is_emb = getattr(param, 'is_embedding_or_output_parameter', False)
            is_merged_offload_expert = getattr(param, 'merged_offload_expert', False)
            if len(param.shape) == 2 and (not is_emb or emb_in_md):
                linear_params.append(param)
            elif len(param.shape) == 3 and is_merged_offload_expert:
                # In OffloadingExpert with FP8, the expert weight is 3D (num_local_experts, out, in).
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    md_overrides = _mddecoupling_config_overrides(config, base_overrides)

    md_kwargs = dict(
        lr=(config.matrix_lr if config.matrix_lr is not None
            else config.muon_lr_factor * (config.lr or 0.0)),
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        hypersphere_mode=config.hypersphere_mode,
        hypersphere_embedding_mode=config.hypersphere_embedding_mode,
        hypersphere_router_mode=config.hypersphere_router_mode,
        hypersphere_tangential_grad=config.hypersphere_tangential_grad,
        hypersphere_preserve_init=config.hypersphere_preserve_init,
        hypersphere_scale_out_proj_init=config.hypersphere_scale_out_proj_init,
        num_layers=model_chunks[0].config.num_layers,
        hypersphere_radius_from_init=config.hypersphere_radius_from_init,
        hidden_size=model_chunks[0].config.hidden_size,
        use_orthogonal_updates=config.use_orthogonal_updates,
        momentum_beta=config.muon_momentum,
        use_nesterov=config.muon_use_nesterov,
        split_qkv=config.muon_split_qkv,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=qkv_split_shapes,
        qkv_dim=model_chunks[0].config.kv_channels,
        fp32_matmul_prec=config.muon_fp32_matmul_prec,
        coefficient_type=config.muon_coefficient_type,
        num_ns_steps=config.muon_num_ns_steps,
        scale_mode=config.muon_scale_mode,
        extra_scale_factor=config.muon_extra_scale_factor,
        pg_collection=pg_collection,
        tp_mode=config.muon_tp_mode,
        hypersphere_gains_mode=config.hypersphere_gains_mode,
        hypersphere_gains_mode_output=config.hypersphere_gains_mode_output,
        hypersphere_gains_mode_embedding=config.hypersphere_gains_mode_embedding,
        hypersphere_gains_mode_router=config.hypersphere_gains_mode_router,
        gains_lr=config.gains_lr if config.gains_lr is not None else config.lr,
        # Gains decay on their own band down to this floor, honouring --min-lr-mode
        # (absolute → gains_min_lr = min(config.min_lr, gains_base) = config.min_lr).
        gains_min_lr=_group_min_lr(
            config, config.gains_lr if config.gains_lr is not None else config.lr
        ),
        gains_betas=(config.adam_beta1, config.adam_beta2),
        gains_eps=config.adam_eps,
        gains_weight_decay=config.weight_decay,
        gain_parametrization=config.gain_parametrization,
        gains_no_clamp_min=config.gains_no_clamp_min,
    )

    optimizers = []

    # Freeze nonlinear params so only MDDecoupling-managed params materialize in these groups.
    for param in nonlinear_params:
        param.requires_grad = False

    md_param_groups = _get_param_groups(model_chunks, config, md_overrides)
    # When NOT layer-wise, expert-parallel params need a separate optimizer instance (their grad
    # stats reduce over the expert TP-PP group). The layer-wise optimizer handles experts itself.
    expert_param_groups = []
    if not layer_wise_distributed_optimizer:
        expert_param_groups = [g for g in md_param_groups if g['is_expert_parallel']]
        md_param_groups = [g for g in md_param_groups if not g['is_expert_parallel']]

    optimizer = MDDecoupling(md_param_groups, **md_kwargs)

    reset_config_bf16 = False
    if config.bf16:
        if layer_wise_distributed_optimizer:
            # Creating master weights before layer-wise sharding wastes memory, so defer master
            # weight creation into the layer-wise wrapper. Unsetting config.bf16 also keeps the
            # adam optimizers below unwrapped so the wrapper can manage them uniformly.
            config.bf16 = False
            reset_config_bf16 = True
        else:
            optimizer = Float16OptimizerWithFloat16Params(optimizer, config, None, _md_init_state_fn)
    else:
        optimizer = FP32Optimizer(optimizer, config, _md_init_state_fn)
    optimizers.append(optimizer)

    if len(expert_param_groups) > 0:
        expert_optimizer = MDDecoupling(expert_param_groups, **md_kwargs)
        if config.bf16:
            expert_optimizer = Float16OptimizerWithFloat16Params(
                expert_optimizer, config, None, _md_init_state_fn
            )
        else:
            expert_optimizer = FP32Optimizer(expert_optimizer, config, _md_init_state_fn)
        setattr(expert_optimizer, 'grad_stats_parallel_group', pg_collection.tp_ep_pp)
        optimizers.append(expert_optimizer)

    # Done with MDDecoupling: unfreeze nonlinear, freeze linear, build the chained Adam for the
    # rest. Linear params are skipped by get_megatron_optimizer because they're frozen.
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    prev_optimizer = config.optimizer
    config.optimizer = 'adam'
    chained_adam = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=md_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )
    config.optimizer = prev_optimizer

    # Unfreeze everything.
    for param in linear_params:
        param.requires_grad = True

    init_fns = [_md_init_state_fn] + len(chained_adam.chained_optimizers) * [_adam_init_state_fn]
    optimizers += chained_adam.chained_optimizers

    if layer_wise_distributed_optimizer:
        log_single_rank(logger, logging.INFO, 'Using LayerWiseDistributedOptimizer for MDDecoupling')
        if reset_config_bf16:
            config.bf16 = True
        return LayerWiseDistributedOptimizer(
            optimizers,
            config,
            pg_collection,
            init_state_fn_list=init_fns,
            model_chunks=model_chunks,
            async_allgather=config.overlap_param_gather,
        )
    return ChainedOptimizer(optimizers)
