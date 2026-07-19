# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.fusions.fused_polynorm_glu import (
    MAX_FUSED_FEATURE_DIM,
    HAVE_TRITON as HAVE_FUSED_PNGLU,
    fused_polynorm_glu_impl,
)
from megatron.core.fusions.fused_gxpr_glu import fused_gxpr_glu_impl
from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule


@jit_fuser
def compiled_polynorm(x, alpha_1, alpha_2, eps: float = 1e-6):
    """Core PolyNorm GLU gate: ``a1*RMSNorm(x) + a2*RMSNorm(x**2)``.

    The RMS normalization is taken over the last (feature) dimension. The math is done in
    fp32 for numerical stability and cast back to the input dtype, mirroring how the
    RMSNorm/LayerNorm layers in this codebase behave under mixed precision.

    ``alpha_1``/``alpha_2`` broadcast against ``x``. They are either a single
    (broadcastable) coefficient of shape ``(1,)`` (dense / single-expert case) or per-token
    coefficients of shape ``(num_tokens, 1)`` (grouped-expert case, where each token already
    carries the coefficient of the expert it was routed to).

    This is the (torch.compile-fused) **gate-only** computation used by the non-Triton fallback
    paths; the CUDA fast path fuses the gate, the ``* x_linear`` and the ``* score`` multiplies in
    a single Triton kernel (see ``fused_polynorm_glu_impl``).
    """
    input_dtype = x.dtype
    x = x.float()

    def norm(t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)

    out = alpha_1 * norm(x) + alpha_2 * norm(x * x)
    return out.to(input_dtype)


@jit_fuser
def compiled_poly3norm(x, alpha_1, alpha_2, alpha_3, eps: float = 1e-6):
    """3rd-order PolyNorm gate: ``a1*RMSNorm(x) + a2*RMSNorm(x**2) + a3*RMSNorm(x**3)``.

    Same conventions as :func:`compiled_polynorm`, extended with one more (odd) term.
    """
    input_dtype = x.dtype
    x = x.float()

    def norm(t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)

    out = alpha_1 * norm(x) + alpha_2 * norm(x * x) + alpha_3 * norm(x * x * x)
    return out.to(input_dtype)


class _AllReduceSumSymmetric(torch.autograd.Function):
    """All-reduce(sum) over ``group`` in BOTH the forward and backward passes.

    Used to turn each rank's partial feature-sum into the full sum when the result is then
    consumed independently on every rank (each rank normalizes its own tokens with the shared
    statistic). Because the reduced value feeds rank-local downstream work, the gradient must
    be summed back across the group — unlike ``reduce_from_tensor_model_parallel_region``
    (forward all-reduce, backward identity), which is only correct when the reduced value feeds
    *replicated* downstream work.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        x = x.clone()
        torch.distributed.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        torch.distributed.all_reduce(grad, group=ctx.group)
        return grad, None


class _SyncGradSum(torch.autograd.Function):
    """Identity in the forward pass; all-reduce(sum) the gradient over ``group`` in backward.

    Applied to the (TP-replicated) alpha coefficients so each rank's partial coefficient
    gradient — a sum over only that rank's feature shard — is completed into the full gradient,
    keeping the replicas in sync. (Same semantics as ``copy_to_tensor_model_parallel_region``,
    but over an arbitrary group so it also works for the expert-tensor-parallel group.)
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        torch.distributed.all_reduce(grad, group=ctx.group)
        return grad, None


class PolyNorm(MegatronModule):
    """Learnable PolyNorm GLU activation — a drop-in replacement for the gate of a gated
    linear unit (e.g. SiLU in SwiGLU).

    In a GLU the first linear layer produces ``[x_glu, x_linear]`` and the block output is
    ``gate(x_glu) * x_linear``. Standard SwiGLU uses ``gate = SiLU``. Here the gate is the
    (2nd-order) PolyNorm::

        gate(x) = |alpha_1| * RMSNorm(x) + |alpha_2| * RMSNorm(x ** 2)

    where ``alpha_1``/``alpha_2`` are learnable (``abs`` keeps them positive).

    ``forward`` takes *both* GLU halves and returns the full ``gate(x_glu) * x_linear * [score]``
    product. On CUDA (and ``tp_size == 1``) the gate, the GLU multiply and the optional per-token
    ``score`` multiply (MoE router probs / per-token scale) are fused into a single Triton kernel
    (see ``megatron.core.fusions.fused_polynorm_glu``) so the op runs close to SwiGLU speed and is
    shape-agnostic over the (variable) MoE token count. Otherwise the gate is computed with the
    torch fallback (``compiled_polynorm`` or, when TP-sharded, ``_tp_forward``) and the
    multiplies are applied in eager torch.

    To support grouped MoE experts (where the activations of all local experts are
    concatenated along the token dimension and processed in a single call) this module holds
    one ``(alpha_1, alpha_2)`` pair *per local expert*: ``alpha_1``/``alpha_2`` have shape
    ``(num_local_experts,)``. When ``tokens_per_expert`` is supplied the
    per-expert coefficients are expanded to per-token coefficients, so every token is gated by the
    coefficients of the expert it was routed to. For a dense MLP (or a ``SequentialMLP``
    expert) ``num_local_experts == 1`` and the single coefficient is broadcast to all tokens.

    Tensor parallelism: the RMSNorm reduces over the ffn feature dimension, which is sharded
    across ``tp_group`` (the main TP group for dense/shared MLPs, the expert-TP group for MoE
    experts). When ``tp_group`` has size > 1, the per-token sum-of-squares is all-reduced over
    the group (forward and backward) so every rank uses the *full-feature* RMS, and the
    replicated ``alpha`` gradients are all-reduced over the group so the replicas stay in sync.
    The result is therefore identical to (and bitwise-consistent across) any TP/ETP degree.

    ``num_terms=3`` (``--pn3glu``) adds a 3rd (odd) term ``|alpha_3| * RMSNorm(x ** 3)``; this
    variant has no fused Triton kernel and always runs the torch fallback. ``num_terms=2``
    (default, ``--pnglu``) is unchanged and keeps the fused-kernel fast path.
    """

    def __init__(
        self,
        num_local_experts: int = 1,
        config=None,
        alpha_init: float = 0.2,
        eps: float = 1e-6,
        tp_group: "torch.distributed.ProcessGroup | None" = None,
        num_terms: int = 2,
    ):
        super().__init__(config=config)
        assert num_terms in (2, 3), f"PolyNorm supports num_terms in (2, 3), got {num_terms}."
        self.num_local_experts = num_local_experts
        self.num_terms = num_terms
        self.alpha_1 = nn.Parameter(torch.full((num_local_experts,), alpha_init))
        self.alpha_2 = nn.Parameter(torch.full((num_local_experts,), alpha_init))
        if num_terms == 3:
            self.alpha_3 = nn.Parameter(torch.full((num_local_experts,), alpha_init))
        self.eps = eps
        # The group over which the ffn feature dimension is sharded. tp_size==1 (no sharding,
        # e.g. local CPU runs or ETP=1 experts) takes the cheap fused path with no collectives.
        self.tp_group = tp_group
        if tp_group is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
            self.tp_size = torch.distributed.get_world_size(group=tp_group)
        else:
            self.tp_size = 1

    def _raw_coeffs(self, x, tokens_per_expert):
        """Return ``(alpha_1, alpha_2, alpha_3_or_None)``, positive, NOT yet broadcast-shaped.

        ``alpha_3`` is ``None`` when ``num_terms == 2``. Kept un-unsqueezed (shape
        ``(num_local_experts,)`` or ``(num_tokens,)``) because the fused Triton kernel expects
        that shape directly; :meth:`_broadcast_coeffs` produces the ``(num_tokens, 1)`` shape
        needed by the torch fallback.
        """
        alpha_1 = torch.abs(self.alpha_1)  # (num_local_experts,)
        alpha_2 = torch.abs(self.alpha_2)
        alpha_3 = torch.abs(self.alpha_3) if self.num_terms == 3 else None

        if self.num_local_experts == 1 or tokens_per_expert is None:
            if self.num_local_experts > 1:
                raise ValueError(
                    "PolyNorm with num_local_experts > 1 requires `tokens_per_expert` so "
                    "the per-expert coefficients can be mapped onto the concatenated tokens."
                )
            return alpha_1, alpha_2, alpha_3

        # Expand per-expert coefficients to per-token coefficients: shape (num_tokens,).
        if isinstance(tokens_per_expert, torch.Tensor):
            tokens_per_expert = tokens_per_expert.tolist()
        tpe_tensor = torch.tensor(tokens_per_expert, device=x.device)
        a1 = torch.repeat_interleave(alpha_1, tpe_tensor)
        a2 = torch.repeat_interleave(alpha_2, tpe_tensor)
        a3 = torch.repeat_interleave(alpha_3, tpe_tensor) if alpha_3 is not None else None
        return a1, a2, a3

    def _broadcast_coeffs(self, alpha_1, alpha_2, alpha_3):
        """Unsqueeze per-token coefficients to ``(num_tokens, 1)`` for the torch fallback."""
        a1b = alpha_1.unsqueeze(-1) if alpha_1.dim() == 1 and self.num_local_experts > 1 else alpha_1
        a2b = alpha_2.unsqueeze(-1) if alpha_2.dim() == 1 and self.num_local_experts > 1 else alpha_2
        a3b = None
        if alpha_3 is not None:
            a3b = alpha_3.unsqueeze(-1) if alpha_3.dim() == 1 and self.num_local_experts > 1 else alpha_3
        return a1b, a2b, a3b

    def _compute_gate_local(self, x, alpha_1, alpha_2, alpha_3):
        """Gate computation when the ffn feature dim is whole on this rank (``tp_size == 1``)."""
        if self.num_terms == 2:
            return compiled_polynorm(x, alpha_1, alpha_2, self.eps)
        return compiled_poly3norm(x, alpha_1, alpha_2, alpha_3, self.eps)

    def forward(self, x_glu, x_linear, tokens_per_expert=None, scores=None):
        """Return ``gate(x_glu) * x_linear * [scores]``.

        Args:
            x_glu: GLU gate half, ``(..., D)`` (``D`` = local ffn feature dim).
            x_linear: GLU linear half, same shape/dtype as ``x_glu``.
            tokens_per_expert: per-local-expert token counts (grouped experts only); maps the
                per-expert coefficients onto the concatenated tokens.
            scores: optional per-token multiplier ``(..., 1)`` (MoE router probs / per-token scale).
        """
        alpha_1, alpha_2, alpha_3 = self._raw_coeffs(x_glu, tokens_per_expert)

        use_fused = (
            self.num_terms == 2
            and HAVE_FUSED_PNGLU
            and x_glu.is_cuda
            and self.tp_size == 1
            and x_glu.shape[-1] <= MAX_FUSED_FEATURE_DIM
            and (self.config is None or getattr(self.config, "pnglu_fusion", True))
        )
        if use_fused:
            # Single fused kernel: gate + (* x_linear) + (* scores), shape-agnostic over tokens.
            return fused_polynorm_glu_impl(x_glu, x_linear, alpha_1, alpha_2, self.eps, scores)

        # Fallback: compute the gate in torch, then apply the multiplies in eager mode.
        a1b, a2b, a3b = self._broadcast_coeffs(alpha_1, alpha_2, alpha_3)
        if self.tp_size == 1:
            # ffn feature dim is whole on this rank: cheap fused per-token norm.
            gate = self._compute_gate_local(x_glu, a1b, a2b, a3b)
        else:
            # ffn feature dim is TP-sharded: reduce the feature statistics across the group.
            gate = self._tp_forward(x_glu, a1b, a2b, a3b)
        out = gate * x_linear
        if scores is not None:
            original_dtype = out.dtype
            out = (out * scores).to(original_dtype)
        return out

    def _tp_forward(self, x, alpha_1, alpha_2, alpha_3=None):
        """TP-invariant path: recover the full-feature RMS from the local feature shards."""
        input_dtype = x.dtype
        xf = x.float()
        # Each ColumnParallel rank holds an equal 1/tp_size slice of the ffn features.
        n_global = xf.shape[-1] * self.tp_size
        # Per-token partial feature sums on this rank: sum(x^2) and sum(x^4) (== sum((x^2)^2)) for
        # RMSNorm(x) and RMSNorm(x^2), plus sum(x^6) (== sum((x^3)^2)) for RMSNorm(x^3) when
        # num_terms == 3. One symmetric all-reduce completes all of them together.
        s1 = xf.pow(2).sum(-1, keepdim=True)
        s2 = xf.pow(2).pow(2).sum(-1, keepdim=True)
        stats = [s1, s2]
        if alpha_3 is not None:
            stats.append(xf.pow(3).pow(2).sum(-1, keepdim=True))
        s = _AllReduceSumSymmetric.apply(torch.cat(stats, dim=-1), self.tp_group)
        inv1 = torch.rsqrt(s[..., 0:1] / n_global + self.eps)
        inv2 = torch.rsqrt(s[..., 1:2] / n_global + self.eps)
        # alpha is replicated across the group; all-reduce its gradient so the replicas stay
        # in sync (forward is identity, so the value is unchanged).
        alpha_1 = _SyncGradSum.apply(alpha_1.float(), self.tp_group)
        alpha_2 = _SyncGradSum.apply(alpha_2.float(), self.tp_group)
        out = alpha_1 * (xf * inv1) + alpha_2 * (xf * xf * inv2)
        if alpha_3 is not None:
            inv3 = torch.rsqrt(s[..., 2:3] / n_global + self.eps)
            alpha_3 = _SyncGradSum.apply(alpha_3.float(), self.tp_group)
            out = out + alpha_3 * (xf * xf * xf * inv3)
        return out.to(input_dtype)


class PolyNormAct(PolyNorm):
    """Non-gated counterpart of PolyNorm/``--pn3glu``: returns the raw polynomial-RMSNorm gate
    value directly, applied to the MLP activation input (not multiplied by a second GLU half).

    ``gate(x) = |alpha_1| * RMSNorm(x) + |alpha_2| * RMSNorm(x**2) + |alpha_3| * RMSNorm(x**3)``

    Always uses ``num_terms=3`` ("up to the ``x**3`` term"). Shares :class:`PolyNorm`'s
    per-(local-)expert coefficients, TP handling, and ``tokens_per_expert`` expansion; only the
    calling convention (single input, no GLU multiply) differs.
    """

    def __init__(
        self,
        num_local_experts: int = 1,
        config=None,
        alpha_init: float = 0.2,
        eps: float = 1e-6,
        tp_group: "torch.distributed.ProcessGroup | None" = None,
    ):
        super().__init__(
            num_local_experts=num_local_experts,
            config=config,
            alpha_init=alpha_init,
            eps=eps,
            tp_group=tp_group,
            num_terms=3,
        )

    def forward(self, x, tokens_per_expert=None):
        """Return ``gate(x)``.

        Args:
            x: activation input, ``(..., D)``.
            tokens_per_expert: per-local-expert token counts (grouped experts only); maps the
                per-expert coefficients onto the concatenated tokens.
        """
        alpha_1, alpha_2, alpha_3 = self._raw_coeffs(x, tokens_per_expert)
        a1b, a2b, a3b = self._broadcast_coeffs(alpha_1, alpha_2, alpha_3)
        if self.tp_size == 1:
            return self._compute_gate_local(x, a1b, a2b, a3b)
        return self._tp_forward(x, a1b, a2b, a3b)


@jit_fuser
def compiled_xpr(x, alpha_p1, alpha_p2, alpha_n, beta):
    """Core XPR activation.

    ``alpha_p2 * x**3 + alpha_p1 * x**2 + beta * x`` for ``x > 0``, and
    ``alpha_n * x * softsign(x) + beta * x`` for ``x <= 0``.

    All coefficients broadcast against ``x``: either a single (broadcastable) coefficient of
    shape ``(1,)`` (dense / single-expert case) or per-token coefficients of shape
    ``(num_tokens, 1)`` (grouped-expert case, where each token already carries the coefficients
    of the expert it was routed to).
    """
    return torch.where(
        x > 0,
        alpha_p2 * x * x * x + alpha_p1 * x * x + beta * x,
        alpha_n * x * F.softsign(x) + beta * x,
    )


@jit_fuser
def compiled_xpr_gate(x, alpha_p1, alpha_p2, alpha_n, beta):
    """XPR GLU gate: ``compiled_xpr(x, ...) / x``, simplified algebraically to avoid the ``0/0``
    at ``x == 0``.

    ``alpha_p2 * x**2 + alpha_p1 * x + beta`` for ``x > 0``, and
    ``alpha_n * softsign(x) + beta`` for ``x <= 0``.
    """
    return torch.where(
        x > 0,
        alpha_p2 * x * x + alpha_p1 * x + beta,
        alpha_n * F.softsign(x) + beta,
    )


class XPR(MegatronModule):
    """Learnable elementwise activation (not a gated unit)::

        XPR(x) = |alpha_p2| * x**3 + |alpha_p1| * x**2 + |beta| * x                    (x > 0)
               = (|beta| + |alpha_n|) * x * softsign(x) + |beta| * x                    (x <= 0)

    ``alpha_p1``, ``alpha_p2``, ``alpha_n``, ``beta`` are learnable (``abs`` keeps them
    positive; the negative-branch coefficient is parameterized as ``beta + |alpha_n|`` so that,
    at init, it equals ``alpha_n_init`` regardless of ``beta_init``).

    To support grouped MoE experts (where the activations of all local experts are concatenated
    along the token dimension and processed in a single call) this module holds one coefficient
    set *per local expert*: each parameter has shape ``(num_local_experts,)``. When
    ``tokens_per_expert`` is supplied the per-expert coefficients are expanded to per-token
    coefficients, so every token is activated with the coefficients of the expert it was routed
    to. For a dense MLP (or a ``SequentialMLP`` expert) ``num_local_experts == 1`` and the single
    coefficient set is broadcast to all tokens.

    No fused kernel is provided yet; this always runs the (torch.compile-fused) eager
    implementation.
    """

    def __init__(
        self,
        num_local_experts: int = 1,
        config=None,
        alpha_p_init: float = 0.8,
        alpha_p2_init: float = 0.4,
        alpha_n_init: float = 0.8,
        beta_init: float = 0.5,
    ):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.alpha_p1 = nn.Parameter(torch.full((num_local_experts,), alpha_p_init))
        self.alpha_p2 = nn.Parameter(torch.full((num_local_experts,), alpha_p2_init))
        self.alpha_n = nn.Parameter(torch.full((num_local_experts,), alpha_n_init - beta_init))
        self.beta = nn.Parameter(torch.full((num_local_experts,), beta_init))

    def _raw_coeffs(self, x, tokens_per_expert):
        """Return ``(alpha_p1, alpha_p2, alpha_n, beta)``, positive, NOT yet broadcast-shaped.

        Shape is ``(num_local_experts,)`` in the dense/shared case (numel 1) or ``(num_tokens,)``
        in the grouped case -- not yet unsqueezed to ``(num_tokens, 1)``. :meth:`_broadcast_coeffs`
        produces that shape for the eager torch path; a fused Triton kernel (see
        :class:`GXPR`) indexes coefficients by row and wants this flat shape directly.
        """
        alpha_p1 = torch.abs(self.alpha_p1)  # (num_local_experts,)
        alpha_p2 = torch.abs(self.alpha_p2)
        beta = torch.abs(self.beta)
        alpha_n = beta + torch.abs(self.alpha_n)

        if self.num_local_experts == 1 or tokens_per_expert is None:
            if self.num_local_experts > 1:
                raise ValueError(
                    f"{type(self).__name__} with num_local_experts > 1 requires "
                    "`tokens_per_expert` so the per-expert coefficients can be mapped onto the "
                    "concatenated tokens."
                )
            return alpha_p1, alpha_p2, alpha_n, beta

        # Expand per-expert coefficients to per-token coefficients: shape (num_tokens,).
        if isinstance(tokens_per_expert, torch.Tensor):
            tokens_per_expert = tokens_per_expert.tolist()
        tpe_tensor = torch.tensor(tokens_per_expert, device=x.device)
        return (
            torch.repeat_interleave(alpha_p1, tpe_tensor),
            torch.repeat_interleave(alpha_p2, tpe_tensor),
            torch.repeat_interleave(alpha_n, tpe_tensor),
            torch.repeat_interleave(beta, tpe_tensor),
        )

    def _broadcast_coeffs(self, alpha_p1, alpha_p2, alpha_n, beta):
        """Unsqueeze per-token coefficients to ``(num_tokens, 1)`` for the eager torch path."""
        if self.num_local_experts == 1:
            return alpha_p1, alpha_p2, alpha_n, beta
        return (
            alpha_p1.unsqueeze(-1),
            alpha_p2.unsqueeze(-1),
            alpha_n.unsqueeze(-1),
            beta.unsqueeze(-1),
        )

    def forward(self, x, tokens_per_expert=None):
        """Return ``XPR(x)``.

        Args:
            x: activation input, ``(..., D)``.
            tokens_per_expert: per-local-expert token counts (grouped experts only); maps the
                per-expert coefficients onto the concatenated tokens.
        """
        alpha_p1, alpha_p2, alpha_n, beta = self._broadcast_coeffs(
            *self._raw_coeffs(x, tokens_per_expert)
        )
        return compiled_xpr(x, alpha_p1, alpha_p2, alpha_n, beta)


class GXPR(XPR):
    """Learnable GLU gate — the gated-linear-unit counterpart of :class:`XPR`.

    Mathematically ``gate(x) = XPR(x) / x``, simplified to avoid the ``0/0`` at ``x == 0``::

        gate(x) = |alpha_p2| * x**2 + |alpha_p1| * x + |beta|                  (x > 0)
                = (|beta| + |alpha_n|) * softsign(x) + |beta|                  (x <= 0)

    ``forward`` takes *both* GLU halves and returns ``gate(x_glu) * x_linear * [scores]``, same
    calling convention as :class:`PolyNorm`. Shares :class:`XPR`'s per-(local-)expert
    coefficients and ``tokens_per_expert`` expansion.

    Unlike PolyNorm, this gate has **no cross-feature reduction** (each output element only
    depends on the corresponding elements of ``x_glu``/``x_linear``), exactly like SwiGLU. Its
    fused path (see ``megatron.core.fusions.fused_gxpr_glu``) is therefore built the same way
    SwiGLU's own fusion is in this codebase -- ``@jit_fuser`` (torch.compile)-fused elementwise
    math plus a hand-derived analytic backward, not a hand Triton kernel -- rather than mirroring
    PolyNorm's Triton kernel. There is no tensor-parallel restriction (correct at any TP/ETP
    degree with zero collectives) and no feature-dimension cap (unlike PolyNorm's
    ``MAX_FUSED_FEATURE_DIM``, since there's no per-row register budget to bound).

    ``beta_init`` defaults to ``0.1`` here (vs. :class:`XPR`/:class:`GXPRY`'s ``0.5``): a smaller
    additive floor term makes ``gate(0)`` closer to 0, more in line with SiLU/GELU-style gates,
    while staying safely away from the ``abs()``-gradient dead zone at exactly 0 (see
    :class:`GXPRV2`, which removes ``beta`` -- and therefore this tradeoff -- entirely).
    """

    def __init__(
        self,
        num_local_experts: int = 1,
        config=None,
        alpha_p_init: float = 0.8,
        alpha_p2_init: float = 0.4,
        alpha_n_init: float = 0.8,
        beta_init: float = 0.1,
    ):
        super().__init__(
            num_local_experts=num_local_experts,
            config=config,
            alpha_p_init=alpha_p_init,
            alpha_p2_init=alpha_p2_init,
            alpha_n_init=alpha_n_init,
            beta_init=beta_init,
        )

    def forward(self, x_glu, x_linear, tokens_per_expert=None, scores=None):
        """Return ``gate(x_glu) * x_linear * [scores]``.

        Args:
            x_glu: GLU gate half, ``(..., D)`` (``D`` = local ffn feature dim).
            x_linear: GLU linear half, same shape/dtype as ``x_glu``.
            tokens_per_expert: per-local-expert token counts (grouped experts only); maps the
                per-expert coefficients onto the concatenated tokens.
            scores: optional per-token multiplier ``(..., 1)`` (MoE router probs / per-token scale).
        """
        alpha_p1, alpha_p2, alpha_n, beta = self._broadcast_coeffs(
            *self._raw_coeffs(x_glu, tokens_per_expert)
        )

        use_fused = x_glu.is_cuda and (self.config is None or getattr(self.config, "gxpr_fusion", True))
        if use_fused:
            return fused_gxpr_glu_impl(x_glu, x_linear, alpha_p1, alpha_p2, alpha_n, beta, scores)

        gate = compiled_xpr_gate(x_glu, alpha_p1, alpha_p2, alpha_n, beta)
        out = gate * x_linear
        if scores is not None:
            original_dtype = out.dtype
            out = (out * scores).to(original_dtype)
        return out


@jit_fuser
def compiled_gxpry_gate(x, y, alpha_p1, alpha_p2, alpha_n, beta):
    """GXPRY gate: the same two pieces as :func:`compiled_xpr_gate`, but the piecewise branch is
    selected by the sign of ``y`` (the GLU linear half) instead of ``x`` (the GLU gate half).

    ``alpha_p2 * x**2 + alpha_p1 * x + beta`` when ``y > 0``, and
    ``alpha_n * softsign(x) + beta`` when ``y <= 0``.
    """
    return torch.where(
        y > 0,
        alpha_p2 * x * x + alpha_p1 * x + beta,
        alpha_n * F.softsign(x) + beta,
    )


class GXPRY(XPR):
    """Learnable GLU gate — like :class:`GXPR`, but the piecewise branch is chosen by the sign
    of ``y`` (the GLU *linear* half, ``x_linear``) instead of ``x`` (the GLU *gate* half,
    ``x_glu``)::

        gate(x, y) = |alpha_p2| * x**2 + |alpha_p1| * x + |beta|                (y > 0)
                   = (|beta| + |alpha_n|) * softsign(x) + |beta|                (y <= 0)

    ``forward`` takes *both* GLU halves and returns ``gate(x_glu, x_linear) * x_linear *
    [scores]``, same calling convention as :class:`GXPR`. Shares :class:`XPR`'s
    per-(local-)expert coefficients and ``tokens_per_expert`` expansion.

    Unlike GXPR (which is algebraically ``XPR(x) / x``, so it degenerates to a well-defined
    non-gated single-input activation), GXPRY's branch condition depends on the *second* input,
    so it has no non-gated counterpart -- it is inherently a two-input (GLU) op. It also has no
    ``0/0`` concern at ``y == 0``: the output is simply ``gate(x, y) * 0 == 0`` regardless of
    which branch ``gate`` took.
    """

    def forward(self, x_glu, x_linear, tokens_per_expert=None, scores=None):
        """Return ``gate(x_glu, x_linear) * x_linear * [scores]``.

        Args:
            x_glu: GLU gate half, ``(..., D)`` (``D`` = local ffn feature dim).
            x_linear: GLU linear half, same shape/dtype as ``x_glu``.
            tokens_per_expert: per-local-expert token counts (grouped experts only); maps the
                per-expert coefficients onto the concatenated tokens.
            scores: optional per-token multiplier ``(..., 1)`` (MoE router probs / per-token scale).
        """
        alpha_p1, alpha_p2, alpha_n, beta = self._broadcast_coeffs(
            *self._raw_coeffs(x_glu, tokens_per_expert)
        )
        gate = compiled_gxpry_gate(x_glu, x_linear, alpha_p1, alpha_p2, alpha_n, beta)
        out = gate * x_linear
        if scores is not None:
            original_dtype = out.dtype
            out = (out * scores).to(original_dtype)
        return out


@jit_fuser
def compiled_gxprv2_gate(x, alpha_p1, alpha_p2, alpha_n):
    """GXPRV2 gate: :class:`GXPR` with ``beta`` removed entirely -- no additive floor term, and
    ``alpha_n`` is no longer coupled to ``beta`` (``an = |alpha_n|`` directly, not
    ``|beta| + |alpha_n|``).

    ``alpha_p2 * x**2 + alpha_p1 * x`` for ``x > 0``, and ``alpha_n * softsign(x)`` for ``x <= 0``.
    """
    return torch.where(
        x > 0,
        alpha_p2 * x * x + alpha_p1 * x,
        alpha_n * F.softsign(x),
    )


class GXPRV2(MegatronModule):
    """Learnable GLU gate -- :class:`GXPR` with ``beta`` removed entirely (not just
    initialized near zero)::

        gate(x) = |alpha_p2| * x**2 + |alpha_p1| * x           (x > 0)
                = |alpha_n| * softsign(x)                        (x <= 0)

    GXPR's ``beta`` played two roles: an additive floor term present in both branches (so
    ``gate(0) == |beta|``, meaning the gate is never fully closed), and it was coupled into the
    negative-branch coefficient (``an = |beta| + |alpha_n|``). Removing it entirely -- rather
    than just initializing it near zero -- gives ``gate(0) == 0`` by construction, matching
    standard gated activations (SiLU, GELU-based gates), and makes ``alpha_n`` a fully
    independent parameter. As with every other coefficient in this activation family, ``alpha_n``
    is used only via ``abs()``, so it still needs a nonzero init (``d|x|/dx == sign(0) == 0`` in
    PyTorch -- an exactly-zero init would receive a permanent zero gradient and never move).

    ``forward`` takes *both* GLU halves and returns ``gate(x_glu) * x_linear * [scores]``, same
    calling convention as :class:`GXPR`. Same per-(local-)expert coefficient /
    ``tokens_per_expert`` handling as the rest of the family. No fused kernel yet; always runs
    the (torch.compile-fused) eager implementation.
    """

    def __init__(
        self,
        num_local_experts: int = 1,
        config=None,
        alpha_p_init: float = 0.8,
        alpha_p2_init: float = 0.4,
        alpha_n_init: float = 0.8,
    ):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.alpha_p1 = nn.Parameter(torch.full((num_local_experts,), alpha_p_init))
        self.alpha_p2 = nn.Parameter(torch.full((num_local_experts,), alpha_p2_init))
        self.alpha_n = nn.Parameter(torch.full((num_local_experts,), alpha_n_init))

    def _coeffs(self, x, tokens_per_expert):
        """Return ``(alpha_p1, alpha_p2, alpha_n)``, positive and expanded per-token."""
        alpha_p1 = torch.abs(self.alpha_p1)  # (num_local_experts,)
        alpha_p2 = torch.abs(self.alpha_p2)
        alpha_n = torch.abs(self.alpha_n)

        if self.num_local_experts == 1 or tokens_per_expert is None:
            if self.num_local_experts > 1:
                raise ValueError(
                    f"{type(self).__name__} with num_local_experts > 1 requires "
                    "`tokens_per_expert` so the per-expert coefficients can be mapped onto the "
                    "concatenated tokens."
                )
            return alpha_p1, alpha_p2, alpha_n

        # Expand per-expert coefficients to per-token coefficients: shape (num_tokens, 1).
        if isinstance(tokens_per_expert, torch.Tensor):
            tokens_per_expert = tokens_per_expert.tolist()
        tpe_tensor = torch.tensor(tokens_per_expert, device=x.device)

        def expand(a):
            return torch.repeat_interleave(a, tpe_tensor).unsqueeze(-1)

        return expand(alpha_p1), expand(alpha_p2), expand(alpha_n)

    def forward(self, x_glu, x_linear, tokens_per_expert=None, scores=None):
        """Return ``gate(x_glu) * x_linear * [scores]``.

        Args:
            x_glu: GLU gate half, ``(..., D)`` (``D`` = local ffn feature dim).
            x_linear: GLU linear half, same shape/dtype as ``x_glu``.
            tokens_per_expert: per-local-expert token counts (grouped experts only); maps the
                per-expert coefficients onto the concatenated tokens.
            scores: optional per-token multiplier ``(..., 1)`` (MoE router probs / per-token scale).
        """
        alpha_p1, alpha_p2, alpha_n = self._coeffs(x_glu, tokens_per_expert)
        gate = compiled_gxprv2_gate(x_glu, alpha_p1, alpha_p2, alpha_n)
        out = gate * x_linear
        if scores is not None:
            original_dtype = out.dtype
            out = (out * scores).to(original_dtype)
        return out


@jit_fuser
def compiled_xr2(x, alpha_p1, alpha_n, beta):
    """Core XR2 activation — :func:`compiled_xpr` without the ``x**3`` term.

    ``alpha_p1 * x**2 + beta * x`` for ``x > 0``, and
    ``alpha_n * x * softsign(x) + beta * x`` for ``x <= 0``.
    """
    return torch.where(
        x > 0,
        alpha_p1 * x * x + beta * x,
        alpha_n * x * F.softsign(x) + beta * x,
    )


@jit_fuser
def compiled_xr2_gate(x, alpha_p1, alpha_n, beta):
    """XR2 GLU gate: ``compiled_xr2(x, ...) / x``, simplified algebraically to avoid the ``0/0``
    at ``x == 0``.

    ``alpha_p1 * x + beta`` for ``x > 0``, and ``alpha_n * softsign(x) + beta`` for ``x <= 0``.
    """
    return torch.where(x > 0, alpha_p1 * x + beta, alpha_n * F.softsign(x) + beta)


class XR2(MegatronModule):
    """Learnable elementwise activation (not a gated unit) — :class:`XPR` without the ``x**3``
    term::

        XR2(x) = |alpha_p1| * x**2 + |beta| * x                                 (x > 0)
               = (|beta| + |alpha_n|) * x * softsign(x) + |beta| * x            (x <= 0)

    Same per-(local-)expert coefficient / ``tokens_per_expert`` handling as :class:`XPR`. No
    fused kernel yet; always runs the (torch.compile-fused) eager implementation.
    """

    def __init__(
        self,
        num_local_experts: int = 1,
        config=None,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta_init: float = 0.5,
    ):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.alpha_p1 = nn.Parameter(torch.full((num_local_experts,), alpha_p_init))
        self.alpha_n = nn.Parameter(torch.full((num_local_experts,), alpha_n_init - beta_init))
        self.beta = nn.Parameter(torch.full((num_local_experts,), beta_init))

    def _coeffs(self, x, tokens_per_expert):
        """Return ``(alpha_p1, alpha_n, beta)``, positive and expanded per-token."""
        alpha_p1 = torch.abs(self.alpha_p1)  # (num_local_experts,)
        beta = torch.abs(self.beta)
        alpha_n = beta + torch.abs(self.alpha_n)

        if self.num_local_experts == 1 or tokens_per_expert is None:
            if self.num_local_experts > 1:
                raise ValueError(
                    f"{type(self).__name__} with num_local_experts > 1 requires "
                    "`tokens_per_expert` so the per-expert coefficients can be mapped onto the "
                    "concatenated tokens."
                )
            return alpha_p1, alpha_n, beta

        # Expand per-expert coefficients to per-token coefficients: shape (num_tokens, 1).
        if isinstance(tokens_per_expert, torch.Tensor):
            tokens_per_expert = tokens_per_expert.tolist()
        tpe_tensor = torch.tensor(tokens_per_expert, device=x.device)

        def expand(a):
            return torch.repeat_interleave(a, tpe_tensor).unsqueeze(-1)

        return expand(alpha_p1), expand(alpha_n), expand(beta)

    def forward(self, x, tokens_per_expert=None):
        """Return ``XR2(x)``.

        Args:
            x: activation input, ``(..., D)``.
            tokens_per_expert: per-local-expert token counts (grouped experts only); maps the
                per-expert coefficients onto the concatenated tokens.
        """
        alpha_p1, alpha_n, beta = self._coeffs(x, tokens_per_expert)
        return compiled_xr2(x, alpha_p1, alpha_n, beta)


class GXR2(XR2):
    """Learnable GLU gate — the gated-linear-unit counterpart of :class:`XR2`.

    Mathematically ``gate(x) = XR2(x) / x``, simplified to avoid the ``0/0`` at ``x == 0``::

        gate(x) = |alpha_p1| * x + |beta|                              (x > 0)
                = (|beta| + |alpha_n|) * softsign(x) + |beta|           (x <= 0)

    ``forward`` takes *both* GLU halves and returns ``gate(x_glu) * x_linear * [scores]``, same
    calling convention as :class:`GXPR`. Shares :class:`XR2`'s per-(local-)expert coefficients
    and ``tokens_per_expert`` expansion.
    """

    def forward(self, x_glu, x_linear, tokens_per_expert=None, scores=None):
        """Return ``gate(x_glu) * x_linear * [scores]``.

        Args:
            x_glu: GLU gate half, ``(..., D)`` (``D`` = local ffn feature dim).
            x_linear: GLU linear half, same shape/dtype as ``x_glu``.
            tokens_per_expert: per-local-expert token counts (grouped experts only); maps the
                per-expert coefficients onto the concatenated tokens.
            scores: optional per-token multiplier ``(..., 1)`` (MoE router probs / per-token scale).
        """
        alpha_p1, alpha_n, beta = self._coeffs(x_glu, tokens_per_expert)
        gate = compiled_xr2_gate(x_glu, alpha_p1, alpha_n, beta)
        out = gate * x_linear
        if scores is not None:
            original_dtype = out.dtype
            out = (out * scores).to(original_dtype)
        return out


class XR2GLU(XR2):
    """Learnable GLU gate where the gate function is :class:`XR2` itself -- not divided by
    ``x`` (unlike :class:`GXR2`) -- the direct analogue of plugging XR2 in as a GLU's gate the
    way SiLU is for SwiGLU::

        XR2GLU(x, y) = XR2(x) * y
                     = (|alpha_p1|*x**2 + |beta|*x) * y                          (x > 0)
                     = ((|beta|+|alpha_n|)*x*softsign(x) + |beta|*x) * y         (x <= 0)

    Unlike GXR2 (algebraically ``XR2(x) / x``, built specifically to avoid a ``0/0`` at
    ``x == 0``), there is no division here and therefore nothing to simplify -- this directly
    reuses :func:`compiled_xr2` (XR2's own activation) as the gate. Shares :class:`XR2`'s
    per-(local-)expert coefficients and ``tokens_per_expert`` expansion. No fused kernel yet;
    always runs the (torch.compile-fused) eager implementation.
    """

    def forward(self, x_glu, x_linear, tokens_per_expert=None, scores=None):
        """Return ``XR2(x_glu) * x_linear * [scores]``.

        Args:
            x_glu: GLU gate half, ``(..., D)`` (``D`` = local ffn feature dim).
            x_linear: GLU linear half, same shape/dtype as ``x_glu``.
            tokens_per_expert: per-local-expert token counts (grouped experts only); maps the
                per-expert coefficients onto the concatenated tokens.
            scores: optional per-token multiplier ``(..., 1)`` (MoE router probs / per-token scale).
        """
        alpha_p1, alpha_n, beta = self._coeffs(x_glu, tokens_per_expert)
        gate = compiled_xr2(x_glu, alpha_p1, alpha_n, beta)
        out = gate * x_linear
        if scores is not None:
            original_dtype = out.dtype
            out = (out * scores).to(original_dtype)
        return out


@jit_fuser
def compiled_xsssglu(x, y, alpha):
    """XSSSGLU gate: ``(|alpha| * softsign(x) + 0.5) * x * y``.

    Softsign already smoothly interpolates between the negative- and positive-``x`` regimes
    (unlike the rest of the XPR/XR2 family, which needs an explicit ``torch.where`` piecewise
    split), so this is a single formula with no branch.
    """
    return (alpha * F.softsign(x) + 0.5) * x * y


class XSSSGLU(MegatronModule):
    """Learnable GLU gate built from a softsign-scaled linear term -- no polynomial term and no
    piecewise positive/negative branching (softsign already handles both signs smoothly)::

        gate(x) = |alpha| * softsign(x) + 0.5
        XSSSGLU(x, y) = gate(x) * x * y

    At ``alpha == 0`` this degenerates to a fixed ``0.5 * x * y`` (a plain linear GLU with a
    constant 0.5 gate); growing ``|alpha|`` pushes ``gate(x)`` toward a soft step between
    ``0.5 - |alpha|`` and ``0.5 + |alpha|`` -- a differentiable, learnable-slope relative of a
    ReLU-ish gate, structurally analogous to SiLU's ``sigmoid(x) * x`` but built from
    ``softsign`` instead of ``sigmoid``. Each (local) expert gets its own coefficient. No fused
    kernel yet; always runs the (torch.compile-fused) eager implementation.
    """

    def __init__(self, num_local_experts: int = 1, config=None, alpha_init: float = 0.8):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.alpha = nn.Parameter(torch.full((num_local_experts,), alpha_init - 0.5))

    def _coeffs(self, x, tokens_per_expert):
        """Return ``alpha``, positive and expanded per-token."""
        alpha = 0.5 + torch.abs(self.alpha)  # (num_local_experts,)

        if self.num_local_experts == 1 or tokens_per_expert is None:
            if self.num_local_experts > 1:
                raise ValueError(
                    f"{type(self).__name__} with num_local_experts > 1 requires "
                    "`tokens_per_expert` so the per-expert coefficients can be mapped onto the "
                    "concatenated tokens."
                )
            return alpha

        # Expand per-expert coefficients to per-token coefficients: shape (num_tokens, 1).
        if isinstance(tokens_per_expert, torch.Tensor):
            tokens_per_expert = tokens_per_expert.tolist()
        tpe_tensor = torch.tensor(tokens_per_expert, device=x.device)
        return torch.repeat_interleave(alpha, tpe_tensor).unsqueeze(-1)

    def forward(self, x_glu, x_linear, tokens_per_expert=None, scores=None):
        """Return ``(|alpha| * softsign(x_glu) + 0.5) * x_glu * x_linear * [scores]``.

        Args:
            x_glu: GLU gate half, ``(..., D)`` (``D`` = local ffn feature dim).
            x_linear: GLU linear half, same shape/dtype as ``x_glu``.
            tokens_per_expert: per-local-expert token counts (grouped experts only); maps the
                per-expert coefficients onto the concatenated tokens.
            scores: optional per-token multiplier ``(..., 1)`` (MoE router probs / per-token scale).
        """
        alpha = self._coeffs(x_glu, tokens_per_expert)
        out = compiled_xsssglu(x_glu, x_linear, alpha)
        if scores is not None:
            original_dtype = out.dtype
            out = (out * scores).to(original_dtype)
        return out


@jit_fuser
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU activation"""
    return torch.pow(F.relu(x), 2)


def rlglu_act(x: torch.Tensor) -> torch.Tensor:
    """RLGLU gate: ``f(x) = max(x, 0) - 0.5 * ln(1 + |x|)``.

    Used as the gate of a gated linear unit (``rlglu_act(x_glu) * x_linear``) via the generic
    (non-fused) GLU path — dispatched by ``activation_func == rlglu_act`` with
    ``gated_linear_unit=True`` and ``bias_activation_fusion=False``. No fused kernel yet, so it
    runs eager. Its derivative decays like ``1/(1+|x|)`` (relative-gradient-error condition
    number -> 1), at the cost of an unbounded (logarithmic) negative tail.

    ``log(1 + |x|)`` is used rather than ``log1p(|x|)`` (equal for |x| >= 0) to match the fused
    kernels, whose ``log1p`` tripped an inductor cubin failure inside the weighted-backward
    reduction; see fused_bias_rlglu.py.
    """
    return torch.relu(x) - 0.5 * torch.log(1 + x.abs())


@jit_fuser
def situ_act(x: torch.Tensor) -> torch.Tensor:
    """SiTU gate: ``f(x) = sigmoid(x) * tanh(x)``.

    Used as the gate of a gated linear unit (``situ_act(x_glu) * x_linear``, i.e.
    ``sigmoid(x_glu) * tanh(x_glu) * x_linear``) via the generic (non-fused) GLU path —
    dispatched by ``activation_func == situ_act`` with ``gated_linear_unit=True`` and
    ``bias_activation_fusion=False``. Elementwise and non-learnable, like SwiGLU's SiLU gate, so
    it needs no dedicated module or fused kernel (runs eager). Unlike the strictly-non-negative
    SiLU/GELU gates, the ``tanh`` factor makes the gate change sign with ``x``: it is ~0 near the
    origin, rises to a bounded positive lobe for ``x > 0`` (``-> +1`` as ``x -> +inf``) and dips
    to a bounded negative lobe for ``x < 0`` (``-> 0`` as ``x -> -inf``, with a minimum in
    between), giving a smooth saturating gate on both sides.
    """
    return torch.sigmoid(x) * torch.tanh(x)


@jit_fuser
def lnglu_act(x: torch.Tensor) -> torch.Tensor:
    """LNGLU gate: ``f(x) = sign(x) * ln(1 + |x|)`` (``== x * ln(|x| + 1) / |x|``).

    Used as the gate of a gated linear unit (``lnglu_act(x_glu) * x_linear``) via the generic
    (non-fused) GLU path -- dispatched by ``activation_func == lnglu_act`` with
    ``gated_linear_unit=True`` and ``bias_activation_fusion=False``. Like SiTU the linear half
    is used linearly, so this is gate-form and needs no dedicated branch. Written with
    ``torch.sign`` so ``x == 0`` maps to 0 (the literal ``x/|x|`` form is ``0/0`` there); the gate
    is a sign-preserving, logarithmically-growing (unbounded but slow) odd function whose
    derivative ``1/(1 + |x|)`` is bounded in ``(0, 1]``, so it needs no epsilon guard.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@jit_fuser
def lnglu_v2_act(x: torch.Tensor) -> torch.Tensor:
    """LNGLU-v2 gate: :func:`lnglu_act` divided by 2 -- ``f(x) = 0.5 * sign(x) * ln(1 + |x|)``.

    Identical to LNGLU in every way (gate-form, generic GLU path, non-learnable, derivative
    ``0.5 / (1 + |x|)`` bounded in ``(0, 0.5]``) except the gate magnitude is halved.
    """
    return 0.5 * torch.sign(x) * torch.log(1 + torch.abs(x))


@jit_fuser
def lnglu_v3_act(x: torch.Tensor) -> torch.Tensor:
    """LNGLU-v3 gate: logarithmic on the positive side, SiLU on the negative side::

        f(x) = ln(1 + |x|)     (x > 0)
             = silu(x)         (x <= 0)

    Used as the gate of a gated linear unit (``f(x_glu) * x_linear``) via the generic GLU path.
    Both branches are 0 at ``x == 0`` (continuous there). Non-learnable, no fused kernel.
    """
    return torch.where(x > 0, torch.log(1 + torch.abs(x)), F.silu(x))


@jit_fuser
def lnglu_v4_act(x: torch.Tensor) -> torch.Tensor:
    """LNGLU-v4 gate: :func:`lnglu_v3_act` with the positive branch replaced by
    ``0.5x / (1 + |0.5x|)`` (softsign of ``0.5x`` on the positive side)::

        f(x) = 0.5x / (1 + |0.5x|)   (x > 0)
             = silu(x)               (x <= 0)

    Used as the gate of a gated linear unit (``f(x_glu) * x_linear``) via the generic GLU path.
    Both branches are 0 at ``x == 0`` (continuous there); the positive branch is bounded (-> 1)
    unlike v3's logarithm. Non-learnable, no fused kernel.
    """
    return torch.where(x > 0, 0.5 * x / (1 + torch.abs(0.5 * x)), F.silu(x))


@jit_fuser
def compiled_situ_v2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """SiTU-v2 GLU: ``softsign(x) * (0.5 + 0.5*softsign(x)) * y``.

    The gate is ``softsign(x)`` multiplied by ``softsign(x)`` rescaled from ``(-1, 1)`` to
    ``(0, 1)`` (``0.5 + 0.5*softsign(x)``); the linear half ``y`` is used linearly. Non-learnable
    and elementwise; no fused kernel, runs eager. ``x`` is the GLU gate half (``x_glu``), ``y`` the
    linear half (``x_linear``). (Kept as a two-input function -- rather than a gate on the generic
    path -- so its dispatch branch is unchanged.)
    """
    s = F.softsign(x)
    return s * (0.5 + 0.5 * s) * y


@jit_fuser
def situ_v3_act(x: torch.Tensor) -> torch.Tensor:
    """SiTU-v3 gate: :func:`situ_act` with the positive side replaced by ``0.5x / (1 + |0.5x|)``::

        f(x) = 0.5x / (1 + |0.5x|)    (x > 0)     (softsign of 0.5x)
             = sigmoid(x) * tanh(x)   (x <= 0)    (SiTU gate)

    Used as the gate of a gated linear unit (``f(x_glu) * x_linear``) via the generic GLU path.
    Both branches are 0 at ``x == 0`` (continuous there). Non-learnable, no fused kernel.
    """
    return torch.where(x > 0, 0.5 * x / (1 + torch.abs(0.5 * x)), torch.sigmoid(x) * torch.tanh(x))


@jit_fuser
def situ_v4_act(x: torch.Tensor) -> torch.Tensor:
    """SiTU-v4 gate: ``f(x) = sign(x - 1) * ln(|x - 1| + 1) + ln(2)``.

    The antiderivative of ``1 / (1 + |x - 1|)`` (its derivative is exactly that, bounded in
    ``(0, 1]``), plus the ``+ln(2)`` bias so the gate passes through the origin: ``f(0) == 0``.
    Used as the gate of a gated linear unit (``f(x_glu) * x_linear``) via the generic GLU path.
    Non-learnable, no fused kernel.
    """
    xm1 = x - 1
    return torch.sign(xm1) * torch.log(torch.abs(xm1) + 1) + 0.6931471805599453


@jit_fuser
def situ_v5_act(x: torch.Tensor) -> torch.Tensor:
    """SiTU-v5 gate: ``f(x) = softsign(x - 1) + 0.5``.

    Softsign shifted right by 1 and up by 0.5, so ``f(0) == 0`` (passes through the origin) and
    ``f -> 1.5`` as ``x -> +inf``, ``f -> -0.5`` as ``x -> -inf``. Used as the gate of a gated
    linear unit (``f(x_glu) * x_linear``) via the generic GLU path. Non-learnable, no fused kernel.
    """
    return F.softsign(x - 1) + 0.5


@jit_fuser
def downscale_glu_transform(x: torch.Tensor) -> torch.Tensor:
    """Square-root down-scaling of the GLU projections (``--downscale-glu``): ``x / sqrt(|x|)``.

    Sign-preserving (``== sign(x) * sqrt(|x|)`` for ``|x|`` >> the floor). Applied elementwise to
    the whole fc1 output (i.e. to both the gate and up projections) before the activation, so it
    composes with whatever GLU gate follows. The ``1e-6`` floor inside the sqrt avoids the ``0/0``
    at ``x == 0`` (giving 0 there) and keeps the backward gradient finite instead of NaN.
    """
    return x / torch.sqrt(torch.abs(x) + 1e-6)


@jit_fuser
def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """Quick GELU activation"""
    return x * torch.sigmoid(1.702 * x)


@jit_fuser
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """Fast GELU activation"""
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
