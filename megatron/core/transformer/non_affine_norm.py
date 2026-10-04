# Copyright (c) 2026, Swiss AI Initiative.
"""Parameter-free pre-normalization, including an affine-free CUDA backward.

The Triton path performs each normalization in a single kernel, without allocating
unit gains or calculating gain gradients. CPU uses the same formula in PyTorch.
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:

    @triton.jit
    def _norm_forward(X, Y, Mean, Rstd, H: tl.constexpr, EPS: tl.constexpr,
                      CENTER: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        x = tl.load(X + row * H + cols, cols < H, other=0).to(tl.float32)
        mean = tl.sum(x, 0) / H if CENTER else 0.0
        centered = tl.where(cols < H, x - mean, 0.0)
        variance = tl.sum(centered * centered, 0) / H
        rstd = tl.rsqrt(variance + EPS)
        tl.store(Y + row * H + cols, centered * rstd, cols < H)
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)

    @triton.jit
    def _norm_backward(X, DY, DX, Mean, Rstd, H: tl.constexpr,
                       CENTER: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        x = tl.load(X + row * H + cols, cols < H, other=0).to(tl.float32)
        dy = tl.load(DY + row * H + cols, cols < H, other=0).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        y = tl.where(cols < H, (x - mean) * rstd, 0.0)
        projection = tl.sum(y * dy, 0) / H
        mean_dy = tl.sum(dy, 0) / H if CENTER else 0.0
        dx = (dy - mean_dy - y * projection) * rstd
        tl.store(DX + row * H + cols, dx, cols < H)


class _NonAffineNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps, center):
        x = x.contiguous()
        h = x.shape[-1]
        rows = x.numel() // h
        y = torch.empty_like(x)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        block = triton.next_power_of_2(h)
        warps = min(8, max(1, block // 256))
        if rows:
            _norm_forward[(rows,)](x, y, mean, rstd, h, eps, center, block,
                                   num_warps=warps)
        ctx.save_for_backward(x, mean, rstd)
        ctx.center, ctx.block, ctx.warps = center, block, warps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, mean, rstd = ctx.saved_tensors
        dx = torch.empty_like(x)
        rows = mean.numel()
        if rows:
            _norm_backward[(rows,)](x, dy.contiguous(), dx, mean, rstd,
                                    x.shape[-1], ctx.center, ctx.block,
                                    num_warps=ctx.warps)
        return dx, None, None


class NonAffineNorm(torch.nn.Module):
    """LayerNorm/RMSNorm with no affine parameters or buffers.

    Accumulate statistics in FP32 for BF16/FP16 input. Sequence parallelism is
    local: each rank normalizes its complete hidden dimension before all-gather.
    """

    def __init__(self, config, hidden_size, eps=None):
        super().__init__()
        if config.normalization not in ("LayerNorm", "RMSNorm"):
            raise ValueError("Non-affine pre-norm supports LayerNorm and RMSNorm only")
        self.hidden_size = hidden_size
        self.eps = config.layernorm_epsilon if eps is None else eps
        self.center = config.normalization == "LayerNorm"

    def forward(self, x):
        if x.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {x.shape[-1]}")
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
            if triton is None:
                raise RuntimeError("Non-affine CUDA pre-norm requires Triton")
            if self.hidden_size > 32768:
                raise ValueError("Non-affine CUDA pre-norm supports hidden sizes up to 32768")
            return _NonAffineNorm.apply(x, self.eps, self.center)
        value = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        if self.center:
            output = F.layer_norm(value, (self.hidden_size,), eps=self.eps)
        else:
            output = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps)
        return output.to(x.dtype)


def build_kda_output_norm(config, hidden_size, existing_norm, fused_norm_cls=None, device=None):
    """Remove the redundant output gain without splitting FLA's fused gate kernel."""
    if fused_norm_cls is not None:
        norm = fused_norm_cls(
            hidden_size,
            elementwise_affine=not config.non_affine_kda_output_norm,
            activation="sigmoid",
            eps=config.layernorm_epsilon,
            device=device,
            dtype=config.params_dtype,
        )
        if norm.weight is not None:
            norm.weight.sequence_parallel = config.sequence_parallel
        return norm
    if config.non_affine_kda_output_norm:
        return NonAffineNorm(config, hidden_size)
    return existing_norm


def build_pre_norm(builder, config, hidden_size, eps=None, **kwargs):
    """Replace only an explicitly identified pre-norm; leave IdentityOp intact."""
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.spec_utils import ModuleSpec, get_module

    module = get_module(builder) if isinstance(builder, ModuleSpec) else builder
    if config.non_affine_pre_norm and module is not IdentityOp:
        return NonAffineNorm(config, hidden_size, eps)
    return builder(config=config, hidden_size=hidden_size,
                   eps=config.layernorm_epsilon if eps is None else eps, **kwargs)


def build_pre_norm_linear(spec, *args, config, **kwargs):
    """Select an affine-free norm + TE linear at pre-norm projection sites only."""
    from dataclasses import replace

    from megatron.core.transformer.spec_utils import ModuleSpec, build_module, get_module

    if config.non_affine_pre_norm:
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
            TENonAffineNormColumnParallelLinear,
        )

        module = get_module(spec) if isinstance(spec, ModuleSpec) else spec
        if isinstance(module, type) and issubclass(module, TELayerNormColumnParallelLinear):
            spec = (replace(spec, module=TENonAffineNormColumnParallelLinear)
                    if isinstance(spec, ModuleSpec) else TENonAffineNormColumnParallelLinear)
    if isinstance(spec, ModuleSpec):
        return build_module(spec, *args, config=config, **kwargs)
    return spec(*args, config=config, **kwargs)
