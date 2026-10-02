# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Protocol, cast

import torch
import torch.nn.functional as F

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.fusions.fused_bias_geglu import (
    bias_geglu_impl,
    quick_gelu,
    weighted_bias_quick_geglu_impl,
)
from megatron.core.activations import (
    GXPR,
    GXPRY,
    GXPRV2,
    GXR2,
    PolyNorm,
    PolyNormAct,
    XPR,
    XR2,
    XR2GLU,
    XSSGLU,
    rlglu_act,
    sssglu_act,
)
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_rlglu import (
    bias_rlglu_impl,
    weighted_bias_rlglu_impl,
)
from megatron.core.fusions.fused_bias_ssglu import (
    bias_ssglu_impl,
    sslu,
    weighted_bias_ssglu_impl,
)
from megatron.core.fusions.fused_bias_sssglu import (
    bias_sssglu_impl,
    weighted_bias_sssglu_impl,
)
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.expert_checkpoint_utils import (
    EXPERT_CKPT_HAS_TE_EXTRA_STATE_KEY,
    apply_swiglu_sharded_factory,
    is_legacy_offloading_checkpoint,
    make_legacy_offloading_load_factory,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import (
    get_tensor_model_parallel_group_if_none,
    nvtx_range_pop,
    nvtx_range_push,
)

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


logger = logging.getLogger(__name__)


class LinearFc1Interface(Protocol):
    """Interface for linear_fc1 module in MLP."""

    def forward(self, hidden_states: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc1 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc1 module."""
        ...


class LinearFc1Builder(Protocol):
    """Protocol describing how to build a linear_fc1 layer in MLP."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        tp_group: torch.distributed.ProcessGroup | None,
        stride: int = 1,
    ) -> LinearFc1Interface:
        """Builds a linear_fc1 layer for MLP."""
        ...


class TEActivationFunctionInterface(Protocol):
    """Interface for activation_function module in MLP."""

    def forward(self, input_: torch.Tensor, /) -> torch.Tensor:
        """Forward method for activation_function module."""
        ...


class TEActivationFunctionBuilder(Protocol):
    """Protocol for activation_function module in MLP."""

    def __call__(self, *, config: TransformerConfig) -> TEActivationFunctionInterface:
        """Builds an activation function module for MLP."""
        ...


class LinearFc2Interface(Protocol):
    """Interface for linear_fc2 module in MLP."""

    def forward(self, hidden_states: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc2 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc2 module."""
        ...


class LinearFc2Builder(Protocol):
    """Protocol describing how to build a linear_fc2 layer in MLP."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> LinearFc2Interface:
        """Builds a linear_fc2 layer for MLP."""
        ...


@dataclass
class MLPSubmodules:
    """
    The dataclass for ModuleSpecs of MLP submodules
    including  linear fc1, activation function, linear fc2.
    """

    linear_fc1: LinearFc1Builder

    linear_fc2: LinearFc2Builder

    activation_func: TEActivationFunctionBuilder | None = None
    """
    Builder for an activation function module; only used if config.use_te_activation_func is True.
    """


class MLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.is_expert = is_expert

        self.input_size = input_size if input_size != None else self.config.hidden_size

        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        if ffn_hidden_size is None:
            if is_expert:
                raise ValueError("MoE MLP requires `ffn_hidden_size`, but it was not provided.")
            warnings.warn(
                "MLP requires ffn_hidden_size, but it was not provided. Using \
                    config.ffn_hidden_size by default.",
                DeprecationWarning,
                stacklevel=2,
            )
            ffn_hidden_size = not_none(self.config.ffn_hidden_size)

        # If this is a gated linear unit we double the output width
        # see https://arxiv.org/pdf/2002.05202.pdf
        # For GLU/SwiGLU, use stride=2 because each TP rank stores interleaved [gate, up] portions.
        # This is critical for correct weight resharding across different TP sizes.
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2
            fc1_stride = 2
            if self.config.use_kitchen:
                # Kitchen Linear doesn't support stride != 1.
                # Weight resharding across TP sizes will have aforementioned problems.
                fc1_stride = 1
        else:
            fc1_stride = 1

        # Use moe_latent_size only for routed experts. 'is_expert' is false for
        # shared_experts.
        use_latent_size = (self.config.moe_latent_size is not None) and is_expert

        self.linear_fc1 = submodules.linear_fc1(
            self.input_size if not use_latent_size else not_none(self.config.moe_latent_size),
            ffn_hidden_size,
            config=self.config,
            init_method=not_none(self.config.init_method),
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc1",
            tp_group=tp_group,
            stride=fc1_stride,
        )

        if self.config.use_te_activation_func and not (submodules.activation_func is None):
            self.activation_func = apply_module(submodules.activation_func(config=self.config))
        else:
            self.activation_func = self.config.activation_func

        # PolyNorm GLU replaces the GLU gate with a learnable module. A dense MLP (and each
        # SequentialMLP / shared expert, which are themselves MLPs) owns a single coefficient
        # pair, so num_local_experts=1 here. tp_group is the group over which this MLP's ffn is
        # sharded, so the norm can reduce feature statistics across it when TP > 1.
        if self.config.pnglu:
            self.polynorm_glu = PolyNorm(
                num_local_experts=1, config=self.config, tp_group=self.tp_group
            )
        if self.config.pn3glu:
            self.polynorm_glu = PolyNorm(
                num_local_experts=1, config=self.config, tp_group=self.tp_group, num_terms=3
            )

        # XPR/GXPR/XR2/GXR2/PolyNormAct: like PolyNorm, a dense MLP (and each SequentialMLP /
        # shared expert, which are themselves MLPs) owns a single coefficient set, so
        # num_local_experts=1 here.
        if self.config.xpr:
            self.xpr_act = XPR(num_local_experts=1, config=self.config)
        if self.config.gxpr:
            self.gxpr_glu = GXPR(num_local_experts=1, config=self.config)
        if self.config.gxpry:
            self.gxpry_glu = GXPRY(num_local_experts=1, config=self.config)
        if self.config.gxprv2:
            self.gxprv2_glu = GXPRV2(num_local_experts=1, config=self.config)
        if self.config.xr2:
            self.xr2_act = XR2(num_local_experts=1, config=self.config)
        if self.config.gxr2:
            self.gxr2_glu = GXR2(num_local_experts=1, config=self.config)
        if self.config.xr2glu:
            self.xr2glu = XR2GLU(num_local_experts=1, config=self.config)
        if self.config.xssglu:
            self.xssglu_glu = XSSGLU(num_local_experts=1, config=self.config)
        if self.config.polynorm:
            self.polynorm_act = PolyNormAct(
                num_local_experts=1, config=self.config, tp_group=self.tp_group
            )

        self.linear_fc2 = submodules.linear_fc2(
            not_none(self.config.ffn_hidden_size),
            not_none(
                self.config.hidden_size if not use_latent_size else self.config.moe_latent_size
            ),
            config=self.config,
            init_method=not_none(self.config.output_layer_init_method),
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc2",
            tp_group=tp_group,
        )

    def forward(
        self, hidden_states: torch.Tensor, per_token_scale: torch.Tensor | None = None, **kwargs
    ):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, bias_parallel = apply_module(self.linear_fc1)(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")

        nvtx_range_push(suffix="activation")
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == sslu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_ssglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == rlglu_act and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_rlglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == sssglu_act and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_sssglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu, ssglu, rlglu, sssglu and quick_gelu with "
                        "per_token_scale in MLP."
                    )
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                elif self.activation_func == sslu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_ssglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                elif self.activation_func == rlglu_act and self.config.gated_linear_unit:
                    intermediate_parallel = bias_rlglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                elif self.activation_func == sssglu_act and self.config.gated_linear_unit:
                    intermediate_parallel = bias_sssglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                else:
                    raise ValueError("Only support fusion of gelu, swiglu, ssglu, rlglu and sssglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            # When PolyNorm fuses the per_token_scale multiply into its kernel we skip the
            # eager post-multiply below (the fc2-bias scaling further down still uses per_token_scale).
            scale_fused = False
            if self.config.gated_linear_unit and (self.config.pnglu or self.config.pn3glu):
                x_glu, x_linear = torch.chunk(intermediate_parallel, 2, dim=-1)
                if (val := self.config.activation_func_clamp_value) is not None:
                    x_glu = x_glu.clamp(min=None, max=val)
                    x_linear = x_linear.clamp(min=-val, max=val)
                if self.config.glu_linear_offset != 0.0:
                    x_linear = x_linear + self.config.glu_linear_offset
                scores = per_token_scale.unsqueeze(-1) if per_token_scale is not None else None
                intermediate_parallel = self.polynorm_glu(x_glu, x_linear, scores=scores)
                scale_fused = per_token_scale is not None
            elif self.config.gated_linear_unit and self.config.gxpr:
                x_glu, x_linear = torch.chunk(intermediate_parallel, 2, dim=-1)
                if (val := self.config.activation_func_clamp_value) is not None:
                    x_glu = x_glu.clamp(min=None, max=val)
                    x_linear = x_linear.clamp(min=-val, max=val)
                if self.config.glu_linear_offset != 0.0:
                    x_linear = x_linear + self.config.glu_linear_offset
                scores = per_token_scale.unsqueeze(-1) if per_token_scale is not None else None
                intermediate_parallel = self.gxpr_glu(x_glu, x_linear, scores=scores)
                scale_fused = per_token_scale is not None
            elif self.config.gated_linear_unit and self.config.gxpry:
                x_glu, x_linear = torch.chunk(intermediate_parallel, 2, dim=-1)
                if (val := self.config.activation_func_clamp_value) is not None:
                    x_glu = x_glu.clamp(min=None, max=val)
                    x_linear = x_linear.clamp(min=-val, max=val)
                if self.config.glu_linear_offset != 0.0:
                    x_linear = x_linear + self.config.glu_linear_offset
                scores = per_token_scale.unsqueeze(-1) if per_token_scale is not None else None
                intermediate_parallel = self.gxpry_glu(x_glu, x_linear, scores=scores)
                scale_fused = per_token_scale is not None
            elif self.config.gated_linear_unit and self.config.gxprv2:
                x_glu, x_linear = torch.chunk(intermediate_parallel, 2, dim=-1)
                if (val := self.config.activation_func_clamp_value) is not None:
                    x_glu = x_glu.clamp(min=None, max=val)
                    x_linear = x_linear.clamp(min=-val, max=val)
                if self.config.glu_linear_offset != 0.0:
                    x_linear = x_linear + self.config.glu_linear_offset
                scores = per_token_scale.unsqueeze(-1) if per_token_scale is not None else None
                intermediate_parallel = self.gxprv2_glu(x_glu, x_linear, scores=scores)
                scale_fused = per_token_scale is not None
            elif self.config.gated_linear_unit and self.config.gxr2:
                x_glu, x_linear = torch.chunk(intermediate_parallel, 2, dim=-1)
                if (val := self.config.activation_func_clamp_value) is not None:
                    x_glu = x_glu.clamp(min=None, max=val)
                    x_linear = x_linear.clamp(min=-val, max=val)
                if self.config.glu_linear_offset != 0.0:
                    x_linear = x_linear + self.config.glu_linear_offset
                scores = per_token_scale.unsqueeze(-1) if per_token_scale is not None else None
                intermediate_parallel = self.gxr2_glu(x_glu, x_linear, scores=scores)
                scale_fused = per_token_scale is not None
            elif self.config.gated_linear_unit and self.config.xr2glu:
                x_glu, x_linear = torch.chunk(intermediate_parallel, 2, dim=-1)
                if (val := self.config.activation_func_clamp_value) is not None:
                    x_glu = x_glu.clamp(min=None, max=val)
                    x_linear = x_linear.clamp(min=-val, max=val)
                if self.config.glu_linear_offset != 0.0:
                    x_linear = x_linear + self.config.glu_linear_offset
                scores = per_token_scale.unsqueeze(-1) if per_token_scale is not None else None
                intermediate_parallel = self.xr2glu(x_glu, x_linear, scores=scores)
                scale_fused = per_token_scale is not None
            elif self.config.gated_linear_unit and self.config.xssglu:
                x_glu, x_linear = torch.chunk(intermediate_parallel, 2, dim=-1)
                if (val := self.config.activation_func_clamp_value) is not None:
                    x_glu = x_glu.clamp(min=None, max=val)
                    x_linear = x_linear.clamp(min=-val, max=val)
                if self.config.glu_linear_offset != 0.0:
                    x_linear = x_linear + self.config.glu_linear_offset
                scores = per_token_scale.unsqueeze(-1) if per_token_scale is not None else None
                intermediate_parallel = self.xssglu_glu(x_glu, x_linear, scores=scores)
                scale_fused = per_token_scale is not None
            elif self.config.gated_linear_unit:

                def glu(x):
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    gate = self.config.activation_func(x_glu)
                    return gate * (x_linear + self.config.glu_linear_offset)

                intermediate_parallel = glu(intermediate_parallel)
            elif self.config.xpr:
                intermediate_parallel = self.xpr_act(intermediate_parallel)
            elif self.config.xr2:
                intermediate_parallel = self.xr2_act(intermediate_parallel)
            elif self.config.polynorm:
                intermediate_parallel = self.polynorm_act(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

            if per_token_scale is not None and not scale_fused:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        nvtx_range_pop(suffix="activation")

        # [s, b, h]
        nvtx_range_push(suffix="linear_fc2")

        output, output_bias = apply_module(self.linear_fc2)(
            cast(torch.Tensor, intermediate_parallel)
        )
        nvtx_range_pop(suffix="linear_fc2")

        if per_token_scale is not None and output_bias is not None:
            # if this MLP is an expert, and bias is required, we add the bias to output directly
            # without doing bda later.
            output += output_bias.unsqueeze(0) * per_token_scale.unsqueeze(-1)
            output_bias = None

        return output, output_bias

    # pylint: disable=missing-function-docstring
    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Return the sharded state dictionary of the module."""
        sharded_state_dict = {}
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        # Load-only compatibility for checkpoints saved by OffloadingExpertsMLP before
        # it adopted the canonical Sequential/TE expert keys and [out, in] layout.
        legacy_offloading = self.is_expert and is_legacy_offloading_checkpoint(metadata)
        has_te_extra_state = (metadata or {}).get(
            EXPERT_CKPT_HAS_TE_EXTRA_STATE_KEY, True
        )
        if legacy_offloading:
            assert not self.config.add_bias_linear, (
                "Legacy offloading checkpoints contain expert weights only and cannot initialize "
                "biased SequentialMLP experts."
            )
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f"{prefix}{name}.", sharded_offsets, metadata)
            if legacy_offloading and name in ("linear_fc1", "linear_fc2"):
                key = f"{prefix}{name}.weight"
                if key in sub_sd:
                    sub_sd[key] = make_legacy_offloading_load_factory(
                        sub_sd[key],
                        name,
                        gated=(name == "linear_fc1" and self.config.gated_linear_unit),
                    )
            elif self.config.gated_linear_unit and name == "linear_fc1":
                for k, v in sub_sd.items():
                    if k in (f"{prefix}{name}.weight", f"{prefix}{name}.bias"):
                        sub_sd[k] = apply_swiglu_sharded_factory(
                            v, sharded_offsets, singleton_local_shards
                        )
            if not has_te_extra_state:
                sub_sd = {k: v for k, v in sub_sd.items() if '_extra_state' not in k}
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict

    def backward_dw(self):
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()
