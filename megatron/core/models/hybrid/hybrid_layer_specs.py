# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Expose the native Mamba specs under their upstream hybrid names."""

from megatron.core.models.mamba.mamba_layer_specs import (
    mamba_inference_stack_spec as hybrid_inference_stack_spec,
    mamba_stack_spec as hybrid_stack_spec,
)
