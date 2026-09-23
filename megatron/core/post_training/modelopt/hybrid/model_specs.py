# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Expose the native Mamba ModelOpt spec under its upstream name."""

from megatron.core.post_training.modelopt.mamba.model_specs import (
    get_mamba_stack_modelopt_spec as get_hybrid_stack_modelopt_spec,
)
