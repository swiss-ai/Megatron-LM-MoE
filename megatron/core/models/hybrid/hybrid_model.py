# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Adapt the upstream hybrid builder to the native Mamba implementation."""

from megatron.core.models.mamba.mamba_model import MambaModel


class HybridModel(MambaModel):
    """Mamba model accepting the renamed upstream stack-spec argument."""

    def __init__(self, *, hybrid_stack_spec, **kwargs):
        super().__init__(mamba_stack_spec=hybrid_stack_spec, **kwargs)
