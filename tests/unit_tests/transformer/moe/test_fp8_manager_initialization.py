# Copyright (c) 2026, ETH Zurich / EPFL / Swiss AI Initiative.

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.experts import OffloadingExpertsMLP
from megatron.core.transformer.moe.experts_offloading_fp8_util import FP8ExpertsParameterManager
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(
    'mode,fp8,debug',
    [
        ('fine-grained', True, False),
        ('coarse-grained', True, False),
        ('fine-grained', False, False),
        ('fine-grained', True, True),
    ],
)
def test_expert_constructor_initializes_fp8_manager(mode, fp8, debug, monkeypatch):
    monkeypatch.setattr(FP8ExpertsParameterManager, '_instance', None)
    Utils.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            num_moe_experts=2,
            moe_ffn_hidden_size=128,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
            perform_initialization=False,
            gated_linear_unit=True,
            add_bias_linear=False,
            gradient_accumulation_fusion=True,
            moe_use_offloading_experts=True,
            moe_use_inplace_fp8_param=fp8,
            moe_offloading_experts_debug_mode=debug,
            moe_offloading_mode=mode,
            moe_offloading_num_chunks=2,
            moe_offloading_num_stages=2,
            moe_offloading_chunk_size=1,
        )
        groups = ProcessGroupCollection.use_mpu_process_groups()
        expert = OffloadingExpertsMLP(2, config, pg_collection=groups)
        if fp8 and not debug:
            manager = FP8ExpertsParameterManager.get_instance()
            assert manager.config is expert.fp8_config
            assert manager.config.coarse_grained_reload == (mode == 'coarse-grained')
            # Additional layers must reuse the singleton rather than replace it.
            OffloadingExpertsMLP(2, config, pg_collection=groups)
            assert FP8ExpertsParameterManager.get_instance() is manager
        else:
            assert FP8ExpertsParameterManager._instance is None
    finally:
        Utils.destroy_model_parallel()
