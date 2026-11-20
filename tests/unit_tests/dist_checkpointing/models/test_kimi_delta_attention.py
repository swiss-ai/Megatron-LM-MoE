# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.

"""Distributed checkpointing for Kimi Delta Attention.

KDA repacks the fused in_proj as [Q, K, V, f_a, g_a, beta] while GDN, whose
sharded_state_dict it inherits, uses [Q, K, V, z, beta, alpha]. Splitting a KDA
in_proj on GDN's sections asks for more rows than the tensor has, so saving a
distributed checkpoint raised before this was overridden. Production defaults to
--ckpt-format torch_dist, so that failure lands at the first --save-interval.
"""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load, load_plain_tensors, save
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.kimi_delta_attention import HAVE_KDA, KimiDeltaAttention
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def initialize_kda(seed, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    default_config_kwargs = dict(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        add_bias_linear=False,
        pipeline_dtype=torch.bfloat16,
        experimental_attention_variant="kda",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )
    default_config_kwargs.update(**config_kwargs)
    config = TransformerConfig(**default_config_kwargs)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    return KimiDeltaAttention(
        config,
        submodules=get_experimental_attention_variant_module_spec(config=config).submodules,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=1.0,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=pg_collection,
    )


@pytest.mark.skipif(not HAVE_KDA, reason="The installed FLA does not provide KDA kernels.")
class TestKimiDeltaAttentionReconfiguration:

    def test_in_proj_split_covers_the_projection(self):
        """The sections must sum to the local in_proj width.

        This is the invariant _split_tensor_factory enforces, checked directly so
        a layout change is caught without needing a full save.
        """
        Utils.initialize_model_parallel(1, 1)
        try:
            model = initialize_kda(1)
            sections, names = model._in_proj_sharded_split()
            assert len(sections) == len(names)
            assert sum(sections) == model.in_proj_dim // model.tp_size
            assert sum(sections) == model.in_proj.weight.shape[0]
        finally:
            Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "src_tp_pp,dest_tp_pp",
        [
            ((1, 1), (1, 1)),  # save/load round trip
            ((2, 1), (2, 1)),  # TP-sharded round trip
            ((1, 1), (2, 1)),  # reshard TP up
            ((2, 1), (1, 1)),  # reshard TP down
        ],
    )
    def test_parallel_reconfiguration_e2e(self, tmp_path_dist_ckpt, src_tp_pp, dest_tp_pp):
        src_tp, src_pp = src_tp_pp
        dest_tp, dest_pp = dest_tp_pp
        Utils.initialize_model_parallel(src_tp, src_pp)
        with (
            TempNamedDir(tmp_path_dist_ckpt / 'test_kda_reconfiguration_A') as ckpt_dir_A,
            TempNamedDir(tmp_path_dist_ckpt / 'test_kda_reconfiguration_B') as ckpt_dir_B,
        ):
            layer_prefix = f'{parallel_state.get_pipeline_model_parallel_rank()}.'
            model_A = initialize_kda(
                1, tensor_model_parallel_size=src_tp, pipeline_model_parallel_size=src_pp
            )
            save(model_A.sharded_state_dict(prefix=layer_prefix), ckpt_dir_A)
            Utils.destroy_model_parallel()

            Utils.initialize_model_parallel(dest_tp, dest_pp)
            model_B = initialize_kda(
                2, tensor_model_parallel_size=dest_tp, pipeline_model_parallel_size=dest_pp
            )
            state_dict = load(model_B.sharded_state_dict(prefix=layer_prefix), ckpt_dir_A)
            model_B.load_state_dict(
                {k.removeprefix(layer_prefix): v for k, v in state_dict.items()}
            )
            save(model_B.sharded_state_dict(prefix=layer_prefix), ckpt_dir_B)
            Utils.destroy_model_parallel()

            Utils.initialize_model_parallel(1, 1)
            diffs = diff(load_plain_tensors(ckpt_dir_A), load_plain_tensors(ckpt_dir_B))
            assert not any(map(bool, diffs)), diffs
        Utils.destroy_model_parallel()
