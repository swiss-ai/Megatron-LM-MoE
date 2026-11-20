# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.

"""Cross-document (THD packed sequence) masking for Kimi Delta Attention."""

import copy

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.kimi_delta_attention import HAVE_KDA, KimiDeltaAttention
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.ssm.packed_seq_utils import (
    assert_masking_isolates_documents,
    assert_packed_backward_matches_dense,
    assert_packed_matches_dense,
    make_packed_seq_params,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(
    ("tp_size", "sp", "cp_size"),
    [
        (1, False, 1),
        (2, False, 1),
        (2, True, 1),
        (1, False, 2),
        (2, True, 2),
    ],
)
@pytest.mark.skipif(not HAVE_KDA, reason="The installed FLA does not provide KDA kernels.")
@pytest.mark.internal
class TestKimiDeltaAttentionPacked:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, tp_size, sp, cp_size):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(123)
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.sp_size = tp_size if sp else 1

        pg_collection = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )
        self.config = TransformerConfig(
            hidden_size=256,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            num_attention_heads=8,
            activation_func=F.silu,
            bf16=True,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=sp,
            context_parallel_size=cp_size,
            experimental_attention_variant="kda",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        self.pg_collection = pg_collection
        self.kda = self._build(self.config)

    def _build(self, config):
        submodules = get_experimental_attention_variant_module_spec(
            config=config
        ).submodules
        return KimiDeltaAttention(
            config,
            submodules=submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=self.pg_collection,
        ).cuda().bfloat16()

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_packed_forward_matches_per_document_dense(self):
        assert_packed_matches_dense(self.kda, self.sp_size, self.cp_size)

    def test_packed_forward_masks_across_documents(self):
        # Document lengths are multiples of 2*cp_size so the THD CP partitioning
        # can cut each one evenly, which is what the data pipeline guarantees.
        assert_masking_isolates_documents(
            self.kda, [64, 32, 96, 64], self.sp_size, self.cp_size,
            self.config.hidden_size,
        )

    def test_packed_backward_matches_per_document_dense(self):
        assert_packed_backward_matches_dense(self.kda, self.sp_size, self.cp_size)

    def test_packed_backward_is_finite(self):
        doc_lengths = [64, 32, 96, 64]
        total_len = sum(doc_lengths)
        device = torch.cuda.current_device()
        packed_seq_params = make_packed_seq_params(doc_lengths, device)
        local_len = total_len // self.sp_size // self.cp_size
        hidden_states = torch.randn(
            local_len, 1, self.config.hidden_size,
            device=device, dtype=torch.bfloat16, requires_grad=True,
        )

        out, _ = self.kda(
            hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
        )
        out.float().square().sum().backward()

        assert torch.isfinite(hidden_states.grad).all()
        assert torch.isfinite(self.kda.in_proj.weight.grad).all()
        assert torch.isfinite(self.kda.A_log.grad).all()
        assert torch.isfinite(self.kda.dt_bias.grad).all()

    @pytest.mark.parametrize("recompute_modules", [["qkv"], ["linear_attn"], ["qkv", "linear_attn"]])
    def test_packed_backward_matches_dense_under_recompute(self, recompute_modules):
        """Selective recompute must not change the packed gradients.

        The production config runs `--recompute-modules layernorm qkv`, so the
        recomputed region re-enters in_proj -> conv1d with the same cu_seqlens in
        the backward. Nothing else in this suite exercises that, and a recompute
        that dropped or mismatched cu_seqlens would still yield finite gradients
        while silently changing them.
        """
        config = copy.deepcopy(self.config)
        config.recompute_granularity = "selective"
        config.recompute_modules = recompute_modules
        layer = self._build(config)
        layer.load_state_dict(self.kda.state_dict())
        assert_packed_backward_matches_dense(layer, self.sp_size, self.cp_size)
