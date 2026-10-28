# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.

"""KDA state recycling with cross-document masking (THD packed sequences, needs the FLA kernels).

The recycled position stores the final state of its first document; in the next iteration every
document of that position starts from it. Checked against the same layer run dense, one document
at a time: the dense path stores the state at the end of its (single-document) sequence and
starts from the stored state, which is exactly what each packed document must reproduce.
"""

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import state_recycle
from megatron.core.ssm.kimi_delta_attention import HAVE_KDA, KimiDeltaAttention
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.ssm.packed_seq_utils import make_packed_seq_params
from tests.unit_tests.test_utilities import Utils

DOCS = [64, 32, 96, 64]


@pytest.mark.skipif(not HAVE_KDA, reason="The installed FLA does not provide KDA kernels.")
@pytest.mark.internal
class TestKimiDeltaAttentionStateRecyclePacked:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(123)
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
            experimental_attention_variant="kda",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
            linear_attention_state_recycle_fraction=1.0,
        )
        pg = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )
        submodules = get_experimental_attention_variant_module_spec(config=self.config).submodules
        self.kda = KimiDeltaAttention(
            self.config, submodules=submodules, layer_number=1, bias=False, conv_bias=False,
            conv_init=1.0, use_qk_l2norm=True, A_init_range=(1, 16), pg_collection=pg,
        ).cuda().bfloat16().train()
        # every data-parallel rank holds one recycled position
        self.gbs = parallel_state.get_data_parallel_world_size()

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def _step(self, iteration, hidden_states, packed_seq_params=None):
        state_recycle.begin_iteration(iteration, self.gbs)
        state_recycle.next_microbatch()
        with torch.no_grad():
            out, _ = self.kda(hidden_states, attention_mask=None, packed_seq_params=packed_seq_params)
        return out

    def test_every_document_starts_from_the_first_documents_state(self):
        device = torch.cuda.current_device()
        params = make_packed_seq_params(DOCS, device)
        bounds = [0] + torch.tensor(DOCS).cumsum(0).tolist()
        torch.manual_seed(0)
        x0 = torch.randn(sum(DOCS), 1, self.config.hidden_size, device=device, dtype=torch.bfloat16)
        x1 = torch.randn_like(x0)

        # iteration 0: nothing stored, the packed forward is the ordinary masked one
        out0 = self._step(0, x0, params)
        stored = self.kda._recycle._next[(0, 0)].clone()
        # iteration 1: every document starts from the stored state
        out1 = self._step(1, x1, params)

        # reference 1: the stored state is the final state of document 0 (dense, from zero)
        self.kda._recycle = state_recycle.RecycleBuffer()
        ref0 = self._step(10, x0[: DOCS[0]])
        torch.testing.assert_close(out0[: DOCS[0]], ref0, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(stored, self.kda._recycle._next[(0, 0)], atol=5e-3, rtol=5e-3)

        # reference 2: each document of iteration 1, alone, started from that state
        for d in range(len(DOCS)):
            self.kda._recycle = state_recycle.RecycleBuffer()
            self.kda._recycle._next = {(0, 0): stored}
            self.kda._recycle.start(20)                  # makes `stored` the previous iteration's
            ref = self._step(20, x1[bounds[d] : bounds[d + 1]])
            torch.testing.assert_close(out1[bounds[d] : bounds[d + 1]], ref, atol=5e-3, rtol=5e-3)

        # and the recycled state matters: from zero, the later documents come out different
        self.kda._recycle = state_recycle.RecycleBuffer()
        fresh = self._step(30, x1, params)
        assert not torch.allclose(fresh[DOCS[0] :], out1[DOCS[0] :], atol=5e-3, rtol=5e-3)
