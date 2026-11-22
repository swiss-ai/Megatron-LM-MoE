# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.
"""Document-aware CP (``--cp-doc-aware``) on SFT-shaped packs: KDA (KCP) and TE attention against CP1 references.

Packs are whole documents padded to ``2 * cp`` tokens, as ``ApertusSFTDataset`` builds them. Each case is sliced with
``get_doc_aware_batch_on_this_cp_rank`` (the path ``pretrain_gpt.get_batch`` takes), run through the CP layer, and
compared token by token, in forward and backward, with the same layer at CP1 on the whole pack. The cases cover all
documents local (no KCP state exchange), local and long documents together, and everything long (plain zigzag).
Run with torchrun --nproc_per_node=4.
"""
import copy
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.doc_aware_cp import get_doc_aware_batch_on_this_cp_rank
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.kimi_delta_attention import HAVE_FLA_CP, HAVE_KDA, KimiDeltaAttention
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from tests.unit_tests.test_utilities import Utils

try:
    import transformer_engine_torch as tex
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
except ImportError:
    tex = None

HIDDEN = 256
HEADS, HEAD_DIM = 8, 64
# 4096 tokens per pack, documents in multiples of 8 (= 2 * cp for cp 4).
CASES = {
    "all_short": [512, 8, 248, 1000, 120, 640, 504, 1000, 64],
    "mixed": [2448, 8, 512, 128, 1000],
    "all_long": [4096],
    "equal_quarters": [1000, 1000, 1000, 1000, 96],
}


def doc_aware_slice(lengths, cp_size, cp_rank, max_padding):
    """This rank's token ids (0-based, -1 = plan padding), its PackedSeqParams, and the full pack's cu_seqlens."""
    device = torch.cuda.current_device()
    cu = torch.tensor([0] + lengths, dtype=torch.int32, device=device).cumsum(0).int()
    batch = {'tokens': torch.arange(1, sum(lengths) + 1, device=device).view(1, -1)}
    batch, params = get_doc_aware_batch_on_this_cp_rank(batch, cu, cp_size, cp_rank, max_padding=max_padding)
    return batch['tokens'].view(-1) - 1, params, cu


def full_params(cu, lengths):
    return PackedSeqParams(
        qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu, cu_seqlens_q_padded=cu, cu_seqlens_kv_padded=cu,
        max_seqlen_q=max(lengths), max_seqlen_kv=max(lengths),
    )


def grads_close(layer, reference, cp_group):
    for (name, param), (_, ref_param) in zip(layer.named_parameters(), reference.named_parameters()):
        if param.grad is None:
            assert ref_param.grad is None, name
            continue
        grad = param.grad.float()
        dist.all_reduce(grad, group=cp_group)
        error = (grad - ref_param.grad.float()).norm() / ref_param.grad.float().norm().clamp_min(1e-8)
        assert error < 0.03, (name, error.item())


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.skipif(not (HAVE_KDA and HAVE_FLA_CP) or tex is None, reason="Needs KDA kernels, fla.ops.cp and TE.")
@pytest.mark.internal
class TestDocAwareCPSFT:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, cp_size):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_size)
        model_parallel_cuda_manual_seed(123)
        self.cp_size = cp_size
        self.tp = parallel_state.get_tensor_model_parallel_group()
        self.cp = parallel_state.get_context_parallel_group()
        self.tp_cp = parallel_state.get_tensor_and_context_parallel_group()
        singletons = [dist.new_group([r]) for r in range(dist.get_world_size())]
        self.own = singletons[dist.get_rank()]
        yield
        Utils.destroy_model_parallel()

    def _config(self, cp_size, **extra):
        return TransformerConfig(
            hidden_size=HIDDEN, num_layers=1, num_attention_heads=HEADS, kv_channels=HEAD_DIM,
            normalization="RMSNorm", use_cpu_initialization=True, activation_func=F.silu, bf16=True,
            params_dtype=torch.bfloat16, tensor_model_parallel_size=1, context_parallel_size=cp_size,
            cp_comm_type="p2p", transformer_impl="transformer_engine", attention_dropout=0.0, hidden_dropout=0.0,
            **extra,
        )

    def _kda(self, config, cp_group, tp_cp_group):
        submodules = get_experimental_attention_variant_module_spec(config=config).submodules
        return KimiDeltaAttention(
            config, submodules=submodules, layer_number=1, bias=False, conv_bias=False, conv_init=1.0,
            use_qk_l2norm=True, A_init_range=(1, 16),
            pg_collection=ProcessGroupCollection(tp=self.tp, cp=cp_group, tp_cp=tp_cp_group),
        ).cuda().bfloat16()

    @pytest.mark.parametrize("case", list(CASES))
    @pytest.mark.parametrize("max_padding", [0.01, 0.1])
    def test_kda_matches_cp1(self, case, max_padding):
        lengths = CASES[case]
        kda = dict(linear_conv_kernel_dim=4, linear_key_head_dim=64, linear_value_head_dim=64,
                   linear_num_key_heads=6, linear_num_value_heads=6, experimental_attention_variant="kda",
                   linear_attention_freq=[1], linear_attention_cp_impl="kcp", linear_attention_safe_output_gate=True)
        config = self._config(self.cp_size, **kda)
        layer = self._kda(config, self.cp, self.tp_cp)
        for param in layer.parameters():  # CPU init is not replicated over CP ranks
            dist.broadcast(param.data, dist.get_process_group_ranks(self.cp)[0], group=self.cp)
        ref_config = copy.deepcopy(config)
        ref_config.context_parallel_size = 1
        reference = self._kda(ref_config, self.own, self.tp)
        reference.load_state_dict(layer.state_dict())

        ids, params, cu = doc_aware_slice(lengths, self.cp_size, self.cp.rank(), max_padding)
        valid = ids >= 0
        torch.manual_seed(777)
        full = torch.randn(sum(lengths), 1, HIDDEN, device='cuda', dtype=torch.bfloat16)
        local = full[ids.clamp_min(0)].masked_fill(~valid.view(-1, 1, 1), 0).clone().requires_grad_()
        ref_input = full.clone().requires_grad_()
        out, _ = layer(local, None, packed_seq_params=params)
        ref_out, _ = reference(ref_input, None, packed_seq_params=full_params(cu, lengths))
        torch.testing.assert_close(out[valid], ref_out.detach()[ids[valid]], atol=5e-3, rtol=5e-3)
        out[valid].float().square().sum().backward()
        ref_out.float().square().sum().backward()
        torch.testing.assert_close(local.grad[valid], ref_input.grad[ids[valid]], atol=8e-3, rtol=2e-2)
        grads_close(layer, reference, self.cp)

    @pytest.mark.parametrize("case", list(CASES))
    @pytest.mark.parametrize("max_padding", [0.01, 0.1])
    def test_te_attention_matches_cp1(self, case, max_padding):
        lengths = CASES[case]
        # conftest's set_env turns flash and fused attention off; TE's unfused path supports neither THD nor CP.
        # Training selects flash for packed SFT batches (--attention-backend auto -> flash).
        os.environ.update(NVTE_FLASH_ATTN='1', NVTE_FUSED_ATTN='0', NVTE_UNFUSED_ATTN='0')

        def attention(cp_size, cp_group):
            return TEDotProductAttention(
                self._config(cp_size), layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type="self",
                cp_comm_type="p2p", pg_collection=ProcessGroupCollection(tp=self.tp, cp=cp_group),
            )

        layer, reference = attention(self.cp_size, self.cp), attention(1, self.own)
        ids, params, cu = doc_aware_slice(lengths, self.cp_size, self.cp.rank(), max_padding)
        valid = ids >= 0
        torch.manual_seed(777)
        full = [torch.randn(sum(lengths), HEADS, HEAD_DIM, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
        local = [x[ids.clamp_min(0)].masked_fill(~valid.view(-1, 1, 1), 0).clone().requires_grad_() for x in full]
        ref = [x.clone().requires_grad_() for x in full]
        out = layer(*local, None, AttnMaskType.causal, packed_seq_params=params)
        ref_out = reference(*ref, None, AttnMaskType.causal, packed_seq_params=full_params(cu, lengths))
        out, ref_out = out.view(len(ids), -1), ref_out.view(sum(lengths), -1)
        # Forward and backward on every rank before any check: a rank that raised early would leave the
        # others in the next CP collective. Tolerances as for bf16 flash attention (1 ulp at |x|~2 is 0.0078).
        out[valid].float().square().sum().backward()
        ref_out.float().square().sum().backward()
        failures = []
        for what, got, want, tol in [("out", out.detach()[valid], ref_out.detach()[ids[valid]], 1e-2)] + [
            (f"grad {n}", g.grad[valid], w.grad[ids[valid]], 2e-2) for n, g, w in zip("qkv", local, ref)
        ]:
            try:
                torch.testing.assert_close(got, want, atol=tol, rtol=tol)
            except AssertionError as error:
                failures.append(f"rank {dist.get_rank()} {what}: {error}")
        failed = torch.tensor([len(failures)], device='cuda')
        dist.all_reduce(failed)
        assert not failures and int(failed) == 0, failures or f"another rank failed ({int(failed)} checks)"
