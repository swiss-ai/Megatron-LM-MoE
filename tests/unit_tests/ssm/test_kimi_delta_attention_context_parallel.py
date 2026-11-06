# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.

"""Context parallelism over packed documents: CP padding and the KCP linear-attention path."""

import copy

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.cp_relayout import build_relayout_plan, relayout, zigzag_indices
from megatron.core.ssm.kimi_delta_attention import HAVE_FLA_CP, HAVE_KDA, KimiDeltaAttention
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import (
    get_batch_on_this_cp_rank,
    get_thd_batch_on_this_cp_rank,
    pad_thd_batch_for_cp,
)
from tests.unit_tests.test_utilities import Utils

try:
    import transformer_engine_torch as tex
except ImportError:
    tex = None

# Odd lengths, a one-token document, and 4096 in total.
DOCUMENTS = [1, 3, 7, 127, 509, 997, 2452]
HIDDEN = 256


def packed_batch(lengths, cp_size, sp_size):
    """Pad and shard like pretrain_gpt.get_batch; tokens are 1-based ids, 0 marks padding."""
    device = torch.cuda.current_device()
    cu_seqlens = torch.tensor([0] + lengths, dtype=torch.int32, device=device).cumsum(0).int()
    total = sum(lengths)
    batch = dict(
        tokens=torch.arange(1, total + 1, device=device).view(1, -1),
        labels=torch.arange(total, device=device).view(1, -1),
        loss_mask=torch.ones(1, total, device=device),
        attention_mask=None,
        position_ids=torch.cat([torch.arange(n, device=device) for n in lengths]).view(1, -1),
    )
    batch, padded, max_seqlen = pad_thd_batch_for_cp(batch, cu_seqlens, cp_size, sp_size)
    batch, params = get_thd_batch_on_this_cp_rank(batch, padded, padded, max_seqlen)
    return batch, params, cu_seqlens


def sp_slice(x, sp_size, sp_rank, dim=0):
    local = x.shape[dim] // sp_size
    return x.narrow(dim, sp_rank * local, local)


def gather_sp(x, group):
    if group.size() == 1:
        return x
    chunks = [torch.empty_like(x) for _ in range(group.size())]
    dist.all_gather(chunks, x.contiguous(), group=group)
    return torch.cat(chunks)


@pytest.mark.parametrize(("tp_size", "cp_size"), [(1, 2), (1, 4), (2, 2)])
@pytest.mark.skipif(
    not (HAVE_KDA and HAVE_FLA_CP) or tex is None,
    reason="Needs KDA kernels, fla.ops.cp and Transformer Engine.",
)
@pytest.mark.internal
class TestKimiDeltaAttentionContextParallel:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, tp_size, cp_size):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        model_parallel_cuda_manual_seed(123)
        self.tp_size, self.cp_size = tp_size, cp_size
        self.sp_size = tp_size
        self.tp = parallel_state.get_tensor_model_parallel_group()
        self.cp = parallel_state.get_context_parallel_group()
        self.tp_cp = parallel_state.get_tensor_and_context_parallel_group()
        # Reference layers run at CP=1: a process group holding only this rank.
        singletons = [dist.new_group([r]) for r in range(dist.get_world_size())]
        self.own = singletons[dist.get_rank()]
        self.config = TransformerConfig(
            hidden_size=HIDDEN,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=6,
            linear_num_value_heads=6,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            num_attention_heads=8,
            activation_func=F.silu,
            bf16=True,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=tp_size > 1,
            context_parallel_size=cp_size,
            experimental_attention_variant="kda",
            linear_attention_freq=[1],
            linear_attention_cp_impl="kcp",
            linear_attention_safe_output_gate=True,
            transformer_impl="transformer_engine",
        )
        yield
        Utils.destroy_model_parallel()

    def _build(self, config, cp_group, tp_cp_group):
        submodules = get_experimental_attention_variant_module_spec(config=config).submodules
        pg_collection = ProcessGroupCollection(tp=self.tp, cp=cp_group, tp_cp=tp_cp_group)
        return KimiDeltaAttention(
            config,
            submodules=submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=pg_collection,
        ).cuda().bfloat16()

    def _layer_and_reference(self, config):
        layer = self._build(config, self.cp, self.tp_cp)
        # CPU initialization is not replicated over CP ranks; KCP needs identical weights.
        for param in layer.parameters():
            dist.broadcast(param.data, dist.get_process_group_ranks(self.cp)[0], group=self.cp)
        ref_config = copy.deepcopy(config)
        ref_config.context_parallel_size = 1
        reference = self._build(ref_config, self.own, self.tp)
        reference.load_state_dict(layer.state_dict())
        return layer, reference

    def _compare_param_grads(self, layer, reference):
        for (name, param), (_, ref_param) in zip(
            layer.named_parameters(), reference.named_parameters()
        ):
            if param.grad is None:
                assert ref_param.grad is None, name
                continue
            grad = param.grad.float()
            dist.all_reduce(grad, group=self.cp)
            ref_grad = ref_param.grad.float()
            if getattr(param, 'sequence_parallel', False):
                # Layer-norm gradients are partial per SP shard until DDP reduces them.
                dist.all_reduce(grad, group=self.tp)
                dist.all_reduce(ref_grad, group=self.tp)
            error = (grad - ref_grad).norm() / ref_grad.norm().clamp_min(1e-8)
            assert error < 0.03, (name, error.item())

    def test_padding_partition_and_relayout(self):
        batch, params, cu_seqlens = packed_batch(DOCUMENTS, self.cp_size, self.sp_size)
        padded = params.cu_seqlens_q_padded
        lengths = (padded[1:] - padded[:-1]).cpu()
        assert (lengths % (2 * self.cp_size * self.sp_size) == 0).all()
        # Every real token lands on exactly one rank; padding has zero loss and is flagged.
        gathered = [torch.empty_like(batch['tokens']) for _ in range(self.cp_size)]
        dist.all_gather(gathered, batch['tokens'], group=self.cp)
        real = torch.cat(gathered).flatten()
        assert torch.equal(real[real != 0].sort().values, torch.arange(1, sum(DOCUMENTS) + 1, device=real.device))
        assert torch.equal(batch['loss_mask'].bool(), batch['tokens'] != 0)
        assert torch.equal(batch['padding_mask'], batch['tokens'] == 0)
        # Middle pipeline stages partition the mask without any tokens.
        stage = dict(tokens=None, labels=None, loss_mask=None, attention_mask=None, position_ids=None,
                     padding_mask=torch.zeros(1, int(padded[-1]), dtype=torch.bool, device='cuda'))
        stage, _ = get_thd_batch_on_this_cp_rank(stage, padded, padded, torch.tensor([params.max_seqlen_q]))
        assert stage['padding_mask'].shape == batch['padding_mask'].shape
        # The host-side zigzag matches TE's partition, and the relayout round-trips.
        cu_cpu = padded.cpu()
        ids = zigzag_indices(cu_cpu, self.cp_size, self.cp.rank()).cuda()
        te_ids = tex.thd_get_partitioned_indices(padded, int(cu_cpu[-1]), self.cp_size, self.cp.rank())
        assert torch.equal(ids, te_ids.long())
        group = self.tp_cp if self.sp_size > 1 else self.cp
        plan = build_relayout_plan(cu_cpu, self.cp_size, self.sp_size, 1, group, 'cuda')
        mine = sp_slice(ids, self.sp_size, self.tp.rank()).float().view(-1, 1).requires_grad_()
        contiguous = relayout(mine, plan)
        slice_len = int(cu_cpu[-1]) // group.size()
        expected = torch.arange(group.rank() * slice_len, (group.rank() + 1) * slice_len, device='cuda')
        assert torch.equal(contiguous.flatten(), expected.float())
        assert torch.equal(relayout(contiguous, plan, inverse=True), mine)
        contiguous.square().sum().backward()
        assert torch.equal(mine.grad, 2 * mine)

    @pytest.mark.parametrize("recompute_modules", [[], ["qkv_fine", "linear_attn"], ["qkv"]])
    def test_packed_matches_cp1_reference(self, recompute_modules):
        config = copy.deepcopy(self.config)
        if recompute_modules:
            config.recompute_granularity = "selective"
            config.recompute_modules = recompute_modules
        layer, reference = self._layer_and_reference(config)
        # Two outstanding forwards with different boundaries before any backward.
        pending = []
        for lengths in (DOCUMENTS, [255, 1, 3840]):
            batch, params, cu_seqlens = packed_batch(lengths, self.cp_size, self.sp_size)
            torch.manual_seed(777)
            full = torch.randn(sum(lengths), 1, HIDDEN, device='cuda', dtype=torch.bfloat16)
            ids = sp_slice(batch['tokens'].view(-1) - 1, self.sp_size, self.tp.rank())
            valid = ids >= 0
            local = full[ids.clamp_min(0)].clone().requires_grad_()
            ref_input = sp_slice(full, self.sp_size, self.tp.rank()).clone().requires_grad_()
            ref_params = PackedSeqParams(
                qkv_format='thd', cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=max(lengths), max_seqlen_kv=max(lengths),
            )
            out, _ = layer(local, None, packed_seq_params=params)
            ref_out, _ = reference(ref_input, None, packed_seq_params=ref_params)
            ref_full = gather_sp(ref_out.detach(), self.tp)
            torch.testing.assert_close(out[valid], ref_full[ids[valid]], atol=5e-3, rtol=5e-3)
            pending.append((out, ref_out, local, ref_input, ids, valid))
        for out, ref_out, local, ref_input, ids, valid in reversed(pending):
            out[valid].float().square().sum().backward()
            ref_out.float().square().sum().backward()
            ref_grad = gather_sp(ref_input.grad, self.tp)
            torch.testing.assert_close(local.grad[valid], ref_grad[ids[valid]], atol=8e-3, rtol=2e-2)
        self._compare_param_grads(layer, reference)

    def test_unpacked_matches_cp1_reference(self):
        layer, reference = self._layer_and_reference(self.config)
        seq, batch_size = 1024, 4
        torch.manual_seed(777)
        full = torch.randn(seq, batch_size, HIDDEN, device='cuda', dtype=torch.bfloat16)
        index = get_batch_on_this_cp_rank({'tokens': torch.arange(seq, device='cuda').view(1, -1)})['tokens'][0]
        index = sp_slice(index, self.sp_size, self.tp.rank())
        local = full[index].clone().requires_grad_()
        ref_input = sp_slice(full, self.sp_size, self.tp.rank()).clone().requires_grad_()
        out, _ = layer(local, None)
        ref_out, _ = reference(ref_input, None)
        torch.testing.assert_close(out, gather_sp(ref_out.detach(), self.tp)[index], atol=5e-3, rtol=5e-3)
        out.float().square().sum().backward()
        ref_out.float().square().sum().backward()
        torch.testing.assert_close(
            local.grad, gather_sp(ref_input.grad, self.tp)[index], atol=8e-3, rtol=2e-2
        )
        self._compare_param_grads(layer, reference)
