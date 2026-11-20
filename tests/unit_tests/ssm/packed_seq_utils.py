# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.

"""Shared helpers for the GDN/KDA cross-document (THD packed sequence) tests."""

import torch

from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams

try:
    import transformer_engine_torch as tex
except ImportError:
    tex = None


def gated_backward_blocked_by_fla():
    """Whether FLA refuses to run the gated chunk backward in this environment.

    fla/ops/common/chunk_o.py raises unconditionally for a gated backward on
    Hopper with 3.4.0 <= triton < 3.7.1 (FLA issue #640), before touching any
    input. That blocks GDN's backward regardless of packing, so tests that need
    it are skipped rather than reported as a failure of this code. KDA is
    unaffected -- chunk_kda has its own backward and does not go through it.
    """
    try:
        from fla.utils import (
            IS_NVIDIA_HOPPER,
            TRITON_ABOVE_3_4_0,
            TRITON_ABOVE_3_7_1,
        )
    except ImportError:
        return False
    return IS_NVIDIA_HOPPER and TRITON_ABOVE_3_4_0 and not TRITON_ABOVE_3_7_1


def make_packed_seq_params(doc_lengths, device):
    """PackedSeqParams for one flat sequence made of ``doc_lengths`` documents."""
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(doc_lengths).cumsum(0)), dtype=torch.int32, device=device
    )
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=max(doc_lengths),
        max_seqlen_kv=max(doc_lengths),
    )


def global_indices_on_this_rank(cu_seqlens, total_len, sp_size, cp_size):
    """Global token positions this rank's hidden_states slice holds.

    Mirrors how the data pipeline shards a packed batch: ``get_thd_batch_on_this_
    cp_rank`` cuts the CP shard per document with ``tex.thd_get_partitioned_indices``,
    then sequence-parallelism takes a contiguous slice of what is left.
    """
    device = cu_seqlens.device
    if cp_size > 1:
        assert tex is not None, "Transformer Engine is required for THD CP partitioning."
        index = tex.thd_get_partitioned_indices(
            cu_seqlens, total_len, cp_size, parallel_state.get_context_parallel_rank()
        )
    else:
        index = torch.arange(total_len, device=device)
    if sp_size > 1:
        local = index.numel() // sp_size
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        index = index[tp_rank * local : (tp_rank + 1) * local]
    return index


def assert_masking_isolates_documents(layer, doc_lengths, sp_size, cp_size, hidden_size):
    """Perturbing the first document must leave every later document's output alone.

    That is exactly what cross-document masking buys: without it the recurrent
    state and the causal convolution both carry document 0 into document 1. The
    check is made on whatever slice of the sequence this rank holds, so it is
    valid for any TP/SP/CP layout.
    """
    device = torch.cuda.current_device()
    total_len = sum(doc_lengths)
    packed_seq_params = make_packed_seq_params(doc_lengths, device)
    cu_seqlens = packed_seq_params.cu_seqlens_q

    torch.manual_seed(0)
    full = torch.randn(total_len, 1, hidden_size, device=device, dtype=torch.bfloat16)
    perturbed = full.clone()
    perturbed[: doc_lengths[0]] = torch.randn_like(perturbed[: doc_lengths[0]])

    index = global_indices_on_this_rank(cu_seqlens, total_len, sp_size, cp_size)
    out_a, _ = layer(full[index], attention_mask=None, packed_seq_params=packed_seq_params)
    out_b, _ = layer(perturbed[index], attention_mask=None, packed_seq_params=packed_seq_params)

    later_docs = index >= doc_lengths[0]
    assert torch.equal(out_a[later_docs], out_b[later_docs]), (
        "documents after the perturbed one changed: masking is leaking across "
        "document boundaries"
    )
    if (~later_docs).any():
        assert not torch.equal(out_a[~later_docs], out_b[~later_docs]), (
            "the perturbed document's own output did not change; the test input "
            "is not exercising the layer"
        )


def assert_packed_matches_dense(layer, sp_size, cp_size, num_docs=4, doc_len=32):
    """Packed forward must equal running each document as its own batch element.

    Only meaningful without sequence/context parallelism, where the dense
    reference sees the same tokens on the same rank; under SP/CP the equivalent
    guarantee is covered by assert_masking_isolates_documents.
    """
    hidden = layer.config.hidden_size
    total_len = num_docs * doc_len
    device = torch.cuda.current_device()

    if sp_size > 1 or cp_size > 1:
        assert_masking_isolates_documents(
            layer, [doc_len] * num_docs, sp_size, cp_size, hidden
        )
        return

    torch.manual_seed(0)
    hidden_states = torch.randn(total_len, 1, hidden, device=device, dtype=torch.bfloat16)
    packed_seq_params = make_packed_seq_params([doc_len] * num_docs, device)

    packed_out, _ = layer(
        hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
    )

    dense_hidden_states = (
        hidden_states.view(num_docs, doc_len, hidden).transpose(0, 1).contiguous()
    )
    dense_out, _ = layer(dense_hidden_states, attention_mask=None)
    dense_out_flat = dense_out.transpose(0, 1).reshape(total_len, 1, hidden)

    torch.testing.assert_close(packed_out, dense_out_flat, atol=5e-3, rtol=5e-3)


def assert_packed_backward_matches_dense(layer, sp_size, cp_size, num_docs=4, doc_len=32):
    """Packed BACKWARD must equal running each document as its own batch element.

    The forward check alone cannot catch a cu_seqlens that is right going in and
    wrong coming back: a mis-scoped backward still produces finite gradients, it
    just produces the wrong ones. Comparing against the per-document dense
    reference pins the gradients themselves, for the input and for every
    parameter that receives one.

    Skipped under SP/CP, where the dense reference would not see the same tokens
    on this rank.
    """
    if sp_size > 1 or cp_size > 1:
        return

    hidden = layer.config.hidden_size
    total_len = num_docs * doc_len
    device = torch.cuda.current_device()
    packed_seq_params = make_packed_seq_params([doc_len] * num_docs, device)

    torch.manual_seed(0)
    base = torch.randn(total_len, 1, hidden, device=device, dtype=torch.bfloat16)

    def run(hidden_states, **kwargs):
        layer.zero_grad(set_to_none=True)
        out, _ = layer(hidden_states, attention_mask=None, **kwargs)
        # Sum of squares over the same element set either way, so the two runs
        # share one scalar objective and their gradients are comparable.
        out.float().square().sum().backward()
        grads = {
            name: p.grad.detach().clone()
            for name, p in layer.named_parameters()
            if p.grad is not None
        }
        return hidden_states.grad.detach().clone(), grads

    packed_input = base.clone().requires_grad_(True)
    packed_grad, packed_params = run(
        packed_input, packed_seq_params=packed_seq_params
    )

    dense_input = (
        base.view(num_docs, doc_len, hidden).transpose(0, 1).contiguous().requires_grad_(True)
    )
    dense_grad, dense_params = run(dense_input)
    dense_grad = dense_grad.transpose(0, 1).reshape(total_len, 1, hidden)

    torch.testing.assert_close(
        packed_grad.float(), dense_grad.float(), atol=2e-2, rtol=2e-2,
        msg=lambda m: f"input gradient mismatch between packed and dense: {m}",
    )
    assert packed_params, "no parameter received a gradient"
    for name in sorted(packed_params):
        torch.testing.assert_close(
            packed_params[name].float(), dense_params[name].float(),
            atol=2e-2, rtol=2e-2,
            msg=lambda m, name=name: f"gradient mismatch for {name}: {m}",
        )
