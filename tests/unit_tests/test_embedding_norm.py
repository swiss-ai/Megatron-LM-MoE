"""Tests for --embedding-norm (a norm with a learnable gain on the embedding output)."""

import argparse

import pytest
import torch

from megatron.core.transformer.transformer_config import TransformerConfig


def test_argument_and_default():
    from megatron.training.arguments import add_megatron_arguments

    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    assert parser.parse_args([]).embedding_norm is False
    assert parser.parse_args(["--embedding-norm"]).embedding_norm is True
    assert TransformerConfig.__dataclass_fields__["embedding_norm"].default is False


def _world_size():
    import os

    return int(os.environ.get("WORLD_SIZE", "1"))


@pytest.mark.skipif(not torch.cuda.is_available() or _world_size() < 2, reason="needs 2 GPUs (torchrun)")
@pytest.mark.parametrize("position", ["rope", "learned_absolute"])
def test_sequence_parallel_gradient_of_the_norm_gain(position):
    """TP=2 with sequence parallelism: the norm gain's gradient, after the TP reduction that
    finalize_model_grads applies to params flagged `sequence_parallel`, equals the gradient of the
    same computation on one device over the full sequence. "rope": the embeddings are
    reduce-scattered before the norm (each rank normalizes its slice, the gradient is summed over
    TP). "learned_absolute": the scatter comes after the norm (each rank normalizes the full
    sequence, the gradient is already complete and must not be summed)."""
    import math

    from megatron.core import parallel_state as ps
    from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(123)
        H, V, S, B = 64, 128, 16, 2
        cfg = TransformerConfig(num_layers=1, hidden_size=H, num_attention_heads=4, tensor_model_parallel_size=2,
                                sequence_parallel=True, embedding_norm=True, scale_embeddings_by_sqrt_hidden=True,
                                hidden_dropout=0.0, normalization="RMSNorm", transformer_impl="transformer_engine",
                                params_dtype=torch.float32)
        emb = LanguageModelEmbedding(config=cfg, vocab_size=V, max_sequence_length=S, position_embedding_type=position).cuda()
        tp, rank, group = ps.get_tensor_model_parallel_world_size(), ps.get_tensor_model_parallel_rank(), ps.get_tensor_model_parallel_group()
        src = ps.get_tensor_model_parallel_src_rank()
        weight = emb.embedding_norm.weight
        assert weight.sequence_parallel is (position == "rope")
        with torch.no_grad():
            weight.copy_(torch.linspace(0.5, 1.5, H, device="cuda"))            # same non-trivial gain on every rank
            if position == "learned_absolute":                                  # replicated table: make it identical
                torch.distributed.broadcast(emb.position_embeddings.weight.data, src=src, group=group)
        gen = torch.Generator().manual_seed(7)                                  # same batch and readout on every rank
        ids = torch.randint(0, V, (B, S), generator=gen).cuda()
        pos = torch.arange(S).repeat(B, 1).cuda()
        R = torch.randn(S, B, H, generator=gen).cuda()

        out = emb(ids, pos)
        assert out.shape == (S // tp, B, H)
        (out * R[rank * (S // tp):(rank + 1) * (S // tp)]).sum().backward()
        grad = weight.grad.clone()
        if cfg.sequence_parallel and getattr(weight, "sequence_parallel", False):
            torch.distributed.all_reduce(grad, group=group)                     # as finalize_model_grads does

        table = [torch.empty_like(emb.word_embeddings.weight) for _ in range(tp)]
        torch.distributed.all_gather(table, emb.word_embeddings.weight.detach(), group=group)
        x = torch.cat(table)[ids]
        if position == "learned_absolute":
            x = x + emb.position_embeddings.weight.detach()[pos]
        x = x.transpose(0, 1) * math.sqrt(H)
        w = weight.detach().clone().requires_grad_()
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.layernorm_epsilon) * w
        (y * R).sum().backward()
        torch.testing.assert_close(grad, w.grad, rtol=1e-4, atol=1e-4)
    finally:
        Utils.destroy_model_parallel()
