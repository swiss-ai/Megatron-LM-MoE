"""Tests for the two input-embedding ablations: per-token row gains (--hypersphere-gains-mode-embedding row)
and the embedding on the chained Adam (--md-input-embedding-external)."""

import argparse

import pytest
import torch

from megatron.core.optimizer.md_decoupling import MDDecoupling, get_megatron_mddecoupling_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from tests.unit_tests.test_md_decoupling import _NoProcessGroups, _TinyMDDecouplingModel
from tests.unit_tests.test_utilities import Utils


def _embedding(rows=6, dim=8):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(rows, dim))
    p.is_embedding_or_output_parameter = True
    p.is_md_embedding_parameter = True
    return p


def _radial_steps(opt, p, coeff, steps=60):
    # loss = sum_i coeff_i * ||row_i||^2: a purely radial gradient (positive coeff wants a smaller row).
    for _ in range(steps):
        p.grad = 2 * coeff[:, None] * p.detach()
        opt.step()


def test_embedding_row_gain_scales_the_stored_row():
    """With row gains on the embedding, the stored weight (what the embedding lookup returns, i.e. what
    enters the residual stream) is gain_i * unit row: a token pushed to shrink gets a smaller row, one
    pushed to grow a larger one, an unused token stays at norm 1. Without the gain every row stays 1."""
    coeff = torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    kw = dict(lr=0.01, hypersphere_mode="row", hypersphere_embedding_mode="row", hypersphere_gains_mode="rowcol",
              gain_parametrization="softplus", use_orthogonal_updates=False, pg_collection=_NoProcessGroups())

    p = _embedding()
    opt = MDDecoupling(params=[p], hypersphere_gains_mode_embedding="row", **kw)
    _radial_steps(opt, p, coeff)
    norms = p.detach().norm(dim=1)
    gain = torch.nn.functional.softplus(opt.state[p]["row_gain"])
    assert "col_gain" not in opt.state[p]
    torch.testing.assert_close(norms, gain, rtol=1e-5, atol=0)
    assert norms[0] < 0.9 and norms[1] > 1.1
    torch.testing.assert_close(norms[2:], torch.ones(4), rtol=1e-5, atol=0)

    q = _embedding()
    opt = MDDecoupling(params=[q], hypersphere_gains_mode_embedding="none", **kw)
    _radial_steps(opt, q, coeff)
    torch.testing.assert_close(q.detach().norm(dim=1), torch.ones(6), rtol=1e-5, atol=0)
    assert "row_gain" not in opt.state[q]


def test_argument_and_default():
    from megatron.training.arguments import _add_training_args

    parser = _add_training_args(argparse.ArgumentParser(allow_abbrev=False))
    assert parser.parse_args([]).md_input_embedding_external is False
    assert parser.parse_args(["--md-input-embedding-external"]).md_input_embedding_external is True
    assert OptimizerConfig().md_input_embedding_external is False


def _owners(optimizer, param):
    return [type(w.optimizer).__name__.replace("AdamW", "Adam") for w in optimizer.chained_optimizers
            for g in w.optimizer.param_groups for q in g["params"] if q is param]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="optimizer wrapper creates CUDA scale")
@pytest.mark.parametrize("external", [False, True])
def test_input_embedding_external_moves_only_the_word_embedding(external):
    """The word embedding goes to the chained Adam (free row norms after a step); the untied LM head
    and every other matrix stay in MDDecoupling. Off (default): everything as before."""
    Utils.initialize_model_parallel()
    try:
        model = _TinyMDDecouplingModel(shared_output=False, device="cuda")
        # weight_decay=0 as in the runs: the chained AdamW would otherwise decay the unused rows too.
        config = OptimizerConfig(optimizer="md_decoupling", lr=0.01, min_lr=0.0, use_orthogonal_updates=False,
                                 weight_decay=0.0,
                                 hypersphere_mode="row", hypersphere_embedding_mode="row",
                                 md_router_use_orthogonal_updates=False, hypersphere_gains_mode_output="none",
                                 md_input_embedding_external=external)
        optimizer = get_megatron_mddecoupling_optimizer(config, [model], use_gloo_process_groups=False)
        emb, head = model.embedding.word_embeddings.weight, model.output_layer.weight
        assert _owners(optimizer, emb) == (["Adam"] if external else ["MDDecoupling"])
        for p in (head, model.router.weight, model.attn.linear_qkv.weight, model.mlp.linear_fc2.weight):
            assert _owners(optimizer, p) == ["MDDecoupling"]
        for p in model.parameters():
            assert p.requires_grad and len(_owners(optimizer, p)) == 1
        # a resume matches the checkpoint's param groups by their identity keys: they must be unique
        for w in optimizer.chained_optimizers:
            w._assert_unique_param_group_keys(w.optimizer.param_groups, "test")
        coeff = torch.tensor([1.0, -1.0, 0, 0, 0, 0, 0, 0], device="cuda")
        emb_norm0 = emb.detach().norm(dim=1)   # Adam does not project: rows keep their own initial norms
        for _ in range(20):
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
            emb.grad = 2 * coeff[:, None] * emb.detach()
            head.grad = 2 * coeff[:, None] * head.detach()
            for w in optimizer.chained_optimizers:
                w.optimizer.step()
        emb_norm, head_norm = emb.detach().norm(dim=1), head.detach().norm(dim=1)
        torch.testing.assert_close(head_norm, torch.ones_like(head_norm), rtol=1e-4, atol=0)
        if external:
            assert emb_norm[0] < 0.95 * emb_norm0[0] and emb_norm[1] > 1.05 * emb_norm0[1]
            torch.testing.assert_close(emb_norm[2:], emb_norm0[2:], rtol=1e-5, atol=0)
        else:
            torch.testing.assert_close(emb_norm, torch.ones_like(emb_norm), rtol=1e-4, atol=0)
    finally:
        Utils.destroy_model_parallel()
