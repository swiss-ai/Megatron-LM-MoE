"""Real CPU tensor tests without importing Megatron's CUDA/TE dependencies.

Run: python tests/unit_tests/transformer/moe/test_marin_qb_cpu.py
AST extraction executes the production function bodies, not copies of them.
"""
import ast
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import List, Optional, Tuple
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[4]


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


update = load_function(
    "megatron/core/transformer/moe/moe_utils.py", "marin_qb_histogram_update",
    {"torch": torch, "Optional": Optional, "Tuple": Tuple},
)


def distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(17)
        logits = torch.randn(37, 8) * 30
        beta = torch.linspace(-3, 3, 8)
        alpha = (logits - beta).topk(3, dim=1).values[:, -1]
        expected, _ = update(logits, alpha, beta, 2)
        # Unequal shards, then one entirely empty shard: all ranks must join
        # the same range/count collectives, with identical pooled estimates.
        for split in (11, 0):
            sl = slice(0, split) if rank == 0 else slice(split, None)
            actual, count = update(logits[sl], alpha[sl], beta, 2, group=dist.group.WORLD)
            torch.testing.assert_close(actual, expected)
            assert count.item() == 37
        actual, count = update(
            logits, alpha, beta, 2, group=dist.group.WORLD,
            padding_mask=torch.ones(37, dtype=torch.bool),
        )
        torch.testing.assert_close(actual, beta)
        assert count.item() == 0
    finally:
        dist.destroy_process_group()


class TestMarinQB(unittest.TestCase):
    def test_order_statistic_within_bin_width(self):
        torch.manual_seed(9)
        for tokens in (31, 128):
            logits = torch.randn(tokens, 8) * 50
            beta = torch.linspace(-2, 2, 8)
            alpha = (logits - beta).topk(3, dim=1).values[:, -1]
            actual, count = update(logits, alpha, beta, 2)
            margins = logits - alpha[:, None]
            q = tokens * 2 / 8
            expected = margins.sort(dim=0, descending=True).values[math.ceil(q) - 1]
            width = (margins.max() - margins.min()) / 10000
            self.assertTrue(torch.all((actual - expected).abs() <= width + 1e-5))
            self.assertEqual(count.item(), tokens)

    def test_padding_does_not_change_grid_or_counts(self):
        torch.manual_seed(3)
        logits = torch.randn(64, 8)
        beta = torch.zeros(8)
        alpha = logits.topk(3, dim=1).values[:, -1]
        expected, _ = update(logits, alpha, beta, 2)
        padded = torch.cat((logits, torch.full((5, 8), float('nan'))))
        cutoffs = torch.cat((alpha, torch.full((5,), float('nan'))))
        mask = torch.arange(69) >= 64
        actual, count = update(padded, cutoffs, beta, 2, padding_mask=mask)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(count.item(), 64)

    def test_empty_and_degenerate(self):
        beta = torch.arange(8).float()
        actual, count = update(torch.empty(0, 8), torch.empty(0), beta, 2)
        torch.testing.assert_close(actual, beta)
        self.assertEqual(count.item(), 0)
        actual, _ = update(torch.full((32, 8), 100.), torch.full((32,), 100.), beta, 2)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue((actual.abs() <= 1e-6).all())

    def test_common_offset_invariance_and_no_gradient(self):
        torch.manual_seed(5)
        logits = torch.randn(64, 8, requires_grad=True)
        beta = torch.zeros(8)
        alpha = logits.topk(3, dim=1).values[:, -1]
        actual, _ = update(logits, alpha, beta, 2)
        shifted, _ = update(logits + 100, alpha + 100, beta, 2)
        torch.testing.assert_close(actual, shifted, atol=2e-5, rtol=1e-5)
        self.assertFalse(actual.requires_grad)

    def test_router_selects_logits_and_defers_beta_update(self):
        combine = load_function(
            "megatron/core/transformer/moe/moe_utils.py", "topk_routing_with_score_function",
            {"torch": torch, "Optional": Optional, "Tuple": Tuple, "os": os},
        )

        ns = {"torch": torch, "Optional": Optional,
              "marin_qb_histogram_update": lambda *a, **kw: update(*a, **{**kw, "group": None}),
              "topk_routing_with_score_function": combine}
        route = load_function("megatron/core/transformer/moe/router.py", "quantile_balancing", ns, "TopKRouter")
        cfg = SimpleNamespace(moe_router_fusion=False, moe_router_num_groups=None,
                              moe_router_group_topk=None, moe_router_quantile_balancing_method="marin_histogram",
                              moe_router_quantile_balancing_marin_num_bins=10000,
                              moe_router_pre_softmax=False, moe_router_topk_scaling_factor=2.5,
                              moe_expert_capacity_factor=None,
                              moe_router_quantile_balancing_freeze=False)
        beta = torch.tensor([0.2, 0.1, 0., -0.1])
        router = SimpleNamespace(config=cfg, tp_cp_group=None, tp_dp_cp_group=object(), training=True,
                                 score_function="sigmoid", topk=2, qb_beta=beta.clone(),
                                 qb_beta_accum=torch.zeros(4), qb_beta_count=torch.zeros((), dtype=torch.long))
        logits = torch.tensor([[106., 105., 100., 99.]]).repeat(8, 1)
        weights, selected, _ = route(router, logits)
        expected = torch.zeros_like(selected).scatter_(1, (logits - beta).topk(2, dim=1).indices, True)
        sigmoid_selected = torch.zeros_like(selected).scatter_(1, (logits.sigmoid() - beta).topk(2, dim=1).indices, True)
        self.assertTrue(torch.equal(selected, expected))
        self.assertFalse(torch.equal(selected, sigmoid_selected))
        torch.testing.assert_close(weights.sum(-1), torch.full((8,), 2.5))
        torch.testing.assert_close(router.qb_beta, beta)
        self.assertEqual(router.qb_beta_count.item(), 8)
        accumulated = router.qb_beta_accum.clone()
        router.training = False
        route(router, logits)
        torch.testing.assert_close(router.qb_beta_accum, accumulated)

        infer = load_function("megatron/core/transformer/moe/router.py", "_forward", ns, "InferenceTopKRouter")
        router.gating = lambda x: x
        router._compiled_topk_routing = combine
        router.expert_bias = None
        router.router_replay = None
        _, indices = infer(router, logits)
        torch.testing.assert_close(indices, (logits - beta).topk(2, dim=1).indices)

    def test_step_boundary_weighted_average_and_centering(self):
        ns = {"torch": torch, "List": List, "Optional": Optional,
              "TransformerConfig": SimpleNamespace,
              "get_attr_wrapped_model": lambda obj, name: getattr(obj, name)}
        finalize = load_function("megatron/core/distributed/finalize_model_grads.py", "_update_router_qb_beta", ns)
        module = SimpleNamespace(qb_beta=torch.zeros(4), training=True,
                                 qb_beta_accum=torch.tensor([2., 4., 8., 10.]),
                                 qb_beta_count=torch.tensor(2))
        model = SimpleNamespace(modules=lambda: [module])
        config = SimpleNamespace(moe_router_quantile_balancing_method="marin_histogram",
                                 moe_router_quantile_balancing_ema=0.,
                                 moe_router_quantile_balancing_freeze=False)
        with patch.object(dist, "all_reduce", return_value=SimpleNamespace(wait=lambda: None)):
            finalize([model], config)
        torch.testing.assert_close(module.qb_beta, torch.tensor([-2., -1., 1., 2.]))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
    def test_two_rank_pooled_histogram(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(distributed_worker, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
