"""CPU tests for the marin_exact quantile-balancing estimator (no CUDA/TE imports).

Run: python tests/unit_tests/transformer/moe/test_marin_qb_exact_cpu.py
AST extraction executes the production function bodies, not copies of them.
"""
import ast
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import List, Optional, Tuple
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[4]
UTILS = "megatron/core/transformer/moe/moe_utils.py"
ROUTER = "megatron/core/transformer/moe/router.py"
FINALIZE = "megatron/core/distributed/finalize_model_grads.py"


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


BASE = {"torch": torch, "Optional": Optional, "Tuple": Tuple, "List": List}
exact = load_function(UTILS, "marin_qb_exact_update", dict(BASE))
# Single pooled forward of the existing (already Marin-checked) per-forward estimator.
pooled = load_function(UTILS, "marin_qb_histogram_update", dict(BASE))
BINS, TOPK = 10000, 3


def make_margins(rows=211, experts=16, seed=3):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, experts, generator=g) * 4 + torch.linspace(-1, 1, experts)
    beta = torch.linspace(-1.5, 1.5, experts)
    alpha = (logits - beta).topk(TOPK + 1, dim=1).values[:, -1]
    return logits - alpha[:, None], beta


def reference(margins, beta, bins=BINS):
    estimate, count = pooled(margins, torch.zeros(margins.shape[0]), beta, TOPK, bins)
    return estimate, count


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        margins, beta = make_margins(rows=97)
        valid = torch.ones(97, dtype=torch.bool)
        expected, _ = reference(margins, beta)
        # Unequal shards, then an entirely empty shard: both ranks must join every
        # collective and agree with the single-process pooled estimate.
        for split in (31, 0, 97):
            sl = slice(0, split) if rank == 0 else slice(split, None)
            est, count, bad = exact(margins[sl], valid[sl], beta, TOPK, BINS, group=dist.group.WORLD)
            torch.testing.assert_close(est, expected, rtol=0, atol=0)
            assert count.item() == 97 and bad.item() == 0
        # A NaN on one rank rejects the update on BOTH ranks.
        poisoned = margins.clone()
        poisoned[5, 2] = float("nan")
        sl = slice(0, 31) if rank == 0 else slice(31, None)
        est, count, bad = exact(poisoned[sl], valid[sl], beta, TOPK, BINS, group=dist.group.WORLD)
        torch.testing.assert_close(est, beta, rtol=0, atol=0)
        assert bad.item() == 1
    finally:
        dist.destroy_process_group()


class TestMarinQBExact(unittest.TestCase):
    def test_matches_single_pooled_forward_for_any_chunking(self):
        margins, beta = make_margins()
        valid = torch.ones(margins.shape[0], dtype=torch.bool)
        expected, expected_count = reference(margins, beta)
        for chunk_rows in (1, 7, 50, 210, 211, 10_000):
            est, count, bad = exact(margins, valid, beta, TOPK, BINS, chunk_rows=chunk_rows)
            torch.testing.assert_close(est, expected, rtol=0, atol=0)
            self.assertEqual(count.item(), expected_count.item())
            self.assertEqual(bad.item(), 0)

    def test_within_one_bin_of_exact_order_statistic(self):
        margins, beta = make_margins(rows=4000, experts=16)
        valid = torch.ones(4000, dtype=torch.bool)
        est, count, _ = exact(margins, valid, beta, TOPK, BINS)
        rank = round(4000 * TOPK / 16)
        truth = margins.topk(rank, dim=0).values[-1]
        width = (margins.max() - margins.min()) / BINS
        self.assertLess((est - truth).abs().max().item(), 2 * width.item())

    def test_padding_rows_are_ignored(self):
        margins, beta = make_margins()
        valid = torch.ones(margins.shape[0], dtype=torch.bool)
        valid[::3] = False
        junk = margins.clone()
        junk[~valid] = 1e9  # would blow up the range if it were counted
        est, count, bad = exact(junk, valid, beta, TOPK, BINS, chunk_rows=40)
        expected, expected_count = reference(margins[valid], beta)
        torch.testing.assert_close(est, expected, rtol=0, atol=0)
        self.assertEqual(count.item(), int(valid.sum()))
        self.assertEqual(bad.item(), 0)
        # A NaN on a padded row is not an observation.
        junk[~valid] = float("nan")
        est, _, bad = exact(junk, valid, beta, TOPK, BINS)
        torch.testing.assert_close(est, expected, rtol=0, atol=0)
        self.assertEqual(bad.item(), 0)

    def test_nonfinite_valid_token_keeps_beta(self):
        margins, beta = make_margins()
        valid = torch.ones(margins.shape[0], dtype=torch.bool)
        for bad_value in (float("nan"), float("inf"), -float("inf")):
            poisoned = margins.clone()
            poisoned[7, 4] = bad_value
            est, count, bad = exact(poisoned, valid, beta, TOPK, BINS, chunk_rows=64)
            torch.testing.assert_close(est, beta, rtol=0, atol=0)
            self.assertEqual(bad.item(), 1)

    def test_empty_and_degenerate(self):
        beta = torch.linspace(-1, 1, 16)
        est, count, bad = exact(torch.empty(0, 16), torch.empty(0, dtype=torch.bool), beta, TOPK, BINS)
        torch.testing.assert_close(est, beta, rtol=0, atol=0)
        self.assertEqual((count.item(), bad.item()), (0, 0))
        est, count, _ = exact(torch.ones(9, 16), torch.zeros(9, dtype=torch.bool), beta, TOPK, BINS)
        torch.testing.assert_close(est, beta, rtol=0, atol=0)
        self.assertEqual(count.item(), 0)
        constant = torch.full((20, 16), 2.0)  # all-equal range: Marin's lo + 1e-6 guard
        est, count, _ = exact(constant, torch.ones(20, dtype=torch.bool), beta, TOPK, BINS)
        self.assertTrue(torch.isfinite(est).all())
        self.assertEqual(count.item(), 20)

    def test_common_offset_shifts_estimate_by_that_offset(self):
        margins, beta = make_margins()
        valid = torch.ones(margins.shape[0], dtype=torch.bool)
        base, _, _ = exact(margins, valid, beta, TOPK, BINS)
        shifted, _, _ = exact(margins + 5.0, valid, beta, TOPK, BINS)
        torch.testing.assert_close(shifted - 5.0, base, rtol=0, atol=2e-3)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
    def test_two_rank_pooled_and_nan_agreement(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(worker, args=(rendezvous,), nprocs=2, join=True)

    def test_router_stores_margins_across_microbatches_without_updating_beta(self):
        combine = load_function(UTILS, "topk_routing_with_score_function", {**BASE, "os": os})
        store = load_function(ROUTER, "_store_marin_exact_margins", {**BASE, "get_num_microbatches": lambda: 3}, "TopKRouter")
        ns = {**BASE, "topk_routing_with_score_function": combine}
        route = load_function(ROUTER, "quantile_balancing", ns, "TopKRouter")
        cfg = SimpleNamespace(
            moe_router_fusion=False, moe_router_num_groups=None, moe_router_group_topk=None,
            moe_router_quantile_balancing_method="marin_exact",
            moe_router_quantile_balancing_marin_exact_store="cpu",
            moe_expert_capacity_factor=None, moe_router_pre_softmax=False,
            moe_router_topk_scaling_factor=2.5)
        beta = torch.tensor([0.2, 0.1, 0.0, -0.1, 0.3, -0.2])
        router = SimpleNamespace(
            config=cfg, tp_cp_group=None, tp_dp_cp_group=None, training=True, score_function="sigmoid",
            topk=2, qb_beta=beta.clone(), qb_beta_accum=None, qb_beta_count=None,
            _qb_exact_margins=None, _qb_exact_valid=None, _qb_exact_rows=0)
        router._store_marin_exact_margins = lambda m, p: store(router, m, p)
        g = torch.Generator().manual_seed(11)
        batches = [torch.randn(n, 6, generator=g) * 3 for n in (4, 5, 4, 9)]  # 4th overflows capacity
        pads = [None, torch.tensor([False, True, False, False, True]), None, None]
        for lg, pm in zip(batches, pads):
            weights, selected, _ = route(router, lg, pm)
            self.assertTrue((selected.sum(-1) == 2).all())
            expected = torch.zeros_like(selected).scatter_(1, (lg - beta).topk(2, dim=1).indices, True)
            self.assertTrue(torch.equal(selected, expected))  # raw logits, not sigmoid scores
            torch.testing.assert_close(weights.sum(-1), torch.full((lg.shape[0],), 2.5))
        torch.testing.assert_close(router.qb_beta, beta)  # beta only moves at the step boundary
        self.assertEqual(router._qb_exact_rows, 22)
        stored = router._qb_exact_margins[:22]
        valid = router._qb_exact_valid[:22]
        want = torch.cat([lg - (lg - beta).topk(3, dim=1).values[:, -1:] for lg in batches])
        torch.testing.assert_close(stored, want)
        self.assertEqual(valid.tolist(), [True] * 4 + [True, False, True, True, False] + [True] * 4 + [True] * 9)
        # Eval / no-grad forwards store nothing.
        router.training = False
        route(router, batches[0])
        self.assertEqual(router._qb_exact_rows, 22)
        # Estimating from the stored buffer equals one pooled forward over the valid rows.
        est, count, _ = exact(stored, valid, beta, 2, BINS, chunk_rows=5)
        ref, ref_count = pooled(want[valid], torch.zeros(int(valid.sum())), beta, 2, BINS)
        torch.testing.assert_close(est, ref, rtol=0, atol=0)
        self.assertEqual(count.item(), ref_count.item())

    def test_rerun_restore_discards_partial_margins(self):
        restore = load_function(ROUTER, "set_router_rerun_state", dict(BASE), "TopKRouter")
        router = SimpleNamespace(qb_beta=torch.zeros(4), _qb_exact_rows=17)
        restore(router, {"qb_beta": torch.ones(4)})
        torch.testing.assert_close(router.qb_beta, torch.ones(4))
        self.assertEqual(router._qb_exact_rows, 0)

    def test_step_boundary_update_centers_and_skips_nonfinite(self):
        recorded = []
        ns = {**BASE, "TransformerConfig": SimpleNamespace,
              "get_attr_wrapped_model": lambda obj, name: getattr(obj, name),
              "get_num_microbatches": lambda: 1,
              "save_to_aux_losses_tracker": lambda name, value, layer, n: recorded.append((name, float(value))),
              "marin_qb_exact_update": lambda *a, **kw: exact(*a, **{**kw, "group": None})}
        update = load_function(FINALIZE, "_update_marin_exact_router_qb_beta", ns)
        margins, beta0 = make_margins(rows=64, experts=8)
        module = SimpleNamespace(
            qb_exact=True, training=True, layer_number=1, qb_beta=beta0.clone(),
            _qb_exact_margins=margins.clone(), _qb_exact_valid=torch.ones(64, dtype=torch.bool),
            _qb_exact_rows=64)
        model = SimpleNamespace(modules=lambda: [module])
        config = SimpleNamespace(num_layers=2, mtp_num_layers=None, moe_router_topk=TOPK,
                                 moe_router_quantile_balancing_marin_num_bins=BINS,
                                 moe_router_quantile_balancing_ema=0.0)
        update([model], config, object())
        est, _, _ = exact(margins, torch.ones(64, dtype=torch.bool), beta0, TOPK, BINS)
        torch.testing.assert_close(module.qb_beta, est - est.mean())
        self.assertAlmostEqual(module.qb_beta.mean().item(), 0.0, places=6)
        self.assertIn(("qb_exact_update_skipped", 0.0), recorded)
        # Non-finite token: the COMPLETE previous beta survives (no centering, no update).
        before = module.qb_beta.clone()
        module._qb_exact_margins[3, 1] = float("nan")
        recorded.clear()
        update([model], config, object())
        torch.testing.assert_close(module.qb_beta, before)
        self.assertIn(("qb_exact_update_skipped", 1.0), recorded)
        self.assertIn(("qb_exact_nonfinite_tokens", 1.0), recorded)
        # Nothing was stored this step (e.g. all-eval): beta is kept.
        module._qb_exact_rows = 0
        update([model], config, object())
        torch.testing.assert_close(module.qb_beta, before)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
