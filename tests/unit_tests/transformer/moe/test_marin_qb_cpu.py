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
compute_pooled = load_function(
    'megatron/core/transformer/moe/moe_utils.py', 'compute_marin_qb_histogram', {'torch': torch})
recover_pooled = load_function(
    'megatron/core/transformer/moe/moe_utils.py', 'recover_marin_qb_beta', {'torch': torch})


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
            bounds = torch.tensor([-256., 256.])
            reference_parts = compute_pooled(logits, alpha, bounds, 10000)
            expected_pooled = recover_pooled(
                *reference_parts[:2], bounds, *reference_parts[2:], beta, 2)
            local_logits, local_alpha = logits[sl], alpha[sl]
            cut = local_logits.shape[0] // 2
            parts = [compute_pooled(local_logits[a:b], local_alpha[a:b], bounds, 10000)
                     for a, b in ((0, cut), (cut, local_logits.shape[0]))]
            hist, tails, bad = sum(p[0] for p in parts), sum(p[1] for p in parts), sum(p[3] for p in parts)
            lo = torch.stack([p[2][0] for p in parts]).min()
            hi = torch.stack([p[2][1] for p in parts]).max()
            for value in (hist, tails, bad):
                dist.all_reduce(value)
            dist.all_reduce(lo, op=dist.ReduceOp.MIN)
            dist.all_reduce(hi, op=dist.ReduceOp.MAX)
            actual_pooled = recover_pooled(hist, tails, bounds, torch.stack((lo, hi)), bad, beta, 2)
            for actual_value, expected_value in zip(actual_pooled, expected_pooled):
                torch.testing.assert_close(actual_value, expected_value)
        actual, count = update(
            logits, alpha, beta, 2, group=dist.group.WORLD,
            padding_mask=torch.ones(37, dtype=torch.bool),
        )
        torch.testing.assert_close(actual, beta)
        assert count.item() == 0
    finally:
        dist.destroy_process_group()


class TestMarinQB(unittest.TestCase):
    def test_fixed_grid_counts_and_beta_are_partition_invariant(self):
        torch.manual_seed(5)
        logits = torch.randn(97, 8) * 3
        beta = torch.linspace(-1, 1, 8)
        alpha = (logits - beta).topk(3, dim=-1).values[:, -1]
        bounds = torch.tensor([-32., 32.])
        expected = compute_pooled(logits, alpha, bounds, 10000)
        for splits in ([0, 97], [0, 11, 97], [0, 0, 11, 37, 97]):
            parts = [compute_pooled(logits[a:b], alpha[a:b], bounds, 10000)
                     for a, b in zip(splits, splits[1:])]
            hist = sum(p[0] for p in parts)
            tails = sum(p[1] for p in parts)
            extrema = torch.stack((torch.stack([p[2][0] for p in parts]).min(),
                                   torch.stack([p[2][1] for p in parts]).max()))
            bad = sum(p[3] for p in parts)
            for actual, wanted in zip((hist, tails, extrema, bad), expected):
                torch.testing.assert_close(actual, wanted)
            result = recover_pooled(hist, tails, bounds, extrema, bad, beta, 2)
            reference = recover_pooled(*expected[:2], bounds, *expected[2:], beta, 2)
            for actual, wanted in zip(result, reference):
                torch.testing.assert_close(actual, wanted)
            self.assertTrue(result[2].item())

    def test_overflow_rejects_bias_but_expands_next_grid(self):
        logits = torch.tensor([[100., 99., 0., -1.]]).repeat(32, 1)
        beta = torch.tensor([0.3, 0.2, -0.1, -0.4])
        alpha = (logits - beta).topk(2, dim=-1).values[:, -1]
        bounds = torch.tensor([-1., 1.])
        hist, tails, extrema, bad = compute_pooled(logits, alpha, bounds, 10000)
        estimate, next_bounds, accept = recover_pooled(hist, tails, bounds, extrema, bad, beta, 1)
        self.assertFalse(accept.item())
        torch.testing.assert_close(estimate, beta)
        self.assertGreater(tails.sum().item(), 0)
        self.assertLess(next_bounds[0].item(), extrema[0].item())
        self.assertGreater(next_bounds[1].item(), extrema[1].item())
        hist, tails, extrema, bad = compute_pooled(logits, alpha, next_bounds, 10000)
        _, _, accept = recover_pooled(hist, tails, next_bounds, extrema, bad, beta, 1)
        self.assertTrue(accept.item())

    def test_nonfinite_valid_rows_reject_update_padding_does_not(self):
        logits = torch.tensor([[2., 1., 0., -1.], [float('nan'), 0., 0., 0.]])
        alpha = torch.tensor([0., float('nan')])
        bounds, beta = torch.tensor([-32., 32.]), torch.zeros(4)
        for mask, expected_bad, expected_accept in (
            (None, 1, False), (torch.tensor([False, True]), 0, True),
        ):
            hist, tails, extrema, bad = compute_pooled(logits, alpha, bounds, 10000, mask)
            self.assertEqual(bad.item(), expected_bad)
            self.assertEqual(hist[0].sum().item(), 1)
            estimate, next_bounds, accept = recover_pooled(hist, tails, bounds, extrema, bad, beta, 1)
            self.assertEqual(accept.item(), expected_accept)
            self.assertTrue(torch.isfinite(estimate).all())
            self.assertTrue(torch.isfinite(next_bounds).all())

    def test_empty_padded_and_batched_layers(self):
        bounds, beta = torch.tensor([-32., 32.]), torch.zeros(4)
        for logits, alpha, mask in (
            (torch.empty(0, 4), torch.empty(0), None),
            (torch.full((3, 4), float('nan')), torch.full((3,), float('nan')),
             torch.ones(3, dtype=torch.bool)),
        ):
            hist, tails, extrema, bad = compute_pooled(logits, alpha, bounds, 100, mask)
            result = recover_pooled(hist, tails, bounds, extrema, bad, beta, 1)
            torch.testing.assert_close(result[0], beta)
            torch.testing.assert_close(result[1], bounds)
            self.assertFalse(result[2].item())
            batched = recover_pooled(hist[None], tails[None], bounds[None], extrema[None], bad[None], beta[None], 1)
            for actual, expected in zip(batched, result):
                torch.testing.assert_close(actual[0], expected)

    def test_grid_contraction_is_limited(self):
        bounds, beta = torch.tensor([-32., 32.]), torch.zeros(4)
        hist, tails, extrema, bad = compute_pooled(torch.zeros(32, 4), torch.zeros(32), bounds, 10000)
        _, next_bounds, _ = recover_pooled(hist, tails, bounds, extrema, bad, beta, 1)
        torch.testing.assert_close(next_bounds, torch.tensor([-28.8, 28.8]))

    def test_rerun_restores_range_and_clears_partial_counts(self):
        ns = {'torch': torch, '_RERUN_ROUTER_BUFFERS': ('qb_beta', 'qb_marin_range')}
        snapshot = load_function('megatron/core/transformer/moe/router.py', 'get_router_rerun_state', ns, 'TopKRouter')
        restore = load_function('megatron/core/transformer/moe/router.py', 'set_router_rerun_state', ns, 'TopKRouter')
        router = SimpleNamespace(qb_beta=torch.arange(4).float(), qb_marin_range=torch.tensor([-32., 32.]),
                                 qb_histogram=torch.ones(4, 100, dtype=torch.long),
                                 qb_marin_tails=torch.ones(4, 2, dtype=torch.long),
                                 qb_marin_bad_tokens=torch.tensor(1), qb_marin_extrema=torch.tensor([-10., 10.]))
        state = snapshot(router)
        router.qb_beta.add_(20)
        router.qb_marin_range.mul_(2)
        restore(router, state)
        torch.testing.assert_close(router.qb_beta, torch.arange(4).float())
        torch.testing.assert_close(router.qb_marin_range, torch.tensor([-32., 32.]))
        self.assertEqual(router.qb_histogram.sum().item(), 0)
        self.assertEqual(router.qb_marin_tails.sum().item(), 0)
        self.assertEqual(router.qb_marin_bad_tokens.item(), 0)
        self.assertEqual(router.qb_marin_extrema.tolist(), [float('inf'), -float('inf')])

    def test_range_checkpoint_roundtrip_and_legacy_bootstrap(self):
        class Parent(torch.nn.Module):
            def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
                return {prefix + key: ('sharded', value) for key, value in self.state_dict().items()}

        class Local:
            def __init__(self, value):
                self.value = value

        tree = ast.parse((ROOT / 'megatron/core/transformer/moe/router.py').read_text(encoding='utf-8'))
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TopKRouter')
        methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in (
            'sharded_state_dict', '_load_from_state_dict', '_save_to_state_dict',
            '_maintain_float32_expert_bias')]
        namespace = {'torch': torch, 'Parent': Parent, 'LocalNonpersistentObject': Local}
        cls = ast.ClassDef(name='Router', bases=[ast.Name(id='Parent', ctx=ast.Load())],
                           keywords=[], body=methods, decorator_list=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                     '<production checkpoint methods>', 'exec'), namespace)

        def make():
            router = namespace['Router']()
            router.register_buffer('qb_beta', torch.arange(4).float())
            router.register_buffer('qb_marin_range', torch.tensor([-32., 32.]))
            return router

        router = make()
        # Old state dict has only beta. Strict loading must still work.
        router.load_state_dict({'qb_beta': torch.ones(4)}, strict=True)
        torch.testing.assert_close(router.qb_beta, torch.ones(4))
        torch.testing.assert_close(router.qb_marin_range, torch.tensor([-32., 32.]))
        router.qb_marin_range.copy_(torch.tensor([-13., 57.]))
        restored = make()
        restored.load_state_dict(router.state_dict(), strict=True)
        torch.testing.assert_close(restored.qb_marin_range, router.qb_marin_range)
        self.assertIsInstance(router.sharded_state_dict(metadata={
            'marin_qb_legacy_range': True})['qb_marin_range'], Local)
        self.assertEqual(router.sharded_state_dict(metadata={})['qb_marin_range'][0], 'sharded')

    def test_pooled_finalization_centers_and_logs_once_per_step(self):
        logs = {}
        ns = {'torch': torch, 'recover_marin_qb_beta': recover_pooled,
              'get_attr_wrapped_model': lambda obj, name: getattr(obj, name),
              'get_num_microbatches': lambda: 4,
              'save_to_aux_losses_tracker': lambda name, value, *args: logs.update({name: value})}
        finalize = load_function('megatron/core/distributed/finalize_model_grads.py',
                                 '_update_marin_router_qb_beta', ns)
        beta = torch.tensor([0.3, 0.2, -0.1, -0.4])
        bounds = torch.tensor([-32., 32.])
        logits = torch.tensor([[3., 2., 1., 0.]]).repeat(32, 1)
        alpha = (logits - beta).topk(2, dim=-1).values[:, -1]
        hist, tails, extrema, bad = compute_pooled(logits, alpha, bounds, 10000)
        expected, expected_bounds, _ = recover_pooled(hist, tails, bounds, extrema, bad, beta, 1)
        module = SimpleNamespace(training=True, qb_beta=beta.clone(), qb_histogram=hist,
                                 qb_marin_range=bounds.clone(), qb_marin_tails=tails,
                                 qb_marin_extrema=extrema, qb_marin_bad_tokens=bad,
                                 layer_number=1, is_mtp_layer=False)
        model = SimpleNamespace(modules=lambda: [module])
        cfg = SimpleNamespace(moe_router_topk=1, moe_router_quantile_balancing_ema=0.25,
                              num_layers=1, mtp_num_layers=None)
        with patch.object(dist, 'all_reduce', return_value=None) as reduce:
            finalize([model], cfg, object())
            self.assertEqual(reduce.call_count, 4)
        wanted = 0.25 * beta + 0.75 * expected
        torch.testing.assert_close(module.qb_beta, wanted - wanted.mean())
        torch.testing.assert_close(module.qb_marin_range, expected_bounds)
        self.assertEqual(logs['qb_histogram_update_skipped'].item(), 0.)
        self.assertEqual(logs['qb_histogram_range_min'].item() / 4, -32.)


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
              "compute_marin_qb_histogram": compute_pooled,
              "topk_routing_with_score_function": combine}
        route = load_function("megatron/core/transformer/moe/router.py", "quantile_balancing", ns, "TopKRouter")
        cfg = SimpleNamespace(moe_router_fusion=False, moe_router_num_groups=None,
                              moe_router_group_topk=None, moe_router_quantile_balancing_method="marin_histogram",
                              moe_router_quantile_balancing_marin_num_bins=10000,
                              moe_router_pre_softmax=False, moe_router_topk_scaling_factor=2.5)
        beta = torch.tensor([0.2, 0.1, 0., -0.1])
        router = SimpleNamespace(config=cfg, tp_cp_group=None, tp_dp_cp_group=object(), training=True,
                                 score_function="sigmoid", topk=2, qb_beta=beta.clone(),
                                 qb_histogram=torch.zeros(4, 10000, dtype=torch.long),
                                 qb_marin_range=torch.tensor([-32., 32.]),
                                 qb_marin_tails=torch.zeros(4, 2, dtype=torch.long),
                                 qb_marin_bad_tokens=torch.zeros((), dtype=torch.long),
                                 qb_marin_extrema=torch.tensor([float('inf'), -float('inf')]))
        logits = torch.tensor([[106., 105., 100., 99.]]).repeat(8, 1)
        weights, selected = route(router, logits)
        expected = torch.zeros_like(selected).scatter_(1, (logits - beta).topk(2, dim=1).indices, True)
        sigmoid_selected = torch.zeros_like(selected).scatter_(1, (logits.sigmoid() - beta).topk(2, dim=1).indices, True)
        self.assertTrue(torch.equal(selected, expected))
        self.assertFalse(torch.equal(selected, sigmoid_selected))
        torch.testing.assert_close(weights.sum(-1), torch.full((8,), 2.5))
        torch.testing.assert_close(router.qb_beta, beta)
        self.assertEqual(router.qb_histogram[0].sum().item(), 8)
        accumulated = router.qb_histogram.clone()
        router.training = False
        route(router, logits)
        torch.testing.assert_close(router.qb_histogram, accumulated)

        infer = load_function("megatron/core/transformer/moe/router.py", "_forward", ns, "InferenceTopKRouter")
        router.gating = lambda x: x
        router._compiled_topk_routing = combine
        router.expert_bias = None
        router.router_replay = None
        _, indices = infer(router, logits)
        torch.testing.assert_close(indices, (logits - beta).topk(2, dim=1).indices)

        # Tiny sigmoid scores expose the denominator epsilon; ordinary scores
        # cannot distinguish 1e-9 from the previous 1e-20 normalization.
        tiny = torch.tensor([[-25., -26., -27., -28.]], requires_grad=True)
        train_weights, train_selected = route(router, tiny)
        infer_weights, infer_indices = infer(router, tiny)
        scores = tiny.sigmoid().gather(1, infer_indices)
        expected_weights = 2.5 * scores / (scores.sum(-1, keepdim=True) + 1e-9)
        torch.testing.assert_close(infer_weights, expected_weights)
        torch.testing.assert_close(train_weights.gather(1, infer_indices), expected_weights)
        self.assertTrue(torch.equal(train_selected.nonzero()[:, 1], infer_indices.flatten()))
        self.assertLess(infer_weights.sum().item(), 0.1)
        train_weights.sum().backward()
        self.assertTrue(torch.isfinite(tiny.grad).all())
        self.assertGreater(tiny.grad.abs().sum().item(), 0.)

        # Other callers retain the original epsilon by default.
        default_weights, _ = combine(
            tiny.detach(), 2, score_function="sigmoid", scaling_factor=2.5,
            precomputed_indices=infer_indices, dense_output=True,
        )
        torch.testing.assert_close(default_weights.sum(), torch.tensor(2.5))

    def test_step_boundary_weighted_average_and_centering(self):
        ns = {"torch": torch, "List": List, "Optional": Optional,
              "TransformerConfig": SimpleNamespace,
              "get_attr_wrapped_model": lambda obj, name: getattr(obj, name)}
        finalize = load_function("megatron/core/distributed/finalize_model_grads.py", "_update_router_qb_beta", ns)
        module = SimpleNamespace(qb_beta=torch.zeros(4), training=True,
                                 qb_beta_accum=torch.tensor([2., 4., 8., 10.]),
                                 qb_beta_count=torch.tensor(2))
        model = SimpleNamespace(modules=lambda: [module])
        config = SimpleNamespace(moe_router_quantile_balancing_method="legacy_average",
                                 moe_router_quantile_balancing_ema=0.)
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
