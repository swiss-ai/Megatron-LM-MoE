# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Standalone CPU checks; no Megatron, TE, or CUDA imports.

python tests/unit_tests/transformer/test_output_logit_logging_cpu.py
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
from typing import Tuple
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "output_logit_logging", ROOT / "megatron/core/transformer/output_logit_logging.py"
)
logging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logging)


class SingleRank:
    def rank(self):
        return 0

    def size(self):
        return 1


def extracted(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


def ce_implementation(fused, group=None):
    """Execute the actual changed CE implementations, including backward."""
    base = "megatron/core/tensor_parallel/cross_entropy.py"
    ns = {
        "torch": torch, "Tuple": Tuple, "output_logit_logging": logging,
        "get_tensor_model_parallel_group": lambda: group,
        "get_tensor_model_parallel_rank": lambda: 0 if group is None else group.rank(),
        "get_tensor_model_parallel_world_size": lambda: 1 if group is None else group.size(),
        "get_pg_rank": lambda pg: 0 if pg is None else pg.rank(),
        "get_pg_size": lambda pg: 1 if pg is None else pg.size(),
        "VocabUtility": SimpleNamespace(
            vocab_range_from_per_partition_vocab_size=lambda size, rank, world: (size * rank, size * (rank + 1))
        ),
    }
    extracted(base, "VocabParallelCrossEntropy", ns)
    if fused:
        base = "megatron/core/fusions/fused_cross_entropy.py"
        ns.update({
            "_gather_predicted_logits": ns["VocabParallelCrossEntropy"].gather_predicted_logits,
            "_exp_logits_inplace": ns["VocabParallelCrossEntropy"].exp_logits_inplace,
            "_scale_gradients_inplace": ns["VocabParallelCrossEntropy"].scale_gradients_inplace,
        })
        for name in ("calculate_logits_max", "calculate_predicted_logits", "calculate_cross_entropy_loss", "calculate_gradients"):
            extracted(base, name, ns)
    cls = extracted(base, "_VocabParallelCrossEntropy", ns)
    if fused:
        return lambda x, y: cls.apply(x, y, SingleRank() if group is None else group)
    return lambda x, y, smoothing=0.0: cls.apply(x, y, smoothing, group)


def observe(logits, loss_mask=None, group=None, labels=None):
    if labels is None:
        labels = torch.zeros(logits.shape[:-1], dtype=torch.long)
    with logging.loss_scope(loss_mask):
        margin = logging.prepare_margin(logits, labels, group)
        maximum = logits.max(-1).values
        sum_exp = (logits - maximum.unsqueeze(-1)).exp().sum(-1)
        target = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        loss = sum_exp.log() + maximum - target
        logging.record_ce(maximum, sum_exp, loss, margin)


def worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    groups = [dist.new_group([0]), dist.new_group([1])]
    try:
        # TP: labels on both shards, a confidently wrong prediction, and collapse.
        full = torch.tensor([[[10., 1., 9., 0.]], [[5., 0., 5., -1.]],
                             [[30., 0., 0., 0.]], [[0., 0., 0., 120.]]])
        labels = torch.tensor([[0], [3], [0], [3]])
        others = full.double().scatter(-1, labels.unsqueeze(-1), -float("inf"))
        expected_margins = full.double().gather(-1, labels.unsqueeze(-1)).squeeze(-1) - others.logsumexp(-1)
        logging.set_active(True)
        with logging.forward_scope(), logging.loss_scope(None):
            margin = logging.prepare_margin(full.chunk(2, -1)[rank], labels, dist.group.WORLD)
            if rank == 0:
                torch.testing.assert_close(margin.double(), expected_margins)
            maximum = full.max(-1).values
            sums = (full - maximum.unsqueeze(-1)).exp().sum(-1)
            logging.record_ce(maximum, sums, sums.log(), margin)
        stats = logging.collect(torch.device("cpu"), dist.group.WORLD, groups[rank], True)
        assert stats["output_logits/max"] == 120.0
        expected = torch.quantile(expected_margins.flatten(), .999).item()
        assert abs(stats["output_logits/target_margin_p999"] - expected) < 1e-5
        assert stats["output_logits/target_margin_gt_fp32_cliff_fraction"] == .5

        # Exercise production fused/unfused CE with real TP2 collectives, and
        # verify that enabling diagnostics leaves the existing training unchanged.
        for fused in (False, True):
            fn = ce_implementation(fused, dist.group.WORLD)
            results = []
            for enabled in (False, True):
                logging.set_active(enabled)
                x = full.chunk(2, -1)[rank].to(torch.bfloat16).contiguous().requires_grad_()
                with logging.forward_scope(), logging.loss_scope(None):
                    loss = fn(x, labels)
                loss.sum().backward()
                results.append((loss.detach().clone(), x.grad.clone()))
                reference_logits = full.double().requires_grad_()
                target_logit = reference_logits.gather(-1, labels.unsqueeze(-1))
                competitors = (reference_logits - target_logit).scatter(
                    -1, labels.unsqueeze(-1), -float("inf")
                )
                reference_loss = torch.nn.functional.softplus(competitors.logsumexp(-1))
                reference_loss.sum().backward()
                torch.testing.assert_close(loss.double(), reference_loss.detach(), rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(
                    x.grad, reference_logits.grad.chunk(2, -1)[rank].to(x.dtype),
                    rtol=.01, atol=1e-30,
                )
                if enabled:
                    stats = logging.collect(torch.device("cpu"), dist.group.WORLD, groups[rank], True)
                    assert abs(stats["output_logits/target_margin_max"] - expected_margins.max().item()) < 1e-5
            for before, after in zip(*results):
                torch.testing.assert_close(before, after, rtol=0, atol=0)

        # DP/CP: unequal token counts and sharply different margin distributions.
        local = torch.tensor([[[1., 0.]], [[2., 0.]], [[3., 0.]]]) if rank == 0 else torch.tensor([[[100., 0.]]])
        logging.set_active(True)
        with logging.forward_scope():
            observe(local, group=groups[rank])
        stats = logging.collect(torch.device("cpu"), groups[rank], dist.group.WORLD, True)
        expected = torch.quantile(torch.tensor([1., 2., 3., 100.], dtype=torch.float64), 0.999).item()
        assert abs(stats["output_logits/target_margin_p999"] - expected) < 1e-7
        assert stats["output_logits/target_margin_max"] == 100.0
    finally:
        dist.destroy_process_group()


class TestOutputLogitLogging(unittest.TestCase):
    def tearDown(self):
        logging.set_active(False)

    def collect(self):
        return logging.collect(torch.device("cpu"), None, None, True)

    def test_default_disabled_and_per_step_opt_in(self):
        self.assertFalse(any(logging.should_log(s) for s in range(1, 1000)))
        self.assertTrue(all(logging.should_log(s, 1) for s in range(1, 1000)))
        self.assertFalse(logging.should_log(199, 200))
        self.assertTrue(logging.should_log(200, 200))

    def test_no_tensor_access_when_disabled(self):
        logging.set_active(False)
        with logging.forward_scope(), logging.loss_scope(None):
            self.assertIsNone(logging.prepare_margin(object(), object(), None))
            logging.record_hidden(object(), None, None, False)
            logging.record_ce(object(), object(), object(), None)

    def test_stats_match_direct_fp32_reference(self):
        x = torch.tensor([[[100., 0., -1.], [2., 1., 0.]], [[3., 1., -2.], [1., 1., 1.]]])
        logging.set_active(True)
        with logging.forward_scope():
            observe(x)
        stats = self.collect()
        lse = x.logsumexp(-1)
        margin = x[..., 0] - x[..., 1:].logsumexp(-1)
        self.assertEqual(stats["output_logits/max"], 100.0)
        self.assertAlmostEqual(stats["output_logits/lse_mean"], lse.double().mean().item(), places=6)
        self.assertAlmostEqual(stats["output_logits/lse_max"], lse.max().item(), places=6)
        self.assertAlmostEqual(stats["output_logits/target_margin_p999"], torch.quantile(margin.double(), .999).item(), places=5)
        self.assertEqual(stats["output_logits/zero_loss_fraction"], .25)
        self.assertEqual(stats["output_logits/target_margin_gt_fp32_cliff_fraction"], .25)

    def test_masked_zero_losses_do_not_count(self):
        x = torch.tensor([[[100., 0.]], [[1., 0.]], [[100., 0.]]])
        logging.set_active(True)
        with logging.forward_scope():
            observe(x, torch.tensor([[0., .5, 0.]]))
        stats = self.collect()
        self.assertEqual(stats["output_logits/zero_loss_fraction"], 0.)
        self.assertEqual(stats["output_logits/max"], 1.)
        self.assertEqual(stats["output_logits/target_margin_p999"], 1.)

    def test_all_masked_omitted(self):
        logging.set_active(True)
        with logging.forward_scope():
            observe(torch.tensor([[[float("nan"), 0.]]]), torch.zeros(1, 1))
        self.assertEqual(self.collect(), {})

    def test_hidden_token_l2_norm_and_mask(self):
        hidden = torch.tensor([[[3., 4.]], [[0., 10.]], [[100., 0.]]])
        logging.set_active(True)
        with logging.forward_scope():
            logging.record_hidden(hidden, torch.tensor([[1., 1., 0.]]), None, False)
        stats = self.collect()
        self.assertEqual(stats["output_logits/final_hidden_norm_mean"], 7.5)
        self.assertEqual(stats["output_logits/final_hidden_norm_max"], 10.)

    def test_rerun_and_scope_reset(self):
        logging.set_active(True)
        with logging.forward_scope():
            observe(torch.tensor([[[100., 0.]]]))
        logging.reset_attempt()
        with logging.forward_scope():
            observe(torch.tensor([[[1., 0.]]]))
        observe(torch.tensor([[[999., 0.]]]))  # backward recompute: outside forward
        self.assertEqual(self.collect()["output_logits/max"], 1.)

    def test_mtp_unscoped_and_eval_losses_are_ignored(self):
        logging.set_active(True)
        with logging.forward_scope():
            self.assertIsNone(logging.prepare_margin(object(), object(), None))
            with logging.loss_scope(None, enabled=False):
                self.assertIsNone(logging.prepare_margin(object(), object(), None))
        self.assertEqual(self.collect(), {})

    def test_exact_percentile_tail_merge(self):
        g = torch.Generator().manual_seed(52)
        for n in (1, 2, 999, 1000, 1001, 10000):
            values = torch.randn(n, generator=g)
            k = logging.percentile_tail_size(n)
            tail = torch.cat([p.topk(min(k, p.numel())).values for p in values.tensor_split(7)])
            expected = torch.quantile(values.double(), .999)
            torch.testing.assert_close(logging.percentile_from_tail(tail, n), expected)

    def test_logs_only_token_vectors_not_vocab(self):
        x = torch.randn(5, 2, 100)
        original = x.clone()
        logging.set_active(True)
        with logging.forward_scope():
            observe(x)
        self.assertEqual(sum(t.numel() for t in logging._margins), 10)
        self.assertTrue(all(t.grad_fn is None for t in logging._margins))
        self.assertIsNone(logging._loss_mask)
        torch.testing.assert_close(x, original, rtol=0, atol=0)

    def test_paper_margin_handles_wrong_targets_ties_and_tiny_competitors(self):
        x = torch.tensor([[[18., 0., 0.]], [[120., 0., 0.]],
                          [[20., 0., 0.]], [[1., 1., 1.]], [[-120., -121., -122.]]])
        labels = torch.tensor([[0], [0], [1], [2], [2]])
        original = x.clone()
        others = x.double().scatter(-1, labels.unsqueeze(-1), -float("inf"))
        reference = x.double().gather(-1, labels.unsqueeze(-1)).squeeze(-1) - others.logsumexp(-1)
        logging.set_active(True)
        # Force multiple workspace chunks to check chunk boundaries as well.
        with patch.object(logging, "_MARGIN_CHUNK_ELEMENTS", 6):
            with logging.forward_scope(), logging.loss_scope(None):
                margin = logging.prepare_margin(x, labels, None)
        torch.testing.assert_close(margin.double(), reference)
        torch.testing.assert_close(x, original, rtol=0, atol=0)
        self.assertLess(margin[2].item(), 0.)  # Confidently wrong is not collapse.
        self.assertTrue(torch.isfinite(margin).all())

    def test_many_competitors_are_not_a_top_two_gap(self):
        x = torch.zeros(1, 1, 1001)
        x[..., 0] = 20.
        logging.set_active(True)
        with logging.forward_scope():
            observe(x)
        stats = self.collect()
        self.assertAlmostEqual(stats["output_logits/target_margin_max"], 20. - torch.log(torch.tensor(1000.)).item(), places=5)
        self.assertEqual(stats["output_logits/target_margin_gt_fp32_cliff_fraction"], 0.)
        self.assertNotIn("output_logits/top1_top2_margin_max", stats)

    def test_native_and_unfused_ce_losses_and_gradients_unchanged(self):
        inputs = torch.tensor([[[100., 0., -2., -3.]], [[1., 2., 3., 4.]]], dtype=torch.bfloat16)
        labels = torch.tensor([[0], [1]])
        for fused in (False, True):
            fn = ce_implementation(fused)
            observations = []
            for enabled in (False, True):
                logging.set_active(enabled)
                x = inputs.clone().requires_grad_()
                with patch.object(dist, "all_reduce", lambda *args, **kwargs: None):
                    with logging.forward_scope(), logging.loss_scope(torch.ones(1, 2)):
                        loss = fn(x, labels)
                    loss.sum().backward()
                observations.append((loss.detach().clone(), x.grad.clone()))
                if enabled:
                    self.assertEqual(self.collect()["output_logits/zero_loss_fraction"], 0.)
            for a, b in zip(*observations):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_corrected_ce_preserves_confident_loss_and_target_gradient(self):
        # The old CE returned zero loss/target gradient above the FP32 cliff.
        for fused in (False, True):
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                logits = torch.zeros(6, 1, 16, dtype=dtype)
                labels = torch.tensor([[0], [3], [5], [15], [9], [7]])
                logits.scatter_(-1, labels.unsqueeze(-1),
                                torch.tensor([5., 10., 17., 20., 40., 80.], dtype=dtype).view(6, 1, 1))
                z = logits.double().requires_grad_()
                target = z.gather(-1, labels.unsqueeze(-1))
                others = (z - target).scatter(-1, labels.unsqueeze(-1), -float("inf"))
                expected = torch.nn.functional.softplus(others.logsumexp(-1))
                expected.sum().backward()
                x = logits.clone().requires_grad_()
                logging.set_active(False)
                with patch.object(dist, "all_reduce", lambda *args, **kwargs: None):
                    loss = ce_implementation(fused)(x, labels)
                    loss.sum().backward()
                torch.testing.assert_close(loss.double(), expected.detach(), rtol=1e-5, atol=0)
                # FP16 legitimately underflows some tiny gradients when returned
                # in the input dtype; compare against the same final cast.
                torch.testing.assert_close(
                    x.grad, z.grad.to(dtype),
                    rtol=1e-5 if dtype == torch.float32 else .01, atol=0,
                )
                self.assertTrue((loss > 0).all())
                if dtype != torch.float16:
                    self.assertTrue((x.grad.gather(-1, labels.unsqueeze(-1)) < 0).all())
                if dtype == torch.float32:
                    self.assertTrue((x.grad.double().sum(-1).abs() <
                                     1e-5 * z.grad.abs().sum(-1)).all())

    def test_target_free_sum_restores_input_and_replays_exactly(self):
        ns = {"torch": torch, "Tuple": Tuple}
        cls = extracted("megatron/core/tensor_parallel/cross_entropy.py", "VocabParallelCrossEntropy", ns)
        exp = torch.tensor([[[1., 1e-9, 1e-10]], [[.1, .2, .3]]])
        before = exp.clone()
        mask, indices = torch.tensor([[False], [True]]), torch.tensor([0, 0])
        sums = cls.sum_exp_logits_excluding_target(exp, mask, indices)
        torch.testing.assert_close(exp, before, rtol=0, atol=0)
        expected = before.clone()
        expected[0, 0, 0] = 0.
        torch.testing.assert_close(sums[1], expected.sum(-1), rtol=0, atol=0)
        repeat = cls.sum_exp_logits_excluding_target(exp, mask, indices)
        for first, second in zip(sums, repeat):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_corrected_label_smoothing_matches_fp64(self):
        x = torch.tensor([[[20., 0., 0., 0.]], [[1., 3., -1., 0.]]], requires_grad=True)
        labels = torch.tensor([[0], [2]])
        z = x.detach().double().requires_grad_()
        logp = z.log_softmax(-1)
        smoothing = .1 * 4 / 3
        expected = -(1 - smoothing) * logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1) - smoothing * logp.mean(-1)
        expected.sum().backward()
        logging.set_active(False)
        with patch.object(dist, "all_reduce", lambda *args, **kwargs: None):
            loss = ce_implementation(False)(x, labels, .1)
            loss.sum().backward()
        torch.testing.assert_close(loss.double(), expected.detach(), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(x.grad.double(), z.grad, rtol=1e-5, atol=1e-7)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
    def test_distributed_tp_candidates_and_dp_percentile(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker, args=(Path(tmp, "rendezvous").resolve().as_uri(),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
