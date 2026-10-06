"""CPU tests for the scaled-softsign MoE router score function.

Run: python tests/unit_tests/transformer/moe/test_scaled_softsign_router_cpu.py
AST extraction executes the production function bodies without importing Megatron's CUDA/TE
dependencies.
"""
import ast
import os
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[4]
MOE_UTILS = "megatron/core/transformer/moe/moe_utils.py"
ROUTER_INPUT_LOGGING = "megatron/core/transformer/moe/router_input_logging.py"


def load(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in names)
        or (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
        )
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


def moe_namespace():
    ns = {
        "torch": torch,
        "os": os,
        "Optional": __import__("typing").Optional,
        "Tuple": __import__("typing").Tuple,
        "_ROUTING_OOB_ACCUM": {},
        "HAVE_TE": False,
        "fused_topk_with_score_function": None,
        "fused_compute_score_for_moe_aux_loss": None,
    }
    return load(
        MOE_UTILS,
        {
            "scaled_softsign",
            "group_limited_topk",
            "topk_routing_with_score_function",
            "compute_routing_scores_for_aux_loss",
        },
        ns,
    )


class ScaledSoftsignTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.ns = moe_namespace()
        self.f = self.ns["scaled_softsign"]

    def test_values_and_range(self):
        x = torch.tensor([-1e6, -3.0, -1.0, 0.0, 1.0, 3.0, 1e6], dtype=torch.float64)
        y = self.f(x)
        expected = 0.5 + 0.5 * x / (1 + x.abs())
        torch.testing.assert_close(y, expected)
        self.assertEqual(self.f(torch.tensor(0.0)).item(), 0.5)
        self.assertTrue(((y > 0) & (y < 1)).all())
        self.assertTrue((y[1:] >= y[:-1]).all())

    def test_gradient_is_polynomial_not_saturating(self):
        x = torch.tensor([0.0, 1.0, -2.0, 5.0, 20.0, 40.0], requires_grad=True)
        self.f(x).sum().backward()
        torch.testing.assert_close(x.grad, 0.5 / (1 + x.detach().abs()) ** 2)
        # Sigmoid's fp32 gradient is exactly 0 at x = 40; scaled softsign's is not.
        xs = torch.tensor(40.0, requires_grad=True)
        torch.sigmoid(xs).backward()
        self.assertEqual(xs.grad.item(), 0.0)
        self.assertGreater(x.grad[-1].item(), 1e-4)

    def test_topk_routing_matches_reference(self):
        topk_routing = self.ns["topk_routing_with_score_function"]
        logits = torch.randn(16, 32, dtype=torch.float32) * 4
        probs, routing_map = topk_routing(
            logits, topk=4, score_function="scaled-softsign", scaling_factor=2.5
        )
        scores = self.f(logits)
        top = scores.topk(4, dim=1).indices
        self.assertTrue(torch.equal(top.sort(dim=1).values, routing_map.nonzero()[:, 1].view(16, 4)))
        ref = scores.gather(1, top)
        ref = ref / ref.sum(dim=1, keepdim=True) * 2.5
        torch.testing.assert_close(probs.gather(1, top), ref)
        torch.testing.assert_close(probs.sum(dim=1), torch.full((16,), 2.5))
        # Same selection as sigmoid: both are monotone in the logit.
        _, sigmoid_map = topk_routing(logits, topk=4, score_function="sigmoid")
        self.assertTrue(torch.equal(routing_map, sigmoid_map))

    def test_expert_bias_and_precomputed_indices(self):
        topk_routing = self.ns["topk_routing_with_score_function"]
        logits = torch.randn(8, 16)
        bias = torch.zeros(16)
        bias[3] = 10.0  # forces expert 3 into every token's top-k
        _, routing_map = topk_routing(
            logits, topk=2, score_function="scaled-softsign", expert_bias=bias
        )
        self.assertTrue(routing_map[:, 3].all())
        indices = torch.tensor([[0, 1]] * 8)
        probs, routing_map = topk_routing(
            logits, topk=2, score_function="scaled-softsign", precomputed_indices=indices
        )
        self.assertTrue(routing_map[:, :2].all() and not routing_map[:, 2:].any())
        s = self.f(logits[:, :2])
        torch.testing.assert_close(probs[:, :2], s / s.sum(dim=1, keepdim=True))

    def test_gradient_reaches_selected_logits(self):
        topk_routing = self.ns["topk_routing_with_score_function"]
        logits = (torch.randn(8, 16) * 30).requires_grad_()
        probs, routing_map = topk_routing(logits, topk=4, score_function="scaled-softsign")
        (probs * torch.randn_like(probs)).sum().backward()
        selected_grad = logits.grad[routing_map]
        self.assertTrue((selected_grad != 0).all())
        self.assertTrue((logits.grad[~routing_map] == 0).all())

    def test_aux_loss_scores(self):
        aux_scores = self.ns["compute_routing_scores_for_aux_loss"]
        logits = torch.randn(8, 16)
        routing_map, scores = aux_scores(logits, topk=2, score_function="scaled-softsign")
        s = self.f(logits)
        torch.testing.assert_close(scores, s / s.sum(dim=1, keepdim=True))
        self.assertEqual(routing_map.sum().item(), 16)


class GateThresholdLoggingTest(unittest.TestCase):
    def test_softsign_cuts_match_gate_values(self):
        ns = {"torch": torch}
        load(
            ROUTER_INPUT_LOGGING,
            {"_SAT_LOGIT", "_GATE_THRESHOLDS", "_SOFTSIGN_SAT_LOGIT", "_SOFTSIGN_GATE_THRESHOLDS"},
            ns,
        )
        f = moe_namespace()["scaled_softsign"]
        for p, cut in ns["_SOFTSIGN_GATE_THRESHOLDS"]:
            self.assertAlmostEqual(f(torch.tensor(cut, dtype=torch.float64)).item(), p, places=12)
        self.assertAlmostEqual(
            f(torch.tensor(ns["_SOFTSIGN_SAT_LOGIT"], dtype=torch.float64)).item(), 0.99, places=12
        )
        self.assertEqual(
            [p for p, _ in ns["_SOFTSIGN_GATE_THRESHOLDS"]],
            [p for p, _ in ns["_GATE_THRESHOLDS"]],
        )


class ConfigLiteralTest(unittest.TestCase):
    def test_literal_includes_scaled_softsign(self):
        source = (ROOT / "megatron/core/transformer/transformer_config.py").read_text(encoding="utf-8")
        self.assertIn(
            "moe_router_score_function: Literal['softmax', 'sigmoid', 'scaled-softsign']", source
        )


if __name__ == "__main__":
    unittest.main()
