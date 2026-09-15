"""Dependency-free architecture and heterogeneous router-collective checks."""
import ast
from functools import wraps
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    'adjacent_experts', ROOT / 'megatron/core/transformer/moe/adjacent_experts.py'
)
adjacent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adjacent)


class TestAdjacentExperts(unittest.TestCase):
    def test_nine_moe_layers(self):
        self.assertEqual(adjacent.adjacent_expert_pairs(10, [0] + [1] * 9),
                         [(1, 2), (3, 4), (5, 6), (7, 8)])

    def test_dense_gap_and_odd_runs(self):
        self.assertEqual(adjacent.adjacent_expert_pairs(8, [1, 1, 1, 0, 1, 1, 1, 1]),
                         [(0, 1), (4, 5), (6, 7)])
        self.assertEqual(adjacent.adjacent_expert_pairs(6, 2), [])
        self.assertEqual(adjacent.adjacent_expert_pairs(4, 1), [(0, 1), (2, 3)])

    def test_alias_only_routed_experts(self):
        layers = [NS(mlp=NS(num_local_experts=128, experts=object(), router=object(),
                             shared_experts=object(), fc1_latent_proj=object())) for _ in range(3)]
        original = [layer.mlp.experts for layer in layers]
        adjacent.tie_adjacent_experts(layers, [(0, 1)])
        self.assertIs(layers[0].mlp.experts, layers[1].mlp.experts)
        self.assertIs(layers[2].mlp.experts, original[2])
        for attr in ('router', 'shared_experts', 'fc1_latent_proj'):
            self.assertIsNot(getattr(layers[0].mlp, attr), getattr(layers[1].mlp, attr))

    def test_parameter_budget(self):
        routed = 3 * 384 * 448
        self.assertEqual(9 * 256 * routed, (4 * 512 + 256) * routed)
        self.assertEqual(9 * 8 * routed, (8 * 8 + 8) * routed)

    def test_production_router_grouping(self):
        source = ast.parse((ROOT / 'megatron/core/distributed/finalize_model_grads.py').read_text(encoding='utf-8'))
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == '_by_expert_pool_size')
        env = dict(wraps=wraps, SimpleNamespace=NS, get_attr_wrapped_model=getattr)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<router-groups>', 'exec'), env)
        routers = [NS(topk=8, layer_number=i, config=NS(num_moe_experts=e))
                   for i, e in enumerate([512, 512, 256])]
        model = [NS(modules=lambda: iter(routers))]
        calls = []
        def collective(view, config, marker):
            calls.append(([r.config.num_moe_experts for r in view[0].modules()], marker))
        wrapped = env['_by_expert_pool_size'](collective)
        wrapped(model, NS(moe_tie_adjacent_experts=True), marker=42)
        self.assertEqual(calls, [([256], 42), ([512, 512], 42)])
        calls.clear()
        wrapped(model, NS(moe_tie_adjacent_experts=False), marker=7)
        self.assertEqual(calls, [([512, 512, 256], 7)])


if __name__ == '__main__':
    unittest.main()
