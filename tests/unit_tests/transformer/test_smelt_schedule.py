"""Dependency-free schedule tests: python -m unittest discover -s tests/unit_tests/transformer -p test_smelt_schedule.py."""
import importlib.util
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('smelt', ROOT / 'megatron/core/transformer/smelt.py')
smelt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smelt)


class TestSmeltSchedule(unittest.TestCase):
    def test_router_visit_capacity_and_normalization(self):
        for start in (-1, 2):
            config = SimpleNamespace(num_layers=10, smelt_loop_start=start, smelt_loop_layers=5)
            visits = [smelt.smelt_layer_visits(config, i) for i in range(1, 11)]
            self.assertEqual(visits, [1, 1, 2, 2, 2, 2, 2, 1, 1, 1])
            for v in visits:
                self.assertEqual((4 * v * 0.25) / v / 4, 0.25)
            self.assertEqual(smelt.smelt_layer_visits(config, 3, True), 1)
        self.assertEqual(smelt.smelt_layer_visits(SimpleNamespace(), None), 1)

    def test_disabled(self):
        self.assertEqual(smelt.smelt_layer_order(10), tuple(range(10)))

    def test_middle_block_order(self):
        self.assertEqual(smelt.smelt_layer_order(10, loop_layers=5),
                         (0, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 8, 9))

    def test_full_loop(self):
        self.assertEqual(smelt.smelt_layer_order(3, loop_layers=3), (0, 1, 2, 0, 1, 2))

    def test_bounds(self):
        for n, start, count in [(10, 8, 5), (10, -2, 5), (10, -1, -1), (10, -1, 11)]:
            with self.assertRaises(ValueError):
                smelt.smelt_layer_order(n, start, count)

    def test_hybrid_execution_counts(self):
        order = smelt.smelt_layer_order(10, 2, 5)
        pattern = [1, 1, 1, 0] * 2 + [1, 0]
        self.assertEqual(sum(pattern[i] for i in order), 11)
        self.assertEqual(sum(i != 0 for i in order), 14)

    def test_actual_residual_transform_scales_bias_after_norm(self):
        # Execute the production helper without importing CUDA/TE dependencies.
        source = ast.parse((ROOT / 'megatron/core/transformer/transformer_layer.py').read_text(encoding='utf-8'))
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'TransformerLayer')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_apply_post_norm')
        identity = type('IdentityOp', (), {})
        env = {'IdentityOp': identity, 'apply_module': lambda module: module}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<residual-helper>', 'exec'), env)
        layer = type('Layer', (), {'residual_output_scale': 0.5})()
        apply = env['_apply_post_norm']
        self.assertEqual(apply(layer, (2.0, 4.0), identity()), (1.0, 2.0))
        self.assertEqual(apply(layer, (2.0, 4.0), lambda x: 3 * x), (9.0, None))
        layer.residual_output_scale = None
        self.assertEqual(apply(layer, (2.0, 4.0), identity()), (2.0, 4.0))


if __name__ == '__main__':
    unittest.main()
