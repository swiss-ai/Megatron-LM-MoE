"""CPU-only diagnostic selection and training-wiring regression checks.

Run directly with Python; no CUDA or Megatron imports required.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[2]


def tree(path):
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))


class TestTrainingDiagnostics(unittest.TestCase):
    def test_parameter_norm_tp_selection(self):
        fn = next(n for n in tree("megatron/training/utils.py").body
                  if isinstance(n, ast.FunctionDef) and n.name == "calc_params_l2_norm")
        # Execute the actual per-parameter selection prefix, before tensor norm kernels.
        loop = next(n for n in fn.body if isinstance(n, ast.For)).body[0]
        prefix = []
        for node in loop.body:
            if isinstance(node, ast.Assert):
                break
            prefix.append(node)
        loop.body = prefix + ast.parse("selected.append(param)").body
        tp, etp = NS(rank=lambda: 1), NS(rank=lambda: 0)
        dense = NS(tensor_model_parallel=False)
        shard = NS(tensor_model_parallel=True)
        expert = NS(tensor_model_parallel=False, allreduce=False)
        tagged_expert = NS(tensor_model_parallel=False, expert_tp=True)
        ns = dict(model_chunk=NS(parameters=lambda: [dense, shard, expert, tagged_expert]),
                  selected=[], data_parallel_group=None,
                  get_data_parallel_group_if_dtensor=lambda p, g: g,
                  mpu=NS(get_tensor_model_parallel_group=lambda: tp,
                         get_expert_tensor_parallel_group=lambda: etp))
        predicate = next(n for n in tree("megatron/core/tensor_parallel/layers.py").body
                         if isinstance(n, ast.FunctionDef) and n.name == "param_is_not_tensor_parallel_duplicate")
        exec(compile(ast.Module(body=[predicate, loop], type_ignores=[]), "selector", "exec"), ns)
        self.assertEqual([id(p) for p in ns["selected"]], [id(shard), id(expert), id(tagged_expert)])
        etp.rank = lambda: 1
        ns["selected"] = []
        exec(compile(ast.Module(body=[loop], type_ignores=[]), "selector", "exec"), ns)
        self.assertEqual([id(p) for p in ns["selected"]], [id(shard)])

    def test_nan_diagnostics_order_and_reruns(self):
        fn = next(n for n in tree("megatron/training/training.py").body
                  if isinstance(n, ast.FunctionDef) and n.name == "train_step")
        loop = next(n for n in fn.body if isinstance(n, ast.While))
        calls = {n.func.id: n.lineno for n in ast.walk(loop)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertLess(calls["nan_debug_new_step"], calls["forward_backward_func"])
        self.assertLess(calls["forward_backward_func"], calls["nan_debug_check_grads"])
        self.assertLess(calls["nan_debug_check_grads"], calls["nan_debug_sanitize_grads"])


if __name__ == "__main__":
    unittest.main()
