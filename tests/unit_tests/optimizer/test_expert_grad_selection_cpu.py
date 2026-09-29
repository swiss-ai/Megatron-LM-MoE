"""Dependency-free tests of the production selector; run this file with Python.

AST extraction avoids importing CUDA/TE. These tests cover selection and metadata,
not tensor operations or collectives; check_expert_grad_norm_distributed.py covers
the real BF16 wrapper, ownership, norm reduction and clipping on four GPUs.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from typing import Optional, Union

ROOT = Path(__file__).resolve().parents[3]


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class Group:
    def __init__(self, rank):
        self._rank = rank

    def rank(self):
        return self._rank


class Scalar(int):
    def __add__(self, other):
        return Scalar(int(self) + int(other))

    __radd__ = __add__

    def item(self):
        return int(self)


class Grad:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def numel(self):
        return len(self.values)


class TestExpertGradSelection(unittest.TestCase):
    def setUp(self):
        namespace = {"List": list, "torch": NS(Tensor=object),
                     "param_is_not_shared": lambda p: not getattr(p, "shared", False),
                     "get_tensor_model_parallel_rank": lambda: 1}
        predicate = load_function("megatron/core/tensor_parallel/layers.py",
                                  "param_is_not_tensor_parallel_duplicate", namespace)
        namespace["tensor_parallel"] = NS(param_is_not_tensor_parallel_duplicate=predicate)
        self.select = load_function("megatron/core/optimizer/optimizer.py",
                                    "get_main_grads_for_grad_norm", namespace, "MegatronOptimizer")
        namespace.update({
            "Optional": Optional, "Union": Union,
            "torch": NS(Tensor=Grad, int64=int, zeros=lambda *a, **kw: Scalar(0),
                        count_nonzero=lambda g: sum(v != 0 for v in g.values),
                        distributed=NS(ProcessGroup=Group, ReduceOp=NS(SUM="sum"),
                                       all_reduce=lambda *a, **kw: None)),
            "get_data_parallel_group_if_dtensor": lambda g, group: group,
            "to_local_if_dtensor": lambda g: g,
        })
        self.count = load_function("megatron/core/optimizer/clip_grads.py",
                                   "count_zeros_fp32", namespace)

    def test_zero_count_expert_and_dense_domains(self):
        expert = self.param(allreduce=False, grad=Grad([0, 0, 1]))
        dense = self.param(grad=Grad([0, 1]))
        self.assertEqual(self.count([expert, dense], None, tp_group=Group(1),
                                    expt_tp_group=Group(0)), 2)
        self.assertEqual(self.count([expert, dense], None, tp_group=Group(0),
                                    expt_tp_group=Group(1)), 1)
        expert.tensor_model_parallel = True
        self.assertEqual(self.count([expert, dense], None, tp_group=Group(0),
                                    expt_tp_group=Group(1)), 3)

    def test_zero_count_shared_missing_and_decoupled(self):
        params = [self.param(allreduce=False, shared=True, grad=Grad([0])),
                  self.param(allreduce=False, grad=None),
                  self.param(expert_tp=True, decoupled_grad=Grad([0, 0, 1]))]
        self.assertEqual(self.count(params, None, use_decoupled_grad=True,
                                    tp_group=Group(1), expt_tp_group=Group(0)), 2)

    def test_zero_count_legacy_fallback(self):
        p = self.param(allreduce=False, grad=Grad([0]))
        self.assertEqual(self.count([p], None, tp_group=Group(1)), 0)

    def test_layerwise_zero_count_passes_groups(self):
        calls = []
        method = load_function("megatron/core/optimizer/layer_wise_optimizer.py",
                               "count_zeros", {"count_zeros_fp32": lambda *a, **kw: calls.append(kw)},
                               "LayerWiseDistributedOptimizer")
        groups = NS(tp=Group(1), expt_tp=Group(0))
        method(NS(chained_optimizers=[NS(get_parameters=lambda: [])], pg_collection=groups,
                  config=NS(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)))
        self.assertIs(calls[0]["tp_group"], groups.tp)
        self.assertIs(calls[0]["expt_tp_group"], groups.expt_tp)

    def selected(self, params, tp=1, etp=0):
        opt = NS(get_parameters=lambda: params, tp_group=Group(tp),
                 config=NS(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False))
        if etp is not None:
            opt.expt_tp_group = Group(etp)
        return self.select(opt)

    def param(self, **kwargs):
        return NS(**dict({"grad": object(), "tensor_model_parallel": False}, **kwargs))

    def test_unique_expert_on_dense_tp_rank_one(self):
        for identity in ({"allreduce": False}, {"expert_tp": True}):
            p = self.param(**identity)
            self.assertEqual(self.selected([p]), [p.grad])

    def test_dense_replica_and_shard(self):
        replica, shard = self.param(), self.param(tensor_model_parallel=True)
        self.assertEqual(self.selected([replica, shard]), [shard.grad])
        self.assertEqual(self.selected([replica], tp=0), [replica.grad])

    def test_expert_tp_replica_and_shard(self):
        replica = self.param(allreduce=False)
        shard = self.param(allreduce=False, tensor_model_parallel=True)
        self.assertEqual(self.selected([replica, shard], tp=0, etp=1), [shard.grad])
        self.assertEqual(self.selected([replica], tp=1, etp=0), [replica.grad])

    def test_shared_and_missing_grads(self):
        self.assertEqual(self.selected([self.param(allreduce=False, shared=True),
                                       self.param(allreduce=False, grad=None)]), [])

    def test_legacy_optimizer_without_expert_group(self):
        self.assertEqual(self.selected([self.param(allreduce=False)], etp=None), [])

    def test_master_metadata_propagation(self):
        path = "megatron/core/optimizer/optimizer.py"
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        attrs = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "_MAIN_PARAM_ROUTING_ATTRS"
                             for t in n.targets))
        namespace = {"_MAIN_PARAM_ROUTING_ATTRS": ast.literal_eval(attrs)}
        propagate = load_function(path, "_propagate_routing_attrs", namespace)
        for flag in (True, False):
            main = NS()
            propagate(main, NS(allreduce=flag))
            self.assertEqual(main.allreduce, flag)

    def test_reported_norm_regression(self):
        selected = []
        for rank in (0, 1):
            expert = self.param(allreduce=False, grad=768 * (rank + 1) ** 2)
            dense = self.param(grad=8)
            selected.extend(self.selected([expert, dense], tp=rank))
        self.assertEqual(sum(selected), 3848)
        self.assertAlmostEqual(math.sqrt(sum(selected)), 62.032249677)


if __name__ == "__main__":
    unittest.main()
