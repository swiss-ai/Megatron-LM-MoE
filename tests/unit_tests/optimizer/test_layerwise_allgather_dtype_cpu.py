"""CPU regression test: layer-wise all-gather with mixed-dtype parameter shards.

KDA decay params stay fp32 next to bf16 weights, so a rank's shard can mix dtypes. The
dense all-gather must gather each dtype separately; one flat buffer/dtype across all
ranks made torch raise "Invalid usage of tensors with different dtypes".

Run this file with Python. The production nested helper is extracted by AST (no CUDA/TE
import) and run on 3 gloo processes. NCCL tolerates uneven all-gather sizes but gloo does
not, so the shim below exchanges tensors with all_gather_object after calling torch's real
per-call dtype check (the check that fired in the cluster crash).
"""
import ast
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "megatron/core/optimizer/layer_wise_optimizer.py"
WORLD = 3
HELPERS = ("_allgather_helper", "_allgather_single_dtype")  # the latter exists once fixed


def load_helper(source_text):
    """Extract the dense all-gather helper (and its per-dtype worker, if any)."""
    tree = ast.parse(source_text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LayerWiseDistributedOptimizer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "allgather_params")
    nodes = [n for n in method.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]

    def all_gather(gather_list, tensor, group=None):
        dist.distributed_c10d._ensure_all_tensors_same_dtype(gather_list, tensor)
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, tensor.clone())
        for dst, src in zip(gather_list, gathered):
            if dst is not tensor:
                dst.copy_(src)

    ns = dict(
        torch=NS(empty=torch.empty, distributed=NS(all_gather=all_gather)),
        _flatten_dense_tensors=_flatten_dense_tensors,
        _unflatten_dense_tensors=_unflatten_dense_tensors,
        get_pg_rank=lambda group: dist.get_rank(),
        get_pg_size=lambda group: dist.get_world_size(),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "layer_wise_optimizer", "exec"), ns)
    return ns["_allgather_helper"]


# (dtype, numel) per param, per owning rank. Rank 1 owns no fp32 param and rank 2 lists its
# fp32 param first, so both "shard without fp32" and "fp32 first" are covered.
LAYOUT = [
    [(torch.bfloat16, 3), (torch.float32, 2)],
    [(torch.bfloat16, 4)],
    [(torch.float32, 1), (torch.bfloat16, 2)],
]


# Single-dtype shards (the common bf16 case) must behave exactly as before.
LAYOUT_BF16 = [[(torch.bfloat16, 5), (torch.bfloat16, 1)], [(torch.bfloat16, 2)], [(torch.bfloat16, 7)]]


def worker(rank, init_file, source_text, layout, errors):
    # Short timeout: a rank that raises before the collective leaves its peers waiting, and a
    # test must fail rather than hang.
    dist.init_process_group(
        "gloo",
        init_method=f"file:///{init_file}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=20),
    )
    try:
        helper = load_helper(source_text)
        # Every rank holds the full model; only the owner's copy is "updated" locally.
        params_list = [
            [torch.zeros(n, dtype=dt) for dt, n in shard] for shard in layout
        ]
        for i, p in enumerate(params_list[rank]):
            p.fill_((rank + 1) * 10 + i + 0.5)
        helper(params_list, None)
        for owner, shard in enumerate(params_list):
            for i, p in enumerate(shard):
                assert p.dtype == layout[owner][i][0], (owner, i, p.dtype)
                want = torch.full_like(p, (owner + 1) * 10 + i + 0.5)
                assert torch.equal(p, want), f"rank {rank}: owner {owner} param {i} = {p} != {want}"
    except BaseException as e:  # noqa: BLE001 - report to the parent, keep the group teardown clean
        errors[rank] = f"{type(e).__name__}: {e}"
    finally:
        dist.destroy_process_group()


def run_world(source_text, layout=LAYOUT):
    """Run the helper on WORLD processes; return {rank: error string} for failed ranks."""
    with tempfile.TemporaryDirectory() as tmp:
        init_file = (Path(tmp) / "rdzv").as_posix()
        with mp.Manager() as manager:
            errors = manager.dict()
            mp.spawn(worker, args=(init_file, source_text, layout, errors), nprocs=WORLD, join=True)
            return dict(errors)


class TestLayerwiseAllgatherDtype(unittest.TestCase):
    def test_mixed_dtype_shards_are_gathered_per_dtype(self):
        errors = run_world(SOURCE.read_text(encoding="utf-8"))
        self.assertEqual(errors, {}, errors)

    def test_single_dtype_shards_unchanged(self):
        errors = run_world(SOURCE.read_text(encoding="utf-8"), LAYOUT_BF16)
        self.assertEqual(errors, {}, errors)


if __name__ == "__main__":
    unittest.main()
