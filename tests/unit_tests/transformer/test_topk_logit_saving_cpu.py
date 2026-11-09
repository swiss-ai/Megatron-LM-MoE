# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Standalone CPU checks; no Megatron, TE, or CUDA imports.

python tests/unit_tests/transformer/test_topk_logit_saving_cpu.py
"""

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
from typing import Tuple
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "topk_logit_saving", ROOT / "megatron/core/transformer/topk_logit_saving.py"
)
saving = importlib.util.module_from_spec(spec)
spec.loader.exec_module(saving)


def ce_inputs(logits):
    """What the CE hook sees: global max and the sum of exp(logit - max) over the vocabulary."""
    logits_max = logits.max(dim=-1).values
    return logits_max, (logits - logits_max[..., None]).exp().sum(dim=-1)


def topk_by_id(logits_2d, k):
    """Reference top-k in the saved order: ids ascending, values to match."""
    top = torch.topk(logits_2d, k, dim=-1)
    ids, order = top.indices.sort(dim=1)
    return ids, top.values.gather(1, order)


def run_step(logits, target, loss_mask, k, clip, tp_group=None):
    saving.set_active(True, k, clip)
    with saving.forward_scope(), saving.loss_scope(loss_mask):
        prepared = saving.prepare(logits, target, tp_group)
        saving.record(prepared, *ce_inputs(logits))
    return prepared


class TopKLogitSavingTest(unittest.TestCase):
    def tearDown(self):
        saving.set_active(False)
        saving.wait()

    def test_schedule(self):
        steps = [s for s in range(1, 41) if saving.should_save(s, 20, 3)]
        self.assertEqual(steps, [1, 2, 3, 21, 22, 23])
        self.assertFalse(saving.should_save(5, 0, 3))
        self.assertFalse(saving.should_save(5, 20, 0))
        # 30B tokens per window at GBS 6144 x seq 8192: 596.05 -> 597 iterations (rounded up).
        self.assertEqual(saving.window_iterations(30_000_000_000, 6144 * 8192, 2000), 597)
        self.assertEqual(saving.window_iterations(10**15, 6144 * 8192, 2000), 2000)
        self.assertEqual(saving.window_iterations(0, 6144 * 8192, 2000), 0)

    def test_roundtrip_matches_reference(self):
        torch.manual_seed(0)
        seq, batch, vocab, k, clip = 6, 3, 1000, 32, 16.0
        logits = torch.randn(seq, batch, vocab) * 4
        target = torch.randint(0, vocab, (seq, batch))
        loss_mask = torch.ones(batch, seq)
        loss_mask[1, 2:] = 0  # masked tokens are not saved
        run_step(logits, target, loss_mask, k, clip)
        with tempfile.TemporaryDirectory() as tmp:
            path = saving.flush(7, tmp, rank=3)
            saving.wait()
            self.assertTrue(path.endswith(os.path.join("iter_0000007", "rank_00003.npz")))
            out = saving.decode(path)

        valid = (loss_mask.transpose(0, 1).reshape(-1) > 0)
        flat = logits.reshape(-1, vocab)[valid]
        ref_ids, ref_values = topk_by_id(flat, k)
        self.assertEqual(out["step"], 7)
        np.testing.assert_array_equal(out["indices"], ref_ids.numpy())
        self.assertTrue(np.all(np.diff(out["indices"], axis=1) > 0))  # ids strictly ascending
        np.testing.assert_array_equal(out["labels"], target.reshape(-1)[valid].numpy())
        np.testing.assert_allclose(out["lse"], torch.logsumexp(flat, dim=-1).numpy(), rtol=1e-5)
        in_range = (ref_values.max(dim=1, keepdim=True).values - ref_values) <= clip
        error = np.abs(out["logits"] - ref_values.numpy())
        self.assertLessEqual(float(error[in_range.numpy()].max()), clip / 510 + 1e-4)
        self.assertEqual(out["logits"].shape, (int(valid.sum()), k))

    def test_logits_beyond_clip_saturate(self):
        values = torch.tensor([[10.0, 9.0, -50.0, float("-inf")]])
        q = saving.quantize(values, 16.0)
        self.assertEqual(q.dtype, torch.uint8)
        self.assertEqual(q[0, 0].item(), 0)
        self.assertEqual(q[0, 2].item(), 255)
        self.assertEqual(q[0, 3].item(), 255)
        self.assertEqual(q[0, 1].item(), round(1.0 / (16.0 / 255)))

    def test_elias_fano_roundtrip(self):
        torch.manual_seed(5)
        # (vocab, k): the real shape, small vocabularies, and the no-low-bits corner cases
        for vocab, k in ((200_064, 256), (1000, 32), (40, 32), (300, 300), (65_536, 1), (4096, 7)):
            rows = 37
            ids = torch.stack([torch.randperm(vocab)[:k].sort().values for _ in range(rows)])
            packed = saving.encode_indices_ef(ids, vocab)
            low_bits, low_bytes, upper_bytes = saving.ef_layout(vocab, k)
            self.assertEqual(tuple(packed.shape), (rows, low_bytes + upper_bytes), (vocab, k))
            decoded = saving.decode_indices_ef(packed.numpy(), vocab, k, chunk=10)
            np.testing.assert_array_equal(decoded, ids.numpy(), err_msg=str((vocab, k)))

    def test_elias_fano_size_at_real_shape(self):
        low_bits, low_bytes, upper_bytes = saving.ef_layout(200_064, 256)
        self.assertEqual((low_bits, low_bytes, upper_bytes), (9, 288, 81))
        self.assertEqual(low_bytes + upper_bytes, 369)  # 1.44 bytes per id, against 3 for 24 bits

    def test_elias_fano_rejects_bad_ids(self):
        with self.assertRaises(ValueError):
            saving.encode_indices_ef(torch.tensor([[3, 3, 9]]), 100)  # not strictly ascending
        with self.assertRaises(ValueError):
            saving.encode_indices_ef(torch.tensor([[1, 2, 100]]), 100)  # outside the vocabulary

    def test_shard_merge_equals_global_topk(self):
        torch.manual_seed(1)
        rows, vocab, k = 5, 400, 16
        logits = torch.randn(rows, vocab)
        shards = logits.chunk(2, dim=1)
        parts = []
        for rank, shard in enumerate(shards):
            values, indices = saving.local_topk(shard, k)
            parts.append((values, indices + rank * shard.shape[1]))
        values, indices = saving.merge_topk(
            torch.cat([p[0] for p in parts], dim=1), torch.cat([p[1] for p in parts], dim=1), k
        )
        ref = torch.topk(logits, k, dim=1)
        self.assertTrue(torch.equal(indices, ref.indices))
        self.assertTrue(torch.equal(values, ref.values))

    def test_tiling_does_not_change_topk(self):
        torch.manual_seed(2)
        logits = torch.randn(23, 300)
        full = torch.topk(logits, 8, dim=1)
        original = saving._TILE_ELEMENTS
        saving._TILE_ELEMENTS = 300 * 4  # four rows per tile, 23 rows -> a partial last tile
        try:
            values, indices = saving.local_topk(logits, 8)
        finally:
            saving._TILE_ELEMENTS = original
        self.assertTrue(torch.equal(indices, full.indices))
        self.assertTrue(torch.equal(values, full.values))

    def test_inactive_and_outside_scope_do_nothing(self):
        logits = torch.randn(4, 2, 50)
        target = torch.zeros(4, 2, dtype=torch.long)
        saving.set_active(False)
        with saving.forward_scope(), saving.loss_scope(None):
            self.assertIsNone(saving.prepare(logits, target, None))
        saving.set_active(True, 4, 16.0)
        self.assertIsNone(saving.prepare(logits, target, None))  # outside the forward/loss scope
        with saving.forward_scope():  # a CE outside loss_scope (MTP, ...) is not recorded
            self.assertIsNone(saving.prepare(logits, target, None))
        self.assertIsNone(saving.flush(1, "unused", 0))

    def test_rejected_attempt_is_dropped(self):
        logits = torch.randn(4, 2, 50)
        target = torch.zeros(4, 2, dtype=torch.long)
        run_step(logits, target, None, 4, 16.0)
        saving.reset_attempt()
        self.assertIsNone(saving.flush(1, "unused", 0))

    def test_mask_shape_mismatch_raises(self):
        logits = torch.randn(4, 2, 50)
        target = torch.zeros(4, 2, dtype=torch.long)
        with self.assertRaises(ValueError):
            run_step(logits, target, torch.ones(2, 5), 4, 16.0)


def extracted(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


def ce_implementation(fused, group):
    """Execute the actual changed CE implementations (native and fused), forward only."""
    base = "megatron/core/tensor_parallel/cross_entropy.py"
    ns = {
        "torch": torch, "Tuple": Tuple, "topk_logit_saving": saving,
        "output_logit_logging": SimpleNamespace(
            prepare_margin=lambda *a, **k: None, record_ce=lambda *a, **k: None
        ),
        "get_tensor_model_parallel_group": lambda: group,
        "get_tensor_model_parallel_rank": lambda: 0,
        "get_tensor_model_parallel_world_size": lambda: 1,
        "get_pg_rank": lambda pg: 0,
        "get_pg_size": lambda pg: 1,
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
        return lambda x, y: cls.apply(x, y, group)
    return lambda x, y: cls.apply(x, y, 0.0, group)


class RealCrossEntropyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        dist.init_process_group(
            "gloo", init_method=f"file://{os.path.join(cls.tmp.name, 'init')}", rank=0, world_size=1
        )

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        cls.tmp.cleanup()

    def tearDown(self):
        saving.set_active(False)
        saving.wait()

    def check(self, fused):
        torch.manual_seed(4)
        seq, batch, vocab, k = 5, 2, 300, 16
        logits = torch.randn(seq, batch, vocab) * 3
        target = torch.randint(0, vocab, (seq, batch))
        ce = ce_implementation(fused, dist.group.WORLD)
        expected = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab), target.reshape(-1), reduction="none"
        ).reshape(seq, batch)

        saving.set_active(False)
        plain = ce(logits.clone(), target)

        saving.set_active(True, k, 16.0)
        with saving.forward_scope(), saving.loss_scope(None):
            loss = ce(logits.clone(), target)
        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(loss, plain)  # the hook does not change the loss

        with tempfile.TemporaryDirectory() as tmp:
            path = saving.flush(1, tmp, 0)
            saving.wait()
            out = saving.decode(path)
        ref_ids, _ = topk_by_id(logits.reshape(-1, vocab), k)
        np.testing.assert_array_equal(out["indices"], ref_ids.numpy())
        np.testing.assert_array_equal(out["labels"], target.reshape(-1).numpy())
        np.testing.assert_allclose(
            out["lse"], torch.logsumexp(logits.reshape(-1, vocab), dim=-1).numpy(), rtol=1e-5
        )

    def test_native_cross_entropy(self):
        self.check(fused=False)

    def test_fused_cross_entropy(self):
        self.check(fused=True)


def _two_rank_worker(rank, init_file, results):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    torch.manual_seed(3)
    full = torch.randn(6, 2, 200)
    shard = full.chunk(2, dim=-1)[rank].contiguous()
    target = torch.randint(0, 200, (6, 2))
    saving.set_active(True, 8, 16.0)
    with saving.forward_scope(), saving.loss_scope(None):
        prepared = saving.prepare(shard, target, dist.group.WORLD)
    if rank == 0:
        ref = torch.topk(full.reshape(-1, 200), 8, dim=-1)
        results[0] = (
            torch.equal(prepared["indices"], ref.indices),
            torch.equal(prepared["values"], ref.values),
        )
    else:
        results[1] = prepared is None
    dist.destroy_process_group()


class TwoRankTest(unittest.TestCase):
    def test_tensor_parallel_merge_over_gloo(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = mp.get_context("spawn").Manager()
            results = manager.dict()
            mp.spawn(
                _two_rank_worker, args=(os.path.join(tmp, "init"), results), nprocs=2, join=True
            )
            self.assertEqual(results[0], (True, True))
            self.assertTrue(results[1])  # only TP rank 0 keeps the result


if __name__ == "__main__":
    unittest.main()
