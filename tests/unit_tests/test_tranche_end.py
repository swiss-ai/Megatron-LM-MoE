"""CPU regression checks for finite-tranche construction and checkpoint exits.

Run directly with Python; executes production functions with distributed/model
dependencies stubbed, so no CUDA or Megatron imports are required.
"""

import ast
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys
import json
import tempfile


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("tranche", ROOT / "megatron/training/tranche.py")
tranche = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tranche)


def production_function(name, namespace):
    tree = ast.parse((ROOT / "megatron/training/training.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), namespace)
    return namespace[name]


def arguments(**changes):
    defaults = dict(
        exit_on_tranche_end=True, tranche_start_samples=None,
        consumed_train_samples=0, consumed_valid_samples=0, iteration=0,
        phase_transition_iterations=None, global_batch_size=8, train_samples=1000,
        train_iters=125, eval_iters=0, eval_interval=100, full_validation=False,
        perform_rl_step=False, skip_train=False, multiple_validation_sets=False,
        save="checkpoints", save_interval=1, save_iters=None,
        load=None, split="100,0,0",
        non_persistent_save_interval=1, exit_signal_handler=False,
        exit_duration_in_mins=None, exit_interval=None,
    )
    defaults.update(changes)
    return SimpleNamespace(**defaults)


class Scalar(int):
    def item(self):
        return int(self)


class FiniteDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class TestTrancheEnd(unittest.TestCase):
    def test_marker_resume_and_next_tranche(self):
        with tempfile.TemporaryDirectory() as directory:
            first = tranche.tranche_identity(["/data/tranche_00/a"], "100,0,0")
            second = tranche.tranche_identity(["/data/tranche_01/a"], "100,0,0")
            tranche.write_tranche_end(directory, first, 0, 24, 3)
            record = json.loads((Path(directory) / tranche.MARKER_NAME).read_text())
            self.assertEqual(record["end_consumed_samples"], 24)
            # Same dataset remains finished; it must not start over at sample zero.
            self.assertEqual(tranche.resolve_tranche_start(directory, first, 24), 0)
            self.assertEqual(tranche.resolve_tranche_start(directory, second, 24), 24)
            # Mid-tranche crash resumes from the same original start.
            self.assertEqual(tranche.resolve_tranche_start(directory, second, 32), 24)
            tranche.write_tranche_end(directory, second, 24, 48, 6)
            self.assertEqual(tranche.resolve_tranche_start(directory, second, 48), 24)
            self.assertEqual(list(Path(directory).glob(".tranche_end-*")), [])

    def test_marker_missing_override_and_older_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(tranche.resolve_tranche_start(directory, "a", 80, fallback_start=40), 40)
            tranche.write_tranche_end(directory, "a", 0, 24, 3)
            with self.assertRaises(ValueError):
                tranche.resolve_tranche_start(directory, "b", 16)
            self.assertEqual(tranche.resolve_tranche_start(directory, "b", 16, explicit_start=8), 8)
            # An older checkpoint from the SAME tranche is safe to replay.
            self.assertEqual(tranche.resolve_tranche_start(directory, "a", 16), 0)

    def test_bad_marker_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / tranche.MARKER_NAME
            path.write_text('{"version": 1, "start_consumed_samples": 24, "end_consumed_samples": 8}')
            with self.assertRaises(ValueError):
                tranche.resolve_tranche_start(directory, "a", 24)

    def test_marker_published_after_checkpoint_finalization(self):
        tree = ast.parse((ROOT / "megatron/training/training.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        calls = {n.func.id: n.lineno for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertLess(calls["maybe_finalize_async_save"], calls["write_tranche_end"])

    def test_command_line_flags(self):
        # Use the real dataclass-to-argparse factory, without package imports.
        modules = []
        for name, path in [
            ("tranche_training_config", "megatron/training/config/training_config.py"),
            ("tranche_argument_utils", "megatron/training/argument_utils.py"),
        ]:
            module_spec = importlib.util.spec_from_file_location(name, ROOT / path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[name] = module
            module_spec.loader.exec_module(module)
            modules.append(module)
        parser = argparse.ArgumentParser()
        modules[1].ArgumentGroupFactory(modules[0].TrainingConfig).build_group(parser, "training")
        defaults = parser.parse_args([])
        self.assertFalse(defaults.exit_on_tranche_end)
        self.assertIsNone(defaults.tranche_start_samples)
        opts = parser.parse_args(["--exit-on-tranche-end", "--tranche-start-samples", "1000"])
        self.assertTrue(opts.exit_on_tranche_end)
        self.assertEqual(opts.tranche_start_samples, 1000)

    def test_boundaries_and_phase_offset(self):
        self.assertEqual(tranche.tranche_end_samples(24, 0, 0, 8), 24)
        self.assertEqual(tranche.tranche_end_samples(27, 8, 8, 8), 24)
        self.assertEqual(tranche.tranche_end_samples(27, 8, 1008, 8), 1024)
        self.assertEqual(tranche.tranche_end_samples(7, 0, 1000, 8), 1000)
        self.assertEqual(tranche.tranche_end_samples(24, 24, 1024, 8), 1024)
        self.assertEqual(tranche.tranche_end_samples(24, 32, 1032, 8), 1032)
        # A batch-size change on resume aligns from the current data cursor.
        self.assertEqual(tranche.tranche_end_samples(27, 8, 8, 6), 26)

    def test_invalid_counters(self):
        for values in [(0, 0, 0, 8), (24, 0, 0, 0), (24, -1, 0, 8), (24, 8, 0, 8)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                tranche.tranche_end_samples(*values)

    def test_build_single_pass_without_changing_schedule(self):
        args = arguments()
        fn = production_function("get_train_valid_test_num_samples", {"get_args": lambda: args})
        self.assertIsNone(fn()[0])
        self.assertEqual((args.train_samples, args.train_iters), (1000, 125))
        args.exit_on_tranche_end = False
        self.assertEqual(fn()[0], 1000)

    def build_loaders(self, args, size=27, rank=0, broadcast_size=None):
        calls = []
        def loader(dataset, consumed):
            if dataset is None:
                return None
            self.assertLess(consumed, len(dataset))
            calls.append(consumed)
            return SimpleNamespace(dataset=dataset, consumed=consumed)

        def broadcast(flags, source):
            self.assertEqual(source, 0)
            if broadcast_size is not None:
                flags[:] = [Scalar(v) for v in [1, 0, 0, broadcast_size]]

        namespace = dict(
            get_args=lambda: args, print_rank_0=lambda *a: None,
            mpu=SimpleNamespace(get_tensor_model_parallel_rank=lambda: rank),
            torch=SimpleNamespace(long=None, tensor=lambda v, **kw: [Scalar(x) for x in v],
                                  distributed=SimpleNamespace(broadcast=broadcast)),
            build_train_valid_test_datasets=lambda p: (FiniteDataset(size), None, None),
            build_pretraining_data_loader=loader,
            tranche_end_samples=tranche.tranche_end_samples,
            get_blend_and_blend_per_split=lambda a: ((["/test/tranche_00/shard"], None), None),
            tranche_identity=tranche.tranche_identity,
            resolve_tranche_start=tranche.resolve_tranche_start,
        )
        fn = production_function("build_train_valid_test_data_loaders", namespace)
        fn(lambda _: None)
        return calls

    def test_resume_uses_existing_tranche_cursor(self):
        args = arguments(consumed_train_samples=8, iteration=1)
        self.assertEqual(self.build_loaders(args), [8])
        self.assertEqual(args.tranche_end_consumed_samples, 24)

    def test_next_tranche_offset_preserves_global_counter(self):
        args = arguments(consumed_train_samples=1000, iteration=125, tranche_start_samples=1000)
        self.assertEqual(self.build_loaders(args), [0])
        self.assertEqual(args.tranche_end_consumed_samples, 1024)
        self.assertEqual(args.consumed_train_samples, 1000)

    def test_loader_reads_marker_without_manual_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            tranche.write_tranche_end(directory, "previous-tranche", 0, 1000, 125)
            args = arguments(consumed_train_samples=1000, iteration=125, load=directory)
            self.assertEqual(self.build_loaders(args), [0])
            self.assertEqual(args.tranche_start_samples, 1000)
            self.assertEqual(args.tranche_end_consumed_samples, 1024)

    def test_phase_transition_offset(self):
        args = arguments(consumed_train_samples=88, iteration=11, phase_transition_iterations=[10])
        self.assertEqual(self.build_loaders(args), [8])
        self.assertEqual(args.tranche_end_consumed_samples, 104)

    def test_completed_resume_does_not_fail_sampler(self):
        args = arguments(consumed_train_samples=24, iteration=3)
        self.assertEqual(self.build_loaders(args, size=24), [0])
        self.assertTrue(tranche.tranche_finished(True, args.tranche_end_consumed_samples, 24))

    def test_tp_rank_without_dataset_receives_boundary(self):
        args = arguments(consumed_train_samples=8, iteration=1)
        self.assertEqual(self.build_loaders(args, rank=1, broadcast_size=27), [])
        self.assertEqual(args.tranche_end_consumed_samples, 24)

    def test_future_tranche_start_rejected(self):
        with self.assertRaises(ValueError):
            self.build_loaders(arguments(tranche_start_samples=1000))

    def test_one_persistent_checkpoint_at_boundary(self):
        args = arguments(consumed_train_samples=24, tranche_end_consumed_samples=24)
        saves, messages = [], []
        namespace = dict(
            get_args=lambda: args, get_timers=lambda: None,
            tranche_finished=tranche.tranche_finished,
            save_checkpoint_and_time=lambda *a, **kw: saves.append((a, kw)),
            print_datetime=messages.append,
        )
        fn = production_function("checkpoint_and_decide_exit", namespace)
        self.assertTrue(fn("model", "optimizer", "scheduler", 3, 42, "context", "iterator"))
        self.assertEqual(len(saves), 1)
        self.assertEqual(saves[0][0][0], 3)
        self.assertNotIn("non_persistent_ckpt", saves[0][1])
        self.assertEqual(saves[0][1]["train_data_iterator"], "iterator")
        self.assertIn("[TRANCHE-END]", messages[0])

    def test_no_fetch_when_already_complete(self):
        tree = ast.parse((ROOT / "megatron/training/training.py").read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        loop = next(n for n in fn.body if isinstance(n, ast.While))
        # Execute the actual pre-fetch exit guard. Reaching the rest of the
        # loop here stands for a data fetch that would exhaust the sampler.
        guard = loop.body[0]
        events = []
        namespace = dict(
            args=arguments(consumed_train_samples=24, tranche_end_consumed_samples=24),
            tranche_finished=tranche.tranche_finished,
            checkpoint_and_decide_exit=lambda *a: events.append("checkpoint") or True,
            model=None, optimizer=None, opt_param_scheduler=None, iteration=3,
            num_floating_point_operations_so_far=0, checkpointing_context=None,
            train_data_iterator=None,
        )
        while_node = ast.While(test=ast.Constant(True), body=[guard] + ast.parse(
            'raise AssertionError("Fetched past tranche boundary")'
        ).body, orelse=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[while_node], type_ignores=[])),
                     "pre-fetch-exit", "exec"), namespace)
        self.assertEqual(events, ["checkpoint"])

    def test_disabled_mode_ignores_stale_boundary(self):
        self.assertFalse(tranche.tranche_finished(False, 24, 24))
        self.assertFalse(tranche.tranche_finished(True, None, 24))
        self.assertFalse(tranche.tranche_finished(True, 24, 16))


if __name__ == "__main__":
    unittest.main()
