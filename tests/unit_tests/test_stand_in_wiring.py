"""CPU checks of the data-loader wiring for the middle-pipeline-stage stand-in datasets.

Executes the production build_train_valid_test_data_loaders with the distributed pieces stubbed (the pattern of
test_tranche_end.py): with BFD packing, or greedy packing with inter-document masking, a rank that built no datasets
must get one stand-in per saved split, with the split and length of the dataset it stands in for, the document-length
paths and the sequence length, and the same loader arguments as the first stage; every rank must take part in the
broadcast; nothing happens otherwise; a save error recorded by rank 0 raises on every rank. The dataset provider of
pretrain_gpt.py saves one array set per split (train, valid or valid<i>, test) on rank 0 only, with the gate on only, and records
instead of raising when a save fails. No CUDA or Megatron imports are required.
"""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def production_function(name, namespace, path="megatron/training/training.py"):
    return production_functions([name], namespace, path)[name]


def production_functions(names, namespace, path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


class Split:
    train, valid, test = "train", "valid", "test"


class Scalar(int):
    def item(self):
        return int(self)


class StandIn:
    def __init__(self, path, length, split, doc_paths=None, seq_length=None):
        self.path, self.length, self.index_split = path, length, split
        self.doc_paths, self.seq_length = doc_paths, seq_length

    def __len__(self):
        return self.length


class Real:
    def __init__(self, length, split):
        self.length, self.index_split = length, split

    def __len__(self):
        return self.length


def arguments(**changes):
    defaults = dict(
        exit_on_tranche_end=False, tranche_start_samples=None, consumed_train_samples=16, consumed_valid_samples=4,
        iteration=2, phase_transition_iterations=None, global_batch_size=8, train_samples=None, train_iters=10,
        eval_iters=1, eval_interval=5, full_validation=False, perform_rl_step=False, skip_train=False,
        multiple_validation_sets=False, pretraining_packing_strategy="bfd", dataloader_inter_document_masking=False,
        seq_length=512, load=None, save="checkpoints", split="99,1,0",
    )
    defaults.update(changes)
    return SimpleNamespace(**defaults)


def run(args, built, saved, doc_paths=None, tp_rank=0):
    """built: what the dataset provider returns on this rank; saved / doc_paths: what rank 0 saved.

    Returns the loader arguments (class, split, length, consumed samples, doc paths, seq length), the broadcasts
    (source rank, payload) and the flags tensors built. Rank 0 broadcasts the objects the function puts in the list
    (its module globals, pre-filled here) and every rank receives exactly those, so the broadcast is a pass-through.
    """
    loaders, broadcasts, flags = [], [], []

    def broadcast_object_list(objects, src):
        broadcasts.append((src, [dict(o) for o in objects]))

    def loader(dataset, consumed):
        if dataset is None:
            return None
        loaders.append((type(dataset).__name__, dataset.index_split, len(dataset), consumed,
                        getattr(dataset, "doc_paths", "n/a"), getattr(dataset, "seq_length", "n/a")))
        return [dataset]

    def tensor(values, **kwargs):
        flags.append(list(values))
        return [Scalar(v) for v in values]

    namespace = dict(
        get_args=lambda: args, print_rank_0=lambda *x: None,
        mpu=SimpleNamespace(get_tensor_model_parallel_rank=lambda: tp_rank),
        torch=SimpleNamespace(long=None, tensor=tensor, distributed=SimpleNamespace(
            broadcast=lambda *x: None, broadcast_object_list=broadcast_object_list)),
        build_train_valid_test_datasets=lambda provider: built, build_pretraining_data_loader=loader,
        NUM_VALID_TOKENS_PATHS=dict(saved), DOC_LENGTHS_PATHS=dict(doc_paths or {}),
        NumValidTokensDataset=StandIn, Split=Split, SAVE_ERROR_KEY="__error__",
        get_blend_and_blend_per_split=lambda args: ((["prefix"], None), None), tranche_identity=lambda *x: "tranche",
        resolve_tranche_start=lambda *x, **k: 0, tranche_end_samples=lambda size, consumed, total, gbs: size,
    )
    provider = lambda sizes: None
    provider.is_distributed = True
    production_function("build_train_valid_test_data_loaders", namespace)(provider)
    return loaders, broadcasts, flags


SAVED = {"train": ("t.npy", 80), "valid": ("v.npy", 24), "test": ("s.npy", 8)}
DOC = {"train": ("to.npy", "tl.npy"), "valid": ("vo.npy", "vl.npy"), "test": ("so.npy", "sl.npy")}
REAL = (Real(80, Split.train), Real(24, Split.valid), Real(8, Split.test))
NONE = (None, None, None)


class TestStandInWiring(unittest.TestCase):
    def test_bfd_without_masking_stand_ins_have_no_doc_paths(self):
        middle, broadcasts, _ = run(arguments(), NONE, SAVED)
        first, _, _ = run(arguments(), REAL, SAVED)
        self.assertEqual([b[0] for b in broadcasts], [0])
        self.assertEqual([x[0] for x in middle], ["StandIn"] * 3)
        self.assertEqual([x[1:4] for x in middle], [x[1:4] for x in first])
        self.assertEqual([x[4] for x in middle], [None] * 3)

    def test_bfd_with_masking_stand_ins_carry_doc_paths_and_seq_length(self):
        args = arguments(dataloader_inter_document_masking=True)
        middle, broadcasts, _ = run(args, NONE, SAVED, DOC)
        self.assertEqual(broadcasts[0][1], [SAVED, DOC], "rank 0 must broadcast both path dicts")
        self.assertEqual([x[:3] for x in middle],
                         [("StandIn", "train", 80), ("StandIn", "valid", 24), ("StandIn", "test", 8)])
        self.assertEqual([x[4] for x in middle], [DOC["train"], DOC["valid"], DOC["test"]])
        self.assertEqual([x[5] for x in middle], [512] * 3)

    def test_greedy_with_masking_broadcasts_and_builds_stand_ins(self):
        args = arguments(pretraining_packing_strategy="greedy", dataloader_inter_document_masking=True)
        saved = {key: (None, length) for key, (_, length) in SAVED.items()}
        middle, broadcasts, _ = run(args, NONE, saved, DOC)
        first, first_broadcasts, _ = run(args, REAL, saved, DOC)
        self.assertEqual(([b[0] for b in broadcasts], [b[0] for b in first_broadcasts]), ([0], [0]))
        self.assertEqual([x[0] for x in middle], ["StandIn"] * 3)
        self.assertEqual([x[1:4] for x in middle], [x[1:4] for x in first])
        self.assertEqual([x[4] for x in middle], [DOC["train"], DOC["valid"], DOC["test"]])
        self.assertEqual([x[0] for x in first], ["Real"] * 3)

    def test_greedy_without_masking_does_nothing(self):
        loaders, broadcasts, _ = run(arguments(pretraining_packing_strategy="greedy"), NONE, SAVED, DOC)
        self.assertEqual((loaders, broadcasts), ([], []))

    def test_other_tp_ranks_take_part_in_the_broadcast_and_get_nothing(self):
        loaders, broadcasts, _ = run(arguments(dataloader_inter_document_masking=True), NONE, SAVED, DOC, tp_rank=1)
        self.assertEqual(loaders, [])
        self.assertEqual([b[0] for b in broadcasts], [0])

    def test_skip_train(self):
        middle, _, _ = run(arguments(dataloader_inter_document_masking=True, skip_train=True), NONE, SAVED, DOC)
        self.assertEqual([x[1] for x in middle], ["valid", "test"])
        self.assertEqual([x[3] for x in middle], [0, 0])

    def test_multiple_validation_sets_full_validation(self):
        args = arguments(multiple_validation_sets=True, full_validation=True, dataloader_inter_document_masking=True)
        saved = {"train": ("t", 80), "valid0": ("v0", 24), "valid1": ("v1", 40), "test": ("s", 8)}
        doc = {key: (key + "o", key + "l") for key in saved}
        middle, _, _ = run(args, (None, [None, None], None), saved, doc)
        first, _, _ = run(args, (Real(80, Split.train), [Real(24, Split.valid), Real(40, Split.valid)],
                                 Real(8, Split.test)), saved, doc)
        self.assertEqual([x[1:4] for x in middle], [x[1:4] for x in first])
        self.assertEqual([x[4] for x in middle], [doc["train"], doc["valid0"], doc["valid1"], doc["test"]])

    def test_tranche_size_from_the_stand_in_equals_the_first_stage(self):
        args = arguments(dataloader_inter_document_masking=True, exit_on_tranche_end=True)
        _, _, middle_flags = run(args, NONE, SAVED, DOC)
        _, _, first_flags = run(args, REAL, SAVED, DOC)
        self.assertEqual(middle_flags[-1], first_flags[-1])
        self.assertEqual(middle_flags[-1][3], 80)

    def test_built_dataset_is_never_replaced(self):
        saved = {"train": ("t", 99), "valid": ("v", 99), "test": ("s", 99)}
        loaders, _, _ = run(arguments(dataloader_inter_document_masking=True), REAL, saved, DOC)
        self.assertEqual([x[0] for x in loaders], ["Real"] * 3)

    def test_recorded_save_error_raises_on_every_rank(self):
        saved = {"__error__": "PermissionError: [Errno 13] Permission denied: 'cache/x.tmp'"}
        for tp_rank in (0, 1):
            with self.assertRaises(RuntimeError) as cm:
                run(arguments(dataloader_inter_document_masking=True), NONE, saved, {}, tp_rank=tp_rank)
            self.assertIn("Permission denied", str(cm.exception))
        with self.assertRaises(RuntimeError):
            run(arguments(dataloader_inter_document_masking=True), REAL, saved, {})


PROVIDER = {"bfd_middle_ranks_read_num_valid_tokens", "greedy_middle_ranks_read_doc_lengths", "train_valid_test_datasets_provider"}


def provide(args, datasets, rank=0, fail=None):
    """Runs the production train_valid_test_datasets_provider with a stub builder that returns `datasets` and stub save
    functions. Returns (save calls as (function, key, dataset), NUM_VALID_TOKENS_PATHS afterwards)."""
    import functools
    calls, nvt_paths, doc_paths = [], {"stale": ("old.npy", 1)}, {"stale": ("o", "l")}

    def saver(name):
        def save(ds, key):
            if fail is not None:
                raise fail
            calls.append((name, key, ds))
            nvt_paths[key] = (name, len(ds) if hasattr(ds, "__len__") else None)
        return save

    class Builder:
        def __init__(self, cls, sizes, is_built, config):
            pass

        def build(self):
            return list(datasets)

    namespace = dict(
        get_args=lambda: args, core_gpt_dataset_config_from_args=lambda args: SimpleNamespace(blend=None, blend_per_split=None),
        SFTDataset="sft", MockGPTDataset="mock", GPTFIMDataset="fim", GPTDataset="gpt", print_rank_0=lambda *x: None,
        partial=functools.partial, is_dataset_built_on_rank=lambda **k: True, BlendedMegatronDatasetBuilder=Builder,
        torch=SimpleNamespace(distributed=SimpleNamespace(get_rank=lambda: rank)),
        NUM_VALID_TOKENS_PATHS=nvt_paths, DOC_LENGTHS_PATHS=doc_paths, SAVE_ERROR_KEY="__error__",
        save_num_valid_tokens=saver("nvt"), save_greedy_document_lengths=saver("greedy"),
    )
    production_functions(PROVIDER, namespace, "pretrain_gpt.py")
    out = namespace["train_valid_test_datasets_provider"]([1, 1, 1])
    assert tuple(out) == tuple(datasets), "the provider returns the built datasets unchanged"
    return calls, nvt_paths


def provider_arguments(**changes):
    defaults = dict(pretraining_packing_strategy="bfd", dataloader_inter_document_masking=True, pipeline_model_parallel_size=4,
                    sft=False, mock_data=False, fim_data=False, exit_on_tranche_end=False)
    defaults.update(changes)
    return SimpleNamespace(**defaults)


class TestProviderSaves(unittest.TestCase):
    def test_one_array_set_per_split_in_key_order(self):
        t, v, s = Real(80, Split.train), Real(24, Split.valid), Real(8, Split.test)
        calls, paths = provide(provider_arguments(), (t, v, s))
        self.assertEqual(calls, [("nvt", "train", t), ("nvt", "valid", v), ("nvt", "test", s)])
        self.assertNotIn("stale", paths, "the dicts are cleared before saving")

    def test_multiple_validation_sets_get_valid_i_keys(self):
        t, v0, v1, s = Real(80, Split.train), Real(24, Split.valid), Real(40, Split.valid), Real(8, Split.test)
        calls, _ = provide(provider_arguments(), (t, [v0, v1], s))
        self.assertEqual([c[1:] for c in calls], [("train", t), ("valid0", v0), ("valid1", v1), ("test", s)])

    def test_missing_splits_are_skipped_and_greedy_uses_its_saver(self):
        t = Real(80, Split.train)
        calls, _ = provide(provider_arguments(pretraining_packing_strategy="greedy"), (t, None, None))
        self.assertEqual(calls, [("greedy", "train", t)])

    def test_nothing_saved_on_other_ranks_or_with_the_gate_off(self):
        t, v, s = Real(80, Split.train), Real(24, Split.valid), Real(8, Split.test)
        self.assertEqual(provide(provider_arguments(), (t, v, s), rank=1)[0], [])
        self.assertEqual(provide(provider_arguments(pipeline_model_parallel_size=2), (t, v, s))[0], [])
        self.assertEqual(provide(provider_arguments(pretraining_packing_strategy="greedy", dataloader_inter_document_masking=False), (t, v, s))[0], [])
        self.assertEqual(provide(provider_arguments(sft=True), (t, v, s))[0], [])

    def test_a_failed_save_is_recorded_instead_of_raised(self):
        t, v, s = Real(80, Split.train), Real(24, Split.valid), Real(8, Split.test)
        calls, paths = provide(provider_arguments(), (t, v, s), fail=PermissionError(13, "Permission denied", "cache/x.tmp"))
        self.assertEqual(calls, [])
        self.assertEqual(set(paths), {"__error__"})
        self.assertIn("PermissionError", paths["__error__"])


if __name__ == "__main__":
    unittest.main()
