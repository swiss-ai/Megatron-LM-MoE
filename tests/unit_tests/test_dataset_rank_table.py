"""CPU check of which (pipeline rank, virtual stage, tensor-parallel rank) chunks build datasets.

Executes the production predicates of pretrain_gpt.py (bfd_middle_ranks_read_num_valid_tokens,
greedy_middle_ranks_read_doc_lengths, is_dataset_built_on_rank) with parallel_state stubbed, over every
TP / PP / VPP / MTP x packing x masking x data-type combination, against the rule of get_batch: on tensor-parallel
rank 0, a chunk needs a data iterator when it is a first or last chunk, or the sequences are packed, or it holds MTP
layers, or the packing is BFD. With the middle-stage stand-in enabled, exactly the physical first and last pipeline
ranks and the MTP chunks build datasets; the other chunks that need an iterator get a stand-in.
"""
import ast
import itertools
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PREDICATES = {"bfd_middle_ranks_read_num_valid_tokens", "greedy_middle_ranks_read_doc_lengths", "is_dataset_built_on_rank"}


def production_predicates():
    tree = ast.parse((ROOT / "pretrain_gpt.py").read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in PREDICATES]
    assert {n.name for n in nodes} == PREDICATES
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pretrain_gpt_predicates", "exec"), namespace)
    return namespace


def rank_table(pp, vpp, tp, packing, masking, sft=False, mock=False, fim=False, mtp_middle=False):
    """Returns {(pp_rank, vp_stage, tp_rank): (builds a dataset, needs an iterator)} and whether the gate is on."""
    namespace = production_predicates()
    args = SimpleNamespace(pretraining_packing_strategy=packing, pipeline_model_parallel_size=pp,
                           dataloader_inter_document_masking=masking, sft=sft, mock_data=mock, fim_data=fim)
    state = {}
    stages = [None] if vpp is None else list(range(vpp))

    def first(ignore_virtual=True, vp_stage=None):
        if not ignore_virtual and vpp is not None and vp_stage != 0:
            return False
        return state["pp"] == 0

    def last(ignore_virtual=True, vp_stage=None):
        if not ignore_virtual and vpp is not None and vp_stage != vpp - 1:
            return False
        return state["pp"] == pp - 1

    def first_or_last(vp_stage):
        ignore_virtual = vp_stage is None
        return first(ignore_virtual, vp_stage) or last(ignore_virtual, vp_stage)

    def mtp(config, ignore_virtual=True, vp_stage=None):
        # A custom layout with the MTP layers in the last chunk of pipeline rank 1.
        return mtp_middle and state["pp"] == 1 and (vp_stage is None or vp_stage == stages[-1])

    namespace.update(
        get_args=lambda: args,
        core_transformer_config_from_args=lambda args: SimpleNamespace(pipeline_model_parallel_layout=None),
        parallel_state=SimpleNamespace(get_tensor_model_parallel_rank=lambda: state["tp"],
                                       is_pipeline_first_stage=first, is_pipeline_last_stage=last),
        is_first_or_last_pipeline_stage=first_or_last, mtp_on_this_rank=mtp,
    )
    packed = sft or masking
    table = {}
    for p, v, t in itertools.product(range(pp), stages, range(tp)):
        state.update(pp=p, tp=t)
        builds = bool(namespace["is_dataset_built_on_rank"](vp_stage=v, is_packed_sequence=packed))
        needs = t == 0 and (first_or_last(v) or packed or mtp(None, False, v) or packing == "bfd")
        table[(p, v, t)] = (builds, needs)
    gate = bool(namespace["bfd_middle_ranks_read_num_valid_tokens"](args)
                or namespace["greedy_middle_ranks_read_doc_lengths"](args))
    return table, gate


GRID = list(itertools.product((1, 2, 3, 4, 8), (None, 2), (1, 2), ("greedy", "bfd"), (False, True)))


class TestDatasetRankTable(unittest.TestCase):
    def test_every_chunk_that_needs_an_iterator_has_one_and_no_chunk_builds_in_vain(self):
        for pp, vpp, tp, packing, masking in GRID:
            for extra in ({}, {"sft": True}, {"mock": True}, {"fim": True}, {"mtp_middle": True}):
                table, gate = rank_table(pp, vpp, tp, packing, masking, **extra)
                for (p, v, t), (builds, needs) in table.items():
                    # With the gate on, a tensor-parallel rank 0 chunk without a dataset gets a stand-in.
                    has_iterator = builds or (gate and t == 0)
                    self.assertTrue(has_iterator or not needs, (pp, vpp, tp, packing, masking, extra, p, v, t))
                    self.assertTrue(needs or not builds, (pp, vpp, tp, packing, masking, extra, p, v, t))

    def test_gate_and_the_ranks_that_build_under_it(self):
        for pp, vpp, tp, packing, masking in GRID:
            table, gate = rank_table(pp, vpp, tp, packing, masking)
            self.assertEqual(gate, pp > 2 and (packing == "bfd" or (packing == "greedy" and masking)),
                             (pp, vpp, tp, packing, masking))
            for (p, v, t), (builds, needs) in table.items():
                if gate:
                    self.assertEqual(builds, t == 0 and p in (0, pp - 1), (pp, vpp, tp, packing, masking, p, v, t))
                else:
                    self.assertEqual(builds, needs, (pp, vpp, tp, packing, masking, p, v, t))
            if gate:
                table, _ = rank_table(pp, vpp, tp, packing, masking, mtp_middle=True)
                for (p, v, t), (builds, needs) in table.items():
                    if p == 1 and t == 0 and (v is None or v == (vpp or 1) - 1):
                        self.assertTrue(builds, ("an MTP chunk on a middle rank needs real data", pp, vpp, tp, packing, masking, v))

    def test_sft_mock_and_fim_data_are_unchanged(self):
        for pp, vpp, tp, packing, masking in GRID:
            for extra in ({"sft": True}, {"mock": True}, {"fim": True}):
                table, gate = rank_table(pp, vpp, tp, packing, masking, **extra)
                self.assertFalse(gate)
                for (p, v, t), (builds, needs) in table.items():
                    self.assertEqual(builds, needs, (pp, vpp, tp, packing, masking, extra, p, v, t))


if __name__ == "__main__":
    unittest.main()
