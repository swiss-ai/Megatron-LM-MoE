"""CPU execution of production batch/loss functions with hardware calls replaced.

Loading function ASTs avoids importing optional model/GPU dependencies. Broadcasts
are recorded and replayed to check source/receiver ordering, shapes, and dtypes;
real distributed execution is covered separately in test_get_batch_pp.py.
"""

import ast
import argparse
from collections import deque
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from megatron.core import utils as core_utils
from megatron.core.packed_seq_params import PackedSeqParams

ROOT = Path(__file__).resolve().parents[3]


def load_function(path, name, namespace):
    namespace.setdefault('Optional', Optional)
    tree = ast.parse((ROOT / path).read_text())
    node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("pp,stage", [(1, 0), (3, 0), (3, 1), (3, 2)])
@pytest.mark.parametrize("ap_sft", [False, True])
@pytest.mark.parametrize('report_assistant', [False, True])
def test_tp_collective_contract_and_existing_batch_keys(
    monkeypatch, pp, stage, ap_sft, report_assistant
):
    args = SimpleNamespace(
        sft=False,
        ap_sft=ap_sft,
        dataloader_inter_document_masking=False,
        ap_sft_report_assistant_loss=ap_sft and report_assistant,
        hybrid_context_parallel=False,
        pipeline_model_parallel_size=pp,
        micro_batch_size=2,
        seq_length=8,
        create_attention_mask_in_dataloader=False,
    )
    state = {"tp": 0}
    mpu = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: state["tp"],
        get_tensor_model_parallel_world_size=lambda: 2,
        get_tensor_model_parallel_src_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
        is_pipeline_first_stage=lambda: stage == 0,
        is_pipeline_last_stage=lambda: stage == pp - 1,
    )
    messages = deque()

    def broadcast(tensor, *args, **kwargs):
        if state["tp"] == 0:
            messages.append(tensor.clone())
        else:
            source = messages.popleft()
            assert source.shape == tensor.shape
            assert source.dtype == tensor.dtype
            tensor.copy_(source)

    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, **_: tensor)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    get_batch = load_function(
        "megatron/training/utils.py",
        "get_batch_on_this_tp_rank",
        {"torch": torch, "mpu": mpu, "get_args": lambda: args},
    )
    raw = {
        "tokens": torch.arange(16).reshape(2, 8),
        "labels": torch.arange(16).reshape(2, 8),
        "loss_mask": torch.ones(2, 8),
        "position_ids": torch.arange(8).repeat(2, 1),
    }
    if ap_sft:
        raw.update(
            cu_seqlens=torch.tensor([[0, 4, 8], [0, 8, 8]], dtype=torch.int32),
            cu_seqlens_padded=torch.tensor([[0, 4, 8], [0, 8, 8]], dtype=torch.int32),
            max_seqlen=torch.tensor([4, 8], dtype=torch.int32),
            padding_mask=torch.tensor([[False] * 6 + [True] * 2] * 2),
            assistant_mask=torch.tensor([[False, True, True, False] * 2] * 2),
        )
    get_batch(iter([raw]))
    state["tp"] = 1
    received = get_batch(None)
    assert not messages, "receiver skipped a source broadcast"
    if ap_sft:
        assert torch.equal(received["cu_seqlens_padded"], raw["cu_seqlens_padded"])
        assert torch.equal(received["padding_mask"], raw["padding_mask"])
        if report_assistant:
            assert torch.equal(received['assistant_mask'], raw['assistant_mask'])
        else:
            assert 'assistant_mask' not in received
    else:
        assert "padding_mask" not in received
        assert "cu_seqlens_padded" not in received
        assert 'assistant_mask' not in received


@pytest.mark.parametrize("stage", ["first", "middle", "last"])
def test_cp_slices_available_tensors_and_padding_on_every_pp_stage(monkeypatch, stage):
    monkeypatch.setattr(core_utils, "is_te_min_version", lambda _: True)
    indices = torch.tensor([0, 3, 4, 7, 8, 9, 14, 15])

    def partition(cu, length, cp, rank):
        assert cu.ndim == 1 and cu.tolist() == [0, 4, 8, 16]
        assert (length, cp, rank) == (16, 2, 0)
        return indices

    monkeypatch.setattr(core_utils, "tex", SimpleNamespace(thd_get_partitioned_indices=partition))
    args = SimpleNamespace(
        sft=False,
        ap_sft=True,
        dataloader_inter_document_masking=False,
        pretraining_packing_strategy="greedy",
    )
    padding = torch.arange(16).reshape(2, 8) % 3 == 0
    raw = {
        "tokens": torch.arange(16).reshape(2, 8) if stage == "first" else None,
        "labels": torch.arange(16).reshape(2, 8) if stage == "last" else None,
        "loss_mask": torch.ones(2, 8) if stage == "last" else None,
        "position_ids": torch.zeros(2, 8, dtype=torch.long) if stage == "first" else None,
        "attention_mask": None,
        "padding_mask": padding,
        "assistant_mask": ~padding if stage == 'last' else None,
        "cu_seqlens": torch.tensor([[0, 4, 8], [0, 8, 8]], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([[0, 4, 8], [0, 8, 8]], dtype=torch.int32),
        "max_seqlen": torch.tensor([4, 8], dtype=torch.int32),
        "local_cp_size": None,
    }
    get_batch = load_function(
        "pretrain_gpt.py",
        "get_batch",
        {
            "get_args": lambda: args,
            "core_transformer_config_from_args": lambda _: None,
            "mtp_on_this_rank": lambda *args, **kwargs: False,
            "is_first_or_last_pipeline_stage": lambda _: stage != "middle",
            "get_batch_on_this_tp_rank": lambda *args, **kwargs: raw,
            "flatten_batch_for_packed_sequences": core_utils.flatten_batch_for_packed_sequences,
            "get_thd_batch_on_this_cp_rank": lambda *args: core_utils.get_thd_batch_on_this_cp_rank(
                *args, cp_size=2, cp_rank=0
            ),
            "PackedSeqParams": PackedSeqParams,
        },
    )
    result = get_batch(None)
    assert len(result) == 8
    assert torch.equal(result[5], padding.reshape(1, -1).index_select(1, indices))
    assert result[6].cu_seqlens_q.ndim == 1
    assert result[6].cu_seqlens_q_padded.ndim == 1
    assert result[6].cu_seqlens_q[-1] == 16
    if stage == "middle":
        assert all(value is None for value in result[:5])
    elif stage == "last":
        assert result[0] is None and result[1].shape == (1, 8)
        assert torch.equal(result[7], (~padding).reshape(1, -1).index_select(1, indices))
    else:
        assert result[0].shape == (1, 8)


def test_packed_pipeline_shape_includes_apertus_sft():
    shape = load_function("megatron/training/training.py", "_pipeline_shape_args", {})
    args = SimpleNamespace(ap_sft=True, seq_length=16, micro_batch_size=3)
    assert shape(args) == (48, 1)
    args.ap_sft = False
    assert shape(args) == (16, 3)


@pytest.mark.parametrize(
    "unsupported",
    [
        None,
        'use_legacy_models',
        'dataloader_fast_cache_load',
        'dataloader_defer_npy_index_mmap',
        'hybrid_context_parallel',
        'mtp_num_layers',
        'modelopt_enabled',
        'fim_data',
        'goldfish_loss',
        'dataloader_inter_document_masking',
        'mock_data',
    ],
)
def test_apertus_validation_guards_use_real_cli_names(unsupported):
    add_args = load_function('megatron/training/arguments.py', '_add_sft_args', {})
    args = add_args(argparse.ArgumentParser()).parse_args(['--ap-sft'])
    args.calculate_per_token_loss = True
    args.create_attention_mask_in_dataloader = True
    if unsupported:
        setattr(args, unsupported, True)
    tree = ast.parse((ROOT / 'megatron/training/arguments.py').read_text())
    validate = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == 'validate_args'
    )
    guard = next(
        node
        for node in validate.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "getattr(args, 'ap_sft', False)"
    )
    code = compile(ast.Module(body=[guard], type_ignores=[]), 'ap_sft_validation', 'exec')
    if unsupported:
        with pytest.raises(AssertionError, match=unsupported):
            exec(code, {'args': args})
    else:
        exec(code, {'args': args})
        assert not args.create_attention_mask_in_dataloader


@pytest.mark.parametrize('ap_sft,packing', [(False, 'greedy'), (False, 'bfd'), (True, 'greedy')])
@pytest.mark.parametrize('graph_impl', ['none', 'local', 'transformer_engine'])
@pytest.mark.parametrize('warmup', [0, 1])
def test_padding_graph_warmup_guard(ap_sft, packing, graph_impl, warmup):
    args = SimpleNamespace(
        ap_sft=ap_sft,
        pretraining_packing_strategy=packing,
        cuda_graph_impl=graph_impl,
        transformer_impl='transformer_engine',
        te_rng_tracker=True,
        cuda_graph_warmup_steps=warmup,
    )
    tree = ast.parse((ROOT / 'megatron/training/arguments.py').read_text())
    validate = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == 'validate_args'
    )
    guard = next(
        node
        for node in validate.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "args.cuda_graph_impl != 'none'"
    )
    code = compile(ast.Module(body=[guard], type_ignores=[]), 'cuda_graph_validation', 'exec')
    namespace = {'args': args, 'os': SimpleNamespace(getenv=lambda *args: '')}
    if graph_impl == 'transformer_engine' and (ap_sft or packing == 'bfd') and warmup == 0:
        with pytest.raises(AssertionError, match='--cuda-graph-warmup-steps must be > 0'):
            exec(code, namespace)
    else:
        exec(code, namespace)


@pytest.mark.parametrize("weights", [[0.25, 2, 0, 0.5], [0, 0, 0, 0]])
def test_weighted_loss_counts_positive_targets_and_backpropagates(weights):
    args = SimpleNamespace(
        ap_sft=True, check_for_nan_in_loss_and_grad=False, check_for_spiky_loss=False
    )
    loss_func = load_function(
        "pretrain_gpt.py",
        "loss_func",
        {
            "torch": torch,
            "Optional": Optional,
            "GPTModel": object,
            "get_args": lambda: args,
            "has_nvidia_modelopt": False,
            "get_rerun_state_machine": lambda: None,
        },
    )
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    weights = torch.tensor(weights)
    loss, count, report = loss_func(weights, losses)
    assert count == (weights > 0).sum()
    assert loss == (weights * losses).sum()
    assert report["lm loss"][1] == count
    (loss / count.clamp_min(1)).backward()
    torch.testing.assert_close(losses.grad, weights / count.clamp_min(1))


@pytest.mark.parametrize('assistant', [[True, True, False, True], [False] * 4])
def test_assistant_loss_is_unweighted_detached_and_does_not_change_gradients(assistant):
    args = SimpleNamespace(
        ap_sft=True, check_for_nan_in_loss_and_grad=False, check_for_spiky_loss=False
    )
    loss_func = load_function(
        'pretrain_gpt.py',
        'loss_func',
        {
            'torch': torch,
            'GPTModel': object,
            'get_args': lambda: args,
            'has_nvidia_modelopt': False,
            'get_rerun_state_machine': lambda: None,
        },
    )
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    weights = torch.tensor([0.0, 0.25, 2.0, 0.5])
    mask = torch.tensor(assistant)
    loss, count, report = loss_func(weights, losses, assistant_mask=mask)
    pair = report['assistant_loss']
    assert pair[0] == losses[mask].sum()
    assert pair[1] == mask.sum()
    assert not pair.requires_grad
    assert torch.isfinite(pair[0] / pair[1].clamp_min(1))
    (loss / count.clamp_min(1)).backward()
    torch.testing.assert_close(losses.grad, weights / count)


def test_assistant_loss_aggregates_sums_and_counts_across_partitions():
    args = SimpleNamespace(
        ap_sft=True, check_for_nan_in_loss_and_grad=False, check_for_spiky_loss=False
    )
    loss_func = load_function(
        'pretrain_gpt.py',
        'loss_func',
        {
            'torch': torch,
            'GPTModel': object,
            'get_args': lambda: args,
            'has_nvidia_modelopt': False,
            'get_rerun_state_machine': lambda: None,
        },
    )
    pairs = []
    for losses, mask in [([1.0, 2.0, 9.0], [True, True, False]), ([6.0], [True])]:
        _, _, report = loss_func(
            torch.zeros(len(losses)), torch.tensor(losses), assistant_mask=torch.tensor(mask)
        )
        pairs.append(report['assistant_loss'])
    total = torch.stack(pairs).sum(0)
    assert total.tolist() == [9.0, 3.0]
    assert total[0] / total[1] == 3


@pytest.mark.parametrize('schedule_plan', [False, True])
def test_forward_step_passes_assistant_mask_to_reporting_callback(schedule_plan):
    args = SimpleNamespace(
        ap_sft=True,
        use_legacy_models=False,
        overlap_moe_expert_parallel_comm=True,
        check_for_nan_in_loss_and_grad=False,
        check_for_spiky_loss=False,
    )
    namespace = {
        'torch': torch,
        'GPTModel': object,
        'get_args': lambda: args,
        'has_nvidia_modelopt': False,
        'get_rerun_state_machine': lambda: None,
    }
    loss_func = load_function('pretrain_gpt.py', 'loss_func', namespace)
    mask = torch.tensor([True, False])
    losses = torch.tensor([2.0, 9.0])
    batch = (None, None, torch.tensor([0.25, 0.0]), None, None, None, None, mask)

    class Model:
        def __call__(self, *args, **kwargs):
            assert 'assistant_mask' not in kwargs
            return losses

        build_schedule_plan = __call__

    class Tracker(nullcontext):
        def __call__(self, **kwargs):
            return nullcontext()

    forward = load_function(
        'pretrain_gpt.py',
        'forward_step',
        {
            'GPTModel': object,
            'get_args': lambda: args,
            'get_timers': lambda: lambda *args, **kwargs: SimpleNamespace(
                start=lambda **kwargs: None, stop=lambda **kwargs: None
            ),
            'get_attr_wrapped_model': lambda *args: None,
            'stimer': Tracker(),
            'get_batch': lambda *args: batch,
            'partial': partial,
            'loss_func': loss_func,
        },
    )
    output, callback = forward(None, Model(), return_schedule_plan=schedule_plan)
    loss, count, report = callback(output)
    assert loss == 0.5 and count == 1
    assert report['assistant_loss'].tolist() == [2.0, 1.0]
