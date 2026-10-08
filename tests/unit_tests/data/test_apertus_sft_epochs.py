"""Exact epoch coverage through indexed data, the production sampler and provider."""

import ast
import copy
import importlib.util
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from tests.unit_tests.data.test_apertus_sft_batch import ROOT, load_function
from tests.unit_tests.data.test_apertus_sft_dataset import (
    ApertusSFTDataset,
    Tokenizer,
    make_dataset,
    write_index,
)

spec = importlib.util.spec_from_file_location(
    'apertus_sft_epochs_cpu', ROOT / 'megatron/training/datasets/apertus_sft_epochs.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
EpochDataset = module.ApertusSFTEpochDataset

tree = ast.parse((ROOT / 'megatron/training/datasets/data_samplers.py').read_text())
sampler_class = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == 'MegatronPretrainingSampler'
)
namespace = {}
exec(compile(ast.Module(body=[sampler_class], type_ignores=[]), 'sampler', 'exec'), namespace)
Sampler = namespace['MegatronPretrainingSampler']


@pytest.mark.parametrize('packing', [None, 'greedy', 'bfd'])
@pytest.mark.parametrize('global_batch,dp,micro', [(8, 2, 2), (4, 1, 1), (16, 4, 1)])
@pytest.mark.parametrize('stored', [False, True])
def test_exact_epochs_cover_conversations_once_and_resume(
    tmp_path, packing, global_batch, dp, micro, stored
):
    rows = [[i, 90, i, 91, 92] for i in range(1, 12)]
    weights = [[0] * 5] + [[0, 0.25, 0, 2, 0] for _ in rows[1:]] if stored else None
    source = make_dataset(
        tmp_path,
        rows,
        weights,
        sft_pack_samples=packing is not None,
        sft_packing_strategy=packing or 'greedy',
        sft_report_assistant_loss=True,
    )
    dataset = EpochDataset(source, 3, global_batch, seed=42)
    rebuilt = EpochDataset(source, 3, global_batch, seed=42)
    assert np.array_equal(dataset.order, rebuilt.order)
    for epoch in range(3):
        start = epoch * dataset.samples_per_epoch
        seen = []
        real_packs = 0
        for rank in range(dp):
            sampler = Sampler(len(dataset), start, micro, rank, dp)
            # Take exactly one padded epoch across all ranks/microbatches.
            loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
            for batch_id, batch in enumerate(loader):
                if batch_id == dataset.samples_per_epoch // (micro * dp):
                    break
                real_packs += (~batch['padding_mask'].all(dim=1)).sum().item()
                for tokens, positions, padding in zip(
                    batch['tokens'], batch['position_ids'], batch['padding_mask']
                ):
                    seen.extend(tokens[(positions == 0) & ~padding].tolist())
                assert not batch['assistant_mask'][batch['padding_mask']].any()
        assert sorted(seen) == list(range(1, 12))
        assert real_packs == len(source)
        for offset in range(len(source), dataset.samples_per_epoch):
            dummy = dataset[start + offset]
            assert not dummy['loss_mask'].any()
            assert not dummy['assistant_mask'].any()
            assert dummy['padding_mask'].all()
    # Checkpoint offsets count physical slots; every DP rank resumes the exact stream.
    for rank in range(dp):
        all_batches = list(Sampler(len(dataset), 0, micro, rank, dp))
        offset = global_batch
        resumed = list(Sampler(len(dataset), offset, micro, rank, dp))
        assert resumed == all_batches[offset // (micro * dp) :]


@pytest.mark.parametrize('virtual', [None, 2])
@pytest.mark.parametrize('has_source', [True, False])
def test_preparation_resolves_schedule_and_reuses_training(
    tmp_path, monkeypatch, virtual, has_source
):
    source = make_dataset(tmp_path, [[1, 90, 2, 91, 92]] * 5)
    calls = []

    def provider(sizes, **kwargs):
        calls.append((sizes, kwargs))
        return (source if has_source else None), 'valid', 'test'

    provider.is_distributed = True
    args = SimpleNamespace(
        virtual_pipeline_model_parallel_size=virtual, global_batch_size=4, ap_sft_epochs=2, seed=123
    )
    real_tensor = torch.tensor
    monkeypatch.setattr(
        torch, 'tensor', lambda *a, **kw: real_tensor(*a, **{**kw, 'device': 'cpu'})
    )
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda count, **_: count.fill_(5))
    wrapped = module.prepare_apertus_sft_epochs(provider, args)
    assert calls[0] == ((None, 0, 0), {'vp_stage': 0} if virtual else {})
    assert args.train_iters == 4
    assert args.ap_sft_real_samples_per_epoch == 5
    assert args.ap_sft_samples_per_epoch == 8
    train, valid, test = wrapped((16, 24, 4), vp_stage=0)
    assert (valid, test) == ('valid', 'test')
    assert calls[-1] == ((0, 24, 4), {'vp_stage': 0} if virtual else {})
    assert wrapped.is_distributed
    if has_source:
        assert len(train) == 16
        assert train is wrapped((16, 24, 4), vp_stage=1)[0]
    else:
        assert train is None


@pytest.mark.parametrize('per_split', [False, True])
@pytest.mark.parametrize('prefix_count', [1, 2])
def test_real_provider_builds_exhaustive_training_and_preserves_validation(
    tmp_path, monkeypatch, per_split, prefix_count
):
    if prefix_count == 2:
        pytest.importorskip('megatron.core.datasets.helpers')
    prefixes = []
    for i in range(prefix_count):
        prefix = tmp_path / f'tokens{i}'
        write_index(prefix, [[1 + i * 10 + j, 90, 2, 91, 92] for j in range(8)], np.int32)
        prefixes.append(str(prefix))
    config = GPTDatasetConfig(
        random_seed=123,
        sequence_length=16,
        tokenizer=Tokenizer(),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        blend=None if per_split else (prefixes, None),
        split=None if per_split else '75,25,0',
        blend_per_split=[(prefixes, None), (prefixes[:1], None), None] if per_split else None,
        path_to_cache=str(tmp_path / 'cache'),
    )
    args = SimpleNamespace(
        ap_sft=True,
        ap_sft_epochs=2,
        goldfish_loss=False,
        virtual_pipeline_model_parallel_size=None,
        global_batch_size=4,
        seed=123,
    )
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    real_tensor = torch.tensor
    monkeypatch.setattr(
        torch, 'tensor', lambda *a, **kw: real_tensor(*a, **{**kw, 'device': 'cpu'})
    )
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda *a, **kw: None)
    provider = load_function(
        'pretrain_gpt.py',
        'train_valid_test_datasets_provider',
        {
            'get_args': lambda: args,
            'core_gpt_dataset_config_from_args': lambda _: copy.deepcopy(config),
            'ApertusSFTDataset': ApertusSFTDataset,
            'BlendedMegatronDatasetBuilder': BlendedMegatronDatasetBuilder,
            'partial': partial,
            'is_dataset_built_on_rank': lambda **_: True,
            'print_rank_0': lambda _: None,
        },
    )
    provider.is_distributed = True
    wrapped = module.prepare_apertus_sft_epochs(provider, args)
    train, valid, test = wrapped((args.train_iters * 4, 20, 0))
    expected = (8 if per_split else 6) * prefix_count
    assert train.real_samples_per_epoch == expected
    # Existing unweighted multi-prefix validation is capped at one source pass.
    expected_validation = 4 if prefix_count == 2 and not per_split else 20
    assert len(valid) == expected_validation and test is None
    for epoch in range(2):
        items = [train[epoch * train.samples_per_epoch + i] for i in range(expected)]
        ids = [item['tokens'][0].item() for item in items]
        assert len(set(ids)) == expected
    for i in range(expected, train.samples_per_epoch):
        dummy = train[i]
        assert not dummy['loss_mask'].any()
        if prefix_count == 2:
            assert dummy['dataset_id'] == 0


@pytest.mark.parametrize(
    'bad',
    [
        {'ap_sft': False},
        {'ap_sft_epochs': 0},
        {'ap_sft_epochs': -1},
        {'train_iters': 1},
        {'train_samples': 8},
        {'dataloader_type': 'cyclic'},
        {'rampup_batch_size': [1, 1, 8]},
        {'phase_transition_iterations': [1]},
        {'skip_train': True},
        {'perform_rl_step': True},
        {'decrease_batch_size_if_needed': True},
        {'global_batch_size': 3},
    ],
)
def test_epoch_cli_rejects_incompatible_options(bad):
    validate = load_function('megatron/training/arguments.py', 'validate_apertus_sft_epochs', {})
    args = SimpleNamespace(
        ap_sft=True,
        ap_sft_epochs=1,
        train_iters=None,
        train_samples=None,
        dataloader_type='single',
        global_batch_size=4,
        micro_batch_size=1,
        data_parallel_size=2,
    )
    validate(args)
    for key, value in bad.items():
        setattr(args, key, value)
    with pytest.raises(AssertionError, match='ap-sft-epochs'):
        validate(args)


def test_epoch_cli_default_leaves_existing_modes_alone():
    validate = load_function('megatron/training/arguments.py', 'validate_apertus_sft_epochs', {})
    validate(SimpleNamespace())


@pytest.mark.parametrize('blend', [None, (['tokens'], [1.0])])
def test_provider_rejects_missing_or_weighted_training(blend):
    args = SimpleNamespace(ap_sft_epochs=1)
    config = SimpleNamespace(blend=blend, blend_per_split=None)
    provider = load_function(
        'pretrain_gpt.py',
        'train_valid_test_datasets_provider',
        {'get_args': lambda: args, 'core_gpt_dataset_config_from_args': lambda _: config},
    )
    with pytest.raises(ValueError, match='without blend weights'):
        provider((None, 0, 0))


def test_empty_training_split_fails_before_optimizer_setup(monkeypatch):
    args = SimpleNamespace(virtual_pipeline_model_parallel_size=None)
    real_tensor = torch.tensor
    monkeypatch.setattr(
        torch, 'tensor', lambda *a, **kw: real_tensor(*a, **{**kw, 'device': 'cpu'})
    )
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda *a, **kw: None)
    with pytest.raises(ValueError, match='nonempty training split'):
        module.prepare_apertus_sft_epochs(lambda sizes: ([], None, None), args)


@pytest.mark.parametrize('consumed', [0, 4, 16])
def test_training_loader_resumes_and_accepts_completed_epoch_run(tmp_path, monkeypatch, consumed):
    source = make_dataset(tmp_path, [[1, 90, 2, 91, 92]] * 5)
    dataset = EpochDataset(source, 2, 4, seed=123)
    args = SimpleNamespace(
        ap_sft_epochs=2,
        train_iters=4,
        global_batch_size=4,
        train_samples=None,
        iteration=consumed // 4,
        consumed_train_samples=consumed,
        consumed_valid_samples=0,
        eval_interval=10,
        eval_iters=0,
        phase_transition_iterations=None,
        perform_rl_step=False,
        skip_train=False,
        full_validation=False,
        multiple_validation_sets=False,
    )
    real_tensor = torch.tensor
    monkeypatch.setattr(
        torch, 'tensor', lambda *a, **kw: real_tensor(*a, **{**kw, 'device': 'cpu'})
    )
    monkeypatch.setattr(torch.distributed, 'broadcast', lambda *a, **kw: None)

    def loader(data, offset):
        if data is None:
            return None
        return torch.utils.data.DataLoader(data, batch_sampler=Sampler(len(data), offset, 1, 0, 1))

    provider = lambda sizes: (dataset, None, None)
    provider.is_distributed = True
    build_loaders = load_function(
        'megatron/training/training.py',
        'build_train_valid_test_data_loaders',
        {
            'get_args': lambda: args,
            'print_rank_0': lambda _: None,
            'torch': torch,
            'build_train_valid_test_datasets': lambda _: (dataset, None, None),
            'build_pretraining_data_loader': loader,
        },
    )
    train, _, _ = build_loaders(provider)
    if consumed == len(dataset):
        assert train is None and not args.do_train
    else:
        assert args.do_train
        batches = list(train)
        assert len(batches) == len(dataset) - consumed
        assert torch.equal(batches[0]['tokens'][0], dataset[consumed]['tokens'])


def test_dummy_tail_does_not_change_loss_denominator_or_gradients(tmp_path):
    source = make_dataset(
        tmp_path, [[1, 90, 2, 91, 92]], [[0, 0.25, 0, 2, 0.5]], sft_report_assistant_loss=True
    )
    dataset = EpochDataset(source, 1, 4, seed=123)
    batch = torch.utils.data.default_collate([dataset[i] for i in range(len(dataset))])
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
    losses = torch.full_like(batch['loss_mask'], 2, requires_grad=True)
    loss, count, report = loss_func(
        batch['loss_mask'], losses, assistant_mask=batch['assistant_mask']
    )
    assert count == 3 and loss == 5.5
    assert report['assistant_loss'].tolist() == [8, 4]
    (loss / count).backward()
    torch.testing.assert_close(losses.grad[0], source[0]['loss_mask'] / 3)
    assert not losses.grad[1:].any()
