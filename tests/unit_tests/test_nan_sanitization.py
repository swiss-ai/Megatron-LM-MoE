"""CPU tests for counted gradient sanitization; no distributed setup required."""

import importlib.util
import math
import os
import tempfile
from datetime import timedelta
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def load_sanitizer(monkeypatch, enabled=True):
    monkeypatch.setenv('NAN_DEBUG_SANITIZE', '1' if enabled else '0')
    path = Path(__file__).resolve().parents[2] / 'megatron/training/nan_debug.py'
    spec = importlib.util.spec_from_file_location('nan_debug_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_counts_and_sanitizes_aliases_once(monkeypatch, capsys):
    sanitizer = load_sanitizer(monkeypatch)
    model = torch.nn.Linear(4, 1, bias=False)
    model.weight.grad = torch.tensor([[float('nan'), float('inf'), -float('inf'), 3.0]])
    model.weight.main_grad = model.weight.grad.view_as(model.weight.grad)
    sanitizer.nan_debug_sanitize_grads(model, 42)
    assert torch.equal(model.weight.grad, torch.tensor([[0.0, 0.0, 0.0, 3.0]]))
    assert capsys.readouterr().out == (
        '[NAN-SANITIZE] rank=0 iter=42 nan=1 inf=2 replaced_with_zero=3\n'
    )


def test_counts_distinct_grad_buffers(monkeypatch, capsys):
    sanitizer = load_sanitizer(monkeypatch)
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[float('nan'), 7.0]])
    model.weight.main_grad = torch.tensor([[float('nan'), float('nan')]])
    sanitizer.nan_debug_sanitize_grads([model], 43)
    assert 'nan=3 inf=0 replaced_with_zero=3' in capsys.readouterr().out
    assert model.weight.grad[0, 1] == 7.0
    assert torch.isfinite(model.weight.main_grad).all()


def test_finite_grads_are_unchanged_and_quiet(monkeypatch, capsys):
    sanitizer = load_sanitizer(monkeypatch)
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[2.0, -4.0]])
    before = model.weight.grad.clone()

    def unexpected_slow_path(*args, **kwargs):
        raise AssertionError('finite gradients must not be counted or sanitized')

    monkeypatch.setattr(torch, 'isnan', unexpected_slow_path)
    monkeypatch.setattr(torch, 'isinf', unexpected_slow_path)
    monkeypatch.setattr(torch, 'nan_to_num_', unexpected_slow_path)
    sanitizer.nan_debug_sanitize_grads(model, 44)
    assert torch.equal(model.weight.grad, before)
    assert capsys.readouterr().out == ''


def test_disabled_does_not_sanitize(monkeypatch, capsys):
    sanitizer = load_sanitizer(monkeypatch, enabled=False)
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.tensor([[float('nan')]])
    sanitizer.nan_debug_sanitize_grads(model, 45)
    assert torch.isnan(model.weight.grad).all()
    assert capsys.readouterr().out == ''


def test_finite_norm_bypasses_scan_and_sanitization(monkeypatch):
    sanitizer = load_sanitizer(monkeypatch)

    def unexpected(*args, **kwargs):
        raise AssertionError('healthy norm must bypass the repair path')

    monkeypatch.setattr(sanitizer, 'nan_debug_sanitize_grads', unexpected)
    optimizer = SimpleNamespace(prepare_grad_norm=unexpected)
    assert sanitizer.nan_debug_sanitize_after_grad_norm(None, optimizer, 46, False, 0.5) == (
        False, 0.5
    )
    # A finite spike must also pass through unchanged to the rerun check.
    assert sanitizer.nan_debug_sanitize_after_grad_norm(None, optimizer, 46, False, 100.0) == (
        False, 100.0
    )


def test_nonfinite_norm_repairs_and_refreshes_optimizer_copy(monkeypatch, capsys):
    sanitizer = load_sanitizer(monkeypatch)
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.main_grad = torch.tensor([[float('nan'), 3.0]])
    prepared_grad = model.weight.main_grad.clone()

    def reprepare():
        prepared_grad.copy_(model.weight.main_grad)
        return False, prepared_grad.norm().item()

    optimizer = SimpleNamespace(prepare_grad_norm=reprepare)
    result = sanitizer.nan_debug_sanitize_after_grad_norm(
        model, optimizer, 47, False, float('nan')
    )
    assert result == (False, 3.0)
    assert torch.equal(prepared_grad, torch.tensor([[0.0, 3.0]]))
    assert 'nan=1 inf=0 replaced_with_zero=1' in capsys.readouterr().out


def _model_with_main_grad(values):
    model = torch.nn.Linear(len(values), 1, bias=False)
    model.weight.main_grad = torch.tensor([values], dtype=torch.float32)
    return model


def test_locally_finite_rank_recomputes_when_a_peer_repairs(monkeypatch, capsys):
    sanitizer = load_sanitizer(monkeypatch)
    # Another rank has 5 bad elements: this rank is finite but must still join the
    # second norm collective.
    monkeypatch.setattr(sanitizer, '_global_max_int', lambda value: 5)
    model = _model_with_main_grad([1.0])
    calls = []

    def reprepare():
        calls.append(True)
        return False, 4.0

    result = sanitizer.nan_debug_sanitize_after_grad_norm(
        model, SimpleNamespace(prepare_grad_norm=reprepare), 48, False, float('inf')
    )
    assert result == (False, 4.0)
    assert calls == [True]
    assert 'max_per_rank=5 threshold=16 action=sanitize' in capsys.readouterr().out


def test_nonfinite_norm_without_nonfinite_grads_is_left_to_rerun(monkeypatch, capsys):
    sanitizer = load_sanitizer(monkeypatch)
    model = _model_with_main_grad([1.0, 2.0])

    def unexpected():
        raise AssertionError('nothing to repair, so no second norm collective')

    result = sanitizer.nan_debug_sanitize_after_grad_norm(
        model, SimpleNamespace(prepare_grad_norm=unexpected), 49, False, float('inf')
    )
    assert result[0] is False and math.isinf(result[1])
    assert 'no gradient element is' in capsys.readouterr().out


def _run_checked_path(sanitizer, n_nan, optimizer_norm=7.0):
    model = _model_with_main_grad([float('nan')] * n_nan + [1.0, 2.0, 3.0])
    calls = []

    def reprepare():
        calls.append(True)
        return False, optimizer_norm

    result = sanitizer.nan_debug_sanitize_after_grad_norm(
        model, SimpleNamespace(prepare_grad_norm=reprepare), 50, False, float('nan')
    )
    return model, result, calls


def test_threshold_boundary_repairs_16_and_reruns_17(monkeypatch, capsys):
    sanitizer = load_sanitizer(monkeypatch)

    model, result, calls = _run_checked_path(sanitizer, 16)
    assert result == (False, 7.0) and calls == [True]
    assert torch.isfinite(model.weight.main_grad).all()
    assert 'nan=16 inf=0 replaced_with_zero=16 max_per_rank=16 threshold=16' in capsys.readouterr().out

    model, result, calls = _run_checked_path(sanitizer, 17)
    # Untouched gradients and the original non-finite norm go back so the step is rerun.
    assert result[0] is False and math.isnan(result[1])
    assert calls == []
    assert int(torch.isnan(model.weight.main_grad).sum()) == 17
    assert 'nan=17 inf=0 replaced_with_zero=0 max_per_rank=17 threshold=16' in capsys.readouterr().out


def test_threshold_env_override_and_invalid_value(monkeypatch):
    sanitizer = load_sanitizer(monkeypatch)

    monkeypatch.setenv('NAN_DEBUG_SANITIZE_MAX', '0')  # count and log only
    model, result, calls = _run_checked_path(sanitizer, 1)
    assert math.isnan(result[1]) and calls == [] and torch.isnan(model.weight.main_grad).any()

    monkeypatch.setenv('NAN_DEBUG_SANITIZE_MAX', '2')
    assert _run_checked_path(sanitizer, 2)[2] == [True]
    assert _run_checked_path(sanitizer, 3)[2] == []

    monkeypatch.setenv('NAN_DEBUG_SANITIZE_MAX', 'not-a-number')  # falls back to 16
    assert _run_checked_path(sanitizer, 16)[2] == [True]
    assert _run_checked_path(sanitizer, 17)[2] == []


def test_inf_counts_toward_the_cap(monkeypatch, capsys):
    sanitizer = load_sanitizer(monkeypatch)
    model = _model_with_main_grad([float('nan')] * 8 + [float('inf')] * 8 + [-float('inf')] + [1.0])
    result = sanitizer.nan_debug_sanitize_after_grad_norm(
        model, SimpleNamespace(prepare_grad_norm=lambda: (False, 1.0)), 51, False, float('nan')
    )
    # 8 NaN + 9 inf = 17 > 16: left to the rerun (original non-finite norm returned,
    # gradients untouched).
    assert result[0] is False and math.isnan(result[1])
    assert int(torch.isnan(model.weight.main_grad).sum()) == 8
    assert int(torch.isinf(model.weight.main_grad).sum()) == 9
    assert 'nan=8 inf=9 replaced_with_zero=0 max_per_rank=17' in capsys.readouterr().out


def _dist_worker(rank, init_file, nans_on_rank0, results, errors):
    os.environ['NAN_DEBUG_SANITIZE'] = '1'
    os.environ.pop('NAN_DEBUG_SANITIZE_MAX', None)
    dist.init_process_group(
        'gloo', init_method=f'file:///{init_file}', rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        path = Path(__file__).resolve().parents[2] / 'megatron/training/nan_debug.py'
        spec = importlib.util.spec_from_file_location('nan_debug_dist', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        grad = torch.ones(1, 40)
        if rank == 0:
            grad[0, :nans_on_rank0] = float('nan')  # only rank 0 holds bad elements
        model = torch.nn.Linear(40, 1, bias=False)
        model.weight.main_grad = grad
        calls = []

        def reprepare():
            calls.append(True)
            return False, 7.0

        found, norm = module.nan_debug_sanitize_after_grad_norm(
            model, SimpleNamespace(prepare_grad_norm=reprepare), 9, False, float('nan')
        )
        results[rank] = (found, None if math.isnan(norm) else norm, len(calls),
                         bool(torch.isfinite(grad).all()))
    except BaseException as e:  # noqa: BLE001 - report to the parent
        errors[rank] = f'{type(e).__name__}: {e}'
    finally:
        dist.destroy_process_group()


def _run_two_ranks(nans_on_rank0):
    with tempfile.TemporaryDirectory() as tmp:
        init_file = (Path(tmp) / 'rdzv').as_posix()
        with mp.Manager() as manager:
            results, errors = manager.dict(), manager.dict()
            mp.spawn(_dist_worker, args=(init_file, nans_on_rank0, results, errors), nprocs=2)
            return dict(results), dict(errors)


def test_ranks_agree_when_only_one_rank_has_bad_elements():
    # 1 bad element on rank 0 only: BOTH ranks repair and recompute (rank 1 is locally
    # finite but must join the collective), so neither is left waiting.
    results, errors = _run_two_ranks(1)
    assert errors == {}, errors
    assert results[0] == (False, 7.0, 1, True)
    assert results[1] == (False, 7.0, 1, True)

    # 20 bad elements on rank 0 only: BOTH ranks leave the step to the rerun, even though
    # rank 1 saw nothing wrong locally.
    results, errors = _run_two_ranks(20)
    assert errors == {}, errors
    assert results[0] == (False, None, 0, False)
    assert results[1] == (False, None, 0, True)
