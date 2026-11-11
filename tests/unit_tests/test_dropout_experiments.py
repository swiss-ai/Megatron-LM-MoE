"""Tests for --embedding-dropout, --block-input-dropout and --clean-loss-interval."""

import argparse
from types import SimpleNamespace

import pytest
import torch

import megatron.training.training as training_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import _block_input_dropout


def _parser():
    from megatron.training.arguments import add_megatron_arguments

    return add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))


def test_arguments_and_defaults():
    p = _parser()
    a = p.parse_args([])
    assert a.embedding_dropout is None and a.block_input_dropout == 0.0 and a.clean_loss_interval == 0
    a = p.parse_args(["--hidden-dropout", "0.15", "--embedding-dropout", "0", "--block-input-dropout", "0.1",
                      "--clean-loss-interval", "20"])
    assert (a.hidden_dropout, a.embedding_dropout, a.block_input_dropout, a.clean_loss_interval) == (0.15, 0.0, 0.1, 20)
    fields = TransformerConfig.__dataclass_fields__
    assert fields["embedding_dropout"].default is None and fields["block_input_dropout"].default == 0.0


def test_block_input_dropout_is_a_new_tensor_and_leaves_the_residual():
    cfg = SimpleNamespace(block_input_dropout=0.25)
    x = torch.ones(64, 4, 256)
    residual = x                                  # what the layer keeps for the add (aliases the input)
    torch.manual_seed(0)
    y = _block_input_dropout(x, cfg, training=True)
    assert y is not x and torch.equal(residual, torch.ones_like(x))
    zero = (y == 0).float().mean().item()
    assert abs(zero - 0.25) < 0.01
    torch.testing.assert_close(y[y != 0], torch.full_like(y[y != 0], 1 / 0.75))
    assert _block_input_dropout(x, cfg, training=False) is x
    assert _block_input_dropout(x, SimpleNamespace(block_input_dropout=0.0), training=True) is x
    assert _block_input_dropout(x, SimpleNamespace(), training=True) is x   # older configs


def test_replay_iterator_serves_the_same_batches_twice_then_continues():
    it = training_module._ReplayIterator(iter(range(10)))
    it.prefetch(3)
    assert [next(it) for _ in range(3)] == [0, 1, 2]
    it.rewind()
    assert [next(it) for _ in range(3)] == [0, 1, 2]
    assert next(it) == 3
    with pytest.raises(AssertionError):
        it.prefetch(2); it.prefetch(2)


def test_clean_loss_evaluates_the_training_batches_in_eval_mode(monkeypatch):
    """_clean_loss puts the model in eval mode for one forward-only pass over the prefetched micro-batches,
    restores train mode and the rerun mode, clears eval-only routing samples, rewinds the batches for the
    training step and returns sum(loss) / sum(tokens)."""
    seen = {}

    class _Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inference_seq_expert_load_samples = []

    model = torch.nn.Sequential(torch.nn.Dropout(0.5), _Router())

    def fake_fb(forward_step_func, data_iterator, model, num_microbatches, forward_only, **kw):
        seen["training"] = [m.training for m in model]
        seen["forward_only"] = forward_only
        seen["batches"] = [next(data_iterator) for _ in range(num_microbatches)]
        model[0][1].inference_seq_expert_load_samples.append(torch.ones(3))
        return [{"lm loss": torch.tensor([6.0, 3.0])}, {"lm loss": torch.tensor([2.0, 1.0])}]

    class _RSM:
        mode = "validate"
        def get_mode(self): return self.mode
        def set_mode(self, m): self.mode = m
    rsm = _RSM()
    monkeypatch.setattr(training_module, "get_rerun_state_machine", lambda: rsm)
    monkeypatch.setattr(training_module, "get_num_microbatches", lambda: 2)
    monkeypatch.setattr(training_module, "get_args", lambda: SimpleNamespace(seq_length=8, micro_batch_size=1, decoder_seq_length=None))
    monkeypatch.setattr(training_module.mpu, "is_pipeline_last_stage", lambda ignore_virtual=True: True)
    monkeypatch.setattr(training_module.mpu, "get_data_parallel_group", lambda with_context_parallel=True: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda t, group=None: t)
    replay = training_module._ReplayIterator(iter(["b0", "b1", "b2"]))
    config = SimpleNamespace(timers="T")

    loss = training_module._clean_loss(None, replay, [model], config, fake_fb)

    assert loss == pytest.approx(8.0 / 4.0)
    assert seen["training"] == [False] and seen["forward_only"] and seen["batches"] == ["b0", "b1"]
    assert model.training and rsm.mode == "validate" and config.timers == "T"
    assert model[1].inference_seq_expert_load_samples == []
    assert [next(replay), next(replay), next(replay)] == ["b0", "b1", "b2"]
