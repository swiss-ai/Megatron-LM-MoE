# Copyright (c) 2026, EPFL / Swiss AI Initiative.

from argparse import Namespace

import pytest
import torch

import tools.checkpoint.merge as checkpoint_merge
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from tests.unit_tests.test_utilities import Utils
from tools.checkpoint.merge import LinearDecay, mean_coefficients, wsm_coefficients


def test_resolve_sources():
    sources = checkpoint_merge.resolve_sources(["run"], [1000, 2000])
    assert sources == ["run/iter_0001000", "run/iter_0002000"]
    assert checkpoint_merge.source_iterations(sources) == [1000, 2000]
    assert checkpoint_merge.source_iterations(["run/iter_0001000", "run/release"]) is None
    with pytest.raises(ValueError, match="exactly one checkpoint root"):
        checkpoint_merge.resolve_sources(["one", "two"], [1000, 2000])


@pytest.mark.parametrize(
    "coefficients, expected",
    [
        (mean_coefficients(4), [0.25] * 4),
        # Stable source: a linear decay to 0.2 over five intervals.
        (wsm_coefficients(5, decay_end=0.2), [0.16, 0.16, 0.16, 0.16, 0.36]),
        # Decayed source: the decay is rebased at the synthetic interval [0, 20) ...
        (
            wsm_coefficients(3, 0.5, LinearDecay(steps=100, end=0.5, start=0), [20, 40, 60]),
            [47 / 682, 635 / 682 - 254 / 301, 254 / 301],
        ),
        # ... and averaged over training steps, so the synthetic interval [-20, 0) counts as 1.
        (
            wsm_coefficients(3, 0.5, LinearDecay(steps=100, end=0.5, start=0), [0, 20, 40]),
            [143 / 1143, 1000 / 1143 - 800 / 1023, 800 / 1023],
        ),
    ],
)
def test_coefficients(coefficients, expected):
    assert coefficients == pytest.approx(expected)


@pytest.mark.parametrize(
    "source_decay, iterations, match",
    [
        (LinearDecay(steps=100, end=0.5), None, "needs the checkpoint iterations"),
        (LinearDecay(steps=100, end=0.0, start=0), [100, 125, 150], "reaches zero"),
    ],
)
def test_wsm_rejects_invalid_source_decay(source_decay, iterations, match):
    with pytest.raises(ValueError, match=match):
        wsm_coefficients(3, source_decay=source_decay, iterations=iterations)


def test_empty_workers_still_enter_load_collectives(monkeypatch):
    # A rank that owns no tensors must still join the collective load, or the others hang.
    calls = []

    def fake_load(templates, source, **kwargs):
        calls.append(source)
        return {"common": "state"}

    monkeypatch.setattr(checkpoint_merge.dist_checkpointing, "load", fake_load)
    assert checkpoint_merge._load_assigned("checkpoint", {}) == {}
    assert calls == ["checkpoint"]


@pytest.fixture
def initialized_model_parallel():
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


def _full(key, value):
    return ShardedTensor.from_rank_offsets(key, torch.full((4,), value))


def test_merge_torch_dist_checkpoints(initialized_model_parallel, tmp_path_dist_ckpt):
    model_keys = ("weight", "router.expert_bias", "router.qb_beta")
    sources = []
    for value, iteration in ((1.0, 123), (5.0, 456)):
        source = tmp_path_dist_ckpt / f"iter_{iteration:07d}"
        source.mkdir()
        state = {key: _full(key, value) for key in model_keys}
        state["optimizer.state.row_gain.weight"] = _full("optimizer.state.row_gain.weight", 99.0)
        state["iteration"] = iteration
        state["args"] = Namespace(iteration=iteration, consumed_train_samples=iteration * 512)
        state["rng_state"] = "excluded"
        dist_checkpointing.save(state, str(source))
        sources.append(str(source))

    # The router balancing state always gets a plain mean (3.0), whatever the coefficients.
    for method, coefficients, expected_weight in (("mean", None, 3.0), ("wsm", [0.75, 0.25], 2.0)):
        release = tmp_path_dist_ckpt / method / "release"
        checkpoint_merge.merge_checkpoint_directories(
            sources, release.parent, coefficients, merge_metadata={"method": method}
        )

        loaded = dist_checkpointing.load({key: _full(key, 0.0) for key in model_keys}, str(release))
        expected = {"weight": expected_weight, "router.expert_bias": 3.0, "router.qb_beta": 3.0}
        for key, value in expected.items():
            torch.testing.assert_close(loaded[key], torch.full((4,), value))
        assert set(dist_checkpointing.load_tensors_metadata(str(release))) == set(model_keys)

        # Progress comes from the last checkpoint; training state is dropped.
        common = dist_checkpointing.load_common_state_dict(str(release))
        assert common["iteration"] == 456
        assert common["args"].consumed_train_samples == 456 * 512
        assert "rng_state" not in common
        assert common["checkpoint_merge_metadata"] == {
            "method": method,
            "source_checkpoints": sources,
            "coefficients": coefficients or [0.5, 0.5],
        }
