# Copyright (c) 2026, EPFL / Swiss AI Initiative.

from argparse import Namespace

import pytest
import torch

import tools.checkpoint.merge as checkpoint_merge
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from tests.unit_tests.test_utilities import Utils
from tools.checkpoint.merge import (
    LinearDecay,
    ema_coefficients,
    mean_coefficients,
    wsm_coefficients,
)


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
        # avg = 1; avg = 0.5 * 2 + 0.5 * avg; avg = 0.5 * 3 + 0.5 * avg.
        (ema_coefficients(3, alpha=0.5), [0.25, 0.25, 0.5]),
        (ema_coefficients(3, alpha=1.0), [0.0, 0.0, 1.0]),
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


def test_ema_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="EMA alpha"):
        ema_coefficients(3, alpha=0.0)


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


def _full(key, value, dtype=torch.float32, size=4):
    return ShardedTensor.from_rank_offsets(key, torch.full((size,), value, dtype=dtype))


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

    # By default the router balancing state gets a plain mean (3.0), whatever the coefficients;
    # with router_state="method" it follows the coefficients like the weight, and with "last" it is
    # copied from the last checkpoint (5.0).
    for method, coefficients, router_state, expected_weight, expected_router in (
        ("mean", None, "mean", 3.0, 3.0),
        ("wsm", [0.75, 0.25], "mean", 2.0, 3.0),
        ("wsm", [0.75, 0.25], "method", 2.0, 2.0),
        ("wsm", [0.75, 0.25], "last", 2.0, 5.0),
    ):
        release = tmp_path_dist_ckpt / f"{method}_{router_state}" / "release"
        checkpoint_merge.merge_checkpoint_directories(
            sources,
            release.parent,
            coefficients,
            merge_metadata={"method": method},
            router_state=router_state,
        )

        loaded = dist_checkpointing.load({key: _full(key, 0.0) for key in model_keys}, str(release))
        expected = {
            "weight": expected_weight,
            "router.expert_bias": expected_router,
            "router.qb_beta": expected_router,
        }
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
            "router_state": router_state,
            "save_fp32": True,
        }


def test_save_fp32_keeps_precision(initialized_model_parallel, tmp_path_dist_ckpt):
    # bf16 sources 1.0 and 1.0078125 (the next bf16 value): their mean 1.00390625 lies halfway
    # between two bf16 values, so a bf16 output rounds it to 1.0, and only fp32 keeps it.
    sources = []
    for value, iteration in ((1.0, 100), (1.0078125, 200)):
        source = tmp_path_dist_ckpt / "fp32" / f"iter_{iteration:07d}"
        source.mkdir(parents=True)
        state = {"weight": _full("weight", value, torch.bfloat16), "iteration": iteration}
        dist_checkpointing.save(state, str(source))
        sources.append(str(source))

    for save_fp32, dtype, expected in (
        (False, torch.bfloat16, 1.0),
        (True, torch.float32, 1.00390625),
    ):
        output = tmp_path_dist_ckpt / "fp32" / f"merged_{save_fp32}"
        checkpoint_merge.merge_checkpoint_directories(sources, output, save_fp32=save_fp32)
        release = str(output / "release")
        assert dist_checkpointing.load_tensors_metadata(release)["weight"].dtype == dtype
        loaded = dist_checkpointing.load({"weight": _full("weight", 0.0)}, release)["weight"]
        torch.testing.assert_close(loaded, torch.full((4,), expected), rtol=0, atol=0)

    # A bf16 model can load the fp32 checkpoint: the loader casts to the model's dtype.
    loaded = dist_checkpointing.load(
        {"weight": _full("weight", 0.0, torch.bfloat16)}, str(output / "release")
    )["weight"]
    assert loaded.dtype == torch.bfloat16


def _save_update_source(root, iteration):
    # Values that are not exact sums of each other, stored in bf16 like a training checkpoint.
    weight = (torch.linspace(-1, 1, 8) * (1 + iteration / 1000) + iteration / 7).to(torch.bfloat16)
    source = root / f"iter_{iteration:07d}"
    source.mkdir(parents=True)
    state = {
        "weight": ShardedTensor.from_rank_offsets("weight", weight),
        "router.expert_bias": ShardedTensor.from_rank_offsets("router.expert_bias", weight * 2),
        "iteration": iteration,
        "args": Namespace(iteration=iteration),
    }
    dist_checkpointing.save(state, str(source))
    return str(source), weight.float()


@pytest.mark.parametrize(
    "name, method, kwargs, router_state, effective",
    [
        ("mean", "mean", {}, "last", lambda n: mean_coefficients(n)),
        ("ema", "ema", dict(ema_alpha=0.3), "method", lambda n: ema_coefficients(n, 0.3)),
        (
            "ema_bc",
            "ema",
            dict(ema_alpha=0.3, ema_bias_correction=True),
            "mean",
            lambda n: ema_coefficients(n, 0.3, bias_correction=True),
        ),
        ("wsm", "wsm", dict(decay_end=0.1), "method", lambda n: wsm_coefficients(n, 0.1)),
    ],
)
def test_update_matches_offline_merge(
    initialized_model_parallel, tmp_path_dist_ckpt, name, method, kwargs, router_state, effective
):
    root = tmp_path_dist_ckpt / f"update_{name}"
    sources, weights = zip(*(_save_update_source(root / "run", it) for it in (100, 200, 300, 400)))
    output = root / "merged"

    def update(source, **overrides):
        settings = dict(method=method, router_state=router_state, **kwargs) | overrides
        checkpoint_merge.update_running_merge(output, [source], **settings)

    for source in sources[:3]:
        update(source)
    # An interrupted update leaves an iteration directory that the tracker never named.
    if torch.distributed.get_rank() == 0:
        (output / "iter_0000400").mkdir()
        (output / "iter_0000400" / "partial").write_text("")
    torch.distributed.barrier()
    update(sources[3])

    # Folding one checkpoint at a time gives the offline merge of all four.
    assert (output / "latest_checkpointed_iteration.txt").read_text() == "400"
    assert [path.name for path in output.glob("iter_*")] == ["iter_0000400"]
    merged = str(output / "iter_0000400")
    assert dist_checkpointing.load_tensors_metadata(merged)["weight"].dtype == torch.float32
    coefficients = effective(4)
    router_coefficients = checkpoint_merge._router_coefficients(
        router_state, coefficients, mean_coefficients(4)
    )
    loaded = dist_checkpointing.load(
        {key: _full(key, 0.0, size=8) for key in ("weight", "router.expert_bias")}, merged
    )
    expected = sum(c * w for c, w in zip(coefficients, weights))
    expected_router = sum(c * w * 2 for c, w in zip(router_coefficients, weights))
    # Streaming rounds to fp32 after every update, the offline sum once: allow a few fp32 ulps.
    torch.testing.assert_close(loaded["weight"], expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(loaded["router.expert_bias"], expected_router, rtol=1e-6, atol=1e-6)
    metadata = dist_checkpointing.load_common_state_dict(merged)["checkpoint_merge_metadata"]
    assert metadata["num_checkpoints"] == 4
    assert metadata["source_checkpoints"] == list(sources)
    assert metadata["iterations"] == [100, 200, 300, 400]
    assert metadata["coefficients"] == pytest.approx(coefficients)
    if method == "wsm":
        assert [p.name for p in (output / "wsm_running_mean").glob("iter_*")] == ["iter_0000400"]

    # Folding the same checkpoint again changes nothing; older or mismatched input is refused.
    update(sources[3])
    assert [path.name for path in output.glob("iter_*")] == ["iter_0000400"]
    older, _ = _save_update_source(root / "run", 50)
    with pytest.raises(ValueError, match="not newer"):
        update(older)
    with pytest.raises(ValueError, match="Settings differ"):
        update(sources[3], router_state="last" if router_state != "last" else "mean")
