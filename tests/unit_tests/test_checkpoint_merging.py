# Copyright (c) 2026, EPFL / Swiss AI Initiative.

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

import tools.checkpoint.merge as checkpoint_merge
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.strategies.torch import _get_filesystem_reader
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


def test_wsm_source_schedule_matches_linear_decay():
    # A per-iteration LR that follows LinearDecay(steps=100, end=0.5, start=0) gives its weights.
    decay = LinearDecay(steps=100, end=0.5, start=0)

    def lr(step):
        return 1.0 - 0.5 * min(max(step, 0), 100) / 100

    assert wsm_coefficients(3, 0.5, lr, [20, 40, 60]) == pytest.approx(
        wsm_coefficients(3, 0.5, decay, [20, 40, 60])
    )


def _bridge_args(**changes):
    # Saved training args of an iteration-based sqrt-bridge run.
    return Namespace(
        **{
            **dict(
                train_iters=1000,
                train_samples=None,
                lr_decay_iters=None,
                lr_warmup_iters=50,
                lr_warmup_fraction=None,
                lr_wsd_decay_iters=None,
                lr_wsd_decay_style=None,
                global_batch_size=8,
                rampup_batch_size=None,
                lr=1e-2,
                min_lr=1e-4,
                lr_warmup_init=0.0,
                lr_decay_style="sqrt-bridge",
                lr_bridge_low_factor=0.5,
                lr_bridge_switch_iters=600,
                lr_bridge_switch_samples=None,
                lr_bridge_log_start=None,
                lr_bridge_rho_ref=None,
                lr_bridge_budget="remaining",
                start_weight_decay=0.1,
                end_weight_decay=0.1,
                weight_decay_incr_style="constant",
                use_checkpoint_opt_param_scheduler=False,
                override_opt_param_scheduler=False,
            ),
            **changes,
        }
    )


def test_wsm_rejects_extrapolating_weights():
    # A source that decays to 0.1 under a target that only decays to 0.9 needs negative weights.
    with pytest.raises(ValueError, match="extrapolate"):
        wsm_coefficients(3, 0.9, LinearDecay(steps=40, end=0.1, start=0), [20, 40, 60])


@pytest.mark.parametrize(
    "changes, saved_lr, match",
    [
        # The weight decay still grows inside the window [400, 700].
        (dict(start_weight_decay=0.0, weight_decay_incr_style="linear"), None, "weight decay"),
        # The run used another schedule than its saved args give (for example after a resume).
        ({}, 1e-3, "rebuilt schedule"),
    ],
)
def test_wsm_group_coefficients_guards(monkeypatch, changes, saved_lr, match):
    args = _bridge_args(**changes)
    lr = checkpoint_merge.checkpoint_lr_schedule(args)
    iterations = [500, 600, 700]

    def common_state(path):
        iteration = int(path.rsplit("_", 1)[1])
        group = {"lr": saved_lr or lr(iteration), "max_lr": args.lr, "min_lr": args.min_lr}
        return {"args": args, "optimizer": {"param_groups": [group]}}

    monkeypatch.setattr(checkpoint_merge.dist_checkpointing, "load_common_state_dict", common_state)
    monkeypatch.setattr(checkpoint_merge.dist_checkpointing, "load_tensors_metadata", lambda _: {})
    sources = [f"run/iter_{iteration}" for iteration in iterations]
    with pytest.raises(ValueError, match=match):
        checkpoint_merge.wsm_group_coefficients(sources, iterations, 1e-4)
    # The same run passes when the guard does not apply.
    if saved_lr is None:
        args.start_weight_decay = 0.1
    else:
        saved_lr = None
    checkpoint_merge.wsm_group_coefficients(sources, iterations, 1e-4)


def test_checkpoint_lr_schedule_rebuilds_sqrt_bridge():
    from megatron.core.optimizer_param_scheduler import build_air_bridge_schedule

    args = _bridge_args()
    lr = checkpoint_merge.checkpoint_lr_schedule(args)
    expected = build_air_bridge_schedule(
        high_lr=1e-2,
        low_lr=5e-3,
        final_lr=1e-4,
        total_steps=8000,
        warmup_steps=400,
        switch_step=4800,
        budget="remaining",
    )
    for iteration in (0, 10, 50, 300, 599, 600, 601, 800, 1000):
        assert lr(iteration) == pytest.approx(expected(iteration * 8))
    assert args.lr_decay_iters is None  # the saved args are not modified

    # Sample-based runs save train_iters as training filled it in; the samples settings must win.
    samples = Namespace(
        **{
            **vars(args),
            "train_iters": 1000,
            "train_samples": 8000,
            "lr_decay_samples": 8000,
            "lr_warmup_iters": 0,
            "lr_warmup_samples": 400,
            "lr_wsd_decay_samples": None,
        }
    )
    lr = checkpoint_merge.checkpoint_lr_schedule(samples)
    for iteration in (0, 10, 50, 300, 600, 800, 1000):
        assert lr(iteration) == pytest.approx(expected(iteration * 8))


@pytest.mark.parametrize(
    "key, ndim, expected",
    [
        ("decoder.layers.0.mlp.router.weight", 2, (2e-3, 1e-4)),
        ("decoder.layers.0.mlp.experts.experts.weight1", 3, (2.74e-2, 1e-4)),
        ("decoder.layers.0.self_attention.linear_qkv.weight", 2, (2.74e-2, 1e-4)),
        ("decoder.layers.0.self_attention.in_proj.weight.query", 2, (2.74e-2, 1e-4)),
        ("embedding.word_embeddings.weight", 2, (5.5e-3, 1e-4)),
        ("output_layer.weight", 2, (5.5e-3, 1e-4)),
        ("decoder.layers.0.self_attention.conv1d.weight", 3, (5.5e-3, 1e-4)),
        ("decoder.final_layernorm.weight", 1, (5.5e-3, 1e-4)),
    ],
)
def test_lr_group(key, ndim, expected):
    # The megachonk routing: --matrix-lr, --router-lr and --lr, all floored at --min-lr.
    args = Namespace(
        lr=5.5e-3,
        min_lr=1e-4,
        min_lr_mode="absolute",
        matrix_lr=2.74e-2,
        router_lr=2e-3,
        use_orthogonal_updates=True,
        md_router_use_orthogonal_updates=None,
    )
    assert checkpoint_merge._lr_group(key, ndim, args) == pytest.approx(expected)
    # With relative min LRs every group decays by the same fraction.
    relative = Namespace(**{**vars(args), "min_lr_mode": "relative"})
    max_lr, min_lr = checkpoint_merge._lr_group(key, ndim, relative)
    assert min_lr == pytest.approx(max_lr * 1e-4 / 5.5e-3)


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
    # copied from the last checkpoint (5.0). A tensor with its own weights (its LR group) uses them.
    for name, method, coefficients, router_state, own, expected_weight, expected_router in (
        ("mean", "mean", None, "mean", None, 3.0, 3.0),
        ("wsm_mean", "wsm", [0.75, 0.25], "mean", None, 2.0, 3.0),
        ("wsm_method", "wsm", [0.75, 0.25], "method", None, 2.0, 2.0),
        ("wsm_last", "wsm", [0.75, 0.25], "last", None, 2.0, 5.0),
        ("wsm_group", "wsm", [0.75, 0.25], "method", {"weight": [0.25, 0.75]}, 4.0, 2.0),
    ):
        release = tmp_path_dist_ckpt / name / "release"
        checkpoint_merge.merge_checkpoint_directories(
            sources,
            release.parent,
            coefficients,
            merge_metadata={"method": method},
            router_state=router_state,
            tensor_coefficients=own,
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
            "sphere_merge": False,
        }


def _stored_chunks(checkpoint, key):
    storage = _get_filesystem_reader(checkpoint).read_metadata().state_dict_metadata[key]
    return sorted((tuple(chunk.offsets), tuple(chunk.sizes)) for chunk in storage.chunks)


def _chunked(key, tensor, *fragmentations):
    # One shard per grid cell, all saved by the main replica on rank 0.
    rank = torch.distributed.get_rank()
    sizes = [length // parts for length, parts in zip(tensor.shape, fragmentations)]
    cells = [[]]
    for axis, parts in enumerate(fragmentations):
        cells = [cell + [(axis, index, parts)] for cell in cells for index in range(parts)]
    shards = []
    for cell in cells:
        index = tuple(slice(i * size, (i + 1) * size) for (_, i, _), size in zip(cell, sizes))
        shards.append(
            ShardedTensor.from_rank_offsets(key, tensor[index].clone(), *cell, replica_id=rank)
        )
    return shards


def test_merge_keeps_source_chunks(initialized_model_parallel, tmp_path_dist_ckpt):
    # A reader loads each stored chunk whole, so the merge stores the chunks of the last source:
    # experts split per expert (like EP), a dense weight split on both axes. The first source
    # stores whole tensors, so reading it into the last source's chunks also works.
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    sources = []
    for value, iteration, chunked in ((1.0, 100, False), (3.0, 200, True)):
        source = tmp_path_dist_ckpt / "chunks" / f"iter_{iteration:07d}"
        if torch.distributed.get_rank() == 0:
            source.mkdir(parents=True)
        torch.distributed.barrier()
        experts, dense = (base + value).to(torch.bfloat16), base * value
        state = {
            "experts": _chunked("experts", experts, *((4, 1) if chunked else (1, 1))),
            "dense": _chunked("dense", dense, *((2, 3) if chunked else (1, 1))),
            "iteration": iteration,
        }
        dist_checkpointing.save(state, str(source))
        sources.append(str(source))

    output = tmp_path_dist_ckpt / "chunks" / "merged"
    checkpoint_merge.merge_checkpoint_directories(sources, output)
    release = str(output / "release")
    for key in ("experts", "dense"):
        assert _stored_chunks(release, key) == _stored_chunks(sources[-1], key)
    rank = torch.distributed.get_rank()
    templates = {
        key: ShardedTensor.from_rank_offsets(key, torch.zeros(4, 6), replica_id=rank)
        for key in ("experts", "dense")
    }
    loaded = dist_checkpointing.load(templates, release)
    torch.testing.assert_close(loaded["experts"], base + 2.0)
    torch.testing.assert_close(loaded["dense"], base * 2.0)


def _unit(tensor):
    # Rows on the unit sphere: every slice along the input axis 1 has length 1.
    return tensor / tensor.norm(dim=1, keepdim=True)


def test_sphere_merge(initialized_model_parallel, tmp_path_dist_ckpt):
    # MDDecoupling saves W = bare * phi(row)[:, None] * phi(col)[None, :] with unit rows of bare;
    # merged experts are [experts, in, out] with row gains [experts, out], col gains [experts, in].
    phi = torch.nn.functional.softplus
    generator = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=generator)

    rank = torch.distributed.get_rank()
    coefficients = [0.25, 0.75]
    root = tmp_path_dist_ckpt / "sphere"
    sources, values = [], []
    for iteration in (100, 200):
        v = {
            "dense": _unit(randn(4, 6)),
            "dense.row": randn(4),
            "dense.col": randn(6),
            "experts": _unit(randn(2, 3, 4)),
            "experts.row": randn(2, 4),
            "experts.col": randn(2, 3),
            # A piece of a split weight: its own row gain, the column gain of the whole weight.
            "qkv.weight.query": _unit(randn(4, 6)),
            "qkv.weight.query.row": randn(4),
            "qkv.weight.col": randn(6),
            # No gains, rows of length 2: the merged rows go back to the source length.
            "embedding": 2 * _unit(randn(5, 6)),
            "norm": randn(6),
        }
        values.append(v)
        fused = {
            "dense": v["dense"] * phi(v["dense.row"])[:, None] * phi(v["dense.col"])[None, :],
            "experts": v["experts"]
            * phi(v["experts.row"])[:, None, :]
            * phi(v["experts.col"])[:, :, None],
            "qkv.weight.query": v["qkv.weight.query"]
            * phi(v["qkv.weight.query.row"])[:, None]
            * phi(v["qkv.weight.col"])[None, :],
            "embedding": v["embedding"],
            "norm": v["norm"],
        }
        gains = {
            "row_gain.dense": v["dense.row"],
            "col_gain.dense": v["dense.col"],
            "row_gain.experts": v["experts.row"],
            "col_gain.experts": v["experts.col"],
            "row_gain.qkv.weight.query": v["qkv.weight.query.row"],
            "col_gain.qkv.weight": v["qkv.weight.col"],
        }
        source = root / f"iter_{iteration:07d}"
        if rank == 0:
            source.mkdir(parents=True)
        torch.distributed.barrier()
        state = {key: _chunked(key, tensor, *([1] * tensor.ndim)) for key, tensor in fused.items()}
        # The dense rows span two chunks on the input axis, as with row-parallel TP.
        state["dense"] = _chunked("dense", fused["dense"], 1, 2)
        state["experts"] = _chunked("experts", fused["experts"], 2, 1, 1)
        for name, gain in gains.items():
            key = f"optimizer.state.{name}"
            state[key] = _chunked(key, gain, *([1] * gain.ndim))
        state["args"] = Namespace(
            gain_parametrization="softplus",
            hypersphere_mode="row",
            hypersphere_router_mode=None,
            hypersphere_embedding_mode="row",
        )
        state["iteration"] = iteration
        dist_checkpointing.save(state, str(source))
        sources.append(str(source))

    output = root / "merged"
    checkpoint_merge.merge_checkpoint_directories(
        sources, output, coefficients, merge_metadata={"method": "wsm"}, sphere_merge=True
    )
    release = str(output / "release")
    assert _stored_chunks(release, "dense") == _stored_chunks(sources[-1], "dense")
    assert dist_checkpointing.load_common_state_dict(release)["checkpoint_merge_metadata"][
        "sphere_merge"
    ]

    def mean(name, fn=lambda x: x):
        return sum(c * fn(v[name]) for c, v in zip(coefficients, values))

    # Rows are scaled back to their source length: 1 for the matrices with gains, 2 for the
    # embedding.
    expected = {
        "dense": _unit(mean("dense"))
        * mean("dense.row", phi)[:, None]
        * mean("dense.col", phi)[None, :],
        "experts": _unit(mean("experts"))
        * mean("experts.row", phi)[:, None, :]
        * mean("experts.col", phi)[:, :, None],
        "qkv.weight.query": _unit(mean("qkv.weight.query"))
        * mean("qkv.weight.query.row", phi)[:, None]
        * mean("qkv.weight.col", phi)[None, :],
        "embedding": 2 * _unit(mean("embedding")),
        "norm": mean("norm"),
    }
    templates = {
        key: ShardedTensor.from_rank_offsets(key, torch.zeros_like(value), replica_id=rank)
        for key, value in expected.items()
    }
    loaded = dist_checkpointing.load(templates, release)
    for key, value in expected.items():
        torch.testing.assert_close(loaded[key], value, rtol=1e-5, atol=1e-6, msg=key)


def test_sphere_merge_flat(initialized_model_parallel, tmp_path_dist_ckpt):
    # In flat mode MDDecoupling normalizes each piece on its own: Q, K and V of every query group
    # together, each GLU half, each expert (and each GLU half of an expert).
    phi = torch.nn.functional.softplus
    generator = torch.Generator().manual_seed(0)
    rank = torch.distributed.get_rank()
    radius = 2.0
    prefix = "decoder.layers.0."
    # 2 query groups of [Q (2 heads x 2 channels), K (2), V (2)] rows.
    qkv_rows = [[0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 12, 13], [6, 7, 14, 15]]
    layout = {
        # key: (shape, pieces as indices, chunk fragmentation)
        "self_attention.linear_qkv.weight": (
            (16, 6),
            [(torch.tensor(rows),) for rows in qkv_rows],
            (1, 1),
        ),
        # GLU halves of 3 rows, stored in chunks of 2 rows that do not line up with them.
        "mlp.linear_fc1.weight": ((6, 6), [(slice(0, 3),), (slice(3, 6),)], (3, 2)),
        "mlp.linear_fc2.weight": ((6, 3), [(slice(None),)], (1, 1)),
        # Merged experts [experts, in, out]: GLU halves of each expert's output.
        "mlp.experts.experts.weight1": (
            (2, 3, 4),
            [(e, slice(None), half) for e in range(2) for half in (slice(0, 2), slice(2, 4))],
            (2, 1, 1),
        ),
        "mlp.experts.experts.weight2": ((2, 2, 3), [(e,) for e in range(2)], (2, 1, 1)),
    }
    args = Namespace(
        gain_parametrization="softplus",
        hypersphere_mode="flat",
        hypersphere_router_mode=None,
        hypersphere_embedding_mode=None,  # the embedding is not on a sphere: averaged directly
        muon_split_qkv=True,
        muon_split_fc1=True,
        num_attention_heads=4,
        num_query_groups=2,
        group_query_attention=True,
        kv_channels=2,
        hidden_size=6,
        attention_output_gate=False,
        ffn_hidden_size=3,
        moe_ffn_hidden_size=2,
    )

    def gain_shapes(shape):
        return ((shape[0],), (shape[1],)) if len(shape) == 2 else ((shape[0], shape[2]), shape[:2])

    def apply(bare, row, col):
        if bare.ndim == 2:
            return bare * phi(row)[:, None] * phi(col)[None, :]
        return bare * phi(row)[:, None, :] * phi(col)[:, :, None]

    def on_pieces(tensor, pieces):
        tensor = tensor.clone()
        for piece in pieces:
            tensor[piece] = tensor[piece] * (radius / tensor[piece].norm())
        return tensor

    coefficients = [0.25, 0.75]
    sources, values = [], []
    for iteration in (100, 200):
        v = {}
        state = {"args": args, "iteration": iteration}
        for name, (shape, pieces, fragmentation) in layout.items():
            key = prefix + name
            bare = on_pieces(torch.randn(*shape, generator=generator), pieces)
            row_shape, col_shape = gain_shapes(shape)
            row = torch.randn(*row_shape, generator=generator)
            col = torch.randn(*col_shape, generator=generator)
            v[key] = (bare, row, col)
            state[key] = _chunked(key, apply(bare, row, col), *fragmentation)
            for kind, gain in (("row", row), ("col", col)):
                gain_key = f"optimizer.state.{kind}_gain.{key}"
                state[gain_key] = _chunked(gain_key, gain, *([1] * gain.ndim))
        embedding = torch.randn(5, 6, generator=generator)
        v["embedding.word_embeddings.weight"] = embedding
        state["embedding.word_embeddings.weight"] = _chunked(
            "embedding.word_embeddings.weight", embedding, 1, 1
        )
        values.append(v)
        source = tmp_path_dist_ckpt / "sphere_flat" / f"iter_{iteration:07d}"
        if rank == 0:
            source.mkdir(parents=True)
        torch.distributed.barrier()
        dist_checkpointing.save(state, str(source))
        sources.append(str(source))

    output = tmp_path_dist_ckpt / "sphere_flat" / "merged"
    checkpoint_merge.merge_checkpoint_directories(sources, output, coefficients, sphere_merge=True)
    release = str(output / "release")
    key = prefix + "mlp.linear_fc1.weight"
    assert _stored_chunks(release, key) == _stored_chunks(sources[-1], key)

    expected = {
        "embedding.word_embeddings.weight": sum(
            c * v["embedding.word_embeddings.weight"] for c, v in zip(coefficients, values)
        )
    }
    for name, (_, pieces, _) in layout.items():
        key = prefix + name
        bare = on_pieces(sum(c * v[key][0] for c, v in zip(coefficients, values)), pieces)
        row = sum(c * phi(v[key][1]) for c, v in zip(coefficients, values))
        col = sum(c * phi(v[key][2]) for c, v in zip(coefficients, values))
        # The gains are merged after phi, so apply them without phi again.
        if bare.ndim == 2:
            expected[key] = bare * row[:, None] * col[None, :]
        else:
            expected[key] = bare * row[:, None, :] * col[:, :, None]
    templates = {
        key: ShardedTensor.from_rank_offsets(key, torch.zeros_like(value), replica_id=rank)
        for key, value in expected.items()
    }
    loaded = dist_checkpointing.load(templates, release)
    for key, value in expected.items():
        torch.testing.assert_close(loaded[key], value, rtol=1e-5, atol=1e-6, msg=key)


@pytest.mark.parametrize(
    "key, modes, expected",
    [
        ("decoder.layers.0.mlp.linear_fc1.weight", dict(hypersphere_mode="row"), "row"),
        ("decoder.layers.0.mlp.linear_fc1.weight", dict(hypersphere_mode="none"), None),
        # The router and the embedding/output layer have their own modes, as in MDDecoupling.
        (
            "decoder.layers.0.mlp.router.weight",
            dict(hypersphere_mode="none", hypersphere_router_mode="row"),
            "row",
        ),
        (
            "embedding.word_embeddings.weight",
            dict(hypersphere_mode="row", hypersphere_embedding_mode="external"),
            None,
        ),
        ("output_layer.weight", dict(hypersphere_mode="row"), None),
        ("decoder.layers.0.mlp.linear_fc1.weight", dict(hypersphere_mode="flat"), "flat"),
        ("decoder.layers.0.mlp.linear_fc1.weight", dict(hypersphere_mode="col"), ValueError),
        ("embedding.word_embeddings.weight", dict(hypersphere_embedding_mode="embed"), ValueError),
    ],
)
def test_sphere_mode(key, modes, expected):
    args = Namespace(**{"hypersphere_mode": None, **modes})
    if expected is ValueError:
        with pytest.raises(ValueError, match="row and flat spheres only"):
            checkpoint_merge._sphere_mode(key, args)
    else:
        assert checkpoint_merge._sphere_mode(key, args) == expected


def test_irregular_source_chunks_stay_one_chunk(monkeypatch):
    chunks = [SimpleNamespace(offsets=(0,), sizes=(3,)), SimpleNamespace(offsets=(3,), sizes=(5,))]
    storage = {"weight": SimpleNamespace(chunks=chunks)}
    reader = SimpleNamespace(read_metadata=lambda: SimpleNamespace(state_dict_metadata=storage))
    monkeypatch.setattr(checkpoint_merge, "_get_filesystem_reader", lambda source: reader)
    meta = ShardedTensor.from_rank_offsets("weight", torch.empty(8, device="meta")).without_data()
    (chunk,) = checkpoint_merge._source_chunks({"weight": meta}, "source").values()
    assert (chunk.global_offset, chunk.local_shape) == ((0,), (8,))


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
    rank = torch.distributed.get_rank()
    if rank == 0:
        source.mkdir(parents=True)
    torch.distributed.barrier()
    state = {
        # Two chunks, so the running merge must keep the source layout.
        "weight": _chunked("weight", weight, 2),
        "router.expert_bias": ShardedTensor.from_rank_offsets(
            "router.expert_bias", weight * 2, replica_id=rank
        ),
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
    rank = torch.distributed.get_rank()
    templates = {
        key: ShardedTensor.from_rank_offsets(key, torch.zeros(8), replica_id=rank)
        for key in ("weight", "router.expert_bias")
    }
    loaded = dist_checkpointing.load(templates, merged)
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
    assert _stored_chunks(merged, "weight") == _stored_chunks(sources[-1], "weight")
    if method == "wsm":
        assert [p.name for p in (output / "wsm_running_mean").glob("iter_*")] == ["iter_0000400"]
        mean = str(output / "wsm_running_mean" / "iter_0000400")
        assert _stored_chunks(mean, "weight") == _stored_chunks(sources[-1], "weight")

    # Folding the same checkpoint again changes nothing; older or mismatched input is refused.
    update(sources[3])
    assert [path.name for path in output.glob("iter_*")] == ["iter_0000400"]
    older, _ = _save_update_source(root / "run", 50)
    with pytest.raises(ValueError, match="not newer"):
        update(older)
    with pytest.raises(ValueError, match="Settings differ"):
        update(sources[3], router_state="last" if router_state != "last" else "mean")


@pytest.mark.parametrize(
    "method, kwargs, effective",
    [
        ("mean", {}, lambda n: mean_coefficients(n)),
        ("ema", dict(ema_alpha=0.3), lambda n: ema_coefficients(n, 0.3)),
        ("wsm", dict(decay_end=0.1), lambda n: wsm_coefficients(n, 0.1)),
    ],
)
def test_update_sphere_merge_matches_offline(
    initialized_model_parallel, tmp_path_dist_ckpt, method, kwargs, effective
):
    # Folding row-sphere checkpoints one at a time with --sphere-merge equals the offline
    # --sphere-merge of all of them: each running merge keeps its gains and row shrinkage.
    phi = torch.nn.functional.softplus
    generator = torch.Generator().manual_seed(0)
    rank = torch.distributed.get_rank()
    root = tmp_path_dist_ckpt / f"update_sphere_{method}"
    args = Namespace(
        gain_parametrization="softplus",
        hypersphere_mode="row",
        hypersphere_router_mode=None,
        hypersphere_embedding_mode="row",
    )

    def randn(*shape):
        return torch.randn(*shape, generator=generator)

    shapes = {"dense": (4, 6), "experts": (2, 3, 4), "embedding.word_embeddings.weight": (5, 6)}
    shapes["norm"] = (6,)
    sources = []
    for iteration in (100, 200, 300, 400):
        gains = {
            "row_gain.dense": randn(4),
            "col_gain.dense": randn(6),
            "row_gain.experts": randn(2, 4),
            "col_gain.experts": randn(2, 3),
        }
        fused = {
            "dense": _unit(randn(4, 6))
            * phi(gains["row_gain.dense"])[:, None]
            * phi(gains["col_gain.dense"])[None, :],
            "experts": _unit(randn(2, 3, 4))
            * phi(gains["row_gain.experts"])[:, None, :]
            * phi(gains["col_gain.experts"])[:, :, None],
            "embedding.word_embeddings.weight": 2 * _unit(randn(5, 6)),
            "norm": randn(6),
        }
        source = root / "run" / f"iter_{iteration:07d}"
        if rank == 0:
            source.mkdir(parents=True)
        torch.distributed.barrier()
        # Rows split over two groups and two chunks each, so row state spans several workers.
        fragmentation = {"dense": (2, 2), "experts": (2, 1, 1)}
        state = {
            key: _chunked(key, tensor, *fragmentation.get(key, [1] * tensor.ndim))
            for key, tensor in fused.items()
        }
        for name, gain in gains.items():
            key = f"optimizer.state.{name}"
            state[key] = _chunked(key, gain, *([1] * gain.ndim))
        state["args"] = args
        state["iteration"] = iteration
        dist_checkpointing.save(state, str(source))
        sources.append(str(source))

    output = root / "merged"
    for source in sources:
        checkpoint_merge.update_running_merge(output, [source], method, sphere_merge=True, **kwargs)
    offline = root / "offline"
    checkpoint_merge.merge_checkpoint_directories(sources, offline, effective(4), sphere_merge=True)

    def load(checkpoint):
        templates = {
            key: ShardedTensor.from_rank_offsets(key, torch.zeros(shape), replica_id=rank)
            for key, shape in shapes.items()
        }
        return dist_checkpointing.load(templates, str(checkpoint))

    running, expected = load(output / "iter_0000400"), load(offline / "release")
    for key in shapes:
        torch.testing.assert_close(running[key], expected[key], rtol=1e-5, atol=1e-6, msg=key)
    # A plain --update cannot continue a sphere running merge.
    with pytest.raises(ValueError, match="Settings differ"):
        checkpoint_merge.update_running_merge(output, [sources[-1]], method, **kwargs)
