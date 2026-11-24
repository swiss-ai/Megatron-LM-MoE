# Copyright (c) 2026, EPFL / Swiss AI Initiative.

"""Merge Megatron ``torch_dist`` checkpoints with generic checkpoint workers.

Offline, ``--checkpoints`` lists every checkpoint to merge and ``--output`` receives a
``release`` checkpoint. With ``--update``, the new checkpoints are folded into a running merge at
``--output``, so the merge can follow a training run that deletes its old checkpoints.
"""

import argparse
import copy
import gc
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.serialization import load_sharded_metadata
from megatron.core.dist_checkpointing.strategies.torch import _get_filesystem_reader


# Router load-balancing state (expert bias, quantile-balancing betas); see --router-state.
_ROUTER_STATE_TENSOR_NAMES = {"expert_bias", "qb_beta"}
_TRAINING_PREFIXES = (
    "optimizer",
    "opt_param_scheduler",
    "lr_scheduler",
    "rng_state",
    "rerun_state_machine",
    "dataloader",
)
_TRACKER = "latest_checkpointed_iteration.txt"
# Growing-window WSM keeps the running mean of all folded checkpoints next to the export.
_WSM_RUNNING_MEAN_DIR = "wsm_running_mean"
# MDDecoupling keeps its learnable gains in the optimizer state; see --sphere-merge.
_GAIN_STATE = "optimizer.state."
# Axes of a weight chunk that each gain covers. 2D weights are [out, in]; merged experts are
# saved as [experts, in, out], with row gains [experts, out] and column gains [experts, in].
_GAIN_AXES = {("row", 2): (0,), ("col", 2): (1,), ("row", 3): (0, 2), ("col", 3): (0, 1)}


def resolve_sources(checkpoints, iterations=None, min_sources=2):
    """Resolve ``--checkpoints`` (and optional ``--iterations``) to checkpoint directories.

    Either pass iteration/release directories (or roots, resolved through their tracker file),
    or pass exactly one root together with the iterations to take from it.
    """
    checkpoints = list(checkpoints)
    if iterations:
        if len(checkpoints) != 1:
            raise ValueError("--iterations needs exactly one checkpoint root")
        sources = [_resolve_checkpoint_dir(checkpoints[0], step) for step in iterations]
    else:
        sources = [_resolve_checkpoint_dir(root, None) for root in checkpoints]
    if len(sources) < min_sources:
        raise ValueError(f"Checkpoint merging requires at least {min_sources} checkpoints")
    return sources


def source_iterations(sources):
    """Return the iteration of each ``iter_*`` source, or None if any source is not one."""
    names = [Path(source).name for source in sources]
    if not all(name.startswith("iter_") for name in names):
        return None
    return [int(name.removeprefix("iter_")) for name in names]


def _iter_name(iteration):
    return f"iter_{iteration:07d}"


def _resolve_checkpoint_dir(root, step):
    root = Path(root)
    if (root / "metadata.json").is_file() or (root / ".metadata").is_file():
        if step is not None:
            raise ValueError(f"Cannot combine direct checkpoint directory {root} with a step")
        return str(root)

    if step is None:
        tracker = root / "latest_checkpointed_iteration.txt"
        if not tracker.is_file():
            raise FileNotFoundError(f"Checkpoint tracker not found: {tracker}")
        value = tracker.read_text().strip()
        if value == "release":
            return str(root / "release")
        step = int(value)
    return str(root / _iter_name(step))


def _is_training_key(key):
    key = str(key)
    return key.startswith(_TRAINING_PREFIXES)


def _tensor_nbytes(metadata):
    return math.prod(metadata.local_shape) * metadata.dtype.itemsize


def _source_chunks(tensor_metadata, source):
    """Split every tensor into the chunks that ``source`` stored.

    A reader loads each stored chunk whole before it takes its slice. With one chunk per tensor,
    every rank that needs a slice (for example one EP rank's experts) reads the whole tensor, so
    the merge keeps the source layout. A layout that is not a regular grid stays one chunk.
    """
    storage = _get_filesystem_reader(source).read_metadata().state_dict_metadata
    chunks = {}
    for key, meta in tensor_metadata.items():
        shape = tuple(meta.global_shape)
        layout = [(tuple(c.offsets), tuple(c.sizes)) for c in storage[key].chunks]
        sizes = layout[0][1]
        regular = (
            all(chunk_sizes == sizes for _, chunk_sizes in layout)
            and all(size > 0 and length % size == 0 for size, length in zip(sizes, shape))
            and len({offsets for offsets, _ in layout}) == len(layout)
            and len(layout) == math.prod(length // size for size, length in zip(sizes, shape))
            and all(o % size == 0 for offsets, _ in layout for o, size in zip(offsets, sizes))
        )
        if not regular:
            layout = [((0,) * len(shape), shape)]
        for index, (offsets, sizes) in enumerate(layout):
            chunks[(key, index)] = replace(
                meta,
                local_shape=sizes,
                global_offset=offsets,
                axis_fragmentations=tuple(length // size for size, length in zip(sizes, shape)),
            )
    return chunks


def _chunk_group(chunk_key, chunk_meta, mode=None):
    """Return the group of chunks one worker merges together.

    A row-mode sphere (and every other tensor) needs whole rows: the chunks that differ only in
    their offset on the input axis 1. A flat-mode sphere needs the whole matrix, or whole experts.
    """
    offsets = chunk_meta.global_offset
    if mode == "flat":
        return (chunk_key[0], offsets[:1] if len(offsets) == 3 else ())
    return (chunk_key[0], offsets[:1] + offsets[2:])


def _chunk_index(chunk_meta, start):
    """Index of a chunk inside the region of its group that begins at ``start``."""
    return tuple(
        slice(offset - origin, offset - origin + size)
        for offset, origin, size in zip(chunk_meta.global_offset, start, chunk_meta.local_shape)
    )


def _assemble(parts, metas):
    """Join the chunks of a group into one tensor; return it and where it begins."""
    ndim = len(metas[0].global_offset)
    start = tuple(min(meta.global_offset[axis] for meta in metas) for axis in range(ndim))
    stop = tuple(
        max(meta.global_offset[axis] + meta.local_shape[axis] for meta in metas)
        for axis in range(ndim)
    )
    whole = torch.empty([b - a for a, b in zip(start, stop)], dtype=torch.float32)
    for part, meta in zip(parts, metas):
        whole[_chunk_index(meta, start)] = part
    return whole, start


def _is_glu_fc1(key, fused_size, checkpoint_args):
    """Whether ``key`` is a fused GLU FC1, whose output holds the gate and up halves."""
    if "shared_experts.linear_fc1.weight" in key:
        size = getattr(checkpoint_args, "moe_shared_expert_intermediate_size", None)
    elif "experts.weight1" in key or "experts.linear_fc1.weight" in key:
        size = getattr(checkpoint_args, "moe_ffn_hidden_size", None)
    elif "linear_fc1.weight" in key:
        size = getattr(checkpoint_args, "ffn_hidden_size", None)
    else:
        return False
    # The activation flags decide gated_linear_unit; a GLU FC1 has twice the FFN size.
    return size is not None and fused_size == 2 * size


def _qkv_split_shapes(checkpoint_args):
    """Per-query-group block sizes of linear_qkv, as get_megatron_mddecoupling_optimizer sets."""
    heads = checkpoint_args.num_attention_heads
    groups = checkpoint_args.num_query_groups if checkpoint_args.group_query_attention else heads
    kv_channels = checkpoint_args.kv_channels or checkpoint_args.hidden_size // heads
    query = heads // groups * kv_channels
    if getattr(checkpoint_args, "attention_output_gate", False):
        return [query, query, kv_channels, kv_channels]
    return [query, kv_channels, kv_channels]


_MLA_PROJECTIONS = (
    "linear_q_up_proj.weight",
    "linear_kv_up_proj.weight",
    "linear_qkv_down_proj.weight",
)


def _flat_split(key, whole, checkpoint_args):
    """Split a flat-mode matrix into the pieces MDDecoupling normalizes one by one.

    ``whole`` is a matrix [out, in] or merged experts [experts, in, out]. Returns the pieces and
    a function that joins them back, from MDDecoupling's own split helpers.
    """
    from megatron.core.optimizer import md_decoupling

    split_fc1 = getattr(checkpoint_args, "muon_split_fc1", False)
    split_qkv = getattr(checkpoint_args, "muon_split_qkv", False)
    if whole.ndim == 3:
        experts = whole.transpose(1, 2)  # MDDecoupling's merged-expert layout [experts, out, in]
        if split_fc1 and _is_glu_fc1(key, experts.shape[1], checkpoint_args):
            parts, merge = md_decoupling._split_glu_fc1(experts, 1)
        else:
            parts, merge = md_decoupling._split_experts(experts), md_decoupling._merge_experts
        return parts, lambda parts: merge(parts).transpose(1, 2)
    if key.endswith(_MLA_PROJECTIONS):
        raise ValueError(f"--sphere-merge does not support flat-mode MLA projections ({key!r})")
    if ".self_attention.in_proj.weight." in key and not split_qkv:
        # The checkpoint stores the KDA in_proj in pieces; unsplit, MDDecoupling normalizes it whole.
        raise ValueError(f"--sphere-merge needs muon_split_qkv for flat-mode KDA in_proj ({key!r})")
    if split_qkv and "linear_qkv.weight" in key:
        shapes = _qkv_split_shapes(checkpoint_args)
        return (
            md_decoupling._split_qkv(whole, shapes),
            lambda parts: md_decoupling._merge_qkv(parts, whole.shape, shapes),
        )
    if split_fc1 and _is_glu_fc1(key, whole.shape[0], checkpoint_args):
        return md_decoupling._split_glu_fc1(whole, 0)
    return [whole], lambda parts: parts[0]


def _slice_norms(mode, key, parts, metas, checkpoint_args):
    """Return the length of every sphere slice of a group: its rows, or its flat-mode pieces."""
    if mode == "row":
        return _row_norms(parts)
    whole, _ = _assemble(parts, metas)
    pieces, _ = _flat_split(key, whole, checkpoint_args)
    return torch.stack(
        [torch.linalg.vector_norm(piece, dtype=torch.float64) for piece in pieces]
    ).float()


def _rescale_slices(mode, key, parts, metas, factors, checkpoint_args):
    """Multiply every sphere slice of a group, in place, by its factor."""
    if mode == "row":
        for part in parts:
            part.mul_(factors.unsqueeze(1))
        return
    whole, start = _assemble(parts, metas)
    pieces, merge = _flat_split(key, whole, checkpoint_args)
    whole = merge([piece * factor for piece, factor in zip(pieces, factors)])
    for part, meta in zip(parts, metas):
        part.copy_(whole[_chunk_index(meta, start)])


def _gain_keys(key, tensors):
    """Return the row and column gain keys of an MDDecoupling weight, or None where it has none."""
    if f"{_GAIN_STATE}flat_gain.{key}" in tensors:
        raise ValueError(f"--sphere-merge does not support flat gains (found for {key!r})")
    row = f"{_GAIN_STATE}row_gain.{key}"
    col = f"{_GAIN_STATE}col_gain.{key}"
    parent = key.rsplit(".", 1)[0]
    if col not in tensors and parent.endswith(".weight"):
        # Pieces of a split weight (in_proj.weight.query, ...) share the column gain of the whole.
        col = f"{_GAIN_STATE}col_gain.{parent}"
    return (row if row in tensors else None), (col if col in tensors else None)


def _sphere_mode(key, checkpoint_args):
    """Return the MDDecoupling sphere mode of ``key``, or None if it is not on a sphere.

    The mode is chosen as in ``MDDecoupling._resolve_mode``. Row and flat modes are supported;
    col and embed raise an error.
    """
    if not hasattr(checkpoint_args, "hypersphere_mode"):
        raise ValueError("--sphere-merge needs args.hypersphere_mode in the checkpoint")
    router_mode = getattr(checkpoint_args, "hypersphere_router_mode", None)
    if key.endswith("mlp.router.weight") and router_mode is not None:
        mode = router_mode
    elif key.endswith(("embedding.word_embeddings.weight", "output_layer.weight")):
        # Without an embedding mode (unset or "external"), a chained Adam trains these, off-sphere.
        mode = getattr(checkpoint_args, "hypersphere_embedding_mode", None)
        mode = None if mode == "external" else mode
    else:
        mode = checkpoint_args.hypersphere_mode
    if mode in (None, "none"):
        return None
    if mode not in ("row", "flat"):
        raise ValueError(
            f"--sphere-merge supports row and flat spheres only; {key!r} uses mode {mode!r}"
        )
    return mode


def _gain_function(checkpoint_args):
    """Return phi, which maps a raw gain to its multiplier, as MDDecoupling's gain_parametrization."""
    mode = getattr(checkpoint_args, "gain_parametrization", None)
    if mode == "softplus":
        return torch.nn.functional.softplus
    if mode == "direct":
        return lambda gain: gain
    raise ValueError(
        f"--sphere-merge needs args.gain_parametrization in the checkpoint, got {mode!r}"
    )


def _gain_slice(gain, kind, offsets, sizes):
    """Return the part of an effective gain that scales one chunk, shaped to broadcast against it."""
    axes = _GAIN_AXES[(kind, len(offsets))]
    part = gain[tuple(slice(offsets[axis], offsets[axis] + sizes[axis]) for axis in axes)]
    if kind == "row":
        return part.unsqueeze(1)
    return part.unsqueeze(0) if len(offsets) == 2 else part.unsqueeze(2)


def _gain_scale(chunk_meta, gains, gain_keys):
    """Return the product of the row and column gains for one chunk (1 if it has no gains)."""
    scale = torch.ones((), dtype=torch.float32)
    for kind, gain_key in zip(("row", "col"), gain_keys):
        if gain_key is not None:
            part = _gain_slice(
                gains[gain_key], kind, chunk_meta.global_offset, chunk_meta.local_shape
            )
            scale = scale * part
    return scale


def _row_norms(parts):
    """Return the length of every row of a row group, over all its chunks along axis 1."""
    return sum((part * part).sum(dim=1) for part in parts).sqrt()


def _report_sphere_merge(sphere_groups):
    """Print, on rank 0, how many matrices --sphere-merge scaled back."""
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, {group[0] for group in sphere_groups})
    if dist.get_rank() == 0:
        print(f"> sphere merge: scaled back {len(set().union(*gathered))} matrices")


def _load_whole(source, keys, metadata):
    """Load whole tensors (the small gain vectors). Several ranks may load the same tensor."""
    templates = {key: metadata[key].without_data() for key in keys}
    loaded = dist_checkpointing.load(templates, source, validate_access_integrity=False)
    return {key: loaded[key].float() for key in keys}


def _owner_map(items, world_size, size_fn):
    """Greedily balance metadata entries by size across checkpoint workers."""
    loads = [0] * world_size
    owners = {}
    for key, value in sorted(items, key=lambda item: (-size_fn(item[1]), str(item[0]))):
        owner = min(range(world_size), key=lambda rank: (loads[rank], rank))
        owners[key] = owner
        loads[owner] += size_fn(value)
    return owners


def _linear_multipliers(progress, end_multiplier):
    if not 0 <= end_multiplier <= 1:
        raise ValueError(f"LR end multiplier must be in [0, 1], got {end_multiplier}")
    return [1.0 - (1.0 - end_multiplier) * position for position in progress]


def _linear_interval_mean(left, right, decay_start, decay_steps, end_multiplier):
    """Average a per-training-step linear schedule over ``[left, right)``."""
    count = right - left
    pre_decay_last = min(right - 1, decay_start)
    total = max(pre_decay_last - left + 1, 0)

    linear_first = max(left, decay_start + 1)
    linear_last = min(right - 1, decay_start + decay_steps)
    linear_count = max(linear_last - linear_first + 1, 0)
    if linear_count:
        position_sum = (linear_first + linear_last) * linear_count / 2 - decay_start * linear_count
        total += linear_count - (1.0 - end_multiplier) * position_sum / decay_steps

    post_decay_count = max(right - max(left, decay_start + decay_steps + 1), 0)
    total += post_decay_count * end_multiplier
    return total / count


@dataclass(frozen=True)
class LinearDecay:
    """A linear LR decay from multiplier 1 to ``end`` over ``steps`` training steps.

    ``start`` is the last step at multiplier 1. None means the decay starts at the synthetic
    checkpoint one interval before the first merged checkpoint.
    """

    steps: int
    end: float
    start: int | None = None


def _check_count(num_checkpoints):
    if num_checkpoints < 1:
        raise ValueError("Checkpoint merging requires at least one checkpoint")


def mean_coefficients(num_checkpoints):
    """Uniform weights."""
    _check_count(num_checkpoints)
    return [1.0 / num_checkpoints] * num_checkpoints


def ema_coefficients(num_checkpoints, alpha, bias_correction=False):
    """Weights of an exponential moving average over the checkpoints, oldest first.

    The average starts at the first checkpoint and then takes ``avg = alpha * new +
    (1 - alpha) * avg`` for each later one, so the first checkpoint keeps ``(1 - alpha)**(n - 1)``.
    The decay is per checkpoint, not per training step. This is Eq. (4) of Li et al. 2025,
    "Model Merging in Pre-training of Large Language Models" (arXiv:2505.12082).

    With ``bias_correction`` the average starts at zero instead, as Adam's moments do, and is
    divided by ``1 - (1 - alpha)**n``. Every checkpoint then gets the geometric weight
    ``alpha * (1 - alpha)**(n - i)``, normalized, so the oldest one is not weighted up.
    """
    _check_count(num_checkpoints)
    if not 0 < alpha <= 1:
        raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")
    geometric = [
        alpha * (1 - alpha) ** (num_checkpoints - 1 - index) for index in range(num_checkpoints)
    ]
    if bias_correction:
        total = 1 - (1 - alpha) ** num_checkpoints
        return [weight / total for weight in geometric]
    return [(1 - alpha) ** (num_checkpoints - 1)] + geometric[1:]


def wsm_coefficients(num_checkpoints, decay_end=1e-4, source_decay=None, iterations=None):
    """WSM weights that emulate a linear decay to ``decay_end`` over the merged checkpoints.

    Each checkpoint is the endpoint of one coarse update. A synthetic checkpoint one interval
    before the first gets weight zero, so it is never loaded. If the checkpoints were already
    produced under ``source_decay``, that decay is divided out first; this needs the
    checkpoint ``iterations``.
    """
    _check_count(num_checkpoints)
    desired = _linear_multipliers(
        [interval / num_checkpoints for interval in range(num_checkpoints)], decay_end
    )

    if source_decay is None:
        original = [1.0] * num_checkpoints
    else:
        if iterations is None:
            raise ValueError("A source decay needs the checkpoint iterations")
        if num_checkpoints < 2:
            raise ValueError("A source decay needs at least two checkpoints")
        iterations = list(iterations)
        if len(iterations) != num_checkpoints:
            raise ValueError("Provide exactly one iteration per checkpoint")
        if any(left >= right for left, right in zip(iterations, iterations[1:])):
            raise ValueError("Checkpoint iterations must be strictly increasing")
        if source_decay.steps <= 0:
            raise ValueError("Source decay steps must be positive")
        synthetic_step = iterations[0] - (iterations[1] - iterations[0])
        decay_start = synthetic_step if source_decay.start is None else source_decay.start
        original = [
            _linear_interval_mean(left, right, decay_start, source_decay.steps, source_decay.end)
            for left, right in zip([synthetic_step, *iterations[:-1]], iterations)
        ]
        if any(multiplier == 0 for multiplier in original):
            raise ValueError("Cannot cancel a source decay whose multiplier reaches zero")
        original = [multiplier / original[0] for multiplier in original]

    ratios = [target / source for target, source in zip(desired, original)]
    coefficients = [left - right for left, right in zip(ratios, ratios[1:])]
    coefficients.append(ratios[-1])
    return coefficients


def _validate_source_metadata(source, canonical_tensors):
    source_tensors = dist_checkpointing.load_tensors_metadata(source)
    for key, expected in canonical_tensors.items():
        if key not in source_tensors:
            raise KeyError(f"Model tensor {key!r} is missing from {source}")
        actual = source_tensors[key]
        if actual.global_shape != expected.global_shape:
            raise ValueError(
                f"Global shape mismatch for {key!r} in {source}: "
                f"{actual.global_shape} != {expected.global_shape}"
            )


def _load_assigned(source, templates):
    # Every rank must enter DCP load collectives, including workers with no assigned entries.
    loaded = dist_checkpointing.load(
        {key: value.without_data() for key, value in templates.items()},
        source,
        validate_access_integrity=True,
    )
    # dist_checkpointing.load merges checkpoint common state into its result.
    return {key: loaded[key] for key in templates}


def _filtered_common_state(source):
    """Filter out training-specific keys from the common state of a checkpoint."""
    common = copy.deepcopy(dist_checkpointing.load_common_state_dict(source))
    common = {key: value for key, value in common.items() if not _is_training_key(key)}

    checkpoint_args = common.get("args")
    if checkpoint_args is not None:
        for name, value in (("no_save_optim", True), ("no_save_rng", True), ("finetune", True)):
            setattr(checkpoint_args, name, value)
    return common


def merge_checkpoint_directories(
    sources,
    output,
    coefficients=None,
    merge_metadata=None,
    router_state="mean",
    save_fp32=True,
    sphere_merge=False,
):
    """Combine model tensors into ``output/release``, using every distributed rank as a worker.

    ``router_state`` selects how the router balancing state (``expert_bias``, ``qb_beta``) is
    merged: "mean" averages it uniformly, "method" uses ``coefficients`` like every other
    tensor, and "last" copies it from the last checkpoint.

    Merged floating-point tensors are saved in fp32 (``save_fp32``, the default) or in their
    source dtype. Keep fp32 when the output is merged again, for example as a running average,
    because rounding it to bf16 at every update discards changes smaller than about 0.4% of a
    weight. A bf16 model still loads an fp32 checkpoint; the loader casts to the model's dtype.

    ``sphere_merge`` is for MDDecoupling (hypersphere) training, which saves every matrix as
    ``bare * phi(row_gain)[:, None] * phi(col_gain)[None, :]`` with ``bare`` on a sphere and the
    gains in the optimizer state. Averaging these weights directly shortens ``bare`` wherever
    the checkpoints point in different directions. With ``sphere_merge``, every matrix on a
    sphere is split into ``bare`` and its effective gains ``phi(g)``. These are averaged
    separately, every sphere slice of the averaged ``bare`` is scaled back to its average source
    length, and the gains are applied again; a matrix without gains, such as the embedding, is
    only scaled back. A slice is a row in row mode, and in flat mode the whole matrix or each
    piece MDDecoupling normalizes on its own (an expert, a GLU half, Q, K or V). All other
    tensors are averaged directly. Which matrices are on a sphere, and in which mode, comes from
    the checkpoint args (``hypersphere_mode``, ``hypersphere_router_mode``,
    ``hypersphere_embedding_mode``), as MDDecoupling chooses it. Col and embed modes raise an
    error.
    """
    sources = list(sources)
    if len(sources) < 2:
        raise ValueError("Checkpoint merging requires at least two checkpoints")
    if coefficients is None:
        coefficients = mean_coefficients(len(sources))
    router_coefficients = _router_coefficients(
        router_state, coefficients, mean_coefficients(len(sources))
    )
    metadata = {
        **copy.deepcopy(merge_metadata or {}),
        "source_checkpoints": sources,
        "coefficients": list(coefficients),
        "router_state": router_state,
        "save_fp32": save_fp32,
        "sphere_merge": sphere_merge,
    }
    output = Path(output)
    _merge(
        sources,
        output / "release",
        coefficients,
        router_coefficients,
        metadata,
        save_fp32,
        sphere_merge=sphere_merge,
    )
    if dist.get_rank() == 0:
        (output / _TRACKER).write_text("release")
        print(f"> saved merged release checkpoint to {output}")
    dist.barrier()


def _router_coefficients(router_state, coefficients, mean):
    if router_state == "mean":
        return mean
    if router_state == "method":
        return coefficients
    if router_state == "last":
        return [0.0] * (len(coefficients) - 1) + [1.0]
    raise ValueError(f"router_state must be 'mean', 'method' or 'last', got {router_state!r}")


def _merge(
    sources,
    checkpoint_dir,
    coefficients,
    router_coefficients,
    metadata,
    save_fp32,
    sphere_merge=False,
):
    """Write ``sum(coefficients[i] * sources[i])`` as a new checkpoint at ``checkpoint_dir``.

    With ``sphere_merge``, matrices on a sphere are merged as described in
    ``merge_checkpoint_directories``.
    """
    if not len(sources) == len(coefficients) == len(router_coefficients):
        raise ValueError("Provide exactly one coefficient per checkpoint")
    for name, values in (("Checkpoint", coefficients), ("Router state", router_coefficients)):
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError(f"{name} coefficients must sum to 1, got {sum(values)}")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    first_source = str(sources[0])

    all_metadata = dist_checkpointing.load_tensors_metadata(first_source)
    tensor_metadata = {
        key: value for key, value in all_metadata.items() if not _is_training_key(key)
    }
    if not tensor_metadata:
        raise ValueError(f"No model tensors found in {first_source}")

    sharded_metadata = load_sharded_metadata(first_source)
    object_metadata = {
        key: value
        for key, value in sharded_metadata.items()
        if isinstance(value, ShardedObject) and not _is_training_key(value.key)
    }

    # Work on the chunks of the last source, the newest training checkpoint, and save the merge
    # with the same chunks, so it reads like that checkpoint.
    common_state = _filtered_common_state(str(sources[-1]))
    checkpoint_args = common_state.get("args")

    # --sphere-merge: the sphere mode and gain keys of every matrix MDDecoupling keeps on a
    # sphere. Every rank checks every key, so a bad setting stops all ranks before the first
    # collective load.
    sphere_tensors = {}
    for key, meta in tensor_metadata.items() if sphere_merge else ():
        shape = tuple(meta.global_shape)
        if not meta.dtype.is_floating_point or len(shape) not in (2, 3):
            continue
        gain_keys = _gain_keys(key, all_metadata)
        # MDDecoupling manages 2D matrices and merged 3D experts (which have gains); other 3D
        # tensors, such as conv1d weights, are not on a sphere.
        mode = (
            None
            if gain_keys == (None, None) and len(shape) == 3
            else _sphere_mode(key, checkpoint_args)
        )
        if mode is None:
            continue
        for kind, gain_key in zip(("row", "col"), gain_keys):
            expected = tuple(shape[axis] for axis in _GAIN_AXES[(kind, len(shape))])
            if gain_key is not None and tuple(all_metadata[gain_key].global_shape) != expected:
                raise ValueError(
                    f"{gain_key!r} has shape {tuple(all_metadata[gain_key].global_shape)}, "
                    f"expected {expected} for {key!r} {shape}"
                )
        sphere_tensors[key] = (mode, gain_keys)
    phi = None
    if any(key for _, keys in sphere_tensors.values() for key in keys):
        phi = _gain_function(checkpoint_args)

    # Work on the chunks of the last source, the newest training checkpoint, and save the merge
    # with the same chunks, so it reads like that checkpoint.
    chunk_metadata = _source_chunks(tensor_metadata, str(sources[-1]))
    # One worker gets every chunk of a group, so it can measure whole sphere slices.
    groups = {}
    for key, value in chunk_metadata.items():
        mode = sphere_tensors.get(key[0], (None, None))[0]
        groups.setdefault(_chunk_group(key, value, mode), []).append(key)
    group_sizes = {
        group: sum(_tensor_nbytes(chunk_metadata[key]) for key in keys)
        for group, keys in groups.items()
    }
    group_owners = _owner_map(list(group_sizes.items()), world_size, lambda size: size)
    object_owners = _owner_map(list(object_metadata.items()), world_size, lambda _: 1)
    my_groups = {group: keys for group, keys in groups.items() if group_owners[group] == rank}
    assigned_tensors = {key: chunk_metadata[key] for keys in my_groups.values() for key in keys}
    assigned_objects = {
        key: value for key, value in object_metadata.items() if object_owners[key] == rank
    }
    sphere_groups = {
        group: sphere_tensors[group[0]] for group in my_groups if group[0] in sphere_tensors
    }
    gain_keys_to_load = sorted({k for _, keys in sphere_groups.values() for k in keys if k})

    accumulators = {}
    static_tensors = {}
    gain_sums = {}  # merged effective gains phi(g), per gain key
    norm_sums = {}  # merged source lengths of the sphere slices of bare, per group
    for source_index, source in enumerate(map(str, sources)):
        _validate_source_metadata(source, tensor_metadata)
        loaded = _load_assigned(source, assigned_tensors)
        coefficient = coefficients[source_index]
        if sphere_merge:
            # Every rank joins this collective load, including ranks with no gains to load.
            gains = {
                key: phi(value)
                for key, value in _load_whole(source, gain_keys_to_load, all_metadata).items()
            }
            for key, gain in gains.items():
                if source_index == 0:
                    gain_sums[key] = gain * coefficient
                else:
                    gain_sums[key].add_(gain, alpha=coefficient)
            for group, (mode, gain_keys) in sphere_groups.items():
                keys = groups[group]
                bare = {}
                for key in keys:
                    scale = _gain_scale(chunk_metadata[key], gains, gain_keys)
                    # A zero gain leaves a zero weight; dividing it by inf keeps bare at zero.
                    divisor = scale.masked_fill(scale == 0, float("inf"))
                    bare[key] = loaded.pop(key).float() / divisor
                norms = _slice_norms(
                    mode,
                    group[0],
                    list(bare.values()),
                    [chunk_metadata[key] for key in keys],
                    checkpoint_args,
                )
                for key in keys:
                    if source_index == 0:
                        accumulators[key] = bare[key].mul_(coefficient)
                    else:
                        accumulators[key].add_(bare[key], alpha=coefficient)
                if source_index == 0:
                    norm_sums[group] = norms * coefficient
                else:
                    norm_sums[group].add_(norms, alpha=coefficient)
                del bare
        for key, tensor in loaded.items():
            if tensor.is_floating_point():
                is_router_state = key[0].rsplit(".", 1)[-1] in _ROUTER_STATE_TENSOR_NAMES
                weights = router_coefficients if is_router_state else coefficients
                weight = weights[source_index]
                if source_index == 0:
                    accumulators[key] = tensor.float().mul_(weight)
                else:
                    accumulators[key].add_(tensor, alpha=weight)
            elif source_index == 0:
                static_tensors[key] = tensor.cpu().clone()
            elif not torch.equal(static_tensors[key], tensor.cpu()):
                raise ValueError(f"Non-floating model tensor {key[0]!r} differs in {source}")
        del loaded
        gc.collect()
        if rank == 0:
            print(f"> accumulated checkpoint {source_index + 1}/{len(sources)}: {source}")

    # Scale each sphere slice of the merged bare direction back to its merged source length, as
    # MDDecoupling projects it, then apply the merged gains again.
    for group, (mode, gain_keys) in sphere_groups.items():
        keys = groups[group]
        parts = [accumulators[key] for key in keys]
        metas = [chunk_metadata[key] for key in keys]
        current = _slice_norms(mode, group[0], parts, metas, checkpoint_args)
        target = norm_sums.pop(group)
        factors = torch.where(current > 0, target / current, torch.zeros_like(current))
        _rescale_slices(mode, group[0], parts, metas, factors, checkpoint_args)
        for key, meta in zip(keys, metas):
            accumulators[key].mul_(_gain_scale(meta, gain_sums, gain_keys))
    if sphere_merge:
        _report_sphere_merge(sphere_groups)

    output_tensors = {}
    for key, tensor_meta in assigned_tensors.items():
        if key in accumulators:
            data = accumulators[key].to(torch.float32 if save_fp32 else tensor_meta.dtype)
        else:
            data = static_tensors[key].to(tensor_meta.dtype)
        output_tensors[key] = replace(tensor_meta, data=data, dtype=data.dtype, replica_id=0)

    loaded_objects = _load_assigned(first_source, assigned_objects)
    output_objects = {
        key: replace(object_meta, data=loaded_objects[key], replica_id=0)
        for key, object_meta in assigned_objects.items()
    }

    state_dict = common_state
    state_dict["checkpoint_merge_metadata"] = copy.deepcopy(metadata)
    for key, value in {**output_tensors, **output_objects}.items():
        if key in state_dict:
            raise KeyError(f"Sharded model key collides with common state: {key!r}")
        state_dict[key] = value

    checkpoint_dir = Path(checkpoint_dir)
    if rank == 0:
        if checkpoint_dir.exists():
            raise FileExistsError(f"Output checkpoint already exists: {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True)
    dist.barrier()

    content_metadata = dist_checkpointing.load_content_metadata(first_source) or {}
    content_metadata = {
        key: value
        for key, value in content_metadata.items()
        if not str(key).startswith("distrib_optim")
    }
    dist_checkpointing.save(
        state_dict,
        str(checkpoint_dir),
        validate_access_integrity=True,
        content_metadata=content_metadata,
    )
    dist.barrier()


def update_running_merge(
    output,
    sources,
    method,
    ema_alpha=None,
    ema_bias_correction=False,
    decay_end=1e-4,
    router_state="mean",
):
    """Fold ``sources`` (``iter_*`` directories, oldest first) into the running merge at ``output``.

    ``output`` is an ordinary checkpoint root: its tracker names ``iter_N``, the merged model up to
    the newest folded checkpoint, saved in fp32. The first call creates it. Each call checks that
    the settings match the stored ones, skips checkpoints that were already folded, and refuses
    older ones. The tracker is replaced only after the new merge is complete, so an interrupted
    update leaves the previous merge in place and can simply be run again.

    mean, ema (with or without bias correction) and wsm are supported. wsm uses a window that grows
    from the first folded checkpoint; it keeps the running mean in ``output/wsm_running_mean``
    and exports ``(1 - e)(N - 1)/N * mean(M_1..M_{N-1}) + (1 - (1 - e)(N - 1)/N) * M_N``.
    """
    if method not in ("mean", "ema", "wsm"):
        raise ValueError(f"Unknown merge method: {method!r}")
    settings = {
        "method": method,
        "ema_alpha": ema_alpha if method == "ema" else None,
        "ema_bias_correction": bool(ema_bias_correction) if method == "ema" else False,
        "decay_end": decay_end if method == "wsm" else None,
        "router_state": router_state,
    }
    if method == "ema" and ema_alpha is None:
        raise ValueError("EMA needs an alpha")
    sources = [str(source) for source in sources]
    iterations = source_iterations(sources)
    if iterations is None:
        raise ValueError("Updating a running merge needs iteration checkpoints (iter_*)")
    for source, iteration in zip(sources, iterations):
        _fold_checkpoint(Path(output), source, iteration, settings)


def _fold_checkpoint(output, source, iteration, settings):
    rank = dist.get_rank()
    tracker = output / _TRACKER
    mean_root = output / _WSM_RUNNING_MEAN_DIR
    is_wsm = settings["method"] == "wsm"

    previous, state = None, None
    if tracker.is_file():
        value = tracker.read_text().strip()
        if not value.isdigit():
            raise ValueError(f"{output} is not a running merge (its tracker names {value!r})")
        previous = int(value)
        common = dist_checkpointing.load_common_state_dict(str(output / _iter_name(previous)))
        state = common.get("checkpoint_merge_metadata", {})
        if not state.get("update"):
            raise ValueError(f"{output} holds a checkpoint that was not created by --update")
        stored = {key: state.get(key) for key in settings}
        if stored != settings:
            raise ValueError(
                f"Settings differ from the running merge at {output}: stored {stored}, "
                f"requested {settings}"
            )
        if iteration <= previous:
            if source in state["source_checkpoints"]:
                if rank == 0:
                    print(f"> skipping {source}: already folded into {output}")
                return
            raise ValueError(f"{source} is not newer than the running merge (iteration {previous})")
    elif output.exists():
        # Only leftovers of an interrupted first update may be here.
        allowed = {_iter_name(iteration), _WSM_RUNNING_MEAN_DIR, _TRACKER + ".tmp"}
        unexpected = sorted(path.name for path in output.iterdir() if path.name not in allowed)
        if unexpected:
            raise FileExistsError(f"{output} has no running-merge tracker but holds {unexpected}")

    num_merged = 0 if state is None else state["num_checkpoints"]
    total = num_merged + 1
    alpha = settings["ema_alpha"]
    running_mean = [num_merged / total, 1 / total]
    if settings["method"] == "mean":
        pair = running_mean
        effective = mean_coefficients(total)
    elif settings["method"] == "ema":
        if settings["ema_bias_correction"]:
            z_old = 1 - (1 - alpha) ** num_merged
            z_new = 1 - (1 - alpha) ** total
            pair = [(1 - alpha) * z_old / z_new, alpha / z_new]
        else:
            pair = [1 - alpha, alpha]
        effective = ema_coefficients(total, alpha, settings["ema_bias_correction"])
    else:
        keep = (1 - settings["decay_end"]) * num_merged / total
        pair = [keep, 1 - keep]
        effective = wsm_coefficients(total, settings["decay_end"])
    router_pair = _router_coefficients(settings["router_state"], pair, running_mean)

    # The running state is the previous merge, except for wsm, whose state is the running mean.
    if state is None:
        inputs, pair, router_pair, running_mean = [source], [1.0], [1.0], [1.0]
    else:
        state_root = mean_root if is_wsm else output
        inputs = [str(state_root / _iter_name(previous)), source]
    folded = {
        "update": True,
        "save_fp32": True,
        "num_checkpoints": total,
        "source_checkpoints": [*(state or {}).get("source_checkpoints", []), source],
        "iterations": [*(state or {}).get("iterations", []), iteration],
    }
    merged_metadata = {**settings, **folded, "coefficients": effective}
    targets = [(output / _iter_name(iteration), pair, router_pair, merged_metadata)]
    if is_wsm:
        mean_settings = {**settings, "method": "mean", "decay_end": None, "router_state": "mean"}
        mean_metadata = {**mean_settings, **folded, "coefficients": mean_coefficients(total)}
        targets.append(
            (mean_root / _iter_name(iteration), running_mean, running_mean, mean_metadata)
        )

    for target, coefficients, router_coefficients, metadata in targets:
        if rank == 0 and target.exists():
            shutil.rmtree(target)  # left by an interrupted update; the tracker never named it
        dist.barrier()
        _merge(inputs, target, coefficients, router_coefficients, metadata, save_fp32=True)

    if rank == 0:
        temporary = output / (_TRACKER + ".tmp")
        temporary.write_text(str(iteration))
        os.replace(temporary, tracker)
        # Keep only the merge the tracker names, including leftovers of an interrupted cleanup.
        for root in (output, mean_root) if is_wsm else (output,):
            for path in root.glob("iter_*"):
                if path.name != _iter_name(iteration):
                    shutil.rmtree(path)
        print(f"> folded {source} into {output} ({total} checkpoints, coefficients {pair})")
    dist.barrier()


def _initialize_distributed(backend):
    if dist.is_initialized():
        return False
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        if "MASTER_ADDR" not in os.environ:
            node_list = os.environ.get("SLURM_STEP_NODELIST", os.environ["SLURM_JOB_NODELIST"])
            os.environ["MASTER_ADDR"] = subprocess.check_output(
                ["scontrol", "show", "hostnames", node_list], text=True
            ).splitlines()[0]
        if "MASTER_PORT" not in os.environ:
            job_id = int(os.environ["SLURM_JOB_ID"].split("_")[0])
            step_id = int(os.environ.get("SLURM_STEP_ID", "0"))
            os.environ["MASTER_PORT"] = str(20000 + (job_id + step_id) % 40000)
    dist.init_process_group(backend=backend)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help=(
            "Checkpoints in chronological order: iteration/release directories or roots "
            "(resolved through their tracker), or one root together with --iterations. "
            "The last checkpoint supplies the progress metadata."
        ),
    )
    parser.add_argument(
        "--iterations", nargs="+", type=int, help="Iterations to take from the single root."
    )
    parser.add_argument("--output", required=True, help="Output checkpoint root.")
    parser.add_argument(
        "--update",
        action="store_true",
        help=(
            "Fold --checkpoints into the running merge at --output (created on first use) instead "
            "of merging them from scratch. Supports mean, ema and wsm (a window that grows from "
            "the first folded checkpoint); the merge is always saved in fp32."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("mean", "ema", "wsm"),
        default="mean",
        help=(
            "mean: uniform average. ema: exponential moving average. "
            "wsm: weights that emulate a linear LR decay."
        ),
    )
    parser.add_argument(
        "--router-state",
        choices=("mean", "method", "last"),
        default="mean",
        help=(
            "Router balancing state (expert_bias, qb_beta): mean averages it uniformly "
            "(default); method uses the same coefficients as --method; last copies it from the "
            "last checkpoint."
        ),
    )
    parser.add_argument(
        "--save-fp32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save merged floating-point tensors in fp32 (default). --no-save-fp32 keeps the "
            "source dtype, for example bf16 for a final export."
        ),
    )
    parser.add_argument(
        "--sphere-merge",
        action="store_true",
        help=(
            "For MDDecoupling (hypersphere) checkpoints: average each sphere matrix's direction "
            "and its effective gains separately, scale the direction back to its average source "
            "length as MDDecoupling projects it (per row, or per flat piece), then apply the "
            "gains again. The sphere modes come from the checkpoint args; row and flat are "
            "supported. Other tensors are averaged directly. Not supported with --update."
        ),
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        help="ema: weight of each newer checkpoint in the running average, in (0, 1].",
    )
    parser.add_argument(
        "--ema-bias-correction",
        action="store_true",
        help=(
            "ema: start the average at zero and divide by 1 - (1 - alpha)**n, so the oldest "
            "checkpoint is not weighted up."
        ),
    )
    wsm = parser.add_argument_group("wsm")
    wsm.add_argument(
        "--decay-end", type=float, help="Final LR multiplier of the emulated decay (default: 1e-4)."
    )
    wsm.add_argument(
        "--source-decay-steps",
        type=int,
        help="Set if the checkpoints come from inside a linear decay: its total length.",
    )
    wsm.add_argument("--source-decay-end", type=float, help="Final multiplier of that decay.")
    wsm.add_argument(
        "--source-decay-start",
        type=int,
        help="Last step before that decay (default: one interval before the first checkpoint).",
    )
    parser.add_argument(
        "--backend",
        choices=("gloo", "nccl"),
        default="gloo",
        help="Process-group backend for worker coordination.",
    )
    args = parser.parse_args()

    if (args.ema_alpha is None) == (args.method == "ema"):
        parser.error("--ema-alpha is required with, and only valid with, --method ema")
    if args.ema_bias_correction and args.method != "ema":
        parser.error("--ema-bias-correction needs --method ema")
    source_flags = (args.source_decay_steps, args.source_decay_end, args.source_decay_start)
    if args.method != "wsm" and any(flag is not None for flag in (args.decay_end, *source_flags)):
        parser.error("--decay-end and --source-decay-* flags need --method wsm")
    if args.decay_end is None:
        args.decay_end = 1e-4
    if any(flag is not None for flag in source_flags):
        if args.source_decay_steps is None or args.source_decay_end is None:
            parser.error("--source-decay-steps and --source-decay-end must be set together")
        source_decay = LinearDecay(
            args.source_decay_steps, args.source_decay_end, args.source_decay_start
        )
    else:
        source_decay = None
    if args.update and source_decay is not None:
        parser.error("--source-decay-* cannot be used with --update")
    if args.update and not args.save_fp32:
        parser.error("--update always saves fp32; --no-save-fp32 is not supported with it")
    if args.update and args.sphere_merge:
        parser.error("--sphere-merge is not supported with --update")

    initialized_here = _initialize_distributed(args.backend)
    try:
        if args.update:
            update_running_merge(
                args.output,
                resolve_sources(args.checkpoints, args.iterations, min_sources=1),
                args.method,
                ema_alpha=args.ema_alpha,
                ema_bias_correction=args.ema_bias_correction,
                decay_end=args.decay_end,
                router_state=args.router_state,
            )
            return
        sources = resolve_sources(args.checkpoints, args.iterations)
        iterations = source_iterations(sources)
        metadata = {"method": args.method, "iterations": iterations}
        if args.method == "mean":
            coefficients = mean_coefficients(len(sources))
        elif args.method == "ema":
            coefficients = ema_coefficients(len(sources), args.ema_alpha, args.ema_bias_correction)
            metadata["ema_alpha"] = args.ema_alpha
            metadata["ema_bias_correction"] = args.ema_bias_correction
        else:
            coefficients = wsm_coefficients(len(sources), args.decay_end, source_decay, iterations)
            metadata["decay_end"] = args.decay_end
            metadata["source_decay"] = asdict(source_decay) if source_decay else None
        if dist.get_rank() == 0:
            print(f"> checkpoint coefficients: {coefficients}")
        merge_checkpoint_directories(
            sources,
            args.output,
            coefficients,
            metadata,
            router_state=args.router_state,
            save_fp32=args.save_fp32,
            sphere_merge=args.sphere_merge,
        )
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
