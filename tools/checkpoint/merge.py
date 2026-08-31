# Copyright (c) 2026, EPFL / Swiss AI Initiative.

"""Merge Megatron ``torch_dist`` checkpoints with generic checkpoint workers."""

import argparse
import copy
import gc
import math
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.dist_checkpointing.serialization import load_sharded_metadata


_ALWAYS_MEAN_TENSOR_NAMES = {"expert_bias", "qb_beta"}
_TRAINING_PREFIXES = (
    "optimizer",
    "opt_param_scheduler",
    "lr_scheduler",
    "rng_state",
    "rerun_state_machine",
    "dataloader",
)


def resolve_sources(checkpoint_roots, checkpoint_steps=None):
    """Resolve roots and optional iterations to concrete checkpoint directories."""
    checkpoint_roots = list(checkpoint_roots)
    checkpoint_steps = list(checkpoint_steps or [])
    if checkpoint_steps:
        if len(checkpoint_roots) == 1:
            checkpoint_roots *= len(checkpoint_steps)
        elif len(checkpoint_roots) != len(checkpoint_steps):
            raise ValueError("Provide one checkpoint root or one root per checkpoint step")
    else:
        checkpoint_steps = [None] * len(checkpoint_roots)

    sources = [
        _resolve_checkpoint_dir(root, step)
        for root, step in zip(checkpoint_roots, checkpoint_steps)
    ]
    if len(sources) < 2:
        raise ValueError("Checkpoint merging requires at least two checkpoints")
    return sources


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
    return str(root / f"iter_{step:07d}")


def _is_training_key(key):
    key = str(key)
    return key.startswith(_TRAINING_PREFIXES)


def _tensor_nbytes(metadata):
    return math.prod(metadata.global_shape) * metadata.dtype.itemsize


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
        position_sum = (
            (linear_first + linear_last) * linear_count / 2 - decay_start * linear_count
        )
        total += linear_count - (1.0 - end_multiplier) * position_sum / decay_steps

    post_decay_count = max(right - max(left, decay_start + decay_steps + 1), 0)
    total += post_decay_count * end_multiplier
    return total / count


def merge_coefficients(
    num_checkpoints,
    method="mean",
    original_schedule="stable",
    original_end_multiplier=None,
    target_end_multiplier=1e-4,
    checkpoint_steps=None,
    original_decay_steps=None,
    original_decay_start_step=None,
):
    """Derive checkpoint coefficients, including WSM decay correction."""
    if num_checkpoints < 2:
        raise ValueError("Checkpoint merging requires at least two checkpoints")
    if method == "mean":
        return [1.0 / num_checkpoints] * num_checkpoints
    if method != "linear-decay":
        raise ValueError(f"Unknown merge method: {method}")

    steps_were_provided = checkpoint_steps is not None
    if checkpoint_steps is None:
        checkpoint_steps = list(range(num_checkpoints))
    else:
        checkpoint_steps = list(checkpoint_steps)
    if len(checkpoint_steps) != num_checkpoints:
        raise ValueError("Provide exactly one step per checkpoint")
    if any(left >= right for left, right in zip(checkpoint_steps, checkpoint_steps[1:])):
        raise ValueError("Checkpoint steps must be strictly increasing")

    num_intervals = num_checkpoints
    desired = _linear_multipliers(
        [interval / num_intervals for interval in range(num_intervals)],
        target_end_multiplier,
    )
    # Treat each selected checkpoint as the endpoint of a coarse update. The preceding
    # synthetic checkpoint has zero final coefficient and does not need to be loaded.
    synthetic_step = checkpoint_steps[0] - (checkpoint_steps[1] - checkpoint_steps[0])
    interval_starts = [synthetic_step, *checkpoint_steps[:-1]]
    if original_schedule == "stable":
        original = [1.0] * num_intervals
    elif original_schedule == "linear-decay":
        if not steps_were_provided:
            raise ValueError("Original linear decay requires checkpoint iteration steps")
        if original_end_multiplier is None or original_decay_steps is None:
            raise ValueError(
                "Original linear decay requires --original-end-multiplier and "
                "--original-decay-steps"
            )
        if original_decay_steps <= 0:
            raise ValueError("Original decay steps must be positive")
        decay_start = (
            synthetic_step if original_decay_start_step is None else original_decay_start_step
        )
        original = [
            _linear_interval_mean(
                left,
                right,
                decay_start,
                original_decay_steps,
                original_end_multiplier,
            )
            for left, right in zip(interval_starts, checkpoint_steps)
        ]
        if original[0] == 0:
            raise ValueError("Cannot rebase an original decay whose multiplier reaches zero")
        original = [multiplier / original[0] for multiplier in original]
        if any(multiplier == 0 for multiplier in original):
            raise ValueError("Cannot cancel an original decay whose multiplier reaches zero")
    else:
        raise ValueError(f"Unknown original schedule: {original_schedule}")

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
        for name, value in (
            ("no_save_optim", True),
            ("no_save_rng", True),
            ("finetune", True),
        ):
            setattr(checkpoint_args, name, value)
    return common


def merge_checkpoint_directories(sources, output, coefficients=None, merge_metadata=None):
    """Combine model tensors using every distributed rank as an I/O worker."""
    sources = list(sources)
    if len(sources) < 2:
        raise ValueError("Checkpoint merging requires at least two checkpoints")
    inferred_method = "mean" if coefficients is None else "custom"
    if coefficients is None:
        coefficients = merge_coefficients(len(sources))
    if len(coefficients) != len(sources):
        raise ValueError("Provide exactly one coefficient per checkpoint")
    coefficient_sum = sum(coefficients)
    if abs(coefficient_sum - 1.0) > 1e-9:
        raise ValueError(f"Checkpoint coefficients must sum to 1, got {coefficient_sum}")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    first_source = sources[0]

    tensor_metadata = {
        key: value
        for key, value in dist_checkpointing.load_tensors_metadata(first_source).items()
        if not _is_training_key(key)
    }
    if not tensor_metadata:
        raise ValueError(f"No model tensors found in {first_source}")

    sharded_metadata = load_sharded_metadata(first_source)
    object_metadata = {
        key: value
        for key, value in sharded_metadata.items()
        if isinstance(value, ShardedObject) and not _is_training_key(value.key)
    }

    tensor_owners = _owner_map(list(tensor_metadata.items()), world_size, _tensor_nbytes)
    object_owners = _owner_map(list(object_metadata.items()), world_size, lambda _: 1)
    assigned_tensors = {
        key: value for key, value in tensor_metadata.items() if tensor_owners[key] == rank
    }
    assigned_objects = {
        key: value for key, value in object_metadata.items() if object_owners[key] == rank
    }

    accumulators = {}
    static_tensors = {}
    for source_index, source in enumerate(sources):
        _validate_source_metadata(source, tensor_metadata)
        loaded = _load_assigned(source, assigned_tensors)
        for key, tensor in loaded.items():
            if tensor.is_floating_point():
                coefficient = (
                    1.0 / len(sources)
                    if key.rsplit(".", 1)[-1] in _ALWAYS_MEAN_TENSOR_NAMES
                    else coefficients[source_index]
                )
                if source_index == 0:
                    accumulators[key] = tensor.float().mul_(coefficient)
                else:
                    accumulators[key].add_(tensor, alpha=coefficient)
            elif source_index == 0:
                static_tensors[key] = tensor.cpu().clone()
            elif not torch.equal(static_tensors[key], tensor.cpu()):
                raise ValueError(f"Non-floating model tensor {key!r} differs in {source}")
        del loaded
        gc.collect()
        if rank == 0:
            print(f"> accumulated checkpoint {source_index + 1}/{len(sources)}: {source}")

    output_tensors = {}
    for key, metadata in assigned_tensors.items():
        if key in accumulators:
            data = accumulators[key].to(metadata.dtype)
        else:
            data = static_tensors[key].to(metadata.dtype)
        output_tensors[key] = replace(metadata, data=data, replica_id=0)

    loaded_objects = _load_assigned(first_source, assigned_objects)
    output_objects = {
        key: replace(metadata, data=loaded_objects[key], replica_id=0)
        for key, metadata in assigned_objects.items()
    }

    state_dict = _filtered_common_state(sources[-1])
    state_dict["checkpoint_merge_metadata"] = {
        "method": inferred_method,
        "source_checkpoints": sources,
        "coefficients": list(coefficients),
        **(copy.deepcopy(merge_metadata) if merge_metadata is not None else {}),
    }
    for key, value in {**output_tensors, **output_objects}.items():
        if key in state_dict:
            raise KeyError(f"Sharded model key collides with common state: {key!r}")
        state_dict[key] = value

    output = Path(output)
    checkpoint_dir = output / "release"
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
    if rank == 0:
        (output / "latest_checkpointed_iteration.txt").write_text("release")
        print(f"> saved merged release checkpoint to {output}")
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
            os.environ["MASTER_ADDR"] = (
                subprocess.check_output(["scontrol", "show", "hostnames", node_list], text=True)
                .splitlines()[0]
            )
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
            "Checkpoint roots or direct iteration/release directories in chronological order; "
            "the final checkpoint supplies progress metadata."
        ),
    )
    parser.add_argument(
        "--checkpoint-steps",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Strictly increasing iterations to merge. One root applies to every listed iteration."
        ),
    )
    parser.add_argument("--output", required=True, help="Output checkpoint root.")
    parser.add_argument(
        "--merge-method",
        choices=("mean", "linear-decay"),
        default="mean",
        help="Checkpoint combination method. linear-decay uses the WSM formula.",
    )
    parser.add_argument(
        "--original-schedule",
        choices=("stable", "linear-decay"),
        default="stable",
        help="LR schedule used while producing the selected checkpoints.",
    )
    parser.add_argument(
        "--original-end-multiplier",
        type=float,
        default=None,
        help="Final LR multiplier of an original linear decay.",
    )
    parser.add_argument(
        "--original-decay-steps",
        type=int,
        default=None,
        help="Total number of steps in the original linear decay.",
    )
    parser.add_argument(
        "--original-decay-start-step",
        type=int,
        default=None,
        help="Original decay start; defaults to the synthetic preceding checkpoint step.",
    )
    parser.add_argument(
        "--target-end-multiplier",
        type=float,
        default=1e-4,
        help="Desired final LR multiplier for linear-decay merging (default: 1e-4).",
    )
    parser.add_argument(
        "--backend",
        choices=("gloo", "nccl"),
        default="gloo",
        help="Process-group backend for worker coordination.",
    )
    args = parser.parse_args()

    initialized_here = _initialize_distributed(args.backend)
    try:
        sources = resolve_sources(args.checkpoints, args.checkpoint_steps)
        names = [Path(source).name for source in sources]
        checkpoint_steps = (
            [int(name.removeprefix("iter_")) for name in names]
            if all(name.startswith("iter_") for name in names)
            else None
        )
        coefficients = merge_coefficients(
            num_checkpoints=len(sources),
            method=args.merge_method,
            original_schedule=args.original_schedule,
            original_end_multiplier=args.original_end_multiplier,
            target_end_multiplier=args.target_end_multiplier,
            checkpoint_steps=checkpoint_steps,
            original_decay_steps=args.original_decay_steps,
            original_decay_start_step=args.original_decay_start_step,
        )
        if dist.get_rank() == 0:
            print(f"> checkpoint coefficients: {coefficients}")
        merge_checkpoint_directories(
            sources,
            args.output,
            coefficients,
            merge_metadata={
                "method": args.merge_method,
                "checkpoint_steps": checkpoint_steps,
                "original_schedule": args.original_schedule,
                "original_end_multiplier": args.original_end_multiplier,
                "original_decay_steps": args.original_decay_steps,
                "original_decay_start_step": args.original_decay_start_step,
                "target_end_multiplier": args.target_end_multiplier,
            },
        )
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
