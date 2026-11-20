# Copyright (c) 2026, EPFL / Swiss AI Initiative.

"""Merge Megatron ``torch_dist`` checkpoints with generic checkpoint workers."""

import argparse
import copy
import gc
import math
import os
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.serialization import load_sharded_metadata


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


def resolve_sources(checkpoints, iterations=None):
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
    if len(sources) < 2:
        raise ValueError("Checkpoint merging requires at least two checkpoints")
    return sources


def source_iterations(sources):
    """Return the iteration of each ``iter_*`` source, or None if any source is not one."""
    names = [Path(source).name for source in sources]
    if not all(name.startswith("iter_") for name in names):
        return None
    return [int(name.removeprefix("iter_")) for name in names]


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


def mean_coefficients(num_checkpoints):
    """Uniform weights."""
    if num_checkpoints < 2:
        raise ValueError("Checkpoint merging requires at least two checkpoints")
    return [1.0 / num_checkpoints] * num_checkpoints


def ema_coefficients(num_checkpoints, alpha):
    """Weights of an exponential moving average over the checkpoints, oldest first.

    The average starts at the first checkpoint and then takes ``avg = alpha * new +
    (1 - alpha) * avg`` for each later one, so the first checkpoint keeps ``(1 - alpha)**(n - 1)``.
    The decay is per checkpoint, not per training step. This is Eq. (4) of Li et al. 2025,
    "Model Merging in Pre-training of Large Language Models" (arXiv:2505.12082).
    """
    if num_checkpoints < 2:
        raise ValueError("Checkpoint merging requires at least two checkpoints")
    if not 0 < alpha <= 1:
        raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")
    return [(1 - alpha) ** (num_checkpoints - 1)] + [
        alpha * (1 - alpha) ** (num_checkpoints - 1 - index) for index in range(1, num_checkpoints)
    ]


def wsm_coefficients(num_checkpoints, decay_end=1e-4, source_decay=None, iterations=None):
    """WSM weights that emulate a linear decay to ``decay_end`` over the merged checkpoints.

    Each checkpoint is the endpoint of one coarse update. A synthetic checkpoint one interval
    before the first gets weight zero, so it is never loaded. If the checkpoints were already
    produced under ``source_decay``, that decay is divided out first; this needs the
    checkpoint ``iterations``.
    """
    if num_checkpoints < 2:
        raise ValueError("Checkpoint merging requires at least two checkpoints")
    desired = _linear_multipliers(
        [interval / num_checkpoints for interval in range(num_checkpoints)], decay_end
    )

    if source_decay is None:
        original = [1.0] * num_checkpoints
    else:
        if iterations is None:
            raise ValueError("A source decay needs the checkpoint iterations")
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
    sources, output, coefficients=None, merge_metadata=None, router_state="mean", save_fp32=True
):
    """Combine model tensors using every distributed rank as an I/O worker.

    ``router_state`` selects how the router balancing state (``expert_bias``, ``qb_beta``) is
    merged: "mean" averages it uniformly, "method" uses ``coefficients`` like every other
    tensor, and "last" copies it from the last checkpoint.

    Merged floating-point tensors are saved in fp32 (``save_fp32``, the default) or in their
    source dtype. Keep fp32 when the output is merged again, for example as a running average,
    because rounding it to bf16 at every update discards changes smaller than about 0.4% of a
    weight. A bf16 model still loads an fp32 checkpoint; the loader casts to the model's dtype.
    """
    sources = list(sources)
    if len(sources) < 2:
        raise ValueError("Checkpoint merging requires at least two checkpoints")
    if coefficients is None:
        coefficients = mean_coefficients(len(sources))
    router_coefficients = {
        "mean": mean_coefficients(len(sources)),
        "method": coefficients,
        "last": [0.0] * (len(sources) - 1) + [1.0],
    }.get(router_state)
    if router_coefficients is None:
        raise ValueError(f"router_state must be 'mean', 'method' or 'last', got {router_state!r}")
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
                is_router_state = key.rsplit(".", 1)[-1] in _ROUTER_STATE_TENSOR_NAMES
                weights = router_coefficients if is_router_state else coefficients
                coefficient = weights[source_index]
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
            data = accumulators[key].to(torch.float32 if save_fp32 else metadata.dtype)
        else:
            data = static_tensors[key].to(metadata.dtype)
        output_tensors[key] = replace(metadata, data=data, dtype=data.dtype, replica_id=0)

    loaded_objects = _load_assigned(first_source, assigned_objects)
    output_objects = {
        key: replace(metadata, data=loaded_objects[key], replica_id=0)
        for key, metadata in assigned_objects.items()
    }

    state_dict = _filtered_common_state(sources[-1])
    state_dict["checkpoint_merge_metadata"] = {
        **copy.deepcopy(merge_metadata or {}),
        "source_checkpoints": sources,
        "coefficients": list(coefficients),
        "router_state": router_state,
        "save_fp32": save_fp32,
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
        "--ema-alpha",
        type=float,
        help="ema: weight of each newer checkpoint in the running average, in (0, 1].",
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

    initialized_here = _initialize_distributed(args.backend)
    try:
        sources = resolve_sources(args.checkpoints, args.iterations)
        iterations = source_iterations(sources)
        metadata = {"method": args.method, "iterations": iterations}
        if args.method == "mean":
            coefficients = mean_coefficients(len(sources))
        elif args.method == "ema":
            coefficients = ema_coefficients(len(sources), args.ema_alpha)
            metadata["ema_alpha"] = args.ema_alpha
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
        )
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
