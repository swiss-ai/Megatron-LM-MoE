#!/usr/bin/env python3
"""Print only KDA A_log model tensors from torch_dist checkpoints, on CPU.

Accept checkpoint roots (iter_* children) or individual iteration directories.
Each output line is JSON. Full tensor values are included unless --summary-only
is selected. Does not instantiate Megatron or read model/optimizer tensors other
than A_log. Run with python, not torchrun.
"""

import argparse
import json
from pathlib import Path
import re
import sys

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def checkpoint_dirs(paths: list[Path]) -> list[Path]:
    found = set()
    for path in paths:
        path = path.expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")
        if (path / '.metadata').is_file():
            found.add(path)
            continue
        children = [p for p in path.glob('iter_*') if p.is_dir()]
        if not children:
            raise ValueError(f"No torch_dist checkpoint or iter_* directories in {path}")
        for child in children:
            if (child / '.metadata').is_file():
                found.add(child.resolve())
            else:
                print(f"SKIP: missing .metadata (possibly incomplete): {child}", file=sys.stderr)
    return sorted(found, key=lambda p: (str(p.parent), int(p.name[5:]) if re.fullmatch(r'iter_\d+', p.name) else -1))


def is_model_alog(key: str) -> bool:
    parts = re.split(r'[./]', key)
    return 'A_log' in parts and not any(
        part.lower() in {'optimizer', 'exp_avg', 'exp_avg_sq', 'fp32_param', 'fp32_from_fp16_params'}
        for part in parts
    )


def read_alog(path: Path) -> dict[str, torch.Tensor]:
    reader = dcp.FileSystemReader(path)
    metadata = reader.read_metadata()
    tensors = {
        key: torch.empty(tuple(value.size), dtype=value.properties.dtype, device='cpu')
        for key, value in metadata.state_dict_metadata.items()
        if isinstance(value, TensorStorageMetadata) and is_model_alog(key)
    }
    if not tensors:
        raise ValueError(f"No model A_log tensor keys in {path}")
    # Keys are already flattened names from the DCP metadata. Reassemble global
    # tensors from all TP/PP shards without starting a distributed process group.
    dcp.load(tensors, storage_reader=reader, planner=dcp.DefaultLoadPlanner(flatten_state_dict=False))
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--summary-only', action='store_true', help='Omit full values; retain shape, dtype, min/max/mean and nonfinite counts.')
    args = parser.parse_args()
    if torch.distributed.is_initialized():
        parser.error('Run with plain python in one process, not torchrun.')
    try:
        paths = checkpoint_dirs(args.paths)
        if not paths:
            raise ValueError('No completed torch_dist checkpoints found')
    except ValueError as exc:
        parser.error(str(exc))
    failed = False
    for path in paths:
        try:
            print(f"Reading A_log only: {path}", file=sys.stderr, flush=True)
            tensors = read_alog(path)
            for key, tensor in sorted(tensors.items()):
                values = tensor.double()
                finite = values[torch.isfinite(values)]
                record = dict(checkpoint=str(path), key=key, shape=list(tensor.shape), dtype=str(tensor.dtype), numel=tensor.numel(),
                              nan_count=int(values.isnan().sum()), inf_count=int(values.isinf().sum()),
                              min=finite.min().item() if finite.numel() else None,
                              max=finite.max().item() if finite.numel() else None,
                              mean=finite.mean().item() if finite.numel() else None)
                if not args.summary_only:
                    # Standard JSON: preserve nonfinite locations as null, with
                    # separate NaN/Inf counts above. Never truncate tensor values.
                    record['values'] = torch.where(torch.isfinite(values), values, float('nan')).tolist()
                    def clean(value):
                        if isinstance(value, list):
                            return [clean(x) for x in value]
                        return value if value == value else None
                    record['values'] = clean(record['values'])
                print(json.dumps(record, allow_nan=False), flush=True)
        except Exception as exc:
            print(f"ERROR: {path}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            failed = True
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
