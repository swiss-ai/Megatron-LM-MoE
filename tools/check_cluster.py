"""Check a training container and NCCL under srun or torchrun before training.

This checks NCCL collectives, not DeepEP/UCCL dispatch or model memory capacity.
"""

import argparse
from datetime import timedelta
import importlib
import os
import socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--require-module", action="append", default=[])
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    for torch_key, slurm_key in (
        ("RANK", "SLURM_PROCID"),
        ("WORLD_SIZE", "SLURM_NTASKS"),
        ("LOCAL_RANK", "SLURM_LOCALID"),
    ):
        if torch_key not in os.environ and slurm_key in os.environ:
            os.environ[torch_key] = os.environ[slurm_key]
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in training container")
    # HANGANG exposes all four GPUs to each task. Use the same local rank
    # convention as Megatron; also accept one GPU bound per task.
    device = local_rank if torch.cuda.device_count() > 1 else 0
    torch.cuda.set_device(device)
    for module in ["transformer_engine.pytorch", *args.require_module]:
        importlib.import_module(module)
    print(
        f"rank={rank}/{world} host={socket.gethostname()} local_rank={local_rank} "
        f"gpu={torch.cuda.get_device_name(device)} torch={torch.__version__} "
        f"cuda={torch.version.cuda} FI_PROVIDER={os.getenv('FI_PROVIDER')} "
        f"NCCL_NET={os.getenv('NCCL_NET')}",
        flush=True,
    )
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout_seconds))
    try:
        value = torch.tensor([rank + 1], device="cuda", dtype=torch.int64)
        dist.all_reduce(value)
        expected_sum = world * (world + 1) // 2
        if value.item() != expected_sum:
            raise RuntimeError(f"all_reduce got {value.item()}, expected {expected_sum}")
        # Each destination receives one element from every source rank.
        send = torch.full((world,), rank, device="cuda", dtype=torch.int64)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send)
        expected = torch.arange(world, device="cuda", dtype=torch.int64)
        if not torch.equal(recv, expected):
            raise RuntimeError(f"all_to_all_single failed on rank {rank}: {recv}")
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print("PASS: required imports, NCCL all_reduce and all_to_all_single", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
