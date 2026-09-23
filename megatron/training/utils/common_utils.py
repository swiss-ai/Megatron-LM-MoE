# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import os
import warnings
from megatron.core._slurm_utils import resolve_slurm_local_rank

def get_local_rank_preinit() -> int:
    """Get the local rank from the environment variable, intended for use before full init.

    Fallback order:
    1. LOCAL_RANK environment variable (torchrun/torchelastic)
    2. SLURM_LOCALID environment variable (SLURM)
    3. Default: 0 (with warning)

    Returns:
        The local rank of the current process.
    """
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])

    slurm_local_rank = resolve_slurm_local_rank()
    if slurm_local_rank is not None:
        return slurm_local_rank

    warnings.warn("Could not determine local rank from LOCAL_RANK or SLURM_LOCALID. Defaulting to local rank 0.")
    return 0
