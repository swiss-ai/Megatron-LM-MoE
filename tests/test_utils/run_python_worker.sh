#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=$(mktemp -d "/tmp/megatron-training-${SLURM_JOB_ID:-local}-rank-${RANK:?}-XXXXXX")
trap 'rm -rf -- "$CACHE_ROOT"' EXIT

export TRITON_HOME=$CACHE_ROOT/triton
export TRITON_CACHE_DIR=$CACHE_ROOT/triton/cache
export TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/inductor
export TORCH_EXTENSIONS_DIR=$CACHE_ROOT/extensions
export CUDA_CACHE_PATH=$CACHE_ROOT/cuda
export CUPY_CACHE_DIR=$CACHE_ROOT/cupy
export PYTHONPYCACHEPREFIX=$CACHE_ROOT/pycache
export FLASHINFER_WORKSPACE_BASE=$CACHE_ROOT/flashinfer

if [[ -n ${PYTHON_BIN:-} ]]; then
    uv run --no-project --python "$PYTHON_BIN" python "$@"
else
    uv run --no-sync python "$@"
fi
