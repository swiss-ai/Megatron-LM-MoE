#!/usr/bin/env bash
# Source inside the test image; dependencies are image-owned.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -f "$REPO_ROOT/.env" ]]; then
    set +x
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

: "${SCRATCH:?Set SCRATCH to an Iopsstor scratch directory}"
export PYTHON_BIN=${IMAGE_PYTHON:-/usr/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing image Python: $PYTHON_BIN" >&2
    return 1
fi
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT PYTHON_EXEC PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$REPO_ROOT
export UV_NO_PROJECT=1 UV_PYTHON=$PYTHON_BIN UV_PYTHON_DOWNLOADS=never

for tool in uv yq envsubst; do
    if ! command -v "$tool" >/dev/null; then
        echo "Missing image tool: $tool; rebuild the image." >&2
        return 1
    fi
done
if ! yq --version | grep -q 'version v4\.44\.1$'; then
    echo 'The test image requires Go yq 4.44.1.' >&2
    return 1
fi

SHARED_CACHE_DIR=/ritom/scratch/cscs/$USER/cache
if [[ ! -d /ritom/scratch/cscs/$USER ]]; then
    SHARED_CACHE_DIR="${LOG_DIR:-${WORK_DIR:-$SCRATCH/tmp/megatron-tests-${SLURM_JOB_ID:-local}}}/cache"
fi
export XDG_CACHE_HOME=$SHARED_CACHE_DIR
export UV_CACHE_DIR=$SHARED_CACHE_DIR/uv
export HF_HOME=$SHARED_CACHE_DIR/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$SHARED_CACHE_DIR/torch
export NLTK_DATA=$SHARED_CACHE_DIR/nltk
