#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
RECIPE=${FUNCTIONAL_RECIPE:-$REPO_ROOT/tests/test_utils/recipes/gh200/gpt.yaml}
TEST_CASE=$1

: "${LOG_DIR:?Set the case output directory}"
: "${DATA_PATH:?Set the shared asset root}"

MODEL=$(yq '.spec.model' "$RECIPE")
TRAINING_SCRIPT=$(yq '.spec.training_script' "$RECIPE")
N_REPEAT=${N_REPEAT:-$(yq '.spec.n_repeat' "$RECIPE")}
CONFIG_SOURCE="$REPO_ROOT/tests/functional_tests/test_cases/$MODEL/$TEST_CASE/model_config.yaml"
GOLDEN_VALUES_PATH="${CONFIG_SOURCE%/*}/golden_values_dev_gh200.json"
CONFIG_COPY="$LOG_DIR/model-config-node-${SLURM_PROCID:-0}.yaml"

mkdir -p "$LOG_DIR"
cp "$CONFIG_SOURCE" "$CONFIG_COPY"
if [[ ${SLURM_PROCID:-0} == 0 ]]; then
    git -C "$REPO_ROOT" rev-parse HEAD > "$LOG_DIR/source-commit.txt"
    git -C "$REPO_ROOT" diff HEAD -- megatron pretrain_gpt.py > "$LOG_DIR/production-changes.patch"
    sha256sum "$CONFIG_SOURCE" > "$LOG_DIR/source-config.sha256"
    if [[ -n ${ENVIRONMENT:-} ]]; then
        cp "$ENVIRONMENT" "$LOG_DIR/environment.toml"
    fi
    DATASET_DIR="$DATA_PATH/text/common_pile/v01_filtered_data"
    if [[ -f "$DATASET_DIR/SHA256SUMS" ]]; then
        cp "$DATASET_DIR/SHA256SUMS" "$LOG_DIR/dataset.sha256"
    fi
    if [[ -f "$DATASET_DIR/source.json" ]]; then
        cp "$DATASET_DIR/source.json" "$LOG_DIR/dataset-source.json"
    fi
fi

TEST_TYPE=$(yq '.TEST_TYPE' "$CONFIG_COPY")
case "$TEST_TYPE" in
    regular|ckpt-resume) ;;
    *)
        echo "Unsupported test type for this asset setup: $TEST_TYPE" >&2
        exit 1
        ;;
esac

NODE_CACHE=$(mktemp -d "/tmp/megatron-functional-${SLURM_JOB_ID:-local}-node-${SLURM_PROCID:-0}-XXXXXX")
trap 'rm -rf -- "$NODE_CACHE"' EXIT
export PYTHONPYCACHEPREFIX=$NODE_CACHE/pycache

cd "$REPO_ROOT"
bash tests/functional_tests/shell_test_utils/run_ci_test.sh \
    "TRAINING_SCRIPT_PATH=$TRAINING_SCRIPT" \
    "TRAINING_PARAMS_PATH=$CONFIG_COPY" \
    "GOLDEN_VALUES_PATH=$GOLDEN_VALUES_PATH" \
    "OUTPUT_PATH=$LOG_DIR" \
    "TENSORBOARD_PATH=$LOG_DIR/tensorboard" \
    "CHECKPOINT_SAVE_PATH=$LOG_DIR/checkpoints" \
    "CHECKPOINT_LOAD_PATH=$LOG_DIR/checkpoint-input" \
    "EMPTY_CHECKPOINT_PATH=$LOG_DIR/empty-checkpoints" \
    "DATA_PATH=$DATA_PATH" \
    "DATA_CACHE_PATH=$LOG_DIR/data-cache" \
    "N_REPEAT=$N_REPEAT" \
    "ENABLE_LIGHTWEIGHT_MODE=false" \
    "RECORD_CHECKPOINTS=false" \
    "RECORD_BASELINE=${RECORD_BASELINE:-false}"
