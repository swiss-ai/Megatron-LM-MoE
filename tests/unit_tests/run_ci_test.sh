#!/bin/bash
set -euxo pipefail

PYTHON_CMD=(uv run --no-sync python)
if [[ -n ${PYTHON_BIN:-} ]]; then
    unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
    PYTHON_CMD=(uv run --no-project --python "$PYTHON_BIN" python)
fi

# Set worker-local caches before Python imports.
if [[ ${1:-} == --worker ]]; then
    shift
    CACHE_ROOT=$(mktemp -d "/tmp/megatron-test-${SLURM_JOB_ID:-local}-rank-${RANK:?}-XXXXXX")
    trap 'rm -rf -- "$CACHE_ROOT"' EXIT

    export TRITON_HOME=$CACHE_ROOT/triton
    export TRITON_CACHE_DIR=$CACHE_ROOT/triton/cache
    export TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/inductor
    export TORCH_EXTENSIONS_DIR=$CACHE_ROOT/extensions
    export CUDA_CACHE_PATH=$CACHE_ROOT/cuda
    export CUPY_CACHE_DIR=$CACHE_ROOT/cupy
    export PYTHONPYCACHEPREFIX=$CACHE_ROOT/pycache
    export FLASHINFER_WORKSPACE_BASE=$CACHE_ROOT/flashinfer

    PYTEST_BASE="$LOG_DIR/pytest/${TEST_PHASE:?}/rank-$RANK"
    mkdir -p "$(dirname "$PYTEST_BASE")"

    PYTEST_ARGS=(
        --basetemp "$PYTEST_BASE"
        --junitxml "$LOG_DIR/$TEST_PHASE.rank-$RANK.xml"
        -o "cache_dir=$LOG_DIR/pytest-cache/rank-$RANK"
    )
    if [[ ${UNIT_TEST_TRACE:-false} == true ]]; then
        PYTEST_ARGS+=(-o faulthandler_timeout=120)
        export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000}
        export TORCH_NCCL_DUMP_ON_TIMEOUT=1
        export TORCH_NCCL_TRACE_CPP_STACK=1
        export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$LOG_DIR/$TEST_PHASE.rank-$RANK.nccl-trace-"
    fi
    if [[ ${UNIT_TEST_FAIL_FAST:-true} == false ]]; then
        PYTEST_ARGS+=(--maxfail=0 --continue-on-collection-errors)
    fi
    if [[ ${UNIT_TEST_COLLECT_ONLY:-false} == true ]]; then
        PYTEST_ARGS+=(--collect-only --continue-on-collection-errors --maxfail=0)
    fi

    EXIT_CODE=0
    "${PYTHON_CMD[@]}" "$@" "${PYTEST_ARGS[@]}" || EXIT_CODE=$?

    echo "$EXIT_CODE" > "$LOG_DIR/$TEST_PHASE.rank-$RANK.exitcode"

    # Let sibling ranks finish after ordinary failures; the parent reads all
    # recorded exit codes. An empty marker phase is also allowed.
    if [[ "$EXIT_CODE" == 5 || ( "$EXIT_CODE" == 1 && ${UNIT_TEST_FAIL_FAST:-true} == false ) ]]; then
        exit 0
    fi
    exit "$EXIT_CODE"
fi

# Parse command line arguments
usage() {
    echo "Usage: $0 --tag {latest|legacy} --environment {lts|dev} --bucket BUCKET [--platform {h100|gh200|gb200}] [--unit-test-repeat N] [--unit-test-timeout MINUTES] [--phase {all|prod|experimental}] --log-dir LOG_DIR"
    exit 1
}

# Get directory of this script
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_PATH/../.."

# Default values
UNIT_TEST_REPEAT=1
UNIT_TEST_TIMEOUT=10
PHASE=all
PLATFORM=h100
LOG_DIR=$(pwd)/logs

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --help)
        usage
        ;;
    --tag)
        TAG="$2"
        shift 2
        ;;
    --environment)
        ENVIRONMENT="$2"
        shift 2
        ;;
    --bucket)
        BUCKET="$2"
        shift 2
        ;;
    --platform)
        PLATFORM="$2"
        shift 2
        ;;
    --unit-test-repeat)
        UNIT_TEST_REPEAT="$2"
        shift 2
        ;;
    --unit-test-timeout)
        UNIT_TEST_TIMEOUT="$2"
        shift 2
        ;;
    --phase)
        PHASE="$2"
        shift 2
        ;;
    --log-dir)
        LOG_DIR="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        usage
        ;;
    esac
done

# Validate required arguments
if [[ -z "${TAG:-}" || -z "${ENVIRONMENT:-}" || -z "${BUCKET:-}" ]]; then
    echo "Error: Missing required arguments"
    usage
fi

# Validate TAG
if [[ "$TAG" != "latest" && "$TAG" != "legacy" ]]; then
    echo "Error: TAG must be either 'latest' or 'legacy'"
    usage
fi

# Validate ENVIRONMENT
if [[ "$ENVIRONMENT" != "lts" && "$ENVIRONMENT" != "dev" ]]; then
    echo "Error: ENVIRONMENT must be either 'dev' or 'dev'"
    usage
fi

# Validate LOG_DIR
if [[ -z "${LOG_DIR:-}" ]]; then
    echo "Error: LOG_DIR is required"
    usage
else
    mkdir -p "$LOG_DIR"
    LOG_DIR=$(realpath "$LOG_DIR")
    export LOG_DIR
fi

if [[ ! "$UNIT_TEST_TIMEOUT" =~ ^[1-9][0-9]*$ || ! "$UNIT_TEST_REPEAT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Timeout minutes and repeat count must be positive integers" >&2
    exit 1
fi
if [[ "$PHASE" != all && "$PHASE" != prod && "$PHASE" != experimental ]]; then
    usage
fi
if [[ "$TAG" == legacy && "$PHASE" == experimental ]]; then
    echo "Legacy tests have no experimental marker phase" >&2
    exit 1
fi

# Convert ENVIRONMENT to lowercase for internal use
ENVIRONMENT=$(echo "$ENVIRONMENT" | tr '[:upper:]' '[:lower:]')

if [[ "$TAG" == "latest" ]]; then
    TEST_PATH=${TEST_PATH:-$(cd "$SCRIPT_PATH/../.." && pwd)}
else
    TEST_PATH=${TEST_PATH:-/opt/megatron-lm-legacy}
fi

cd "$TEST_PATH"

MARKER=()
if [[ "$TAG" == "legacy" ]]; then
    MARKER+=("not internal")
fi

if [[ "$ENVIRONMENT" == "lts" ]]; then
    MARKER+=("not flaky")
fi

if [[ "$ENVIRONMENT" == "dev" ]]; then
    MARKER+=("not flaky_in_dev")
fi

MARKER_ARG=$(printf "%s" "${MARKER[0]}")
for element in "${MARKER[@]:1}"; do
    MARKER_ARG+=" and $element"
done

export BUCKET
IGNORE_ARGS=()
IGNORE_OUTPUT=$("${PYTHON_CMD[@]}" tests/unit_tests/find_test_cases.py "$BUCKET" "$PLATFORM")
while IFS= read -r line; do
    [[ -n "$line" ]] && IGNORE_ARGS+=("$line")
done <<< "$IGNORE_OUTPUT"

echo "------ARGUMENTS for SLURM ---"
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NUM_NODES=${NUM_NODES:-${SLURM_NNODES:-1}}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODE_RANK=${SLURM_PROCID:-${SLURM_NODEID:-0}}
DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes "$NUM_NODES"
    --master_addr "$MASTER_ADDR"
    --master_port "$MASTER_PORT"
    --node_rank "$NODE_RANK"
    --log-dir "$LOG_DIR"
    --tee "0:3"
    --redirects "3"
)

# Reduce memory usage by NCCL
export NCCL_MAX_NCHANNELS=1
export NCCL_NVLS_ENABLE=0
export ONE_LOGGER_JOB_CATEGORY=test

TORCHRUN=(
    "${PYTHON_CMD[@]}" -m torch.distributed.run
    "${DISTRIBUTED_ARGS[@]}"
)
TEST_RESULT=0
run_phase() {
    local phase_result=0
    timeout --signal=TERM --kill-after=30s "${UNIT_TEST_TIMEOUT}m" \
        "${TORCHRUN[@]}" \
        --no-python bash "$SCRIPT_PATH/run_ci_test.sh" --worker "$@" || phase_result=$?
    if (( phase_result != 0 )); then
        # Let the parent end the Slurm phase if a launcher fails.
        echo "$phase_result" > "$LOG_DIR/$TEST_PHASE.node-$NODE_RANK.launcher-exitcode"
        return "$phase_result"
    fi
}

TEST_TARGET=${BUCKET%/\*\*/\*.py}
for i in $(seq "$UNIT_TEST_REPEAT"); do
    if [[ "$PHASE" == all || "$PHASE" == prod ]]; then
        echo "Running prod test suite."
        export TEST_PHASE="repeat-$i-prod"
        run_phase -m coverage run \
            --parallel-mode \
            --data-file="$LOG_DIR/.coverage.unit_tests" \
            --source=megatron/core \
            -m pytest -xvs \
            "${IGNORE_ARGS[@]}" \
            -m "not experimental and ${MARKER_ARG}" \
            "$TEST_TARGET"
    fi

    if [[ "$TAG" == "latest" && "$PHASE" != prod ]]; then
        export TEST_PHASE="repeat-$i-experimental"
        run_phase -m pytest -xvs \
            --experimental \
            "${IGNORE_ARGS[@]}" \
            -m "experimental and ${MARKER_ARG}" \
            "$TEST_TARGET"
    fi

    # The Alps parent checks all ranks after its entire Slurm phase step exits.
    if [[ "$NODE_RANK" == 0 && "$PHASE" == all ]]; then
        PHASES=(prod)
        if [[ "$TAG" == latest ]]; then
            PHASES+=(experimental)
        fi
        for phase in "${PHASES[@]}"; do
            for ((rank = 0; rank < NUM_NODES * GPUS_PER_NODE; rank++)); do
                exit_file="$LOG_DIR/repeat-$i-$phase.rank-$rank.exitcode"
                if [[ ! -f "$exit_file" ]]; then
                    TEST_RESULT=1
                else
                    rank_result=$(< "$exit_file")
                    if [[ "$rank_result" != 0 && "$rank_result" != 5 ]]; then
                        TEST_RESULT=$rank_result
                    fi
                fi
            done
        done
    fi

    if [[ "$NODE_RANK" == 0 && "$PHASE" == all && -f "$LOG_DIR/repeat-$i-prod.rank-0.exitcode" ]]; then
        PROD_RESULT=$(< "$LOG_DIR/repeat-$i-prod.rank-0.exitcode")
        EXPERIMENTAL_RESULT=5
        if [[ "$TAG" == latest ]]; then
            EXPERIMENTAL_RESULT=1
            if [[ -f "$LOG_DIR/repeat-$i-experimental.rank-0.exitcode" ]]; then
                EXPERIMENTAL_RESULT=$(< "$LOG_DIR/repeat-$i-experimental.rank-0.exitcode")
            fi
        fi
        if [[ "$PROD_RESULT" == 5 && "$EXPERIMENTAL_RESULT" == 5 ]]; then
            echo "No tests selected in either marker phase: $BUCKET" >&2
            TEST_RESULT=5
            if [[ ${UNIT_TEST_FAIL_FAST:-true} != false ]]; then
                exit 5
            fi
        fi
    fi
done

# Each node has the same shared log directory; only the first combines coverage.
if [[ "$NODE_RANK" == 0 && "$PHASE" != prod ]] &&
    compgen -G "$LOG_DIR/.coverage.unit_tests.*" > /dev/null; then
    "${PYTHON_CMD[@]}" -m coverage combine -q --data-file="$LOG_DIR/.coverage.unit_tests" "$LOG_DIR"
fi
exit "$TEST_RESULT"
