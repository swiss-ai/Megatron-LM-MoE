#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RECIPE_DIR="$REPO_ROOT/tests/test_utils/recipes/gh200"

usage() {
    echo "Usage: $0 unit [all|GROUP|--list]"
    echo "       $0 functional [all|CASE|--list]"
    echo
    echo "Settings: NODES=2 PARTITION=preemptable ACCOUNT=csstaff TIME=05:00:00"
    echo "          ENVIRONMENT=<Alps EDF> LOG_DIR=<scratch output directory>"
    echo "          DATA_PATH=<asset root> FUNCTIONAL_RECIPE=<family recipe YAML>"
    echo "          RECORD_BASELINE=true records candidates in LOG_DIR"
    echo "          UNIT_TEST_COLLECT_ONLY=true checks unit collection without running tests"
    echo "          UNIT_TEST_FAIL_FAST=false continues tests and marker phases after failures"
    echo "          UNIT_TEST_TIMEOUT=30 bounds each unit marker phase in minutes"
    echo "          PUBLISH_RESULTS=true dispatches a GitHub report after either suite"
    echo "Dependencies and tools are used directly from the test image."
    echo "NODES is the allocation size. Each node has four GH200 GPUs."
}

run_on_node() {
    export GPUS_PER_NODE=$1
    export LOG_DIR=$2
    local test_suite=$3
    shift 3

    cd "$REPO_ROOT"

    MASTER_ADDR=$(scontrol show hostnames "$SLURM_STEP_NODELIST" | head -n 1)
    export MASTER_ADDR
    export MASTER_PORT=${MASTER_PORT:-29500}
    export NUM_NODES=$SLURM_STEP_NUM_NODES

    source "$REPO_ROOT/tests/test_utils/setup_environment.sh"
    unset RANK LOCAL_RANK WORLD_SIZE

    case "$test_suite" in
        unit)
            if [[ ${SLURM_PROCID:-0} == 0 && -n ${SLURM_STEP_ID:-} ]]; then
                printf '%s\n' "$SLURM_STEP_ID" > "$LOG_DIR/slurm-${2:-all}.step-id"
            fi
            exec bash tests/unit_tests/run_ci_test.sh \
                --tag latest \
                --environment dev \
                --platform gh200 \
                --log-dir "$LOG_DIR" \
                --unit-test-timeout "${UNIT_TEST_TIMEOUT:-30}" \
                --phase "${2:-all}" \
                --bucket "$1"
            ;;
        functional)
            exec bash tests/functional_tests/shell_test_utils/run_alps_test.sh "$1"
            ;;
    esac
}

run_step() {
    local nodes=$1
    local workers_per_node=$2
    local output_dir=$3
    shift 3

    local mounts="$SCRATCH:$SCRATCH,$HOME:$HOME,/capstor:/capstor,/iopsstor:/iopsstor,/ritom:/ritom"
    if [[ "$1" == unit ]]; then
        mounts+=",$UNIT_TEST_DATA_DIR:/opt/data"
    fi

    mkdir -p "$output_dir"

    local phase=${3:-all} prefix=node code=0 step_id
    local -a step_limit=() watchdog=()
    if [[ "$1" == unit ]]; then
        prefix="$phase.node"
        step_limit=(--time="$UNIT_TEST_TIMEOUT")
        # Slurm owns remote-rank cleanup; this also bounds an unresponsive srun client.
        watchdog=(timeout --signal=TERM --kill-after=30s "$((UNIT_TEST_TIMEOUT * 60 + 60))s")
    fi

    "${watchdog[@]}" srun \
        "${step_limit[@]}" \
        --jobid="$SLURM_JOB_ID" \
        --nodes="$nodes" \
        --ntasks="$nodes" \
        --ntasks-per-node=1 \
        --cpus-per-task=288 \
        --gpus-per-node=4 \
        --mpi=pmix \
        --network=disable_rdzv_get \
        --input=none \
        --kill-on-bad-exit=1 \
        --environment="$ENVIRONMENT" \
        --container-mounts="$mounts" \
        --output="$output_dir/$prefix-%t.out" \
        --error="$output_dir/$prefix-%t.err" \
        bash "$REPO_ROOT/tests/run_tests.sh" \
        --node "$workers_per_node" "$output_dir" "$@" || code=$?

    if [[ "$1" == unit && "$code" != 0 ]]; then
        if [[ "$code" == 124 || "$code" == 137 ]]; then
            # Never leave a remote step behind if the local watchdog had to kill srun.
            if [[ -f "$output_dir/slurm-$phase.step-id" ]]; then
                step_id=$(< "$output_dir/slurm-$phase.step-id")
                if [[ "$step_id" =~ ^[0-9]+$ ]]; then
                    scancel --signal=KILL "$SLURM_JOB_ID.$step_id" || true
                fi
            fi
        elif grep -q 'DUE TO TIME LIMIT' "$output_dir/$phase.node-"*.err 2>/dev/null; then
            code=124
        elif [[ -f "$output_dir/slurm-$phase.step-id" ]]; then
            step_id=$(< "$output_dir/slurm-$phase.step-id")
            if [[ "$step_id" =~ ^[0-9]+$ ]] &&
                timeout 5s sacct -nP -j "$SLURM_JOB_ID.$step_id" --format=State | grep -q '^TIMEOUT'; then
                code=124
            fi
        fi
    fi
    return "$code"
}

functional_block_reason() {
    local test_case=$1
    local reason config_file mock_data asset model

    reason=$(TEST_CASE="$test_case" yq \
        '.products[] | select(.test_case[] == strenv(TEST_CASE)) | .blocked // ""' "$RECIPE")
    if [[ -n "$reason" ]]; then
        echo "$reason"
        return
    fi

    model=$(yq '.spec.model' "$RECIPE")
    config_file="$REPO_ROOT/tests/functional_tests/test_cases/$model/$test_case/model_config.yaml"
    if [[ ! -f "$config_file" ]]; then
        echo "Missing model config: $config_file"
        return
    fi

    mock_data=$(yq '.MODEL_ARGS."--mock-data" // false' "$config_file")
    if [[ "$mock_data" != true ]]; then
        for asset in \
            my-gpt3_00_text_document.bin \
            my-gpt3_00_text_document.idx \
            bpe/vocab.json \
            bpe/merges.txt; do
            if [[ ! -f "$DATA_PATH/text/common_pile/v01_filtered_data/$asset" ]]; then
                echo "Missing dataset asset: $asset"
                return
            fi
        done
    fi

    local golden_file="${config_file%/*}/golden_values_dev_gh200.json"
    if [[ ${RECORD_BASELINE:-false} != true && ! -f "$golden_file" ]]; then
        echo "Missing GH200 baseline: $golden_file"
    fi
}

has_runnable_functional_case() {
    local group ranks test_case reason
    while read -r group ranks test_case; do
        if [[ "$SELECTED_GROUP" != all && "$SELECTED_GROUP" != "$group" ]]; then
            continue
        fi
        reason=$(functional_block_reason "$test_case")
        if [[ -z "$reason" ]] && (( ranks <= NODES * 4 )); then
            return 0
        fi
    done <<< "$TEST_GROUPS"
    return 1
}

record_result() {
    awk -F '\t' -v OFS='\t' -v group="$1" -v result="$2" -v code="$3" \
        '$1 == group { $3 = result; $4 = code } { print }' \
        "$LOG_DIR/results.tsv" > "$LOG_DIR/results.tsv.tmp"
    mv "$LOG_DIR/results.tsv.tmp" "$LOG_DIR/results.tsv"
}

run_groups() {
    local group ranks test_case reason
    local nodes workers_per_node exit_code result phase phase_code phase_result rank rank_code
    local missing_junit missing_exit bad_rank
    local suite_exit_code=0

    printf 'group\tranks\tresult\texit_code\n' > "$LOG_DIR/results.tsv"
    while read -r group ranks test_case; do
        if [[ "$SELECTED_GROUP" == all || "$SELECTED_GROUP" == "$group" ]]; then
            printf '%s\t%s\tNOT_RUN\t-\n' "$group" "$ranks" >> "$LOG_DIR/results.tsv"
        fi
    done <<< "$TEST_GROUPS"

    while read -r group ranks test_case; do
        if [[ "$SELECTED_GROUP" != all && "$SELECTED_GROUP" != "$group" ]]; then
            continue
        fi

        nodes=$(( (ranks + 3) / 4 ))
        workers_per_node=$(( ranks / nodes ))
        reason=

        if (( nodes > NODES )); then
            reason="Needs $nodes nodes ($ranks ranks)"
        elif [[ "$TEST_SUITE" == functional ]]; then
            reason=$(functional_block_reason "$test_case")
        fi

        if [[ -n "$reason" ]]; then
            echo "BLOCKED: $group: $reason"
            record_result "$group" BLOCKED 1
            suite_exit_code=1
            continue
        fi

        echo "RUNNING: $group ($ranks ranks on $nodes nodes)"
        # Keep unfinished groups visible if the allocation ends early.
        record_result "$group" INCOMPLETE 1
        exit_code=0
        result=COMPLETED
        if [[ "$TEST_SUITE" == unit ]]; then
            mkdir -p "$LOG_DIR/$group"
            printf 'phase\tranks\tresult\tstep_exit_code\tmissing_junit\tmissing_exit\tfailed_ranks\n' \
                > "$LOG_DIR/$group/phase-results.tsv"
            for phase in prod experimental; do
                phase_code=0
                run_step "$nodes" "$workers_per_node" "$LOG_DIR/$group" \
                    unit "$test_case" "$phase" || phase_code=$?
                missing_junit=0
                missing_exit=0
                bad_rank=0
                # Check rank artifacts after every node has left the phase.
                for ((rank = 0; rank < ranks; rank++)); do
                    [[ -f "$LOG_DIR/$group/repeat-1-$phase.rank-$rank.xml" ]] || \
                        missing_junit=$((missing_junit + 1))
                    if [[ -f "$LOG_DIR/$group/repeat-1-$phase.rank-$rank.exitcode" ]]; then
                        rank_code=$(< "$LOG_DIR/$group/repeat-1-$phase.rank-$rank.exitcode")
                        if [[ "$rank_code" != 0 && "$rank_code" != 5 ]]; then bad_rank=$((bad_rank + 1)); fi
                    else
                        missing_exit=$((missing_exit + 1))
                    fi
                done
                if [[ "$phase_code" == 124 ]]; then
                    phase_result=TIMED_OUT
                    result=TIMED_OUT
                    exit_code=124
                elif (( missing_junit || missing_exit )); then
                    phase_result=INCOMPLETE
                    if [[ "$result" != TIMED_OUT ]]; then result=INCOMPLETE; exit_code=1; fi
                elif (( phase_code || bad_rank )); then
                    phase_result=FAILED
                    if [[ "$result" == COMPLETED ]]; then result=FAILED; exit_code=1; fi
                else
                    phase_result=COMPLETED
                fi
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$phase" "$ranks" "$phase_result" "$phase_code" "$missing_junit" "$missing_exit" "$bad_rank" \
                    >> "$LOG_DIR/$group/phase-results.tsv"
                echo "$phase_result: $group/$phase (missing JUnit: $missing_junit; exit markers: $missing_exit)"
                if [[ ${UNIT_TEST_FAIL_FAST:-true} != false && \
                    ( "$phase_result" == FAILED || "$phase_result" == INCOMPLETE ) ]]; then
                    break
                fi
            done
            if [[ "$result" == COMPLETED && ${UNIT_TEST_COLLECT_ONLY:-false} != true ]] &&
                ! grep -q '<testcase[ >]' "$LOG_DIR/$group/"repeat-1-*.rank-0.xml; then
                echo "No tests selected in either marker phase: $group" >&2
                result=FAILED
                exit_code=5
            fi
        else
            run_step "$nodes" "$workers_per_node" "$LOG_DIR/$group" \
                "$TEST_SUITE" "$test_case" || exit_code=$?
            if (( exit_code != 0 )); then result=FAILED; fi
        fi

        if (( exit_code == 0 )); then
            if [[ "$TEST_SUITE" == unit && ${UNIT_TEST_COLLECT_ONLY:-false} == true ]]; then
                result=COLLECTED
            fi
            if [[ "$TEST_SUITE" == functional && ${RECORD_BASELINE:-false} == true ]]; then
                result=RECORDED
            fi
            echo "$result: $group"
            record_result "$group" "$result" 0
        else
            echo "$result: $group (exit $exit_code)"
            record_result "$group" "$result" "$exit_code"
            suite_exit_code=1
        fi
    done <<< "$TEST_GROUPS"

    return "$suite_exit_code"
}

publish_results_on_exit() {
    local test_code=$? publish_code=0
    bash "$REPO_ROOT/tests/publish_results.sh" "$LOG_DIR" \
        --suite "$TEST_SUITE" --publish || publish_code=$?
    if (( publish_code != 0 )); then
        echo "GitHub report dispatch failed (exit $publish_code); results remain in $LOG_DIR" >&2
    fi
    if (( test_code != 0 )); then
        exit "$test_code"
    fi
    exit "$publish_code"
}

case "${1:-}" in
    --help)
        usage
        exit 0
        ;;
    --node)
        shift
        run_on_node "$@"
        exit
        ;;
    unit)
        RECIPE="$RECIPE_DIR/unit-tests.yaml"
        ;;
    functional)
        RECIPE=${FUNCTIONAL_RECIPE:-$RECIPE_DIR/gpt.yaml}
        export FUNCTIONAL_RECIPE="$RECIPE"
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac

TEST_SUITE=$1
SELECTED_GROUP=${2:-all}
TEST_GROUPS=$(yq '. as $recipe | .products[] | [.name // .test_case[0], .ranks // $recipe.spec.ranks, .test_case[0]] | @tsv' "$RECIPE")

if [[ "$SELECTED_GROUP" == --list ]]; then
    printf '%s\n' "$TEST_GROUPS"
    exit 0
fi
if (( $# > 2 )); then
    usage >&2
    exit 1
fi
if [[ "$SELECTED_GROUP" != all ]] &&
    ! awk -v group="$SELECTED_GROUP" '$1 == group { found = 1 } END { exit !found }' <<< "$TEST_GROUPS"; then
    echo "Unknown test group: $SELECTED_GROUP" >&2
    exit 1
fi

: "${SCRATCH:?Set SCRATCH to an Iopsstor scratch directory}"

export PUBLISH_RESULTS=${PUBLISH_RESULTS:-false}
if [[ "$PUBLISH_RESULTS" != true && "$PUBLISH_RESULTS" != false ]]; then
    echo "PUBLISH_RESULTS must be true or false" >&2
    exit 1
fi
if [[ "$PUBLISH_RESULTS" == true ]]; then
    for tool in gh uv; do
        if ! command -v "$tool" >/dev/null; then
            echo "PUBLISH_RESULTS=true requires $tool on the launcher host" >&2
            exit 1
        fi
    done
fi

DEFAULT_LOG_ROOT=$SCRATCH/tmp
if [[ "$TEST_SUITE" == functional ]]; then
    DEFAULT_LOG_ROOT=/ritom/scratch/cscs/$USER/tmp
fi
LOG_DIR=${LOG_DIR:-$DEFAULT_LOG_ROOT/megatron-tests-$(date +%Y%m%dT%H%M%S)-$$}
mkdir -p "$LOG_DIR"
LOG_DIR=$(realpath "$LOG_DIR")
if [[ -n $(find "$LOG_DIR" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo "LOG_DIR must be empty; choose a new directory: $LOG_DIR" >&2
    exit 1
fi
export LOG_DIR
export DATA_PATH=${DATA_PATH:-/capstor/store/cscs/swissai/infra01/megatron-test-assets}
export RECORD_BASELINE=${RECORD_BASELINE:-false}
export ENVIRONMENT=${ENVIRONMENT:-$RECIPE_DIR/environment.toml}
export PARTITION=${PARTITION:-preemptable}
export NODES=${NODES:-${SLURM_JOB_NUM_NODES:-2}}
export UNIT_TEST_TIMEOUT=${UNIT_TEST_TIMEOUT:-30}
if [[ "$TEST_SUITE" == unit && ! "$UNIT_TEST_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "UNIT_TEST_TIMEOUT must be a positive number of minutes" >&2
    exit 1
fi
if [[ "$TEST_SUITE" == functional ]] && ! has_runnable_functional_case; then
    mkdir "$LOG_DIR/.run-lock"
    cp "$RECIPE" "$LOG_DIR/recipe.yaml"
    if [[ "$PUBLISH_RESULTS" == true ]]; then
        {
            echo "Commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
            echo "Branch: $(git -C "$REPO_ROOT" branch --show-current)"
        } > "$LOG_DIR/launch.txt"
        trap publish_results_on_exit EXIT
    fi
    run_groups
    exit
fi

if [[ ! -f "$ENVIRONMENT" ]]; then
    echo "Missing EDF: $ENVIRONMENT" >&2
    exit 1
fi
if [[ "$TEST_SUITE" == unit ]]; then
    if [[ ! -d "$DATA_PATH/unit-tests" ]]; then
        echo "Missing unit-test fixtures: $DATA_PATH/unit-tests" >&2
        exit 1
    fi
    export UNIT_TEST_DATA_DIR="$LOG_DIR/unit-test-data"
fi
if ! sinfo -h -p "$PARTITION" -o '%a' | grep -qx up; then
    echo "Partition $PARTITION is unavailable" >&2
    exit 1
fi

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "Requesting $NODES nodes. Tests start when the allocation is ready."
    exec salloc \
        --account="${ACCOUNT:-csstaff}" \
        --partition="$PARTITION" \
        --nodes="$NODES" \
        --ntasks="$NODES" \
        --ntasks-per-node=1 \
        --cpus-per-task=288 \
        --gpus-per-node=4 \
        --exclusive \
        --time="${TIME:-05:00:00}" \
        bash "$REPO_ROOT/tests/run_tests.sh" "$@"
fi

mkdir "$LOG_DIR/.run-lock"
cp "$RECIPE" "$LOG_DIR/recipe.yaml"
if [[ "$TEST_SUITE" == unit ]]; then
    cp -rL --reflink=auto -- "$DATA_PATH/unit-tests" "$UNIT_TEST_DATA_DIR.tmp-$$"
    chmod -R u+w "$UNIT_TEST_DATA_DIR.tmp-$$"
    mv "$UNIT_TEST_DATA_DIR.tmp-$$" "$UNIT_TEST_DATA_DIR"
fi

{
    echo "Source: $REPO_ROOT"
    echo "Commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "Branch: $(git -C "$REPO_ROOT" branch --show-current)"
    echo "EDF: $ENVIRONMENT"
    echo "Job: $SLURM_JOB_ID"
    echo "Logs: $LOG_DIR"
    echo "Data: $DATA_PATH"
    if [[ "$TEST_SUITE" == unit ]]; then
        echo "Unit phase timeout: $UNIT_TEST_TIMEOUT minutes"
        echo "Unit fixtures: $UNIT_TEST_DATA_DIR"
    fi
} | tee "$LOG_DIR/launch.txt"
if [[ "$TEST_SUITE" == unit ]]; then
    printf '%s\n' "$TEST_GROUPS" > "$LOG_DIR/unit-groups.tsv"
    git -C "$REPO_ROOT" diff HEAD -- tests/unit_tests tests/run_tests.sh tests/test_utils \
        > "$LOG_DIR/test-changes.patch"
    for file in tests/run_tests.sh tests/unit_tests/run_ci_test.sh \
        tests/unit_tests/find_test_cases.py tests/test_utils/setup_environment.sh \
        tests/test_utils/run_python_worker.sh; do
        mkdir -p "$LOG_DIR/launcher-source/$(dirname "$file")"
        cp "$REPO_ROOT/$file" "$LOG_DIR/launcher-source/$file"
    done
fi
if [[ $(realpath "$ENVIRONMENT") != "$LOG_DIR/environment.toml" ]]; then
    cp "$ENVIRONMENT" "$LOG_DIR/environment.toml"
fi

if [[ "$PUBLISH_RESULTS" == true ]]; then
    trap publish_results_on_exit EXIT
fi
run_groups
