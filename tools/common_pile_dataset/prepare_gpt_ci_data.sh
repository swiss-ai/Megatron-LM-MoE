#!/usr/bin/env bash
set -euo pipefail
umask 0002

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-/capstor/store/cscs/swissai/infra01/megatron-test-assets}
OUTPUT_DIR="$DATA_ROOT/text/common_pile/v01_filtered_data"

: "${SCRATCH:?Set SCRATCH to an Iopsstor scratch directory}"

WORK_DIR=${WORK_DIR:-$SCRATCH/tmp/megatron-gpt-data-$(date +%Y%m%dT%H%M%S)-$$}
mkdir -p "$WORK_DIR"
WORK_DIR=$(realpath "$WORK_DIR")

source "$REPO_ROOT/tests/test_utils/setup_environment.sh"
export PYTHONPYCACHEPREFIX="$WORK_DIR/pycache"
export TOKENIZERS_PARALLELISM=false

PYTHON_COMMAND=(uv run --no-project --python "$PYTHON_BIN" python)

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output already exists: $OUTPUT_DIR" >&2
    exit 1
fi

"${PYTHON_COMMAND[@]}" -c 'import datasets, nltk, torch, transformers'

OUTPUT_PARENT=$(dirname "$OUTPUT_DIR")
mkdir -p "$OUTPUT_PARENT"
LOCK_DIR="$OUTPUT_PARENT/.gpt-preparation.lock"
if ! mkdir "$LOCK_DIR"; then
    echo "Another build holds the lock: $LOCK_DIR" >&2
    exit 1
fi
trap 'rmdir -- "$LOCK_DIR"' EXIT

STAGING_DIR=$(mktemp -d "$OUTPUT_PARENT/.gpt-build-XXXXXX")
chmod g+rx "$STAGING_DIR"
echo "Staging output: $STAGING_DIR"
echo "Working files: $WORK_DIR"

"${PYTHON_COMMAND[@]}" - "$STAGING_DIR/source.json" > "$WORK_DIR/revisions.txt" <<'PY'
import json
import sys
import urllib.request

sources = {
    "dataset": "https://huggingface.co/api/datasets/common-pile/comma_v0.1_training_dataset",
    "tokenizer": "https://huggingface.co/api/models/openai-community/gpt2",
}
revisions = {}
for name, url in sources.items():
    with urllib.request.urlopen(url, timeout=60) as response:
        revisions[name] = json.load(response)["sha"]

with open(sys.argv[1], "w") as metadata:
    json.dump({"sources": sources, "revisions": revisions, "documents": 12000000}, metadata, indent=2)
    metadata.write("\n")

print(revisions["dataset"], revisions["tokenizer"])
PY

read -r DATASET_REVISION GPT2_REVISION < "$WORK_DIR/revisions.txt"

git -C "$REPO_ROOT" rev-parse HEAD > "$STAGING_DIR/source-commit.txt"
if [[ -n ${ENVIRONMENT:-} ]]; then
    cp "$ENVIRONMENT" "$STAGING_DIR/environment.toml"
fi
git -C "$REPO_ROOT" diff -- \
    tools/common_pile_dataset/create_common_pile_ci_dataset.py \
    tools/preprocess_data.py > "$STAGING_DIR/preparation.patch"
cp "$REPO_ROOT/tools/common_pile_dataset/create_common_pile_ci_dataset.py" \
    "$STAGING_DIR/preparation-script.py"

"${PYTHON_COMMAND[@]}" - > "$STAGING_DIR/packages.json" <<'PY'
import importlib.metadata
import json

packages = ["datasets", "transformers", "tokenizers", "nltk", "numpy", "torch"]
versions = {name: importlib.metadata.version(name) for name in packages}
print(json.dumps(versions, indent=2))
PY

"${PYTHON_COMMAND[@]}" \
    "$REPO_ROOT/tools/common_pile_dataset/create_common_pile_ci_dataset.py" \
    --output-dir "$STAGING_DIR" \
    --work-dir "$WORK_DIR" \
    --megatron-dir "$REPO_ROOT" \
    --dataset-name common-pile/comma_v0.1_training_dataset \
    --dataset-revision "$DATASET_REVISION" \
    --gpt2-revision "$GPT2_REVISION" \
    --num-documents 12000000 \
    --models gpt \
    --download-vocab \
    --keep-jsonl \
    --workers "${WORKERS:-4}"

cd "$REPO_ROOT"
"${PYTHON_COMMAND[@]}" - "$STAGING_DIR/my-gpt3_00_text_document" <<'PY'
import sys

from megatron.core.datasets.indexed_dataset import IndexedDataset

dataset = IndexedDataset(sys.argv[1])
documents = len(dataset.document_indices) - 1
if documents != 12000000:
    raise RuntimeError(f"Expected 12000000 indexed documents, received {documents}")
if len(dataset) == 0 or dataset[0].size == 0:
    raise RuntimeError("The indexed dataset contains no usable tokens")
print(f"Verified {documents} indexed documents")
PY

(
    cd "$STAGING_DIR"
    sha256sum \
        my-gpt3_00_text_document.bin \
        my-gpt3_00_text_document.idx \
        bpe/vocab.json \
        bpe/merges.txt > SHA256SUMS
    sha256sum --check SHA256SUMS
)

if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output was created by another process: $OUTPUT_DIR" >&2
    exit 1
fi

mv -T "$STAGING_DIR" "$OUTPUT_DIR"
echo "Dataset ready: $OUTPUT_DIR"
echo "Raw JSONL retained in: $WORK_DIR"
