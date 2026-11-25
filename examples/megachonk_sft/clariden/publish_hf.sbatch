#!/bin/bash
#SBATCH --job-name=sft1540-publish
#SBATCH -A infra01
#SBATCH -p xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/users/dmelikidze/transfer/logs/%j-%x.out
#SBATCH --error=/users/dmelikidze/transfer/logs/%j-%x.out
# After sft1540_convert.sbatch: make the export self-contained (norm-aware modeling code, auto_map, SFT tokenizer,
# README; finalize_hf_export.sh from the KISTI backup with Clariden paths) and upload it to the private repo
# swiss-ai/megachonk-sft-v1-16k-iter_0001540 (megachonk_upload.py: skips files already there, checks sizes).
set -euo pipefail
# aarch64 compute node: hfup-node venv; x86 xfer node: hf-x86 venv
PY=/users/dmelikidze/venvs/hfup-node/bin/python; [ "$(uname -m)" = aarch64 ] || PY=/iopsstor/scratch/cscs/dmelikidze/venvs/hf-x86/bin/python
S=/iopsstor/scratch/cscs/dmelikidze
B=$HOME/kisti-backup-20261123-megachonk-sft/megachonk-sft
E=$S/megachonk-hf/sft-v1-16k-iter_0001540
export C=$B/embnorm-hf-code T=$B/hfconverter/hf-reference-sft
bash $B/finalize_hf_export.sh "$E" 1540
export SRC=$S/hf-upload/megachonk-sft-v1-16k-iter_0001540 REPO=swiss-ai/megachonk-sft-v1-16k-iter_0001540
mkdir -p "$SRC"
for f in "$E"/*; do [ -f "$f" ] && ln -sfn "$f" "$SRC/$(basename "$f")"; done
export HF_HOME=$S/huggingface HF_XET_HIGH_PERFORMANCE=1
for attempt in 1 2 3 4; do
    $PY -u /users/dmelikidze/transfer/megachonk_upload.py && { echo "$(date) PUBLISHED $REPO"; exit 0; }
    sleep 300
done
echo "ERROR: upload failed 4 times"; exit 1
