#!/bin/bash
# Clariden: full-depth warm-ups (64 nodes, random init) then the resume of the MegaChonk SFT run from iteration 1,540.
set -euo pipefail
S=/iopsstor/scratch/cscs/dmelikidze/megachonk-sft
L=${L:-$(cd "$(dirname "$0")" && pwd)/sft_megachonk.sh}   # this folder
FULL_TAR=$S/triton-cache/warm/sft-megachonk-l61-seq16384-tp2-pp16-ep16-etp1-cp2-kcp-mbs1.tar
[ -f "$FULL_TAR" ] || cp "$S/triton-cache/warm/debug-l9-tp2-pp2-ep8-cp2.tar" "$FULL_TAR"   # seed with the debug kernels
cd "$S"
COMMON="SFT_DATA_ROOT=$S/data/sft_v1_v1_16k,SFT_DATA_GLOB=tokens,EPOCHS=1,TRITON_TAR=$FULL_TAR"
W1=$(sbatch --parsable --nodes=64 --time=01:00:00 --job-name=sft-megachonk-warmup \
     --export=ALL,$COMMON,WARMUP_NODES=64,GBS=64,WARMUP_ITERS=3 "$L" --warmup)
W2=$(sbatch --parsable --nodes=64 --time=01:00:00 --job-name=sft-megachonk-warmup --dependency=afterany:$W1 \
     --export=ALL,$COMMON,WARMUP_NODES=64,GBS=64,WARMUP_ITERS=3 "$L" --warmup)
WANDB_API_KEY="$(cat ~/.wandb-api-key)"; export WANDB_API_KEY
R=$(sbatch --parsable --nodes=64 --time=12:00:00 --dependency=afterany:$W2,afterok:4010227 \
     --export=ALL,$COMMON,GBS=512,EXP_NAME=sft/megachonk-sft-v1-16k-gbs512,AUTO_REQUEUE=1,TOTAL_ITERS=1890,ALLOW_COLD_START=1,WANDB_PROJECT=ap2-sft-ablations,WANDB_ENTITY=apertus \
     "$L" --resume)
echo "warm-up 1: $W1, warm-up 2: $W2 (after $W1), resume: $R (after $W2 and the checkpoint download 4010227)"
