#!/bin/bash
# Unattended launch of the MegaChonk SFT run once the rebalanced mix is rendered on KISTI (user, 2026-11-22: batch
# 512, 16K context with longer conversations removed, real 96,000 weights, 64 nodes). Runs detached on the login node:
#   nohup setsid bash ~/projects/megachonk-sft/launch_when_ready.sh > /scratch/hanint06/megachonk-sft/logs/launch_when_ready.log 2>&1 &
# 1. wait for <render>/train/tokens.{bin,idx} + manifest.json, sizes unchanged for 10 min
# 2. require the rehearsal job (REHEARSAL) to have COMPLETED
# 3. filter to conversations <= 16,384 tokens (filter_long_index.py; new index, .bin linked)
# 4. sbatch sft-megachonk.sh: GBS 512, 1 epoch, saves every 220 steps (newest kept), 12 h jobs chained (AUTO_REQUEUE)
set -uo pipefail
R=${R:-/scratch/hanint06/sft-tokens/megachonk/sft_v1_v1__apertus2_v1}
F=${F:-/scratch/hanint06/megachonk-sft/data/sft_v1_v1_16k}
REHEARSAL=${REHEARSAL:-1337167}
L=/home01/hanint06/projects/megachonk-sft
GBS=512; SEQ=16384
echo "$(date) start: waiting for $R"
for i in $(seq 1 2880); do                       # up to 10 days
    if [ -f "$R/train/tokens.bin" ] && [ -f "$R/train/tokens.idx" ] && [ -f "$R/manifest.json" ]; then
        s1=$(stat -c %s "$R/train/tokens.bin" "$R/train/tokens.idx" | paste -sd+); sleep 600
        s2=$(stat -c %s "$R/train/tokens.bin" "$R/train/tokens.idx" | paste -sd+)
        [ "$s1" = "$s2" ] && break
        echo "$(date) render still growing"
    else
        sleep 300
    fi
done
[ -f "$R/manifest.json" ] || { echo "$(date) ERROR: render never completed"; exit 1; }
echo "$(date) render complete: $(ls -la $R/train | tr '\n' ' ')"
# Wait for the rehearsal to end (it may still be queued), then launch only if it COMPLETED.
for i in $(seq 1 1440); do
    st=$(sacct -j "$REHEARSAL" -X -n -o State | head -1 | tr -d ' ')
    case "$st" in PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|"") sleep 120 ;; *) break ;; esac
done
echo "$(date) rehearsal $REHEARSAL: $st"
[ "$st" = COMPLETED ] || { echo "$(date) NOT LAUNCHING: rehearsal job $REHEARSAL is '$st', not COMPLETED"; exit 1; }
rm -rf "$F"; mkdir -p "$F"
python3 "$L/filter_long_index.py" "$R/train" "$F/all" --max-tokens $SEQ || { echo "$(date) ERROR: filter failed"; exit 1; }
# Iterations for the requeue chain: packs of 16K at >= 97% utilization (v0.2: 99%), rounded up; a slight overestimate
# only means one last successor job that finds training finished.
kept=$(python3 -c "
import sys; sys.path.insert(0, '$L'); from filter_long_index import read_index
print(int(read_index('$F/all/tokens.idx')[1].sum(dtype='int64')))")
T=$(python3 -c "import math; print(math.ceil($kept / ($SEQ * 0.97) / $GBS))")
echo "$(date) kept tokens $kept -> TOTAL_ITERS $T"
WANDB_API_KEY="$(cat /home01/hanint06/.wandb-api-key)"; export WANDB_API_KEY
cd /scratch/hanint06/megachonk-sft
# Compile caches node-local: the group scratch is at 97 % of its 10M-file limit and refused the caches' files
# (rehearsal 1336104, Errno 122); the warm archive stays on scratch.
export JIT_LOCAL=1 JITCACHE_ROOT=/tmp/megachonk-jit-hanint06 \
    TRITON_TAR=/scratch/hanint06/megachonk-sft/triton-cache/warm/sft-megachonk-l61-seq16384-tp2-pp16-ep16-etp1-cp2-kcp-mbs1.tar
export GBS EPOCHS=1 EXP_NAME=sft/megachonk-sft-v1-16k-gbs512 \
    BASE_CKPT_DIR=/scratch/hanint06/megachonk-sft/checkpoints/megachonk-iter96000-megatron \
    SFT_DATA_ROOT="$F" SFT_DATA_GLOB=tokens AUTO_REQUEUE=1 TOTAL_ITERS=$T WANDB_PROJECT=ap2-sft-ablations WANDB_ENTITY=apertus
sbatch --time=12:00:00 "$L/sft-megachonk.sh" && echo "$(date) LAUNCHED"
