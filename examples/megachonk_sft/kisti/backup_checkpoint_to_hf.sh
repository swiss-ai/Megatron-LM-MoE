#!/bin/bash
# KISTI compute ended 2026-11-23 10:12 KST (SFT stopped at step 1,574 of 1,890). Back up the newest SFT checkpoint
# (Megatron DCP, iteration 1,540, 5.5 TB incl. optimizer) to a private HF repo from the data mover, so Clariden can
# download and convert it. Retries until UPLOAD_DONE (each batch is its own commit; reruns skip committed files).
#   setsid nohup bash backup_1540.sh > /scratch/hanint06/megachonk-sft/backup-1540.log 2>&1 < /dev/null &
set -uo pipefail
export REPO=swiss-ai/megachonk-sft-v1-16k-iter_0001540-megatron
export SRC=/scratch/hanint06/megachonk-sft/runs/results/ckpts/sft/megachonk-sft-v1-16k-gbs512/iter_0001540
export PATH_PREFIX=iter_0001540 BATCH_GIB=400
source <(grep "^export HF_TOKEN=" /home01/hanint06/projects/rl-infra/site.env)
export HF_HOME=/scratch/hanint06/tmp/hf_home_megachonk HF_XET_HIGH_PERFORMANCE=1
L=/home01/hanint06/projects/megachonk-sft
for attempt in $(seq 1 30); do
    echo "$(date '+%F %T') attempt $attempt"
    nice /home01/hanint06/venvs/hfdl-x86/bin/python -u "$L/hf_upload_batched.py" && { echo "$(date '+%F %T') BACKUP_OK"; exit 0; }
    sleep 300
done
echo "$(date '+%F %T') BACKUP_FAILED"; exit 1
