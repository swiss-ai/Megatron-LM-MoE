#!/bin/bash
#SBATCH --job-name=megachonk-export
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=megatron-lm
#SBATCH --output=/scratch/hanint06/megachonk-sft/logs/%x-%j.out
# MegaChonk SFT checkpoint (native Megatron torch_dist, iteration folder) -> HF, as the Clariden converter exported the
# pretraining checkpoints (hfconverter convert_checkpoint.py + Apertus2 key check, config-only reference, TP1 PP4 EP8 on
# 32 GPUs) with the converter's Megatron (42e7a63 + --embedding-norm) and Bridge (6ccdb3c + embedding_norm) worktrees.
# The reference has the SFT data's tokenizer (Apertus_2_instruct-b801ff1).
#   CKPT=<iter dir> OUT=<hf dir> sbatch export_sft_to_hf.sbatch
set -euo pipefail
CKPT=${CKPT:?iteration folder of the SFT checkpoint}
OUT=${OUT:?HF output folder}
H=/home01/hanint06/projects/megachonk-sft/hfconverter
REF=${REF:-$H/hf-reference-sft}
M=/home01/hanint06/projects/megatron-lm-moe-42e7a63-embnorm
B=/home01/hanint06/projects/megatron-bridge-apertus2-embnorm
IMAGE=/scratch/hanint02/images/nemo-rl-apertus2-cfcfede4-ucclfix.sqsh
[ ! -e "$OUT" ] || { echo "ERROR: $OUT exists"; exit 1; }
mkdir -p "$(dirname "$OUT")"
first=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$first-hsn0 MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
echo "rendezvous $MASTER_ADDR ($(getent hosts $MASTER_ADDR | awk '{print $1}')), $CKPT -> $OUT (reference $REF)"
srun --mpi=pmix --network=disable_rdzv_get --container-image=$IMAGE --container-writable \
  --container-mounts=/scratch/hanint02:/scratch/hanint02 -l -u bash -c "
    export RANK=\$SLURM_PROCID WORLD_SIZE=\$SLURM_NTASKS LOCAL_RANK=\$SLURM_LOCALID
    export BRIDGE_ROOT=$B MEGATRON_ROOT=$M HFCONVERTER_ROOT=$H
    export PYTHONPATH=$M:$B/src:$B/scripts/conversion:$H PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export HF_MODULES_CACHE=/tmp/hfmod-\$SLURM_PROCID TMPDIR=/tmp/exp-\$SLURM_PROCID CUDA_DEVICE_MAX_CONNECTIONS=1
    export TRITON_CACHE_DIR=/tmp/triton-exp-\$SLURM_PROCID TORCHINDUCTOR_CACHE_DIR=/tmp/ind-exp-\$SLURM_PROCID
    export NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_SOCKET_IFNAME=hsn0 GLOO_SOCKET_IFNAME=hsn0 NCCL_IB_DISABLE=1
    mkdir -p \$TMPDIR
    unset VIRTUAL_ENV PYTHONHOME
    exec /usr/bin/python3 -u $H/check_apertus2_export_keys.py export $CKPT $REF $OUT \
      --tp 1 --pp 4 --ep 8 --etp 1 --dtype bf16 --reference-mode config-only \
      --pipeline-layout 'Et*16|t*15|t*15|t*15L' \
      --model-override kda_legacy_gate_out_proj_bias=false \
      --model-override linear_attn_a_log_per_channel=true \
      --model-override 'moe_router_quantile_balancing_method=\"marin_histogram\"' \
      --model-override moe_router_quantile_balancing_marin_num_bins=10000 \
      --model-override embedding_norm=true \
      --trust-remote-code --max-shard-size-gib 16"
[ -f "$OUT/CONFIG_ONLY_EXPORT_COMPLETE" ] || { echo "ERROR: export not marked complete"; exit 1; }
grep -q '"model.embedding_norm.weight"' "$OUT/model.safetensors.index.json" || { echo "ERROR: no embedding norm in export"; exit 1; }
cp /home01/hanint06/sft-data-env/tokenizer/Apertus_2_instruct-b801ff1/generation_config.json "$OUT/" 2>/dev/null || true
ls "$OUT" | wc -l; du -sh "$OUT"
echo EXPORT_DONE
