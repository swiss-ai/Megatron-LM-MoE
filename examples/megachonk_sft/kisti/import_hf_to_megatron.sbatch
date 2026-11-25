#!/bin/bash
#SBATCH --job-name=megachonk-import
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=megatron-lm
#SBATCH --output=/scratch/hanint06/megachonk-sft/logs/%x-%j.out
# HF megachonk 96,000 (staged, with the embedding-norm modeling code) -> Megatron torch_dist for SFT, with the
# converter's Megatron (42e7a63 + --embedding-norm) and Bridge (6ccdb3c + embedding_norm mapping) worktrees, in the
# Apertus 2 NeMo-RL image (the image the Clariden converter runs). Little cross-node traffic: NCCL over sockets on
# Slingshot (hsn0); the rendezvous on the first node's hsn0 address.
set -euo pipefail
HF=${HF:-/scratch/hanint06/models/apertus2-megachonk-iter96000}
OUT=${OUT:-/scratch/hanint06/megachonk-sft/checkpoints/megachonk-iter96000-megatron}
M=/home01/hanint06/projects/megatron-lm-moe-42e7a63-embnorm
B=/home01/hanint06/projects/megatron-bridge-apertus2-embnorm
SCRIPT=/home01/hanint06/projects/megachonk-sft/import_hf_to_megatron.py
IMAGE=/scratch/hanint02/images/nemo-rl-apertus2-cfcfede4-ucclfix.sqsh
[ ! -e "$OUT" ] || { echo "ERROR: $OUT exists"; exit 1; }
mkdir -p "$(dirname "$OUT")"
first=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$first-hsn0 MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
echo "rendezvous $MASTER_ADDR ($(getent hosts $MASTER_ADDR | awk '{print $1}')), $HF -> $OUT"
srun --mpi=pmix --network=disable_rdzv_get --container-image=$IMAGE --container-writable \
  --container-mounts=/scratch/hanint02:/scratch/hanint02 -l -u bash -c "
    export RANK=\$SLURM_PROCID WORLD_SIZE=\$SLURM_NTASKS LOCAL_RANK=\$SLURM_LOCALID
    export PYTHONPATH=$M:$B/src PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export HF_MODULES_CACHE=/tmp/hfmod-\$SLURM_PROCID TMPDIR=/tmp/imp-\$SLURM_PROCID CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_SOCKET_IFNAME=hsn0 GLOO_SOCKET_IFNAME=hsn0 NCCL_IB_DISABLE=1
    mkdir -p \$TMPDIR
    unset VIRTUAL_ENV PYTHONHOME
    exec /usr/bin/python3 -u $SCRIPT $HF $OUT"
ls -la "$OUT"; du -sh "$OUT"
echo IMPORT_DONE
