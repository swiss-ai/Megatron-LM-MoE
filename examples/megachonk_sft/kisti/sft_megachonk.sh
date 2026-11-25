#!/bin/bash
#SBATCH --job-name=sft-megachonk
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                           # KISTI: GPU partitions require --gres
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --mem=0                                # KISTI: whole node; 800G exceeds the usable ~780G
#SBATCH --time=12:00:00
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=megatron-lm   # KISTI: mandatory job tag, see `showappl`
#SBATCH --dependency=singleton   # serialize jobs with this name
#SBATCH --no-requeue             # successors are separate jobs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# MegaChonk SFT on KISTI HANGANG: hanint05's Chonk SFT launcher (/scratch/hanint05/runs/apertus-2-sft/tests/
# sft-chonk-test.sh, 2026-11-05) with MegaChonk's architecture (pretraining launcher megachonk.sh: 61 layers, hidden
# 8192, 256 experts top-8, latent 4096, 3 dense + 58 MoE layers, KDA 3:1 with the last layer global, NoPE, per-channel
# A_log, embedding norm), the Megatron worktree ~/projects/megatron-lm-moe-sft (branch sft-megachonk = main 3a68f6307,
# which has --embedding-norm, + hanint05's feat/sft_trainer_ap2), hanint05's image, packages, tokenizer and v0.2 data.
# Ray is not used; torch.distributed meets at the first node's Slingshot address (<node>-hsn0), not its Ethernet name.
# Debug: `bash sft-megachonk.sh --warmup` (random weights, WARMUP_ITERS iterations, no checkpoints, no W&B);
# NUM_LAYERS=9 (or 13, 17, ...: 4k+1 layers) with PP=2 runs a shallow copy on WARMUP_NODES=8.
# Usage: sbatch sft-megachonk.sh [--resume]; DRY_RUN=1 bash sft-megachonk.sh

set -euo pipefail

WARMUP=0
REQUIRE_RESUME=0
RESUME_ARGS=()
while [ $# -gt 0 ]; do
    case $1 in
        --warmup) WARMUP=1; shift ;;
        --resume) REQUIRE_RESUME=1; RESUME_ARGS=(--resume); shift ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--warmup | --resume]"; exit 1 ;;
    esac
done
[ "$WARMUP" = 0 ] || [ "$REQUIRE_RESUME" = 0 ] \
    || { echo "ERROR: --warmup and --resume cannot be combined"; exit 1; }

# Resolve the submitted script path; Slurm runs a spool copy.
SELF=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SELF=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1 || true)
    [ -f "$SELF" ] || echo "WARNING: could not read this job's script path from scontrol; falling back to ${BASH_SOURCE[0]}"
fi
[ -f "$SELF" ] || SELF=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "$SELF")" && pwd)
JOB_NAME=$(sed -n 's/^#SBATCH --job-name=\([^ ]*\).*/\1/p' "$SELF" | head -1)   # the training job's singleton name (header above)
[ -n "$JOB_NAME" ] || { echo "ERROR: no '#SBATCH --job-name=' line in $SELF"; exit 1; }

# Clone only when absent; use existing checkouts unchanged.
fetch_megatron() {   # $1 = target dir, $2 = repository URL, $3 = branch or tag
    local dir=$1 repo=$2 ref=$3 tmp
    [ -f "$dir/pretrain_gpt.py" ] && return 0
    if [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then echo "ERROR: $dir exists but holds no pretrain_gpt.py; remove it or point MEGATRON_LM_DIR elsewhere"; return 1; fi
    echo "megatron  cloning $repo ($ref) into $dir"
    mkdir -p "$(dirname "$dir")" && tmp=$(mktemp -d "$dir.clone.XXXXXX") \
        && git clone --quiet --depth 1 --branch "$ref" "$repo" "$tmp" \
        || { [ -z "${tmp:-}" ] || rm -rf "$tmp"; echo "ERROR: could not clone $repo ($ref); clone it yourself and set MEGATRON_LM_DIR"; return 1; }
    mv -T "$tmp" "$dir" 2>/dev/null || { rm -rf "$tmp"; [ -f "$dir/pretrain_gpt.py" ] || { echo "ERROR: could not move the clone to $dir"; return 1; }; }
}

# Paths (environment overrides supported).
KISTI_SCRATCH=${KISTI_SCRATCH:-/scratch/$USER}       # flash, 15-day access purge: runs, data, caches, images
KISTI_APPSDATA=${KISTI_APPSDATA:-/appsdata/$USER}    # no purge: tokenizer, Python packages
MEGATRON_REPO=${MEGATRON_REPO:-https://github.com/swiss-ai/Megatron-LM-MoE.git}
MEGATRON_REF=${MEGATRON_REF:-feat/sft_trainer_ap2}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-/home01/hanint06/projects/megatron-lm-moe-sft}
HANINT05_APPSDATA=/appsdata/hanint05                 # image, packages, tokenizer and SFT data, read only
CONTAINER_IMAGE=${CONTAINER_IMAGE:-$HANINT05_APPSDATA/images/ngc-pytorch-25.12-kisti-chonk-uccl329641dc.sqsh}
# Versioned SFT data folder of <dataset> symlinks (/appsdata/hanint05/sft-data/README.md).
SFT_DATA_ROOT=${SFT_DATA_ROOT:-$HANINT05_APPSDATA/sft-data/v0.2-chonk-16k-balanced}
SFT_DATA_GLOB=${SFT_DATA_GLOB:-tokens}                      # prefix glob under SFT_DATA_ROOT, without .bin/.idx
SFT_DATA_DEPTH=${SFT_DATA_DEPTH:-2}                        # KISTI: 2 also finds <root>/<dataset>/<glob>.bin (versioned sft-data folders)
LOSS_MASK=${LOSS_MASK:-auto}                                # auto | on | off (README.md)
# Must match the data's tokenizer.
TOKENIZER_MODEL=${TOKENIZER_MODEL:-$HANINT05_APPSDATA/apertus-omni-tokenizer/tokenizers/Apertus_2_instruct}
WORK_DIR=${WORK_DIR:-$KISTI_SCRATCH/megachonk-sft/runs}   # checkpoints, tensorboard, wandb
# Native MegaChonk pretraining checkpoint (with its gains); none on KISTI yet: only --warmup runs until it is copied.
BASE_CKPT_DIR=${BASE_CKPT_DIR:-$KISTI_SCRATCH/megachonk-sft/checkpoints/megachonk-pretraining-native}
BASE_CKPT_ITER=${BASE_CKPT_ITER:-}                          # set to start from an intermediate iteration
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-$KISTI_SCRATCH/megachonk-sft/dataset-cache}
WANDB_PROJECT=${WANDB_PROJECT:-ap2-sft-ablations}
WANDB_ENTITY=${WANDB_ENTITY:-apertus}
# WANDB_API_KEY enables W&B.

if [ -z "${DRY_RUN:-}" ]; then
    fetch_megatron "$MEGATRON_LM_DIR" "$MEGATRON_REPO" "$MEGATRON_REF" || exit 1
elif [ ! -f "$MEGATRON_LM_DIR/pretrain_gpt.py" ]; then
    echo "megatron  (DRY_RUN) $MEGATRON_LM_DIR is absent; a real run clones $MEGATRON_REPO ($MEGATRON_REF) there"
fi
[ -f "$CONTAINER_IMAGE" ]                  || { echo "ERROR: container image not found: $CONTAINER_IMAGE"; exit 1; }
[ -d "$SFT_DATA_ROOT" ]                    || { echo "ERROR: SFT_DATA_ROOT not found: $SFT_DATA_ROOT"; exit 1; }
for ctl in output_start_token output_end_token wait_token; do
    grep -q "$ctl" "$TOKENIZER_MODEL/tokenizer_config.json" 2>/dev/null \
        || { echo "ERROR: $TOKENIZER_MODEL registers no extra_special_tokens.$ctl; --ap-sft needs output_start_token, output_end_token and wait_token"; exit 1; }
done
[ -f "$TOKENIZER_MODEL/tokenizer.json" ]   || { echo "ERROR: no tokenizer.json under TOKENIZER_MODEL=$TOKENIZER_MODEL"; exit 1; }
if [ ! -f "$MEGATRON_LM_DIR/megatron/training/datasets/apertus_sft_dataset.py" ] && [ -d "$MEGATRON_LM_DIR" ]; then
    echo "ERROR: $MEGATRON_LM_DIR has no megatron/training/datasets/apertus_sft_dataset.py;"
    echo "       --ap-sft needs the SFT trainer branch (MEGATRON_REF=$MEGATRON_REF)"
    [ -n "${DRY_RUN:-}" ] || exit 1
fi

RUN_TAG=${RUN_TAG-megachonk}                                # RUN_TAG= drops it
SEQ_LEN=${SEQ_LEN:-16384}   # KISTI: 16k for now (Alps default 8192); warm-up and training must match
SEQ_PER_RANK=${SEQ_PER_RANK:-8192}   # tokens/rank/micro-batch, CP = SEQ_LEN/SEQ_PER_RANK; 8192 was Chonk's memory ceiling

# Model architecture follows the MegaChonk checkpoint; NUM_LAYERS=4k+1 (default 61) builds a shallower copy for debugging.
NUM_LAYERS=${NUM_LAYERS:-61}
[ $(( (NUM_LAYERS - 1) % 4 )) -eq 0 ] && [ "$NUM_LAYERS" -ge 5 ] \
    || { echo "ERROR: NUM_LAYERS must be 4k+1 (KDA 3:1 with a final global layer), got $NUM_LAYERS"; exit 1; }
EXP_NAME=${EXP_NAME:-sft/megachonk-l$NUM_LAYERS-seq$SEQ_LEN${RUN_TAG:+-$RUN_TAG}}
CKPT_DIR=${CKPT_DIR:-$WORK_DIR/results/ckpts/$EXP_NAME}
if [ "$REQUIRE_RESUME" = 1 ] && [ ! -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: --resume requires an SFT checkpoint in CKPT_DIR=$CKPT_DIR; check EXP_NAME/WORK_DIR or set CKPT_DIR"
    exit 1
fi
TENSORBOARD_DIR=$WORK_DIR/results/runs/tb-${SLURM_JOB_ID:-interactive}
WANDB_SAVE_DIR=$WORK_DIR/results/runs
NSYS_DIR=$WORK_DIR/results/nsys/${SLURM_JOB_ID:-interactive}
# hanint05's installed packages, read only (their install markers exist, so nothing is installed into them).
PACKAGE_DIR=${PACKAGE_DIR:-$HANINT05_APPSDATA/packages/ngc-25.12-kisti-chonk}   # wandb, omegaconf, transformers, emerging_optimizers
FLA_DIR=${FLA_DIR:-$HANINT05_APPSDATA/packages/fla-0.5.2}   # flash-linear-attention 0.5.2, ahead of the container's 0.4.0
JITCACHE_ROOT=${JITCACHE_ROOT:-$KISTI_SCRATCH/megachonk-sft/triton-cache}   # per-node Triton/inductor caches: triton/job-<id>/node-<n>

HIDDEN=8192
FFN_HIDDEN=14336              # dense-layer MLP (the first 3 layers)
NUM_HEADS=64                  # head_dim = 128
NUM_KV_HEADS=16               # GQA 4:1 on the global softmax layers
INIT_STD=0.0110485            # 1/sqrt(HIDDEN)
KDA_GROUPS=$(( (NUM_LAYERS - 1) / 4 ))

# Parallelism and batch sizes. Two virtual stages per pipeline rank: the EP all-to-all overlap
# (--overlap-moe-expert-parallel-comm, MOE_OVERLAP=1) requires an interleaved schedule (job 1322138), as hanint05's Chonk
# layout had (8 stages on PP4). Full depth: PP16 with the 32-stage layout listed in the pretraining launcher
# (megachonk.sh); a shallow copy: PP2 with 4 stages.
MBS=${MBS:-1}
GBS=${GBS:-256}
TP=${TP:-2}                  # as in pretraining; splits the output layer and the embedding (see --sequence-parallel below)
EP=${EP:-16}
ETP=1
if [ -z "${PP_LAYOUT:-}" ]; then
    if [ "$NUM_LAYERS" -eq 61 ]; then PP=${PP:-16}; PP_LAYOUT="Et|(tt|)*15,t|(tt|)*14,tL"
    else PP=${PP:-2}; _q=$(( NUM_LAYERS / 4 )); PP_LAYOUT="Et*$_q|t*$_q|t*$_q|t*$(( NUM_LAYERS - 3 * _q ))L"; fi
else
    PP=${PP:?PP must be set with PP_LAYOUT}
fi
CP=$(( SEQ_LEN / SEQ_PER_RANK ))
CP_IMPL=${CP_IMPL:-kcp}

# SFT-specific Triton archive, one per depth, sequence length, parallel layout and micro-batch size.
CP_TAG=cp$CP; [ "$CP" -eq 1 ] || CP_TAG+=-$CP_IMPL
TRITON_TAR=${TRITON_TAR:-$JITCACHE_ROOT/warm/sft-megachonk-l$NUM_LAYERS-seq$SEQ_LEN-tp$TP-pp$PP-ep$EP-etp$ETP-$CP_TAG-mbs$MBS.tar}
MIN_TRITON_ARCHIVE_KEYS=${MIN_TRITON_ARCHIVE_KEYS:-300}     # sanity floor; full archives hold ~420-660 keys

# Checkpoint frequency and timed exit.
SAVE_INTERVAL=${SAVE_INTERVAL:-220}  # user, 2026-11-21: every 220 steps, only the newest checkpoint kept (prune_old_checkpoints)
EXIT_DURATION_MINS=${EXIT_DURATION_MINS:-705}    # exit cleanly 15 min before the 12 h limit
DIST_TIMEOUT_MIN=${DIST_TIMEOUT_MIN:-60}  # 32 virtual stages compile one after another on a cold cache (job 1322902)         # rank 0 must survive the one-time dataset index build at the barrier
NUM_WORKERS=${NUM_WORKERS:-1}                    # data loader workers per rank (KISTI; Alps uses 1)

# Untuned scalar, matrix (Muon), and gains learning rates.
LR=${LR:-6.16e-4}
MLR=${MLR:-3.08e-4}  # 10x below hanint05's Chonk v0.2 default: lower matrix LR was better on Chonk (user, 2026-11-21)
GAINS_LR=${GAINS_LR:-6.16e-4}
MIN_LR=${MIN_LR:-6.16e-5}
# Fractional warm-up adapts to the derived training length.
LR_WARMUP_FRACTION=${LR_WARMUP_FRACTION:-0.03}

# EPOCHS counts full dataset passes; Megatron derives train_iters after packing.
EPOCHS=${EPOCHS:-2}
SEED=${SEED:-42}                                 # shuffles conversations before packing and the packs every epoch
PACK=${PACK:-1}
PACKING_STRATEGY=${PACKING_STRATEGY:-bfd}
MAX_DOCS_PER_BIN=${MAX_DOCS_PER_BIN:-64}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}           # larger than any run: evaluation never fires, but it must be a number
# TOTAL_ITERS is the logged SFT iteration total; it only controls chaining.
AUTO_REQUEUE=${AUTO_REQUEUE:-0}
TOTAL_ITERS=${TOTAL_ITERS:-}
MAX_REQUEUES=${MAX_REQUEUES:-20}
REQUEUE_COUNT=${REQUEUE_COUNT:-0}
if [ "$WARMUP" = 0 ] && [ "$AUTO_REQUEUE" = 1 ] && [[ ! "$TOTAL_ITERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: AUTO_REQUEUE=1 needs TOTAL_ITERS=<N> from sft-preflight.py or the 'Apertus SFT: ... ; <N> iterations' log line"
    exit 1
fi
HYPERSPHERE_PRESERVE_INIT=${HYPERSPHERE_PRESERVE_INIT:-1}
MUON_LOG=${MUON_LOG:-0}                          # 1 adds Muon diagnostics, which currently hit a TorchDynamo recompile limit
MUON_LOG_INTERVAL=${MUON_LOG_INTERVAL:-5}
# Performance/debug switches; changing them may need a new warm-up.
CUDA_GRAPHS=${CUDA_GRAPHS:-0}                    # MegaChonk: 0, as marin_exact routing (pretraining's) cannot be captured (job 1322113)
MOE_DISPATCHER=${MOE_DISPATCHER:-flex}           # flex (+deepep) | alltoall
MOE_OVERLAP=${MOE_OVERLAP:-0}  # MegaChonk: off (full recompute requires it off; job 1325531 passed this way)
DELAY_WGRAD=${DELAY_WGRAD:-0}  # requires MOE_OVERLAP=1                    # disable when debugging backward corruption
CKPT_STAGING=${CKPT_STAGING:-streamed}           # pinned | streamed
MOE_FP8_DISPATCH=${MOE_FP8_DISPATCH:-0}  # requires MOE_OVERLAP=1
MOE_OFFLOAD=${MOE_OFFLOAD:-0}  # MegaChonk: off. The imported checkpoint has the standard per-expert layout (offloading uses fused weight1/weight2); off was also faster (job 1325853: 415 tokens/s/GPU, 45 GB)
MOE_OFFLOAD_MAIN_GRAD=${MOE_OFFLOAD_MAIN_GRAD:-0}   # needs device-aware scale_gradients for SFT loss scaling
RECOMPUTE_MODULES=${RECOMPUTE_MODULES:-"layernorm linear_attn qkv shared_experts"}
# RECOMPUTE=full recomputes every layer in backward (less activation memory, slower); default: selective modules.
RECOMPUTE=${RECOMPUTE:-full}  # MegaChonk: selective recompute ran out of GPU memory on middle stages (job 1323486)
if [ "$RECOMPUTE" = full ]; then RECOMPUTE_ARGS=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)
else RECOMPUTE_ARGS=(--recompute-granularity selective --recompute-modules $RECOMPUTE_MODULES); fi

NSYS_PROFILER=${NSYS_PROFILER:-false}
NSYS_RANKS="64 128"
NSYS_START_ITER=50
NSYS_END_ITER=51

# Triton warm-up allocation.
WARMUP_NODES=${WARMUP_NODES:-8}
WARMUP_TIME=${WARMUP_TIME:-01:00:00}
WARMUP_ITERS=${WARMUP_ITERS:-6}            # iterations before --exit-interval stops the run
WARMUP_DATA_SHARDS=${WARMUP_DATA_SHARDS:-8}

# Warm-up uses a separate job name and singleton slot.
if [ -z "${DRY_RUN:-}" ]; then
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        [ "$WARMUP" = 1 ] || { echo "ERROR: training is submitted with 'sbatch $(basename "$SELF")' (DRY_RUN=1 prints the argv instead)"; exit 1; }
        mkdir -p logs   # the #SBATCH --output path is relative to the submit directory, as for 'sbatch $(basename "$SELF")'
        WARMUP_JOBID=$(sbatch --parsable --nodes="$WARMUP_NODES" --time="$WARMUP_TIME" --job-name="$JOB_NAME-warmup" "$SELF" --warmup)
        echo "warm-up: submitted job $WARMUP_JOBID ($WARMUP_NODES nodes, $WARMUP_TIME, job name $JOB_NAME-warmup, log logs/$JOB_NAME-warmup-$WARMUP_JOBID.out)"
        exit 0
    elif [ "$WARMUP" = 1 ]; then
        [ "${SLURM_JOB_NUM_NODES:-0}" -le "$WARMUP_NODES" ] \
            || { echo "ERROR: --warmup on $SLURM_JOB_NUM_NODES nodes; a warm-up pass needs WARMUP_NODES=$WARMUP_NODES. Submit it with 'bash $(basename "$SELF") --warmup'."; exit 1; }
        [ "${SLURM_JOB_NAME:-}" != "$JOB_NAME" ] \
            || { echo "ERROR: --warmup under the training job name $JOB_NAME would take its singleton slot. Submit it with 'bash $(basename "$SELF") --warmup'."; exit 1; }
    fi
fi

case "$MUON_LOG" in 0|1) ;; *) echo "ERROR: MUON_LOG must be 0 or 1 (got '$MUON_LOG')"; exit 1 ;; esac
case "$CUDA_GRAPHS" in 0|1) ;; *) echo "ERROR: CUDA_GRAPHS must be 0 or 1 (got '$CUDA_GRAPHS')"; exit 1 ;; esac
case "$CP_IMPL" in kcp|a2a) ;; *) echo "ERROR: CP_IMPL must be kcp or a2a (got '$CP_IMPL')"; exit 1 ;; esac
case "$CKPT_STAGING" in pinned|streamed) ;; *) echo "ERROR: CKPT_STAGING must be pinned or streamed (got '$CKPT_STAGING')"; exit 1 ;; esac
for _k in MOE_OVERLAP MOE_FP8_DISPATCH MOE_OFFLOAD MOE_OFFLOAD_MAIN_GRAD DELAY_WGRAD HYPERSPHERE_PRESERVE_INIT; do
    case "${!_k}" in 0|1) ;; *) echo "ERROR: $_k must be 0 or 1 (got '${!_k}')"; exit 1 ;; esac
done; unset _k
# Validate context-parallel layout and support.
[ $(( SEQ_LEN % SEQ_PER_RANK )) -eq 0 ] \
    || { echo "ERROR: SEQ_LEN ($SEQ_LEN) must be a multiple of SEQ_PER_RANK ($SEQ_PER_RANK)"; exit 1; }
if [ "$CP" -gt 1 ]; then
    [ $(( SEQ_LEN % (2 * CP) )) -eq 0 ] \
        || { echo "ERROR: SEQ_LEN ($SEQ_LEN) must be a multiple of 2*CP ($(( 2 * CP ))); Megatron asserts this"; exit 1; }
    if [ "$CP_IMPL" = a2a ] && [ $(( NUM_HEADS % CP )) -ne 0 ]; then
        echo "ERROR: CP_IMPL=a2a needs $NUM_HEADS % CP == 0 (CP=$CP); use CP_IMPL=kcp"; exit 1
    fi
    if [ -f "$MEGATRON_LM_DIR/megatron/core/transformer/transformer_config.py" ]; then
        grep -q "linear_attention_cp_impl" "$MEGATRON_LM_DIR/megatron/core/transformer/transformer_config.py" \
            || { echo "ERROR: SEQ_LEN=$SEQ_LEN needs CP=$CP, but $MEGATRON_LM_DIR has no linear_attention_cp_impl."; \
                 echo "       Update $MEGATRON_LM_DIR to the current feat/sft_trainer_ap2, or run at SEQ_LEN=$SEQ_PER_RANK (CP=1)."; exit 1; }
    elif [ -z "${DRY_RUN:-}" ]; then
        echo "ERROR: cannot verify CP support: $MEGATRON_LM_DIR/megatron/core/transformer/transformer_config.py is absent"; exit 1
    fi
fi

# Both data and expert parallel groups must divide the GPU count.
GPUS=${SLURM_NTASKS:-$(( ${SLURM_NNODES:-0} * 4 ))}
_gcd () { local a=$1 b=$2 t; while [ "$b" -ne 0 ]; do t=$b; b=$(( a % b )); a=$t; done; echo "$a"; }
_p1=$(( TP * PP * CP )); _p2=$(( ETP * EP * PP ))
NODE_STEP=$(( (_p1 / $(_gcd "$_p1" "$_p2") * _p2 + 3) / 4 ))
if [ "$GPUS" -gt 0 ]; then
    DP=$(( GPUS / _p1 ))
    [ $(( GPUS % _p1 )) -eq 0 ] && [ $(( GPUS % _p2 )) -eq 0 ] \
        || { echo "ERROR: $(( GPUS / 4 )) nodes does not divide into TP$TP/PP$PP/CP$CP/EP$EP; use a multiple of $NODE_STEP nodes"; exit 1; }
    [ $(( GBS % (MBS * DP) )) -eq 0 ] \
        || { echo "ERROR: GBS ($GBS) must be divisible by MBS ($MBS) x DP ($DP) = $(( MBS * DP )); set GBS= or MBS="; exit 1; }
    # The interleaved pipeline schedule needs at least PP micro-batches per rank.
    [ $(( GBS / (MBS * DP) )) -ge "$PP" ] \
        || { echo "ERROR: $(( GBS / (MBS * DP) )) micro-batches/rank < PP=$PP; use fewer nodes or a larger GBS"; exit 1; }
    echo "layout     $(( GPUS / 4 )) nodes = $GPUS GPUs | TP$TP PP$PP EP$EP CP$CP -> DP=$DP, expert-DP=$(( GPUS / _p2 ))"
    echo "sequence   seq_len=$SEQ_LEN MBS=$MBS GBS=$GBS -> $(( MBS * SEQ_PER_RANK )) tokens/rank/microbatch, $(( GBS / (MBS * DP) )) microbatches/rank"
fi

echo "START TIME: $(date)"

# Environment shared by all tasks.
# Tests: TEST_CUDA_DEVICE_MAX_CONNECTIONS overrides the hardware queue count (default 1, as in sft-chonk.sh).
export CUDA_DEVICE_MAX_CONNECTIONS=${TEST_CUDA_DEVICE_MAX_CONNECTIONS:-1}
echo "CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CPP_LOG_LEVEL=WARNING
export OMP_NUM_THREADS=18
export NUMEXPR_NUM_THREADS=18
export NUMEXPR_MAX_THREADS=512
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/tmp/$USER-${SLURM_JOB_ID:-$$}
export NCCL_GRAPH_REGISTER=0              # required by the router CUDA graphs
# Remove a host venv that would leak invalid paths into the container.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "env  dropping leaked VIRTUAL_ENV=$VIRTUAL_ENV from the job environment"
    PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "$VIRTUAL_ENV/bin" | paste -sd: -)
    export PATH; unset VIRTUAL_ENV
fi
# Prevent core dumps (a crashed rank writes ~60 GB). Slurm can leave the tasks' core limit unchanged (KISTI excludes
# CORE from PropagateResourceLimits), so IN_CONTAINER sets it again.
ulimit -c 0
# UCCL/DeepEP tuning for EP16 and FP8 dispatch.
export UCCL_EP_TRANSPORT=cxi
export UCCL_EP_DISPATCH_CONFIG=24,40,512,32,512
export UCCL_EP_COMBINE_CONFIG=24,7,512,32,512
export UCCL_EP_CPU_TIMEOUT_SECS=6000
# KISTI: [env] of the Alps EDF, forwarded into the container by pyxis.
export PMIX_MCA_psec=native
export NCCL_PXN_DISABLE=1
export NCCL_CROSS_NIC=0
export NCCL_PROTO=^LL128 FI_PROVIDER=cxi NCCL_IB_DISABLE=1   # KISTI: LL128 gives wrong inter-node all-reduce results
# KISTI: Slingshot through the host libfabric (CXI) and the site aws-ofi-nccl plugin, on more than one node only
# (single-node jobs get no Slingshot VNI). LD_LIBRARY_PATH does not reach the container; it travels as KISTI_NET_LD_PATH.
CONTAINER_NET_MOUNTS=""
KISTI_NET_LD_PATH=""
if [ "${KISTI_HOST_NET:-$(( ${SLURM_NNODES:-2} > 1 ))}" = 1 ]; then
    LIBFABRIC_DIR=${LIBFABRIC_DIR:-/opt/cray/libfabric/2.3.1}
    OFI_NCCL_DIR=${OFI_NCCL_DIR:-/apps/library/aws-ofi-nccl/1.20.0/aarch64}
    CONTAINER_NET_MOUNTS=$LIBFABRIC_DIR:$LIBFABRIC_DIR,/usr/lib64/libcxi.so.1:/opt/hostlibs/libcxi.so.1,/usr/lib64/libjson-c.so.5:/opt/hostlibs/libjson-c.so.5
    KISTI_NET_LD_PATH=/opt/hostlibs:$LIBFABRIC_DIR/lib64:$OFI_NCCL_DIR/lib
    export NCCL_NET_PLUGIN=$OFI_NCCL_DIR/lib/libnccl-net-ofi.so NCCL_NET="AWS Libfabric"
    # As set by `module load aws-ofi-nccl` on HANGANG.
    export FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DISABLE_HOST_REGISTER=1 FI_CXI_DEFAULT_CQ_SIZE=131072 \
        FI_CXI_RDZV_PROTO=alt_read FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_GET_MIN=0 \
        FI_CXI_DEFAULT_TX_SIZE=2048 FI_CXI_RX_MATCH_MODE=hybrid NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 GLOO_SOCKET_IFNAME=hsn0
fi
# MegaChonk keeps A_log per channel (Chonk: per head).
KDA_LEVERS="KDA_USE_GATE_IN_KERNEL=1 KDA_FUSED_GATE=1 KDA_QK_L2NORM_IN_KERNEL=1 KDA_CONV1D_CUDA=1 KDA_FUSED_LOW_RANK_GATHER=1 KDA_ALOG_PER_CHANNEL=1"

if [ -z "${DRY_RUN:-}" ]; then
    # Check resolution separately: export would hide a failed command substitution.
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1 || true)
    [ -n "$MASTER_ADDR" ] || { echo "ERROR: could not resolve the first node of $SLURM_JOB_NODELIST"; exit 1; }
    # The node name resolves to the 1 Gbit management Ethernet (10.161.*); the rendezvous goes over Slingshot.
    if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
        getent hosts "$MASTER_ADDR-hsn0" > /dev/null \
            || { echo "ERROR: $MASTER_ADDR-hsn0 does not resolve; the rendezvous must use the Slingshot address"; exit 1; }
        MASTER_ADDR=$MASTER_ADDR-hsn0
        echo "rendezvous $MASTER_ADDR ($(getent hosts "$MASTER_ADDR" | awk '{print $1}'))"
    fi
else
    MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-25678}

# Megatron arguments.
TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --main-grads-dtype fp32
)

NETWORK_SIZE_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN
    --ffn-hidden-size $FFN_HIDDEN
    --num-attention-heads $NUM_HEADS
    --group-query-attention
    --num-query-groups $NUM_KV_HEADS
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --rotary-base 10000
    --make-vocab-size-divisible-by 128
    --normalization RMSNorm
    --qk-layernorm
    --untie-embeddings-and-output-weights
    --attention-backend auto
    --seq-length $SEQ_LEN
    --scale-embeddings-by-sqrt-hidden
    --residual-output-scaling           # x = x + f(x)/sqrt(2*num_layers), the single depth-scaling mechanism
    # Hybrid attention: 3 KDA layers per global layer; final layer global.
    --experimental-attention-variant kda
    --linear-attention-safe-output-gate # bounded decay gate (needed for FlashKDA inference)
    --attention-output-gate             # gates only the global softmax layers
    --linear-attention-freq "([1,1,1,0]*$KDA_GROUPS+[0])"   # 3 KDA : 1 global, last layer global
    --no-rope-freq 1
    --linear-conv-kernel-dim 4
    --linear-key-head-dim 128
    --linear-value-head-dim 128
    --linear-num-key-heads 64           # HIDDEN/128
    --linear-num-value-heads 64
    --sssglu
    --sandwich-norm
    --embedding-norm                    # RMSNorm with a learnable gain on the embedding output (pretraining from 90,000)
    --enable-experimental
)

MOE_ARGS=(
    --num-experts 256
    --moe-router-topk 8
    --moe-ffn-hidden-size 3072
    --moe-shared-expert-intermediate-size 6144
    --moe-latent-size 4096
    --moe-layer-freq "([0]*3+[1]*$(( NUM_LAYERS - 3 )))"   # first 3 layers dense
    --expert-tensor-parallel-size 1     # experts sharded over EP only
    --pipeline-model-parallel-layout "$PP_LAYOUT"
    # Quantile balancing without seq_aux_loss.
    --moe-router-score-function sigmoid
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
    --moe-router-load-balancing-type quantile_balancing
    --moe-router-quantile-balancing-method marin_exact   # as in pretraining
    --moe-router-quantile-balancing-marin-num-bins 10000
    --moe-router-quantile-balancing-marin-exact-store cuda   # margin buffer on the GPU, not pinned host memory
    --moe-aux-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-per-layer-logging
)

# MoE execution switches for performance tuning and debugging.
case "$MOE_DISPATCHER" in
    flex)     MOE_ARGS+=(--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep) ;;
    alltoall) MOE_ARGS+=(--moe-token-dispatcher-type alltoall) ;;
    *)        echo "ERROR: MOE_DISPATCHER must be flex or alltoall (got '$MOE_DISPATCHER')"; exit 1 ;;
esac
if [ "$MOE_OVERLAP" = 1 ]; then
    MOE_ARGS+=(--overlap-moe-expert-parallel-comm)
fi
if [ "$DELAY_WGRAD" = 1 ]; then
    MOE_ARGS+=(--delay-wgrad-compute)
fi
if [ "$MOE_FP8_DISPATCH" = 1 ]; then
    MOE_ARGS+=(--moe-use-fp8-dispatch)
fi
if [ "$MOE_OFFLOAD" = 1 ]; then
    MOE_ARGS+=(
        --moe-use-offloading-experts
        --moe-offloading-num-chunks 8
        --moe-offloading-num-stages 2
        --moe-use-inplace-fp8-param
        --moe-use-extra-fp8-param-storage
        --moe-offloading-mode coarse-grained
    )
    [ "$MOE_OFFLOAD_MAIN_GRAD" = 0 ] || MOE_ARGS+=(--moe-offload-main-grad)
    # MoE activations kept on the host (moe_input moe_fc1_output); MOE_OFFLOAD_ACTIVATIONS=none keeps them on the GPU.
    [ "${MOE_OFFLOAD_ACTIVATIONS:-none}" = none ] || MOE_ARGS+=(--moe-offload-activations $MOE_OFFLOAD_ACTIVATIONS)  # MegaChonk: none (host copies grew to 170 GB per rank, jobs 1324968/1325137)
fi

if [ "$CUDA_GRAPHS" = 1 ]; then
    MOE_ARGS+=(
        --cuda-graph-impl transformer_engine
        --cuda-graph-scope ${CUDA_GRAPH_SCOPE:-moe_router moe_preprocess}
        # Warm up before graph capture so replay retains the MoE padding mask.
        --cuda-graph-warmup-steps ${CUDA_GRAPH_WARMUP_STEPS:-1}
    )
fi

SFT_ARGS=(
    --ap-sft
    --calculate-per-token-loss          # required by --ap-sft: loss = sum(w * CE) / count(w > 0)
    --ap-sft-report-assistant-loss
)
# Warm-up uses a short sample budget instead of epochs.
[ "$WARMUP" = 1 ] || SFT_ARGS+=(--ap-sft-epochs $EPOCHS)
if [ "$PACK" = 1 ]; then
    SFT_ARGS+=(--ap-sft-pack-samples --ap-sft-packing-strategy "$PACKING_STRATEGY" --max-docs-per-bin-sft $MAX_DOCS_PER_BIN)
fi

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --exit-duration-in-mins $EXIT_DURATION_MINS
    --log-interval 1
    --eval-iters 0
    --optimizer md_decoupling
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 100
    --log-memory-interval 20
    --log-device-memory-used
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --no-create-attention-mask-in-dataloader
    --distributed-timeout-minutes $DIST_TIMEOUT_MIN
    # Rerun steps whose gradient norm exceeds the threshold.
    --no-check-for-nan-in-loss-and-grad
    --check-grad-norm
    --check-grad-norm-threshold 5.0
    "${RECOMPUTE_ARGS[@]}"
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

# MD/Muon requires the layer-wise optimizer; keep overlap-param-gather off.
OPTIMIZER_ARGS=(
    --matrix-lr $MLR
    --gains-lr $GAINS_LR
    --min-lr-mode absolute              # every group floors at --min-lr
    --use-orthogonal-updates
    --muon-coefficient-type polar_express
    --muon-scale-mode shape_up
    --muon-router-scale-mode none
    --muon-momentum 0.95
    --muon-use-nesterov
    --hypersphere-mode row
    --hypersphere-embedding-mode row
    --hypersphere-router-mode row
    --hypersphere-radius-mode fan_in
    --md-router-use-orthogonal-updates True
    --hypersphere-gains-mode rowcol
    --hypersphere-gains-mode-router rowcol
    --hypersphere-gains-mode-embedding none
    --gain-parametrization softplus
    --use-layer-wise-distributed-optimizer
    --overlap-grad-reduce
)
# Fresh gains preserve loaded magnitudes; resume restores saved gains.
if [ "$HYPERSPHERE_PRESERVE_INIT" = 1 ]; then
    OPTIMIZER_ARGS+=(--hypersphere-preserve-init)
fi
if [ "$MUON_LOG" = 1 ]; then
    OPTIMIZER_ARGS+=(
        --log-muon-gains
        --log-muon-sparsity
        --log-muon-per-layer
        --muon-sparsity-thresholds 1e-8 1e-12
        --muon-log-interval $MUON_LOG_INTERVAL
    )
fi

LEARNING_RATE_ARGS=(
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style cosine
    --lr-warmup-fraction $LR_WARMUP_FRACTION
    # Unset decay length follows the derived train_iters.
)

INITIALIZATION_ARGS=(
    --seed $SEED
    --init-method-std $INIT_STD
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --context-parallel-size $CP
    --expert-model-parallel-size $EP
)
if [ "$CP" -gt 1 ]; then
    DISTRIBUTED_ARGS+=(--linear-attention-cp-impl "$CP_IMPL")
fi
# Tensor parallelism (pretraining: TP2) needs sequence parallelism with MoE. It also splits the output layer: with
# TP1 the layer-wise distributed optimizer gives the whole 200k x 8192 output matrix to one rank, whose fp32 copies
# for the update ran out of memory (rank 23, jobs 1322381 and 1322467).
if [ "$TP" -gt 1 ]; then
    DISTRIBUTED_ARGS+=(--sequence-parallel)
fi

LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --log-params-norm
    --log-memory-to-tensorboard
    --tensorboard-dir "$TENSORBOARD_DIR"
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
)

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
)

# Read matching data prefixes in stable order; use an argument file above 256 prefixes.
mapfile -t DATA_PATH_LIST < <(find -L "$SFT_DATA_ROOT" -maxdepth "$SFT_DATA_DEPTH" -type f -name "${SFT_DATA_GLOB}.bin" | sed -E 's/\.bin$//' | LC_ALL=C sort -u)
[ ${#DATA_PATH_LIST[@]} -gt 0 ] || { echo "ERROR: no ${SFT_DATA_GLOB}.bin under $SFT_DATA_ROOT"; exit 1; }
# Match the loader's loss-weight lookup.
BAD_WEIGHTS=(); MISSING_WEIGHTS=()
for _p in "${DATA_PATH_LIST[@]}"; do
    [ -f "$_p.idx" ] || { echo "ERROR: $_p.bin has no matching $_p.idx"; exit 1; }
    _c=("$_p.loss_weights" "$_p.loss_mask")
    [ "$(basename "$_p")" = tokens ] && _c+=("$(dirname "$_p")/loss_weights" "$(dirname "$_p")/loss_mask")
    _h=()
    for _x in "${_c[@]}"; do [ -f "$_x.idx" ] && _h+=("$_x"); done
    case ${#_h[@]} in
        1) ;;
        0) MISSING_WEIGHTS+=("$_p") ;;
        *) BAD_WEIGHTS+=("$_p -> ${_h[*]}") ;;
    esac
done
unset _p _c _h _x
weights_msg() {
    if [ ${#BAD_WEIGHTS[@]} -gt 0 ]; then
        printf '%d prefix(es) have MORE than one loss_weights/loss_mask index, which the loader rejects: %s' \
            "${#BAD_WEIGHTS[@]}" "${BAD_WEIGHTS[0]}"
    else
        printf '%d of %d prefix(es) have no loss_weights/loss_mask index, e.g. %s' \
            "${#MISSING_WEIGHTS[@]}" "${#DATA_PATH_LIST[@]}" "${MISSING_WEIGHTS[0]:-?}"
    fi
}
[ ${#BAD_WEIGHTS[@]} -eq 0 ] || { echo "ERROR: $(weights_msg)"; exit 1; }
case "$LOSS_MASK" in
    auto|on)
        if [ ${#MISSING_WEIGHTS[@]} -eq 0 ]; then
            LOSS_MASK_USED="stored (--ap-sft-load-loss-mask)"; SFT_ARGS+=(--ap-sft-load-loss-mask)
        elif [ "$LOSS_MASK" = on ]; then
            echo "ERROR: LOSS_MASK=on but $(weights_msg); set LOSS_MASK=auto or export the weights"; exit 1
        else
            LOSS_MASK_USED="derived from Apertus-2 controls ($(weights_msg))"
        fi ;;
    off) LOSS_MASK_USED="derived from Apertus-2 controls (LOSS_MASK=off)" ;;
    *)   echo "ERROR: LOSS_MASK must be auto, on or off (got '$LOSS_MASK')"; exit 1 ;;
esac
if [ "$WARMUP" = 1 ] && [ ${#DATA_PATH_LIST[@]} -gt "$WARMUP_DATA_SHARDS" ]; then
    DATA_PATH_LIST=("${DATA_PATH_LIST[@]:0:$WARMUP_DATA_SHARDS}")
fi
if [ ${#DATA_PATH_LIST[@]} -gt 256 ]; then
    DATA_ARGS_FILE=$DATASET_CACHE_DIR/sft_data_paths-${SLURM_JOB_ID:-local}.txt
    [ -n "${DRY_RUN:-}" ] && DATA_ARGS_FILE=$TMPDIR/sft_data_paths-dry.txt
    mkdir -p "$(dirname "$DATA_ARGS_FILE")"
    printf '%s\n' "${DATA_PATH_LIST[@]}" > "$DATA_ARGS_FILE"
    DATA_ARGS=(--data-args-path "$DATA_ARGS_FILE" --split 100,0,0)
else
    DATA_ARGS=(--train-data-path "${DATA_PATH_LIST[@]}")
fi
DATA_ARGS+=(
    --data-cache-path "$DATASET_CACHE_DIR"
    --num-workers $NUM_WORKERS
)

NSYS_SELECT=""
if [ "$WARMUP" = 1 ]; then
    # Warm-up uses random weights and never loads/saves checkpoints.
    # KISTI: at least WARMUP_ITERS steps of samples, so WARMUP_ITERS > 10 reaches later-iteration kernels (as kisti-docs chonk.sh).
    WARMUP_STEPS=$(( WARMUP_ITERS > 10 ? WARMUP_ITERS : 10 ))
    RUN_ARGS=(
        --train-samples $(( WARMUP_STEPS * GBS ))
        --eval-interval $EVAL_INTERVAL
        --exit-interval $WARMUP_ITERS
    )
    export WANDB_MODE=disabled
else
    RUN_ARGS=(
        --eval-interval $EVAL_INTERVAL
        --load "$CKPT_DIR"
        --exit-on-missing-checkpoint
        --ckpt-format torch_dist
        --ckpt-staging $CKPT_STAGING    # streamed suits the ~1 TB load
    )
    # Fully parallel load exchanges loaded shards over NCCL with extra GPU buffers: rank 0 (embedding stage) ran out of
    # GPU memory loading the 96,000 import with it (job 1325717). CKPT_FULLY_PARALLEL_LOAD=1 turns it back on.
    [ "${CKPT_FULLY_PARALLEL_LOAD:-0}" = 0 ] || RUN_ARGS+=(--ckpt-fully-parallel-load)
    # SAVE=0 (short debug runs on real weights): no --save, so no ~5.5 TB checkpoint at the exit interval.
    [ "${SAVE:-1}" = 0 ] || RUN_ARGS+=(--save "$CKPT_DIR" --save-interval $SAVE_INTERVAL)
    # Fresh SFT uses base weights; resume restores SFT optimizer state and progress.
    if [ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
        RESUME_FROM="$CKPT_DIR ($(tr -dc '0-9' < "$CKPT_DIR/latest_checkpointed_iteration.txt"))"
    else
        [ -f "$BASE_CKPT_DIR/latest_checkpointed_iteration.txt" ] \
            || { echo "ERROR: no Megatron checkpoint under BASE_CKPT_DIR=$BASE_CKPT_DIR"; exit 1; }
        RESUME_FROM="base: $BASE_CKPT_DIR (${BASE_CKPT_ITER:-$(tr -dc '0-9' < "$BASE_CKPT_DIR/latest_checkpointed_iteration.txt")})"
        RUN_ARGS+=(--pretrained-checkpoint "$BASE_CKPT_DIR" --finetune)
        if [ -n "$BASE_CKPT_ITER" ]; then RUN_ARGS+=(--ckpt-step "$BASE_CKPT_ITER"); fi
    fi
    if [ "$NSYS_PROFILER" = true ]; then
        RUN_ARGS+=(
            --profile
            --profile-step-start $NSYS_START_ITER
            --profile-step-end $NSYS_END_ITER
            --profile-ranks $NSYS_RANKS       # unquoted on purpose: two tokens, "64 128"
        )
        NSYS_SELECT="NSYS=''; case \" $NSYS_RANKS \" in *\" \$SLURM_PROCID \"*) NSYS='nsys profile -s none --trace=nvtx,cudnn,cublas,cuda --output=$NSYS_DIR/nsys-${EXP_NAME##*/}-rank'\$SLURM_PROCID' --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop';; esac;"
    fi
    if [ -n "${WANDB_API_KEY:-}" ]; then
        export WANDB_PROJECT
        export WANDB_NAME=$EXP_NAME-${SLURM_JOB_ID:-interactive}
        RUN_ARGS+=(
            --wandb-project "$WANDB_PROJECT"
            --wandb-exp-name "$EXP_NAME-${SLURM_JOB_ID:-interactive}"
            --wandb-save-dir "$WANDB_SAVE_DIR"
        )
        if [ -n "$WANDB_ENTITY" ]; then
            export WANDB_ENTITY
            RUN_ARGS+=(--wandb-entity "$WANDB_ENTITY")
        fi
    else
        export WANDB_MODE=disabled
    fi
fi

MEGATRON_ARGS=(
    "${TRANSFORMER_ENGINE_ARGS[@]}"
    "${NETWORK_SIZE_ARGS[@]}"
    "${MOE_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${REGULARIZATION_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${LEARNING_RATE_ARGS[@]}"
    "${INITIALIZATION_ARGS[@]}"
    "${MIXED_PRECISION_ARGS[@]}"
    "${DISTRIBUTED_ARGS[@]}"
    "${LOGGING_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${RUN_ARGS[@]}"
    ${MEGATRON_EXTRA_ARGS:-}                     # extra Megatron flags, unquoted (word-split) on purpose
)

# Local rank 0 seeds the Triton cache; other ranks wait.
triton_warm_seed() {
    local dest=$1 tar_path=$2 stamp=$1/.triton_warm_done staging i
    [ -f "$tar_path" ] || { echo "[triton-warm] no archive at $tar_path -- cold start (will JIT)" >&2; mkdir -p "$dest"; return 0; }
    if [ "${SLURM_LOCALID:-0}" = 0 ]; then
        if [ ! -f "$stamp" ]; then
            staging=$dest.staging.$$
            rm -rf "$staging"; mkdir -p "$staging"
            if tar -xf "$tar_path" -C "$staging" 2>/dev/null; then
                : > "$staging/.triton_warm_done"
                # Publish the completion stamp only after all cache files are available.
                if ! mv -T "$staging" "$dest" 2>/dev/null; then
                    rm -f "$staging/.triton_warm_done"; mkdir -p "$dest"
                    cp -a "$staging/." "$dest/" 2>/dev/null; rm -rf "$staging"; : > "$stamp"
                fi
                echo "[triton-warm] seeded $dest from $(basename "$tar_path") ($(ls -1 "$dest" | wc -l) entries)" >&2
            else
                echo "[triton-warm] FAILED to extract $tar_path -- cold start (will JIT)" >&2
                rm -rf "$staging"; mkdir -p "$dest"
            fi
        fi
    else
        for i in $(seq 1 900); do [ -f "$stamp" ] && break; sleep 1; done
        [ -f "$stamp" ] || echo "[triton-warm] localid=$SLURM_LOCALID timed out waiting for the seed -- proceeding" >&2
    fi
    mkdir -p "$dest"
}

# Install missing container packages once per allocation, under a lock.
install_python_deps() {
    local pkg=$1 marker=$1/.deps_installed_moe_kisti_v1 lock=$1/.install.lockdir install="pip install --quiet" waited=0
    mkdir -p "$pkg"
    [ -f "$marker" ] && return 0
    while ! mkdir "$lock" 2>/dev/null; do
        [ -f "$marker" ] && return 0
        [ "$waited" -lt 1800 ] || { echo "install_python_deps: $lock held for 30 min without $marker appearing (stale lock from a killed job? rmdir it)" >&2; return 1; }
        sleep 5; waited=$((waited + 5))
    done
    [ -f "$marker" ] && { rmdir "$lock"; return 0; }
    command -v uv >/dev/null 2>&1 && install="uv pip install --quiet"
    { $install --target="$pkg" wandb omegaconf transformers \
        && { python3 -c "import emerging_optimizers" 2>/dev/null \
             || $install --no-deps --target="$pkg" 'git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0'; } \
        && touch "$marker"; } || { rmdir "$lock"; return 1; }
    rmdir "$lock"
}

# Archive the union of old and newly compiled kernels.
triton_warm_build() {
    local src=$1 out=$2 seed=${3:-} merge map d k path n_new=0
    merge=$(mktemp -d "$TMPDIR/triton-warm-merge.XXXXXX") || return 1
    map=$(mktemp) || { rm -rf "$merge"; return 1; }
    for d in "$src"/node-*; do
        [ -d "$d" ] || continue
        for path in "$d"/*; do
            [ -e "$path" ] || continue
            printf '%s\t%s\n' "${path##*/}" "$path"
        done
    done | sort -u -k1,1 > "$map"
    while IFS=$'\t' read -r k path; do
        [ -n "$k" ] && [ ! -e "$merge/$k" ] || continue
        cp -a "$path" "$merge/$k" 2>/dev/null && n_new=$((n_new + 1))
    done < "$map"
    rm -f "$map"
    [ "$n_new" -gt 0 ] || { echo "triton_warm_build: nothing collected from $src" >&2; rm -rf "$merge"; return 1; }
    if [ -n "$seed" ] && [ -f "$seed" ]; then
        tar -xf "$seed" -C "$merge" --skip-old-files 2>/dev/null || echo "triton_warm_build: could not merge the previous archive $seed" >&2
    fi
    if ! mkdir -p "$(dirname "$out")" \
        || ! tar -cf "$out.tmp" -C "$merge" . \
        || ! mv "$out.tmp" "$out"; then
        rm -rf "$merge"
        rm -f "$out.tmp"
        return 1
    fi
    rm -rf "$merge"
}
archive_keys() { tar -tf "$1" | awk -F/ 'NF>1 && $2!=""{print $2}' | sort -u | wc -l; }

# ===== BEGIN WORKAROUND: libfabric CXI strtok race -- KEEP UNTIL THE MOUNTED LIBFABRIC IS FIXED =====================
# libfabric's CXI provider (prov/cxi/src/cxip_nic.c, cxip_gen_auth_key_ss_env_get_svc_id) finds a NIC's Slingshot
# service ID with strtok over SLINGSHOT_DEVICES and SLINGSHOT_SVC_IDS. strtok is not thread-safe: when UCCL-EP's proxy
# threads open their domains together, one can miss its NIC and the rank aborts with
# "fi_domain failed: Function not implemented" (5 of 12 chonk starts on HANGANG, 2026-10-29).
# Listing this rank's NIC (cxi<local rank>, the one UCCL-EP uses) first ends the lookup at the first token.
# Affected: HANGANG's /opt/cray/libfabric/2.3.1 (LIBFABRIC_DIR) and upstream libfabric up to v2.7.0 / main.
# Remove only when the libfabric mounted into the container uses strtok_r there (or UCCL-EP retries -FI_ENOSYS).
own_nic_first() {
    local own=cxi${SLURM_LOCALID:-0} devs svcs i n d s
    [ -n "${SLINGSHOT_DEVICES:-}" ] && [ -n "${SLINGSHOT_SVC_IDS:-}" ] || return 0
    IFS=, read -ra devs <<< "$SLINGSHOT_DEVICES"
    IFS=, read -ra svcs <<< "$SLINGSHOT_SVC_IDS"
    [ ${#devs[@]} -eq ${#svcs[@]} ] || return 0
    for i in "${!devs[@]}"; do [ "${devs[$i]}" = "$own" ] && break; done
    [ "${devs[$i]}" = "$own" ] || return 0
    d=$own; s=${svcs[$i]}
    for n in "${!devs[@]}"; do [ "$n" = "$i" ] || { d+=,${devs[$n]}; s+=,${svcs[$n]}; }; done
    export SLINGSHOT_DEVICES=$d SLINGSHOT_SVC_IDS=$s
    [ "${SLURM_PROCID:-0}" -ge 4 ] || echo "own_nic_first: rank ${SLURM_PROCID:-0} SLINGSHOT_DEVICES=$d SLINGSHOT_SVC_IDS=$s"
}
# ===== END WORKAROUND: libfabric CXI strtok race ===================================================================

NET_LD_EXPORT=""
[ -z "$KISTI_NET_LD_PATH" ] || NET_LD_EXPORT="export LD_LIBRARY_PATH=${KISTI_NET_LD_PATH@Q}:\${LD_LIBRARY_PATH:-}"
# KISTI: pyxis flags replace the Alps EDF; the site mounts /scratch, /appsdata, /home01 and /apps.
CONTAINER_ARGS=(
    --container-image="$CONTAINER_IMAGE"
    --container-workdir="$PWD"
    --container-writable
)
# Joined without a trailing comma, which pyxis rejects ("--container-mounts: invalid format", job 1320441).
_mounts=()
[ -z "$CONTAINER_NET_MOUNTS" ] || _mounts+=("$CONTAINER_NET_MOUNTS")
[ -z "${EXTRA_CONTAINER_MOUNTS:-}" ] || _mounts+=("$EXTRA_CONTAINER_MOUNTS")
[ -z "${JIT_LOCAL:-}" ] || _mounts+=("$JITCACHE_ROOT:$JITCACHE_ROOT")
[ ${#_mounts[@]} -eq 0 ] || CONTAINER_ARGS+=(--container-mounts="$(IFS=,; echo "${_mounts[*]}")")
unset _mounts
# Tests: EXTRA_CONTAINER_MOUNTS adds src:dst mounts, e.g. a patched package over the installed one.
[ -z "${EXTRA_CONTAINER_MOUNTS:-}" ] || echo "EXTRA MOUNTS: $EXTRA_CONTAINER_MOUNTS"

# Expand host paths now; escaped SLURM/TMPDIR variables resolve inside each task.
IN_CONTAINER="$(declare -f triton_warm_seed install_python_deps own_nic_first)
ulimit -c 0
# WORKAROUND libfabric CXI strtok race (see own_nic_first): keep until the mounted libfabric is fixed.
own_nic_first
export MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT
export RANK=\$SLURM_PROCID WORLD_SIZE=\$SLURM_NTASKS LOCAL_RANK=\$SLURM_LOCALID LOCAL_WORLD_SIZE=$(( ${SLURM_NTASKS:-4} / ${SLURM_NNODES:-1} ))
export PYTHONPATH=${FLA_DIR@Q}:${MEGATRON_LM_DIR@Q}:${PACKAGE_DIR@Q}
$NET_LD_EXPORT
export $KDA_LEVERS
export TRITON_CACHE_DIR=${JITCACHE_ROOT@Q}/triton/job-${SLURM_JOB_ID:-interactive}/node-\$SLURM_NODEID
export TORCHINDUCTOR_CACHE_DIR=${JITCACHE_ROOT@Q}/inductor/job-${SLURM_JOB_ID:-interactive}/node-\$SLURM_NODEID
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
mkdir -p \"\$TORCHINDUCTOR_CACHE_DIR\" \"\$TMPDIR\"
triton_warm_seed \"\$TRITON_CACHE_DIR\" ${TRITON_TAR@Q}
cd ${MEGATRON_LM_DIR@Q}
${NSYS_SELECT}
install_python_deps ${PACKAGE_DIR@Q} && \$NSYS python3 pretrain_gpt.py ${MEGATRON_ARGS[*]@Q}"

if [ -n "${DRY_RUN:-}" ]; then
    echo "DATA:      ${#DATA_PATH_LIST[@]} prefix(es) under $SFT_DATA_ROOT (glob ${SFT_DATA_GLOB})"
    echo "LOSSMASK:  $LOSS_MASK_USED"
    echo "LOAD:      ${RESUME_FROM:-<warm-up: random init>}"
    echo "=== Megatron argv (DRY_RUN) ==="; printf '%s\n' "${MEGATRON_ARGS[@]}"
    echo "=== in-container command (DRY_RUN) ==="; echo "$IN_CONTAINER" | sed -n '/^export MASTER_ADDR/,$p'
    if [ "$WARMUP" = 1 ]; then
        echo "=== warm-up submission (DRY_RUN) ==="
        echo "sbatch --parsable --nodes=$WARMUP_NODES --time=$WARMUP_TIME --job-name=$JOB_NAME-warmup $SELF --warmup"
    fi
    exit 0
fi

mkdir -p "$WORK_DIR/results/runs" "$DATASET_CACHE_DIR" "$PACKAGE_DIR" "$JITCACHE_ROOT" "$TMPDIR"
[ "$WARMUP" = 1 ] || mkdir -p "$CKPT_DIR"
[ "$WARMUP" = 1 ] || [ "$NSYS_PROFILER" != true ] || mkdir -p "$NSYS_DIR"

# Require a warm archive unless cold starts are explicitly allowed.
if [ -f "$TRITON_TAR" ]; then
    SEED_KEYS=$(archive_keys "$TRITON_TAR") || { echo "ERROR: cannot list $TRITON_TAR (corrupt archive? move it away and run the warm-up again)"; exit 1; }
    echo "TRITON ARCHIVE: $TRITON_TAR ($SEED_KEYS keys)"
else
    SEED_KEYS=0
    if [ "$WARMUP" = 1 ]; then
        echo "TRITON ARCHIVE: none yet; this pass compiles from scratch -> $TRITON_TAR"
    elif [ "${ALLOW_COLD_START:-0}" = 1 ]; then
        echo "TRITON ARCHIVE: none at $TRITON_TAR -- ALLOW_COLD_START=1, compiling at run time (may hang at iteration 0)"
    else
        echo "ERROR: no Triton archive at $TRITON_TAR. Run 'bash $(basename "$SELF") --warmup' first (or ALLOW_COLD_START=1)."
        exit 1
    fi
fi

# Install FLA 0.5.2 and the Swiss AI gate kernel atomically.
install_fla() {   # runs inside the container on one node; $1 = FLA_DIR
    local dst=$1 tmp=$1.tmp.$$
    if [ ! -f "$dst/fla/__init__.py" ]; then
        rm -rf "$tmp"
        pip install --quiet --no-deps --ignore-installed --target="$tmp" 'flash-linear-attention==0.5.2' 'fla-core==0.5.2' || return 1
        curl -sfL https://raw.githubusercontent.com/swiss-ai/flash-linear-attention/main/fla/ops/kda/gate.py -o "$tmp/fla/ops/kda/gate.py" || return 1
        if ! mv -T "$tmp" "$dst" 2>/dev/null; then
            if [ -f "$dst/fla/__init__.py" ]; then
                rm -rf "$tmp"  # Another job finished first.
            else
                rm -rf "$dst" && mv -T "$tmp" "$dst" || return 1
            fi
        fi
    fi
    PYTHONPATH=$dst python3 -c "import fla; assert fla.__version__.startswith('0.5.2'), 'got fla '+fla.__version__; print('fla', fla.__version__, 'OK')"
}
srun --nodes=1 --ntasks=1 --overlap --network=disable_rdzv_get "${CONTAINER_ARGS[@]}" \
    bash -c "$(declare -f install_fla); install_fla ${FLA_DIR@Q}"

# Queue the successor before training so it survives a crash or timeout.
training_complete() {
    local marker=$CKPT_DIR/latest_checkpointed_iteration.txt iter
    [ -n "$TOTAL_ITERS" ] || return 1
    [ -f "$marker" ] || return 1
    iter=$(tr -dc '0-9' < "$marker"); [ -n "$iter" ] || return 1
    [ "$iter" -ge "$TOTAL_ITERS" ]
}
REQUEUE_JOBID=""
if [ "$WARMUP" = 0 ] && [ "$AUTO_REQUEUE" = 1 ] && [ "$REQUEUE_COUNT" -lt "$MAX_REQUEUES" ] && ! training_complete; then
    # Preserve placement; use a fresh rendezvous port. Submission failure is nonfatal.
    REQUEUE_TIME=$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null | tr -d ' ' || true)
    REQUEUE_JOBID=$(sbatch --parsable \
        --dependency=afterany:"$SLURM_JOB_ID",singleton \
        --nodes="$SLURM_NNODES" ${REQUEUE_TIME:+--time="$REQUEUE_TIME"} --job-name="$SLURM_JOB_NAME" \
        ${SLURM_JOB_ACCOUNT:+--account="$SLURM_JOB_ACCOUNT"} --partition="$SLURM_JOB_PARTITION" \
        ${SLURM_JOB_QOS:+--qos="$SLURM_JOB_QOS"} \
        --chdir="$SLURM_SUBMIT_DIR" \
        --export=ALL,AUTO_REQUEUE=1,TOTAL_ITERS="$TOTAL_ITERS",MAX_REQUEUES="$MAX_REQUEUES",REQUEUE_COUNT=$(( REQUEUE_COUNT + 1 )),MASTER_PORT=$(( 20000 + RANDOM % 20000 )) \
        "$SELF" "${RESUME_ARGS[@]}") || REQUEUE_JOBID=""
    if [ -n "$REQUEUE_JOBID" ]; then
        echo "auto-requeue: pre-submitted job $REQUEUE_JOBID (afterany:$SLURM_JOB_ID, link $(( REQUEUE_COUNT + 1 ))/$MAX_REQUEUES)"
    else
        echo "WARNING: auto-requeue: sbatch failed; this job trains without a successor (resubmit by hand when it ends)"
    fi
fi

echo "MODE:      $([ "$WARMUP" = 1 ] && echo "warm-up ($WARMUP_ITERS iterations, ${#DATA_PATH_LIST[@]} shards)" || echo training)"
echo "EXP_NAME:  $EXP_NAME"
echo "CKPT_DIR:  $CKPT_DIR"
echo "LOAD:      ${RESUME_FROM:-<warm-up: random init>}"
echo "DATA:      ${#DATA_PATH_LIST[@]} prefix(es) under $SFT_DATA_ROOT (glob ${SFT_DATA_GLOB})"
echo "LOSSMASK:  $LOSS_MASK_USED"
echo "SFT:       $EPOCHS epoch(s) | pack=$PACK/$PACKING_STRATEGY | MUON_LOG=$MUON_LOG"
echo "MEGATRON:  $MEGATRON_LM_DIR @ $(git -C "$MEGATRON_LM_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "CONTAINER: $CONTAINER_IMAGE"
echo "NODES:     $SLURM_NNODES ($SLURM_NTASKS GPUs) | TP$TP PP$PP EP$EP CP$CP -> DP=$DP"
echo "CMD:       python3 pretrain_gpt.py ${MEGATRON_ARGS[*]@Q}"

# Keep only the newest checkpoint (user, 2026-11-21; a full one is ~5.5 TB): delete every iter_* below the iteration in
# latest_checkpointed_iteration.txt, which Megatron writes only after a save completed. Runs on the batch node every
# 5 minutes while training, once more at the end.
prune_old_checkpoints() {
    local latest d it
    [ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ] || return 0
    latest=$(tr -dc '0-9' < "$CKPT_DIR/latest_checkpointed_iteration.txt"); [ -n "$latest" ] || return 0
    for d in "$CKPT_DIR"/iter_[0-9]*; do
        [ -d "$d" ] || continue
        it=$(basename "$d"); it=$((10#${it#iter_}))
        if [ "$it" -lt "$latest" ]; then echo "prune: removing $d (newest is $latest)"; rm -rf "$d"; fi
    done
}
PRUNER_PID=""
if [ "$WARMUP" = 0 ]; then
    prune_old_checkpoints
    ( while sleep 300; do prune_old_checkpoints; done ) &
    PRUNER_PID=$!
fi
SRUN_RC=0
# test: node-local Triton/inductor cache (JIT_LOCAL=1) keeps ~300k cache files off /scratch
[ -z "${JIT_LOCAL:-}" ] || srun --ntasks-per-node=1 --overlap mkdir -p "$JITCACHE_ROOT"
srun -lu --mpi=pmix --network=disable_rdzv_get \
    "${CONTAINER_ARGS[@]}" \
    --cpus-per-task "$SLURM_CPUS_PER_TASK" --wait 120 \
    bash -c "$IN_CONTAINER" || SRUN_RC=$?

echo "END TIME: $(date) (srun exit code $SRUN_RC)"
if [ -n "$PRUNER_PID" ]; then kill "$PRUNER_PID" 2>/dev/null || true; prune_old_checkpoints; fi
[ -z "${JIT_LOCAL:-}" ] || srun --ntasks-per-node=1 --overlap rm -rf "$JITCACHE_ROOT/triton/job-$SLURM_JOB_ID" "$JITCACHE_ROOT/inductor/job-$SLURM_JOB_ID" || true

if [ "$WARMUP" = 1 ]; then
    # Retain newly compiled kernels even after a failed warm-up.
    CACHE_ROOT=$JITCACHE_ROOT/triton/job-$SLURM_JOB_ID
    [ -d "$CACHE_ROOT" ] || { echo "ERROR: no Triton cache at $CACHE_ROOT"; exit 1; }
    triton_warm_build "$CACHE_ROOT" "$TRITON_TAR.new" "$TRITON_TAR" && mv "$TRITON_TAR.new" "$TRITON_TAR" \
        || { echo "ERROR: no compiled kernels under $CACHE_ROOT (srun exit $SRUN_RC); the archive is unchanged"; exit 1; }
    KEYS=$(archive_keys "$TRITON_TAR") || { echo "ERROR: cannot list $TRITON_TAR"; exit 1; }
    echo "ARCHIVE:   $TRITON_TAR ($KEYS keys, $(du -h "$TRITON_TAR" | cut -f1)); seeded $SEED_KEYS, added $(( KEYS - SEED_KEYS ))"
    rm -rf "$CACHE_ROOT" "$JITCACHE_ROOT/inductor/job-$SLURM_JOB_ID"
    if [ "$KEYS" -lt "$MIN_TRITON_ARCHIVE_KEYS" ]; then
        echo "WARNING:   only $KEYS keys, below the $MIN_TRITON_ARCHIVE_KEYS floor. Run the warm-up again; it seeds from this archive and compiles the rest."
        exit 1
    elif [ "$SRUN_RC" != 0 ] && [ "$KEYS" -le "$SEED_KEYS" ]; then
        # No new kernels after an early failure does not establish saturation.
        echo "WARNING:   the pass failed (srun exit $SRUN_RC) and added no kernels; the archive is unchanged. Fix the failure and run the warm-up again."
        exit 1
    elif [ "$KEYS" -le "$SEED_KEYS" ]; then
        echo "SATURATED: no new kernels this pass. Ready for training."
    else
        # Accept a failed teardown if the pass still added kernels.
        [ "$SRUN_RC" = 0 ] || echo "NOTE:      srun exit $SRUN_RC (teardown crash after the iterations); the kernels it compiled are in the archive."
        echo "Added $(( KEYS - SEED_KEYS )) kernels this pass. Run the warm-up again until it reports SATURATED."
    fi
    exit 0
fi

[ "${KEEP_JITCACHE:-0}" = 1 ] || rm -rf "$JITCACHE_ROOT/triton/job-$SLURM_JOB_ID" "$JITCACHE_ROOT/inductor/job-$SLURM_JOB_ID"

if [ -n "$REQUEUE_JOBID" ]; then
    if training_complete; then
        echo "auto-requeue: iteration budget reached, cancelling pending job $REQUEUE_JOBID"; scancel "$REQUEUE_JOBID" || true
    else
        echo "auto-requeue: training incomplete, pending job $REQUEUE_JOBID will resume"
    fi
fi
exit "$SRUN_RC"
