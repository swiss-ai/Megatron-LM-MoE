#!/bin/bash
#SBATCH --job-name=mc-eval
#SBATCH --account=in0138
#SBATCH --partition=intTest
#SBATCH --comment=field=nlp;appl=pytorch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --propagate=NONE
#SBATCH --output=/scratch/hanint06/megachonk-sft/evals/logs/%x-%j.out

# MegaChonk before/after evaluation: serve one HF export (bf16) on 4 nodes with vLLM (TP4 inside a node, PP4 across
# nodes, multi-node without Ray as ~/glm53-traces/serve.sh, NCCL over sockets on hsn0) and run evals-post-train's
# lm-eval launcher against it from the head node (prebuilt eval image, as kisti/serve_eval.sbatch).
# Server: NeMo-RL image (vLLM 0.26) with the RL stack's gpu_model_runner.py and the embedding-norm Apertus 2 model file
# (~/projects/megachonk-sft/embnorm-vllm/apertus2.py) mounted over the image's copies.
# Settings (sbatch --export=ALL,...):
#   MODEL   HF export directory (required)       NAME  run name (default: basename of MODEL)
#   RAW     raw-continuation tasks, "task=shots" separated by ":" (default gsm8k=8:mmlu=5:arc_challenge=25:global_mmlu=5)
#           No chat template, greedy, the base model's tokenizer (adds <s>) for every model: same prompts for all.
#   CHAT    Apertus 2 format tasks, "task=repeats" separated by ":" (default none), effort high, thinking sampling
#   MAXLEN  server --max-model-len (8192)        LIMIT  samples per task (all)     CONC  concurrent requests (64)
set -uo pipefail
REPO=/home01/hanint06/projects/evals-post-train
MODEL=${MODEL:?MODEL: HF export directory}
NAME=${NAME:-$(basename "$MODEL")}
RAW=${RAW-gsm8k=8:mmlu=5:arc_challenge=25:global_mmlu=5}
CHAT=${CHAT:-}
MAXLEN=${MAXLEN:-8192} LIMIT=${LIMIT:-} CONC=${CONC:-64} GPU_UTIL=${GPU_UTIL:-0.88}
# 8192 batched tokens OOMed the last PP stage in the prompt-logprob log_softmax (job 1346816, MMLU).
MAX_BATCHED=${MAX_BATCHED:-2048}
# SERVE_TOK: tokenizer the server uses (default: the model dir). The SFT export's tokenizer adds no <s>; raw tasks of
# an SFT model take SERVE_TOK=<base model dir> so the server tokenizes the prompts as for the base model.
SERVE_TOK=${SERVE_TOK:-}
LOAD_TIMEOUT_MIN=${LOAD_TIMEOUT_MIN:-120}
SERVE_IMAGE=/scratch/hanint06/container-images/nemo-rl-apertus2-cfcfede4-ucclfix-cxisetup.sqsh
EVAL_IMAGE=/scratch/hanint06/container-images/evals-post-train-eval-6e354c289658.sqsh
PATCH=/home01/hanint06/projects/nemo-rl-apertus2-vllm
EMBNORM=/home01/hanint06/projects/megachonk-sft/embnorm-vllm/apertus2.py
SERVE_PYDEPS=/scratch/hanint06/nemo-rl-apertus2/pydeps-gym
VLLM_PY=/usr/local/lib/python3.12/dist-packages/vllm
BASE_TOK=/scratch/hanint06/models/apertus2-megachonk-iter96000
CHAT_TOK=/scratch/hanint06/ept/tokenizers/apertus-omni-tokenizer/tokenizers/Apertus_2_instruct
OUT=/scratch/hanint06/megachonk-sft/evals/runs/$NAME
PORT=8011
mkdir -p "$OUT"
# Marker for unattended pipelines (final_pipeline.sh): the exit code of this job.
trap 'echo $? > "$OUT/EVAL_FINISHED-$SLURM_JOB_ID"' EXIT
[[ -f "$MODEL/config.json" && -s "$EVAL_IMAGE" && -f "$BASE_TOK/tokenizer.json" && -f "$CHAT_TOK/tokenizer.json" ]] \
  || { echo "missing model, eval image or tokenizer"; exit 1; }
grep -q '"embedding_norm": true' "$MODEL/config.json" || { echo "config.json has no embedding_norm: true"; exit 1; }

NNODES=$SLURM_JOB_NUM_NODES
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
HEAD_ADDR=$(srun --nodes=1 --ntasks=1 -w "$HEAD" ip -4 -o addr show hsn0 | awk '{print $4}' | cut -d/ -f1 | head -1)
[[ "$HEAD_ADDR" == 10.169.* ]] || { echo "no Slingshot address on $HEAD: '$HEAD_ADDR'"; exit 1; }
echo "$(date +%T) job $SLURM_JOB_ID on $SLURM_JOB_NODELIST (head $HEAD $HEAD_ADDR): $MODEL as $NAME"
echo "raw tasks: ${RAW:-none}; chat tasks: ${CHAT:-none}; max-model-len $MAXLEN; batched $MAX_BATCHED; util $GPU_UTIL; server tokenizer ${SERVE_TOK:-model dir}; limit ${LIMIT:-all}"
( cd "$REPO" && echo "evals-post-train $(git rev-parse HEAD)" ); md5sum "$EMBNORM"

srun --nodes="$NNODES" --ntasks-per-node=1 --cpu-bind=none --network=disable_rdzv_get \
  --container-image="$SERVE_IMAGE" --container-writable --no-container-mount-home \
  --container-mounts=/scratch/hanint06:/scratch/hanint06,/home01/hanint06:/home01/hanint06,$PATCH/gpu_model_runner.py:$VLLM_PY/v1/worker/gpu_model_runner.py,$EMBNORM:$VLLM_PY/model_executor/models/apertus2.py \
  bash -c "export PYTHONPATH=$SERVE_PYDEPS NCCL_SOCKET_IFNAME=hsn0 GLOO_SOCKET_IFNAME=hsn0 NCCL_NET=Socket NCCL_IB_DISABLE=1 NCCL_PROTO=^LL128
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1 TOKENIZERS_PARALLELISM=false VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 VLLM_HOST_IP=\$(ip -4 -o addr show hsn0 | awk '{print \$4}' | cut -d/ -f1 | head -1)
C=/tmp/mc-eval-cache; export HF_HOME=\$C/hf VLLM_CACHE_ROOT=\$C/vllm XDG_CACHE_HOME=\$C FLASHINFER_WORKSPACE_BASE=\$C/flashinfer TRITON_CACHE_DIR=\$C/triton; mkdir -p \$C/vllm
R=\$SLURM_NODEID; H=''; [ \$R != 0 ] && H=--headless
echo \"\$(hostname) node rank \$R, own address \$VLLM_HOST_IP\"
exec vllm serve $MODEL --served-model-name mc --host 127.0.0.1 --port $PORT \
  --tensor-parallel-size 4 --pipeline-parallel-size $NNODES --nnodes $NNODES --node-rank \$R --master-addr $HEAD_ADDR \$H \
  --max-model-len $MAXLEN --gpu-memory-utilization $GPU_UTIL --max-num-seqs 256 --max-num-batched-tokens $MAX_BATCHED ${SERVE_TOK:+--tokenizer $SERVE_TOK} \
  --trust-remote-code" > "$OUT/vllm-$SLURM_JOB_ID.log" 2>&1 &
server=$!
start=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  kill -0 $server 2>/dev/null || { echo "vLLM died:"; grep -v -E "^\s*$" "$OUT/vllm-$SLURM_JOB_ID.log" | tail -40; exit 1; }
  (( $(date +%s) - start > LOAD_TIMEOUT_MIN * 60 )) && { echo "no health after $LOAD_TIMEOUT_MIN min"; kill $server; exit 1; }
  sleep 15
done
echo "$(date +%T) vLLM ready after $(( $(date +%s) - start ))s"
grep -E "KV cache|Maximum concurrency|weights took|Model loading took" "$OUT/vllm-$SLURM_JOB_ID.log" | head -8

# One launcher call per task, in the eval image on the head node (the server listens on its 127.0.0.1).
run_task() {  # $1 task  $2 label  $3.. launcher flags
  local task=$1 label=$2; shift 2
  echo "$(date +%T) == $label"
  srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" --cpu-bind=none \
    --container-image="$EVAL_IMAGE" --container-writable --no-container-mount-home \
    --container-mounts=/scratch/hanint06:/scratch/hanint06,/home01/hanint06:/home01/hanint06 \
    bash -c "cd $REPO
export PATH=$REPO/scripts/direct_serving/inline_slurm:/opt/lm-eval-venv/bin:\$PATH LM_EVAL_VENV=/opt/lm-eval-venv EVAL_PREBUILT_ENV=1 EVAL_PREBUILT_HARNESS_ROOT=/opt/lm-eval-harness
export TARGET_API_KEY=local SCRATCH=/scratch/hanint06 HF_HOME=/scratch/hanint06/ept/hf HOME=/scratch/hanint06/ept/home
mkdir -p \$HOME
bash scripts/lm_eval_harness/launch_evaluations.sh single --task $task --model mc --name $NAME-$label \
  --backend openai --api-base-url http://127.0.0.1:$PORT/v1 --api-model-name mc --api-num-concurrent $CONC \
  ${LIMIT:+--limit $LIMIT} --failure-policy fail-fast --logs-root $OUT/logs --no-wandb $*" &
  local ev=$!
  while kill -0 $ev 2>/dev/null; do
    kill -0 $server 2>/dev/null || { echo "vLLM died during $label"; tail -40 "$OUT/vllm-$SLURM_JOB_ID.log"; kill $ev; return 2; }
    sleep 15
  done
  wait $ev
}

failed=0 dead=0
for item in ${RAW//:/ }; do
  task=${item%%=*} shots=${item#*=}
  run_task "$task" "$task-${shots}shot-raw" --tokenizer "$BASE_TOK" --no-chat-template --num-fewshot "$shots" \
    --sampling-profile greedy --max-length "$MAXLEN"
  rc=$?; [[ $rc == 2 ]] && { failed=1 dead=1; break; }; [[ $rc != 0 ]] && failed=1
  echo "$(date +%T) == $task finished rc=$rc"
done
[[ $dead == 0 ]] && for item in ${CHAT//:/ }; do
  task=${item%%=*} rep=${item#*=}
  run_task "$task" "$task-apertus2-high" --tokenizer "$CHAT_TOK" --chat-format apertus2 --max-length "$MAXLEN" \
    --reasoning-effort high --repeats "$rep"
  rc=$?; [[ $rc == 2 ]] && { failed=1; break; }; [[ $rc != 0 ]] && failed=1
  echo "$(date +%T) == $task finished rc=$rc"
done
kill $server 2>/dev/null; wait $server 2>/dev/null
echo "$(date +%T) exit $failed"
exit $failed
