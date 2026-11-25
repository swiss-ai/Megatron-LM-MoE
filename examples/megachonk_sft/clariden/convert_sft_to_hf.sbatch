#!/bin/bash
# HF export of the MegaChonk SFT checkpoint (iteration 1,540), submitted by sft1540_fetch.sbatch with CKPT=<iter dir>.
# Same converter, worktrees and overrides as megachonk_convert_embnorm.sbatch (pretraining 96,000) and as the KISTI
# export of SFT step 880 (export_sft_to_hf.sbatch); the reference is hf-reference-sft (SFT tokenizer, embedding_norm).
set -euo pipefail
export SCRATCH=/iopsstor/scratch/cscs/dmelikidze
CKPT=${CKPT:?}
REF=$HOME/kisti-backup-20261123-megachonk-sft/megachonk-sft/hfconverter/hf-reference-sft
export HF_OUTPUT=$SCRATCH/megachonk-hf/sft-v1-16k-iter_0001540
echo "$(date) checkpoint: $CKPT -> $HF_OUTPUT (reference $REF)"
if [[ -f "$HF_OUTPUT/CONFIG_ONLY_EXPORT_COMPLETE" ]]; then echo "export is complete already"; exit 0; fi
rm -rf "$HF_OUTPUT"
H=$HOME/projects/hfconverter
export BRIDGE_ROOT=$HOME/projects/megatron-bridge-apertus2-embnorm MEGATRON_ROOT=$HOME/projects/megatron-lm-moe-42e7a63-embnorm
export RUN_ENV=$SCRATCH/megachonk-debug/convert-env.toml LOCAL_HFCONVERTER_PATH=$H
export CONVERTER_SCRIPT=$H/check_apertus2_export_keys.py CPU_MEMORY_NODES=0,1,2,3
srun --nodes=1 --ntasks=1 --network=disable_rdzv_get --environment="$RUN_ENV" \
  --container-mounts="${SCRATCH}:${SCRATCH},${HOME}:${HOME},/ritom:/ritom" \
  bash -c 'unset VIRTUAL_ENV UV_PYTHON PYTHONHOME; export UV_CACHE_DIR=/ritom/scratch/cscs/$USER/cache/uv; mkdir -p "$UV_CACHE_DIR"; uv run --no-project python -c "import sys; print(\"uv cache warm:\", sys.version.split()[0])"'
bash $H/convert_bridge.sbatch export "$CKPT" "$REF" "$HF_OUTPUT" \
  --tp 1 --pp 4 --ep ${EP:-8} --etp 1 --dtype bf16 --reference-mode config-only \
  --pipeline-layout "Et*16|t*15|t*15|t*15L" \
  --model-override kda_legacy_gate_out_proj_bias=false \
  --model-override linear_attn_a_log_per_channel=true \
  --model-override "moe_router_quantile_balancing_method=\"marin_histogram\"" \
  --model-override moe_router_quantile_balancing_marin_num_bins=10000 \
  --model-override embedding_norm=true \
  --trust-remote-code --max-shard-size-gib 16
[[ -f "$HF_OUTPUT/CONFIG_ONLY_EXPORT_COMPLETE" ]] || { echo "ERROR: export not marked complete"; exit 1; }
grep -q '"model.embedding_norm.weight"' "$HF_OUTPUT/model.safetensors.index.json" || { echo "ERROR: no embedding norm"; exit 1; }
grep -q '"embedding_norm": true' "$HF_OUTPUT/config.json" || { echo "ERROR: config lacks embedding_norm"; exit 1; }
ls "$HF_OUTPUT" | wc -l; du -sh "$HF_OUTPUT"
echo "$(date) EXPORT_DONE"
