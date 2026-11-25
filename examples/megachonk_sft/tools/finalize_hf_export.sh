#!/bin/bash
# Make a MegaChonk SFT HF export self-contained (runnable with transformers trust_remote_code, and stageable for
# rl-infra): modeling code with the embedding norm, auto_map, the SFT tokenizer's extra files, a README.
#   bash finalize_hf_export.sh <hf export dir> <iteration>
set -euo pipefail
D=$1; IT=$2
C=${C:-/home01/hanint06/projects/megachonk-sft/embnorm-hf-code}
T=${T:-/home01/hanint06/sft-data-env/tokenizer/Apertus_2_instruct-b801ff1}
[ -f "$D/CONFIG_ONLY_EXPORT_COMPLETE" ] || { echo "ERROR: $D is not a complete export"; exit 1; }
cp -f "$C/modeling_apertus2.py" "$C/configuration_apertus2.py" "$C/MODELING_SOURCE" "$D/"
for f in tokenizer.json tokenizer_config.json special_tokens_map.json generation_config.json; do cp -f "$T/$f" "$D/$f"; done
python3 - "$D/config.json" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
assert c.get("embedding_norm") is True, "config.json lacks embedding_norm: true"
c.setdefault("auto_map", {}).update({
    "AutoConfig": "configuration_apertus2.Apertus2Config",
    "AutoModel": "modeling_apertus2.Apertus2Model",
    "AutoModelForCausalLM": "modeling_apertus2.Apertus2ForCausalLM",
})
json.dump(c, open(p, "w"), indent=2)
print("config.json: auto_map added, embedding_norm true")
PY
n=$(ls "$D"/model-*.safetensors | wc -l)
cat > "$D/README.md" <<MD
---
library_name: transformers
tags: [apertus2, megachonk, sft, private]
---
# MegaChonk SFT v1 (16K), iteration $IT

Apertus 2 MegaChonk (594B MoE, 61 layers, KDA 3:1 + global attention, 256 experts top-8, embedding norm) after
supervised fine-tuning, exported from the Megatron checkpoint at SFT iteration $IT of 1,890.

- Base: MegaChonk pretraining iteration 96,000 (with \`--embedding-norm\`, from pretraining iteration 90,000 on).
- SFT data: rebalanced chonk-sft-v0.3 mix (code ~8B, STEM 7.6B, Lean ~2.8B, tool use, chat, web search, medical,
  instruction following, factuality, safety, perturbed), conversations longer than 16,384 tokens removed:
  6,992,225 conversations, 15.8B tokens. Apertus 2 message format, tokenizer Apertus_2_instruct (b801ff1).
- Training: hanint05's Megatron SFT trainer (swiss-ai/Megatron-LM-MoE feat/sft_trainer_ap2 merged into main
  3a68f6307), 64 nodes, TP2 PP16 EP16 CP2, global batch 512 x 16K, md_decoupling (Muon) with matrix LR 3.08e-4,
  LR 6.16e-4 cosine to 6.16e-5, loss on assistant tokens.
- Files: $n weight shards (bf16), config.json with \`embedding_norm: true\`, the tokenizer used for the SFT data.

## Running it

The modeling code here (\`modeling_apertus2.py\`, \`configuration_apertus2.py\`: hfconverter c2ab4910948b plus the
embedding norm, \`model.embedding_norm\`, an RMSNorm with a learnable gain applied after \`embedding_multiplier\`) is
required: code without the norm loads the weights but drops the gain and produces garbage (seen on the pretraining
iteration-90,000 export). Load with \`trust_remote_code=True\`. Prompts use the Apertus 2 message format
(apertus-common), not a Jinja chat template.

## rl-infra / vLLM / Megatron

- vLLM: the Apertus 2 model file needs the norm (patch \`apertus2-embedding-norm.patch\` on KISTI in
  \`/home01/hanint06/projects/megachonk-sft/embnorm-vllm/\`).
- Megatron-Bridge (HF -> Megatron for the trainer): \`embnorm-patches/bridge-0001-apertus2-embedding_norm.patch\`;
  Megatron-LM-MoE: the \`--embedding-norm\` commit (596c5271d; \`embnorm-patches/megatron-0001-embedding-norm.patch\`).
- rl-infra's \`stage_model.sh\` checks \`MODELING_SOURCE == "hfconverter c2ab4910948b"\` in link mode and fetches that
  code (without the norm) in copy mode: this folder's \`MODELING_SOURCE\` says "+ embedding_norm" on purpose, so link
  mode refuses it until rl-infra accepts the norm-aware code. Do not stage it in copy mode.
MD
echo "finalized $D ($n shards)"
