# MegaChonk SFT v1 (16K)

Supervised fine-tuning of Apertus 2 MegaChonk (594B MoE) with the Apertus SFT trainer of `feat/sft_trainer_ap2`
(`--ap-sft`). This branch is `main` 3a68f6307 + `feat/sft_trainer_ap2` 8983149fb merged + one fix
(`torch_dist` load plans over a gloo group), plus the launchers and tools of the run in this folder.

## Model

594B parameters (560B in routed experts): 61 layers, hidden 8192, 256 experts top-8 with latent size 4096, 2 shared
experts, 3 dense + 58 MoE layers, KDA 3:1 with the last layer global, NoPE, per-channel `A_log`, and the embedding
norm (`--embedding-norm`, pretraining from iteration 90,000 on). Base: pretraining iteration 96,000 (4.83T tokens),
HF export imported to Megatron with Megatron-Bridge (`kisti/import_hf_to_megatron.*`).

## Data

Rebalanced chonk-sft-v0.3 mix: code ~8B tokens, STEM 7.6B, Lean ~2.8B, tool use 5.4B, chat 2.7B, web search,
medical, instruction following, factuality, safety, perturbed; 7,373,813 conversations, 28.6B tokens, Apertus 2
message format, tokenizer `Apertus_2_instruct` (apertus-omni-tokenizer b801ff1). Conversations longer than 16,384
tokens are removed, not truncated (`tools/filter_long_index.py` writes a filtered `tokens.idx`; the trainer itself
would truncate): 6,992,225 conversations, 15.8B tokens. Best-fit-decreasing packing: 967,337 packs of 16K, 99.41 %
utilization; one epoch = 1,890 steps at global batch 512.

## Training setup

| | |
|---|---|
| Hardware | 64 GH200 nodes (256 GPUs) |
| Parallelism | TP2 + sequence parallel, PP16 with 2 virtual stages (`Et\|(tt\|)*15,t\|(tt\|)*14,tL`), EP16, CP2 (`kcp`), DP4 |
| Batch | 512 x 16,384 tokens (8.4M tokens per step), micro-batch 1 |
| Optimizer | `md_decoupling` (Muon, polar express, Nesterov 0.95), layer-wise distributed optimizer |
| Learning rates | LR 6.16e-4, cosine to 6.16e-5, 3 % warm-up; matrix LR 3.08e-4 (10x below the Chonk SFT default: lower was better on Chonk); gains LR 6.16e-4 |
| Routing | quantile balancing `marin_exact` (as pretraining), aux loss 1e-3, flex dispatcher with DeepEP |
| Memory | full recompute, no expert offloading, MoE comm overlap off; 45 GB max allocated per GPU |
| Checkpoints | every 220 steps, newest kept (5.5 TB each with optimizer state) |

## Run

| Step | Assistant loss | Grad norm | LR | s/step | tokens/s/GPU | TFLOP/s/GPU |
|---|---|---|---|---|---|---|
| 1 | 1.048 | 2.21 | 1.1e-5 | (compile) | | |
| 220 | 0.683 | 0.106 | 6.05e-4 | 51.4 | 638 | 196 |
| 440 | 0.630 | 0.107 | 5.58e-4 | 51.4 | 637 | 196 |
| 880 | 0.582 | 0.090 | 3.83e-4 | 51.5 | 636 | 196 |
| 1,320 | 0.566 | 0.087 | 1.84e-4 | 51.6 | 635 | 195 |
| 1,540 | 0.556 | 0.083 | 1.10e-4 | 51.6 | 635 | 195 |
| 1,574 | 0.553 | 0.082 | 1.01e-4 | 51.4 | 638 | 196 |

MFU ~19.8 %. Steps 1-1,574 ran on KISTI Hangang (jobs 1337283 and 1342408, 2026-11-22/23); KISTI compute ended
2026-11-23 10:12 KST during step 1,574. The step 1,540 checkpoint (with optimizer state) was moved to CSCS Clariden
through a private HF repo, and the run resumes there (`clariden/`).

## Checkpoints (private Hugging Face repos)

| Repo | Content |
|---|---|
| `swiss-ai/megachonk-sft-v1-16k-iter_0000880` | SFT step 880, HF format (bf16, norm-aware modeling code, `auto_map`, SFT tokenizer) |
| `swiss-ai/megachonk-sft-v1-16k-iter_0001540` | SFT step 1,540, HF format |
| `swiss-ai/megachonk-sft-v1-16k-iter_0001540-megatron` | SFT step 1,540, Megatron `torch_dist` checkpoint with optimizer state (to resume) |
| `swiss-ai/megachonk-sft-v1-data-apertus2` | the tokenized SFT data and the 16K-filtered index |

Load the HF exports with `trust_remote_code=True`; prompts use the Apertus 2 message format (apertus-common).
Code without the embedding norm loads these weights silently and generates one repeated token.

## Evaluation (before and after SFT)

lm-eval (evals-post-train) over vLLM in bf16, raw continuation without chat template, greedy, the same tokenizer and
`<s>` for both models (`evals/`):

| Benchmark | Pretraining 96,000 | SFT 880 |
|---|---|---|
| GSM8K, 8-shot, strict match | 64.6 | **77.9** |
| MMLU, 5-shot | 68.5 | **73.6** |
| ARC-Challenge, 25-shot, acc_norm | 67.2 | **68.1** |
| Global-MMLU Lite, 5-shot, 15 languages | 65.0 | **68.6** |

Global-MMLU Lite improves in all 15 languages (+1.5 to +6.2 points: Chinese 68.3 -> 69.8, Yoruba 45.8 -> 52.0).

## Files

- `kisti/sft_megachonk.sh`: the launcher of the KISTI run (hanint05's Chonk SFT launcher adapted to MegaChonk:
  architecture, layout, embedding norm, marin_exact routing, gloo-group load, checkpoint pruning, auto-requeue).
  `kisti/launch_when_ready.sh` shows the exact environment of the real run.
- `clariden/sft_megachonk.sh`: the same launcher on Clariden (pretraining EDF `alps-pytorch2512-a139.toml`, Alps
  network from the EDF, account/partition, only `omegaconf` installed on top of the image);
  `clariden/submit_resume.sh` submits the full-depth warm-up and the resume from step 1,540.
- `kisti/import_hf_to_megatron.*`, `kisti/export_sft_to_hf.sbatch`, `clariden/convert_sft_to_hf.sbatch`: HF <->
  Megatron conversion with Megatron-Bridge (the export runs on 4 or 8 nodes, PP4 with EP4 or EP8).
- `tools/`: 16K filter, export finalizer (modeling code, `auto_map`, tokenizer, README), private HF uploads (batched
  commits for multi-TB folders); `clariden/hf_fetch_fast.sbatch`: HF downloads on a compute node (3.4 GB/s).
- `evals/`: vLLM serving on 4 nodes (TP4 x PP4, embedding-norm model file) + evals-post-train lm-eval, and the table.
- `embedding_norm/`: the embedding-norm patches for vLLM's Apertus 2 model and for Megatron-Bridge
  (Megatron-LM-MoE has `--embedding-norm` on `main`).
