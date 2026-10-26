# Apertus-2 SFT

## Training modes

All modes use `pretrain_gpt.py` and the usual data-path flags.

| Mode | Input | Processing |
| --- | --- | --- |
| `--sft` | Local JSON/JSONL `messages: [{role, content}, …]` | HF Datasets loads rows; the conversation tokenizer creates targets during loading. |
| `--ap-sft` | Indexed tokens | Scans Apertus-2 controls for output-only weights. |
| `--ap-sft --ap-sft-load-loss-mask` | Indexed tokens and float weights | Uses stored weights. |

The [JSON loader](../megatron/training/datasets/sft_dataset.py) supports ordinary
chat messages, not native Apertus-2 dicts or direct HF Hub loading. Prepare
Apertus-2 JSON/HF data externally with `apertus-common`, then export indexed files.

The [AP loader](../megatron/training/datasets/apertus_sft_dataset.py) expects one
nonempty conversation per sequence/document: `tokens.{bin,idx}` plus optional
`loss_weights.{bin,idx}`. Weights are float32/float64 and match token lengths and
document boundaries. Pass the `tokens` prefix as the data path.

## Weighting and normalization

Stored weights describe each token before target shifting. Zero suppresses it;
positive values scale its loss. Weights must be finite, nonnegative, and
representable as float32.

Without stored weights, entire output frames (headers/controls included) and
every `<|wait|>` get weight one; inputs get zero. IDs come from tokenizer metadata.

`--mask-loss-token-ids` zeroes training weights for matching target tokens in
either loading mode. It leaves input tokens and `assistant_loss` unchanged.

AP requires `--calculate-per-token-loss`:

```text
loss = sum(weight × target_cross_entropy) / count(weight > 0)
```

Dividing by the weight sum cancels uniform scaling; converting it to an integer
truncates fractional weights. Positive-target counts avoid both problems.
Gradients and reporting use the same global numerator/count across microbatches
and DP/CP ranks. An all-zero mask contributes zero language-model loss.
The count excludes targets suppressed by `--mask-loss-token-ids`.
AP rejects `--use-torch-fsdp2` because its wrapper does not implement this
gradient normalization.

`--ap-sft-report-assistant-loss` adds `assistant_loss`: unweighted mean
cross-entropy over output targets, including those suppressed by training weights.
This optional metric requires Apertus control metadata in either loading mode.

## Packing and parallelism

Each sample is one conversation, or with `--ap-sft-pack-samples` several whole
conversations packed `greedy` or `bfd` (`--ap-sft-packing-strategy`). Labels and
weights shift per conversation, conversation boundaries reset attention and
positions, and long conversations keep their prefix. Indices are cached; every
epoch is shuffled independently. AP needs THD-capable attention and rejects
`--pretraining-packing-strategy bfd`.

With CP or sequence parallelism, each conversation is padded to a multiple of
`lcm(2 × CP, CP × SP)` with CP > 1, else SP (SP = TP size with
`--sequence-parallel`, else 1). Padding has zero weight and is marked in the MoE
padding mask, so CP works without `--dataloader-inter-document-masking` or
`--cp-doc-aware`.

Transformer Engine CUDA graphs need `--cuda-graph-warmup-steps` ≥ 1 so capture
includes the MoE padding mask. Set `--cuda-graph-scope` to scopes without `attn`
(e.g. `moe_router moe_preprocess`): the default captures whole layers, and TE
replay does not accept packed-sequence metadata.

## Exact epochs

`--ap-sft-epochs N` (with `--ap-sft --calculate-per-token-loss`) replaces
`--train-iters`/`--train-samples` and trains every conversation or pack of the
training split exactly once per epoch. The last global batch of each epoch is
filled with zero-loss samples instead of repeats, e.g. 1,000 samples at global
batch 64 take 16 steps including 24 dummy samples.

Requirements: the `single` loader, a global batch divisible by micro-batch size ×
DP size, and training data without blend weights. Batch rampup, training phases
and automatic batch reduction are unsupported. Consumed-sample counters include
the dummy samples, so resuming with the same data and settings continues exactly.
