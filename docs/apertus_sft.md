# SFT modes and loss weighting

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
positive values scale its loss. This supports different weights for thinking,
replies, tool calls, and claims, including suppressing a turn corrected later.
Weights must be finite, nonnegative, and representable as float32.

Without stored weights, entire output frames (headers/controls included) and
every `<|wait|>` get weight one; inputs get zero. IDs come from tokenizer metadata.

AP requires `--calculate-per-token-loss`:

```text
loss = sum(weight × target_cross_entropy) / count(weight > 0)
```

Dividing by the weight sum cancels uniform scaling; converting it to an integer
truncates fractional weights. Positive-target counts avoid both problems.
Gradients and reporting use the same global numerator/count across microbatches
and DP/CP ranks. An all-zero mask contributes zero language-model loss.

`--ap-sft-report-assistant-loss` adds `assistant_loss`: unweighted mean
cross-entropy over output targets, including those suppressed by training weights.
It uses a separate shifted mask and the same global sum/count reporting reduction.
This optional metric requires Apertus control metadata in either loading mode.

## Reuse and changes

- **Old Apertus trainer:** Reused whole-conversation sampling, greedy/BFD packing,
  cached indices, and independently shuffled epochs. Replaced concatenated
  `[tokens, mask]` storage with indexed pairs. Labels/weights shift per
  conversation, with a zero-weight final target. Lengths define attention
  boundaries and position resets; physical CP/SP padding is separate from weights.
- **MM branch:** Adopted the supervised-token denominator from this repo's
  `origin/multimodality/main`, commit `c8d533834`; `4e3ab6796` scoped it to GPT.
  AP enables it automatically. The old SwissAI MM branch still uses
  `int(loss_mask.sum())`. Modality lookup/reporting and sample equalization
  were not ported.
- **MCore:** Reused existing global gradient normalization in
  [schedules.py](../megatron/core/pipeline_parallel/schedules.py) and
  [finalize_model_grads.py](../megatron/core/distributed/finalize_model_grads.py).
  Added AP's denominator and zero-count reporting protection.

Optional packing: `--ap-sft-pack-samples --ap-sft-packing-strategy greedy|bfd`.
Long conversations retain their prefix. AP requires THD-capable attention.
Transformer Engine CUDA graphs require `--cuda-graph-warmup-steps` ≥ 1 so capture
includes the MoE padding mask. Captured scopes must leave attention outside the
graph: current TE attention/full-layer replay rejects packed-sequence metadata.
CPU encoding/export tests pass; CUDA training remains unverified.

## Exact epochs

Use `--ap-sft --ap-sft-epochs 1 --calculate-per-token-loss` to train on every
conversation in the **training split** exactly once. With packing, every pack is
used once; packing and prefix truncation stay the same. `--ap-sft-epochs N`
independently shuffles each epoch and fills its final global batch with dummy
samples whose training/assistant masks are zero. Real samples are never repeated
to fill a batch. For example, 1,000 samples with global batch 64 take 16 steps,
including 24 dummy samples.

This replaces `--train-iters`/`--train-samples`. Use the `single` loader (default),
a fixed global batch divisible by micro-batch size × DP size, and training
prefixes without explicit blend weights; stored token loss weights are supported.
Multiple unweighted datasets are covered exhaustively;
validation/test sizing is unchanged. Batch rampup, training phases, and automatic
batch reduction are unsupported. LR settings use iteration-based flags; the
schedule length is derived before optimizer setup. Consumed-sample counters
include dummy slots, and resuming with the same data/config continues at the
saved offset. Without this option, existing sample/iteration behavior remains.
