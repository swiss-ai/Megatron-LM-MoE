# Top-k logit dumps for distillation

During training we can save the model's top-k output logits for every token, to use as teacher
targets later. The code is in `megatron/core/transformer/topk_logit_saving.py` (branch
`feat/save-topk-logits`). `tools/decode_topk_logits.py` reads the files with numpy only.

## What is saved

For every token of a saved step (including loss-masked ones):

| Field | Type | Meaning |
|---|---|---|
| `indices` | int64 `[T, k]` | token ids of the top-k, ascending within a token |
| `logits` | float32 `[T, k]` | their logits, in the same order (see Precision) |
| `lse` | float32 `[T]` | log-sum-exp over the **whole** vocabulary (log-prob = `logit - lse`) |
| `labels` | int32 `[T]` | the target token at that position |
| `position` | int32 `[T]` | position inside the sequence (0 to seq-1) |
| `sample` | int32 `[T]` | `microbatch * batch + index in the microbatch`, counted per rank and step |
| `loss_valid` | bool `[T]` | True if the token counts in the training loss |

`k` defaults to 256. The `top1` logit is stored separately (float32) and `logits` are rebuilt from it.

## Where and how files are named

`<--save-topk-logits-dir>/iter_<step>/rank_<dp>.npz`

- One file per step per data-parallel rank (`<dp>` includes context parallelism). At world size
  2048 (TP2 / PP16 / EP16) there are 64 ranks, so 64 files per step, about 0.5 GB each.
- Only the last pipeline stage's TP rank 0 writes.
- A rank's rows are its microbatches in order. Within a microbatch the rows are
  sequence-major, so `sample == s` gives one sample's tokens in order with `position` 0 to seq-1.

## Which steps are saved

`--save-topk-logits-interval N` starts a window at steps 1, N+1, 2N+1, ...
`--save-topk-logits-tokens T` sets the window length to `ceil(T / (global_batch_size * seq_length))`
whole iterations, capped at N. Every token of those iterations is saved.

At GBS 6144 and sequence length 8192, one iteration is 50,331,648 tokens, so 30B tokens is a
window of 597 iterations.

**Windows written so far (fill in):**

- TODO: the interval and tokens used, and the first iteration written with format version 2.

## Reading

```python
from decode_topk_logits import load              # tools/decode_topk_logits.py

d = load("iter_0083001/rank_00000.npz")          # rows=N reads only the first N tokens
d["indices"], d["logits"], d["lse"], d["labels"]
d["position"], d["sample"], d["loss_valid"]
log_prob = d["logits"] - d["lse"][:, None]
```

or, from a shell, a summary of a directory or file:

```bash
python tools/decode_topk_logits.py /ritom/datacache/cscs/swissai/infra01/topk-logits
```

Files are large; use `rows=` for a quick look. `decode()` in `topk_logit_saving.py` returns the same
fields if you already have Megatron's environment.

### Format versions

- **Version 1** (earlier iterations, for example 79501 to 79910) has no `position`, `sample` or
  `loss_valid`, and **loss-masked tokens are missing**. The reader returns `None` for the missing
  fields. These files cannot be aligned with the data by themselves.
- **Version 2** (current) has all fields and every token.

## Precision and size

- **Exact:** the token ids, `top1`, `lse` and the labels.
- **Lossy:** the other logits are uint8 fixed-point below the top-1 logit, in steps of
  `clip / 255` (default clip 16, so about 0.063 nats). The error is at most `clip / 510`
  (about 0.031 nats) for logits within `clip` nats of the top-1. Logits further below are clipped
  to `clip`. That is a 1.03x relative change in probability at worst; it is not a float format
  (not fp8).
- **Ids** are stored with Elias-Fano coding: the sorted ids, as low bits plus a unary-coded high
  part, about 1.44 bytes per id at k = 256 and a 200k vocabulary.
- **Truncation:** only the top-k is kept. The mass outside is `1 - sum(exp(logit - lse))`.
- **Size:** 646 bytes per token (256 `q`, 369 ids, 21 bytes of `top1`, `lse`, label, `position`,
  `sample`, `loss_valid`). That is about 32.5 GB per iteration and about 19.4 TB per 597-iteration
  window.
- **Cost while saving:** about 6% more time per iteration inside a window, measured on version 1
  (40.7 s against 38.2 s).

## Using it for distillation

```python
t_logp = torch.from_numpy(d["logits"] - d["lse"][:, None])      # teacher log-probs [T, k]
ids = torch.from_numpy(d["indices"])
s_logp = torch.log_softmax(s_logits.float(), -1).gather(1, ids)  # student at the same ids
p = t_logp.exp()
t_tail = (1 - p.sum(1)).clamp_min(1e-12)                         # mass outside the top-k
s_tail = (1 - s_logp.exp().sum(1)).clamp_min(1e-12)
kl = (p * (t_logp - s_logp)).sum(1) + t_tail * (t_tail.log() - s_tail.log())
loss = kl[d["loss_valid"]].mean()                                # mirror the training loss mask
```

This treats everything outside the top-k as one bucket. Drop the tail term to ignore it.

### Rebuilding the input

The labels are tokens 1 to seq of each sample. The input at position `p` is the label at `p-1`.
The input at position 0 is **not** saved.

## Caveats

- **The teacher moves.** These logits are from the training forward pass at that step, before the
  update on that batch. Different steps are different teachers; they are not the final model.
- **Document boundaries are not saved.** The run uses inter-document masking, so attention was
  reset at document boundaries when these logits were computed. To reproduce them, a student needs
  the same boundaries. If documents are separated by an end-of-document token, you may be able to
  infer them from the labels. This has not been checked for the Apertus tokenizer.
- **Sample order.** `sample` is a counter within a rank and step. It does not give the dataset
  index; mapping it to the dataset needs the sampler's order for that iteration and rank.
- **Padded vocabulary.** Token ids are in the padded vocabulary.

## Launching

Add to the training command (`EXTRA_NETWORK_ARGS` in the size script):

```bash
--save-topk-logits-dir /ritom/datacache/cscs/swissai/infra01/topk-logits
--save-topk-logits-interval 2000
--save-topk-logits-tokens 30000000000
```

`--save-topk-logits-k` (default 256) and `--save-topk-logits-clip` (default 16.0) are optional.
Saving is skipped, with a warning, with CUDA graphs on, MoE EP overlap on, or the TE cross-entropy
implementation. The background writer has no backpressure: if the filesystem cannot absorb about
0.5 GB per rank per iteration, queued writes accumulate in host memory.

## Tests

```bash
python tests/unit_tests/transformer/test_topk_logit_saving_cpu.py
```
