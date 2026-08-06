# Per-modality loss weighting

Static per-modality loss weights with the same flag interface as the dense Apertus
Megatron: `--vision-weight W` and `--audio-weight W` (default 1.0 = normal loss;
`0.0` fully masks the modality). Weights are fixed for the run.

## Masking approach

Classification is ID-based, not span-based. A modality owns its content-token range
`[offset, offset + vocab_size)` and every ID in its `structure_token_ids` map. All
other IDs are text. This handles samples cut through a modality span and assigns
placeholder tokens such as `<|image|>` to their declared modality without scanning for
start/end pairs.

For prediction position `i`, the effective mask is:

```text
effective_loss_mask[i] = base_loss_mask[i] * modality_weight(labels[i])
```

The target `labels[i]`, rather than the input token, determines which prediction is
weighted. Existing zeros from padding, EOD masking, or another masking stage remain
zero.

## Technical flow

| Stage | Implementation |
|---|---|
| Parse tokenizer layout | [`tokenizer_extra_metadata.py`](../megatron/core/tokenizers/utils/tokenizer_extra_metadata.py) reads `omnimodal_config`, validates disjoint content/structure IDs, and produces ordered `ModalityInfo` objects. The full schema is documented in [`omnimodal_metadata.md`](omnimodal_metadata.md). |
| Build dataset config | [`pretrain_gpt.py`](../pretrain_gpt.py), `core_gpt_dataset_config_from_args`, forwards tokenizer metadata and only non-default weights into `GPTDatasetConfig`. |
| Classify token IDs | [`modality_lut.py`](../megatron/core/tokenizers/utils/modality_lut.py), `_create_modality_index_lut`, builds an int8 vocabulary LUT: `0` is text and `k` is the one-based index of a modality. Both content and structure IDs are assigned. |
| Convert indices to weights | `_create_modality_weight_lut` maps index `0` and unweighted modalities to `1.0`, then maps configured modalities to their weight. `_LUT_CACHE` shares immutable LUTs per process by layout, weights, vocabulary size, and device. |
| Apply the mask | `GPTDataset.__getitem__` first creates the normal loss mask and clears padding. It lazily obtains the weight LUT and multiplies it with `weight_lut[labels]`. Padding labels become `-100` after lookup; batch-padding samples are fully ignored. |
| Consume the mask | [`pretrain_gpt.py`](../pretrain_gpt.py), `loss_func`, computes `sum(token_loss * loss_mask)`. Non-binary weights require supervised-token and per-token normalization so the configured weight is not cancelled by the denominator. |
| Optional reporting | With `--log-per-modality-loss`, `modality_loss_report` uses the same modality-index LUT on the training device, with `args.padded_vocab_size`, so reporting and dataset masking use identical token ownership. |

Dataset LUTs are created lazily in each dataloader worker. Training ranks create only
the modality-index LUT needed for optional reporting. Static weights belong in the cached
weight LUT; dynamic schedules would instead apply a small modality-to-weight vector at
training time.

## Separate per-modality loss tracking

With `--log-per-modality-loss`, omni tokenizers additionally report each modality
([`modality_loss_report`](../pretrain_gpt.py); content + structure ids) and `text`
(every remaining label id). The flag is off by default because these metrics add
full-tensor reductions per microbatch. Two metrics are emitted per token group in
the `lm loss` `[numerator, denominator]` convention:

| Metric         | Pair                                    | Reads as                                      |
|----------------|-----------------------------------------|-----------------------------------------------|
| `<name> loss`  | Same weighted sum and denominator rule as `lm loss` | The group's contribution under the active normalization mode. |
| `<name> error` | `[raw sum, raw token count]`            | Unmasked, unweighted mean cross-entropy over valid targets. |

Without `--normalize-by-num-supervised-tokens`, `<name> loss` divides by the
loss-mask sum (cast to an integer), matching `lm loss`; supported weights in this
mode are `0.0` and `1.0`. With the flag, it divides by the number of positions
whose loss mask is greater than zero, so a fractional modality weight scales the
numerator but not the denominator. `<name> error` ignores both rules. The group
loss numerators and denominators add up to the `lm loss` pair for the supported
configurations. Targets labeled `-100` are excluded from error reporting; this
covers GPT/SFT padding and ignored SFT targets.

Denominators are clamped to ≥ 1 in the per-step reduction (`training.py`), so an
empty group reports `0.0`, not NaN. A modality weighted to `0.0` has zero weighted
count, so its `<name> loss` is `0.0` — track it via `<name> error`.

## Requirements & validation

- Requires an omni tokenizer: non-default weights fail `GPTDatasetConfig` validation
  without matching `tokenizer_extra_metadata.omni`.
- Weights must be finite and ≥ 0 (`validate_args`). Any weight other than `1.0`
  requires `--calculate-per-token-loss`; non-binary weights additionally require
  `--normalize-by-num-supervised-tokens`. Modality layout is validated at metadata
  extraction (see `docs/omnimodal_metadata.md`), not re-validated by the LUT builders.
- Pretraining path only: rejected with `--sft` (SFTDataset builds its own loss mask);
  modality weighting and reporting are implemented by `pretrain_gpt.py` only.
- ModelOpt rejects supervised-token normalization and per-modality loss logging.

## Loss normalization: `--normalize-by-num-supervised-tokens`

Switches `loss_func`'s `num_tokens` — the `lm loss` denominator and, under
`--calculate-per-token-loss`, the gradient normalizer — from `loss_mask.sum()` to
`(loss_mask > 0).sum()`, the true supervised-token count, in `pretrain_gpt.py`.
When is the reported `lm loss` the objective the gradients optimize, and what is it
normalized by?

| Configuration                                          | Reported ≡ optimized?                                                                    | Normalizer                                     |
|--------------------------------------------------------|------------------------------------------------------------------------------------------|------------------------------------------------|
| 0/1 mask, neither flag                                 | Only if every microbatch has the same supervised count (full-length, no pad/eod masking) | per-microbatch count                           |
| 0/1 mask + `--calculate-per-token-loss`                | Yes, exact                                                                               | true token count (this flag is a no-op)        |
| Non-binary weights + both flags                        | **Yes, exact**                                                                           | **true supervised-token count**                |

**Non-binary weights require both flags.** `--calculate-per-token-loss` makes
reported and optimized loss identical: the raw weighted sum goes backward, gradients
are summed across DP and divided once by the globally reduced `num_tokens` in
`finalize_model_grads` — the very tensors the report uses. This flag makes that
shared normalizer the actual token count. `validate_args` rejects non-binary weights
without this pair, avoiding per-microbatch weight cancellation and integer truncation.
Weight `0.0` requires `--calculate-per-token-loss` on its own (the normalize flag is
a no-op for binary masks): without it, contiguous zero-weight modality spans make
the per-microbatch and per-CP-rank local normalization redistribute gradient weight
onto the surviving tokens.

Fine print:

- **0/1 masks**: objective and `lm loss` are identical with or without the flag;
  text-only runs (no per-group report) are a strict no-op. Per-modality loss uses
  the same denominator switch as `lm loss`.
- **Weight `0.0`** behaves like padding in both modes (drops from numerator and
  denominator), but the flag makes `w → 0` discontinuous: at any `w > 0` the tokens
  count fully in the denominator, at exactly `0` they vanish. Anneal to a small ε
  rather than `0` if a continuous ramp-down matters.
- The int cast of `num_tokens` becomes exact (counts, not truncated fractional sums).
  Under `--calculate-per-token-loss` the balance of MoE aux/z-loss and MTP losses
  against the main loss is untouched (one uniform global rescale); in default mode
  the flag shifts that balance with fractional weights, since aux/MTP keep their own
  normalizers — another reason to pair the flags.

## Possible later extension

Dynamic weight schedules (the old repo's `--{m}-weight-decay` family): a per-step
weight cannot be baked into prefetched dataset samples, so it would be applied in
`forward_step`/`loss_func` on top of the static dataset-side weighting kept here.
