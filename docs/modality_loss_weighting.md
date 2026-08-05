# Per-modality loss weighting

Static per-modality loss weights with the same flag interface as the dense Apertus
Megatron: `--vision-weight W` and `--audio-weight W` (default 1.0 = normal loss;
`0.0` fully masks the modality). Weights are fixed for the run.

## How tokens are classified (id-based, not span-based)

The vocabulary of an omni tokenizer is partitioned by modality, so classification is a
pure function of the label id: a modality owns its **content range**
`[offset, offset + vocab_size)` plus its **structure tokens**
(`ModalityInfo.structure_ids`: `<|img_start|>`, `<|image|>`, ...). Both get the
modality's weight via a float LUT over the vocab, applied in `GPTDataset.__getitem__`
as `loss_mask = loss_mask * lut[labels]` (`_create_modality_weight_lut` in
`megatron/core/datasets/gpt_dataset.py`; module-memoized per (weights, vocab, device);
wired via `GPTDatasetConfig.modality_weights`, entries at 1.0 omitted).

Because membership never depends on position, samples that start or end mid-modality
are weighted correctly without start/end anchors (the old implementation weighted
content ranges only), and placeholder `<|image|>`/`<|audio|>` occurrences in text
context carry their modality's weight by the same rule.

## Separate per-modality loss tracking

For omni tokenizers the reported loss is additionally decomposed per category
(`modality_loss_report` in `pretrain_gpt.py`, no flag needed): each modality
(content + structure ids) and `text` (every remaining label id). Three metrics per
category, all in the `lm loss` `[numerator, denominator]` convention:

| Metric                 | Pair                                                                                                            | Reads as                                                                                                                                     |
|------------------------|-----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `<name> loss`          | `[weighted sum, weighted count]`                                                                                | Mean loss over *supervised* tokens. The modality weight cancels in the ratio, so this is the true per-token loss whatever `--{m}-weight` is. |
| `<name> weighted loss` | `[weighted sum, raw count]`, or `[weighted sum, supervised count]` under `--normalize-by-num-supervised-tokens` | The category's actual contribution to `lm loss`, spread over its tokens. Scales linearly with the weight.                                    |
| `<name> error`         | `[raw sum, raw count]`                                                                                          | Ignores loss mask and weight entirely. **The only one that stays meaningful at weight `0.0`.**                                               |

Categories partition the vocabulary, so the `<name> loss` sums add up exactly to the
`lm loss` sum. Denominator additivity depends on the mode: by default the
`<name> loss` counts sum to `lm loss`'s denominator (up to int truncation with
fractional weights); under `--normalize-by-num-supervised-tokens` it is the
`<name> weighted loss` pairs that sum to `lm loss` exactly.

Denominators are clamped to ≥ 1 in the per-step reduction (`training.py`), so an
empty category reports `0.0`, not NaN. A modality weighted to `0.0` has zero weighted
count, so its `<name> loss` is a constant `0.0` — track it via `<name> error`.

## Requirements & validation

- Requires an omni tokenizer: non-default weights fail `GPTDatasetConfig` validation
  without matching `tokenizer_extra_metadata.omni`.
- Weights must be ≥ 0 (`validate_args`); modality layout is validated at metadata
  extraction (see `docs/omnimodal_metadata.md`), not re-validated by the LUT builders.
- Composes with goldfish loss (scaling commutes with drop-zeroing).
- Pretraining path only: rejected with `--sft` (SFTDataset builds its own loss mask);
  `pretrain_mamba.py` accepts the modality-weight flags but does not apply them (it
  does honor `--normalize-by-num-supervised-tokens`).

## Loss normalization: `--normalize-by-num-supervised-tokens`

Switches `loss_func`'s `num_tokens` — the `lm loss` denominator and, under
`--calculate-per-token-loss`, the gradient normalizer — from `loss_mask.sum()` to
`(loss_mask > 0).sum()`, the true supervised-token count, in both `pretrain_gpt.py`
and `pretrain_mamba.py`. When is the reported `lm loss` the objective the gradients
optimize, and what is it normalized by?

| Configuration                                          | Reported ≡ optimized?                                                                    | Normalizer                                     |
|--------------------------------------------------------|------------------------------------------------------------------------------------------|------------------------------------------------|
| 0/1 mask, neither flag                                 | Only if every microbatch has the same supervised count (full-length, no pad/eod masking) | per-microbatch count                           |
| 0/1 mask + `--calculate-per-token-loss`                | Yes, exact                                                                               | true token count (this flag is a no-op)        |
| Fractional weights, neither flag                       | No — and the weight partially cancels out of the gradients                               | per-microbatch `Σw`                            |
| Fractional weights + `--calculate-per-token-loss` only | Yes (report and gradient share the same tensors)                                         | `Σw`, int-truncated — not a per-token quantity |
| Fractional weights + both flags                        | **Yes, exact**                                                                           | **true supervised-token count**                |

**With fractional weights, run both flags.** `--calculate-per-token-loss` makes
reported and optimized loss identical: the raw weighted sum goes backward, gradients
are summed across DP and divided once by the globally reduced `num_tokens` in
`finalize_model_grads` — the very tensors the report uses. This flag makes that
shared normalizer the actual token count. Without either flag, each microbatch is
divided by its own weighted sum, so the weight largely cancels out of the gradients
(a 100% vision microbatch at weight 0.5 produces the same gradients as weight 1.0);
with `--calculate-per-token-loss` alone the objective is `Σ(w·l)/Σw`, inflated by
`N/Σw` relative to a per-token loss. `validate_args` warns if this flag is set
without `--calculate-per-token-loss`.

Fine print:

- **0/1 masks**: objective and `lm loss` are identical with or without the flag;
  text-only runs (no per-category report) are a strict no-op. Omni runs still see the
  `<name> weighted loss` denominators switch to supervised counts wherever the mask
  has zeros (eod masking, padding, goldfish drops).
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