# Per-modality loss weighting

Static per-modality loss weights with the same flag interface as the dense Apertus
Megatron: `--vision-weight W` and `--audio-weight W` (default 1.0 = normal loss;
`0.0` fully masks the modality). Weights are fixed for the run.

## How tokens are classified (id-based, not span-based)

The vocabulary of an omni tokenizer is partitioned by modality, so classification is a
pure function of the label id:

- **content tokens**: the contiguous range `[offset, offset + vocab_size)` from
  `omnimodal_config.modalities[*]`;
- **structure tokens** (`<|img_start|>`, `<|img_end_of_row|>`, `<|image|>`, ...): the
  values of the per-modality `structure_token_ids` map in the same modality entry
  (`ModalityInfo.structure_ids`).

Both sets get the modality's weight via a float lookup table over the vocab
(`_create_modality_weight_lut` in `megatron/core/datasets/gpt_dataset.py`), applied in
`GPTDataset.__getitem__` as `loss_mask = loss_mask * lut[labels]` (out-of-place: keeps
the cacheable loss-mask path intact). Wiring: `--{m}-weight` flags →
`GPTDatasetConfig.modality_weights` (entries at the default 1.0 are omitted). The LUT
is module-memoized per (weights, vocab, device), like the goldfish tables, so blended
runs don't pay per-blend-component copies.

Because membership never depends on position, samples that begin or end mid-modality
are weighted correctly — e.g. an image spanning ≥3 samples whose middle samples carry
no start/end anchors. (This extends the old implementation, which weighted content
ranges only.) Placeholder occurrences of `<|image|>`/`<|audio|>` in text context carry
their modality's weight, by the same id-based rule.

## Separate per-modality loss tracking

For omni tokenizers the reported loss is additionally decomposed per category
(`modality_loss_report` in `pretrain_gpt.py`, no flag needed). A category is a modality
(label in its content range or among its structure tokens) or `text` (every remaining
label id). Three metrics are emitted per category, all in the `lm loss`
`[numerator, denominator]` convention (reduced over microbatches and data-parallel
ranks), matching the set the dense Apertus Megatron reported:

| Metric | Pair | Reads as |
|---|---|---|
| `<name> loss` | `[weighted sum, weighted count]` | Mean loss over *supervised* tokens. The modality weight cancels in the ratio, so this is the true per-token loss whatever `--{m}-weight` is. |
| `<name> weighted loss` | `[weighted sum, raw count]` | The category's actual contribution to `lm loss`, spread over all its tokens. Scales linearly with the weight. |
| `<name> error` | `[raw sum, raw count]` | Ignores loss mask and weight entirely. **The only one that stays meaningful at weight `0.0`.** |

The categories partition the vocabulary (structure ids are validated disjoint across
modalities), so the `<name> loss` sums add up exactly to the `lm loss` sum; the counts
match `lm loss`'s denominator exactly for 0/1 masks and to within int truncation when
fractional weights are active.

Denominators are clamped to a minimum of 1 in the per-step reduction
(`training.py`), so a category with zero tokens in a global batch reports `0.0` rather
than NaN. Note the consequence for `<name> loss` of a modality weighted to `0.0`: its
weighted count is always zero, so it reports a constant `0.0`. Use `<name> error` to
track a fully masked modality.

## Requirements & validation

- Requires an omni tokenizer: non-default weights fail `GPTDatasetConfig` validation
  without `tokenizer_extra_metadata.omni` or when the metadata doesn't describe the
  named modality.
- `validate_args`: weights must be ≥ 0.
- Modality layout (offsets, ranges, `structure_token_ids`, cross-modality
  disjointness) is validated at metadata extraction (see
  `docs/omnimodal_metadata.md`); the LUT builders themselves do not re-validate.
- Composes with goldfish loss (scaling commutes with drop-zeroing).
- `loss_func` returns the loss sum plus `num_tokens = loss_mask.sum()` truncated to
  int; normalization happens downstream (per-step in the train loop, per-microbatch in
  the schedule). Fractional weights therefore make "lm loss" a weighted average over a
  weighted token count, exact only up to that int truncation.
- `pretrain_gpt.py` pretraining path only: rejected with `--sft` (SFTDataset builds its
  own loss mask and ignores modality weights), and `pretrain_mamba.py` accepts the
  shared flags but does not apply them.

## Possible later extension

Dynamic weight decay over training (the old repo's `--{m}-weight-decay` +
schedule/min/max/start/end flags): a per-step weight cannot be baked into prefetched
dataset samples, so it would be applied in `forward_step`/`loss_func` (overwriting
still-active modality entries with the scheduled weight each step) on top of the
static dataset-side weighting kept here.
