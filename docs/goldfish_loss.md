# Goldfish loss

`--goldfish-loss` enables Goldfish loss ([Hans et al., 2024](https://github.com/ahans30/goldfish-loss))
during pretraining: for each position, a hash of the window of `--goldfish-h` labels
ending at it (order-sensitive dot product with fixed-seed odd int64 coefficients, mod a
fixed prime, looked up in a fixed-seed random table) decides whether the position is
dropped from the loss with probability `1/--goldfish-k`. Because the decision is a pure
function of the local token context, the same token in the same context is always
dropped — verbatim memorization is mitigated while training stays fully deterministic
and reproducible. (The coefficient hash replaces the original product hash, whose id-0
and id-1 labels degenerate the window key.)

Implementation: `megatron/core/datasets/gpt_dataset.py` (`apply_goldfish`, called from
`GPTDataset.__getitem__`). Only `loss_mask` is zeroed at dropped positions; the labels
reaching the model are unchanged. Since it acts per-sample at the dataset level, it
composes with both the dense-batch path and THD/`--use-packed-seq-params` packing.

Flags:

- `--goldfish-loss` — master switch.
- `--goldfish-k` (default 50) — drop probability is `1/k`; must be >= 2.
- `--goldfish-h` (default 50) — context width (tokens hashed); must be in
  `(0, seq_length)`. The first `h-1` positions of a sample are never dropped.

## Special-token exemption

Special tokens (text control tokens and modality structure tokens) must never be
dropped from the loss. The exempt set is the model's **full** special-token id set
(`ModelSpecialTokens.full_ids` in
`megatron/core/tokenizers/utils/tokenizer_extra_metadata.py`), applied through a
boolean lookup table in `apply_goldfish`. Extraction is HuggingFace-only and reads the
underlying HF tokenizer directly:

- text-only tokenizers: `all_special_ids` ∪ added tokens flagged special;
- omni tokenizers: the same set filtered to ids below `base_vocab_size`, plus every
  modality structure-token id. Modality *content* tokens stay Goldfish-eligible even
  though omni artifacts flag them special.

The metadata is read from the tokenizer once per process during initialization
(`set_global_variables`, right after the global tokenizer is built) onto
`args.tokenizer_extra_metadata`, and carried to dataloader workers via
`GPTDatasetConfig.tokenizer_extra_metadata`. Fail-loud rules:

- goldfish + a non-HuggingFace-backed tokenizer (sentencepiece/tiktoken/byte-level/
  null) → error at startup (the exempt set would be silently empty);
- an `omnimodal_config` whose tokenizer declares no special tokens → error at startup;
- a text-only HF tokenizer with no special tokens → runs without exemptions, warning
  logged at dataset build.

Set `GOLDFISH_EXEMPT_LOG=1` to log a sample of drops cancelled by the exemption.

## Possible later extensions

- **Manual extra exemption ids** (`--goldfish-extra-exemption-ids`): a CLI list of
  additional token ids to union into the exempt set, for tokenizers whose special ids
  are not fully declared. This existed during development and was removed in favor of
  acting purely on the tokenizer-declared special-token ids; the LUT-based exemption in
  `apply_goldfish` already accepts arbitrary id sets, so re-adding it is only a matter
  of restoring the flag, a `GPTDatasetConfig` field, and the union in
  `GPTDataset.__init__` (plus validating ids `< vocab_size` to avoid an out-of-range
  LUT index).
- Per-modality exemption policies (e.g. exempt only structure tokens but allow drops on
  content tokens of a specific modality) via name-based lookups from the tokenizer.
