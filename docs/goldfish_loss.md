# Goldfish loss

`--goldfish-loss` enables Goldfish loss ([Hans et al., NeurIPS 2024, arXiv:2406.10209](https://arxiv.org/abs/2406.10209);
[reference implementation](https://github.com/ahans30/goldfish-loss))
during pretraining: for each position, a hash of the window of `--goldfish-h` labels
ending at it (order-sensitive dot product with fixed-seed odd int64 coefficients, mod a
fixed prime, looked up in a fixed-seed random table) decides whether the position is
dropped from the loss with probability `1/--goldfish-k`. Because the decision is a pure
function of the local token context, the same token in the same context is always
dropped — verbatim memorization is mitigated while training stays fully deterministic
and reproducible. (The coefficient hash replaces the original product hash, whose id-0
and id-1 labels degenerate the window key. Like the reference implementation — and
unlike the paper's prose, which describes hashing the h *preceding* tokens — the
hashed window includes the decided label itself.)

Implementation: `megatron/core/datasets/goldfish.py` (`apply_goldfish` plus the
memoized hash/exemption state), called from `GPTDataset.__getitem__` in
`megatron/core/datasets/gpt_dataset.py`. Only `loss_mask` is zeroed at dropped
positions; the labels reaching the model are unchanged. Drops apply to the **train
split only**: validation and test samples are never dropped, so eval losses stay
comparable to non-goldfish runs. Rejected with `--sft` (SFTDataset builds its own loss
mask) and applied by `pretrain_gpt.py` only — other entrypoints do not forward the
goldfish config into their dataset builds. Since it acts per-sample at the dataset
level, it composes with the dense-batch path, with pretraining packing
(`--pretraining-packing-strategy greedy|bfd`) and with
`--dataloader-inter-document-masking` (the mask is finalized before the `cu_seqlens`
return branch). In packed samples a hash window may span document boundaries: a
duplicated document's drop mask is identical wherever it lands, except its first `h-1`
positions, which vary with the packing neighbor.

Determinism caveat: the hash tables come from fixed-seed `torch.rand`/`torch.randint`
streams, which PyTorch guarantees stable only within a version/platform — upgrading
PyTorch changes the drop mask.

Flags:

- `--goldfish-loss` — master switch.
- `--goldfish-k` (default 50) — drop probability is `1/k`; must be >= 2.
- `--goldfish-h` (default 50) — context width (tokens hashed); must be in
  `(0, seq_length)`. The first `h-1` positions of a sample are never dropped.

## Special-token exemption

Special tokens (BOS/EOS/PAD/UNK, chat-role and tool delimiters, reserved control
tokens, ...) must never be dropped from the loss. The exempt set is the tokenizer's
**full** special-token id set (`ModelSpecialTokens.full_ids` in
`megatron/core/tokenizers/utils/tokenizer_extra_metadata.py`), applied through a
boolean lookup table in `apply_goldfish`. Exempt tokens still participate in the hash
of neighbouring windows; they just cannot themselves be removed from the loss.

Extraction is HuggingFace-only: Megatron walks its tokenizer wrapper chain
(`_tokenizer` / `tokenizer` attributes) to the underlying `transformers` tokenizer and
takes `all_special_ids` ∪ every entry of `added_tokens_decoder` flagged `special`.
For a plain Apertus 2 tokenizer this yields ids 0–123 (`<unk>`, `<s>`, `</s>`,
`<pad>`, the `<|...|>` chat/tool delimiters, `<think>`/`<reflection>`, PII markers, and
the reserved `<SPECIAL_n>` slots); everything else stays Goldfish-eligible.

The metadata is read from the tokenizer once per process during initialization
(`set_global_variables`, right after the global tokenizer is built) onto
`args.tokenizer_extra_metadata`, and carried to dataloader workers via
`GPTDatasetConfig.tokenizer_extra_metadata`. Fail-loud rules:

- goldfish + a non-HuggingFace-backed tokenizer (sentencepiece/tiktoken/byte-level/
  null) → error at startup (the exempt set would be silently empty);
- an HF tokenizer with no special tokens → runs without exemptions, warning logged at
  dataset build.

Set `GOLDFISH_EXEMPT_LOG=1` to log a sample of drops cancelled by the exemption.

## Possible later extensions

- **Manual extra exemption ids** (`--goldfish-extra-exemption-ids`): a CLI list of
  additional token ids to union into the exempt set, for tokenizers whose special ids
  are not fully declared; the LUT-based exemption already accepts arbitrary id sets
  (validate ids `< vocab_size` to avoid an out-of-range LUT index).
- **Multimodal tokenizers**: the omni-aware variant (filtering modality *content* ids
  out of the exempt set and adding modality structure tokens) lives on the
  `multimodality/main` branch.
