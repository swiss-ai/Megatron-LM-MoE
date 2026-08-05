# Omnimodal tokenizer metadata: the config contract

What Megatron expects in a tokenizer's `tokenizer_config.json`, and how it is consumed.
Canonical parser (and in-code spec):
`megatron/core/tokenizers/utils/tokenizer_extra_metadata.py`.
Producer: [apertus-omni-tokenizer](https://github.com/swiss-ai/apertus-omni-tokenizer).

Extraction is **HuggingFace-only**: Megatron walks its tokenizer wrapper chain to the
underlying `transformers` tokenizer and reads `init_kwargs` (the extra keys of
`tokenizer_config.json`) plus the live special-token surface. Non-HF tokenizers
(sentencepiece, tiktoken, byte-level, null) carry no extra metadata — an error when
goldfish loss is enabled, an info log otherwise. Both transformers 4.x and 5.x runtimes
are supported; note that artifacts *saved* with transformers 5.x cannot be loaded by a
4.x runtime (5.x writes `tokenizer_class: TokenizersBackend`).

## Keys

Top level of `tokenizer_config.json`:

| Key | Required | Meaning |
|---|---|---|
| `base_vocab_size` | for omni tokenizers | text vocab size (exclusive end of text ids) |
| `omnimodal_config` | no (absent ⇒ non-omni, everything below is skipped) | the payload |

Inside `omnimodal_config` (no other keys allowed):

| Key | Required | Meaning |
|---|---|---|
| `omni_special_token_offset` | yes | must equal top-level `base_vocab_size` |
| `modalities[*].name` | yes | unique modality name (`"vision"`, `"audio"`, ...) |
| `modalities[*].offset` + `vocab_size` | yes | content block; `token_id = offset + codebook_index`; `offset >= base_vocab_size`, ranges must not overlap |
| `modalities[*].start_token` / `end_token` | yes | boundary anchor ids; must appear in `structure_token_ids` |
| `modalities[*].structure_token_ids` | yes | THIS modality's full structure-token name → id map |

Validation at extraction (startup, all ranks): strict field whitelists and integer
types; content ranges ordered and non-overlapping (modalities are canonicalized into
offset order); structure-token ids may live in the base vocab or a separate omni
special range but never inside a content range, and must lie inside the declared
layout. Legacy Apertus 1.5 configs (no per-modality `structure_token_ids`) are
**rejected by design** — correct multimodal masking cannot be derived from ranges
alone.

The model's full special-token id set is *not* taken from this config: it is read from
the HF tokenizer itself (`all_special_ids` ∪ added tokens flagged special), filtered to
ids below `base_vocab_size`, then unioned with every modality's structure-token ids.
Omni content tokens are excluded even though artifacts flag them special.

## Consumers

- `args.tokenizer_extra_metadata` (`TokenizerExtraMetadata`, populated once in
  `set_global_variables`) — the single access path; forwarded to dataloader workers via
  `GPTDatasetConfig.tokenizer_extra_metadata`. `.special_tokens` always present,
  `.omni` (`OmniMetadata`) only for omni tokenizers.
- **Goldfish exemption** (planned) ← full special-token set (`ModelSpecialTokens.full_ids`).
- **Per-modality loss weighting** (`--vision-weight`/`--audio-weight`) ← content range
  + `ModalityInfo.structure_ids`; see `docs/modality_loss_weighting.md`.
- **Per-modality loss tracking** (`vision loss`/`audio loss`/`text loss` metrics) ←
  same classification, `modality_loss_report` in `pretrain_gpt.py`.
