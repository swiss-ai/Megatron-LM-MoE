# Apertus 2 Bridge compatibility

This branch starts at Swiss Megatron-LM
`96b9751981de6a259baa8d5bce471416f4c362a2` and supports the companion Bridge
`apertus2/kda_channels` changes for per-channel KDA checkpoints.

The companion native branch is `apertus2/bridge-compat`. The tested Bridge
revision is `b6b1857ea5b2e06b93525805c04631ee13ee6583` in
`mac-mvak/Megatron-Bridge`.

`KimiDeltaAttention` accepts optional boolean `a_log_per_channel` and
`output_gate_bias` arguments. Explicit values override legacy defaults. Omitting
`a_log_per_channel` retains the `KDA_ALOG_PER_CHANNEL` behavior; omitting
`output_gate_bias` retains the existing general bias setting. The dedicated gate
bias flag affects only `gate_out_proj`, leaving other projections unchanged.

Bridge's current configuration/build API requires interfaces absent from this
Swiss revision. The accompanying compatibility files originate from NVIDIA
Megatron-LM `6513e3e23d6b5eda6a1c934990b15e804237732b`, the revision pinned by
the reviewed Bridge branch:

- `megatron/training/models/` and `vocab_utils.py`: model configurations,
  builders, and distributed wrapping.
- `megatron/training/config/`: serialization/container helpers and tokenizer
  configuration, preserving existing Swiss training configuration fields.
- Rank, Slurm, logging, storage-client and shared CUDA capture-stream helpers.

The original `megatron/training/utils.py` is retained as the new package's
`__init__.py`. Hybrid model/spec names adapt to this fork's existing Mamba
implementation. Distributed wrapping tolerates the absence of the upstream
`gtp_remat` group and preserves explicitly marked FP32 parameters plus FP32
router buffers before a recursive BF16/FP16 cast. These are deliberate
adaptations to the Swiss implementation, not a merge of upstream model math.

Validation covers the Bridge config/build/serialization utilities, native KDA
reference and gradient tests, and small native/HF Apertus 2 hybrid models with
TP1 and TP2. The Bridge README describes the commands.

## Full checkpoint validation (2026-09-23)

- The complete `megachonk_iter_0000008` distributed checkpoint loaded through
  Bridge on 32 GPUs with TP=1, PP=8 and EP=4. All 593,659,004,544 parameter
  elements were on GPUs and passed finite-value checks.
- The loaded model exported to 239 HF safetensors shards containing 46,051
  tensors. Every exported tensor matched the independent hfconverter export
  exactly in shape and value. The 90 KDA decay tensors were promoted losslessly
  from BF16 to FP32; other tensor dtypes matched the reference.
- Exported modeling/configuration code matched hfconverter byte-for-byte.
- A separate complete load with native offloading experts also passed. The
  native offloading implementation and its `weight1`/`weight2` checkpoint
  format were not changed.
- Small offloading checkpoint round trips passed with EP=1/2 and PP=2 in both
  fine-grained and coarse-grained modes, including exact HF comparisons.

Other model families, full-model inference/training, optimizer restoration,
FSDP execution and native cached KDA inference were not validated. Temporary
GPU expert merges used the existing CPU fallback when GPU memory ran out;
every final parameter in the Bridge GPU load remained on a GPU.
