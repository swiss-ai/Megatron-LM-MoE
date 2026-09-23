# Swiss Apertus2 / Bridge compatibility

This companion branch starts from Swiss Megatron-LM `96b9751981de6a259baa8d5bce471416f4c362a2`.
Bridge `fd1638f582e4d058962075a32225a8b97b86ff64` expects upstream model-builder
and config APIs absent from that fork. The backport source is its pinned
NVIDIA Megatron-LM commit `6513e3e23d6b5eda6a1c934990b15e804237732b`.

Backported components are the `training/models` builders, `vocab_utils`,
configuration instantiation/serialization helpers, `TokenizerConfig`, rank and
Slurm helpers, the shared CUDA capture stream accessor, and the MSC wrapper.
The existing training `utils.py` is retained verbatim as `utils/__init__.py`
with the new logging/local-rank helper modules beside it. Hybrid API aliases
delegate to this fork's Mamba implementation. Existing Swiss attention, MoE,
activation, normalization, and checkpoint implementations remain authoritative.

Local adaptations:

- KDA accepts explicit `a_log_per_channel` and `output_gate_bias`; omitted
  options preserve the existing environment/default behavior.
- Distributed wrapping preserves marked FP32 parameters and router buffers
  before the BF16/FP16 cast, including their original values.
- Parameter-count logging tolerates the absent upstream `gtp_remat` group.

Validation exercises Bridge imports, configuration/YAML conversion, actual
native model construction, weight conversion and hybrid forward parity at
TP=1/2, and KDA backward at TP=1/2. Full Megachonk loading and HF export also
passed on 32 GPUs with TP=1, PP=8 and EP=4, with exact comparisons of all
46,051 exported tensors. See the repository-root `BRIDGE_COMPATIBILITY.md`
for the tested Bridge revision, native-offloading checks and validation scope.
The matching Bridge Apertus2 README contains the reproducible GPU commands.
