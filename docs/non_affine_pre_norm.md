# Non-affine pre-norm for MuonMD

Enable `--non-affine-pre-norm` when training with `--optimizer md_decoupling` and
`--hypersphere-gains-mode col` or `rowcol`. The launcher validates these requirements. It is
explicitly opt-in; selecting MuonMD alone does not change the architecture.

The option removes pre-norm gain and bias tensors entirely, rather than freezing
unit-valued parameters. The option requires RMSNorm, which computes
`x / sqrt(mean(x**2) + eps)`. LayerNorm's additive bias cannot be absorbed by
column gains, so LayerNorm is not enabled for this option. Column gains supply learned
input scaling. Sandwich/post norms, QK norms, KDA/GDN output norms and the final
model norm retain their affine parameters.

## Optional KDA output norm

Also enable `--non-affine-kda-output-norm` to remove the per-head RMSNorm gain
before KDA's `out_proj`. This is independent of `--non-affine-pre-norm`, and
requires KDA, RMSNorm, and MuonMD with column gains. The sigmoid output gate
commutes with the gain, so `out_proj`'s column gains can absorb it (repeated
across value heads). QK and sandwich norms remain affine.

For the fused output path, FLA's `FusedRMSNormGated(elementwise_affine=False)`
preserves norm/gate fusion and skips gain loads, multiplication and gain-gradient
reductions. This API is supported by the container's pinned FLA v0.5.2. Scalar
or disabled gate modes use the parameter-free norm with the existing gate logic.
No additional recomputation or matrix replay is introduced. GPU throughput
remains to be benchmarked; normalization and gate work remain.

Supported sites are transformer input and pre-MLP norms for standard self-attention
and KDA/GDN: either standalone, or fused into the following QKV/input/FC1 matrix.
"QKV" here identifies the input prenorm fused into the QKV projection; it does
not mean the separate query/key normalization. Cross-attention, Mamba and MLA
are outside this option's scope.
Routed and shared experts' plain linear projections remain plain linear layers.
Expert FP8 weight offload is independent of these changes.

## Backend and performance

CUDA LayerNorm/RMSNorm use parameter-free Triton forward and backward kernels,
with FP32 statistics for BF16/FP16 input. The backward computes only input
gradients: no gain/bias gradient reductions. Triton is required on CUDA; hidden
sizes up to 32768 are supported. CPU uses PyTorch without affine parameters.

Transformer Engine LayerNormLinear requires affine tensors. At the relevant
pre-norm sites this option replaces it with a parameter-free norm followed by
TE ColumnParallelLinear. Linear weight names, tensor/sequence parallelism,
FP8 linear execution and delayed weight gradients are retained. Native TE
norm-linear/quantization fusion is not retained. A throughput improvement is
not guaranteed: measure forward/backward step time on the actual GPU, including
activation recomputation and communication. TE operation fuser and fused TP
inference kernels are not supported with this option.

## Checkpoints

Use the same setting when saving and loading checkpoints. Affine pre-norm keys
are absent, and optimizer groups differ. There is no automatic conversion of
learned gains or optimizer state from an affine model. An affine checkpoint
loaded with `strict=False` discards those gains and changes the model unless
they were identity; that is not an exact resume.

## Validation

CPU tests cover forward/backward agreement, BF16, irregular hidden dimensions,
non-contiguous input, double-precision gradcheck, and empty/zero input. CUDA
numerics tests run the same comparisons against PyTorch when CUDA/Triton exist.
Transformer-layer integration tests verify parameter removal and that other
norms remain affine. GPU throughput requires a separate benchmark.
