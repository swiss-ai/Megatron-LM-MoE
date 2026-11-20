# Activation RMS logging

Per-layer activation RMS logging to W&B and TensorBoard is disabled by default
(`--activation-norm-log-interval 0`). Enable every 200 optimizer steps with
`--activation-norm-log-interval 200`. Layer indices in metric names are zero-based
global decoder indices.

Metrics:

- `activation_norm/residual_input_rms_layer_N`: residual stream entering the layer,
  before the attention input normalization.
- `activation_norm/attention_output_rms_layer_N`: attention branch output, including
  a separate linear bias, before sandwich normalization, residual scaling/addition,
  and dropout.
- `activation_norm/mlp_output_rms_layer_N`: dense or combined MoE branch output,
  including a separate bias, at the equivalent boundary before post-normalization
  and residual addition. This is not the internal expert FC1 or gate activation.

RMS is `sqrt(sum(x**2) / number_of_valid_elements)` pooled over tokens,
microbatches, and ranks, rather than an average of per-microbatch norms. Padding
is excluded using the model's already CP/SP-local `[batch, sequence]` mask. TP
replicas duplicate numerator and denominator equally. Pipeline stages contribute
only the layers they own. Full-sequence padding masks cannot be substituted for
the local mask.

The observer uses bounded FP32 copies (about 1 MiB, or one sequence position if
larger) and retains only two FP64 scalars per layer/metric. It never stores an
activation reference or autograd graph and does not reload CPU-offloaded tensors.
It adds no tensor reductions on unsampled steps. On sampled steps one world
all-reduce pools the small scalar array, followed by a host transfer for logging.
CPU-offloaded expert weights, gradients, and activations remain supported.

Only scheduled forward callbacks enable recording, so backward recomputation
is excluded even when the original checkpoint forward runs under `no_grad`.
Every rerun clears the preceding attempt's observations. Logged values therefore
describe the last completed attempt, not all attempts added together.

CUDA graphs and MoE EP communication overlap are currently unsupported: training
skips these diagnostics and prints a warning once when either is enabled. This
avoids capturing logging kernels or reporting incomplete split-forward metrics.
MTP statistics are not part of this initial decoder-layer diagnostic.

Standalone CPU validation (requires PyTorch; no CUDA/TE imports):

```bash
python tests/unit_tests/transformer/test_activation_norm_logging_cpu.py
```

GPU/offload integration and throughput must additionally be validated on the
training cluster. Output-logit statistics are not implemented by this change.
