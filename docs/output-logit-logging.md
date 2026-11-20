# Output-logit diagnostics

Opt in to per-step GPT output statistics with:

```bash
--output-logit-log-interval 1
```

The default is `0` (disabled). Set `200` to sample every 200 steps instead.
These metrics are independent of `--activation-norm-log-interval` and are sent
through the existing W&B/TensorBoard scalar logging path.

## Metrics and definitions

All logit statistics refer to the primary language-model output, not MoE router
logits or MTP heads. Only tokens whose loss mask is positive are included. Thus
padding, prompt-only tokens, and Goldfish-dropped tokens do not artificially
inflate the zero-loss fraction. Fractional positive mask weights count as one
token, rather than reweighting the diagnostic distribution.

| Metric (`output_logits/` prefix) | Definition |
| --- | --- |
| `max` | Maximum output logit over valid tokens and all TP vocabulary shards |
| `lse_mean`, `lse_max` | Mean/maximum per-token `log(sum(exp(logits)))` |
| `zero_loss_fraction` | Fraction of valid tokens whose actual FP32 CE loss equals exactly zero, before training loss scaling/masking |
| `target_margin_p999`, `target_margin_max` | Exact pooled 99.9th percentile/maximum of `z_target - logsumexp(z_non_target)`, the margin in Barron et al. |
| `target_margin_gt_fp32_cliff_fraction` | Fraction of valid tokens with target margin greater than `24*ln(2)` (approximately 16.636), the theoretical FP32 rounding cliff |
| `final_hidden_norm_mean`, `final_hidden_norm_max` | Mean/maximum per-token L2 norm of the final hidden vector supplied to the output projection (after the decoder final norm) |

Means are token-weighted across microbatches and DP/CP ranks. TP replicas count
only once for vocabulary statistics. Final hidden states are SP-local when
sequence parallelism is active; otherwise only TP rank zero counts them.
MTP/unscoped cross-entropy calls and backward recomputation are excluded. Every
rerun clears the previous attempt's diagnostics.

The margin follows Equation (2) of [A Fix for Softmax and Cross-Entropy](https://arxiv.org/html/2610.31642v1).
It uses the actual label, so confidently incorrect predictions have negative margins.
It replaces the previous `top1_top2_margin_*` metrics with new keys: their historical
values are not comparable. The cliff fraction is a theoretical risk indicator,
not a measurement of backward gradients; actual CE rounding depends on the reduction.
`zero_loss_fraction` measures the actual training CE. This branch now includes the
native fused/unfused CE correction from [NVIDIA/Megatron-LM #8477](https://github.com/NVIDIA/Megatron-LM/pull/8477):
non-target exponentials are summed separately, loss uses `softplus`, and the target
gradient uses `-sum_exp_others / sum_exp_logits`. This correction is active even
when diagnostics are disabled. Consequently, zero-loss values before and after
the fix are not directly comparable. The target-margin cliff fraction still
measures exposure to the old formula's rounding regime; a nonzero fraction does
not mean the corrected CE is collapsing. TE CE and attention are unaffected.

## Cost and precision

Native fused and unfused vocabulary-parallel CE already produce FP32 logits,
global token maxima, and global exponential sums. Diagnostics reuse those
values before CE overwrites logits. To measure the paper's margin even when CE
has rounded its denominator to one, diagnostics separately exclude the target
and make two vocabulary passes: a non-target maximum, then a shifted exponential
sum. Three TP reductions combine target logits, non-target maxima, and sums;
only token vectors cross the network. Each scratch tile is bounded to about 32 MiB
or one vocabulary row if larger. This costs more than the old top-two gap and
only runs on sampled steps. LSE and zero-loss statistics reuse CE intermediates.

No full logits or hidden states are retained. Each output-stage TP0 rank retains
one FP32 margin per local token until step end. To compute an **exact** pooled
p99.9, it sends only its largest `K=ceil(0.001*(N-1))+1` candidates (padded to K)
to the output-stage DP/CP root. That root gathers at most `K * DP * CP` FP32
values and merges their upper tails. This avoids a full global margin gather,
but the percentile has additional communication/workspace cost; per-step
logging is not free. Two small world reductions pool sums and maxima and make
the results available on the logging rank.

For GBS6144, sequence8192 and DP96/CP1, N is at most 50,331,648 tokens, K=50,333.
The root gather buffer is about 18.4 MiB, with further concatenation/top-k
workspace. Local margin retention is at most about 2 MiB for TP0. Loss masking
can lower N. These estimates do not include kernel-specific top-k workspace.

Logging is disabled with a one-time warning for TE CE, CUDA graphs, or MoE EP
communication-overlap schedules: those paths require separate integration to
observe their internal intermediates/split execution safely. The current
implementation supports standard GPT native/unfused CE with CPU expert
offloading. Diagnostics do not modify the corrected loss or its gradients.

## Validation

Run inside the training container:

```bash
python tests/unit_tests/transformer/test_output_logit_logging_cpu.py
```

Standalone CPU tests cover the native and unfused production CE function bodies,
loss/gradient invariance including real TP2 collectives, masking, zero losses,
wrong targets, tied logits, large margins, hidden norms, rerun/scope reset,
exact tail percentiles, and two-process Gloo TP/DP reductions. CUDA/NCCL,
offloading integration and throughput additionally need cluster validation.
