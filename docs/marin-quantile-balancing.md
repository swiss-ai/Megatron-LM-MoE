# Marin histogram QB

Opt in with `--moe-router-quantile-balancing-method marin_histogram`.
The existing `histogram` default is unchanged. The new mode uses raw logits
for selection, retaining the configured expert combination score function.
For sigmoid weights normalized to 2.5, use:

```bash
--moe-router-load-balancing-type quantile_balancing
--moe-router-score-function sigmoid
--moe-router-topk-scaling-factor 2.5
--moe-router-quantile-balancing-method marin_histogram
--moe-router-quantile-balancing-marin-num-bins 10000
```

In the cluster size file
`/ritom/scratch/cscs/ahuang/megatron-apertus-moe/_research/launch/framework/sizes/megachonk/megachonk.sh`,
the method can be selected with:

```bash
EXTRA_NETWORK_ARGS+=(--moe-router-quantile-balancing-method marin_histogram)
```

The pretrain launcher already sets sigmoid and a 2.5 scale. It also adds sequence
auxiliary loss, which this change does not disable. `MOE_SEQ_AUX_LOSS_COEFF=0`
disables that separate objective if matching Marin's absence of balancing loss
is desired. Router z-loss settings are independent and unchanged.

## Reference and adaptation

Reference: [Marin's hero EP model at 12d8b6f](https://github.com/marin-community/marin/blob/12d8b6f/experiments/grug/moe_hero_ep/model.py),
`_qb_beta_hist` and `_bincount_upper_quantile` (lines 783–836).
This is the histogram path associated with their large-model recipe, rather than
the local-quantile Grug baseline.

Selection uses `topk(logits - beta)` (Marin uses additive `bias = -beta`).
For top-k sigmoid combination with k > 1, training and inference use
`scale * sigmoid(selected_logits) / (sum(sigmoid(selected_logits)) + 1e-9)`,
matching Marin's denominator epsilon. Other routing methods retain `1e-20`.
This reduces the total combine weight when all selected sigmoid scores are tiny;
it does not change expert selection. Existing Marin checkpoints can resume without
converting weights or optimizer state, but their tiny-score behavior changes.
The top-(k+1) cutoff defines raw-logit margins. Every layer holds one fixed grid
throughout an optimizer step and adds local integer counts for each microbatch.
At finalization, counts are summed across TP+DP+CP and one upper-tail quantile is
decoded per expert, followed by EMA and mean-centering. There are no QB forward
collectives. With identical starting beta/range and tokens, partitioning tokens
across microbatches or ranks gives identical pooled counts.

**Range approximation relative to Marin:** Marin determines its range from the
current distributed batch. Sequential gradient accumulation cannot know that
range without retaining or recomputing earlier margins. Here each layer starts
at [-32, 32] and adapts the NEXT step's grid from pooled finite extrema: include
the observed range with 10% padding, expanding immediately but contracting at
most 5% of the previous span per edge. Quantiles landing in a clipped edge bin
preserve the whole layer's previous beta while the range adapts. A nonfinite valid
token row also prevents the layer's beta/range update; padding is excluded. This
protects the QB state, not the forward output or gradients of that bad token.
All-padding steps preserve beta and range too.

W&B/TensorBoard log `qb_histogram_underflow_fraction`,
`qb_histogram_overflow_fraction` (fractions of finite token-expert margins),
`qb_histogram_nonfinite_tokens`, `qb_histogram_update_skipped`, and
`qb_histogram_range_min`/`qb_histogram_range_max` (the grid used that step).
These are step-level metrics, not divided down by the accumulation count.

With 256 experts and 10,000 bins the persistent int64 histogram is 19.5 MiB per
MoE layer; finalization temporarily stacks one additional histogram per local
layer. No full-step margins are retained. GPU memory/throughput and CUDA graph
compatibility must still be measured on the target setup.

## Checkpoints

Raw-logit and sigmoid-domain beta values are not interchangeable. Do not silently
switch a trained `histogram` checkpoint to this method: the branch does not reset
or convert its saved thresholds. Use a fresh run or an explicitly prepared
checkpoint with reinitialized QB thresholds, then assess the routing transient.
Checkpoints produced with `marin_histogram` retain the method through the existing
checkpoint-argument loader and use raw-logit selection during inference too.
Existing Marin checkpoints from the microbatch-quantile implementation load
without resetting beta, model, optimizer, scheduler, or counters. Missing range
state bootstraps at [-32, 32]; new checkpoints persist each layer's range.
Distributed-checkpoint content metadata distinguishes these formats so strict
loading does not request a missing range shard from an old checkpoint. Reruns
restore beta/range and clear the rejected pass's temporary histogram statistics.

## Validation

From `/ritom/scratch/cscs/ahuang/megatron-apertus-moe`:

```bash
python3 tests/unit_tests/transformer/moe/test_marin_qb_cpu.py
```

These tests execute extracted production function bodies with real PyTorch CPU
tensors and a two-process Gloo group, avoiding CUDA/TE imports. They cover
quantile accuracy, saturation-sensitive selection, padding, empty shards,
degenerate ranges, shift invariance, and deferred updates. They do not validate
the full NCCL pipeline, optimizer-step integration, or training convergence.
