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
The top-(k+1) cutoff defines raw-logit margins. Two scalar all-reduces find their
current global extrema; an integer histogram reduction pools counts. The upper
quantile is interpolated within the crossing bin. Padding contributes neither
extrema nor counts; empty global forwards preserve the previous threshold.

**Gradient accumulation changes the semantics:** each microbatch uses its own
globally agreed live grid, and its resulting quantile is weighted by valid-token
count. These estimates are averaged at the optimizer-step boundary before the
existing EMA and mean-centering. With one microbatch this is a global-batch
histogram; with multiple microbatches it is an average of global-microbatch
quantiles, NOT a pooled full-step quantile. Histograms on different grids cannot
simply be added. No full-step routing margins are retained in memory.

There are three new forward collectives per routed layer per microbatch, unlike
the existing bounded-score histogram's step-boundary-only communication. More
bins and temporary margin tensors also cost memory. GPU throughput and CUDA graph
compatibility must be measured on the target setup.

## Checkpoints

Raw-logit and sigmoid-domain beta values are not interchangeable. Do not silently
switch a trained `histogram` checkpoint to this method: the branch does not reset
or convert its saved thresholds. Use a fresh run or an explicitly prepared
checkpoint with reinitialized QB thresholds, then assess the routing transient.
Checkpoints produced with `marin_histogram` retain the method through the existing
checkpoint-argument loader and use raw-logit selection during inference too.

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
