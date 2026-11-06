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

## `marin_exact`: Marin's single global-batch estimator under microbatching

`--moe-router-quantile-balancing-method marin_exact` reproduces Marin's estimator exactly over a
whole optimizer step, independent of TP, DP and the microbatch count. Unlike `marin_histogram`
(per-forward quantiles averaged across microbatches) it never averages quantiles.

* Each routed forward stores its raw-logit margins (`logits - alpha`, fp32, `[tokens, experts]`)
  plus a validity mask. There are **no forward-pass collectives**.
* At the step boundary (`finalize_model_grads`) one pass finds the exact global `[min, max]`
  (two reductions over TP+DP+CP), a second pass builds the per-expert integer histogram on that
  live grid (`--moe-router-quantile-balancing-marin-num-bins`, default 10000; summed over
  TP+DP+CP), and the upper quantile is interpolated in the crossing bin exactly as in
  `_bincount_upper_quantile`. Then EMA (0.0 = replacement) and mean-centering apply.
* Guard (not in Marin): if any valid token has a non-finite margin, or the step is empty, the
  layer keeps its complete previous beta. Logged as `qb_exact_update_skipped` and
  `qb_exact_nonfinite_tokens` (per layer).

**Memory.** The retained margins cost `tokens_per_gpu_per_step x num_experts x 4 B` per MoE
layer, i.e. about 0.4 GB for 393k tokens and 256 experts. `--moe-router-quantile-balancing-marin-exact-store`
chooses where they live: `cpu` (default) copies each microbatch to pinned host memory
asynchronously (one 4 MB D2H copy per layer per microbatch, no GPU memory), `cuda` keeps them on
the GPU. The buffer is sized from `get_num_microbatches()` on the first forward and only grows if
a later step has more rows.

**Constraints.** Requires `cuda_graph_impl none` (Python-side write cursor; validated in
`TransformerConfig`). The router must run once per microbatch with grad enabled; activation
recompute that re-ran the router with grad enabled would store its margins twice. A rerun
(`set_router_rerun_state`) resets the write cursor. Nothing extra is checkpointed beyond `qb_beta`.
The combine-weight epsilon is `1e-9`, as for `marin_histogram`.

Tests: `python tests/unit_tests/transformer/moe/test_marin_qb_exact_cpu.py` (chunk-size
invariance bit-equal to one pooled forward, padding, NaN/empty guards, 2-rank Gloo agreement,
router storage, rerun reset, step-boundary update).
