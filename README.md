<div align="center">

Megatron-LM and Megatron Core
=============================

<h4>GPU-optimized library for training transformer models at scale</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.15.0-green)](./CHANGELOG.md)
[![license](https://img.shields.io/badge/license-Apache-blue)](./LICENSE)

<div align="left">

## swiss-ai fork: shared ablation setup (optional `_research/` submodule)

**Only needed if you intend to use the shared ablation setup.** Otherwise you can ignore this.

If you do want it: `_research/` is a **git submodule** → [`swiss-ai/pretrain`](https://github.com/swiss-ai/pretrain), tracking branch `apertus-2-pretrain-ablations`, holding our Apertus-2 MoE pretraining launcher + ablation configs. They live there so a one-argument config change is a commit to `pretrain` — pulled independently — rather than a push to this repo's `main` that everyone must re-sync to.

**First-time setup**

```bash
# fresh clone: pull the submodule too
git clone --recurse-submodules git@github.com:swiss-ai/Megatron-LM-MoE.git

# existing clone — _research changes from a tracked directory into a submodule, which
# does NOT auto-populate on pull, so initialise it once:
git submodule update --init _research
```

**Everyday workflow for ablations**

```bash
# change a config → commit to pretrain (never touches Megatron main)
cd _research && git add -p && git commit && git push

# stay current with others' config changes
cd _research && git pull            # you're on branch apertus-2-pretrain-ablations
# ...or from the repo root:
git submodule update --remote _research
```

The submodule pointer pinned in this repo deliberately lags what's checked out; everyone pulls `_research` directly to stay current, and the pin is bumped only occasionally. Each run logs `PRETRAIN CONFIG COMMIT: <sha>[-dirty]` — that logged SHA, not the pin, is the source of truth for which config produced a run.

## Muon weight logging

For `--optimizer muon` or `--optimizer dist_muon`, pass `--log-muon-sparsity` and/or
`--log-muon-param-rms` to log the Muon-managed matrices by parameter family. Metrics use
`muon/sparsity/<family>/fraction-below-<threshold>` and `muon/params/<family>/rms`. The default
sparsity thresholds are `1e-20`, `1e-10`, and `1e-30`; override them with
`--muon-sparsity-thresholds`. Use `--muon-log-interval N` to override the default
`--log-interval` cadence, and `--log-muon-per-layer` to additionally emit
`muon/layers/<layer>/...` metrics. Sparsity and RMS are computed per logical matrix and then
averaged equally within each family, with TP shards combined and replicated DP copies counted
once. Pass `--log-muon-gains` to additionally log LayerNorm gains.

## Muon-MD logging

For `--optimizer md_decoupling`, pass `--log-muon-gains` to write effective gain
`mean`, `rms`, `effective-rms`, `min`, and `max` to TensorBoard and Weights & Biases at
the Muon-MD logging interval. Metrics use `muon-md/gains/<family>/<row|col|flat>/<stat>` for
routers, embeddings, outputs, attention, experts, MoE latent projections, dense MLPs, and
unclassified matrices. Values are
transformed to the multipliers applied to weights, with TP shards combined and replicated TP/DP
copies counted once. Row-column configurations also log their combined gain-field RMS at
`muon-md/gain-field/<family>/rms`. Softplus gains additionally log per-axis saturation based on
the softplus derivative, plus combined log scale and row-column imbalance under
`muon-md/gauge/<family>/...`. Effective RMS, combined scale, and gauge values are computed for
each matrix first and then averaged with equal matrix weight, so row and column gains from
unrelated matrices are never paired. Each slice of a merged
expert tensor counts as a separate matrix. The legacy `mean`, `rms`, `min`, `max`, and saturation
fraction remain element-level distribution summaries.

Pass `--log-muon-sparsity` to report effective-weight sparsity at
`muon-md/sparsity/<family>/fraction-below-<threshold>`. The default thresholds are `1e-20`,
`1e-10`, and `1e-30`; override them with, for example,
`--muon-sparsity-thresholds 1e-8 1e-12`. Use `--muon-log-interval N` to set the collection
and logging cadence; when omitted, it inherits `--log-interval`. Logging runs after Muon-MD
reapplies the gains, so the model parameter already contains the effective weight, for example
$W_{\mathrm{eff},ij}=W_{ij}\phi(r_i)\phi(c_j)$ for row-column gains. For each logical matrix $m$,

$$
\operatorname{sparsity}_{\tau}(W_{\mathrm{eff},m})
=\frac{\#\{(i,j): |W_{\mathrm{eff},m,ij}|<\tau\}}
       {\#\{(i,j)\}}.
$$

The family metric is the equal-weight average of these per-matrix fractions. TP shards are
combined before computing the fraction, and each slice of a merged expert tensor is treated as a
separate expert matrix. If a threshold is below the parameter dtype's smallest positive value,
the metric counts exact zeros without underflowing the comparison threshold.

Pass `--log-muon-param-rms` to report effective-parameter RMS at
`muon-md/params/<family>/rms`. For each logical matrix $m$ this is

$$
\operatorname{RMS}(W_m)=\frac{\lVert W_m\rVert_F}{\sqrt{N_m}}
=\sqrt{\frac{\sum_{i,j}W_{m,ij}^2}{N_m}}.
$$

As with sparsity, $W_m$ already includes its applied gains, TP shards are combined, merged experts
remain separate matrices, and the family metric averages the per-matrix RMS values equally.

The logged values are computed as follows:

For a raw gain vector $g$, let $e=\phi(g)$ be its effective multiplier. The element-weighted
statistics are

$$
\operatorname{mean}(e)=\frac{\sum_i e_i}{N},\qquad
\operatorname{RMS}(e)=\sqrt{\frac{\sum_i e_i^2}{N}}.
$$

`min` and `max` are the extrema over the same effective-gain elements. For softplus gains,

$$
\text{saturated-fraction}
=\frac{\#\{i:\operatorname{sigmoid}(g_i)<10^{-2}\}}{N}.
$$

For a TP-sharded gain axis, each rank first contributes $[\sum_i e_i^2,\,N]$, plus
$\sum_i\log e_i$ for softplus gains. Their TP `SUM` all-reduce reconstructs the corresponding
full-axis sums and count. The matrix-level statistics are computed only after that reduction.

For matrices $m=1,\ldots,M$ in a family and axis,

$$
\text{effective-rms}
=\frac{1}{M}\sum_{m=1}^{M}\operatorname{RMS}(e_m).
$$

Muon-MD applies row and column gains by broadcasting, so each matrix entry is multiplied as
$W'_{ij}=W_{ij}r_i c_j$. The combined multiplier is therefore the outer-product gain field
$rc^\top$. Its RMS factorizes exactly as

$$
\operatorname{RMS}(rc^\top)=\operatorname{RMS}(r)\operatorname{RMS}(c).
$$

For $M$ matched row and column gains $r_m,c_m$, the logged family value is

$$
\text{gain-field-rms}
  =\frac{1}{M}\sum_{m=1}^{M}\operatorname{RMS}(r_m)\operatorname{RMS}(c_m).
$$

This describes only the multiplicative gain field. It is not the RMS of the resulting effective
weight $W\odot rc^\top$.

For positive softplus gains, row and column gains have a scale redundancy: replacing
$r$ with $a r$ and $c$ with $c/a$ leaves the gain field $rc^\top$ unchanged. The gauge metrics
separate the meaningful combined scale from this otherwise invisible redistribution:

- `combined-log-scale` is
  $\operatorname{mean}(\log r)+\operatorname{mean}(\log c)$, the log of the product of the row
  and column geometric means. It tracks their combined multiplicative scale.
- `row-col-imbalance` is
  $\operatorname{mean}(\log r)-\operatorname{mean}(\log c)$, the log ratio between those geometric
  means. It tracks whether scale is moving from the column gains into the row gains or vice versa.

For the $M_+$ matched softplus matrices, the logged family averages are

$$
\begin{aligned}
\text{combined-log-scale}
  &=\frac{1}{M_+}\sum_{m=1}^{M_+}
    \left(\operatorname{mean}(\log r_m)+\operatorname{mean}(\log c_m)\right),\\
\text{row-col-imbalance}
  &=\frac{1}{M_+}\sum_{m=1}^{M_+}
    \left(\operatorname{mean}(\log r_m)-\operatorname{mean}(\log c_m)\right).
\end{aligned}
$$

If `row-col-imbalance` drifts while `combined-log-scale` and `gain-field-rms` remain stable, the row
and column gains are mostly rescaling each other rather than changing the combined gain field.

Thus, `rms` weights every gain element equally, whereas `effective-rms`, `gain-field-rms`, and the
gauge metrics weight every matrix equally.

Pass `--log-muon-per-layer` to additionally emit whichever Muon-MD metrics are enabled for
each global, zero-based transformer layer under `muon-md/layers/<layer>/...`. It should be used
selectively for large models because it creates one curve per layer, family, axis, and statistic.

## About

This repository contains two components: **Megatron-LM** and **Megatron Core**.

**Megatron-LM** is a reference example that includes Megatron Core plus pre-configured training scripts. Best for research teams, learning distributed training, and quick experimentation.

**Megatron Core** is a composable library with GPU-optimized building blocks for custom training frameworks. It provides transformer building blocks, advanced parallelism strategies (TP, PP, DP, EP, CP), mixed precision support (FP16, BF16, FP8, FP4), and model architectures. Best for framework developers and ML engineers building custom training pipelines.

**[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** provides bidirectional Hugging Face ↔ Megatron checkpoint conversion with production-ready recipes.

## Getting Started

**Install from PyPI:**

```bash
uv pip install megatron-core
```

**Or clone and install from source:**

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
uv pip install -e .
```

> **Note:** Building from source can use a lot of memory. If the build runs out of memory, limit parallel compilation jobs by setting `MAX_JOBS` (e.g. `MAX_JOBS=4 uv pip install -e .`).

For NGC container setup and all installation options, see the **[Installation Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html)**.

- **[Your First Training Run](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)** - End-to-end training examples with data preparation
- **[Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)** - Scale training across GPUs with TP, PP, DP, EP, and CP
- **[Contribution Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)** - How to contribute to Megatron Core

# Latest News

- **[2026/03]** **Deprecating Python 3.10 support:** We're officially dropping Python 3.10 support with the upcoming 0.17.0 release. Downstream applications must raise their lower boundary to 3.12 to stay compatible with MCore.
- **[2026/01]** **[Dynamic Context Parallelism](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)** - Up to 1.48x speedup for variable-length sequence training with adaptive CP sizing.
- **[2025/12]** **Megatron Core development has moved to GitHub!** All development and CI now happens in the open. We welcome community contributions.
- **[2025/10]** **[Megatron Dev Branch](https://github.com/NVIDIA/Megatron-LM/tree/dev)** - early access branch with experimental features.
- **[2025/10]** **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - Bidirectional converter for interoperability between Hugging Face and Megatron checkpoints, featuring production-ready recipes for popular models.
- **[2025/08]** **[MoE Q3-Q4 2025 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - Comprehensive roadmap for MoE features including DeepSeek-V3, Qwen3, advanced parallelism strategies, FP8 optimizations, and Blackwell performance enhancements.
- **[2025/08]** **[GPT-OSS Model](https://github.com/NVIDIA/Megatron-LM/issues/1739)** - Advanced features including YaRN RoPE scaling, attention sinks, and custom activation functions are being integrated into Megatron Core.
- **[2025/06]** **[Megatron MoE Model Zoo](https://github.com/yanring/Megatron-MoE-ModelZoo)** - Best practices and optimized configurations for training DeepSeek-V3, Mixtral, and Qwen3 MoE models with performance benchmarking and checkpoint conversion tools.
- **[2025/05]** Megatron Core v0.11.0 brings new capabilities for multi-data center LLM training ([blog](https://developer.nvidia.com/blog/turbocharge-llm-training-across-long-haul-data-center-networks-with-nvidia-nemo-framework/)).

<details>
<summary>Previous News</summary>

- **[2024/07]** Megatron Core v0.7 improves scalability and training resiliency and adds support for multimodal training ([blog](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-Megatron-Core-functionalities/)).
- **[2024/06]** Megatron Core added supports for Mamba-based models. Check out our paper [An Empirical Study of Mamba-based Language Models](https://arxiv.org/pdf/2406.07887) and [code example](https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba).
- **[2024/01 Announcement]** NVIDIA has released the core capabilities in **Megatron-LM** into [**Megatron Core**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) in this repository. Megatron Core expands upon Megatron-LM's GPU-optimized techniques with more cutting-edge innovations on system-level optimizations, featuring composable and modular APIs.

</details>

# Project Structure

```
Megatron-LM/
├── megatron/
│   ├── core/                    # Megatron Core (kernels, parallelism, building blocks)
│   │   ├── models/              # Transformer models
│   │   ├── transformer/         # Transformer building blocks
│   │   ├── tensor_parallel/     # Tensor parallelism
│   │   ├── pipeline_parallel/   # Pipeline parallelism
│   │   ├── distributed/         # Distributed training (FSDP, DDP)
│   │   ├── optimizer/           # Optimizers
│   │   ├── datasets/            # Dataset loaders
│   │   ├── inference/           # Inference engines and server
│   │   └── export/              # Model export (e.g. TensorRT-LLM)
│   ├── training/                # Training scripts
│   ├── legacy/                  # Legacy components
│   ├── post_training/           # Post-training (quantization, distillation, pruning, etc.)
│   └── rl/                      # Reinforcement learning (RLHF, etc.)
├── examples/                    # Ready-to-use training examples
├── tools/                       # Utility tools
├── tests/                       # Comprehensive test suite
└── docs/                        # Documentation
```

# Performance Benchmarking

For our latest performance benchmarking results, please refer to [NVIDIA Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html).

Our codebase efficiently trains models from 2B to 462B parameters across thousands of GPUs, achieving up to **47% Model FLOP Utilization (MFU)** on H100 clusters.

![Model table](images/model_table.png)

**Benchmark Configuration:**

- **Vocabulary size**: 131,072 tokens
- **Sequence length**: 4096 tokens
- **Model scaling**: Varied hidden size, attention heads, and layers to achieve target parameter counts
- **Communication optimizations**: Fine-grained overlapping with DP (`--overlap-grad-reduce`, `--overlap-param-gather`), TP (`--tp-comm-overlap`), and PP (enabled by default)

**Key Results:**

- **6144 H100 GPUs**: Successfully benchmarked 462B parameter model training
- **Superlinear scaling**: MFU increases from 41% to 47-48% with model size
- **End-to-end measurement**: Throughputs include all operations (data loading, optimizer steps, communication, logging)
- **Production ready**: Full training pipeline with checkpointing and fault tolerance
- *Note: Performance results measured without training to convergence*

## Weak Scaling Results

Our weak scaled results show superlinear scaling (MFU increases from 41% for the smallest model considered to 47-48% for the largest models); this is because larger GEMMs have higher arithmetic intensity and are consequently more efficient to execute.

![Weak scaling](images/weak_scaling.png)

## Strong Scaling Results

We also strong scaled the standard GPT-3 model (our version has slightly more than 175 billion parameters due to larger vocabulary size) from 96 H100 GPUs to 4608 GPUs, using the same batch size of 1152 sequences throughout. Communication becomes more exposed at larger scale, leading to a reduction in MFU from 47% to 42%.

![Strong scaling](images/strong_scaling.png)

# Roadmaps

- **[MoE Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - DeepSeek-V3, Qwen3, advanced parallelism, FP8 optimizations, and Blackwell enhancements

# Resources

## Getting Help

- 📖 **[Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)** - Official documentation
- 🐛 **[Issues](https://github.com/NVIDIA/Megatron-LM/issues)** - Bug reports and feature requests

## Contributing

We ❤️ contributions! Ways to contribute:

- 🐛 **Report bugs** - Help us improve reliability
- 💡 **Suggest features** - Shape the future of Megatron Core
- 📝 **Improve docs** - Make Megatron Core more accessible
- 🔧 **Submit PRs** - Contribute code improvements

**→ [Contributing Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)**

## Citation

If you use Megatron in your research or project, we appreciate that you use the following citations:

```bibtex
@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}
```
