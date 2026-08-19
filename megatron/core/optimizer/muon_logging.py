# Copyright (c) 2026, EPFL / Swiss AI Initiative.

"""Distributed TensorBoard/W&B statistics for Muon and Muon-MD parameters.

The computation has three stages:

1. :func:`collect_md_gain_stats` accumulates element-level gain distributions and saturation.
2. :func:`_accumulate_md_matrix_stats` reconstructs TP-sharded matrix gains, then
   :func:`_accumulate_reduced_matrix_batch` computes per-matrix RMS, scale, gauge, and
   effective-weight sparsity values.
3. :func:`_append_md_gain_stats` converts the globally reduced sums into logged averages.
"""

import math
from typing import Dict, List

import torch

from megatron.core.utils import (
    get_pg_rank,
    get_pg_size,
    nvtx_range_pop,
    nvtx_range_push,
)

from .layer_wise_optimizer import LayerWiseDistributedOptimizer

_GAIN_AXES = ("row", "col", "flat")
_MATRIX_SQUARE_SUM, _MATRIX_ELEMENT_COUNT, _MATRIX_LOG_SUM, _WEIGHT_SQUARE_SUM, _WEIGHT_ELEMENT_COUNT, _MATRIX_NEAR_ZERO_START = range(6)  # fmt: skip

# Columns of ``totals``: element-weighted effective-gain distribution statistics.
_GAIN_SUM, _GAIN_SQUARE_SUM, _GAIN_ELEMENT_COUNT, _GAIN_SOFTPLUS_COUNT, _GAIN_SATURATED_COUNT, _GAIN_STAT_COUNT = range(6)  # fmt: skip

# Columns of ``matrix_axis_totals``: equal-weighted per-matrix effective RMS.
_MATRIX_RMS_SUM, _MATRIX_COUNT, _MATRIX_AXIS_STAT_COUNT = range(3)

# Fixed columns of ``matrix_totals``; one sparsity-sum column per threshold follows these.
_SCALE_SUM, _PAIR_COUNT, _COMBINED_LOG_SCALE_SUM, _ROW_COL_IMBALANCE_SUM, _SOFTPLUS_PAIR_COUNT, _SPARSITY_MATRIX_COUNT, _PARAM_RMS_SUM, _PARAM_RMS_COUNT, _SPARSITY_SUM_START = range(9)  # fmt: skip

_GAIN_FAMILIES = (
    "router",
    "embedding",
    "output",
    "attention-in",
    "attention-out",
    "expert-in",
    "expert-out",
    "moe-latent-in",
    "moe-latent-out",
    "dense-mlp-in",
    "dense-mlp-out",
    "layernorm",
    "unclassified",
)


def _matrix_weight_stats(
    matrix_weights: torch.Tensor, thresholds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count near-zero values and accumulate squared weights."""
    absolute = matrix_weights.abs()
    near_zero_counts = torch.stack(
        [(absolute < threshold).sum(dim=1) for threshold in thresholds.unbind()], dim=1
    )
    return near_zero_counts, matrix_weights.float().square().sum(dim=1)


@torch.compile(dynamic=False, fullgraph=True)
def _compiled_matrix_weight_stats(
    matrix_weights: torch.Tensor, thresholds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # NOTE: With the compilation we remove the problem of having the sparsity generate two full temporal matrices and also iterating over the whole matrix 3 times.
    return _matrix_weight_stats(matrix_weights, thresholds)


def _softplus_matrix_gain_stats(raw_gains: torch.Tensor) -> torch.Tensor:
    """Compute the two softplus matrix reductions in one pass."""
    effective_gains = torch.nn.functional.softplus(raw_gains.to(torch.float64))
    return torch.stack(
        (effective_gains.square().sum(dim=1), effective_gains.log().sum(dim=1)), dim=1
    )


@torch.compile(dynamic=True, fullgraph=True)
def _compiled_softplus_matrix_gain_stats(raw_gains: torch.Tensor) -> torch.Tensor:
    return _softplus_matrix_gain_stats(raw_gains)


def _gain_log_family(name: str, param: torch.Tensor) -> str:
    """Assign a stable, low-cardinality logging family while the parameter name is available."""
    if param.ndim == 1 and (
        name.endswith("layer_norm_weight")
        or (name.endswith(".weight") and "norm" in name.rpartition(".")[0].lower())
    ):
        return "layernorm"
    if getattr(param, "is_router", False):
        return "router"
    if getattr(param, "is_md_embedding_parameter", False):
        return "embedding"
    if getattr(param, "is_md_output_parameter", False):
        return "output"

    is_out = getattr(param, "is_out_proj", False)
    if "fc1_latent_proj" in name:
        return "moe-latent-in"
    if "fc2_latent_proj" in name:
        return "moe-latent-out"
    if "experts" in name:
        return "expert-out" if is_out else "expert-in"
    if "attention" in name:
        return "attention-out" if is_out else "attention-in"
    if ".mlp." in name:
        return "dense-mlp-out" if is_out else "dense-mlp-in"
    return "unclassified"


def _include_gain_in_global_stats(
    md_optimizer: torch.optim.Optimizer,
    param: torch.Tensor,
    axis: str,
    dp_state_is_sharded: bool,
) -> bool:
    """Return whether this rank owns a unique logical copy of a gain tensor."""
    pg_collection = md_optimizer.pg_collection
    if pg_collection is None:
        return True

    is_expert = getattr(param, "expert_tp", False)
    if not dp_state_is_sharded:
        dp_group = (
            getattr(pg_collection, "expt_dp", None)
            if is_expert
            else getattr(pg_collection, "dp_cp", None)
        )
        if dp_group is not None and get_pg_rank(dp_group) != 0:
            return False

    retained_dim = {"row": 0, "col": 1, "flat": None}[axis]
    gain_is_tp_sharded = (
        retained_dim is not None
        and getattr(param, "partition_dim", None) == retained_dim
    )
    if not gain_is_tp_sharded:
        tp_group = (
            getattr(pg_collection, "expt_tp", None)
            if is_expert
            else getattr(pg_collection, "tp", None)
        )
        if tp_group is not None and get_pg_rank(tp_group) != 0:
            return False
    return True


def _include_param_in_matrix_stats(
    md_optimizer: torch.optim.Optimizer, param: torch.Tensor, dp_state_is_sharded: bool
) -> bool:
    """Select one DP/CP owner while retaining every unique TP/EP matrix shard."""
    if md_optimizer.pg_collection is None or dp_state_is_sharded:
        return True
    is_expert = getattr(param, "expert_tp", False)
    dp_group = (
        getattr(md_optimizer.pg_collection, "expt_dp", None)
        if is_expert
        else getattr(md_optimizer.pg_collection, "dp_cp", None)
    )
    return dp_group is None or get_pg_rank(dp_group) == 0


def _accumulate_reduced_matrix_batch(
    reduced: torch.Tensor,
    metadata,
    matrix_axis_totals: torch.Tensor,
    matrix_totals: torch.Tensor,
    sparsity_thresholds: tuple[float, ...],
    log_gains: bool,
    log_param_rms: bool,
) -> None:
    """Compute per-matrix metrics after TP has reconstructed each gain axis.

    ``effective-rms`` is computed separately for every matrix. A row/column pair from that same
    matrix produces ``gain-field/rms`` and, for softplus gains, the two gauge metrics. Sparsity
    is the fraction of gain-baked weight values whose magnitude is below each configured
    threshold. The resulting values are summed here and divided by their matrix counts in
    :func:`_append_md_gain_stats`.
    """
    family_indices = torch.tensor(
        [family for family, _, count, _ in metadata for _ in range(count)],
        dtype=torch.long,
        device=reduced.device,
    )
    expanded_layers = [
        (-1 if layer is None else layer)
        for _, layer, count, _ in metadata
        for _ in range(count)
    ]
    layers = torch.tensor(expanded_layers, dtype=torch.long, device=reduced.device)
    softplus = torch.tensor(
        [enabled for _, _, count, enabled in metadata for _ in range(count)],
        dtype=torch.bool,
        device=reduced.device,
    )
    layer_positions = torch.tensor(
        [
            position
            for position, layer in enumerate(expanded_layers)
            if 0 <= layer and layer + 1 < matrix_axis_totals.size(0)
        ],
        dtype=torch.long,
        device=reduced.device,
    )

    # Per-matrix gain effective RMS is formed here, after TP has combined the squared sums and
    # element counts for any sharded gain axis: sqrt(sum(phi(g)^2) / number of gain elements).
    # These matrix RMS values feed ``gains/.../effective-rms``; they are distinct from the
    # element-weighted gain RMS computed later in ``collect_md_gain_stats``.
    gain_counts = reduced[:, :, _MATRIX_ELEMENT_COUNT]
    axis_valid = gain_counts > 0
    matrix_rms = torch.sqrt(
        reduced[:, :, _MATRIX_SQUARE_SUM] / gain_counts.clamp_min(1)
    )
    if log_gains:
        buckets = family_indices[:, None] * len(_GAIN_AXES) + torch.arange(
            len(_GAIN_AXES), device=reduced.device
        )
        axis_indices = buckets.flatten()
        axis_values = torch.stack(
            (matrix_rms, axis_valid.to(matrix_rms.dtype)), dim=2
        ).view(-1, 2)
        if layer_positions.numel():
            layer_buckets = buckets.index_select(0, layer_positions)
            layer_scopes = layers.index_select(0, layer_positions) + 1
            axis_indices = torch.cat(
                (
                    axis_indices,
                    (
                        layer_scopes[:, None] * matrix_axis_totals.size(1)
                        + layer_buckets
                    ).flatten(),
                )
            )
            layer_matrix_rms = matrix_rms.index_select(0, layer_positions)
            layer_axis_valid = axis_valid.index_select(0, layer_positions)
            axis_values = torch.cat(
                (
                    axis_values,
                    torch.stack(
                        (layer_matrix_rms, layer_axis_valid.to(matrix_rms.dtype)), dim=2
                    ).view(-1, 2),
                )
            )
        matrix_axis_totals.view(-1, _MATRIX_AXIS_STAT_COUNT).index_add_(
            0, axis_indices, axis_values
        )

    matrix_values = torch.zeros(
        (reduced.size(0), matrix_totals.size(2)),
        dtype=matrix_totals.dtype,
        device=matrix_totals.device,
    )
    weight_element_count = reduced[:, 0, _WEIGHT_ELEMENT_COUNT]
    weight_valid = weight_element_count > 0
    if sparsity_thresholds:
        # ``_matrix_weight_stats`` counted abs(effective weight) < threshold per matrix. Convert
        # those counts into per-matrix fractions before families/layers average the matrices.
        matrix_values[:, _SPARSITY_MATRIX_COUNT] = weight_valid
        matrix_values[:, _SPARSITY_SUM_START:] = reduced[
            :, 0, _MATRIX_NEAR_ZERO_START:
        ] / weight_element_count[:, None].clamp_min(1)

    # Frobenious RMS norm of the gain-baked weight (the effective weight)
    if log_param_rms:
        # ``params/.../rms`` = sqrt(sum(W_effective^2) / numel(W_effective)), computed per matrix
        # here and then averaged with equal weight across matrices in the same family/scope.
        matrix_values[:, _PARAM_RMS_SUM] = torch.sqrt(
            reduced[:, 0, _WEIGHT_SQUARE_SUM] / weight_element_count.clamp_min(1)
        )
        matrix_values[:, _PARAM_RMS_COUNT] = weight_valid

    if log_gains:
        row, col = reduced[:, :2].unbind(dim=1)
        row_rms, col_rms = matrix_rms[:, :2].unbind(dim=1)
        row_col_valid = row[:, _MATRIX_ELEMENT_COUNT].gt(0) & col[
            :, _MATRIX_ELEMENT_COUNT
        ].gt(0)

        # RMS of the complete outer-product gain field: RMS(r c^T) = RMS(r) * RMS(c).
        scale = row_rms * col_rms
        matrix_values[:, _SCALE_SUM] = torch.where(row_col_valid, scale, 0)
        matrix_values[:, _PAIR_COUNT] = row_col_valid

        # Gauge metrics use positive effective softplus gains. Their sum tracks the combined
        # multiplicative log scale; their difference exposes row/column rescaling redundancy.
        softplus_pair = row_col_valid & softplus
        row_mean_log = row[:, _MATRIX_LOG_SUM] / row[
            :, _MATRIX_ELEMENT_COUNT
        ].clamp_min(1)
        col_mean_log = col[:, _MATRIX_LOG_SUM] / col[
            :, _MATRIX_ELEMENT_COUNT
        ].clamp_min(1)
        matrix_values[:, _COMBINED_LOG_SCALE_SUM] = torch.where(
            softplus_pair, row_mean_log + col_mean_log, 0
        )
        matrix_values[:, _ROW_COL_IMBALANCE_SUM] = torch.where(
            softplus_pair, row_mean_log - col_mean_log, 0
        )
        matrix_values[:, _SOFTPLUS_PAIR_COUNT] = softplus_pair

    matrix_indices = family_indices
    if layer_positions.numel():
        matrix_indices = torch.cat(
            (
                matrix_indices,
                (layers.index_select(0, layer_positions) + 1) * matrix_totals.size(1)
                + family_indices.index_select(0, layer_positions),
            )
        )
        matrix_values = torch.cat(
            (matrix_values, matrix_values.index_select(0, layer_positions))
        )
    matrix_totals.view(-1, matrix_totals.size(2)).index_add_(
        0, matrix_indices, matrix_values
    )


def _accumulate_md_matrix_stats(
    md_optimizers: List[torch.optim.Optimizer],
    dp_state_is_sharded: bool,
    family_indices: Dict[str, int],
    matrix_axis_totals: torch.Tensor,
    matrix_totals: torch.Tensor,
    sparsity_thresholds: tuple[float, ...],
    log_gains: bool,
    log_param_rms: bool,
) -> None:
    """Build the sufficient statistics needed for matrix-level metrics.

    Each record contains gain statistics plus the near-zero and element counts of the gain-baked
    weight. TP all-reduce reconstructs sharded matrices and gain axes before matrix metrics are
    computed. Merged 3D expert parameters retain one record per local expert matrix.
    """
    batches = {}
    for md_optimizer in md_optimizers:
        for param, state in md_optimizer.state.items():
            if not _include_param_in_matrix_stats(
                md_optimizer, param, dp_state_is_sharded
            ):
                continue
            present_axes = [axis for axis in _GAIN_AXES if f"{axis}_gain" in state]
            if not present_axes and not sparsity_thresholds and not log_param_rms:
                continue

            is_expert = getattr(param, "expert_tp", False)
            tp_group = None
            if md_optimizer.pg_collection is not None:
                tp_group = (
                    getattr(md_optimizer.pg_collection, "expt_tp", None)
                    if is_expert
                    else getattr(md_optimizer.pg_collection, "tp", None)
                )
            tp_rank = get_pg_rank(tp_group) if tp_group is not None else 0
            matrix_count = param.size(0) if param.ndim == 3 else 1
            matrix_stat_count = _MATRIX_NEAR_ZERO_START + len(sparsity_thresholds)
            record = torch.zeros(
                (matrix_count, len(_GAIN_AXES), matrix_stat_count),
                dtype=torch.float64,
                device=param.device,
            )
            partition_dim = getattr(param, "partition_dim", None)
            if (sparsity_thresholds or log_param_rms) and (
                partition_dim in {0, 1} or tp_rank == 0
            ):
                # MDDecoupling reapplies gains to the parameter at the end of every optimizer
                # step, before logging runs. ``param`` is therefore already the effective weight.
                matrix_weights = param.detach().reshape(matrix_count, -1)
                dtype_info = torch.finfo(matrix_weights.dtype)
                smallest_positive = dtype_info.tiny * dtype_info.eps
                weight_square_sum = None
                if sparsity_thresholds:
                    # This is where exact effective-weight sparsity is measured. Merged expert
                    # tensors are reshaped with one row per expert, so every expert remains a
                    # separate matrix rather than one combined 3D sparsity measurement.
                    # The smallest representable positive threshold selects exact zeros without
                    # underflowing the comparison value in the weight dtype.
                    thresholds = matrix_weights.new_tensor(
                        [
                            max(threshold, smallest_positive)
                            for threshold in sparsity_thresholds
                        ]
                    )
                    # NOTE: For now MuonMD always have the master FP32 parametes, so we will always have them in GPU
                    matrix_stats = (
                        _compiled_matrix_weight_stats
                        if matrix_weights.is_cuda
                        else _matrix_weight_stats
                    )
                    near_zero_counts, weight_square_sum = matrix_stats(
                        matrix_weights, thresholds
                    )
                    record[:, 0, _MATRIX_NEAR_ZERO_START:] = near_zero_counts
                if log_param_rms and weight_square_sum is None:
                    # Parameter RMS needs the same sum(W_effective^2) as the sparsity kernel. If
                    # sparsity is disabled, compute that sufficient statistic directly here.
                    weight_square_sum = torch.linalg.vector_norm(
                        matrix_weights, dim=1, dtype=torch.float32
                    ).square()
                if weight_square_sum is not None:
                    record[:, 0, _WEIGHT_SQUARE_SUM] = weight_square_sum
                record[:, 0, _WEIGHT_ELEMENT_COUNT] = matrix_weights.size(1)
            for axis_index, axis in enumerate(_GAIN_AXES):
                raw_gain = state.get(f"{axis}_gain")
                if raw_gain is None or not log_gains:
                    continue
                retained_dim = {"row": 0, "col": 1, "flat": None}[axis]
                gain_is_tp_sharded = (
                    retained_dim is not None
                    and getattr(param, "partition_dim", None) == retained_dim
                )
                if not gain_is_tp_sharded and tp_rank != 0:
                    continue

                raw_matrix_gains = (
                    raw_gain.reshape(matrix_count, -1)
                    if param.ndim == 3
                    else raw_gain.reshape(1, -1)
                )
                # Matrix-level gain statistics are based on effective gains phi(raw_gain). Store
                # their squared sum for effective RMS and, for softplus, their log sum for gauge
                # metrics. TP combines these sufficient statistics before either value is formed.
                if md_optimizer.gain_parametrization == "softplus":
                    gain_stats = (
                        _compiled_softplus_matrix_gain_stats
                        if raw_gain.is_cuda
                        else _softplus_matrix_gain_stats
                    )(raw_matrix_gains)
                    record[:, axis_index, _MATRIX_SQUARE_SUM] = gain_stats[:, 0]
                    record[:, axis_index, _MATRIX_LOG_SUM] = gain_stats[:, 1]
                else:
                    record[:, axis_index, _MATRIX_SQUARE_SUM] = (
                        raw_matrix_gains.to(torch.float64).square().sum(dim=1)
                    )
                record[:, axis_index, _MATRIX_ELEMENT_COUNT] = raw_matrix_gains.size(1)

            records, metadata = batches.setdefault(tp_group, ([], []))
            records.append(record)
            family = getattr(param, "md_gain_log_family", "unclassified")
            metadata.append(
                (
                    family_indices.get(family, family_indices["unclassified"]),
                    getattr(param, "md_gain_log_layer", None),
                    matrix_count,
                    getattr(md_optimizer, "gain_parametrization", "direct") == "softplus",
                )
            )

    pending = []
    for tp_group, (records, metadata) in batches.items():
        reduced = torch.cat(records)
        work = None
        if tp_group is not None and get_pg_size(tp_group) > 1:
            work = torch.distributed.all_reduce(reduced, group=tp_group, async_op=True)
        pending.append((tp_group, reduced, metadata, work))

    for tp_group, reduced, metadata, work in pending:
        if work is not None:
            work.wait()
        if tp_group is not None and get_pg_rank(tp_group) != 0:
            continue
        _accumulate_reduced_matrix_batch(
            reduced,
            metadata,
            matrix_axis_totals,
            matrix_totals,
            sparsity_thresholds,
            log_gains,
            log_param_rms,
        )


def _append_md_gain_stats(
    stats: Dict[str, float],
    totals: List,
    minima: List,
    maxima: List,
    matrix_axis_totals: List,
    matrix_totals: List,
    sparsity_thresholds: tuple[float, ...],
    prefix: str = "muon-md",
) -> None:
    """Turn reduced sufficient statistics into the final scalar metric values.

    ``mean``, ``rms``, saturation, ``min``, and ``max`` describe all gain elements in a bucket.
    ``effective-rms``, combined scale, gauge, and sparsity metrics instead average
    already-computed matrix values, so differently sized matrices receive equal weight.
    """
    for family_index, family in enumerate(_GAIN_FAMILIES):
        for axis_index, axis in enumerate(_GAIN_AXES):
            bucket = family_index * len(_GAIN_AXES) + axis_index
            total, sum_square, count, softplus_count, saturated_count = totals[bucket]
            if count == 0:
                continue
            gain_prefix = f"{prefix}/gains/{family}/{axis}"
            stats[f"{gain_prefix}/mean"] = total / count
            stats[f"{gain_prefix}/rms"] = math.sqrt(sum_square / count)
            matrix_rms_sum, matrix_count = matrix_axis_totals[bucket]
            stats[f"{gain_prefix}/effective-rms"] = matrix_rms_sum / matrix_count
            stats[f"{gain_prefix}/min"] = minima[bucket]
            stats[f"{gain_prefix}/max"] = maxima[bucket]
            if softplus_count:
                stats[f"{gain_prefix}/saturated-fraction"] = (
                    saturated_count / softplus_count
                )

        scale_sum, pair_count, combined_sum, imbalance_sum, softplus_pair_count = (
            matrix_totals[family_index][:_SPARSITY_MATRIX_COUNT]
        )
        if pair_count:
            stats[f"{prefix}/gain-field/{family}/rms"] = scale_sum / pair_count
        if softplus_pair_count:
            stats[f"{prefix}/gauge/{family}/combined-log-scale"] = (
                combined_sum / softplus_pair_count
            )
            stats[f"{prefix}/gauge/{family}/row-col-imbalance"] = (
                imbalance_sum / softplus_pair_count
            )
        sparsity_matrix_count = matrix_totals[family_index][_SPARSITY_MATRIX_COUNT]
        if sparsity_matrix_count:
            for threshold_index, threshold in enumerate(sparsity_thresholds):
                sparsity_sum = matrix_totals[family_index][
                    _SPARSITY_SUM_START + threshold_index
                ]
                stats[f"{prefix}/sparsity/{family}/fraction-below-{threshold}"] = (
                    sparsity_sum / sparsity_matrix_count
                )
        param_rms_sum = matrix_totals[family_index][_PARAM_RMS_SUM]
        param_rms_count = matrix_totals[family_index][_PARAM_RMS_COUNT]
        if param_rms_count:
            stats[f"{prefix}/params/{family}/rms"] = param_rms_sum / param_rms_count


def collect_md_gain_stats(
    optimizer,
    per_layer: bool = False,
    sparsity_thresholds=(1e-20, 1e-10, 1e-30),
    log_gains: bool = True,
    log_sparsity: bool = True,
    log_param_rms: bool = True,
) -> Dict[str, float]:
    """Collect selected global and optionally per-layer Muon-MD statistics."""
    from .md_decoupling import MDDecoupling

    return _collect_muon_stats(
        optimizer,
        MDDecoupling,
        "muon-md",
        per_layer,
        sparsity_thresholds,
        log_gains,
        log_sparsity,
        log_param_rms,
    )


def collect_muon_stats(
    optimizer,
    per_layer: bool = False,
    sparsity_thresholds=(1e-20, 1e-10, 1e-30),
    log_gains: bool = True,
    log_sparsity: bool = True,
    log_param_rms: bool = True,
) -> Dict[str, float]:
    """Collect standard Muon LayerNorm gain, weight sparsity, and RMS statistics."""
    from .muon import TensorParallelMuon

    return _collect_muon_stats(
        optimizer,
        TensorParallelMuon,
        "muon",
        per_layer,
        sparsity_thresholds,
        log_gains,
        log_sparsity,
        log_param_rms,
    )


@torch.no_grad()
def _collect_muon_stats(
    optimizer,
    optimizer_class,
    namespace: str,
    per_layer: bool,
    sparsity_thresholds,
    log_gains: bool,
    log_sparsity: bool,
    log_param_rms: bool,
) -> Dict[str, float]:
    """Collect selected global and optionally per-layer Muon-family statistics.

    Scope 0 holds the global aggregates; scope ``layer + 1`` holds a layer's aggregates. Local
    sufficient statistics are accumulated on-device, all scopes are reduced together, and only
    the final scalar divisions happen on CPU.
    """
    if not log_gains and not log_sparsity and not log_param_rms:
        return {}
    sparsity_thresholds = (
        tuple(dict.fromkeys(float(value) for value in sparsity_thresholds))
        if log_sparsity
        else ()
    )
    if log_sparsity and (
        not sparsity_thresholds
        or any(not math.isfinite(value) or value <= 0 for value in sparsity_thresholds)
    ):
        raise ValueError(f"{namespace} sparsity thresholds must be finite positive values")

    wrapped_optimizers = getattr(optimizer, "chained_optimizers", (optimizer,))
    inner_optimizers = [
        getattr(wrapped, "optimizer", wrapped) for wrapped in wrapped_optimizers
    ]
    md_optimizers = [
        wrapped for wrapped in inner_optimizers if isinstance(wrapped, optimizer_class)
    ]
    layernorm_params = [
        param
        for inner_optimizer in inner_optimizers
        for group in inner_optimizer.param_groups
        for param in group["params"]
        if log_gains and getattr(param, "md_gain_log_family", None) == "layernorm"
    ]

    logged_params = [
        param for md_optimizer in md_optimizers for param in md_optimizer.state
    ] + layernorm_params

    dp_state_is_sharded = isinstance(optimizer, LayerWiseDistributedOptimizer)
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if not md_optimizers and not distributed:
        return {}

    nvtx_range_push(f"{namespace}/logging")
    bucket_count = len(_GAIN_FAMILIES) * len(_GAIN_AXES)
    device = (
        logged_params[0].device
        if logged_params
        else (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    )

    layer_count = 0
    if per_layer:
        layer_count = max(
            (
                getattr(param, "md_gain_log_layer", -1) + 1
                for param in logged_params
            ),
            default=0,
        )
        if distributed:
            layer_count_tensor = torch.tensor(
                layer_count, dtype=torch.int64, device=device
            )
            torch.distributed.all_reduce(
                layer_count_tensor, op=torch.distributed.ReduceOp.MAX
            )
            layer_count = int(layer_count_tensor.item())
    scope_count = layer_count + 1
    totals = torch.zeros(
        (scope_count, bucket_count, _GAIN_STAT_COUNT),
        dtype=torch.float64,
        device=device,
    )
    minima = torch.full(
        (scope_count, bucket_count), float("inf"), dtype=torch.float64, device=device
    )
    maxima = torch.full_like(minima, float("-inf"))
    matrix_axis_totals = torch.zeros(
        (scope_count, bucket_count, _MATRIX_AXIS_STAT_COUNT),
        dtype=torch.float64,
        device=device,
    )
    matrix_totals = torch.zeros(
        (
            scope_count,
            len(_GAIN_FAMILIES),
            _SPARSITY_SUM_START + len(sparsity_thresholds),
        ),
        dtype=torch.float64,
        device=device,
    )

    family_indices = {name: index for index, name in enumerate(_GAIN_FAMILIES)}
    nvtx_range_push(f"{namespace}/logging/matrix-statistics")
    _accumulate_md_matrix_stats(
        md_optimizers,
        dp_state_is_sharded,
        family_indices,
        matrix_axis_totals,
        matrix_totals,
        sparsity_thresholds,
        log_gains,
        log_param_rms,
    )
    nvtx_range_pop(f"{namespace}/logging/matrix-statistics")
    nvtx_range_push(f"{namespace}/logging/gain-statistics")
    gain_entries = []
    for md_optimizer in md_optimizers:
        for param, state in md_optimizer.state.items():
            if not log_gains:
                continue
            family = getattr(param, "md_gain_log_family", "unclassified")
            family_index = family_indices.get(family, family_indices["unclassified"])
            for axis_index, axis in enumerate(_GAIN_AXES):
                raw_gain = state.get(f"{axis}_gain")
                if raw_gain is None or not _include_gain_in_global_stats(
                    md_optimizer, param, axis, dp_state_is_sharded
                ):
                    continue
                bucket = family_index * len(_GAIN_AXES) + axis_index
                layer = getattr(param, "md_gain_log_layer", None)
                layer_scope = layer + 1 if per_layer and layer is not None else None
                gain_entries.append(
                    (
                        raw_gain,
                        bucket,
                        layer_scope,
                        md_optimizer.gain_parametrization == "softplus",
                    )
                )
    # LayerNorm scales are direct 1D gains owned by the chained Adam optimizer. Add one for
    # zero-centered gamma before element statistics, and average each norm vector's RMS equally.
    if log_gains and md_optimizers:
        layernorm_bucket = (
            family_indices["layernorm"] * len(_GAIN_AXES) + _GAIN_AXES.index("flat")
        )
        for param in layernorm_params:
            if not _include_gain_in_global_stats(
                md_optimizers[0], param, "flat", dp_state_is_sharded
            ):
                continue
            effective_gain = param.detach() + getattr(param, "md_layernorm_gain_offset", 0.0)
            layer = getattr(param, "md_gain_log_layer", None)
            layer_scope = layer + 1 if per_layer and layer is not None else None
            gain_entries.append((effective_gain, layernorm_bucket, layer_scope, False))
            rms = effective_gain.to(torch.float64).square().mean().sqrt()
            matrix_axis_totals[0, layernorm_bucket, _MATRIX_RMS_SUM] += rms
            matrix_axis_totals[0, layernorm_bucket, _MATRIX_COUNT] += 1
            if layer_scope is not None:
                matrix_axis_totals[layer_scope, layernorm_bucket, _MATRIX_RMS_SUM] += rms
                matrix_axis_totals[layer_scope, layernorm_bucket, _MATRIX_COUNT] += 1
    if gain_entries:
        # These are element-level gain distribution metrics: mean, RMS, min, max, and softplus
        # saturation. Unlike ``effective-rms`` above, every gain element has equal weight here;
        # differently sized matrices therefore contribute different numbers of observations.
        entry_counts = torch.tensor(
            [entry[0].numel() for entry in gain_entries],
            dtype=torch.long,
            device=device,
        )
        raw_values = torch.cat(
            [entry[0].detach().flatten() for entry in gain_entries]
        ).to(torch.float64)
        entry_ids = torch.repeat_interleave(
            torch.arange(len(gain_entries), device=device), entry_counts
        )
        softplus_values = torch.tensor(
            [entry[3] for entry in gain_entries], dtype=torch.bool, device=device
        )[entry_ids]
        effective_gain = torch.where(
            softplus_values, torch.nn.functional.softplus(raw_values), raw_values
        )
        # Saturation is evaluated on each raw softplus gain, not on the weight matrix. The raw
        # threshold below is exactly equivalent to sigmoid(raw_gain) < 1e-2.
        # sigmoid(raw_gain) is d softplus(raw_gain) / d raw_gain. Below 1e-2, the
        # effective multiplier changes by less than 1% of the raw-gain update.
        saturated = softplus_values & raw_values.lt(math.log(1e-2 / (1 - 1e-2)))
        gain_values = torch.stack(
            (
                torch.segment_reduce(effective_gain, "sum", lengths=entry_counts),
                torch.segment_reduce(
                    effective_gain.square(), "sum", lengths=entry_counts
                ),
                entry_counts.to(torch.float64),
                torch.segment_reduce(
                    softplus_values.to(torch.float64), "sum", lengths=entry_counts
                ),
                torch.segment_reduce(
                    saturated.to(torch.float64), "sum", lengths=entry_counts
                ),
            ),
            dim=1,
        )
        gain_minima = torch.segment_reduce(effective_gain, "min", lengths=entry_counts)
        gain_maxima = torch.segment_reduce(effective_gain, "max", lengths=entry_counts)
        entry_buckets = torch.tensor(
            [entry[1] for entry in gain_entries], dtype=torch.long, device=device
        )
        flat_totals = totals.view(-1, _GAIN_STAT_COUNT)
        flat_minima = minima.view(-1)
        flat_maxima = maxima.view(-1)
        flat_totals.index_add_(0, entry_buckets, gain_values)
        flat_minima.scatter_reduce_(0, entry_buckets, gain_minima, reduce="amin")
        flat_maxima.scatter_reduce_(0, entry_buckets, gain_maxima, reduce="amax")

        layer_entries = [
            index for index, entry in enumerate(gain_entries) if entry[2] is not None
        ]
        if layer_entries:
            layer_indices = [
                gain_entries[index][2] * bucket_count + gain_entries[index][1]
                for index in layer_entries
            ]
            layer_entries = torch.tensor(layer_entries, dtype=torch.long, device=device)
            layer_indices = torch.tensor(layer_indices, dtype=torch.long, device=device)
            flat_totals.index_add_(0, layer_indices, gain_values[layer_entries])
            flat_minima.scatter_reduce_(
                0, layer_indices, gain_minima[layer_entries], reduce="amin"
            )
            flat_maxima.scatter_reduce_(
                0, layer_indices, gain_maxima[layer_entries], reduce="amax"
            )
    nvtx_range_pop(f"{namespace}/logging/gain-statistics")

    nvtx_range_push(f"{namespace}/logging/global-reductions")
    if distributed:
        works = []
        if log_gains:
            works.extend(
                (
                    torch.distributed.all_reduce(totals, async_op=True),
                    torch.distributed.all_reduce(
                        minima, op=torch.distributed.ReduceOp.MIN, async_op=True
                    ),
                    torch.distributed.all_reduce(
                        maxima, op=torch.distributed.ReduceOp.MAX, async_op=True
                    ),
                    torch.distributed.all_reduce(matrix_axis_totals, async_op=True),
                )
            )
        works.append(torch.distributed.all_reduce(matrix_totals, async_op=True))
        for work in works:
            work.wait()
    nvtx_range_pop(f"{namespace}/logging/global-reductions")

    nvtx_range_push(f"{namespace}/logging/cpu-transfer")
    totals, minima, maxima, matrix_axis_totals, matrix_totals = (
        tensor.cpu().tolist()
        for tensor in (totals, minima, maxima, matrix_axis_totals, matrix_totals)
    )
    nvtx_range_pop(f"{namespace}/logging/cpu-transfer")
    stats = {}
    nvtx_range_push(f"{namespace}/logging/format-statistics")
    for scope in range(scope_count):
        prefix = namespace if scope == 0 else f"{namespace}/layers/{scope - 1}"
        _append_md_gain_stats(
            stats,
            totals[scope],
            minima[scope],
            maxima[scope],
            matrix_axis_totals[scope],
            matrix_totals[scope],
            sparsity_thresholds,
            prefix=prefix,
        )
    nvtx_range_pop(f"{namespace}/logging/format-statistics")
    nvtx_range_pop(f"{namespace}/logging")
    return stats
