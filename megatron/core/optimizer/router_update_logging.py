# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-expert relative update statistics for MoE router weights.

Answers "does the router move too fast at the step where z-loss jumps?". Each router row W_i
is one expert's routing vector. Around one optimizer step we log, per layer:

- ``rel_update``: ||W_i' - W_i|| / ||W_i||, the per-expert relative update (max/mean/median).
- ``angle``: the rotation of each row's direction in radians, independent of its norm, so it
  isolates direction updates (Muon/Adam) from magnitude changes (gains) (max/mean).
- ``norm_change``: | ||W_i'|| / ||W_i|| - 1 |, the magnitude part of the update (max).
- ``row_norm``: ||W_i'|| after the step (max), the logit scale ceiling for a unit-RMS input.
- ``frob_rel_update``: ||W' - W||_F / ||W||_F for the whole router matrix.

Rows are read from the optimizer's own (fp32 master) parameters, so under the layer-wise
distributed optimizer only the owning rank contributes and no all-gather is needed. Router
weights are replicated across TP/DP, so a MAX all-reduce over the world merges ranks: every
copy of a layer holds identical values and ranks without the layer contribute zeros.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch

_STATS = (
    "rel_update_max",
    "rel_update_mean",
    "rel_update_median",
    "angle_max",
    "angle_mean",
    "norm_change_max",
    "row_norm_max",
    "frob_rel_update",
)
# Extra column marking which layers reported, so a layer with genuinely zero updates is still
# distinguishable from a layer that is not present on this rank.
_PRESENT = len(_STATS)

_snapshot: Optional[List[Tuple[torch.Tensor, int, torch.Tensor]]] = None
_last_stats: Optional[Dict[str, float]] = None


def _router_params(optimizer) -> List[Tuple[torch.Tensor, int]]:
    """Router weight params the optimizer steps on this rank, with their 0-based layer index."""
    wrapped_optimizers = getattr(optimizer, "chained_optimizers", (optimizer,))
    params = []
    seen = set()
    for wrapped in wrapped_optimizers:
        inner = getattr(wrapped, "optimizer", wrapped)
        for group in getattr(inner, "param_groups", ()):
            for param in group["params"]:
                if not getattr(param, "is_router", False) or param.ndim != 2:
                    continue
                if id(param) in seen:
                    continue
                layer = getattr(param, "md_gain_log_layer", None)
                if layer is None:
                    continue
                seen.add(id(param))
                params.append((param, int(layer)))
    return params


@torch.no_grad()
def snapshot_router_weights(optimizer) -> None:
    """Record router weights before the optimizer step."""
    global _snapshot
    _snapshot = [
        (param, layer, param.detach().to(torch.float32, copy=True))
        for param, layer in _router_params(optimizer)
    ]


@torch.no_grad()
def collect_router_update_stats(num_layers: int) -> None:
    """Compare router weights with the pre-step snapshot and reduce per-layer statistics.

    Must be called on every rank (it all-reduces), right after the optimizer step.
    """
    global _snapshot, _last_stats
    snapshot, _snapshot = _snapshot, None
    if snapshot is None:
        return

    device = (
        snapshot[0][0].device
        if snapshot
        else torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    stats = torch.zeros((num_layers, len(_STATS) + 1), dtype=torch.float32, device=device)
    for param, layer, before in snapshot:
        if not 0 <= layer < num_layers:
            continue
        after = param.detach().to(torch.float32)
        norm_before = before.norm(dim=1).clamp_min(1e-12)
        norm_after = after.norm(dim=1)
        rel_update = (after - before).norm(dim=1) / norm_before
        # atan2(|perpendicular|, parallel) instead of arccos(cosine): arccos is ill-conditioned
        # near 1 (fp32 noise ~3e-4 rad), which is the size of typical per-step router rotations.
        unit_before = before / norm_before.unsqueeze(1)
        parallel = (after * unit_before).sum(dim=1)
        perpendicular = (after - parallel.unsqueeze(1) * unit_before).norm(dim=1)
        angle = torch.atan2(perpendicular, parallel)
        frob_rel = (after - before).norm() / before.norm().clamp_min(1e-12)
        stats[layer] = torch.stack(
            (
                rel_update.max(),
                rel_update.mean(),
                rel_update.median(),
                angle.max(),
                angle.mean(),
                (norm_after / norm_before - 1.0).abs().max(),
                norm_after.max(),
                frob_rel,
                torch.ones((), device=device),
            )
        )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)

    values = stats.cpu().tolist()
    result = {}
    global_max = {name: 0.0 for name in ("rel_update_max", "angle_max", "norm_change_max")}
    for layer, row in enumerate(values):
        if row[_PRESENT] == 0.0:
            continue
        for index, name in enumerate(_STATS):
            if math.isfinite(row[index]):
                result[f"router_update/{name}_layer_{layer}"] = row[index]
        for name in global_max:
            global_max[name] = max(global_max[name], row[_STATS.index(name)])
    if result:
        for name, value in global_max.items():
            result[f"router_update/{name}"] = value
    _last_stats = result


def pop_router_update_stats() -> Optional[Dict[str, float]]:
    """Return and clear the statistics from the most recent collection."""
    global _last_stats
    stats, _last_stats = _last_stats, None
    return stats
