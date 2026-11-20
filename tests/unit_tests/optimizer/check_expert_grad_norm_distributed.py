"""Four-GPU reproduction; no checkpoint, dataset, or optimizer update required.

Run from an environment configured for swiss-ai/Megatron-LM-MoE:
  PYTHONPATH=$PWD python -m torch.distributed.run --standalone \
      --nproc-per-node=4 tests/unit_tests/optimizer/check_expert_grad_norm_distributed.py

Exit 0 means the norm and clipping match the independent reference in all cases.
"""
import json
import math
import os
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from megatron.core import parallel_state as ps
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.clip_grads import (
    clip_grad_by_total_norm_fp32,
    get_grad_norm_fp32,
)
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import (
    set_defaults_if_not_set_tensor_model_parallel_attributes as set_defaults,
)
from megatron.core.transformer.moe.experts import OffloadingExpertsMLP


def reference_norm(entries):
    # Independent oracle: deduplicate logical parameter identities, not TP tags.
    local = [(key, grad.detach().double().cpu()) for key, grad in entries]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    unique = {}
    for rank_entries in gathered:
        for key, grad in rank_entries:
            if key in unique:
                torch.testing.assert_close(grad, unique[key], rtol=0, atol=0)
            else:
                unique[key] = grad
    return math.sqrt(sum(g.square().sum().item() for g in unique.values()))


def run_case(tp, merged):
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tp, pipeline_model_parallel_size=1,
        expert_model_parallel_size=2, expert_tensor_parallel_size=1,
        order="tp-cp-ep-dp-pp", create_gloo_process_groups=False,
    )
    groups = ProcessGroupCollection.use_mpu_process_groups()
    groups.dp_cp = ps.get_data_parallel_group(with_context_parallel=True)
    groups.expt_dp = ps.get_expert_data_parallel_group()
    ep_rank = groups.ep.rank()
    experts, hidden, width = 2, 8, 16

    def allocate(shape):
        return torch.nn.Parameter(torch.empty(
            shape, dtype=torch.bfloat16, device="cpu" if merged else "cuda",
            pin_memory=merged,
        ))

    if merged:
        # The inplace-FP8 branch stores merged BF16 parameters [E, out, in].
        w1 = allocate((experts, 2 * width, hidden))
        w2 = allocate((experts, hidden, width))
        named = [("weight1", w1), ("weight2", w2)]
    else:
        w1 = [allocate((2 * width, hidden)) for _ in range(experts)]
        w2 = [allocate((hidden, width)) for _ in range(experts)]
        named = [(f"weight1.{i}", p) for i, p in enumerate(w1)]
        named += [(f"weight2.{i}", p) for i, p in enumerate(w2)]

    init = SimpleNamespace(num_local_experts=experts, config=SimpleNamespace(
        params_dtype=torch.bfloat16, init_method=torch.nn.init.ones_,
        output_layer_init_method=torch.nn.init.ones_,
    ))
    # Execute the repository's initializer, including its TP-attribute writes.
    OffloadingExpertsMLP._init_expert_weights_like_te(
        init, w1, w2, 2 * width, hidden, hidden, width, True,
    )
    for _, p in named:
        p.allreduce = False
        p.expert_tp = True
        if merged:
            p.merged_offload_expert = True
        set_defaults(p)
        assert p.tensor_model_parallel == (not merged)

    dense = torch.nn.Parameter(torch.ones(hidden, device="cuda", dtype=torch.bfloat16))
    dense.allreduce = True
    set_defaults(dense)
    names = {id(p): f"expert.{ep_rank}.{name}" for name, p in named}
    names[id(dense)] = "dense_replicated"

    # SGD only supplies param_groups. The real LayerWise/BF16 wrappers perform
    # ownership, master creation, gradient preparation and norm collection.
    raw = torch.optim.SGD([
        {"params": [p for _, p in named], "is_expert_parallel": True},
        {"params": [dense], "is_expert_parallel": False},
    ], lr=0.01)
    optimizer = LayerWiseDistributedOptimizer(
        [raw], OptimizerConfig(bf16=True, clip_grad=1.0,
                              use_distributed_optimizer=False,
                              overlap_param_gather=False),
        pg_collection=groups, async_allgather=False,
    )
    # Explicit already-synchronized gradients: EP0=1, EP1=2, dense=1.
    # Expert-DP peers have identical gradients; different EP IDs are unique.
    for _, p in named:
        p.main_grad = torch.full_like(p, 1 + ep_rank, dtype=torch.float32)
    dense.main_grad = torch.ones_like(dense, dtype=torch.float32)
    assert optimizer.prepare_grads() is False

    entries, corrected, owned = [], [], []
    for child in optimizer.chained_optimizers:
        selected = {id(g) for g in child.get_main_grads_for_grad_norm()}
        for models, masters in zip(child.float16_groups, child.fp32_from_float16_groups):
            for model, master in zip(models, masters):
                assert master.grad is not None
                assert master.allreduce == model.allreduce
                entries.append((names[id(model)], master.grad))
                owned.append({"key": names[id(model)],
                              "included": id(master.grad) in selected,
                              "master_tp": master.tensor_model_parallel,
                              "master_allreduce": getattr(master, "allreduce", None)})
                # Diagnostic selector only; no source or parameter mutation.
                group = groups.tp if model.allreduce else groups.expt_tp
                if master.tensor_model_parallel or group.rank() == 0:
                    corrected.append(master.grad)

    expected = reference_norm(entries)
    # Analytic cross-check independent of both selectors and rank assignment.
    assert math.isclose(expected, math.sqrt(768 * (1**2 + 2**2) + 8), rel_tol=1e-12)
    original = float(optimizer.get_grad_norm())
    fixed = float(get_grad_norm_fp32(corrected, grad_stats_parallel_group=None))
    assert math.isclose(fixed, expected, rel_tol=2e-6)
    assert math.isclose(original, expected, rel_tol=2e-6)

    clipped = {}
    for label, norm in (("original", original), ("corrected", fixed)):
        copies = [torch.nn.Parameter(torch.zeros_like(g)) for _, g in entries]
        for p, (_, grad) in zip(copies, entries):
            p.grad = grad.clone()
        clip_grad_by_total_norm_fp32(copies, 1.0, norm)
        clipped[label] = reference_norm([(key, p.grad) for (key, _), p in zip(entries, copies)])
    assert math.isclose(clipped["corrected"], 1.0, abs_tol=3e-6)
    assert math.isclose(clipped["original"], 1.0, abs_tol=3e-6)
    # Every unique element is now zero, so count must include both EP owners
    # and exactly one copy of the replicated dense parameter.
    for _, grad in entries:
        grad.zero_()
    zero_count = optimizer.count_zeros()
    assert zero_count == 768 * 2 + 8, zero_count
    ownership = [None] * dist.get_world_size()
    dist.all_gather_object(ownership, {
        "rank": dist.get_rank(), "tp_rank": groups.tp.rank(),
        "ep_rank": ep_rank, "owned": owned,
    })
    if dist.get_rank() == 0:
        print(json.dumps({"tp": tp, "ep": 2, "etp": 1,
                          "layout": "merged" if merged else "ordinary",
                          "original_norm": original, "reference_norm": expected,
                          "corrected_norm": fixed, "actual_norm_after_clip_1": clipped,
                          "zero_count": zero_count,
                          "ownership": ownership}), flush=True)
    dist.barrier()
    ps.destroy_model_parallel()


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    assert dist.get_world_size() == 4
    if dist.get_rank() == 0:
        print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                          "nccl": torch.cuda.nccl.version(),
                          "gpu": torch.cuda.get_device_name()}), flush=True)
    for tp, merged in ((2, True), (1, True), (2, False)):
        run_case(tp, merged)
    dist.destroy_process_group()
