# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing import load
from megatron.core.dist_checkpointing import save
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory, apply_factories
import megatron.core.optimizer.layer_wise_optimizer as layer_wise_module
import megatron.core.optimizer.md_decoupling as md_module
from megatron.core.optimizer import HAVE_EMERGING_OPTIMIZERS
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.md_decoupling import MDDecoupling
from megatron.core.optimizer.md_decoupling import _get_muon_scale_factor
from megatron.core.optimizer.md_decoupling import _glu_fc1_split_dim
from megatron.core.optimizer.md_decoupling import _md_init_state_fn
from megatron.core.optimizer.md_decoupling import _split_qkv
from megatron.core.optimizer.md_decoupling import get_megatron_mddecoupling_optimizer
from megatron.core.optimizer.optimizer import FP32Optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.core.transformer.moe.fp8_utils import make_fused_experts_sharded_factory
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


requires_cuda_and_emerging = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_EMERGING_OPTIMIZERS,
    reason="CUDA and emerging_optimizers are required for MDDecoupling orthogonal updates",
)



requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this MDDecoupling test"
)

class _NoProcessGroups:
    tp = None
    expt_tp = None


def _step_sum_loss(model, input_tensor):
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()


def _record_md_split_output(param, grad, **md_kwargs):
    split_qkv = md_kwargs.pop("split_qkv", True)
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        split_qkv=split_qkv,
        pg_collection=None,
        tp_mode="duplicated",
        **md_kwargs,
    )
    calls = []

    def record_split(split_grad, tp_group, partition_dim, use_radius_scale=False, is_router=False):
        del tp_group, partition_dim, use_radius_scale, is_router
        calls.append(split_grad.detach().clone())
        return torch.full_like(split_grad, float(len(calls)))

    optimizer._orthogonalize_tensor = record_split
    return optimizer._orthogonalize_param(
        param, grad, is_qkv=getattr(param, "is_qkv", False)
    ), calls


def test_md_decoupling_kda_in_proj_split_uses_local_dim0_tp_shapes():
    local_shapes = (2, 2, 3, 3, 1, 4)
    param = torch.nn.Parameter(torch.empty(sum(local_shapes), 5))
    param.is_kda_in_proj = True
    param.kda_split_shapes = tuple(2 * rows for rows in local_shapes)
    param.partition_dim = 0
    grad = torch.arange(param.numel(), dtype=torch.float32).view_as(param)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_kda_in_proj_fn=lambda p: getattr(p, "is_kda_in_proj", False),
    )

    assert [call.shape for call in calls] == [
        torch.Size([rows, param.size(1)]) for rows in local_shapes
    ]
    expected = torch.cat(
        [
            torch.full((rows, param.size(1)), i, dtype=output.dtype)
            for i, rows in enumerate(local_shapes, 1)
        ]
    )
    assert torch.equal(output, expected)


def test_md_decoupling_builder_routes_kda_decay_parameters_to_adam(monkeypatch):
    class _FakeModelChunk:
        def __init__(self):
            self.config = SimpleNamespace(
                num_attention_heads=2,
                num_query_groups=1,
                kv_channels=4,
                multi_latent_attention=False,
                num_layers=1,
                hidden_size=4,
            )
            self.in_proj = torch.nn.Parameter(torch.ones(8, 4))
            self.A_log = torch.nn.Parameter(torch.ones(2))
            self.dt_bias = torch.nn.Parameter(torch.ones(8))
            self.A_log.is_kda_decay_parameter = True
            self.dt_bias.is_kda_decay_parameter = True

        def named_parameters(self):
            return iter(
                [
                    ("decoder.layers.0.self_attention.in_proj.weight", self.in_proj),
                    ("decoder.layers.0.self_attention.A_log", self.A_log),
                    ("decoder.layers.0.self_attention.dt_bias", self.dt_bias),
                ]
            )

    model = _FakeModelChunk()
    captured = {}

    def fake_get_param_groups(model_chunks, config, config_overrides):
        del config, config_overrides
        params = [
            p
            for model_chunk in model_chunks
            for _, p in model_chunk.named_parameters()
            if p.requires_grad
        ]
        return [{"params": params, "is_expert_parallel": False}]

    def fake_md_decoupling(params, **kwargs):
        del kwargs
        captured["md"] = [p for group in params for p in group["params"]]
        return SimpleNamespace(param_groups=params, state={})

    def fake_get_megatron_optimizer(config, model_chunks, **kwargs):
        del kwargs
        captured["scalar_optimizer"] = config.optimizer
        captured["adam"] = [
            p
            for model_chunk in model_chunks
            for _, p in model_chunk.named_parameters()
            if p.requires_grad
        ]
        return SimpleNamespace(chained_optimizers=[])

    monkeypatch.setattr(md_module, "_get_param_groups", fake_get_param_groups)
    monkeypatch.setattr(md_module, "MDDecoupling", fake_md_decoupling)
    monkeypatch.setattr(md_module, "FP32Optimizer", lambda optimizer, *args: optimizer)
    monkeypatch.setattr(md_module, "get_megatron_optimizer", fake_get_megatron_optimizer)
    monkeypatch.setattr(
        md_module,
        "ChainedOptimizer",
        lambda optimizers: SimpleNamespace(chained_optimizers=optimizers),
    )

    config = OptimizerConfig(
        optimizer="md_decoupling",
        lr=0.01,
        min_lr=0.0,
        use_orthogonal_updates=False,
    )
    config.hypersphere_mode = None
    config.hypersphere_embedding_mode = None
    config.hypersphere_router_mode = None
    config.hypersphere_gains_mode = None
    config.use_distributed_optimizer = False
    config.fp16 = False
    config.bf16 = False
    overrides = md_module._mddecoupling_config_overrides(config, {})
    matrix_key = next(
        key
        for key in overrides
        if getattr(key.predicate, "name", None) == "md_non_embedding_or_output_matrix"
    )
    assert matrix_key.predicate(model.in_proj)
    assert not matrix_key.predicate(model.A_log)
    assert not matrix_key.predicate(model.dt_bias)
    md_module.get_megatron_mddecoupling_optimizer(
        config,
        [model],
        config_overrides={},
        pg_collection=_NoProcessGroups(),
    )

    assert len(captured["md"]) == 1 and captured["md"][0] is model.in_proj
    assert len(captured["adam"]) == 2
    assert captured["scalar_optimizer"] == "adam"
    assert captured["adam"][0] is model.A_log
    assert captured["adam"][1] is model.dt_bias


@requires_cuda_and_emerging
@pytest.mark.parametrize(
    ("preserve_init", "expected_weight_norm"),
    [(False, math.sqrt(2.0)), (True, 13.0)],
)
def test_md_muon_normalizes_update_to_fixed_hypersphere_norm(
    monkeypatch, preserve_init, expected_weight_norm
):
    param = torch.nn.Parameter(
        torch.tensor([[3.0, 4.0], [0.0, 12.0]], device="cuda")
    )
    optimizer = MDDecoupling(
        params=[param],
        lr=0.1,
        weight_decay=0.0,
        hypersphere_mode="flat",
        hypersphere_preserve_init=preserve_init,
        use_orthogonal_updates=True,
        momentum_beta=0.0,
        use_nesterov=False,
        normalize_update_to_weight_norm=True,
        pg_collection=_NoProcessGroups(),
        tp_mode="duplicated",
    )
    raw_update = torch.tensor([[1.0, -2.0], [3.0, -4.0]], device="cuda")
    monkeypatch.setattr(
        optimizer,
        "_orthogonalize_tensor",
        lambda *args, **kwargs: raw_update.clone(),
    )
    # Disable the post-step projection so the exact applied update can be recovered from the
    # parameter delta; the first-step cache happens before that projection and weight decay.
    monkeypatch.setattr(optimizer, "_normalize", lambda *args, **kwargs: None)
    before = param.detach().clone()
    param.grad = torch.ones_like(param)

    optimizer.step()

    applied_update = (before - param) / 0.1
    fixed_norm = expected_weight_norm
    assert torch.allclose(
        torch.linalg.vector_norm(applied_update),
        torch.tensor(fixed_norm, device="cuda"),
    )
    assert torch.allclose(
        applied_update / torch.linalg.vector_norm(applied_update),
        raw_update / torch.linalg.vector_norm(raw_update),
    )
    assert optimizer._fixed_weight_norms[param][0].item() == pytest.approx(fixed_norm)

    # The target is cached; a subsequent call measures only the update, not the weight.
    norm_calls = 0
    compiled_squared_norm = md_module._local_squared_norm

    def count_squared_norm(tensor):
        nonlocal norm_calls
        norm_calls += 1
        return compiled_squared_norm(tensor)

    monkeypatch.setattr(md_module, "_local_squared_norm", count_squared_norm)
    second = optimizer._normalize_muon_update_blocks(
        param, [raw_update * 3], None
    )[0]
    assert norm_calls == 1
    assert torch.linalg.vector_norm(second).item() == pytest.approx(fixed_norm, rel=1e-6)


@requires_cuda_and_emerging
def test_md_update_weight_norm_supersedes_shape_up_scaling(monkeypatch):
    param = torch.nn.Parameter(torch.ones((3, 2), device="cuda"))
    optimizer = MDDecoupling(
        params=[param],
        lr=0.1,
        hypersphere_mode="flat",
        hypersphere_radius_mode="fan_in",
        scale_mode="spectral",
        normalize_update_to_weight_norm=True,
        pg_collection=_NoProcessGroups(),
        tp_mode="duplicated",
    )
    monkeypatch.setattr(md_module, "newton_schulz_tp", lambda grad, **kwargs: grad)
    monkeypatch.setattr(
        md_module,
        "_get_muon_scale_factor",
        lambda *args, **kwargs: pytest.fail("shape scaling must be skipped"),
    )

    update = torch.arange(1, 7, dtype=torch.float32, device="cuda").view(3, 2)
    torch.testing.assert_close(optimizer._orthogonalize_tensor(update, None, None), update)


@requires_cuda
def test_md_update_weight_norm_is_applied_per_existing_logical_split():
    common = dict(
        lr=0.1,
        hypersphere_mode="flat",
        hypersphere_preserve_init=True,
        normalize_update_to_weight_norm=True,
        pg_collection=_NoProcessGroups(),
        tp_mode="duplicated",
    )
    flag = lambda name: lambda p: getattr(p, name, False)
    specs = [
        (
            (8, 4),
            {"is_qkv": True},
            True,
            False,
            dict(split_qkv=True, is_qkv_fn=flag("is_qkv"), qkv_split_shapes=(4, 2, 2)),
        ),
        ((8, 4), {"glu_split_dim": 0}, False, False, dict(split_fc1=True)),
        (
            (6, 4),
            {"is_qkv_down_proj": True},
            False,
            False,
            dict(
                split_qkv=True,
                is_qkv_down_proj_fn=flag("is_qkv_down_proj"),
                qkv_down_proj_split_shapes=(4, 2),
            ),
        ),
        (
            (4, 8),
            {"is_kv_up_proj": True},
            False,
            False,
            dict(
                split_qkv=True,
                split_mla_per_head=True,
                is_kv_up_proj_fn=flag("is_kv_up_proj"),
                kv_up_proj_split_shapes=(1, 1),
            ),
        ),
        (
            (6, 4),
            {"is_q_up_proj": True},
            False,
            False,
            dict(
                split_mla_per_head=True,
                is_q_up_proj_fn=flag("is_q_up_proj"),
                q_up_proj_head_dim=2,
            ),
        ),
        (
            (2, 8, 3),
            {"glu_split_dim": 1, "merged_offload_expert": True},
            False,
            True,
            dict(split_fc1=True),
        ),
        ((2, 4, 3), {"merged_offload_expert": True}, False, True, {}),
    ]

    for shape, attrs, is_qkv, is_merged_expert, optimizer_kwargs in specs:
        param = torch.nn.Parameter(
            torch.arange(1, math.prod(shape) + 1, dtype=torch.float32, device="cuda").view(shape)
        )
        for name, value in attrs.items():
            setattr(param, name, value)
        optimizer = MDDecoupling(params=[param], **common, **optimizer_kwargs)
        optimizer._cache_fixed_weight_norms(param, is_qkv, False, False, is_merged_expert)
        optimizer._orthogonalize_tensor = lambda grad, *args, **kwargs: grad
        normalized = optimizer._orthogonalize_param(
            param,
            torch.linspace(0.1, 3.0, param.numel(), device="cuda").view_as(param),
            is_qkv=is_qkv,
            is_merged_offload_expert=is_merged_expert,
        )
        weight_blocks, _ = optimizer._logical_blocks(param, param, is_qkv, is_merged_expert)
        update_blocks, _ = optimizer._logical_blocks(param, normalized, is_qkv, is_merged_expert)

        assert len(weight_blocks) > 1
        torch.testing.assert_close(
            torch.stack([torch.linalg.vector_norm(block) for block in update_blocks]),
            torch.stack([torch.linalg.vector_norm(block) for block in weight_blocks]),
        )


def _gqa_qkv_optimizer(param, **kwargs):
    return MDDecoupling(
        params=[param],
        lr=0.01,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=(4, 2, 2),
        pg_collection=_NoProcessGroups(),
        tp_mode="duplicated",
        **kwargs,
    )


def _glu_fc1_optimizer(param, **kwargs):
    return MDDecoupling(
        params=[param],
        lr=0.01,
        split_fc1=True,
        pg_collection=_NoProcessGroups(),
        tp_mode="duplicated",
        **kwargs,
    )


def _assert_qkv_split_flat_norms(optimizer, tensor, expected_norm):
    parts = _split_qkv(tensor, optimizer.qkv_split_shapes)
    expected = torch.full((len(parts),), expected_norm, dtype=tensor.dtype, device=tensor.device)
    actual = torch.stack([torch.linalg.vector_norm(part) for part in parts])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _assert_qkv_split_tangent(optimizer, param, grad):
    p_parts = _split_qkv(param, optimizer.qkv_split_shapes)
    g_parts = _split_qkv(grad, optimizer.qkv_split_shapes)
    residuals = torch.stack(
        [
            (p_part * g_part).sum().abs()
            / (torch.linalg.vector_norm(p_part) * torch.linalg.vector_norm(g_part)).clamp_min(1e-12)
            for p_part, g_part in zip(p_parts, g_parts)
        ]
    )
    torch.testing.assert_close(residuals, torch.zeros_like(residuals), rtol=1e-5, atol=1e-6)


def _mla_kv_up_proj_optimizer(param, **kwargs):
    return MDDecoupling(
        params=[param],
        lr=0.01,
        split_qkv=True,
        is_kv_up_proj_fn=lambda p: getattr(p, "is_kv_up_proj", False),
        kv_up_proj_split_shapes=(1, 1),
        split_mla_per_head=True,
        pg_collection=_NoProcessGroups(),
        tp_mode="duplicated",
        **kwargs,
    )


def _assert_split_flat_norms(optimizer, param, tensor, expected_norm, is_qkv=False):
    parts, _ = optimizer._split_param_tensor(param, tensor, is_qkv=is_qkv)
    expected = torch.full((len(parts),), expected_norm, dtype=tensor.dtype, device=tensor.device)
    actual = torch.stack([torch.linalg.vector_norm(part) for part in parts])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _assert_split_tangent(optimizer, param, grad, is_qkv):
    p_parts, _ = optimizer._split_param_tensor(param, param, is_qkv=is_qkv)
    g_parts, _ = optimizer._split_param_tensor(param, grad, is_qkv=is_qkv)
    residuals = torch.stack(
        [
            (p_part * g_part).sum().abs()
            / (torch.linalg.vector_norm(p_part) * torch.linalg.vector_norm(g_part)).clamp_min(1e-12)
            for p_part, g_part in zip(p_parts, g_parts)
        ]
    )
    torch.testing.assert_close(residuals, torch.zeros_like(residuals), rtol=1e-5, atol=1e-6)


def _bare_param_from_gains(optimizer, param):
    state = optimizer.state[param]
    bare_param = param.detach().clone()
    if "flat_gain" in state:
        bare_param.div_(optimizer._phi(state["flat_gain"]))
    if "row_gain" in state:
        bare_param.div_(optimizer._phi(state["row_gain"])[:, None])
    if "col_gain" in state:
        bare_param.div_(optimizer._phi(state["col_gain"])[None, :])
    return bare_param


class _TinyMDDecouplingModel(torch.nn.Module):
    def __init__(self, shared_output=False, offloading_expert=False, device="cpu"):
        super().__init__()
        self.config = SimpleNamespace(
            context_parallel_size=1,
            hidden_size=8,
            kv_channels=4,
            moe_use_inplace_fp8_param=offloading_expert,
            moe_use_offloading_experts=offloading_expert,
            num_attention_heads=2,
            num_layers=1,
            num_query_groups=1,
        )
        self.ddp_config = SimpleNamespace(
            num_distributed_optimizer_instances=1,
            use_distributed_optimizer=False,
            use_megatron_fsdp=False,
        )
        self.embedding = torch.nn.Module()
        self.embedding.word_embeddings = torch.nn.Embedding(8, 8, device=device)
        self.output_layer = torch.nn.Linear(8, 8, bias=False, device=device)
        self.router = torch.nn.Linear(8, 4, bias=False, device=device)
        self.attn = torch.nn.Module()
        self.attn.linear_qkv = torch.nn.Linear(8, 24, bias=False, device=device)
        self.mlp = torch.nn.Module()
        self.mlp.linear_fc2 = torch.nn.Linear(8, 8, bias=False, device=device)
        self.norm = torch.nn.LayerNorm(8, device=device)
        if offloading_expert:
            self.experts = torch.nn.Module()
            self.experts.weight2 = torch.nn.Parameter(torch.ones(2, 8, 8, device=device))

        self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
        self.output_layer.weight.is_embedding_or_output_parameter = True
        if shared_output:
            self.output_layer.weight.shared_embedding = True


def test_md_decoupling_recipe_defaults():
    config = OptimizerConfig()

    assert config.hypersphere_mode == "flat"
    assert config.hypersphere_embedding_mode == "row"
    assert config.hypersphere_router_mode == "row"
    assert config.hypersphere_radius_mode == "shape_native"
    assert config.muon_tp_mode == "duplicated"
    assert config.hypersphere_gains_mode == "rowcol"
    assert config.hypersphere_gains_mode_output == "inherit"
    assert config.hypersphere_gains_mode_embedding == "none"
    assert config.hypersphere_gains_mode_router == "rowcol"
    assert config.use_orthogonal_updates is True
    assert config.gain_parametrization == "softplus"
    assert config.muon_router_scale_mode == "none"
    assert config.muon_split_fc1 is True


def test_md_decoupling_router_scale_mode_resolution():
    # Default: routers get "none" (constant 1.0) while matrices follow scale_mode. A router of
    # shape (num_experts, hidden) is non-square, so shape_up would give >1; "none" pins it to 1.
    optimizer = MDDecoupling(
        params=[torch.nn.Parameter(torch.ones(2, 2))],
        lr=0.01,
        scale_mode="shape_up",
        pg_collection=None,
    )
    assert optimizer.router_scale_mode == "none"
    assert optimizer._resolve_scale_mode(is_router=True) == "none"
    assert optimizer._resolve_scale_mode(is_router=False) == "shape_up"
    # The resolved router mode yields a constant 1.0 regardless of the router's aspect ratio,
    # so the update is width-invariant (num_experts=128 fixed while hidden scales).
    for hidden in (128, 768, 1536):
        assert _get_muon_scale_factor(
            128, hidden, mode=optimizer._resolve_scale_mode(is_router=True)
        ) == 1.0
    # shape_up on a router WOULD track width — this is the behavior being excluded.
    assert _get_muon_scale_factor(128, 768, mode="shape_up") > 1.0

    # Override: routers can be made to follow a mode explicitly.
    overridden = MDDecoupling(
        params=[torch.nn.Parameter(torch.ones(2, 2))],
        lr=0.01,
        scale_mode="shape_up",
        router_scale_mode="spectral",
        pg_collection=None,
    )
    assert overridden._resolve_scale_mode(is_router=True) == "spectral"
    assert overridden._resolve_scale_mode(is_router=False) == "shape_up"


def test_md_decoupling_router_gains_mode_override():
    param = torch.nn.Parameter(torch.ones(2, 2))
    param.is_router = True
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        hypersphere_gains_mode="rowcol",
        hypersphere_gains_mode_router="none",
        pg_collection=None,
    )

    assert optimizer._resolve_gains_mode(param) == "none"


def test_md_decoupling_default_gains_mode_resolution():
    normal = torch.nn.Parameter(torch.ones(2, 2))
    embedding = torch.nn.Parameter(torch.ones(2, 2))
    output = torch.nn.Parameter(torch.ones(2, 2))
    router = torch.nn.Parameter(torch.ones(2, 2))
    embedding.is_md_embedding_parameter = True
    output.is_md_output_parameter = True
    router.is_router = True
    optimizer = MDDecoupling(
        params=[normal, embedding, output, router],
        lr=0.01,
        hypersphere_gains_mode="rowcol",
        hypersphere_gains_mode_output="inherit",
        hypersphere_gains_mode_embedding="none",
        hypersphere_gains_mode_router="rowcol",
        pg_collection=None,
    )

    assert optimizer._resolve_gains_mode(normal) == "rowcol"
    assert optimizer._resolve_gains_mode(embedding) == "none"
    assert optimizer._resolve_gains_mode(output) == "rowcol"
    assert optimizer._resolve_gains_mode(router) == "rowcol"


def test_md_decoupling_gains_mode_none_disables_gain_state():
    param = torch.nn.Parameter(torch.ones(2, 2))
    param.is_router = True
    param.grad = torch.ones_like(param)
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        hypersphere_gains_mode="none",
        hypersphere_gains_mode_router="rowcol",
        use_orthogonal_updates=False,
        pg_collection=None,
    )

    optimizer.step()

    gain_state_keys = {
        "flat_gain",
        "flat_gain_m",
        "flat_gain_v",
        "row_gain",
        "row_gain_m",
        "row_gain_v",
        "col_gain",
        "col_gain_m",
        "col_gain_v",
    }
    assert optimizer.hypersphere_gains_mode is None
    assert optimizer._resolve_gains_mode(param) is None
    assert gain_state_keys.isdisjoint(optimizer.state[param])


def test_md_decoupling_embedding_gain_override_wins_for_tied_output():
    param = torch.nn.Parameter(torch.ones(2, 2))
    param.is_md_embedding_parameter = True
    param.is_md_output_parameter = True
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        hypersphere_gains_mode="rowcol",
        hypersphere_gains_mode_embedding="none",
        hypersphere_gains_mode_output="flat",
        pg_collection=None,
    )

    assert optimizer._resolve_gains_mode(param) == "none"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="optimizer wrapper creates CUDA scale")
def test_md_decoupling_builder_tags_embedding_output_and_shared_output():
    Utils.initialize_model_parallel()
    try:
        untied_model = _TinyMDDecouplingModel(shared_output=False, device="cuda")
        shared_model = _TinyMDDecouplingModel(shared_output=True, device="cuda")
        offload_model = _TinyMDDecouplingModel(
            offloading_expert=True, device="cuda"
        ).bfloat16()
        config = OptimizerConfig(
            optimizer="md_decoupling",
            lr=0.01,
            min_lr=0.0,
            use_orthogonal_updates=False,
        )
        offload_config = OptimizerConfig(
            optimizer="md_decoupling",
            lr=0.01,
            min_lr=0.0,
            bf16=True,
            use_orthogonal_updates=False,
        )

        optimizer = get_megatron_mddecoupling_optimizer(
            config,
            [untied_model],
            use_gloo_process_groups=False,
        )
        shared_optimizer = get_megatron_mddecoupling_optimizer(
            config,
            [shared_model],
            use_gloo_process_groups=False,
        )
        offload_optimizer = get_megatron_mddecoupling_optimizer(
            offload_config,
            [offload_model],
            use_gloo_process_groups=False,
        )
        md_optimizer = optimizer.chained_optimizers[0].optimizer
        shared_md_optimizer = shared_optimizer.chained_optimizers[0].optimizer
        offload_md_optimizer = offload_optimizer.chained_optimizers[0].optimizer

        assert untied_model.embedding.word_embeddings.weight.is_md_embedding_parameter is True
        assert not hasattr(untied_model.embedding.word_embeddings.weight, "is_md_output_parameter")
        assert untied_model.output_layer.weight.is_md_output_parameter is True
        assert not hasattr(untied_model.output_layer.weight, "is_md_embedding_parameter")
        assert untied_model.router.weight.is_router is True
        assert untied_model.attn.linear_qkv.weight.is_qkv is True
        assert untied_model.mlp.linear_fc2.weight.is_out_proj is True
        assert (
            md_optimizer._resolve_gains_mode(untied_model.embedding.word_embeddings.weight)
            == "none"
        )
        assert md_optimizer._resolve_gains_mode(untied_model.output_layer.weight) == "rowcol"
        assert md_optimizer._resolve_gains_mode(untied_model.router.weight) == "rowcol"

        assert shared_model.output_layer.weight.is_md_embedding_parameter is True
        assert not hasattr(shared_model.output_layer.weight, "is_md_output_parameter")
        assert shared_md_optimizer._resolve_gains_mode(shared_model.output_layer.weight) == "none"

        assert offload_model.experts.weight2.expert_tp is True
        assert offload_model.experts.weight2.merged_offload_expert is True
        assert offload_model.experts.weight2.is_out_proj is True
        offload_main_param = offload_model.experts.weight2.main_param
        assert offload_main_param.merged_offload_expert is True
        assert offload_main_param.is_out_proj is True
        assert any(
            p is offload_main_param
            for group in offload_md_optimizer.param_groups
            for p in group["params"]
        )
    finally:
        Utils.destroy_model_parallel()


def test_md_decoupling_direct_gains_no_clamp_min_round_trip():
    param = torch.nn.Parameter(torch.tensor([[2.0, -4.0], [6.0, -8.0]]))
    original = param.detach().clone()
    param.grad = torch.ones_like(param)
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        hypersphere_gains_mode="flat",
        gain_parametrization="direct",
        gains_no_clamp_min=True,
        pg_collection=None,
    )
    optimizer.state[param]["flat_gain"] = torch.tensor(-2.0)

    gain_grads = optimizer._preprocess_gains(param)

    torch.testing.assert_close(param, original / -2.0)
    torch.testing.assert_close(gain_grads["flat_gain"], torch.tensor(2.0))

    optimizer._apply_gains(param)

    torch.testing.assert_close(param, original)


def test_md_decoupling_sharded_state_dict_includes_projected_gain_tensors():
    param = torch.nn.Parameter(torch.ones(3, 4))
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        hypersphere_gains_mode="rowcol",
        pg_collection=None,
    )
    megatron_optimizer = FP32Optimizer(
        optimizer,
        OptimizerConfig(optimizer='md_decoupling'),
        _md_init_state_fn,
    )
    model_sharded_state_dict = {
        'linear.weight': ShardedTensor.from_rank_offsets('linear.weight', param)
    }

    state_dict = megatron_optimizer.sharded_state_dict(
        model_sharded_state_dict,
        is_loading=True,
    )
    param_state = state_dict['state'][0]

    assert isinstance(param_state['exp_avg'], ShardedTensor)
    assert isinstance(param_state['row_gain'], ShardedTensor)
    assert isinstance(param_state['row_gain_m'], ShardedTensor)
    assert isinstance(param_state['row_gain_v'], ShardedTensor)
    assert isinstance(param_state['col_gain'], ShardedTensor)
    assert isinstance(param_state['col_gain_m'], ShardedTensor)
    assert isinstance(param_state['col_gain_v'], ShardedTensor)

    assert param_state['row_gain'].global_shape == (3,)
    assert param_state['col_gain'].global_shape == (4,)


@pytest.mark.parametrize(
    ('partition_dim', 'gain_kind', 'expected_global_shape', 'expected_fragmentations'),
    (
        (0, 'row', (6,), (2,)),
        (0, 'col', (4,), (1,)),
        (0, 'flat', (), ()),
        (1, 'row', (3,), (1,)),
        (1, 'col', (8,), (2,)),
        (1, 'flat', (), ()),
    ),
)
def test_md_gain_projection_turns_dropped_tp_axis_into_replica(
    partition_dim, gain_kind, expected_global_shape, expected_fragmentations
):
    local_shape = (3, 4)
    gain_shape = {'row': (3,), 'col': (4,), 'flat': ()}[gain_kind]

    for tp_rank in (0, 1):
        param = torch.nn.Parameter(torch.ones(local_shape))
        model_shard = ShardedTensor.from_rank_offsets(
            'linear.weight',
            param,
            (partition_dim, tp_rank, 2),
            replica_id=(0, 0, 0),
            allow_shape_mismatch=True,
        )
        optimizer = MDDecoupling(
            params=[param],
            lr=0.01,
            hypersphere_gains_mode='rowcol',
            pg_collection=None,
        )
        gain_shard = optimizer.build_sharded_optimizer_state(
            model_shard,
            torch.ones(gain_shape),
            f'{gain_kind}_gain',
            f'optimizer.state.{gain_kind}_gain',
        )

        assert isinstance(gain_shard, ShardedTensor)
        assert gain_shard.global_shape == expected_global_shape
        assert gain_shard.axis_fragmentations == expected_fragmentations
        assert gain_shard.allow_shape_mismatch is (gain_kind == 'row')

        gain_axis = {'row': 0, 'col': 1, 'flat': None}[gain_kind]
        if gain_axis == partition_dim:
            assert gain_shard.global_offset == (tp_rank * local_shape[gain_axis],)
            assert gain_shard.replica_id == (0, 0, 0)
        else:
            assert gain_shard.global_offset == (0,) * len(expected_global_shape)
            assert gain_shard.replica_id == (0, tp_rank, 0)


def test_md_gain_projection_preserves_layer_and_expert_axes():
    param = torch.nn.Parameter(torch.ones(2, 3, 4))
    model_shard = ShardedTensor.from_rank_offsets(
        'experts.weight',
        param,
        (0, 5, 8),  # layer
        (1, 1, 2),  # expert ownership
        replica_id=(0, 0, 0),
        prepend_axis_num=1,
    )
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        hypersphere_gains_mode='rowcol',
        pg_collection=None,
    )

    expected = {
        'row': ((8, 4, 3), (5, 2, 0), (8, 2, 1), (2, 3)),
        'col': ((8, 4, 4), (5, 2, 0), (8, 2, 1), (2, 4)),
        'flat': ((8, 4), (5, 2), (8, 2), (2,)),
    }
    for gain_kind, (global_shape, global_offset, fragmentations, local_shape) in expected.items():
        gain_shard = optimizer.build_sharded_optimizer_state(
            model_shard,
            torch.ones(local_shape),
            f'{gain_kind}_gain',
            f'optimizer.state.{gain_kind}_gain',
        )
        assert isinstance(gain_shard, ShardedTensor)
        assert gain_shard.prepend_axis_num == 1
        assert gain_shard.global_shape == global_shape
        assert gain_shard.global_offset == global_offset
        assert gain_shard.axis_fragmentations == fragmentations
        assert gain_shard.replica_id == (0, 0, 0)


def test_md_gain_projection_supports_dim0_split_factories():
    param = torch.nn.Parameter(torch.ones(8, 4))
    model_shard = ShardedTensor.from_rank_offsets(
        'linear.weight',
        param,
        (0, 1, 2),
        replica_id=(0, 0, 0),
    )
    model_factory = apply_swiglu_sharded_factory(model_shard, ())
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        hypersphere_gains_mode='rowcol',
        pg_collection=None,
    )

    row_gain = torch.arange(8, dtype=torch.float32)
    row_factory = optimizer.build_sharded_optimizer_state(
        model_factory,
        row_gain,
        'row_gain',
        'optimizer.state.row_gain',
    )
    assert isinstance(row_factory, ShardedTensorFactory)
    row_state = {'row_gain': row_factory}
    apply_factories(row_state)
    row_shards = list(nested_values(row_state))
    assert [shard.local_shape for shard in row_shards] == [(4,), (4,)]
    assert [shard.global_offset for shard in row_shards] == [(4,), (12,)]
    assert all(shard.global_shape == (16,) for shard in row_shards)
    torch.testing.assert_close(row_factory.merge_fn([shard.data for shard in row_shards]), row_gain)

    col_gain = torch.arange(4, dtype=torch.float32)
    col_factory = optimizer.build_sharded_optimizer_state(
        model_factory,
        col_gain,
        'col_gain',
        'optimizer.state.col_gain',
    )
    assert isinstance(col_factory, ShardedTensorFactory)
    col_state = {'col_gain': col_factory}
    apply_factories(col_state)
    col_shards = list(nested_values(col_state))
    assert len(col_shards) == 1
    assert col_shards[0].global_shape == (4,)
    assert col_shards[0].global_offset == (0,)
    assert col_shards[0].replica_id == (0, 1, 0)
    torch.testing.assert_close(col_factory.merge_fn([col_shards[0].data]), col_gain)


def test_md_projected_row_gain_reshards(tmp_path_dist_ckpt):
    Utils.initialize_model_parallel(1, 1)
    try:
        def make_row_shard(rank, fragments, local_rows, values):
            param = torch.nn.Parameter(torch.ones(local_rows, 3))
            model_shard = ShardedTensor.from_rank_offsets(
                'linear.weight',
                param,
                (0, rank, fragments),
                replica_id=(0, 0, 0),
            )
            optimizer = MDDecoupling(
                params=[param],
                lr=0.01,
                hypersphere_gains_mode='row',
                pg_collection=None,
            )
            return optimizer.build_sharded_optimizer_state(
                model_shard,
                values,
                'row_gain',
                'optimizer.state.row_gain',
            )

        source_shards = [
            make_row_shard(0, 2, 4, torch.arange(0, 4, dtype=torch.float32)),
            make_row_shard(1, 2, 4, torch.arange(4, 8, dtype=torch.float32)),
        ]
        destination_shards = [
            make_row_shard(rank, 4, 2, torch.empty(2)) for rank in range(4)
        ]

        with TempNamedDir(tmp_path_dist_ckpt / 'md_projected_gain_reshard', sync=True) as ckpt_dir:
            save({'row_gain': source_shards}, ckpt_dir)
            loaded = load({'row_gain': destination_shards}, ckpt_dir)

        torch.testing.assert_close(
            torch.cat(loaded['row_gain']),
            torch.arange(8, dtype=torch.float32),
        )
    finally:
        Utils.destroy_model_parallel()


@requires_cuda
@pytest.mark.skipif(Utils.world_size < 4, reason="TP resharding test requires four ranks")
@pytest.mark.parametrize(
    ('source_tp', 'destination_tp'), ((1, 4), (2, 4), (4, 2), (4, 1))
)
def test_md_projected_gains_reshard_across_tp(
    tmp_path_dist_ckpt, source_tp, destination_tp
):
    global_rows = 8
    columns = 3

    def make_gain_state(tp_size, source_values):
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        dp_rank = parallel_state.get_data_parallel_rank()
        local_rows = global_rows // tp_size
        row_start = tp_rank * local_rows
        param = torch.nn.Parameter(torch.ones(local_rows, columns, device='cuda'))
        model_shard = ShardedTensor.from_rank_offsets(
            'linear.weight',
            param,
            (0, tp_rank, tp_size),
            replica_id=(0, 0, dp_rank),
        )
        optimizer = MDDecoupling(
            params=[param],
            lr=0.01,
            hypersphere_gains_mode='rowcol',
            pg_collection=None,
        )
        gains = {
            'row_gain': torch.arange(
                row_start, row_start + local_rows, dtype=torch.float32, device='cuda'
            ),
            'col_gain': torch.arange(columns, dtype=torch.float32, device='cuda') + 100,
            'flat_gain': torch.tensor(200.0, device='cuda'),
        }
        if not source_values:
            gains = {name: torch.full_like(value, -1) for name, value in gains.items()}
        return {
            name: optimizer.build_sharded_optimizer_state(
                model_shard,
                value,
                name,
                f'optimizer.state.{name}',
            )
            for name, value in gains.items()
        }

    Utils.initialize_model_parallel(source_tp, 1)
    try:
        with TempNamedDir(
            tmp_path_dist_ckpt / f'md_gain_tp_{source_tp}_to_{destination_tp}', sync=True
        ) as ckpt_dir:
            save(make_gain_state(source_tp, source_values=True), ckpt_dir)
            Utils.destroy_model_parallel()

            Utils.initialize_model_parallel(destination_tp, 1)
            loaded = load(make_gain_state(destination_tp, source_values=False), ckpt_dir)

            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            local_rows = global_rows // destination_tp
            row_start = tp_rank * local_rows
            torch.testing.assert_close(
                loaded['row_gain'],
                torch.arange(
                    row_start,
                    row_start + local_rows,
                    dtype=torch.float32,
                    device='cuda',
                ),
            )
            torch.testing.assert_close(
                loaded['col_gain'],
                torch.arange(columns, dtype=torch.float32, device='cuda') + 100,
            )
            torch.testing.assert_close(loaded['flat_gain'], torch.tensor(200.0, device='cuda'))
    finally:
        Utils.destroy_model_parallel()


@requires_cuda
@pytest.mark.skipif(Utils.world_size < 4, reason="TP resharding test requires four ranks")
def test_md_output_row_gain_reshards_across_tp_with_different_padded_vocab_size(
    tmp_path_dist_ckpt,
):
    source_tp, source_padded_vocab = 2, 10
    destination_tp, destination_padded_vocab = 4, 12
    columns = 3
    state_offsets = {'row_gain': 0, 'row_gain_m': 100, 'row_gain_v': 200}

    def make_gain_state(tp_size, padded_vocab_size, source_values):
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        dp_rank = parallel_state.get_data_parallel_rank()
        local_rows = padded_vocab_size // tp_size
        row_start = tp_rank * local_rows
        param = torch.nn.Parameter(torch.ones(local_rows, columns, device='cuda'))
        model_shard = ShardedTensor.from_rank_offsets(
            'output_layer.weight',
            param,
            (0, tp_rank, tp_size),
            replica_id=(0, 0, dp_rank),
            allow_shape_mismatch=True,
        )
        optimizer = MDDecoupling(
            params=[param],
            lr=0.01,
            hypersphere_gains_mode='row',
            pg_collection=None,
        )
        return {
            state_key: optimizer.build_sharded_optimizer_state(
                model_shard,
                torch.arange(
                    row_start,
                    row_start + local_rows,
                    dtype=torch.float32,
                    device='cuda',
                )
                + offset
                if source_values
                else torch.full((local_rows,), -1.0, device='cuda'),
                state_key,
                f'optimizer.state.{state_key}',
            )
            for state_key, offset in state_offsets.items()
        }

    Utils.initialize_model_parallel(source_tp, 1)
    try:
        with TempNamedDir(
            tmp_path_dist_ckpt / 'md_output_gain_padded_vocab_tp_2_to_4', sync=True
        ) as ckpt_dir:
            save(make_gain_state(source_tp, source_padded_vocab, source_values=True), ckpt_dir)
            Utils.destroy_model_parallel()

            Utils.initialize_model_parallel(destination_tp, 1)
            loaded = load(
                make_gain_state(
                    destination_tp,
                    destination_padded_vocab,
                    source_values=False,
                ),
                ckpt_dir,
            )

            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            local_rows = destination_padded_vocab // destination_tp
            row_start = tp_rank * local_rows
            for state_key, offset in state_offsets.items():
                expected = torch.cat(
                    (
                        torch.arange(
                            source_padded_vocab,
                            dtype=torch.float32,
                            device='cuda',
                        )
                        + offset,
                        torch.zeros(
                            destination_padded_vocab - source_padded_vocab,
                            device='cuda',
                        ),
                    )
                )
                torch.testing.assert_close(
                    loaded[state_key], expected.narrow(0, row_start, local_rows)
                )
    finally:
        Utils.destroy_model_parallel()


@requires_cuda
@pytest.mark.skipif(Utils.world_size < 4, reason="EP resharding test requires four ranks")
@pytest.mark.parametrize(
    ('source_ep', 'destination_ep'), ((1, 4), (2, 4), (4, 2), (4, 1))
)
def test_md_projected_gains_reshard_across_ep(
    tmp_path_dist_ckpt, source_ep, destination_ep
):
    global_experts = 8
    rows = 3
    columns = 4

    def make_gain_state(ep_size, source_values):
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        expert_dp_rank = parallel_state.get_expert_data_parallel_rank()
        local_experts = global_experts // ep_size
        expert_start = ep_rank * local_experts
        param = torch.nn.Parameter(
            torch.ones(local_experts, rows, columns, device='cuda')
        )

        model_factory = make_fused_experts_sharded_factory(
            param,
            '',
            'weight',
            num_local_experts=local_experts,
            local_expert_indices_offset=expert_start,
            num_global_experts=global_experts,
            sharded_offsets=(),
            replica_id=(0, 0, expert_dp_rank),
            singleton_local_shards=False,
        )
        optimizer = MDDecoupling(
            params=[param],
            lr=0.01,
            hypersphere_gains_mode='rowcol',
            pg_collection=None,
        )
        expert_ids = torch.arange(
            expert_start,
            expert_start + local_experts,
            dtype=torch.float32,
            device='cuda',
        )
        gains = {
            'row_gain': expert_ids[:, None] * 10 + torch.arange(rows, device='cuda'),
            'col_gain': expert_ids[:, None] * 10 + torch.arange(columns, device='cuda'),
            'flat_gain': expert_ids,
        }
        if not source_values:
            gains = {name: torch.full_like(value, -1) for name, value in gains.items()}
        return {
            name: optimizer.build_sharded_optimizer_state(
                model_factory,
                value,
                name,
                f'optimizer.state.{name}',
            )
            for name, value in gains.items()
        }

    Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=source_ep)
    try:
        with TempNamedDir(
            tmp_path_dist_ckpt / f'md_gain_ep_{source_ep}_to_{destination_ep}', sync=True
        ) as ckpt_dir:
            save(make_gain_state(source_ep, source_values=True), ckpt_dir)
            Utils.destroy_model_parallel()

            Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=destination_ep)
            loaded = load(make_gain_state(destination_ep, source_values=False), ckpt_dir)

            ep_rank = parallel_state.get_expert_model_parallel_rank()
            local_experts = global_experts // destination_ep
            expert_start = ep_rank * local_experts
            expert_ids = torch.arange(
                expert_start,
                expert_start + local_experts,
                dtype=torch.float32,
                device='cuda',
            )
            torch.testing.assert_close(
                loaded['row_gain'],
                expert_ids[:, None] * 10 + torch.arange(rows, device='cuda'),
            )
            torch.testing.assert_close(
                loaded['col_gain'],
                expert_ids[:, None] * 10 + torch.arange(columns, device='cuda'),
            )
            torch.testing.assert_close(loaded['flat_gain'], expert_ids)
    finally:
        Utils.destroy_model_parallel()


def _md_sharded_optimizer(param):
    optimizer = MDDecoupling(
        params=[
            {
                'params': [param],
                'wd_mult': 1.0,
                'is_expert_parallel': False,
                'is_decoupled_lr': False,
            }
        ],
        lr=0.01,
        hypersphere_gains_mode="rowcol",
        pg_collection=None,
    )
    return FP32Optimizer(
        optimizer,
        OptimizerConfig(optimizer='md_decoupling'),
        _md_init_state_fn,
    )


def _linear_weight_sharded_state(param):
    return {'linear.weight': ShardedTensor.from_rank_offsets('linear.weight', param)}


def test_md_decoupling_torch_dist_round_trips_gain_tensors(tmp_path_dist_ckpt):
    Utils.initialize_model_parallel(1, 1)
    try:
        expected_gains = (
            ('row_gain', 2.0, (3,)),
            ('row_gain_m', 3.0, (3,)),
            ('row_gain_v', 4.0, (3,)),
            ('col_gain', 5.0, (4,)),
            ('col_gain_m', 6.0, (4,)),
            ('col_gain_v', 7.0, (4,)),
        )
        param = torch.nn.Parameter(torch.ones(3, 4))
        megatron_optimizer = _md_sharded_optimizer(param)
        megatron_optimizer.sharded_state_dict(
            _linear_weight_sharded_state(param),
            is_loading=True,
        )
        optimizer = megatron_optimizer.optimizer
        for name, value, _ in expected_gains:
            optimizer.state[param][name].fill_(value)

        with TempNamedDir(tmp_path_dist_ckpt / 'md_gain_state_round_trip', sync=True) as ckpt_dir:
            save(
                megatron_optimizer.sharded_state_dict(_linear_weight_sharded_state(param)),
                ckpt_dir,
            )

            loaded_param = torch.nn.Parameter(torch.ones(3, 4))
            loaded_megatron_optimizer = _md_sharded_optimizer(loaded_param)
            loaded_state_dict = load(
                loaded_megatron_optimizer.sharded_state_dict(
                    _linear_weight_sharded_state(loaded_param),
                    is_loading=True,
                ),
                ckpt_dir,
            )
            loaded_megatron_optimizer.load_state_dict(loaded_state_dict)

        loaded_state = loaded_megatron_optimizer.optimizer.state[loaded_param]
        for name, value, shape in expected_gains:
            torch.testing.assert_close(loaded_state[name], torch.full(shape, value))
    finally:
        Utils.destroy_model_parallel()


@requires_cuda_and_emerging
def test_md_decoupling_qkv_split():
    qkv_size = 3 * 8 * 4
    hidden_size = 64
    qkv_split_shapes = (8, 8, 8)

    torch.manual_seed(42)
    input_tensor = torch.randn(8, hidden_size, dtype=torch.float32, device="cuda")

    model_split = torch.nn.Linear(
        hidden_size, qkv_size, bias=False, dtype=torch.float32, device="cuda"
    )
    model_no_split = torch.nn.Linear(
        hidden_size, qkv_size, bias=False, dtype=torch.float32, device="cuda"
    )
    model_split.weight.data.fill_(1.0)
    model_no_split.weight.data.copy_(model_split.weight.data)
    model_split.weight.is_qkv = True

    optimizer_split = MDDecoupling(
        params=[model_split.weight],
        lr=0.01,
        use_orthogonal_updates=True,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=qkv_split_shapes,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )
    optimizer_no_split = MDDecoupling(
        params=[model_no_split.weight],
        lr=0.01,
        use_orthogonal_updates=True,
        split_qkv=False,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    original_weight = model_split.weight.data.clone()
    _step_sum_loss(model_split, input_tensor)
    optimizer_split.step()
    weight_with_split = model_split.weight.data.clone()

    _step_sum_loss(model_no_split, input_tensor)
    optimizer_no_split.step()
    weight_without_split = model_no_split.weight.data.clone()

    assert not torch.equal(weight_with_split, original_weight)
    assert not torch.equal(weight_without_split, original_weight)
    assert not torch.equal(weight_with_split, weight_without_split)


def _fan_in_optimizer(param, **kwargs):
    kwargs.setdefault("scale_mode", "shape_up")
    kwargs.setdefault("tp_mode", "duplicated")
    kwargs.setdefault("pg_collection", _NoProcessGroups())
    return MDDecoupling(
        params=[param], lr=0.01, hypersphere_radius_mode="fan_in", **kwargs
    )


@pytest.mark.parametrize("shape", [(8, 4), (4, 8), (6, 6)])
@pytest.mark.parametrize("mode", ["row", "flat"])
def test_md_decoupling_fan_in_radius_puts_weight_norm_at_sqrt_out(shape, mode):
    """fan_in normalizes every row to unit L2, i.e. ||W||_F = sqrt(d_out) for any shape, in both
    the row and flat sphere modes."""
    size_out, size_in = shape
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(size_out, size_in))
    optimizer = _fan_in_optimizer(
        param, hypersphere_mode=mode, hypersphere_preserve_init=True
    )

    with torch.no_grad():
        optimizer._normalize(param, param)

    torch.testing.assert_close(
        torch.linalg.matrix_norm(param.detach()),
        torch.tensor(math.sqrt(size_out)),
        rtol=1e-5,
        atol=1e-5,
    )
    if mode == "row":
        torch.testing.assert_close(
            torch.linalg.vector_norm(param.detach(), dim=1),
            torch.ones(size_out),
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize("shape", [(8, 4), (4, 8), (6, 6)])
def test_md_decoupling_fan_in_radius_and_update_scale_agree(shape):
    """The fan-in update scale puts ||U||_F on the same sqrt(d_out) sphere as the weight, and is
    exactly shape_up * _init_radius_scale (the composition the notes derive)."""
    size_out, size_in = shape
    optimizer = _fan_in_optimizer(
        torch.nn.Parameter(torch.zeros(*shape)), hypersphere_mode="row"
    )

    # Weight radii: flat -> sqrt(d_out), row -> 1, col -> sqrt(d_out/d_in) (col is rejected at
    # construction, but the radius rule is the same sqrt(|slice| / d_in) for all three).
    assert optimizer._target_slice_radius(None, size_out, size_in) == pytest.approx(
        math.sqrt(size_out)
    )
    assert optimizer._target_slice_radius(1, size_out, size_in) == pytest.approx(1.0)
    assert optimizer._target_slice_radius(0, size_out, size_in) == pytest.approx(
        math.sqrt(size_out / size_in)
    )

    update_scale = optimizer._fan_in_update_scale(size_out, size_in)
    assert update_scale == pytest.approx(
        _get_muon_scale_factor(size_out, size_in, mode="shape_up")
        * optimizer._init_radius_scale(size_out, size_in)
    )
    # Newton-Schulz returns unit singular values, so ||orth||_F = sqrt(min(d_out, d_in)).
    assert math.sqrt(min(shape)) * update_scale == pytest.approx(math.sqrt(size_out))


@pytest.mark.parametrize("shape", [(8, 4), (4, 8)])
def test_md_decoupling_fan_in_update_norm_matches_sphere_radius(shape, monkeypatch):
    """End-to-end on the update path: a semi-orthogonal Newton-Schulz output comes out of
    _orthogonalize_param with ||U||_F = sqrt(d_out) = the weight's fan-in radius."""
    size_out, size_in = shape
    monkeypatch.setattr(md_module, "newton_schulz_tp", lambda g, **kwargs: g, raising=False)
    param = torch.nn.Parameter(torch.zeros(size_out, size_in))
    optimizer = _fan_in_optimizer(param, hypersphere_mode="row")

    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(max(shape), min(shape)))
    semi_orthogonal = q if size_out >= size_in else q.T

    update = optimizer._orthogonalize_param(param, semi_orthogonal, use_radius_scale=True)

    torch.testing.assert_close(
        torch.linalg.matrix_norm(update),
        torch.tensor(math.sqrt(size_out)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_md_decoupling_fan_in_leaves_router_update_scale_alone(monkeypatch):
    """Routers keep router_scale_mode ("none" -> 1.0): their sphere radius is sqrt(num_experts),
    which the bare Newton-Schulz output already has, so no fan-in rescale is applied."""
    monkeypatch.setattr(md_module, "newton_schulz_tp", lambda g, **kwargs: g, raising=False)
    param = torch.nn.Parameter(torch.zeros(4, 16))
    optimizer = _fan_in_optimizer(
        param, hypersphere_mode="row", hypersphere_router_mode="row"
    )

    assert optimizer._use_radius_scale("row", is_router=False) is True
    assert optimizer._use_radius_scale("flat", is_router=False) is True
    assert optimizer._use_radius_scale(None, is_router=False) is False
    assert optimizer._use_radius_scale("row", is_router=True) is False

    grad = torch.ones(4, 16)
    router_update = optimizer._orthogonalize_param(
        param, grad, use_radius_scale=False, is_router=True
    )
    torch.testing.assert_close(router_update, grad)


@pytest.mark.parametrize("radius_mode", ["shape_native", "init"])
def test_md_decoupling_non_fan_in_radius_modes_rescale_flat_only(radius_mode):
    """'init' (and the default) only move the flat-mode sphere, and keep rescaling routers."""
    optimizer = MDDecoupling(
        params=[torch.nn.Parameter(torch.zeros(8, 4))],
        lr=0.01,
        hypersphere_mode="flat",
        hypersphere_radius_mode=radius_mode,
        hidden_size=8,
        pg_collection=_NoProcessGroups(),
        tp_mode="duplicated",
    )

    assert optimizer._use_radius_scale("flat", is_router=False) is True
    assert optimizer._use_radius_scale("row", is_router=False) is False
    assert optimizer._use_radius_scale("flat", is_router=True) is True
    expected = 1.0 if radius_mode == "shape_native" else math.sqrt(4 / 8)
    assert optimizer._init_radius_scale(8, 4) == pytest.approx(expected)
    assert optimizer._target_slice_radius(1, 8, 4) == pytest.approx(1.0)
    assert optimizer._target_slice_radius(None, 8, 4) == pytest.approx(
        math.sqrt(8) * expected
    )


def test_md_decoupling_fan_in_uses_global_sizes_under_tp(monkeypatch):
    """The fan-in radius is derived from GLOBAL (TP-unsharded) sizes, so a matrix lands on the
    same sphere at any TP degree. Simulated here with tp_size=2 metadata over a single shard."""
    monkeypatch.setattr(md_module, "get_pg_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    local_out, size_in = 4, 8
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(local_out, size_in))
    param.partition_dim = 0  # column-parallel: d_out is sharded, so global d_out = 8
    optimizer = _fan_in_optimizer(
        param, hypersphere_mode="flat", hypersphere_preserve_init=True
    )

    with torch.no_grad():
        optimizer._normalize(param, param)

    # all_reduce is stubbed out, so the shard normalizes to unit Frobenius and is then placed on
    # the GLOBAL radius sqrt(d_out) = sqrt(local_out * tp_size), not the local sqrt(4).
    torch.testing.assert_close(
        torch.linalg.matrix_norm(param.detach()),
        torch.tensor(math.sqrt(local_out * 2)),
        rtol=1e-5,
        atol=1e-5,
    )
    assert optimizer._global_sizes(param, partition_dim=0) == [local_out * 2, size_in]
    assert optimizer._global_sizes(param, partition_dim=1) == [local_out, size_in * 2]
    assert optimizer._global_sizes(param, partition_dim=None) == [local_out, size_in]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"tp_mode": "blockwise"}, "blockwise"),
        ({"scale_mode": "spectral"}, "shape_up"),
        ({"hypersphere_mode": "col"}, "hypersphere_mode"),
        ({"hypersphere_mode": "embed"}, "hypersphere_mode"),
        ({"hypersphere_embedding_mode": "col"}, "hypersphere_embedding_mode"),
        ({"hypersphere_router_mode": "embed"}, "hypersphere_router_mode"),
    ],
)
def test_md_decoupling_fan_in_rejects_inconsistent_settings(kwargs, match):
    kwargs.setdefault("hypersphere_mode", "row")
    with pytest.raises(AssertionError, match=match):
        _fan_in_optimizer(torch.nn.Parameter(torch.zeros(8, 4)), **kwargs)


def test_md_decoupling_unknown_radius_mode_raises():
    with pytest.raises(ValueError, match="hypersphere_radius_mode"):
        MDDecoupling(
            params=[torch.nn.Parameter(torch.zeros(8, 4))],
            lr=0.01,
            hypersphere_mode="row",
            hypersphere_radius_mode="fanin",
            pg_collection=_NoProcessGroups(),
            tp_mode="duplicated",
        )


@requires_cuda_and_emerging
@pytest.mark.parametrize("tp_mode", ["duplicated", "blockwise", "distributed"])
def test_md_decoupling_different_tp_modes_single_rank(tp_mode):
    torch.manual_seed(42)
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device="cuda")
    model.requires_grad_(True)
    model.weight.data.normal_(0, 0.02)

    optimizer = MDDecoupling(
        params=[model.weight],
        lr=0.01,
        weight_decay=0.0,
        use_orthogonal_updates=True,
        momentum_beta=0.95,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode=tp_mode,
    )

    torch.manual_seed(42)
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device="cuda")
    original_weight = model.weight.data.clone()
    _step_sum_loss(model, input_tensor)
    optimizer.step()

    assert not torch.equal(model.weight.data, original_weight)


@requires_cuda_and_emerging
@pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMDDecouplingMultiRankTP:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        world = int(os.getenv("WORLD_SIZE", "1"))
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
        yield
        Utils.destroy_model_parallel()

    def create_tp_model_and_optimizer(self, tp_mode):
        rank = int(os.getenv("RANK", "0"))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        torch.manual_seed(42 + rank)
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device="cuda")
        model.requires_grad_(True)
        model.weight.data.normal_(0, 0.02)
        model.weight.partition_dim = 0

        optimizer = MDDecoupling(
            params=[model.weight],
            lr=0.01,
            weight_decay=0.0,
            use_orthogonal_updates=True,
            momentum_beta=0.95,
            num_ns_steps=5,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
        )

        return model, optimizer

    @pytest.mark.parametrize("tp_mode", ["duplicated", "distributed"])
    def test_md_decoupling_modes_multirank_update(self, tp_mode):
        model, optimizer = self.create_tp_model_and_optimizer(tp_mode)

        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device="cuda")
        original_weight = model.weight.data.clone()
        _step_sum_loss(model, input_tensor)
        optimizer.step()

        assert not torch.equal(model.weight.data, original_weight)

    def test_md_decoupling_blockwise_mode_multirank_update(self):
        model, optimizer = self.create_tp_model_and_optimizer("blockwise")

        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device="cuda")
        original_weight = model.weight.data.clone()
        _step_sum_loss(model, input_tensor)
        optimizer.step()

        assert not torch.equal(model.weight.data, original_weight)

    @pytest.mark.parametrize(
        "hypersphere_mode, partition_dim", [("flat", 0), ("row", 1), ("row", 0), ("flat", 1)]
    )
    def test_md_decoupling_fan_in_radius_is_global_under_tp(
        self, hypersphere_mode, partition_dim
    ):
        """The real TP check: whichever axis is sharded, the fan-in sphere is the GLOBAL one, so
        the reassembled matrix has ||W||_F = sqrt(global d_out) (equivalently unit global rows)."""
        rank = int(os.getenv("RANK", "0"))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_size = torch.distributed.get_world_size(group=pg_collection.tp)
        local_shape = (8, 16)

        torch.manual_seed(42 + rank)
        weight = torch.nn.Parameter(torch.randn(*local_shape, device="cuda"))
        weight.partition_dim = partition_dim
        # Constructing with preserve_init=False projects onto the sphere right away.
        MDDecoupling(
            params=[weight],
            lr=0.01,
            hypersphere_mode=hypersphere_mode,
            hypersphere_radius_mode="fan_in",
            scale_mode="shape_up",
            pg_collection=pg_collection,
            tp_mode="duplicated",
        )

        global_sizes = list(local_shape)
        global_sizes[partition_dim] *= tp_size
        # Every shard holds a disjoint slice of the global matrix, so the global Frobenius norm is
        # the TP sum of the local ones either way.
        squared_frobenius = (weight.detach().double() ** 2).sum()
        torch.distributed.all_reduce(squared_frobenius, group=pg_collection.tp)

        torch.testing.assert_close(
            squared_frobenius,
            torch.tensor(float(global_sizes[0]), dtype=torch.float64, device="cuda"),
            rtol=1e-5,
            atol=1e-4,
        )

        if hypersphere_mode == "row":
            row_squared = (weight.detach().double() ** 2).sum(dim=1)
            if partition_dim == 1:
                torch.distributed.all_reduce(row_squared, group=pg_collection.tp)
            torch.testing.assert_close(
                row_squared, torch.ones_like(row_squared), rtol=1e-5, atol=1e-5
            )


def test_md_decoupling_gqa_qkv_split_mechanics():
    param = torch.nn.Parameter(torch.empty(8, 4))
    param.is_qkv = True
    grad = torch.arange(32, dtype=torch.float32).view(8, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=(4, 2, 2),
    )

    assert [call.shape for call in calls] == [
        torch.Size([4, 4]),
        torch.Size([2, 4]),
        torch.Size([2, 4]),
    ]
    expected = torch.tensor([1] * 4 + [2] * 2 + [3] * 2).view(8, 1)
    assert torch.equal(output, expected.expand_as(output))


def test_md_decoupling_gqa_split_flat_normalization_is_block_local():
    param = torch.nn.Parameter(torch.arange(1, 33, dtype=torch.float32).view(8, 4))
    param.is_qkv = True
    optimizer = _gqa_qkv_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._normalize(param, param, is_qkv=True)

    _assert_qkv_split_flat_norms(optimizer, param, expected_norm=2.0)


def test_md_decoupling_gqa_split_tangential_grad_is_block_local():
    param = torch.nn.Parameter(torch.arange(1, 33, dtype=torch.float32).view(8, 4))
    param.is_qkv = True
    grad = torch.arange(33, 65, dtype=torch.float32).view(8, 4)
    optimizer = _gqa_qkv_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_tangential_grad=True,
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._project_tangent_inplace(param, grad, is_qkv=True)

    _assert_qkv_split_tangent(optimizer, param, grad)


def test_md_decoupling_gqa_split_row_normalization():
    param = torch.nn.Parameter(torch.arange(1, 33, dtype=torch.float32).view(8, 4))
    param.is_qkv = True
    optimizer = _gqa_qkv_optimizer(
        param,
        hypersphere_mode="row",
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._normalize(param, param, is_qkv=True)

    for part in _split_qkv(param, optimizer.qkv_split_shapes):
        row_norms = torch.linalg.vector_norm(part, dim=1)
        torch.testing.assert_close(row_norms, torch.ones_like(row_norms), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("name", "shape", "gated", "expected_dim"),
    [
        ("decoder.mlp.linear_fc1.weight", (8, 3), True, 0),
        ("decoder.mlp.shared_experts.linear_fc1.weight", (8, 3), True, 0),
        ("decoder.mlp.experts.local_experts.0.linear_fc1.weight", (8, 3), True, 0),
        ("decoder.mlp.experts.linear_fc1.weight0", (8, 3), True, 0),
        ("decoder.mlp.experts.weight1_expert_0", (3, 8), True, 1),
        ("decoder.mlp.experts.weight1", (2, 8, 3), True, 1),
        ("decoder.mlp.linear_fc1.weight", (8, 3), False, None),
        ("decoder.mlp.linear_fc2.weight", (3, 4), True, None),
    ],
)
def test_md_decoupling_glu_fc1_layout_detection(name, shape, gated, expected_dim):
    param = torch.nn.Parameter(torch.empty(shape))
    assert _glu_fc1_split_dim(name, param, gated) == expected_dim


@pytest.mark.parametrize(
    ("shape", "split_dim", "expected_part_shapes"),
    [
        ((8, 3), 0, [(4, 3), (4, 3)]),
        ((3, 8), 1, [(3, 4), (3, 4)]),
        ((2, 8, 3), 1, [(4, 3), (4, 3), (4, 3), (4, 3)]),
    ],
)
def test_md_decoupling_glu_fc1_orthogonalizes_each_logical_matrix(
    shape, split_dim, expected_part_shapes
):
    param = torch.nn.Parameter(torch.empty(shape))
    param.glu_split_dim = split_dim
    grad = torch.arange(math.prod(shape), dtype=torch.float32).view(shape)

    output, calls = _record_md_split_output(
        param,
        grad,
    )

    assert [tuple(call.shape) for call in calls] == expected_part_shapes
    assert output.shape == grad.shape


@pytest.mark.parametrize(("shape", "split_dim"), [((8, 3), 0), ((3, 8), 1), ((2, 8, 3), 1)])
def test_md_decoupling_glu_fc1_flat_normalization_is_block_local(shape, split_dim):
    param = torch.nn.Parameter(torch.arange(1, math.prod(shape) + 1, dtype=torch.float32).view(shape))
    param.glu_split_dim = split_dim
    optimizer = _glu_fc1_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._normalize(
            param,
            param,
            is_merged_offload_expert=param.ndim == 3,
        )

    parts, _ = optimizer._split_param_tensor(param, param)
    expected_norms = torch.tensor(
        [max(part.shape) ** 0.5 for part in parts], dtype=param.dtype
    )
    actual_norms = torch.stack([torch.linalg.vector_norm(part) for part in parts])
    torch.testing.assert_close(actual_norms, expected_norms, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode", ["row", "flat"])
@pytest.mark.parametrize(
    "shape, split_dim", [((8, 2), 0), ((6, 12), 0), ((2, 8, 2), 1)]
)
def test_md_decoupling_fan_in_radius_is_per_glu_fc1_block(shape, split_dim, mode, monkeypatch):
    """A fused GLU fc1 is two logical [ffn, hidden] matrices, so fan_in puts each half on its own
    sqrt(ffn) sphere (fused total sqrt(2*ffn) = sqrt(d_out), same as with split_fc1=False), and the
    update follows the same split so ||U|| = ||W|| holds per half."""
    monkeypatch.setattr(md_module, "newton_schulz_tp", lambda g, **kwargs: g, raising=False)
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(*shape))
    param.glu_split_dim = split_dim
    optimizer = _fan_in_optimizer(
        param, split_fc1=True, hypersphere_mode=mode, hypersphere_preserve_init=True
    )
    is_merged = param.ndim == 3

    with torch.no_grad():
        optimizer._normalize(param, param, is_merged_offload_expert=is_merged)

    # Each half is [ffn, hidden] -> radius sqrt(ffn) regardless of row vs flat.
    parts, _ = optimizer._split_param_tensor(param, param)
    block_out = parts[0].size(0)
    _assert_split_flat_norms(optimizer, param, param, expected_norm=math.sqrt(block_out))
    fused_out = shape[split_dim] if not is_merged else shape[0] * shape[1]
    torch.testing.assert_close(
        torch.linalg.vector_norm(param.detach()),
        torch.tensor(math.sqrt(fused_out)),
        rtol=1e-5,
        atol=1e-5,
    )

    # Update side: a semi-orthogonal grad per half comes back at that half's radius.
    grad = torch.zeros_like(param)
    grad_parts, merge = optimizer._split_param_tensor(param, grad)
    q, _ = torch.linalg.qr(torch.randn(max(grad_parts[0].shape), min(grad_parts[0].shape)))
    semi_orthogonal = q if grad_parts[0].size(0) >= grad_parts[0].size(1) else q.T
    for part in grad_parts:
        part.copy_(semi_orthogonal)
    grad.copy_(merge(grad_parts))

    update = optimizer._orthogonalize_param(
        param, grad, use_radius_scale=True, is_merged_offload_expert=is_merged
    )
    _assert_split_flat_norms(
        optimizer, param, update, expected_norm=math.sqrt(block_out)
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(update),
        torch.tensor(math.sqrt(fused_out)),
        rtol=1e-5,
        atol=1e-5,
    )


@requires_cuda_and_emerging
@pytest.mark.parametrize("num_ns_steps", [5, 15, 25])
def test_md_decoupling_num_ns_steps(num_ns_steps):
    torch.manual_seed(42)
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device="cuda")
    model.weight.data.normal_(0, 0.02)

    optimizer = MDDecoupling(
        params=[model.weight],
        lr=0.01,
        use_orthogonal_updates=True,
        num_ns_steps=num_ns_steps,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device="cuda")
    original_weight = model.weight.data.clone()
    _step_sum_loss(model, input_tensor)
    optimizer.step()

    assert not torch.equal(model.weight.data, original_weight)
    assert optimizer.num_ns_steps == num_ns_steps


@requires_cuda_and_emerging
@pytest.mark.parametrize("use_nesterov", [True, False])
def test_md_decoupling_nesterov(use_nesterov):
    torch.manual_seed(42)
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device="cuda")
    model.weight.data.normal_(0, 0.02)

    optimizer = MDDecoupling(
        params=[model.weight],
        lr=0.01,
        use_orthogonal_updates=True,
        use_nesterov=use_nesterov,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device="cuda")
    original_weight = model.weight.data.clone()
    _step_sum_loss(model, input_tensor)
    optimizer.step()

    assert not torch.equal(model.weight.data, original_weight)
    assert optimizer.use_nesterov is use_nesterov


def test_md_decoupling_mla_split_flat_normalization_is_block_local():
    param = torch.nn.Parameter(torch.arange(1, 33, dtype=torch.float32).view(4, 8))
    param.is_kv_up_proj = True
    optimizer = _mla_kv_up_proj_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._normalize(param, param, is_qkv=False)

    _assert_split_flat_norms(optimizer, param, param, expected_norm=8**0.5)
    torch.testing.assert_close(torch.linalg.vector_norm(param), torch.tensor(32**0.5))

@requires_cuda
def test_md_decoupling_mla_split_gains_step_preserves_bare_split_norms():
    param = torch.nn.Parameter(
        torch.arange(1, 33, dtype=torch.float32, device="cuda").view(4, 8)
    )
    param.is_kv_up_proj = True
    optimizer = _mla_kv_up_proj_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_gains_mode="row",
        gains_lr=0.05,
        use_orthogonal_updates=False,
    )

    grad_scale = torch.linspace(0.1, 3.2, param.numel(), device=param.device).view_as(param)
    loss = (param * grad_scale).sum()
    loss.backward()
    optimizer.step()

    state = optimizer.state[param]
    row_gain = state["row_gain"]
    assert row_gain.shape == (param.size(0),)
    assert torch.isfinite(param).all()
    assert torch.isfinite(row_gain).all()
    assert not torch.allclose(row_gain, torch.ones_like(row_gain))

    bare_param = param.detach() / optimizer._phi(row_gain)[:, None]
    _assert_split_flat_norms(optimizer, param, bare_param, expected_norm=8**0.5)

def test_md_decoupling_mla_split_tangential_grad_is_block_local():
    param = torch.nn.Parameter(torch.arange(1, 33, dtype=torch.float32).view(4, 8))
    param.is_kv_up_proj = True
    grad = torch.linspace(-1.5, 2.5, param.numel()).view_as(param)
    optimizer = _mla_kv_up_proj_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_tangential_grad=True,
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._project_tangent_inplace(param, grad, is_qkv=False)

    _assert_split_tangent(optimizer, param, grad, is_qkv=False)

def test_md_decoupling_mla_split_row_normalization():
    param = torch.nn.Parameter(torch.arange(1, 33, dtype=torch.float32).view(4, 8))
    param.is_kv_up_proj = True
    optimizer = _mla_kv_up_proj_optimizer(
        param,
        hypersphere_mode="row",
        hypersphere_preserve_init=True,
    )

    with torch.no_grad():
        optimizer._normalize(param, param, is_qkv=False)

    parts, _ = optimizer._split_param_tensor(param, param, is_qkv=False)
    for part in parts:
        torch.testing.assert_close(
            torch.linalg.vector_norm(part, dim=1),
            torch.ones(part.size(0), dtype=part.dtype),
            rtol=1e-5,
            atol=1e-5,
        )

@requires_cuda
@pytest.mark.parametrize(
    ("gain_mode", "expected_state_keys"),
    [
        ("flat", ("flat_gain",)),
        ("rowcol", ("row_gain", "col_gain")),
    ],
)
def test_md_decoupling_mla_split_gain_modes_preserve_bare_split_norms(
    gain_mode, expected_state_keys
):
    param = torch.nn.Parameter(
        torch.arange(1, 33, dtype=torch.float32, device="cuda").view(4, 8)
    )
    param.is_kv_up_proj = True
    optimizer = _mla_kv_up_proj_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_gains_mode=gain_mode,
        gains_lr=0.05,
        gain_parametrization="softplus",
        use_orthogonal_updates=False,
    )

    grad_scale = torch.linspace(0.1, 3.2, param.numel(), device=param.device).view_as(param)
    loss = (param * grad_scale).sum()
    loss.backward()
    optimizer.step()

    state = optimizer.state[param]
    for key in expected_state_keys:
        assert key in state
        assert torch.isfinite(state[key]).all()
        assert not torch.allclose(optimizer._phi(state[key]), torch.ones_like(state[key]))
    _assert_split_flat_norms(
        optimizer,
        param,
        _bare_param_from_gains(optimizer, param),
        expected_norm=8**0.5,
    )

@requires_cuda
@pytest.mark.parametrize(
    ("gain_mode", "expected_state_keys"),
    [
        ("flat", ("flat_gain",)),
        ("rowcol", ("row_gain", "col_gain")),
    ],
)
def test_md_decoupling_gqa_split_gain_modes_preserve_bare_split_norms(
    gain_mode, expected_state_keys
):
    param = torch.nn.Parameter(
        torch.arange(1, 129, dtype=torch.float32, device="cuda").view(16, 8)
    )
    param.is_qkv = True
    optimizer = _gqa_qkv_optimizer(
        param,
        hypersphere_mode="flat",
        hypersphere_gains_mode=gain_mode,
        gains_lr=0.05,
        gain_parametrization="softplus",
        use_orthogonal_updates=False,
    )

    grad_scale = torch.linspace(0.1, 12.8, param.numel(), device=param.device).view_as(param)
    loss = (param * grad_scale).sum()
    loss.backward()
    optimizer.step()

    state = optimizer.state[param]
    for key in expected_state_keys:
        assert key in state
        assert torch.isfinite(state[key]).all()
        assert not torch.allclose(optimizer._phi(state[key]), torch.ones_like(state[key]))
    _assert_split_flat_norms(
        optimizer,
        param,
        _bare_param_from_gains(optimizer, param),
        expected_norm=8**0.5,
        is_qkv=True,
    )

def test_md_decoupling_mla_kv_up_proj_split():
    param = torch.nn.Parameter(torch.empty(10, 4))
    param.is_kv_up_proj = True
    grad = torch.arange(40, dtype=torch.float32).view(10, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_kv_up_proj_fn=lambda p: getattr(p, "is_kv_up_proj", False),
        kv_up_proj_split_shapes=(3, 2),
    )

    assert [call.shape for call in calls] == [torch.Size([6, 4]), torch.Size([4, 4])]
    expected = torch.tensor([1, 1, 1, 2, 2, 1, 1, 1, 2, 2]).view(10, 1)
    assert torch.equal(output, expected.expand_as(output))

def test_md_decoupling_mla_kv_up_proj_split_uses_local_dim0_tp_shapes():
    param = torch.nn.Parameter(torch.empty(10, 4))
    param.is_kv_up_proj = True
    param.partition_dim = 0
    grad = torch.arange(40, dtype=torch.float32).view(10, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_kv_up_proj_fn=lambda p: getattr(p, "is_kv_up_proj", False),
        kv_up_proj_split_shapes=(12, 8),
    )

    assert [call.shape for call in calls] == [torch.Size([6, 4]), torch.Size([4, 4])]
    assert torch.equal(output[:6], torch.ones_like(output[:6]))
    assert torch.equal(output[6:], torch.full_like(output[6:], 2.0))

@pytest.mark.parametrize("split_qkv", [False, True])
def test_md_decoupling_mla_kv_up_proj_split_per_head(split_qkv):
    param = torch.nn.Parameter(torch.empty(10, 4))
    param.is_kv_up_proj = True
    grad = torch.arange(40, dtype=torch.float32).view(10, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_kv_up_proj_fn=lambda p: getattr(p, "is_kv_up_proj", False),
        kv_up_proj_split_shapes=(3, 2),
        split_mla_per_head=True,
        split_qkv=split_qkv,
    )

    assert [call.shape for call in calls] == [
        torch.Size([3, 4]),
        torch.Size([2, 4]),
        torch.Size([3, 4]),
        torch.Size([2, 4]),
    ]
    expected = torch.tensor([1] * 3 + [2] * 2 + [3] * 3 + [4] * 2).view(10, 1)
    assert torch.equal(output, expected.expand_as(output))

def test_md_decoupling_mla_kv_up_proj_per_head_ignores_head_partition_dim(monkeypatch):
    param = torch.nn.Parameter(torch.arange(1, 41, dtype=torch.float32).view(10, 4))
    param.is_kv_up_proj = True
    param.partition_dim = 0
    grad = torch.arange(40, dtype=torch.float32).view(10, 4)
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        split_qkv=True,
        is_kv_up_proj_fn=lambda p: getattr(p, "is_kv_up_proj", False),
        kv_up_proj_split_shapes=(3, 2),
        split_mla_per_head=True,
        hypersphere_mode="flat",
        hypersphere_preserve_init=True,
        pg_collection=_NoProcessGroups(),
        tp_mode="distributed",
    )
    partition_dims = []

    def record_partition_dim(split_grad, tp_group, partition_dim, use_radius_scale=False, is_router=False):
        del tp_group, use_radius_scale, is_router
        partition_dims.append(partition_dim)
        return torch.full_like(split_grad, float(len(partition_dims)))

    optimizer._orthogonalize_tensor = record_partition_dim
    output = optimizer._orthogonalize_param(param, grad, is_qkv=False, use_radius_scale=True)

    assert partition_dims == [None, None, None, None]
    expected = torch.tensor([1] * 3 + [2] * 2 + [3] * 3 + [4] * 2).view(10, 1)
    assert torch.equal(output, expected.expand_as(output))

    def fail_all_reduce(*args, **kwargs):
        del args, kwargs
        raise AssertionError("KV-up per-head splits should not all-reduce across TP ranks")

    monkeypatch.setattr(torch.distributed, "all_reduce", fail_all_reduce)

    with torch.no_grad():
        optimizer._normalize(param, param, is_qkv=False)

    _assert_split_flat_norms(optimizer, param, param, expected_norm=2.0)

def test_md_decoupling_mla_q_up_proj_split_per_head():
    param = torch.nn.Parameter(torch.empty(12, 4))
    param.is_q_up_proj = True
    grad = torch.arange(48, dtype=torch.float32).view(12, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_q_up_proj_fn=lambda p: getattr(p, "is_q_up_proj", False),
        q_up_proj_head_dim=4,
        split_mla_per_head=True,
    )

    assert [call.shape for call in calls] == [
        torch.Size([4, 4]),
        torch.Size([4, 4]),
        torch.Size([4, 4]),
    ]
    expected = torch.tensor([1] * 4 + [2] * 4 + [3] * 4).view(12, 1)
    assert torch.equal(output, expected.expand_as(output))

def test_md_decoupling_mla_q_up_proj_per_head_ignores_head_partition_dim(monkeypatch):
    param = torch.nn.Parameter(torch.arange(1, 49, dtype=torch.float32).view(12, 4))
    param.is_q_up_proj = True
    param.partition_dim = 0
    grad = torch.arange(48, dtype=torch.float32).view(12, 4)
    optimizer = MDDecoupling(
        params=[param],
        lr=0.01,
        split_qkv=True,
        is_q_up_proj_fn=lambda p: getattr(p, "is_q_up_proj", False),
        q_up_proj_head_dim=4,
        split_mla_per_head=True,
        hypersphere_mode="flat",
        hypersphere_preserve_init=True,
        pg_collection=_NoProcessGroups(),
        tp_mode="distributed",
    )
    partition_dims = []

    def record_partition_dim(split_grad, tp_group, partition_dim, use_radius_scale=False, is_router=False):
        del tp_group, use_radius_scale, is_router
        partition_dims.append(partition_dim)
        return torch.full_like(split_grad, float(len(partition_dims)))

    optimizer._orthogonalize_tensor = record_partition_dim
    output = optimizer._orthogonalize_param(param, grad, is_qkv=False, use_radius_scale=True)

    assert partition_dims == [None, None, None]
    expected = torch.tensor([1] * 4 + [2] * 4 + [3] * 4).view(12, 1)
    assert torch.equal(output, expected.expand_as(output))

    def fail_all_reduce(*args, **kwargs):
        del args, kwargs
        raise AssertionError("q-up per-head splits should not all-reduce across TP ranks")

    monkeypatch.setattr(torch.distributed, "all_reduce", fail_all_reduce)

    with torch.no_grad():
        optimizer._normalize(param, param, is_qkv=False)

    _assert_split_flat_norms(optimizer, param, param, expected_norm=2.0)

def test_md_decoupling_mla_qkv_down_proj_split_mechanics():
    param = torch.nn.Parameter(torch.empty(5, 4))
    param.is_qkv_down_proj = True
    grad = torch.arange(20, dtype=torch.float32).view(5, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_qkv_down_proj_fn=lambda p: getattr(p, "is_qkv_down_proj", False),
        qkv_down_proj_split_shapes=(2, 3),
    )

    assert [call.shape for call in calls] == [torch.Size([2, 4]), torch.Size([3, 4])]
    assert torch.equal(output[:2], torch.ones_like(output[:2]))
    assert torch.equal(output[2:], torch.full_like(output[2:], 2.0))

def test_md_decoupling_mla_qkv_down_proj_split_uses_local_dim0_tp_shapes():
    param = torch.nn.Parameter(torch.empty(6, 4))
    param.is_qkv_down_proj = True
    param.partition_dim = 0
    grad = torch.arange(24, dtype=torch.float32).view(6, 4)

    output, calls = _record_md_split_output(
        param,
        grad,
        is_qkv_down_proj_fn=lambda p: getattr(p, "is_qkv_down_proj", False),
        qkv_down_proj_split_shapes=(4, 8),
    )

    assert [call.shape for call in calls] == [torch.Size([2, 4]), torch.Size([4, 4])]
    assert torch.equal(output[:2], torch.ones_like(output[:2]))
    assert torch.equal(output[2:], torch.full_like(output[2:], 2.0))

def test_md_decoupling_mla_param_tags_copy_to_main_param():
    param = torch.empty(2, 2)
    tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
    param.is_kv_up_proj = True
    param.is_q_up_proj = True
    param.is_qkv_down_proj = True
    param.is_kda_in_proj = True
    param.kda_split_shapes = (1, 1, 1, 1, 1, 1)
    main_param = torch.empty_like(param)

    tensor_parallel.copy_tensor_model_parallel_attributes(main_param, param)

    assert main_param.is_kv_up_proj
    assert main_param.is_q_up_proj
    assert main_param.is_qkv_down_proj
    assert main_param.is_kda_in_proj
    assert main_param.kda_split_shapes == (1, 1, 1, 1, 1, 1)

@pytest.mark.parametrize(
    ("split_mla_per_head", "expected_q_up_proj_head_dim"),
    [(False, None), (True, 8)],
)
def test_md_decoupling_builder_tags_mla_and_gqa_parameters(
    monkeypatch, split_mla_per_head, expected_q_up_proj_head_dim
):
    class _FakeModelChunk:
        def __init__(self):
            self.config = SimpleNamespace(
                num_attention_heads=8,
                num_query_groups=2,
                kv_channels=4,
                multi_latent_attention=True,
                qk_head_dim=6,
                v_head_dim=5,
                qk_pos_emb_head_dim=2,
                q_lora_rank=3,
                kv_lora_rank=7,
                num_layers=4,
                hidden_size=5,
            )
            self.qkv = torch.nn.Parameter(torch.ones(48, 5))
            self.kv_up = torch.nn.Parameter(torch.ones(88, 5))
            self.q_up = torch.nn.Parameter(torch.ones(64, 5))
            self.qkv_down = torch.nn.Parameter(torch.ones(12, 5))
            self.named = [
                ("decoder.layers.0.self_attention.linear_qkv.weight", self.qkv),
                ("decoder.layers.0.self_attention.linear_kv_up_proj.weight", self.kv_up),
                ("decoder.layers.0.self_attention.linear_q_up_proj.weight", self.q_up),
                (
                    "decoder.layers.0.self_attention.linear_qkv_down_proj.weight",
                    self.qkv_down,
                ),
            ]

        def named_parameters(self):
            return iter(self.named)

    class _FakeOptimizerWrapper:
        def __init__(self, optimizer, config, init_state_fn=None):
            del init_state_fn
            self.optimizer = optimizer
            self.config = config
            self.param_groups = optimizer.param_groups
            self.state = optimizer.state
            self.is_stub_optimizer = False

        def get_parameters(self):
            return [p for group in self.param_groups for p in group["params"]]

    def fake_get_param_groups(model_chunks, config, config_overrides):
        del config, config_overrides
        params = [
            p
            for model_chunk in model_chunks
            for _, p in model_chunk.named_parameters()
            if p.requires_grad
        ]
        return [{"params": params, "is_expert_parallel": False}]

    def fake_get_megatron_optimizer(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(chained_optimizers=[])

    monkeypatch.setattr(md_module, "_get_param_groups", fake_get_param_groups)
    monkeypatch.setattr(md_module, "FP32Optimizer", _FakeOptimizerWrapper)
    monkeypatch.setattr(md_module, "get_megatron_optimizer", fake_get_megatron_optimizer)

    model_chunk = _FakeModelChunk()
    config = OptimizerConfig(optimizer="md_decoupling", lr=0.01, min_lr=0.0)
    config.use_orthogonal_updates = False
    config.hypersphere_mode = "flat"
    config.hypersphere_embedding_mode = None
    config.hypersphere_router_mode = None
    config.hypersphere_gains_mode = None
    config.muon_split_qkv = True
    config.muon_split_mla_per_head = split_mla_per_head
    config.use_distributed_optimizer = False
    config.fp16 = False
    config.bf16 = False

    chained = md_module.get_megatron_mddecoupling_optimizer(
        config,
        [model_chunk],
        config_overrides={},
        pg_collection=_NoProcessGroups(),
    )

    optimizer = chained.chained_optimizers[0].optimizer
    assert optimizer.qkv_split_shapes == [16, 4, 4]
    assert optimizer.q_up_proj_head_dim == expected_q_up_proj_head_dim
    assert optimizer.qkv_down_proj_split_shapes == (3, 9)
    assert model_chunk.qkv.is_qkv
    assert model_chunk.kv_up.is_kv_up_proj
    assert model_chunk.q_up.is_q_up_proj
    assert model_chunk.qkv_down.is_qkv_down_proj

    kv_grad = torch.arange(88 * 5, dtype=torch.float32).view(88, 5)
    kv_parts, merge_kv_parts = optimizer._split_param_tensor(model_chunk.kv_up, kv_grad)
    # MultiLatentAttention.forward views linear_kv_up_proj output as
    # [tokens, num_heads, qk_head_dim + v_head_dim] before splitting K and V.
    kv_per_head = kv_grad.view(8, 11, 5)
    if split_mla_per_head:
        expected_parts = [
            part
            for head in kv_per_head.unbind(0)
            for part in torch.split(head, (6, 5), dim=0)
        ]
    else:
        expected_parts = [
            kv_per_head[:, :6].reshape(48, 5),
            kv_per_head[:, 6:].reshape(40, 5),
        ]
    assert len(kv_parts) == len(expected_parts)
    for actual, expected in zip(kv_parts, expected_parts):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(merge_kv_parts(kv_parts), kv_grad)
    assert optimizer.kv_up_proj_split_shapes == (6, 5)

def test_md_decoupling_layerwise_preserves_mla_and_gqa_parameter_tags(monkeypatch):
    class _FakeOptimizer:
        def __init__(self, params):
            self.config = SimpleNamespace()
            self.param_groups = [{"params": params, "is_expert_parallel": False}]
            self.state = {}
            self.is_stub_optimizer = False

        def get_parameters(self):
            return [p for group in self.param_groups for p in group["params"]]

    monkeypatch.setattr(layer_wise_module, "get_pg_size", lambda group: 2)
    monkeypatch.setattr(layer_wise_module, "get_pg_rank", lambda group: 0)

    qkv = torch.nn.Parameter(torch.ones(16, 4))
    qkv.is_qkv = True
    kv_up = torch.nn.Parameter(torch.ones(8, 4))
    kv_up.is_kv_up_proj = True
    q_up = torch.nn.Parameter(torch.ones(12, 4))
    q_up.is_q_up_proj = True
    qkv_down = torch.nn.Parameter(torch.ones(5, 4))
    qkv_down.is_qkv_down_proj = True

    optimizer = _FakeOptimizer([qkv, kv_up, q_up, qkv_down])
    config = SimpleNamespace(bf16=False)
    pg_collection = SimpleNamespace(dp_cp=object(), expt_dp=object())

    layerwise = LayerWiseDistributedOptimizer([optimizer], config, pg_collection)

    sharded_params = [p for shard in layerwise.dp_cp_params_list for p in shard]
    assert any(p is qkv for p in sharded_params) and qkv.is_qkv
    assert any(p is kv_up for p in sharded_params) and kv_up.is_kv_up_proj
    assert any(p is q_up for p in sharded_params) and q_up.is_q_up_proj
    assert any(p is qkv_down for p in sharded_params) and qkv_down.is_qkv_down_proj
