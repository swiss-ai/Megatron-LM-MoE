# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state, tensor_parallel
import megatron.core.optimizer.muon as muon_module
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import HAVE_EMERGING_OPTIMIZERS, HAVE_EO_V02, OptimizerConfig
from megatron.core.optimizer.muon import (
    TensorParallelMuon,
    get_megatron_muon_optimizer,
    get_supported_coefficient_types,
    validate_coefficient_type,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# Skip all tests in this file for LTS versions or when emerging_optimizers is missing
pytestmark = [
    pytest.mark.skipif(
        Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
        reason="Skip muon optimizer for LTS test",
    ),
    pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package is not installed"
    ),
]

requires_eo_v02 = pytest.mark.skipif(
    not HAVE_EO_V02, reason="emerging_optimizers >= 0.2 is required"
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(80, 48)
        self.fc2 = nn.Linear(48, 32)
        self.fc3 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc5 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def test_muon_optimizer_smoke():
    """Smoke test for TensorParallelMuon optimizer."""
    # Create a simple linear model for testing
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    # Create TensorParallelMuon optimizer
    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum_beta=0.95,
        use_nesterov=True,
        weight_decay=0.01,
        use_decoupled_weight_decay=True,
        split_qkv=False,
        fp32_matmul_prec="medium",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        pg_collection=None,
        mode="duplicated",
    )

    # Test basic properties
    assert optimizer is not None, "Optimizer should not be None"
    assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
    assert len(optimizer.param_groups) > 0, "Optimizer should have at least one parameter group"

    # Test forward and backward pass
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Store original weight
    original_weight = model.weight.data.clone()

    # Test optimizer step
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    # Test zero_grad
    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    # Test state_dict and load_state_dict
    state_dict = optimizer.state_dict()
    assert 'state' in state_dict, "State dict should contain state"
    assert 'param_groups' in state_dict, "State dict should contain param_groups"

    # Load state dict should not raise error
    optimizer.load_state_dict(state_dict)


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonOptimizerMultiRank:
    """Test class for Muon optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP.

        Args:
            model: Model to wrap

        Returns:
            DDP-wrapped model
        """
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def test_get_megatron_muon_optimizer_smoke(self):
        """Smoke test for get_megatron_muon_optimizer function."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # Ensure all parameters require gradients
        for param in model.parameters():
            assert param.requires_grad, "All parameters should require gradients"

        # Create optimizer config for Muon
        optimizer_config = OptimizerConfig(
            optimizer='muon',  # This will be changed internally to 'adam' for non-linear params
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,  # Muon doesn't support distributed optimizer
            muon_momentum=0.95,
            muon_use_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
        )

        # Test creating the optimizer
        optimizer = get_megatron_muon_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=True,
            layer_wise_distributed_optimizer=False,
        )

        # Test basic properties
        assert optimizer is not None, "Optimizer should not be None"
        assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
        assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"
        assert len(optimizer.chained_optimizers) >= 1, "Should have at least one chained optimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Test optimizer step
        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated after optimizer step"

        # Test zero_grad
        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(
                param.grad == 0
            ), f"Gradients should be zeroed for all parameters"

        # Test state_dict and load_state_dict
        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, list), "State dict should be a list"

        # Load state dict should not raise error
        optimizer.load_state_dict(state_dict)

    def test_get_megatron_muon_optimizer_validation(self):
        """Test validation logic for get_megatron_muon_optimizer."""
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # Test 1: Distributed optimizer should raise exception
        optimizer_config_dist = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=True,  # This should cause an exception
        )

        with pytest.raises(Exception, match='muon with dist optimizer is not supported'):
            get_megatron_muon_optimizer(config=optimizer_config_dist, model_chunks=[model])

        # Test 2: FP16 should raise exception
        optimizer_config_fp16 = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            fp16=True,  # This should cause an exception
            use_distributed_optimizer=False,
        )

        with pytest.raises(Exception, match='muon with fp16 is not supported'):
            get_megatron_muon_optimizer(config=optimizer_config_fp16, model_chunks=[model])

        # Test 3: Invalid num_ns_steps should raise exception
        optimizer_config_invalid_ns = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_num_ns_steps=0,  # This should cause an exception
        )

        with pytest.raises(ValueError, match='num_ns_steps must be at least 1'):
            get_megatron_muon_optimizer(config=optimizer_config_invalid_ns, model_chunks=[model])

    def test_get_megatron_muon_optimizer_layer_wise(self):
        """Test get_megatron_muon_optimizer with layer-wise distributed optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_momentum=0.95,
            muon_use_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
        )

        # Test with layer_wise_distributed_optimizer=True
        optimizer = get_megatron_muon_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=True,
            layer_wise_distributed_optimizer=True,
        )

        # Verify it's a LayerWiseDistributedOptimizer
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        assert isinstance(
            optimizer, LayerWiseDistributedOptimizer
        ), "Should return LayerWiseDistributedOptimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Test optimizer step
        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"
        assert grad_norm is not None or grad_norm is None, "Grad norm should be returned"


@pytest.mark.parametrize("mode", ["duplicated", "blockwise", "distributed"])
def test_muon_optimizer_different_modes_single_rank(mode):
    """Test TensorParallelMuon optimizer with different modes on single rank.

    When TP size is 1, all modes should produce the same result.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.normal_(0, 0.02)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum_beta=0.95,
        weight_decay=0.0,  # Disable weight decay for deterministic comparison
        num_ns_steps=5,
        pg_collection=None,
        mode=mode,
    )

    # Use fixed input for deterministic results
    torch.manual_seed(42)
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with mode={mode}"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonOptimizerMultiRankTP:
    """Test class for Muon optimizer with multi-rank and tensor parallel setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test with tensor parallel."""
        world = int(os.getenv('WORLD_SIZE', '1'))
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
        yield
        Utils.destroy_model_parallel()

    def create_tp_model_and_optimizer(self, mode):
        """Create model with TP and optimizer.

        Args:
            mode: Muon optimizer mode

        Returns:
            tuple: (model, optimizer, pg_collection)
        """
        rank = int(os.getenv('RANK', '0'))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        # Create model with partition_dim for TP
        torch.manual_seed(42 + rank)
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
        model.requires_grad_(True)
        model.weight.data.normal_(0, 0.02)
        model.weight.partition_dim = 0  # Set partition dimension for TP

        optimizer = TensorParallelMuon(
            params=[model.weight],
            lr=0.01,
            momentum_beta=0.95,
            weight_decay=0.0,
            num_ns_steps=5,
            pg_collection=pg_collection,
            mode=mode,
        )

        return model, optimizer

    @pytest.mark.parametrize("mode", ["duplicated", "distributed"])
    def test_muon_optimizer_modes_multirank_same_result(self, mode):
        """Test that duplicated and distributed modes produce same results with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer(mode)

        # Use fixed input for deterministic results
        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        # Verify weight was updated
        assert not torch.equal(
            model.weight.data, original_weight
        ), f"Weight should be updated with mode={mode}"

    def test_muon_optimizer_blockwise_mode_different_result(self):
        """Test that blockwise mode produces different results than duplicated/distributed with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer("blockwise")

        # Use fixed input for deterministic results
        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        # Verify weight was updated
        assert not torch.equal(
            model.weight.data, original_weight
        ), "Weight should be updated with mode=blockwise"


# All non-custom coefficient types supported by emerging_optimizers.
_TESTABLE_COEFFICIENT_TYPES = (
    [t for t in get_supported_coefficient_types() if t != "custom"] if HAVE_EO_V02 else []
)

# A reasonable default NS step count for testing; get_coefficient_iterator
# cycles/repeats coefficients so any step count works with any type.
_DEFAULT_NS_STEPS = 5


@pytest.mark.parametrize("coefficient_type", _TESTABLE_COEFFICIENT_TYPES)
def test_muon_optimizer_coefficient_types(coefficient_type):
    """Test TensorParallelMuon optimizer with different coefficient types."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        coefficient_type=coefficient_type,
        num_ns_steps=_DEFAULT_NS_STEPS,
        pg_collection=None,
        mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with coefficient_type={coefficient_type}"


@pytest.mark.parametrize("scale_mode", ["spectral", "unit_rms_norm", "shape_scaling"])
def test_muon_optimizer_scale_modes(scale_mode):
    """Test TensorParallelMuon optimizer with different scale modes."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        scale_mode=scale_mode,
        num_ns_steps=5,
        pg_collection=None,
        mode="duplicated",
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with scale_mode={scale_mode}"


@pytest.mark.parametrize("use_nesterov", [True, False])
def test_muon_optimizer_nesterov(use_nesterov):
    """Test TensorParallelMuon optimizer with and without Nesterov momentum."""
    model = torch.nn.Linear(50, 25, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum_beta=0.9,
        use_nesterov=use_nesterov,
        num_ns_steps=5,
        pg_collection=None,
        mode="duplicated",
    )

    input_tensor = torch.randn(16, 50, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with use_nesterov={use_nesterov}"


def test_muon_optimizer_multiple_steps():
    """Test TensorParallelMuon optimizer across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum_beta=0.95,
        weight_decay=0.01,
        num_ns_steps=5,
        pg_collection=None,
        mode="duplicated",
    )

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    # Verify weights changed at each step
    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


def test_muon_optimizer_qkv_split():
    """Test TensorParallelMuon optimizer with QKV splitting."""
    # Create a model with QKV-like parameter
    qkv_size = 3 * 64 * 16  # Combined Q, K, V dimensions, 16 heads x 64 per head
    hidden_size = 1024
    model = torch.nn.Linear(hidden_size, qkv_size, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    # Mark parameter as QKV
    model.weight.is_qkv = True

    # QKV split shapes: [Q_size, K_size, V_size]
    qkv_split_shapes = (64, 64, 64)

    # Test with split_qkv=True
    optimizer_split = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        num_ns_steps=5,
        pg_collection=None,
        mode="duplicated",
    )

    input_tensor = torch.randn(16, hidden_size, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer_split.step()
    weight_with_split = model.weight.data.clone()

    assert not torch.equal(
        weight_with_split, original_weight
    ), "QKV weight should be updated with split_qkv=True"

    # Reset model and test with split_qkv=False
    model.weight.data.fill_(1.0)
    optimizer_no_split = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=False,
        num_ns_steps=5,
        pg_collection=None,
        mode="duplicated",
    )

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    optimizer_no_split.step()
    weight_without_split = model.weight.data.clone()

    assert not torch.equal(
        weight_without_split, original_weight
    ), "QKV weight should be updated with split_qkv=False"

    # Ensure the two results are different
    assert not torch.equal(
        weight_with_split, weight_without_split
    ), "Weights should be different between split_qkv=True and split_qkv=False"


def _record_muon_split_output(param, grad, **muon_kwargs):
    split_qkv = muon_kwargs.pop("split_qkv", True)
    optimizer = TensorParallelMuon(
        params=[param],
        lr=0.01,
        split_qkv=split_qkv,
        num_ns_steps=5,
        pg_collection=None,
        mode="duplicated",
        **muon_kwargs,
    )
    calls = []

    def record_split(split_grad, tp_group, partition_dim):
        del tp_group, partition_dim
        calls.append(split_grad.detach().clone())
        return torch.full_like(split_grad, float(len(calls)))

    optimizer.scaled_orthogonalize_fn = record_split
    return optimizer.orthogonalize(param, grad), calls


def test_muon_glu_fc1_orthogonalizes_gate_and_up_separately():
    param = torch.nn.Parameter(torch.empty(8, 4, device='cuda'))
    param.glu_split_dim = 0
    grad = torch.arange(32, dtype=torch.float32, device='cuda').view_as(param)

    output, calls = _record_muon_split_output(param, grad, split_fc1=True)

    assert [tuple(call.shape) for call in calls] == [(4, 4), (4, 4)]
    assert torch.equal(output[:4], torch.ones_like(output[:4]))
    assert torch.equal(output[4:], torch.full_like(output[4:], 2.0))


def test_muon_optimizer_kda_in_proj_split_uses_local_dim0_tp_shapes():
    local_shapes = (2, 2, 3, 3, 1, 4)
    param = torch.nn.Parameter(torch.empty(sum(local_shapes), 5, device='cuda'))
    param.is_kda_in_proj = True
    param.kda_split_shapes = tuple(2 * rows for rows in local_shapes)
    param.partition_dim = 0
    grad = torch.arange(param.numel(), dtype=torch.float32, device='cuda').view_as(param)

    output, calls = _record_muon_split_output(
        param,
        grad,
        is_kda_in_proj_fn=lambda p: getattr(p, 'is_kda_in_proj', False),
    )

    assert [call.shape for call in calls] == [
        torch.Size([rows, param.size(1)]) for rows in local_shapes
    ]
    expected = torch.cat(
        [torch.full((rows, param.size(1)), i, device='cuda') for i, rows in enumerate(local_shapes, 1)]
    )
    assert torch.equal(output, expected)


def test_muon_builder_routes_kda_decay_parameters_to_scalar_optimizer(monkeypatch):
    class _FakeModelChunk:
        def __init__(self):
            self.config = SimpleNamespace(
                num_attention_heads=2,
                num_query_groups=1,
                kv_channels=4,
                multi_latent_attention=False,
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

    def fake_tensor_parallel_muon(params, **kwargs):
        del kwargs
        captured["muon"] = [p for group in params for p in group["params"]]
        return SimpleNamespace(param_groups=params, state={})

    def fake_get_megatron_optimizer(config, model_chunks, **kwargs):
        del kwargs
        captured["scalar_optimizer"] = config.optimizer
        captured["scalar"] = [
            p
            for model_chunk in model_chunks
            for _, p in model_chunk.named_parameters()
            if p.requires_grad
        ]
        return SimpleNamespace(chained_optimizers=[])

    monkeypatch.setattr(muon_module, "_get_param_groups", fake_get_param_groups)
    monkeypatch.setattr(muon_module, "TensorParallelMuon", fake_tensor_parallel_muon)
    monkeypatch.setattr(muon_module, "FP32Optimizer", lambda optimizer, *args: optimizer)
    monkeypatch.setattr(muon_module, "get_megatron_optimizer", fake_get_megatron_optimizer)
    monkeypatch.setattr(
        muon_module,
        "ChainedOptimizer",
        lambda optimizers: SimpleNamespace(chained_optimizers=optimizers),
    )

    config = OptimizerConfig(optimizer="muon", lr=0.01, min_lr=0.0)
    config.use_distributed_optimizer = False
    config.fp16 = False
    config.bf16 = False
    muon_module.get_megatron_muon_optimizer(
        config,
        [model],
        config_overrides={},
        pg_collection=SimpleNamespace(),
    )

    assert len(captured["muon"]) == 1 and captured["muon"][0] is model.in_proj
    assert len(captured["scalar"]) == 2
    assert captured["scalar_optimizer"] == "adam"
    assert captured["scalar"][0] is model.A_log
    assert captured["scalar"][1] is model.dt_bias


@pytest.mark.parametrize(
    ("split_mla_per_head", "expected_q_up_proj_head_dim"),
    [(False, None), (True, 8)],
)
def test_muon_builder_uses_per_head_mla_kv_split_shapes(
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
                q_lora_rank=None,
                gated_linear_unit=True,
            )
            self.kv_up = torch.nn.Parameter(torch.ones(88, 5))
            self.grouped_fc1 = torch.nn.Parameter(torch.ones(8, 5))
            self.offloaded_fc1 = torch.nn.Parameter(torch.ones(5, 8))

        def named_parameters(self):
            return iter(
                [
                    ("decoder.layers.0.self_attention.linear_kv_up_proj.weight", self.kv_up),
                    ("decoder.layers.0.mlp.experts.linear_fc1.weight0", self.grouped_fc1),
                    ("decoder.layers.0.mlp.experts.weight1_expert_0", self.offloaded_fc1),
                ]
            )

    captured_kwargs = {}

    def fake_tensor_parallel_muon(params, **kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(param_groups=params, state={})

    def fake_get_param_groups(model_chunks, config, config_overrides):
        del config, config_overrides
        return [
            {
                "params": [param for model in model_chunks for _, param in model.named_parameters()],
                "is_expert_parallel": False,
            }
        ]

    monkeypatch.setattr(muon_module, "TensorParallelMuon", fake_tensor_parallel_muon)
    monkeypatch.setattr(muon_module, "_get_param_groups", fake_get_param_groups)
    monkeypatch.setattr(muon_module, "FP32Optimizer", lambda optimizer, *args: optimizer)
    monkeypatch.setattr(
        muon_module,
        "get_megatron_optimizer",
        lambda *args, **kwargs: SimpleNamespace(chained_optimizers=[]),
    )
    monkeypatch.setattr(
        muon_module,
        "ChainedOptimizer",
        lambda optimizers: SimpleNamespace(chained_optimizers=optimizers),
    )

    config = OptimizerConfig(optimizer="muon", lr=0.01, min_lr=0.0)
    config.muon_split_qkv = True
    config.muon_split_mla_per_head = split_mla_per_head
    config.use_distributed_optimizer = False
    config.fp16 = False
    config.bf16 = False

    model_chunk = _FakeModelChunk()
    muon_module.get_megatron_muon_optimizer(
        config,
        [model_chunk],
        pg_collection=SimpleNamespace(),
    )

    assert captured_kwargs["kv_up_proj_split_shapes"] == (6, 5)
    assert captured_kwargs["q_up_proj_head_dim"] == expected_q_up_proj_head_dim
    assert model_chunk.grouped_fc1.glu_split_dim == 0
    assert not hasattr(model_chunk.offloaded_fc1, "glu_split_dim")


def test_muon_optimizer_mla_kv_up_proj_split():
    param = torch.nn.Parameter(torch.empty(10, 4, device='cuda'))
    param.is_kv_up_proj = True
    grad = torch.arange(40, dtype=torch.float32, device='cuda').view(10, 4)

    output, calls = _record_muon_split_output(
        param,
        grad,
        is_kv_up_proj_fn=lambda p: getattr(p, 'is_kv_up_proj', False),
        kv_up_proj_split_shapes=(3, 2),
    )

    assert [call.shape for call in calls] == [torch.Size([6, 4]), torch.Size([4, 4])]
    expected = torch.tensor(
        [1, 1, 1, 2, 2, 1, 1, 1, 2, 2], device='cuda'
    ).view(10, 1)
    assert torch.equal(output, expected.expand_as(output))


def test_muon_optimizer_mla_kv_up_proj_split_uses_local_dim0_tp_shapes():
    param = torch.nn.Parameter(torch.empty(10, 4, device='cuda'))
    param.is_kv_up_proj = True
    param.partition_dim = 0
    grad = torch.arange(40, dtype=torch.float32, device='cuda').view(10, 4)

    output, calls = _record_muon_split_output(
        param,
        grad,
        is_kv_up_proj_fn=lambda p: getattr(p, 'is_kv_up_proj', False),
        kv_up_proj_split_shapes=(12, 8),
    )

    assert [call.shape for call in calls] == [torch.Size([6, 4]), torch.Size([4, 4])]
    assert torch.equal(output[:6], torch.ones_like(output[:6]))
    assert torch.equal(output[6:], torch.full_like(output[6:], 2.0))


@pytest.mark.parametrize("split_qkv", [False, True])
def test_muon_optimizer_mla_kv_up_proj_split_per_head(split_qkv):
    param = torch.nn.Parameter(torch.empty(10, 4, device='cuda'))
    param.is_kv_up_proj = True
    grad = torch.arange(40, dtype=torch.float32, device='cuda').view(10, 4)

    output, calls = _record_muon_split_output(
        param,
        grad,
        is_kv_up_proj_fn=lambda p: getattr(p, 'is_kv_up_proj', False),
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
    expected = torch.tensor(
        [1] * 3 + [2] * 2 + [3] * 3 + [4] * 2, device='cuda'
    ).view(10, 1)
    assert torch.equal(output, expected.expand_as(output))


def test_muon_optimizer_mla_kv_up_proj_per_head_ignores_head_partition_dim():
    param = torch.nn.Parameter(torch.empty(10, 4, device='cuda'))
    param.is_kv_up_proj = True
    param.partition_dim = 0
    grad = torch.arange(40, dtype=torch.float32, device='cuda').view(10, 4)
    optimizer = TensorParallelMuon(
        params=[param],
        lr=0.01,
        split_qkv=True,
        num_ns_steps=5,
        is_kv_up_proj_fn=lambda p: getattr(p, 'is_kv_up_proj', False),
        kv_up_proj_split_shapes=(3, 2),
        split_mla_per_head=True,
        pg_collection=None,
        mode="distributed",
    )
    partition_dims = []

    def record_partition_dim(split_grad, tp_group, partition_dim):
        del tp_group
        partition_dims.append(partition_dim)
        return torch.full_like(split_grad, float(len(partition_dims)))

    optimizer.scaled_orthogonalize_fn = record_partition_dim
    output = optimizer.orthogonalize(param, grad)

    assert partition_dims == [None, None, None, None]
    expected = torch.tensor(
        [1] * 3 + [2] * 2 + [3] * 3 + [4] * 2, device='cuda'
    ).view(10, 1)
    assert torch.equal(output, expected.expand_as(output))


def test_muon_optimizer_mla_q_up_proj_split_per_head():
    param = torch.nn.Parameter(torch.empty(12, 4, device='cuda'))
    param.is_q_up_proj = True
    grad = torch.arange(48, dtype=torch.float32, device='cuda').view(12, 4)

    output, calls = _record_muon_split_output(
        param,
        grad,
        is_q_up_proj_fn=lambda p: getattr(p, 'is_q_up_proj', False),
        q_up_proj_head_dim=4,
        split_mla_per_head=True,
    )

    assert [call.shape for call in calls] == [
        torch.Size([4, 4]),
        torch.Size([4, 4]),
        torch.Size([4, 4]),
    ]
    expected = torch.tensor([1] * 4 + [2] * 4 + [3] * 4, device='cuda').view(12, 1)
    assert torch.equal(output, expected.expand_as(output))


def test_muon_optimizer_mla_q_up_proj_per_head_ignores_head_partition_dim():
    param = torch.nn.Parameter(torch.empty(12, 4, device='cuda'))
    param.is_q_up_proj = True
    param.partition_dim = 0
    grad = torch.arange(48, dtype=torch.float32, device='cuda').view(12, 4)
    optimizer = TensorParallelMuon(
        params=[param],
        lr=0.01,
        split_qkv=True,
        num_ns_steps=5,
        is_q_up_proj_fn=lambda p: getattr(p, 'is_q_up_proj', False),
        q_up_proj_head_dim=4,
        split_mla_per_head=True,
        pg_collection=None,
        mode="distributed",
    )
    partition_dims = []

    def record_partition_dim(split_grad, tp_group, partition_dim):
        del tp_group
        partition_dims.append(partition_dim)
        return torch.full_like(split_grad, float(len(partition_dims)))

    optimizer.scaled_orthogonalize_fn = record_partition_dim
    output = optimizer.orthogonalize(param, grad)

    assert partition_dims == [None, None, None]
    expected = torch.tensor([1] * 4 + [2] * 4 + [3] * 4, device='cuda').view(12, 1)
    assert torch.equal(output, expected.expand_as(output))


def test_muon_optimizer_mla_qkv_down_proj_split_mechanics():
    param = torch.nn.Parameter(torch.empty(5, 4, device='cuda'))
    param.is_qkv_down_proj = True
    grad = torch.arange(20, dtype=torch.float32, device='cuda').view(5, 4)

    output, calls = _record_muon_split_output(
        param,
        grad,
        is_qkv_down_proj_fn=lambda p: getattr(p, 'is_qkv_down_proj', False),
        qkv_down_proj_split_shapes=(2, 3),
    )

    assert [call.shape for call in calls] == [torch.Size([2, 4]), torch.Size([3, 4])]
    assert torch.equal(output[:2], torch.ones_like(output[:2]))
    assert torch.equal(output[2:], torch.full_like(output[2:], 2.0))


def test_muon_optimizer_mla_qkv_down_proj_split_uses_local_dim0_tp_shapes():
    param = torch.nn.Parameter(torch.empty(6, 4, device='cuda'))
    param.is_qkv_down_proj = True
    param.partition_dim = 0
    grad = torch.arange(24, dtype=torch.float32, device='cuda').view(6, 4)

    output, calls = _record_muon_split_output(
        param,
        grad,
        is_qkv_down_proj_fn=lambda p: getattr(p, 'is_qkv_down_proj', False),
        qkv_down_proj_split_shapes=(4, 8),
    )

    assert [call.shape for call in calls] == [torch.Size([2, 4]), torch.Size([4, 4])]
    assert torch.equal(output[:2], torch.ones_like(output[:2]))
    assert torch.equal(output[2:], torch.full_like(output[2:], 2.0))


def test_muon_mla_param_tags_copy_to_main_param():
    param = torch.empty(2, 2)
    tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
    param.is_kv_up_proj = True
    param.is_q_up_proj = True
    param.is_qkv_down_proj = True
    main_param = torch.empty_like(param)

    tensor_parallel.copy_tensor_model_parallel_attributes(main_param, param)

    assert main_param.is_kv_up_proj
    assert main_param.is_q_up_proj
    assert main_param.is_qkv_down_proj


def test_muon_optimizer_extra_scale_factor():
    """Test TensorParallelMuon optimizer with different extra_scale_factor values."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        extra_scale_factor=2.0,
        num_ns_steps=5,
        pg_collection=None,
        mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated with extra_scale_factor"


@requires_eo_v02
def test_get_supported_coefficient_types_returns_tuple():
    """Test that get_supported_coefficient_types returns a non-empty tuple of strings."""
    supported = get_supported_coefficient_types()
    assert isinstance(supported, tuple)
    assert len(supported) > 0
    for t in supported:
        assert isinstance(t, str)


@requires_eo_v02
def test_get_supported_coefficient_types_contains_known_types():
    """Test that the known coefficient types are present in the supported set."""
    supported = get_supported_coefficient_types()
    for expected in ("simple", "quintic", "polar_express"):
        assert expected in supported, f"Expected '{expected}' in supported types {supported}"


@requires_eo_v02
def test_validate_coefficient_type_accepts_valid():
    """Test that validate_coefficient_type does not raise for valid types."""
    for t in get_supported_coefficient_types():
        validate_coefficient_type(t)  # should not raise


def test_validate_coefficient_type_rejects_invalid():
    """Test that validate_coefficient_type raises ValueError for an invalid type."""
    with pytest.raises(ValueError, match="Unsupported muon coefficient type"):
        validate_coefficient_type("nonexistent_type_xyz")


def test_muon_optimizer_invalid_coefficient_type():
    """Test that TensorParallelMuon raises ValueError for an invalid coefficient_type."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)

    with pytest.raises(ValueError, match="Unsupported muon coefficient type"):
        TensorParallelMuon(
            params=[model.weight],
            lr=0.01,
            coefficient_type="nonexistent_type_xyz",
            num_ns_steps=5,
            pg_collection=None,
            mode="duplicated",
        )


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonCoefficientTypeMultiRank:
    """Test coefficient_type integration through get_megatron_muon_optimizer."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    @pytest.mark.parametrize("coefficient_type", _TESTABLE_COEFFICIENT_TYPES)
    def test_get_megatron_muon_optimizer_coefficient_type(self, coefficient_type):
        """Test that coefficient_type flows through get_megatron_muon_optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_coefficient_type=coefficient_type,
            muon_num_ns_steps=_DEFAULT_NS_STEPS,
            muon_tp_mode="duplicated",
        )

        optimizer = get_megatron_muon_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=True,
            layer_wise_distributed_optimizer=False,
        )

        assert optimizer is not None

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()


@pytest.mark.parametrize("num_ns_steps", [5, 15, 25])
def test_muon_optimizer_num_ns_steps(num_ns_steps):
    """Test TensorParallelMuon optimizer with different numbers of Newton-Schulz steps."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        coefficient_type="quintic",
        num_ns_steps=num_ns_steps,
        pg_collection=None,
        mode="duplicated",
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with num_ns_steps={num_ns_steps}"
