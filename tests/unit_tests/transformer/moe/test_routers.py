# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


from types import SimpleNamespace
from typing import cast

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import (
    consume_inference_router_violation_metrics,
    expert_load_entropy,
    get_updated_expert_bias,
    qb_dual_update,
    router_gating_linear,
    topk_routing_with_score_function,
)
from megatron.core.transformer.moe.router import Router, TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

try:
    # Check availability of TE fused router ops
    from megatron.core.extensions.transformer_engine import (
        fused_topk_with_score_function as _fused_topk_with_score_function,
    )

    HAVE_ROUTER_FUSION = _fused_topk_with_score_function is not None
except Exception:  # pragma: no cover - defensive
    HAVE_ROUTER_FUSION = False


def test_qb_dual_update_uses_column_quantile():
    scores = torch.tensor(
        [
            [0.1, 0.5, 0.2, 0.0],
            [0.7, 0.3, 0.4, 0.2],
            [0.6, 0.9, 0.1, 0.8],
            [0.2, 0.4, 0.6, 0.3],
            [0.9, 0.0, 0.5, 0.7],
            [0.3, 0.8, 0.7, 0.4],
        ],
        dtype=torch.float32,
    )
    beta = torch.zeros(scores.shape[1], dtype=torch.float32)
    topk = 2

    indices, beta_local = qb_dual_update(scores, topk, beta)

    topk_result = scores.topk(topk + 1, dim=1)
    expected_indices = topk_result.indices[:, :-1]
    alpha = topk_result.values[:, -1:]
    col_target = scores.shape[0] * topk // scores.shape[1]
    expected_beta = (scores - alpha).topk(col_target + 1, dim=0).values[-1]

    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(beta_local, expected_beta)


def test_expert_load_entropy_is_normalized():
    loads = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )

    torch.testing.assert_close(expert_load_entropy(loads), torch.tensor([1.0, 0.0, 1.0]))
    torch.testing.assert_close(expert_load_entropy(torch.tensor([[3.0]])), torch.ones(1))


def test_topk_routing_uses_precomputed_indices_for_probs():
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [0.0, 3.0, 2.0],
        ],
        dtype=torch.float32,
    )
    precomputed_indices = torch.tensor([[2, 1], [0, 2]], dtype=torch.long)

    probs, routing_map = topk_routing_with_score_function(
        logits,
        topk=2,
        score_function="sigmoid",
        precomputed_indices=precomputed_indices,
    )

    selected_scores = torch.gather(torch.sigmoid(logits), dim=1, index=precomputed_indices)
    expected_probs = selected_scores / selected_scores.sum(dim=-1, keepdim=True)
    expected_sparse_probs = torch.zeros_like(logits).scatter(1, precomputed_indices, expected_probs)
    expected_routing_map = torch.zeros_like(logits).scatter(1, precomputed_indices, 1).bool()

    torch.testing.assert_close(probs, expected_sparse_probs)
    assert torch.equal(routing_map, expected_routing_map)


def _metric_buffer_router(device):
    router = torch.nn.Module()
    router.config = SimpleNamespace(
        num_moe_experts=4,
        mtp_num_layers=None,
        mtp_use_repeated_layer=False,
        enable_cuda_graph=False,
        cuda_graph_impl="none",
    )
    router.is_mtp_layer = False
    router.register_buffer("mbs_expert_load_samples", None, persistent=False)
    router.register_buffer("seq_expert_load_samples", None, persistent=False)
    router.register_buffer(
        "expert_load_sample_count", torch.zeros((), dtype=torch.long, device=device), persistent=False
    )
    return router


def test_expert_load_samples_use_stable_registered_buffers(monkeypatch):
    monkeypatch.setattr("megatron.core.transformer.moe.router.get_num_microbatches", lambda: 2)
    router = _metric_buffer_router("cpu")
    first_mbs = torch.tensor([4.0, 2.0, 1.0, 1.0, 4.0])
    first_seq = torch.stack((first_mbs, first_mbs))

    TopKRouter._record_expert_load_samples(router, first_mbs, first_seq)
    mbs_data_ptr = router.mbs_expert_load_samples.data_ptr()
    seq_data_ptr = router.seq_expert_load_samples.data_ptr()
    second_mbs = first_mbs + 1
    second_seq = first_seq + 1
    TopKRouter._record_expert_load_samples(router, second_mbs, second_seq)

    assert router.mbs_expert_load_samples.data_ptr() == mbs_data_ptr
    assert router.seq_expert_load_samples.data_ptr() == seq_data_ptr
    assert router.expert_load_sample_count.item() == 2
    torch.testing.assert_close(router.mbs_expert_load_samples, torch.stack((first_mbs, second_mbs)))
    torch.testing.assert_close(router.seq_expert_load_samples, torch.stack((first_seq, second_seq)))
    assert "mbs_expert_load_samples" in router._buffers
    assert "seq_expert_load_samples" in router._buffers


def test_cuda_graph_metric_buffers_cannot_be_resized(monkeypatch):
    router = _metric_buffer_router("cpu")
    router.config.enable_cuda_graph = True
    sample = torch.zeros(5)
    monkeypatch.setattr("megatron.core.transformer.moe.router.get_num_microbatches", lambda: 2)
    TopKRouter._record_expert_load_samples(router, sample, None)
    monkeypatch.setattr("megatron.core.transformer.moe.router.get_num_microbatches", lambda: 3)

    with pytest.raises(RuntimeError, match="Cannot resize router metric buffer"):
        TopKRouter._record_expert_load_samples(router, sample, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_expert_load_sample_buffer_updates_during_cuda_graph_replay(monkeypatch):
    monkeypatch.setattr("megatron.core.transformer.moe.router.get_num_microbatches", lambda: 2)
    router = _metric_buffer_router("cuda")
    static_mbs = torch.zeros(5, device="cuda")
    static_seq = torch.zeros((2, 5), device="cuda")

    # Allocate the registered buffers before capture, as the normal graph warmup does.
    TopKRouter._record_expert_load_samples(router, static_mbs, static_seq)
    router.expert_load_sample_count.zero_()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        TopKRouter._record_expert_load_samples(router, static_mbs, static_seq)
    router.expert_load_sample_count.zero_()

    first_mbs = torch.arange(5, dtype=torch.float32, device="cuda")
    first_seq = torch.stack((first_mbs, first_mbs + 1))
    static_mbs.copy_(first_mbs)
    static_seq.copy_(first_seq)
    graph.replay()
    second_mbs = first_mbs + 10
    second_seq = first_seq + 10
    static_mbs.copy_(second_mbs)
    static_seq.copy_(second_seq)
    graph.replay()
    torch.cuda.synchronize()

    assert router.expert_load_sample_count.item() == 2
    torch.testing.assert_close(router.mbs_expert_load_samples, torch.stack((first_mbs, second_mbs)))
    torch.testing.assert_close(router.seq_expert_load_samples, torch.stack((first_seq, second_seq)))


class TestTop2Router:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        init_num_microbatches_calculator(
            rank=0,
            rampup_batch_size=None,
            global_batch_size=1,
            micro_batch_size=1,
            data_parallel_size=1,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")
        num_moe_experts = 4
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        submodules = get_gpt_layer_local_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        self.sequential_mlp = MoELayer(self.transformer_config, submodules.mlp.submodules)
        self.router = cast(Router, self.sequential_mlp.router)

    def teardown_method(self, method):
        unset_num_microbatches_calculator()
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.router, Router)

        num_weights = sum([p.numel() for p in self.router.parameters()])
        assert num_weights == 12 * 4, num_weights

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("moe_router_pre_softmax", [(True), (False)])
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward(self, moe_router_pre_softmax, score_function):
        with torch.no_grad():
            self.router = self.router.cuda()
            self.router.config.moe_router_pre_softmax = moe_router_pre_softmax
            self.router.config.moe_router_score_function = score_function
            # [num tokens, hidden size]
            hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
            hidden_states = hidden_states.cuda().bfloat16()
            scores, indices = self.router(hidden_states)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_ROUTER_FUSION,
        reason="TE fused router ops not available",
    )
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward_fusion_equivalence(self, score_function):
        with torch.no_grad():
            self.router = self.router.cuda()
            self.router.config.moe_router_score_function = score_function
            hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
            hidden_states = hidden_states.cuda().bfloat16()

            # Unfused
            self.router.config.moe_router_fusion = False
            scores_ref, routing_ref = self.router(hidden_states)

            # Fused
            self.router.config.moe_router_fusion = True
            scores_fused, routing_fused = self.router(hidden_states)

            assert torch.equal(routing_ref, routing_fused), "Routing map mismatch"
            torch.testing.assert_close(scores_ref, scores_fused)
            # restore the config
            self.router.config.moe_router_fusion = False

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_inference_violation_metric_collection(self):
        self.router = self.router.cuda()
        self.router.eval()
        hidden_states = torch.randn(
            (8, 2, self.router.config.hidden_size), device='cuda', dtype=torch.bfloat16
        )

        # Disabled by default, and eval with gradients enabled must not collect samples.
        with torch.no_grad():
            self.router(hidden_states)
        assert not self.router.inference_mbs_expert_load_samples
        assert not self.router.inference_seq_expert_load_samples
        self.router.config.moe_router_inference_violation_metrics = ['mbs', 'seq']
        self.router(hidden_states)
        assert not self.router.inference_mbs_expert_load_samples
        assert not self.router.inference_seq_expert_load_samples

        with torch.no_grad():
            self.router(hidden_states)
        assert len(self.router.inference_mbs_expert_load_samples) == 1
        assert len(self.router.inference_seq_expert_load_samples) == 1
        assert self.router.inference_mbs_expert_load_samples[0].shape == (5,)
        assert self.router.inference_seq_expert_load_samples[0].shape == (2, 5)
        assert self.router.inference_mbs_expert_load_samples[0][-1] == 16
        torch.testing.assert_close(
            self.router.inference_seq_expert_load_samples[0][:, -1],
            torch.tensor([8.0, 8.0], device='cuda'),
        )
        assert not self.router.mbs_expert_load_samples
        assert not self.router.seq_expert_load_samples

        metrics = consume_inference_router_violation_metrics(self.router)
        assert set(metrics) == {
            f'moe_router_{scope}_{stat}_violation'
            for scope in ('mbs', 'seq')
            for stat in ('max', 'min', 'median', 'std')
        }
        assert not self.router.inference_mbs_expert_load_samples
        assert not self.router.inference_seq_expert_load_samples

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_aux_loss(self):
        self.sequential_mlp = self.sequential_mlp.cuda()

        # Without aux loss
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
        hidden_states = hidden_states.cuda().bfloat16()
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() == 0

        # With aux loss
        self.transformer_config.moe_aux_loss_coeff = 1
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() > 0

        # With Z loss
        self.transformer_config.moe_aux_loss_coeff = 0
        self.transformer_config.moe_z_loss_coeff = 1
        self.sequential_mlp.router.weight.grad.fill_(0)
        out = self.sequential_mlp(hidden_states)[0]
        out.sum().mul_(0).backward()
        assert self.sequential_mlp.router.weight.grad.abs().sum() > 0

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_with_padding_mask(self):
        """Test that padding mask correctly excludes padding tokens from routing."""
        self.router = self.router.cuda()
        seq_len = 32
        batch_size = 2
        hidden_size = self.router.config.hidden_size

        # Create input with shape [seq_len, batch_size, hidden_size]
        hidden_states = torch.randn((seq_len, batch_size, hidden_size)).cuda().bfloat16()

        # Create padding mask: first half valid, second half padding
        # padding_mask shape: [seq_len, batch_size]
        # Convention: True = padding (exclude), False = valid (include)
        padding_mask = torch.zeros((seq_len, batch_size), dtype=torch.bool, device='cuda')
        padding_mask[seq_len // 2 :, :] = True  # Second half is padding

        # Test forward pass with padding mask
        with torch.no_grad():
            probs_with_mask, routing_map_with_mask = self.router(
                hidden_states, padding_mask=padding_mask
            )

            # Test forward pass without padding mask (only valid tokens)
            hidden_states_valid = hidden_states[: seq_len // 2, :, :]
            probs_without_mask, routing_map_without_mask = self.router(hidden_states_valid)

            # The valid part of routing with mask should match routing without mask
            probs_valid_part = probs_with_mask.reshape(seq_len, batch_size, -1)[
                : seq_len // 2, :, :
            ]
            probs_valid_part = probs_valid_part.reshape(-1, probs_valid_part.shape[-1])

            # Check that shapes are as expected
            assert probs_with_mask.shape == (
                seq_len * batch_size,
                self.router.config.num_moe_experts,
            )
            assert routing_map_with_mask.shape == (
                seq_len * batch_size,
                self.router.config.num_moe_experts,
            )

            # Verify that probs for valid tokens are similar
            assert torch.equal(probs_valid_part, probs_without_mask)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_dtype(self):
        self.router = self.router.cuda()
        self.sequential_mlp = self.sequential_mlp.cuda()
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size), dtype=torch.bfloat16)
        hidden_states = hidden_states.cuda()

        # Test with default setting (bf16)
        self.router.config.moe_router_dtype = None
        with torch.no_grad():
            scores, routing_map = self.router(hidden_states)
            out = self.sequential_mlp(hidden_states)
            assert scores.dtype == torch.bfloat16, "Router output should be bf16 by default"
            assert out[0].dtype == torch.bfloat16

        # Test with fp32 enabled
        self.router.config.moe_router_dtype = 'fp32'
        with torch.no_grad():
            scores, routing_map = self.router(hidden_states)
            out = self.sequential_mlp(hidden_states)
            assert scores.dtype == torch.float32, "Router output should be fp32 when enabled"
            assert out[0].dtype == torch.bfloat16
            self.sequential_mlp.config.moe_token_dispatcher_type = "alltoall"
            out = self.sequential_mlp(hidden_states)
            assert out[0].dtype == torch.bfloat16
            self.sequential_mlp.config.moe_token_dispatcher_type = "allgather"

        # Test with fp64 enabled
        self.router.config.moe_router_dtype = 'fp64'
        with torch.no_grad():
            scores, routing_map = self.router(hidden_states)
            out = self.sequential_mlp(hidden_states)
            assert scores.dtype == torch.float64, "Router output should be fp64 when enabled"
            assert out[0].dtype == torch.bfloat16

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_force_load_balancing(self):
        hidden_states = torch.randn(
            (32, 2, self.router.config.hidden_size), device="cuda", dtype=torch.bfloat16
        )
        hidden_states.requires_grad = True

        # First forward pass with normal routing
        normal_scores, normal_routing_map = self.router(hidden_states)

        # Second forward pass with force load balancing
        self.router.config.moe_router_force_load_balancing = True
        force_scores, force_routing_map = self.router(hidden_states)

        assert normal_scores.shape == force_scores.shape
        assert normal_routing_map.shape == force_routing_map.shape
        assert torch.equal(normal_scores, force_scores) == False

        # Backward pass for force load balancing
        self.router.zero_grad()
        force_scores.sum().backward()
        assert hidden_states.grad is not None
        assert self.router.weight.grad.norm() > 0

        self.router.config.moe_router_force_load_balancing = False

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("capacity_factor", [None, 1.0, 2.0])
    @pytest.mark.parametrize("drop_policy", ["probs", "position"])
    @pytest.mark.parametrize("pad_to_capacity", [True, False])
    def test_token_dropping(self, capacity_factor, drop_policy, pad_to_capacity):
        if capacity_factor is None and pad_to_capacity:
            pytest.skip("Capacity factor is None, so no token dropping should be applied")

        num_tokens = 32
        self.router = self.router.cuda()
        self.router.config.moe_expert_capacity_factor = capacity_factor
        self.router.config.moe_token_drop_policy = drop_policy
        self.router.config.moe_pad_expert_input_to_capacity = pad_to_capacity

        hidden_states = torch.randn(
            (num_tokens, self.router.config.hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        hidden_states.requires_grad = True
        probs, routing_map = self.router(hidden_states)

        if capacity_factor is not None:
            if pad_to_capacity:
                assert (
                    routing_map.sum().item()
                    == num_tokens * self.router.config.moe_router_topk * capacity_factor
                )
            else:
                assert (
                    routing_map.sum().item()
                    <= num_tokens * self.router.config.moe_router_topk * capacity_factor
                )
        else:
            assert routing_map.sum().item() == num_tokens * self.router.config.moe_router_topk

        # restore the config
        self.router.config.moe_expert_capacity_factor = None
        self.router.config.moe_token_drop_policy = "probs"
        self.router.config.moe_pad_expert_input_to_capacity = False


class TestGroupLimitedRouter:
    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=8,
            context_parallel_size=1,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")

        num_moe_experts = 16
        self.transformer_config = TransformerConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=8,
            context_parallel_size=1,
            num_moe_experts=num_moe_experts,
            moe_router_topk=4,
            moe_router_group_topk=2,
            moe_router_num_groups=8,
            moe_router_pre_softmax=True,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0,
            moe_router_dtype='fp32',
            moe_token_dispatcher_type="alltoall",
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )

        # init MoE layer
        submodules = get_gpt_layer_local_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        self.moe_layer = MoELayer(self.transformer_config, submodules.mlp.submodules).cuda()
        self.router = cast(Router, self.moe_layer.router)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.router, Router)

        num_weights = sum([p.numel() for p in self.router.parameters()])
        assert (
            num_weights
            == self.transformer_config.hidden_size * self.transformer_config.num_moe_experts
        ), num_weights

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("moe_router_group_topk,moe_router_num_groups", [(3, 8), (2, 4)])
    @pytest.mark.parametrize("moe_router_pre_softmax", [(True), (False)])
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward(
        self, moe_router_group_topk, moe_router_num_groups, moe_router_pre_softmax, score_function
    ):
        with torch.no_grad():
            self.router.config.moe_router_group_topk = moe_router_group_topk
            self.router.config.moe_router_num_groups = moe_router_num_groups
            self.router.config.moe_router_pre_softmax = moe_router_pre_softmax
            self.router.config.moe_router_score_function = score_function
            if moe_router_pre_softmax:
                self.router.config.moe_router_topk_scaling_factor = 16.0

            seq_len = 128
            batch_size = 4
            num_tokens = seq_len * batch_size
            # hidden_states shape: [seq_len, batch_size, hidden_size]
            hidden_states = (
                torch.randn((seq_len, batch_size, self.router.config.hidden_size)).cuda().bfloat16()
            )
            scores, routing_map = self.router(hidden_states)
            assert scores.shape == (num_tokens, self.router.config.num_moe_experts), scores.shape
            assert routing_map.shape == (
                num_tokens,
                self.router.config.num_moe_experts,
            ), routing_map.shape

            group_routing_map = (
                routing_map.reshape(num_tokens, moe_router_num_groups, -1).max(dim=-1).values
            )
            assert torch.all(group_routing_map.sum(dim=-1) <= moe_router_group_topk)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_ROUTER_FUSION,
        reason="TE fused router ops not available",
    )
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward_fusion_equivalence(self, score_function):
        with torch.no_grad():
            self.router = self.router.cuda()
            self.router.score_function = score_function
            seq_len = 32
            batch_size = 4
            hidden_states = torch.randn((seq_len, batch_size, self.router.config.hidden_size))
            hidden_states = hidden_states.cuda().bfloat16()

            # Unfused
            self.router.config.moe_router_fusion = False
            scores_ref, routing_ref = self.router(hidden_states)

            # Fused
            self.router.config.moe_router_fusion = True
            scores_fused, routing_fused = self.router(hidden_states)

            assert torch.equal(routing_ref, routing_fused), "Routing map mismatch"
            torch.testing.assert_close(scores_ref, scores_fused)
            # restore the config
            self.router.config.moe_router_fusion = False


class TestAuxLossFreeTop2Router:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=8)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        print("done intializing")
        num_moe_experts = 8
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            expert_model_parallel_size=8,
            moe_router_load_balancing_type="none",  # No aux loss
            moe_router_score_function="sigmoid",  # Using sigmoid scoring
            moe_router_enable_expert_bias=True,  # Enable expert bias
            moe_router_bias_update_rate=0.1,  # Set bias update rate
            moe_router_topk=2,
            moe_router_violation_metrics=[],
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        submodules = get_gpt_layer_local_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        self.moe_layer = MoELayer(self.transformer_config, submodules.mlp.submodules)
        self.router = cast(Router, self.moe_layer.router)
        assert self.router.expert_bias is not None
        assert self.router.local_tokens_per_expert is not None

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_router_forward_aux_free(self):
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size))
        hidden_states = hidden_states.cuda().bfloat16()
        self.router = self.router.cuda()

        # First forward pass
        initial_bias = self.router.expert_bias.clone()
        scores1, indices1 = self.router(hidden_states)
        initial_tokens = self.router.local_tokens_per_expert.clone()
        updated_bias = get_updated_expert_bias(
            self.router.local_tokens_per_expert,
            self.router.expert_bias,
            self.router.config.moe_router_bias_update_rate,
        )

        # Verify expert bias was updated
        assert not torch.equal(initial_bias, updated_bias), "Expert bias should be updated"

        # Basic output checks
        assert scores1.shape == (64, 8), "Router scores shape mismatch"
        assert indices1.shape == (64, 8), "Router indices shape mismatch"

        # Print some debug info
        print("Updated bias after first forward pass:", updated_bias)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_ROUTER_FUSION,
        reason="TE fused router ops not available",
    )
    @pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
    def test_router_forward_fusion_equivalence(self, score_function):
        with torch.no_grad():
            # Build two fresh routers to avoid bias update interference
            submodules = get_gpt_layer_local_submodules(
                num_experts=self.transformer_config.num_moe_experts, moe_grouped_gemm=False
            )
            moe_layer_ref = MoELayer(self.transformer_config, submodules.mlp.submodules)
            moe_layer_fused = MoELayer(self.transformer_config, submodules.mlp.submodules)
            router_ref = moe_layer_ref.router.cuda()
            router_fused = moe_layer_fused.router.cuda()

            # Ensure identical initial parameters/state
            router_fused.weight.copy_(router_ref.weight)
            expert_bias_sample = torch.randn_like(router_ref.expert_bias)
            router_ref.expert_bias.copy_(expert_bias_sample)
            router_fused.expert_bias.copy_(expert_bias_sample)

            router_ref.config.moe_router_score_function = score_function
            router_fused.config.moe_router_score_function = score_function

            hidden_states = torch.randn((32, 2, router_ref.config.hidden_size))
            hidden_states = hidden_states.cuda().bfloat16()

            # Unfused
            router_ref.config.moe_router_fusion = False
            scores_ref, routing_ref = router_ref(hidden_states)

            # Fused
            router_fused.config.moe_router_fusion = True
            scores_fused, routing_fused = router_fused(hidden_states)

            assert torch.equal(routing_ref, routing_fused)
            torch.testing.assert_close(scores_ref, scores_fused)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("router_dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_router_gating_linear(router_dtype):
    tols = dict(rtol=2.0e-2, atol=1.0e-3)

    ref_inp = torch.randn((4096, 7168), dtype=torch.bfloat16, device="cuda")
    ref_weight = torch.randn((256, 7168), dtype=torch.bfloat16, device="cuda")
    ref_inp.requires_grad = True
    ref_weight.requires_grad = True
    bwd_input = torch.randn((4096, 256), dtype=router_dtype, device="cuda")

    ref_output = torch.nn.functional.linear(ref_inp.to(router_dtype), ref_weight.to(router_dtype))
    ref_output.backward(bwd_input)

    inp = ref_inp.detach()
    weight = ref_weight.detach()
    inp.requires_grad = True
    weight.requires_grad = True
    bias = None
    output = router_gating_linear(inp, weight, bias, router_dtype)
    output.backward(bwd_input)

    assert output.dtype == router_dtype
    assert ref_inp.grad.dtype == ref_inp.dtype
    assert ref_weight.grad.dtype == ref_weight.dtype
    assert torch.allclose(output, ref_output, **tols)
    assert torch.allclose(inp.grad, ref_inp.grad, **tols)
    assert torch.allclose(weight.grad, ref_weight.grad, **tols)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("router_dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_router_gating_linear_bias(router_dtype):
    tols = dict(rtol=2.0e-2, atol=1.0e-3)

    ref_inp = torch.randn((4096, 7168), dtype=router_dtype, device="cuda")
    ref_weight = torch.randn((256, 7168), dtype=router_dtype, device="cuda")
    ref_bias = torch.randn((256,), dtype=router_dtype, device="cuda")
    ref_inp.requires_grad = True
    ref_weight.requires_grad = True
    ref_bias.requires_grad = True
    bwd_input = torch.randn((4096, 256), dtype=router_dtype, device="cuda")

    ref_output = torch.nn.functional.linear(
        ref_inp.to(router_dtype), ref_weight.to(router_dtype), ref_bias.to(router_dtype)
    )
    ref_output.backward(bwd_input)

    inp = ref_inp.detach()
    weight = ref_weight.detach()
    bias = ref_bias.detach()
    inp.requires_grad = True
    weight.requires_grad = True
    bias.requires_grad = True
    output = router_gating_linear(inp, weight, bias, router_dtype)
    output.backward(bwd_input)

    assert output.dtype == router_dtype
    assert ref_inp.grad.dtype == ref_inp.dtype
    assert ref_weight.grad.dtype == ref_weight.dtype
    assert ref_bias.grad.dtype == ref_bias.dtype
    assert torch.allclose(output, ref_output, **tols)
    assert torch.allclose(inp.grad, ref_inp.grad, **tols)
    assert torch.allclose(weight.grad, ref_weight.grad, **tols)
    assert torch.allclose(bias.grad, ref_bias.grad, **tols)


class TestRoutingPaddingMaskOrientation:
    """MoELayer.route must exclude exactly the padded tokens, in the right order.

    The mask arrives from the model as [bsz, seq_length] while hidden_states is
    [seq_length, bsz, hidden]. A missing or doubled transpose still excludes the
    right NUMBER of tokens, so the counts below are built to distinguish which
    tokens were excluded, not just how many.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        init_num_microbatches_calculator(
            rank=0,
            rampup_batch_size=None,
            global_batch_size=2,
            micro_batch_size=2,
            data_parallel_size=1,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.num_experts = 8
        self.config = TransformerConfig(
            num_layers=2,
            hidden_size=self.num_experts,
            num_attention_heads=4,
            num_moe_experts=self.num_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="none",
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rate=0.1,
            moe_router_topk=1,
            add_bias_linear=False,
        )
        submodules = get_gpt_layer_local_submodules(
            num_experts=self.num_experts, moe_grouped_gemm=False
        )
        self.moe_layer = MoELayer(self.config, submodules.mlp.submodules)
        self.router = cast(Router, self.moe_layer.router)

    def teardown_method(self, method):
        unset_num_microbatches_calculator()
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_route_excludes_exactly_the_padded_tokens(self):
        seq_len, bsz = 4, 2
        self.moe_layer = self.moe_layer.cuda()
        self.router = cast(Router, self.moe_layer.router)

        # Identity gating: flattened token k (seq-major) selects expert k.
        with torch.no_grad():
            self.router.weight.copy_(torch.eye(self.num_experts, device="cuda"))
        hidden_states = (
            torch.eye(seq_len * bsz, self.config.hidden_size, device="cuda")
            .reshape(seq_len, bsz, self.config.hidden_size)
            .contiguous()
            * 10.0
        )

        # [bsz, seq_length], True = padding. Deliberately asymmetric: sample 0 keeps
        # 2 tokens, sample 1 keeps 3, so b-major and s-major flattenings disagree.
        padding_mask = torch.tensor(
            [[False, False, True, True], [False, False, False, True]], device="cuda"
        )
        # transpose -> [s, b] -> flatten seq-major: F F F F T F T T
        expected = torch.tensor([1, 1, 1, 1, 0, 1, 0, 0], device="cuda", dtype=torch.float32)
        # what a dropped transpose would have produced, kept to prove the test bites
        wrong_if_not_transposed = torch.tensor(
            [1, 1, 0, 0, 1, 1, 1, 0], device="cuda", dtype=torch.float32
        )

        self.router.local_tokens_per_expert.zero_()
        self.moe_layer.route(hidden_states, padding_mask)

        torch.testing.assert_close(self.router.local_tokens_per_expert, expected)
        assert not torch.equal(self.router.local_tokens_per_expert, wrong_if_not_transposed)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_route_without_mask_counts_every_token(self):
        seq_len, bsz = 4, 2
        self.moe_layer = self.moe_layer.cuda()
        self.router = cast(Router, self.moe_layer.router)
        with torch.no_grad():
            self.router.weight.copy_(torch.eye(self.num_experts, device="cuda"))
        hidden_states = (
            torch.eye(seq_len * bsz, self.config.hidden_size, device="cuda")
            .reshape(seq_len, bsz, self.config.hidden_size)
            .contiguous()
            * 10.0
        )
        self.router.local_tokens_per_expert.zero_()
        self.moe_layer.route(hidden_states, None)
        torch.testing.assert_close(
            self.router.local_tokens_per_expert, torch.ones(self.num_experts, device="cuda")
        )


def test_every_route_call_site_passes_a_padding_mask():
    """Guard the regression class: a bare mlp.route(x) silently disables the mask.

    Three call sites (the EP-overlap schedule node and both TE CUDA-graph replay
    branches) used to drop it, so the exclusion was a no-op under
    --overlap-moe-expert-parallel-comm while still being applied elsewhere.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[4] / "megatron" / "core"
    offenders = []
    # Every way into MoE routing: route(), _forward_mlp(), and calling the MoELayer
    # directly. Enumerating only the ones I thought of is how this bug class recurred:
    # the CUDA-graph replay path called _forward_mlp positionally, so padding_mask fell
    # back to its default of None.
    call = re.compile(
        r"(?<!def )\b(?:self|layer|super\(\))\.(?:mlp\.)?(?:route|_forward_mlp)\("
        r"|(?<!def )\b(?:self|layer)\.mlp\((?!\s*\))"
    )
    for path in root.rglob("*.py"):
        if "inference" in path.parts:  # flask @bp.route decorators
            continue
        text = path.read_text()
        for match in call.finditer(text):
            depth, i = 0, match.end() - 1
            while i < len(text):                      # slice to the matching paren
                depth += (text[i] == "(") - (text[i] == ")")
                if depth == 0:
                    break
                i += 1
            args = text[match.end() : i]
            if "*args" in args and "**kwargs" in args:
                continue  # forwarding wrapper: passes on whatever it was given
            if "intermediate_tensors" in args and "padding_mask" not in args:
                continue  # a later pipeline step (dispatch/postprocess); does not route
            if "padding_mask" not in args:
                offenders.append(f"{path.relative_to(root)}: {match.group(0)}{args[:50]}")
    assert not offenders, "MoE routing entered without a padding_mask:\n" + "\n".join(offenders)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("pad", ["none", "some", "all"])
def test_masked_qb_histogram_matches_the_compacting_path(pad):
    """The mask must only change HOW padding is excluded, never the counts.

    PR #76 excluded it as scores[~padding_mask], which is correct but synchronizes the
    device. Padded rows now go to a scratch expert block that is sliced off, so the two
    have to agree exactly -- including with nothing padded and with everything padded.
    """
    from megatron.core.transformer.moe.moe_utils import compute_qb_histogram

    torch.manual_seed(0)
    num_tokens, num_experts, num_bins = 512, 16, 64
    scores = torch.randn(num_tokens, num_experts, device="cuda")
    alpha = scores.max(dim=1).values
    beta = torch.rand(num_experts, device="cuda")
    padding_mask = torch.zeros(num_tokens, dtype=torch.bool, device="cuda")
    if pad == "some":
        padding_mask[300:] = True
    elif pad == "all":
        padding_mask[:] = True

    masked = compute_qb_histogram(scores, alpha, beta, num_bins, padding_mask=padding_mask)
    keep = ~padding_mask
    # beta sets the bin edges and is not indexed, so the two paths bin identically.
    compacted = (
        compute_qb_histogram(scores[keep], alpha[keep], beta, num_bins)
        if bool(keep.any())
        else torch.zeros_like(masked)
    )
    assert torch.equal(masked, compacted)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_route_marks_the_layer_as_routing_with_a_padding_mask():
    """CUDA graph capture decides from this flag whether the graph takes a mask input.

    Capture reads get_layer_static_inputs, not the live forward, so it needs some record
    that this layer routes with a mask. Capture happens after the warmup forwards, so a
    flag set on the first masked route() is set by then. It must stay False otherwise, or
    graphs for runs without BFD padding would gain an input they never receive.
    """
    Utils.initialize_model_parallel(1, 1)
    init_num_microbatches_calculator(
        rank=0, rampup_batch_size=None, global_batch_size=2, micro_batch_size=2,
        data_parallel_size=1,
    )
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    try:
        num_experts, seq_len, bsz, hidden = 4, 4, 2, 8
        config = TransformerConfig(
            num_layers=1, hidden_size=hidden, num_attention_heads=4,
            num_moe_experts=num_experts, use_cpu_initialization=True,
            moe_router_load_balancing_type="none", moe_router_score_function="sigmoid",
            moe_router_topk=1, add_bias_linear=False,
        )
        submodules = get_gpt_layer_local_submodules(
            num_experts=num_experts, moe_grouped_gemm=False
        )
        layer = MoELayer(config, submodules.mlp.submodules).cuda()
        hidden_states = torch.randn(seq_len, bsz, hidden, device="cuda")

        assert layer.routes_with_padding_mask is False
        layer.route(hidden_states, None)
        assert layer.routes_with_padding_mask is False, "no mask must not arm the flag"

        layer.route(hidden_states, torch.zeros(bsz, seq_len, dtype=torch.bool, device="cuda"))
        assert layer.routes_with_padding_mask is True
    finally:
        unset_num_microbatches_calculator()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_masked_qb_histogram_is_capturable_and_not_frozen():
    """Capture the masked histogram, then replay it under a different mask."""
    from megatron.core.transformer.moe.moe_utils import compute_qb_histogram

    num_tokens, num_experts, num_bins = 512, 16, 32
    scores = torch.rand(num_tokens, num_experts, device="cuda")
    alpha = torch.rand(num_tokens, device="cuda")
    beta = torch.rand(num_experts, device="cuda") * 0.2
    mask = torch.zeros(num_tokens, dtype=torch.bool, device="cuda")
    accum = torch.zeros(num_experts, num_bins, dtype=torch.long, device="cuda")

    def step():
        accum.add_(compute_qb_histogram(scores, alpha, beta, num_bins, padding_mask=mask))

    def expected(n_valid):
        return n_valid * num_experts

    mask[400:] = True
    step()
    torch.cuda.synchronize()
    accum.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    accum.zero_()
    mask.fill_(False)
    mask[400:] = True
    graph.replay()
    torch.cuda.synchronize()
    assert accum.sum().item() == expected(400)

    accum.zero_()
    mask.fill_(False)
    mask[100:] = True
    graph.replay()
    torch.cuda.synchronize()
    assert accum.sum().item() == expected(100), "mask was frozen at capture"


class TestPaddingMaskBufferAliasing:
    """route() must not retain the caller's mask tensor.

    TE copies graph inputs into its own static buffers, so the mask a caller hands in may
    legitimately be reused or overwritten afterwards. If any op saved a VIEW of it for
    backward, that later write would silently change this microbatch's gradients.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        init_num_microbatches_calculator(
            rank=0,
            rampup_batch_size=None,
            global_batch_size=2,
            micro_batch_size=2,
            data_parallel_size=1,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        unset_num_microbatches_calculator()
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward_is_unaffected_by_a_later_microbatch(self):
        num_experts, seq_len, bsz, hidden = 8, 8, 2, 16
        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden,
            num_attention_heads=4,
            num_moe_experts=num_experts,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=1e-2,
            moe_z_loss_coeff=1e-3,
            moe_router_score_function="sigmoid",
            moe_router_topk=2,
            add_bias_linear=False,
            use_cpu_initialization=True,
        )
        submodules = get_gpt_layer_local_submodules(
            num_experts=num_experts, moe_grouped_gemm=False
        )
        mask_a = torch.zeros((bsz, seq_len), dtype=torch.bool, device="cuda")
        mask_a[:, 5:] = True
        mask_b = torch.zeros((bsz, seq_len), dtype=torch.bool, device="cuda")
        mask_b[:, 1:] = True

        def grad_of(clobber_after_forward):
            _set_random_seed(seed_=123, data_parallel_random_init=False)
            layer = MoELayer(config, submodules.mlp.submodules).cuda()
            layer.train()
            hidden_states = torch.randn(
                seq_len, bsz, hidden, device="cuda", requires_grad=True
            )
            mask = mask_a.clone()
            probs, _ = layer.route(hidden_states, mask)
            if clobber_after_forward:
                mask.copy_(mask_b)  # a later microbatch reuses the caller's tensor
            probs.sum().backward()
            return hidden_states.grad.clone()

        torch.testing.assert_close(grad_of(False), grad_of(True), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_capture_static_inputs_take_a_padding_mask_shaped_like_hidden_states():
    """TE captures from get_layer_static_inputs, not from the live forward's kwargs.

    So the placeholder must be there exactly when the layer routes with a mask, and it
    must follow the sharded shape of hidden_states, since TE copies the live mask into it
    on every replay and a shape mismatch there would be a hard error at the first replay.
    """
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.enums import CudaGraphScope
    from megatron.core.transformer.transformer_layer import TransformerLayer

    Utils.initialize_model_parallel(1, 1)
    init_num_microbatches_calculator(
        rank=0, rampup_batch_size=None, global_batch_size=2, micro_batch_size=2,
        data_parallel_size=1,
    )
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    try:
        num_experts, seq_len, mbs, hidden = 4, 8, 2, 8
        config = TransformerConfig(
            num_layers=1, hidden_size=hidden, num_attention_heads=4,
            num_moe_experts=num_experts, use_cpu_initialization=True,
            moe_router_load_balancing_type="none", moe_router_score_function="sigmoid",
            moe_router_topk=1, add_bias_linear=False,
        )
        spec = get_gpt_layer_local_spec(num_experts=num_experts, moe_grouped_gemm=False)
        layer = TransformerLayer(config, spec.submodules).cuda()
        assert layer.is_moe_layer
        # The production scope: attention stays eager, capture takes hidden_states (+ mask).
        layer.config.cuda_graph_scope = [CudaGraphScope.moe_router, CudaGraphScope.moe_preprocess]

        assert "padding_mask" not in layer.get_layer_static_inputs(seq_len, mbs)

        layer.mlp.route(
            torch.randn(seq_len, mbs, hidden, device="cuda"),
            torch.zeros(mbs, seq_len, dtype=torch.bool, device="cuda"),
        )
        static = layer.get_layer_static_inputs(seq_len, mbs)
        assert static["padding_mask"].dtype == torch.bool
        assert static["padding_mask"].shape == (mbs, seq_len)
        assert static["padding_mask"].shape == tuple(reversed(static["hidden_states"].shape[:2]))

        # Under sequence parallelism hidden_states arrive sequence-sharded and so does the
        # live mask (GPTModel._preprocess scatters it), so the placeholder must shrink too.
        layer.config.sequence_parallel = True
        layer.config.tensor_model_parallel_size = 2
        static = layer.get_layer_static_inputs(seq_len, mbs)
        assert static["padding_mask"].shape == (mbs, seq_len // 2)
        assert static["padding_mask"].shape == tuple(reversed(static["hidden_states"].shape[:2]))
    finally:
        unset_num_microbatches_calculator()
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_expert_bias_ignores_padding_tokens():
    """Expert-bias counts exclude padded rows for any token/expert dimensions.

    Single rank on purpose: TestAuxLossFreeTop2Router's fixture asks for 8 expert-parallel
    ranks, so a test placed there errors at setup in a one-GPU session and never runs.
    """
    Utils.initialize_model_parallel(1, 1)
    init_num_microbatches_calculator(
        rank=0, rampup_batch_size=None, global_batch_size=2, micro_batch_size=2,
        data_parallel_size=1,
    )
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    try:
        num_moe_experts = 8
        config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4,
            num_moe_experts=num_moe_experts, use_cpu_initialization=True,
            moe_router_load_balancing_type="none", moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True, moe_router_bias_update_rate=0.1,
            moe_router_topk=2, add_bias_linear=False,
        )
        submodules = get_gpt_layer_local_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        router = cast(Router, MoELayer(config, submodules.mlp.submodules).router).cuda()
        assert router.local_tokens_per_expert is not None

        routing_map = torch.zeros((5, 8), dtype=torch.bool, device="cuda")
        routing_map[0, [0, 1]] = True
        routing_map[1, [2, 3]] = True
        routing_map[2, [0, 4]] = True
        routing_map[3, [5, 6]] = True
        routing_map[4, [1, 7]] = True
        padding_mask = torch.tensor([False, True, False, True, False], device="cuda")

        router.local_tokens_per_expert.zero_()
        router._apply_expert_bias(routing_map, padding_mask)

        expected = torch.tensor(
            [2, 2, 0, 0, 1, 0, 0, 1], device="cuda", dtype=router.local_tokens_per_expert.dtype
        )
        torch.testing.assert_close(router.local_tokens_per_expert, expected)
    finally:
        unset_num_microbatches_calculator()
        Utils.destroy_model_parallel()
