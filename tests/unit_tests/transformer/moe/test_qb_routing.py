# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import replace
from typing import cast

import pytest
import torch

import megatron.core.transformer.moe.router as router_module
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker,
    compute_qb_histogram,
    get_moe_layer_wise_logging_tracker,
    qb_dual_update,
    recover_qb_beta_from_histogram,
)
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class TestQBDualUpdate:
    """Pure-tensor tests for the quantile-balancing dual update (CPU, no distributed)."""

    @pytest.mark.internal
    @pytest.mark.parametrize("m,n,k", [(64, 8, 2), (40, 8, 1), (10, 4, 1)])
    def test_column_quantile_contract(self, m, n, k):
        """qb_beta_local is the (col_target+1)-th largest of (scores - alpha) per expert."""
        torch.manual_seed(1)
        scores = torch.randn(m, n)
        beta = torch.zeros(n)

        _, beta_local = qb_dual_update(scores, k, beta, update_beta=True)

        alpha = (scores - beta).topk(k + 1, dim=1).values[:, -1:]
        adjusted = scores - alpha
        col_target = m * k // n
        expected = adjusted.sort(dim=0, descending=True).values[col_target]
        torch.testing.assert_close(beta_local, expected)

    @pytest.mark.internal
    def test_single_step_reduces_imbalance(self):
        """One bias update corrects a systematic per-expert preference."""
        torch.manual_seed(2)
        m, n, k = 512, 8, 2
        scores = torch.randn(m, n)
        # Experts 0 and 1 intrinsically attractive -> over-selected at zero bias.
        scores[:, 0] += 4.0
        scores[:, 1] += 2.0
        col_target = m * k // n

        idx0, beta_local = qb_dual_update(scores, k, torch.zeros(n), update_beta=True)
        counts0 = torch.bincount(idx0.flatten(), minlength=n)

        beta1 = beta_local - beta_local.mean()  # mirror the caller's re-centering
        idx1 = (scores - beta1).topk(k, dim=1).indices
        counts1 = torch.bincount(idx1.flatten(), minlength=n)

        imbalance0 = (counts0 - col_target).abs().sum()
        imbalance1 = (counts1 - col_target).abs().sum()
        assert imbalance1 < imbalance0


class TestQBHistogram:
    """Pure-tensor tests for the pooled histogram estimator."""

    @pytest.mark.internal
    def test_accumulation_and_recovered_quantile(self):
        torch.manual_seed(4)
        num_tokens, num_experts, topk, num_bins = 4096, 8, 2, 1000
        scores = torch.rand(num_tokens, num_experts)
        beta = torch.linspace(-0.1, 0.1, num_experts)
        alpha = (scores - beta).topk(topk + 1, dim=1).values[:, -1]
        full_histogram = compute_qb_histogram(scores, alpha, beta, num_bins)
        histogram = compute_qb_histogram(scores[:1000], alpha[:1000], beta, num_bins)
        histogram += compute_qb_histogram(scores[1000:], alpha[1000:], beta, num_bins)

        assert torch.equal(histogram, full_histogram)
        assert torch.equal(
            histogram.sum(dim=1),
            torch.full((num_experts,), num_tokens, dtype=torch.int64),
        )
        estimated_beta = recover_qb_beta_from_histogram(histogram, beta, topk)
        required_bias = alpha.unsqueeze(1) - scores
        exact_beta = -torch.quantile(required_bias, topk / num_experts, dim=0)
        bin_width = (-beta.min() + beta.max() + 2.0) / num_bins

        assert torch.all((estimated_beta - exact_beta).abs() <= bin_width + 1e-6)


@pytest.mark.internal
@pytest.mark.parametrize(
    "load_balancing_type",
    [
        ["quantile_balancing", "aux_loss"],
        ["quantile_balancing", "global_aux_loss"],
        ["quantile_balancing", "none"],
    ],
)
def test_qb_only_combines_with_seq_aux_loss(load_balancing_type):
    with pytest.raises(ValueError, match="quantile_balancing can only be combined"):
        TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=8,
            use_cpu_initialization=True,
            moe_router_load_balancing_type=load_balancing_type,
            moe_aux_loss_coeff=[0, 0],
        )


class TestQuantileBalancingRouter:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.num_moe_experts = 8
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=self.num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="quantile_balancing",
            moe_router_quantile_balancing_method="average",
            moe_router_score_function="softmax",
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        self.submodules = get_submodules(
            get_gpt_layer_local_submodules(
                num_experts=self.num_moe_experts, moe_grouped_gemm=False
            ).mlp
        )
        assert isinstance(self.submodules, MoESubmodules)
        self.moe_layer = MoELayer(self.transformer_config, self.submodules)
        self.router = cast(Router, self.moe_layer.router)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_qb_buffers_registered(self):
        assert self.router.qb_beta is not None
        assert self.router.qb_beta.shape == (self.num_moe_experts,)
        assert self.router.qb_beta.dtype == torch.float32
        assert self.router.qb_beta_accum is not None
        assert self.router.qb_beta_count is not None
        assert self.router.qb_histogram is None

    @pytest.mark.internal
    def test_qb_histogram_buffer_registered_only_for_histogram_method(self):
        config = replace(
            self.transformer_config,
            moe_router_quantile_balancing_method="histogram",
            moe_router_quantile_balancing_num_bins=32,
            moe_router_score_function="sigmoid",
        )
        router = cast(Router, MoELayer(config, self.submodules).router)

        assert router.qb_histogram is not None
        assert router.qb_histogram.shape == (self.num_moe_experts, 32)
        assert router.qb_histogram.dtype == torch.int64
        assert router.qb_beta_accum is None
        assert router.qb_beta_count is None

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_frozen_qb_routes_with_loaded_beta_without_collecting_updates(self):
        config = replace(self.transformer_config, moe_router_quantile_balancing_freeze=True)
        router = cast(Router, MoELayer(config, self.submodules).router).cuda()
        router.train()
        with torch.no_grad():
            router.qb_beta.zero_()
            router.qb_beta[:2] = 10.0

        logits = torch.zeros((2, self.num_moe_experts), device="cuda", dtype=torch.bfloat16)
        _, routing_map = router.quantile_balancing(logits)

        assert not routing_map[:, :2].any()
        torch.testing.assert_close(router.qb_beta_accum, torch.zeros_like(router.qb_beta_accum))
        assert router.qb_beta_count.item() == 0

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_qb_seq_aux_loss_uses_qb_dispatch_and_raw_aux_map(self, monkeypatch):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=self.num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type=["quantile_balancing", "seq_aux_loss"],
            moe_router_score_function="softmax",
            moe_router_topk=2,
            moe_aux_loss_coeff=[0, 0.5],
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        router = cast(Router, MoELayer(config, self.submodules).router).cuda()
        router.set_layer_number(1)
        clear_aux_losses_tracker()

        raw_topk_experts = [0, 1]
        qb_experts = [6, 7]

        captured = {}
        original_compute_aux = router_module.compute_routing_scores_for_aux_loss

        def capture_compute_aux(*args, **kwargs):
            aux_routing_map, aux_scores = original_compute_aux(*args, **kwargs)
            captured["routing_map"] = aux_routing_map.detach().clone()
            return aux_routing_map, aux_scores

        monkeypatch.setattr(
            router_module, "compute_routing_scores_for_aux_loss", capture_compute_aux
        )

        logits = torch.zeros((4, 2, self.num_moe_experts), device="cuda", dtype=torch.bfloat16)
        logits[..., raw_topk_experts[0]] = 6.0
        logits[..., raw_topk_experts[1]] = 5.0
        logits.requires_grad_()
        with torch.no_grad():
            router.qb_beta.zero_()
            router.qb_beta[qb_experts] = -10.0

        probs, routing_map = router.routing(logits)

        expected_qb_routing_map = torch.zeros_like(routing_map)
        expected_qb_routing_map[:, qb_experts] = True
        expected_aux_routing_map = torch.zeros_like(routing_map)
        expected_aux_routing_map[:, raw_topk_experts] = True

        assert torch.equal(routing_map, expected_qb_routing_map)
        assert torch.equal(captured["routing_map"], expected_aux_routing_map)
        assert "seq_load_balancing_loss" in get_moe_layer_wise_logging_tracker()

        probs.sum().mul(0).backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_qb_beta_update_ignores_padding_logits(self, monkeypatch):
        self.router = self.router.cuda()
        self.router.train()

        seq_len = 4
        batch_size = 2
        logits = torch.arange(
            seq_len * batch_size * self.num_moe_experts, dtype=torch.float32, device="cuda"
        ).reshape(seq_len, batch_size, self.num_moe_experts)
        padding_mask = torch.tensor(
            [
                [False, False],
                [True, False],
                [True, True],
                [False, True],
            ],
            dtype=torch.bool,
            device="cuda",
        )
        expected_valid_logits = logits.reshape(-1, self.num_moe_experts)[
            ~padding_mask.reshape(-1)
        ]
        expected_valid_scores = torch.softmax(expected_valid_logits, dim=-1)

        calls = []

        def fake_qb_dual_update(scores, topk, beta, update_beta=True):
            calls.append(
                {
                    "scores": scores.detach().clone(),
                    "update_beta": update_beta,
                }
            )
            indices = torch.arange(topk, device=scores.device).expand(scores.shape[0], -1)
            return indices, scores.mean(dim=0)

        monkeypatch.setattr(router_module, "qb_dual_update", fake_qb_dual_update)

        _, routing_map = self.router.routing(logits.to(torch.bfloat16), padding_mask=padding_mask)

        assert len(calls) == 1
        assert calls[0]["update_beta"] is True
        torch.testing.assert_close(calls[0]["scores"], expected_valid_scores)
        torch.testing.assert_close(self.router.qb_beta_accum, expected_valid_scores.mean(dim=0))
        assert self.router.qb_beta_count.item() == 1
        assert routing_map.shape == (seq_len * batch_size, self.num_moe_experts)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_legacy_average_qb_uses_raw_logits(self, monkeypatch):
        config = replace(
            self.transformer_config,
            moe_router_quantile_balancing_method="legacy_average",
        )
        router = cast(Router, MoELayer(config, self.submodules).router).cuda()
        router.train()

        logits = torch.tensor(
            [
                [3.0, 1.0, 0.9, -1.0, -2.0, -3.0, -4.0, -5.0],
                [1.0, 3.0, 0.9, -1.0, -2.0, -3.0, -4.0, -5.0],
            ],
            device="cuda",
        )
        with torch.no_grad():
            router.qb_beta.zero_()
            router.qb_beta[0] = 1.0
        expected_qb_scores = logits.to(torch.bfloat16).float()
        expected_indices = (expected_qb_scores - router.qb_beta).topk(router.topk, dim=1).indices
        expected_routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(
            1, expected_indices, True
        )
        captured = {}

        def fake_qb_dual_update(scores, topk, beta, update_beta=True):
            captured["scores"] = scores.detach().clone()
            return expected_indices, scores.mean(dim=0)

        monkeypatch.setattr(router_module, "qb_dual_update", fake_qb_dual_update)

        _, routing_map = router.quantile_balancing(logits.to(torch.bfloat16))

        torch.testing.assert_close(captured["scores"], expected_qb_scores)
        assert torch.equal(routing_map, expected_routing_map)
        torch.testing.assert_close(router.qb_beta_accum, expected_qb_scores.mean(dim=0))
        assert router.qb_beta_count.item() == 1

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_qb_histogram_accumulates_valid_tokens_without_forward_collective(
        self, monkeypatch
    ):
        config = replace(
            self.transformer_config,
            moe_router_quantile_balancing_method="histogram",
            moe_router_quantile_balancing_num_bins=16,
            moe_router_score_function="sigmoid",
        )
        router = cast(Router, MoELayer(config, self.submodules).router).cuda()
        router.train()

        logits = torch.arange(
            6 * self.num_moe_experts, dtype=torch.float32, device="cuda"
        ).reshape(6, 1, self.num_moe_experts)
        padding_mask = torch.tensor(
            [[False], [True], [False], [False], [True], [False]],
            dtype=torch.bool,
            device="cuda",
        )
        # Keep the row shape static for CUDA graph capture. The histogram helper
        # excludes padded rows using the mask without compacting at this call site.
        all_scores = torch.sigmoid(logits.reshape(-1, self.num_moe_experts))
        expected_alpha = (
            (all_scores - router.qb_beta).topk(router.topk + 1, dim=1).values[:, -1]
        )
        captured = {}
        expected_histogram = torch.arange(
            self.num_moe_experts * 16, dtype=torch.int64, device="cuda"
        ).reshape(self.num_moe_experts, 16)

        def fake_compute_qb_histogram(scores, alpha, beta, num_bins, padding_mask=None):
            captured["scores"] = scores.detach().clone()
            captured["alpha"] = alpha.detach().clone()
            captured["beta"] = beta.detach().clone()
            captured["num_bins"] = num_bins
            captured["padding_mask"] = (
                None if padding_mask is None else padding_mask.detach().clone()
            )
            return expected_histogram

        def fail_all_gather(*args, **kwargs):
            raise AssertionError("histogram QB must not all-gather during forward")

        monkeypatch.setattr(
            router_module, "compute_qb_histogram", fake_compute_qb_histogram
        )
        monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fail_all_gather)

        for _ in range(2):
            router.quantile_balancing(
                logits.to(torch.bfloat16).reshape(-1, self.num_moe_experts),
                padding_mask=padding_mask.reshape(-1),
            )

        torch.testing.assert_close(captured["scores"], all_scores)
        torch.testing.assert_close(captured["alpha"], expected_alpha)
        torch.testing.assert_close(captured["beta"], router.qb_beta)
        assert captured["num_bins"] == 16
        assert torch.equal(captured["padding_mask"], padding_mask.reshape(-1))
        assert torch.equal(router.qb_histogram, 2 * expected_histogram)

    @pytest.mark.internal
    def test_non_qb_router_has_no_qb_buffers(self):
        config = replace(
            self.transformer_config,
            moe_router_load_balancing_type="aux_loss",
        )
        router = MoELayer(config, self.submodules).router
        assert router.qb_beta is None
        assert router.qb_beta_accum is None
        assert router.qb_beta_count is None
        assert router.qb_histogram is None

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        ("score_function", "moe_router_pre_softmax"),
        [("softmax", True), ("softmax", False), ("sigmoid", False)],
    )
    def test_qb_router_forward(self, score_function, moe_router_pre_softmax):
        self.router = self.router.cuda()
        self.router.config.moe_router_score_function = score_function
        self.router.score_function = score_function
        self.router.config.moe_router_pre_softmax = moe_router_pre_softmax

        num_tokens = 32 * 2
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size)).cuda().bfloat16()
        with torch.no_grad():
            probs, routing_map = self.router(hidden_states)

        assert probs.shape == (num_tokens, self.num_moe_experts)
        assert routing_map.shape == (num_tokens, self.num_moe_experts)
        # Each token selects exactly topk distinct experts.
        assert routing_map.sum().item() == num_tokens * self.router.topk

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_qb_beta_accumulates_in_training(self):
        self.router = self.router.cuda()
        self.router.train()
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size)).cuda().bfloat16()

        assert self.router.qb_beta_count.item() == 0
        self.router(hidden_states)
        assert self.router.qb_beta_count.item() == 1
        assert self.router.qb_beta_accum.abs().sum().item() > 0
        self.router(hidden_states)
        assert self.router.qb_beta_count.item() == 2

        # No accumulation outside the training path (eval / recompute).
        accum_before = self.router.qb_beta_accum.clone()
        with torch.no_grad():
            self.router(hidden_states)
        assert self.router.qb_beta_count.item() == 2
        torch.testing.assert_close(self.router.qb_beta_accum, accum_before)
