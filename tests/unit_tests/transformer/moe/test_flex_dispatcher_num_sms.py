# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
from unittest.mock import Mock

import pytest
import torch

from megatron.core.transformer.moe import token_dispatcher
from megatron.core.transformer.moe.token_dispatcher import _DeepepManager, _HybridEPManager
from megatron.core.transformer.transformer_config import TransformerConfig


def make_config(moe_flex_dispatcher_num_sms=None):
    return TransformerConfig(
        hidden_size=16,
        num_attention_heads=8,
        num_layers=1,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_flex_dispatcher_num_sms=moe_flex_dispatcher_num_sms,
    )


@pytest.mark.parametrize(
    ("override", "expected"),
    [(None, 16), (0, 0), (7, 7)],
)
def test_hybrid_ep_uses_common_sm_override(monkeypatch, override, expected):
    config = make_config(override)
    dispatch = Mock(
        return_value=(
            torch.ones(1, 1),
            torch.ones(1),
            None,
            torch.ones(1, dtype=torch.long),
            object(),
        )
    )
    monkeypatch.setattr(token_dispatcher, "hybrid_ep_dispatch", dispatch)

    manager = _HybridEPManager(group=None, num_local_experts=1, num_experts=1, config=config)
    manager.routing_map = torch.ones(1, 1, dtype=torch.bool)
    manager.token_probs = torch.ones(1, 1)
    manager.dispatch(torch.ones(1, 1))

    assert dispatch.call_args.kwargs["num_sms_dispatch_api"] == expected
    assert dispatch.call_args.kwargs["num_sms_combine_api"] == expected


@pytest.mark.parametrize(
    ("override", "expected"),
    [(None, 20), (0, 0), (7, 7)],
)
def test_deepep_uses_common_sm_override(monkeypatch, override, expected):
    config = make_config(override)
    monkeypatch.setattr(token_dispatcher, "fused_dispatch", object())
    set_num_sms = Mock()
    monkeypatch.setattr(token_dispatcher, "set_deepep_num_sms", set_num_sms)

    _DeepepManager(
        group=None,
        num_local_experts=1,
        router_topk=2,
        num_experts=1,
        config=config,
    )

    set_num_sms.assert_called_once_with(expected)


def test_flex_dispatcher_sm_override_is_constructible_and_serializable():
    config = make_config(7)
    serialized = dataclasses.asdict(config)

    assert config.moe_flex_dispatcher_num_sms == 7
    assert serialized["moe_flex_dispatcher_num_sms"] == 7
    assert dataclasses.replace(config, moe_flex_dispatcher_num_sms=None).moe_flex_dispatcher_num_sms is None
