"""CUDA/Transformer Engine integration tests for adjacent expert sharing."""
import io

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def test_tied_expert_gradients_and_resume():
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=4, hidden_size=64, num_attention_heads=4,
            num_moe_experts=4, moe_layer_freq=[0, 1, 1, 1],
            moe_router_topk=2, moe_ffn_hidden_size=32, moe_latent_size=32,
            moe_shared_expert_intermediate_size=32,
            moe_tie_adjacent_experts=True, moe_router_load_balancing_type='none',
            hidden_dropout=0, attention_dropout=0, gradient_accumulation_fusion=False,
        )
        def build():
            return TransformerBlock(config, get_gpt_decoder_block_spec(config, True)).cuda()
        block = build()
        first, second, tail = [layer.mlp for layer in block.layers[1:]]
        assert first.experts is second.experts
        assert tail.experts is not first.experts
        assert [m.config.num_moe_experts for m in (first, second, tail)] == [8, 8, 4]
        assert first.router is not second.router
        assert first.shared_experts is not second.shared_experts
        assert first.fc1_latent_proj is not second.fc1_latent_proj
        weights = list(first.experts.parameters())
        ids = [id(p) for p in block.parameters()]
        assert all(ids.count(id(p)) == 1 for p in weights)

        # The shared Parameter receives the sum of contributions from each layer.
        x = torch.randn(8, 2, 64, device='cuda')
        a = first(x)[0].square().sum()
        b = second(x)[0].square().sum()
        ga = torch.autograd.grad(a, weights, retain_graph=True, allow_unused=True)
        gb = torch.autograd.grad(b, weights, retain_graph=True, allow_unused=True)
        (a + b).backward()
        for p, g1, g2 in zip(weights, ga, gb):
            if g1 is not None and g2 is not None:
                torch.testing.assert_close(p.grad, g1 + g2)
        block.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)
        mask = torch.triu(torch.ones(1, 1, 8, 8, device='cuda', dtype=torch.bool), 1)
        out = block(x, mask)
        out.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in block.parameters() if p.grad is not None)
        optimizer.step()
        block.sharded_state_dict()  # Exercise heterogeneous physical-layer shard construction.

        checkpoint = io.BytesIO()
        torch.save({'model': block.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=True)
        restored = build()
        restored.load_state_dict(saved['model'], strict=True)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_optimizer.load_state_dict(saved['optimizer'])
        assert restored.layers[1].mlp.experts is restored.layers[2].mlp.experts
        torch.testing.assert_close(block(x, mask), restored(x, mask))
        assert len(optimizer.state) == len(restored_optimizer.state)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize('kwargs', [
    {'smelt_loop_layers': 2}, {'pipeline_model_parallel_size': 2},
    {'recompute_granularity': 'full'}, {'cuda_graph_impl': 'local'},
])
def test_unsupported_modes(kwargs):
    with pytest.raises(ValueError, match='Adjacent expert tying'):
        TransformerConfig(num_layers=4, hidden_size=64, num_attention_heads=4,
                          num_moe_experts=4, moe_tie_adjacent_experts=True, **kwargs)
