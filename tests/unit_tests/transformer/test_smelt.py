"""GPU integration coverage for weight-tied forward/backward and checkpoint layout."""
import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def test_smelt_tied_forward_backward():
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=4, hidden_size=64, num_attention_heads=4,
            smelt_loop_layers=2, hidden_dropout=0, attention_dropout=0,
            gradient_accumulation_fusion=False,
        )
        block = TransformerBlock(config, get_gpt_layer_with_transformer_engine_spec()).cuda()
        assert len(block.layers) == 4
        assert [l.residual_output_scale for l in block.layers] == [None, 0.5, 0.5, None]
        keys = set(block.state_dict())
        assert not any(k.startswith('layers.4.') for k in keys)
        x = torch.randn(8, 2, 64, device='cuda', requires_grad=True)
        mask = torch.triu(torch.ones(1, 1, 8, 8, device='cuda', dtype=torch.bool), diagonal=1)
        out = block(x, mask)
        out.square().sum().backward()
        grad = x.grad.clone()
        grads = {n: p.grad.clone() for n, p in block.named_parameters() if p.grad is not None}
        block.zero_grad(set_to_none=True)
        reference = x.detach().clone().requires_grad_(True)
        y = reference
        for i in (0, 1, 2, 1, 2, 3):
            y, _ = block.layers[i](hidden_states=y, attention_mask=mask)
        y = block.final_layernorm(y)
        y.square().sum().backward()
        torch.testing.assert_close(out, y)
        torch.testing.assert_close(grad, reference.grad)
        for n, p in block.named_parameters():
            if n in grads:
                torch.testing.assert_close(grads[n], p.grad)
        block.load_state_dict(block.state_dict(), strict=True)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize('kwargs', [
    {'pipeline_model_parallel_size': 2},
    {'recompute_granularity': 'full'},
    {'linear_attention_carry_state': True},
    {'cuda_graph_impl': 'local'},
])
def test_smelt_unsupported_modes(kwargs):
    with pytest.raises(ValueError, match='SMELT'):
        TransformerConfig(num_layers=4, hidden_size=64, num_attention_heads=4,
                          smelt_loop_layers=2, **kwargs)
