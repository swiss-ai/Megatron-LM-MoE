# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.ssm.kda_cudnn import HAVE_CUDNN_KDA, cudnn_kda
from megatron.core.ssm.kimi_delta_attention import HAVE_KDA, chunk_kda
from megatron.core.transformer import TransformerConfig


def _kda_config(**kwargs):
    config = dict(
        num_layers=1,
        hidden_size=2048,
        num_attention_heads=16,
        experimental_attention_variant="kda",
        linear_attention_freq=[1],
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_attention_safe_output_gate=True,
        linear_attention_backend="cudnn",
    )
    return TransformerConfig(**{**config, **kwargs})


def test_cudnn_backend_rejects_context_parallelism():
    with pytest.raises(AssertionError, match="context parallelism"):
        _kda_config(context_parallel_size=2)


def test_cudnn_backend_rejects_other_head_dims():
    with pytest.raises(AssertionError, match="head dims of 128"):
        _kda_config(linear_key_head_dim=64, linear_value_head_dim=64)


def test_cudnn_backend_requires_safe_gate():
    with pytest.raises(AssertionError, match="safe_output_gate"):
        _kda_config(linear_attention_safe_output_gate=False)


@pytest.mark.skipif(not (HAVE_KDA and HAVE_CUDNN_KDA), reason="Needs FLA KDA and cuDNN KDA.")
@pytest.mark.parametrize("per_channel", [False, True])
def test_cudnn_kda_matches_fla(per_channel):
    lower_bound = -5.0
    torch.manual_seed(0)
    num_heads, head_dim = 4, 128
    cu_seqlens = torch.tensor([0, 300, 1100, 1157, 2048], dtype=torch.int32, device="cuda")
    num_tokens = int(cu_seqlens[-1])
    shape = (num_tokens, num_heads, head_dim)
    q, k, v, alpha = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    beta = torch.randn(num_tokens, num_heads, device="cuda").sigmoid()
    A_log = torch.empty(num_heads * (head_dim if per_channel else 1), device="cuda").uniform_(1, 16).log()
    dt_bias = torch.empty(num_heads * head_dim, device="cuda").uniform_(1e-3, 1e-1).expm1().log()
    do = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    def run(backend):
        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, alpha, beta, A_log, dt_bias)]
        q_, k_, v_, alpha_, beta_, A_log_, dt_bias_ = leaves
        if backend == "cudnn":
            o, _ = cudnn_kda(
                q_, k_, v_, alpha_, beta_, A_log_, dt_bias_, cu_seqlens, lower_bound=lower_bound
            )
        else:
            o, _ = chunk_kda(
                q_[None], k_[None], v_[None], alpha_[None], beta_[None],
                A_log=A_log_, dt_bias=dt_bias_, cu_seqlens=cu_seqlens.long(),
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                safe_gate=True, lower_bound=lower_bound,
            )
            o = o[0]
        o.backward(do)
        return [o] + [t.grad for t in leaves]

    for name, x, y in zip(
        ("o", "dq", "dk", "dv", "dalpha", "dbeta", "dA_log", "ddt_bias"), run("cudnn"), run("fla")
    ):
        rel = (x.float() - y.float()).norm() / y.float().norm()
        assert rel < 2e-2, f"{name}: relative error {rel:.2e}"
