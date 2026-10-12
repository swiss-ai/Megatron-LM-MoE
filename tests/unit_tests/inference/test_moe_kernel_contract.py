# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPU contract regressions for the completed inference MoE kernel ports."""

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.internal,
]


def _routes(max_tokens, topk=2):
    # Unique experts within each token's top-k; rows are deliberately non-uniform.
    return torch.stack(
        [
            torch.arange(max_tokens, device="cuda") % 4,
            (torch.arange(max_tokens, device="cuda") + 1) % 4,
        ],
        dim=1,
    )[:, :topk]


def _expected_rows(route, valid, alignment, num_experts=4):
    result = []
    for expert in range(num_experts):
        tokens = torch.where(route[:valid] == expert)
        expert_tokens = tokens[0]
        choices = tokens[1]
        result.extend((int(token), int(choice)) for token, choice in zip(expert_tokens, choices))
        if expert_tokens.numel():
            result.extend([(-1, -1)] * ((-expert_tokens.numel()) % alignment))
    return result


@pytest.mark.parametrize("fused_quant", [False, True])
def test_count_and_permute_ignore_garbage_tail(fused_quant):
    from megatron.core.inference.moe.permute import (
        compute_local_tokens_per_expert,
        permute_and_quantize_mxfp8,
        permute_tokens,
    )

    max_tokens, hidden_dim, alignment = 5, 64, 4
    hidden = torch.arange(max_tokens * hidden_dim, device="cuda", dtype=torch.float32)
    hidden = (hidden.reshape(max_tokens, hidden_dim) / 100).to(torch.bfloat16)
    probs = torch.tensor([[0.2, 0.8]] * max_tokens, device="cuda")
    routes = _routes(max_tokens)
    valid_routes = routes.clone()
    routes[3:] = 0  # local-expert garbage must not contribute outside valid_tokens
    alignment = 128 if fused_quant else alignment
    valid = torch.zeros((), dtype=torch.int32, device="cuda")
    permute = permute_and_quantize_mxfp8 if fused_quant else permute_tokens

    for prefix in (0, 1, 3, max_tokens):
        if prefix == max_tokens:
            routes.copy_(valid_routes)
        valid.fill_(prefix)
        counts = compute_local_tokens_per_expert(routes, 0, 4, valid)
        expected_counts = torch.bincount(routes[:prefix].reshape(-1), minlength=4)
        torch.testing.assert_close(counts, expected_counts.to(torch.int32))
        result, out_probs, src, offsets = permute(
            hidden, probs, routes, 0, 4, valid, alignment=alignment
        )
        used = int(offsets[-1].item())
        expected_rows = _expected_rows(routes, prefix, alignment)
        assert sorted(src[:used].tolist()) == sorted(token for token, _ in expected_rows)
        start = 0
        for expert in range(4):
            end = int(offsets[expert].item())
            for row in range(start, end):
                token = int(src[row].item())
                if token >= 0:
                    choice = int((routes[token] == expert).nonzero()[0].item())
                    torch.testing.assert_close(out_probs[row], probs[token, choice])
                    if fused_quant:
                        assert torch.count_nonzero(result.data[row].view(torch.uint8)).item() > 0
                    else:
                        torch.testing.assert_close(result[row], hidden[token])
            start = end
        assert offsets.dtype == torch.int32


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_unpermute_fp32_contract_and_preallocated_output(out_dtype):
    from megatron.core.inference.moe.permute import permute_tokens, unpermute_tokens

    tokens, width = 5, 8
    hidden = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16)
    routes = _routes(tokens)
    probs = torch.tensor([[0.25, 0.75]] * tokens, device="cuda")
    valid = torch.tensor(3, dtype=torch.int32, device="cuda")
    _, pp, src, offsets = permute_tokens(hidden, probs, routes, 0, 4, valid, alignment=4)
    used = int(offsets[-1].item())
    expert = torch.randn(src.numel(), width, device="cuda", dtype=torch.bfloat16)
    # These rows must be skipped before their garbage map, probs, or data are read.
    expert[used:] = float("nan")
    pp[used:] = float("nan")
    src[used:] = 0
    out = torch.full((tokens, width), 123.0, device="cuda", dtype=out_dtype)
    identity, storage = id(out), out.data_ptr()
    got = unpermute_tokens(expert, pp, src, tokens, offsets[-1:], valid, out=out)
    assert id(got) == identity and got.data_ptr() == storage and got.dtype == out_dtype
    local = unpermute_tokens(expert, pp, src, tokens, offsets[-1:], valid)
    assert local.dtype == torch.float32
    ref = torch.zeros_like(out, dtype=torch.float32)
    for row in range(used):
        token = int(src[row].item())
        if token >= 0:
            ref[token] += expert[row].float() * pp[row]
    torch.testing.assert_close(local[:3], ref[:3], atol=1e-6, rtol=1e-6)
    tolerance = (
        dict(atol=1e-6, rtol=1e-6) if out_dtype == torch.float32 else dict(atol=0.015, rtol=0.02)
    )
    torch.testing.assert_close(got[:3].float(), ref[:3], **tolerance)
    assert torch.all(got[3:] == 123.0)


def test_cuda_graph_replay_varying_prefix_permute_activation_unpermute():
    from megatron.core.inference.moe.activations import padded_squared_relu
    from megatron.core.inference.moe.permute import permute_tokens, unpermute_tokens

    max_tokens, width = 5, 8
    hidden = torch.randn(max_tokens, width, device="cuda", dtype=torch.bfloat16)
    routes = _routes(max_tokens)
    probs = torch.tensor([[0.4, 0.6]] * max_tokens, device="cuda")
    valid = torch.zeros((), dtype=torch.int32, device="cuda")
    out = torch.full((max_tokens, width), 77.0, device="cuda", dtype=torch.float32)
    output_ref = torch.zeros_like(out)

    def run():
        perm, pp, src, offsets = permute_tokens(hidden, probs, routes, 0, 4, valid, alignment=4)
        activated = padded_squared_relu(perm, src, offsets[-1:])
        return unpermute_tokens(activated, pp, src, max_tokens, offsets[-1:], valid, out=out)

    # Compile and initialize kernels outside capture on a side stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    assert captured is out
    storage = out.data_ptr()
    for prefix in (3, 1, max_tokens, 0):
        out.fill_(77.0)
        valid.fill_(prefix)
        hidden.normal_()
        graph.replay()
        assert out.data_ptr() == storage
        ref = torch.zeros_like(output_ref)
        for token in range(prefix):
            for choice in range(2):
                value = torch.relu(hidden[token].float()).square().to(torch.bfloat16).float()
                ref[token] += value * probs[token, choice]
        torch.testing.assert_close(out[:prefix], ref[:prefix], atol=0.08, rtol=0.02)
        assert torch.all(out[prefix:] == 77.0)


def test_mcore_fused_moe_bf16_grouped_mm_matches_dense_reference():
    from megatron.core.inference.moe import fused_moe

    if not fused_moe.HAVE_GROUPED_MM:
        pytest.skip("torch.nn.functional.grouped_mm is unavailable")

    from megatron.core.inference.moe.fused_moe import ActivationType, mcore_fused_moe

    tokens, experts, hidden_dim, ffn_dim = 3, 4, 16, 16
    hidden = torch.randn(tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
    routes = _routes(tokens)
    probs = torch.tensor([[0.3, 0.7]] * tokens, device="cuda")
    fc1 = torch.randn(experts, ffn_dim, hidden_dim, device="cuda", dtype=torch.bfloat16)
    fc2 = torch.randn(experts, hidden_dim, ffn_dim, device="cuda", dtype=torch.bfloat16)
    valid = torch.tensor(tokens, dtype=torch.int32, device="cuda")
    actual = mcore_fused_moe(
        hidden, probs, fc1, fc2, ActivationType.SQUARED_RELU, experts, 0, valid, routes
    )
    expected = torch.zeros(tokens, hidden_dim, device="cuda", dtype=torch.float32)
    for token in range(tokens):
        for choice in range(2):
            expert = int(routes[token, choice].item())
            first = (hidden[token] @ fc1[expert].T).to(torch.bfloat16)
            activated = torch.relu(first.float()).square().to(torch.bfloat16)
            projected = (activated @ fc2[expert].T).to(torch.bfloat16)
            expected[token] += projected.float() * probs[token, choice]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual[:tokens], expected, atol=0.15, rtol=0.03)
