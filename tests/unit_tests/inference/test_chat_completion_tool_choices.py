# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from quart import Quart

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints import (
    chat_completions,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_choice,parallel,max_tokens,count,content,finish",
    [
        ("auto", True, 8, 2, "answer ", "tool_calls"),
        ("auto", False, 8, 1, "answer ", "tool_calls"),
        ("required", True, 8, 2, "", "tool_calls"),
        ({"type": "function", "function": {"name": "search"}}, True, 8, 2, "", "stop"),
        ("none", True, 8, 0, None, "stop"),
        ("auto", True, 2, 2, "answer ", "length"),
    ],
)
async def test_chat_tool_choice_response_contract(
    tool_choice, parallel, max_tokens, count, content, finish
):
    """Exercise the real endpoint and parser with a completed unit-test inference result."""
    generated_text = (
        'answer <tool_call><function=search><parameter=query>hello</parameter>'
        '</function></tool_call><tool_call><function=lookup></function></tool_call>'
    )
    tokenizer = SimpleNamespace(
        eod=-1,
        eos_id=0,
        tokenize=lambda text: [1, 2],
        detokenize=lambda tokens, **kwargs: generated_text if tokens == [3, 4] else "prompt",
    )
    submitted = []

    def add_request(prompt, sampling_params):
        submitted.append((prompt, sampling_params))
        future = asyncio.get_running_loop().create_future()
        future.set_result(
            {
                "uid": "completion-unit-id",
                "status": "COMPLETED",
                "prompt_tokens": [1, 2],
                "prompt_length": 2,
                "generated_tokens": [3, 4],
                "generated_log_probs": [],
                "routing_indices": None,
                "sampling_params": asdict(sampling_params),
            }
        )
        return 0, future

    app = Quart(__name__)
    app.register_blueprint(chat_completions.bp)
    app.config.update(
        client=SimpleNamespace(add_request_with_id=add_request),
        tokenizer=tokenizer,
        parsers=["qwen3-coder-tool"],
        verbose=False,
    )
    tools = [
        {"type": "function", "function": {"name": name, "parameters": {}}}
        for name in ("search", "lookup")
    ]
    response = await app.test_client().post(
        '/v1/chat/completions',
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel,
            "max_tokens": max_tokens,
        },
    )
    assert response.status_code == 200, await response.get_data(as_text=True)
    result = await response.get_json()
    assert len(submitted) == 1
    assert result["id"] == "completion-unit-id"
    choice = result["choices"][0]
    assert len(choice["message"].get("tool_calls", [])) == count
    assert choice["message"]["content"] == (generated_text if content is None else content)
    assert choice["finish_reason"] == finish
