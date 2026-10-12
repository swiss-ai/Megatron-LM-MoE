# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
from unittest.mock import MagicMock, patch

import msgpack
import pytest
import zmq

from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.sampling_params import SamplingParams

pytestmark = pytest.mark.asyncio


def _make_client(deserialize: bool = False):
    """Build an InferenceClient with a mocked zmq Context/Socket."""
    fake_socket = MagicMock(name="zmq_socket")
    fake_context = MagicMock(name="zmq_context")
    fake_context.socket.return_value = fake_socket
    with patch("megatron.core.inference.inference_client.zmq.Context", return_value=fake_context):
        client = InferenceClient("tcp://127.0.0.1:5555", deserialize=deserialize)
    return client, fake_context, fake_socket


async def test_inference_client_lifecycle():
    """End-to-end lifecycle of InferenceClient with mocked zmq sockets:
    construct → start (CONNECT handshake) → add_request (SUBMIT_REQUEST) →
    listener delivers ENGINE_REPLY → control signal (pause + set epoch) →
    stop (cancels listener, cancels pending futures, closes socket).

    Per reviewer guidance, the per-step assertions are intentionally bundled
    into one test because the contract is the ordering, not the steps in
    isolation."""
    client, fake_context, fake_socket = _make_client()

    # Construction: DEALER socket connected, HWMs at 0, counters initialized.
    fake_socket.connect.assert_called_once_with("tcp://127.0.0.1:5555")
    opts = {call.args[0]: call.args[1] for call in fake_socket.setsockopt.call_args_list}
    assert opts[zmq.SNDHWM] == 0 and opts[zmq.RCVHWM] == 0
    assert client.next_request_id == 0
    assert client.completion_futures == {}

    # start(): handshake sends CONNECT, expects CONNECT_ACK, spawns listener task.
    # We stage two recv_multipart() replies: the CONNECT_ACK during handshake,
    # and an ENGINE_REPLY for the request we'll add below. Messages arrive as
    # [metadata, body] frames; the body is only present on replies that carry
    # one. Subsequent recvs raise zmq.Again so the listener yields back to the
    # event loop.
    recv_queue = [
        [msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True)],
        [
            msgpack.packb([Headers.ENGINE_REPLY.value, 0], use_bin_type=True),
            msgpack.packb({"foo": "bar"}, use_bin_type=True),
        ],
    ]

    def fake_recv(*args, **kwargs):
        if recv_queue:
            return recv_queue.pop(0)
        raise zmq.Again()

    fake_socket.recv_multipart.side_effect = fake_recv

    client.start()
    assert isinstance(client.listener_task, asyncio.Task)
    sent_connect = fake_socket.send.call_args.args[0]
    assert msgpack.unpackb(sent_connect, raw=False)[0] == Headers.CONNECT.value

    # add_request frames the submission as [metadata, prompt, block_hashes, media]
    # so the coordinator can route it without decoding the prompt or the media.
    fut = client.add_request("hello", SamplingParams(temperature=0.5))
    assert isinstance(fut, asyncio.Future)
    assert client.next_request_id == 1
    assert 0 in client.request_submission_times
    submit_meta, submit_prompt, submit_hashes, submit_media = (
        fake_socket.send_multipart.call_args.args[0]
    )
    submit_payload = msgpack.unpackb(submit_meta, raw=False)
    assert submit_payload[0] == Headers.SUBMIT_REQUEST.value
    assert submit_payload[1] == 0
    assert submit_payload[2]["temperature"] == 0.5
    assert msgpack.unpackb(submit_prompt, raw=False) == "hello"
    # This client was told no block size, so it reports None -- "I did not hash" --
    # and the coordinator hashes on its behalf.
    assert msgpack.unpackb(submit_hashes, raw=False) is None
    # Text-only, so the media frame is present but empty: the frame count is fixed
    # so the coordinator can reject a malformed submission on arity alone.
    assert msgpack.unpackb(submit_media, raw=False) is None

    # Listener delivers the reply: future resolves with payload + injected latency.
    # Submission-time entry is popped on completion.
    result = await asyncio.wait_for(fut, timeout=2.0)
    assert result["foo"] == "bar"
    assert "latency" in result
    assert 0 not in client.request_submission_times
    assert 0 not in client.completion_futures

    # Control helpers send the matching Headers byte (PAUSE used as a representative;
    # the dispatch table is one ctype-style mapping shared across all helpers).
    fake_socket.send.reset_mock()
    client.pause_engines()
    assert msgpack.unpackb(fake_socket.send.call_args.args[0], raw=False)[0] == Headers.PAUSE.value
    client.set_generation_epoch(42)
    epoch_payload = msgpack.unpackb(fake_socket.send.call_args.args[0], raw=False)
    assert epoch_payload[0] == Headers.SET_GENERATION_EPOCH.value
    assert epoch_payload[1] == 42

    # Submit a second request so we have a pending future for stop() to cancel.
    pending = client.add_request("p2", SamplingParams())

    # stop(): cancels listener, cancels pending futures, closes socket + terminates ctx.
    client.stop()
    await asyncio.sleep(0)  # allow cancellation to propagate
    assert client.listener_task.cancelled() or client.listener_task.done()
    assert pending.cancelled()
    assert client.completion_futures == {}
    fake_socket.close.assert_called_once_with(linger=0)
    fake_context.term.assert_called_once_with()


async def test_add_request_accepts_text_only_multimodal_default():
    client, _, fake_socket = _make_client()
    future = client.add_request("hello", SamplingParams(), multi_modal_data=None)
    assert isinstance(future, asyncio.Future)
    assert client.next_request_id == 1
    frames = fake_socket.send_multipart.call_args.args[0]
    assert len(frames) == 4
    assert msgpack.unpackb(frames[3], raw=False) is None
    client.stop()
    assert future.cancelled()


@pytest.mark.parametrize("payload", [{}, {"images": [b"image"]}, {"videos": [b"video"]}])
async def test_add_request_rejects_multimodal_data_before_submission(payload):
    client, _, fake_socket = _make_client()
    with pytest.raises(NotImplementedError, match="multi_modal_data"):
        client.add_request("hello", SamplingParams(), multi_modal_data=payload)
    assert client.next_request_id == 0
    assert client.completion_futures == {}
    assert client.request_submission_times == {}
    fake_socket.send_multipart.assert_not_called()
    client.stop()


async def test_inference_client_connect_handshake_rejects_unexpected_reply():
    """If the coordinator replies with anything other than CONNECT_ACK during
    the handshake, the client raises AssertionError synchronously — this is a
    fatal protocol mismatch, not a recoverable error. Separated from the
    lifecycle test because it short-circuits before any state is established."""
    client, _, fake_socket = _make_client()
    fake_socket.recv_multipart.return_value = [
        msgpack.packb([Headers.STOP.value], use_bin_type=True)
    ]
    with pytest.raises(AssertionError):
        client._connect_with_inference_coordinator()


async def test_add_request_with_kv_handoff_returns_future():
    """Non-streaming handoffs use the normal final-reply future path."""
    client, _, fake_socket = _make_client()
    params = SamplingParams(temperature=0.5)

    future = client.add_request_with_kv_handoff([1, 2, 3], params, {"agent": "prefill"}, [10, 11])

    assert isinstance(future, asyncio.Future)
    assert params.streaming is False
    assert client.completion_futures == {0: future}
    assert client.streams == {}
    # Framed as [metadata, prompt, src_block_ids]: src_block_ids names one block
    # per block_size_tokens of prompt, so it grows with the prompt and travels as
    # its own body rather than in the metadata frame the coordinator decodes.
    meta_frame, prompt_frame, blocks_frame = fake_socket.send_multipart.call_args.args[0]
    metadata = msgpack.unpackb(meta_frame, raw=False)
    assert metadata[0] == Headers.SUBMIT_REQUEST_WITH_KV.value
    assert metadata[1] == 0
    assert metadata[2]["streaming"] is False
    assert metadata[3] == {"agent": "prefill"}
    assert msgpack.unpackb(prompt_frame, raw=False) == [1, 2, 3]
    assert msgpack.unpackb(blocks_frame, raw=False) == [10, 11]
    future.cancel()


def _configured_client(policy=None, block_size=4):
    from megatron.core.inference.config import PrefixCachingCoordinatorPolicy

    fake_socket = MagicMock(name="zmq_socket")
    fake_context = MagicMock(name="zmq_context")
    fake_context.socket.return_value = fake_socket
    with patch("megatron.core.inference.inference_client.zmq.Context", return_value=fake_context):
        client = InferenceClient(
            "tcp://127.0.0.1:5555",
            block_size_tokens=block_size,
            prefix_caching_coordinator_policy=policy
            or PrefixCachingCoordinatorPolicy.LONGEST_PREFIX,
        )
    return client, fake_socket


def _hash_frame(fake_socket):
    return msgpack.unpackb(fake_socket.send_multipart.call_args.args[0][2], raw=False)


async def test_unconfigured_client_says_it_did_not_hash():
    """The regression that matters: None, never [].

    Callers that build an InferenceClient directly -- MegatronAsyncLLM,
    megatron.rl, the coordinator example -- configure neither the block size nor
    the policy. Reporting an empty list for them reads as "hashed, nothing
    matched", so the coordinator skips its own hashing and prefix-affinity
    routing silently degrades to load balancing with nothing raising.
    """
    fake_socket = MagicMock(name="zmq_socket")
    fake_context = MagicMock(name="zmq_context")
    fake_context.socket.return_value = fake_socket
    with patch("megatron.core.inference.inference_client.zmq.Context", return_value=fake_context):
        client = InferenceClient("tcp://127.0.0.1:5555")
    client.add_request(list(range(8)), SamplingParams()).cancel()
    assert _hash_frame(fake_socket) is None


async def test_configured_client_hashes_a_text_prompt():
    import torch

    from megatron.core.inference.inference_request import compute_block_hashes_batched

    client, fake_socket = _configured_client()
    tokens = list(range(8))
    client.add_request(tokens, SamplingParams()).cancel()
    assert _hash_frame(fake_socket) == compute_block_hashes_batched(
        torch.tensor(tokens, dtype=torch.int64), block_size=4
    )


async def test_text_only_client_does_not_expose_media_submission():
    client, fake_socket = _configured_client()
    with pytest.raises(NotImplementedError, match="multi_modal_data"):
        client.add_request([1, 2], SamplingParams(), multi_modal_data={"image": b"jpeg"})
    fake_socket.send_multipart.assert_not_called()
    assert not client.completion_futures


async def test_load_balanced_policy_skips_hashing():
    """Nobody reads them under LOAD_BALANCED, so computing them is pure overhead."""
    from megatron.core.inference.config import PrefixCachingCoordinatorPolicy

    client, fake_socket = _configured_client(PrefixCachingCoordinatorPolicy.LOAD_BALANCED)
    client.add_request(list(range(8)), SamplingParams()).cancel()
    assert _hash_frame(fake_socket) == []


async def test_string_prompt_is_left_to_the_coordinator():
    """Hashing needs token ids, and the client has no tokenizer."""
    client, fake_socket = _configured_client()
    client.add_request("some prompt text", SamplingParams()).cancel()
    assert _hash_frame(fake_socket) is None


async def test_add_request_with_id_returns_the_id_abort_needs():
    """The id handed back is the one that reaches the coordinator as ABORT_REQUEST.

    A non-streaming HTTP response writes nothing to the socket while it
    generates, so a client that disconnects mid-generation is never discovered
    as a broken pipe. The handler has to abort explicitly, and abort_request
    takes an id -- with only the future in hand there is nothing to name, and
    cancelling the future alone leaves the engine generating. add_request keeps
    returning the future alone, delegating here for the id sequence.
    """
    client, _, fake_socket = _make_client()

    delegated = client.add_request("a", SamplingParams())
    request_id, future = client.add_request_with_id("hello", SamplingParams())

    assert isinstance(future, asyncio.Future)
    assert request_id == 1, "add_request must consume an id from the same counter"
    assert client.completion_futures == {0: delegated, request_id: future}
    # The submission travels as multipart frames; header and id are in frame 0.
    submitted = msgpack.unpackb(fake_socket.send_multipart.call_args.args[0][0], raw=False)
    assert submitted[0] == Headers.SUBMIT_REQUEST.value
    assert submitted[1] == request_id

    client.abort_request(request_id)

    aborted = msgpack.unpackb(fake_socket.send.call_args.args[0], raw=False)
    assert aborted == [Headers.ABORT_REQUEST.value, request_id]
    # The abort has to drop local state too, or the handler leaks a future that
    # nothing will ever resolve.
    assert request_id not in client.completion_futures
    assert request_id in client.aborted_request_ids

    delegated.cancel()


def _consume_reply_for_a_completed_request(client):
    """Submit, then stand in for _recv_task delivering the final reply."""
    request_id, future = client.add_request_with_id("hello", SamplingParams())
    # _recv_task pops the future from completion_futures immediately before
    # resolving it.
    client.completion_futures.pop(request_id)
    future.set_result({"tokens": [1]})
    return request_id


def _consume_reply_for_a_finished_stream(client):
    """Same, for the streaming path, whose AsyncStream aborts on close."""
    stream = client.add_request_streaming("hello", SamplingParams())
    # _recv_task pops the stream before delivering the final frame.
    client.streams.pop(stream.request_id)
    return stream.request_id


@pytest.mark.parametrize(
    "finish_request",
    [_consume_reply_for_a_completed_request, _consume_reply_for_a_finished_stream],
    ids=["completed_future", "finished_stream"],
)
async def test_abort_request_ignores_a_request_with_no_local_state(finish_request):
    """A completed request must not be recorded in aborted_request_ids.

    That set is pruned in exactly one place -- _recv_task, when an ENGINE_REPLY
    arrives for the id. Once the reply has been consumed no further reply will
    ever arrive, so recording the id there leaks an entry for the lifetime of
    the process: the same unbounded growth this abort path exists to prevent,
    moved from the engine's batch into the client. The new callers hit this
    routinely -- gather propagates on the first failure while siblings that
    already succeeded are aborted after completion.
    """
    client, _, fake_socket = _make_client()

    request_id = finish_request(client)
    fake_socket.send.reset_mock()

    client.abort_request(request_id)

    assert request_id not in client.aborted_request_ids
    # The coordinator has already dropped its mapping for a finished request,
    # so the ABORT_REQUEST send would be wasted too.
    fake_socket.send.assert_not_called()
