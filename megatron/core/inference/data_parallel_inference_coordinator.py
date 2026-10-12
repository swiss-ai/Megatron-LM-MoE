# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import errno
import faulthandler
import json
import logging
import signal
import socket
import time
from collections import deque
from enum import Enum, auto
from multiprocessing import Event
from multiprocessing.connection import Connection

import numpy as np
import torch

from megatron.core.inference.config import PrefixCachingCoordinatorPolicy, routes_on_prefix
from megatron.core.inference.headers import Headers, UnknownHeaderError
from megatron.core.inference.inference_request import compute_block_hashes_batched
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)

try:
    import zmq

    HAVE_ZMQ = True
except:
    HAVE_ZMQ = False

try:
    import msgpack

    HAVE_MSGPACK = True
except:
    HAVE_MSGPACK = False

# Register faulthandler to emit stack traces upon process kill.
faulthandler.enable()
faulthandler.register(signal.SIGTERM, all_threads=False, chain=True)
faulthandler.register(signal.SIGINT, all_threads=False, chain=True)


class DataParallelInferenceCoordinator:
    """
    Coordinates inference requests between clients and distributed model engines.

    This class acts as a central server. It uses a ZMQ ROUTER socket to manage
    communication flows between multiple clients and multiple data parallel ranks.

    The coordinator's main responsibilities are:
    1.  **Worker Registration**: It waits for a specified number of data parallel ranks
        (representing distributed model instances) to connect and register themselves.
    2.  **Client Connection**: It accepts connections from external clients, like
        `InferenceClient`, and performs a simple handshake.
    3.  **Request Forwarding**: It receives inference requests from clients, assigns a
        unique server-side request ID, tokenizes the prompt, and forwards the request
        to one of the available data parallel ranks using load-balanced (and,
        when prefix caching is enabled, prefix-affinity-aware) routing.
    4.  **Response Routing**: It receives completed results from
        the data parallel ranks and routes them back to the original client that made the
        request.
    5.  **Control Signal Broadcasting**: It relays control signals (e.g., PAUSE, STOP)
        from a client to all connected data parallel ranks.

    Attributes:
        router_socket (zmq.Socket): The central ZMQ ROUTER socket for all communication.
        data_parallel_size (int): The number of data parallel workers to expect.
        identities_of_data_parallel_ranks (deque): A deque holding the ZMQ
            identities of connected data parallel instances, used for request routing.
        request_id_to_client_id (dict): Maps server-side request IDs to the ZMQ
            identity of the client that initiated the request.
        request_id_to_client_request_id (dict): Maps server-side request IDs to the
            original request ID provided by the client.
        next_request_id (int): A counter for generating unique server-side request IDs.
    """

    class CoordinatorState(Enum):
        """State machine for the coordinator."""

        RUNNING = auto()
        PAUSED = auto()
        SUSPENDED = auto()
        STOPPING = auto()

    def __init__(
        self,
        pipe_connection: Connection,
        data_parallel_size: int,
        tokenizer,
        max_requests,
        inference_coordinator_port: int | None = None,
        deterministic_mode: bool = False,
        block_size_tokens: int | None = None,
        enable_prefix_caching: bool = False,
        prefix_caching_coordinator_policy: PrefixCachingCoordinatorPolicy = (
            PrefixCachingCoordinatorPolicy.LONGEST_PREFIX
        ),
        prefix_caching_routing_alpha: float = 1.0,
        prefix_cache_ttl_seconds: float = 300.0,
        schedule_output_path: str | None = None,
        hostname: str | None = None,
    ):
        """
        Initializes the inference coordinator.

        This sets up the ZMQ context and a ROUTER socket, binding it to the given
        port. It then enters a blocking loop to wait for all expected data parallel
        ranks to connect before proceeding.

        Args:
            pipe_connection (Connection): A connecting pipe to the parent process.
            data_parallel_size (int): The number of data parallel instances that are
                expected to connect.
            tokenizer: The tokenizer to use for prompt tokenization and detokenization.
            inference_coordinator_port (Optional[int]): The TCP port number to bind the server to.
            prefix_caching_routing_alpha (float): Relative-load penalty coefficient:
                score = normalized_prefix_depth - alpha * relative_load.
            max_requests (int): Max concurrent requests per rank, used to
                compute normalized_load for prefix-aware scoring.
        """
        assert HAVE_ZMQ, (
            "please install the pyzmq library to use DataParallelInferenceCoordinator\n"
            "pip install pyzmq"
        )
        assert HAVE_MSGPACK, (
            "please install the messagepack library to use DataParallelInferenceCoordinator\n"
            "pip install msgpack"
        )
        self.pipe_connection = pipe_connection
        self.data_parallel_size = data_parallel_size
        self.context = zmq.Context()

        # This is the central router socket
        # 1. data parallel ranks connect to this socket to register themselves
        # 2. Users connect to this socket and submit their requests. We transmit them to
        #    data parallel ranks according to the configured routing policy
        # 3. data parallel ranks return completed requests to this socket. We route them back to
        #    the user that had submitted the request originally.

        # Get local IP.
        local_ip = hostname or socket.gethostname()

        self.router_socket = self.context.socket(zmq.ROUTER)
        # Raise error if the other side of the connection has dropped.
        self.router_socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        is_bound = False
        if inference_coordinator_port is not None:
            try:
                self.router_socket.bind(f"tcp://{local_ip}:{inference_coordinator_port}")
                is_bound = True
            except zmq.error.ZMQError as e:
                if e.errno == errno.EADDRINUSE:
                    logging.warning(
                        f"Port {inference_coordinator_port} is already in use. "
                        "Binding to a random available port instead."
                    )
            except Exception:
                logging.warning(
                    f"Unknown error when binding to port {inference_coordinator_port}. "
                    "Attempting to bind to a random available port instead."
                )
        if not is_bound:
            self.router_socket.bind_to_random_port(f"tcp://{local_ip}")
        self.addr = self.router_socket.getsockopt_string(zmq.LAST_ENDPOINT)

        # Send the address to the parent process.
        self.pipe_connection.send(self.addr)
        self.pipe_connection.close()

        logging.info("Inference Coordinator: waiting for connections from data parallel ranks...")
        # First wait for all data parallel ranks to establish connections.
        self.identities_of_data_parallel_ranks = deque([])
        self.removed_engine_identities = set()
        # time.sleep(5)  # Give data parallel ranks time to spawn and connect.
        for _ in range(data_parallel_size):
            identity, _ = self.router_socket.recv_multipart()
            assert identity not in self.identities_of_data_parallel_ranks
            self.identities_of_data_parallel_ranks.append(identity)
        logging.info("Inference Coordinator: Connected with data parallel ranks...")

        # In deterministic mode, sort identities for consistent scheduling order.
        if deterministic_mode:
            self.identities_of_data_parallel_ranks = deque(
                sorted(self.identities_of_data_parallel_ranks)
            )

        self.request_id_to_client_id = {}
        self.request_id_to_client_request_id = {}
        self.client_request_to_request_id = {}
        self.request_id_to_rank = {}  # Maps request_id → rank identity for pending count tracking
        self.removed_engine_identities = set()

        self.next_request_id = 0
        self.known_clients = set()
        self.tokenizer = tokenizer
        self.state = self.CoordinatorState.RUNNING

        # Prefix caching state for routing.
        self.block_size_tokens = block_size_tokens
        self.enable_prefix_caching = enable_prefix_caching
        self.prefix_caching_coordinator_policy = prefix_caching_coordinator_policy
        self.prefix_caching_routing_alpha = prefix_caching_routing_alpha
        self.prefix_cache_ttl_seconds = prefix_cache_ttl_seconds
        self.max_requests = max_requests
        assert self.max_requests is not None and self.max_requests > 0

        # Schedule recording.
        self.schedule_output_path = schedule_output_path
        self.schedule_records = [] if schedule_output_path else None

        # Deterministic rank index mapping (sorted identity -> 0-based index).
        sorted_identities = sorted(self.identities_of_data_parallel_ranks)
        self.identity_to_rank_index = {
            identity: idx for idx, identity in enumerate(sorted_identities)
        }

        # Numpy arrays for vectorized scoring (indexed by rank index).
        n_ranks = len(sorted_identities)
        self._identities_list = list(sorted_identities)  # rank_index → identity
        self._pending_counts = np.zeros(n_ranks, dtype=np.int32)

        # Hash → {rank_idx: touch_time}, with monotonic timestamps.
        self._hash_table: dict[int, dict[int, float]] = {}
        self._hash_expiry: deque = deque()

    def get_least_loaded_data_parallel_rank(self):
        """
        Selects the data parallel rank with the fewest in-flight requests.

        Ties are broken by lowest rank index for deterministic behavior.

        Returns:
            bytes: The ZMQ identity of the least-loaded data parallel rank.
        """
        if not self._identities_list:
            raise RuntimeError("No engines connected")
        best_idx = int(np.argmin(self._pending_counts))
        return self._identities_list[best_idx]

    def _register_rank_identity(self, identity):
        """Register a new rank identity in the scoring data structures.

        Called when a rank dynamically connects to a running coordinator
        (e.g. in tests that spawn the coordinator with data_parallel_size=0
        and let engines register after the fact).
        """
        if identity in self.identity_to_rank_index:
            return
        new_idx = len(self._identities_list)
        self.identity_to_rank_index[identity] = new_idx
        self._identities_list.append(identity)
        self._pending_counts = np.append(self._pending_counts, np.int32(0))
        logging.info(
            "Coordinator: registered engine %s as rank index %d (now %d engines)",
            identity,
            new_idx,
            len(self._identities_list),
        )

    def _remove_engine(self, identity):
        """Remove a disconnected engine from all routing bookkeeping.
        Called both during shutdown and when an engine becomes unreachable mid-operation
        (e.g. zmq.EHOSTUNREACH in _send_to_engine). The O(n) index-shifting and hash-table
        rebuild are acceptable because the number of connected engines is small; optimize
        only if dynamic registration/deregistration at high engine counts becomes a use case.
        """
        self.identities_of_data_parallel_ranks.remove(identity)
        self.removed_engine_identities.add(identity)
        idx = self.identity_to_rank_index.pop(identity, None)
        if idx is None:
            return
        self._identities_list.pop(idx)
        self._pending_counts = np.delete(self._pending_counts, idx)
        # Shift indices for engines that came after the removed slot.
        for ident in self.identity_to_rank_index:
            if self.identity_to_rank_index[ident] > idx:
                self.identity_to_rank_index[ident] -= 1
        # Drop hash-table entries for the removed rank; shift indices above it.
        new_hash_table = {}
        for h, rank_ts in self._hash_table.items():
            # h is hash index
            # rank_ts is a dict mapping rank_idx → timestamp
            new_row = {}
            for r, ts in rank_ts.items():
                if r == idx:
                    # skip this rank as it is removed
                    continue
                new_r = r - 1 if r > idx else r
                new_row[new_r] = ts
            if new_row:
                new_hash_table[h] = new_row
        self._hash_table = new_hash_table
        logging.warning(
            "Coordinator: removed engine %s (now %d engines)",
            identity,
            len(self.identities_of_data_parallel_ranks),
        )

    def _send_to_engine(self, identity, frames):
        """Send a message to an engine, removing it from the pool if unreachable.

        Args:
            identity: ZMQ identity of the target engine.
            frames (list): Raw frames to send, metadata frame first.

        Returns:
            True if the send succeeded, False if the engine was unreachable and removed.
        """
        try:
            self.router_socket.send_multipart([identity, *frames])
            return True
        except zmq.error.ZMQError as e:
            if e.errno == zmq.EHOSTUNREACH:
                self._remove_engine(identity)
                return False
            raise

    def compute_request_hashes(self, prompt):
        """Compute block hashes for a prompt on CPU.

        Callers decide whether hashes are wanted at all: computing them requires
        the decoded prompt, and decoding it is the cost the caller is usually
        trying to avoid. See handle_submit_request.

        Args:
            prompt: Either a string (to be tokenized) or a list of token IDs.

        Returns:
            List of integer block hashes, or empty list if prefix caching is disabled.
        """
        if isinstance(prompt, str):
            tokens = self.tokenizer.tokenize(prompt)
        else:
            tokens = list(prompt)
        token_tensor = torch.tensor(tokens, dtype=torch.int64)
        return compute_block_hashes_batched(token_tensor, self.block_size_tokens)

    def get_best_data_parallel_rank(self, request_hashes):
        """Select the best DP rank based on prefix cache affinity and load.

        Uses score = cache_score - alpha * relative_load, with reusable prefix
        depth normalized by prompt blocks and load measured against the fleet mean.

        Args:
            request_hashes: List of block hashes for the request.

        Returns:
            bytes: The ZMQ identity of the selected data parallel rank.
        """
        if self.prefix_caching_coordinator_policy == PrefixCachingCoordinatorPolicy.LOAD_BALANCED:
            return self.get_least_loaded_data_parallel_rank()

        # Without prefix caching (or when the request has no hashes to match on)
        # fall back to load-balanced routing.
        if not self.enable_prefix_caching or not request_hashes:
            return self.get_least_loaded_data_parallel_rank()

        self._expire_rank_hashes(time.monotonic())
        prefix_blocks = self._prefix_depth_vector(request_hashes)
        if not prefix_blocks.any():
            return self.get_least_loaded_data_parallel_rank()
        cache_score = prefix_blocks / len(request_hashes)
        n_ranks = len(self._identities_list)
        mean_load = float(self._pending_counts.mean()) if n_ranks else 0.0
        relative_load = (self._pending_counts - mean_load) / max(1.0, mean_load)
        scores = cache_score - self.prefix_caching_routing_alpha * relative_load
        order = np.lexsort((np.arange(n_ranks), self._pending_counts, -scores))
        return self._identities_list[int(order[0])]

    def _update_rank_hashes(self, rank_identity, request_hashes):
        """Record that a rank owns the given hashes.

        Args:
            rank_identity: ZMQ identity of the target rank.
            request_hashes: List of block hashes assigned to this rank.
        """
        rank_idx = self.identity_to_rank_index[rank_identity]
        # One timestamp for the whole call keeps the expiry queue sorted.
        now = time.monotonic()
        for h in request_hashes:
            self._hash_table.setdefault(h, {})[rank_idx] = now
            self._hash_expiry.append((now, h))
        # Swept here rather than on a timer: the coordinator is a single event
        # loop with nothing else to run it, and expiry only needs to keep pace
        # with the traffic that grows the table.
        self._expire_rank_hashes(now)

    def _expire_rank_hashes(self, now):
        """Drop hash entries no request has touched for the TTL.

        The coordinator's table is a guess about what each engine still holds: it
        sees blocks being routed, never blocks being evicted. Left alone the guess
        only gets staler, and the coordinator keeps sending requests to a rank for
        a prefix it dropped long ago, paying a cold prefill and passing up a rank
        that could have served it.

        The queue is insertion-ordered, so expired entries form a prefix of it and
        the sweep stops at the first live one -- the cost is what it evicts, not
        the size of the table. An entry re-routed since it was queued carries a
        newer timestamp than the queue entry and is left alone, which is what
        makes a stale duplicate harmless.
        """
        ttl = self.prefix_cache_ttl_seconds
        while self._hash_expiry:
            ts, h = self._hash_expiry[0]
            if now - ts <= ttl:
                break
            self._hash_expiry.popleft()
            row = self._hash_table.get(h)
            if row is None:
                continue
            for rank_idx in [r for r, touched in row.items() if touched <= ts]:
                del row[rank_idx]
            if not row:
                del self._hash_table[h]

    def _prefix_depth_vector(self, hashes):
        """Return each rank's contiguous prefix depth, in blocks.

        Prefix cache
        reuse requires an unbroken chain from the first block -- each block hash
        chains the previous one's digest -- so a rank only benefits up to its
        first miss. Walking forward and dropping ranks as they miss gives each
        rank its true depth, and the loop exits as soon as no rank is left.

        Counting forward also refuses to credit a rank for a deep block whose
        prefix has been evicted: that KV cannot be reused, because reaching it
        requires the blocks before it.

        """
        n_ranks = len(self._identities_list)
        depth = np.zeros(n_ranks, dtype=np.int64)
        alive = np.ones(n_ranks, dtype=bool)
        for h in hashes:
            row = self._hash_table.get(h)
            if row is None:
                break
            present = np.zeros(n_ranks, dtype=bool)
            rank_idxs = np.fromiter(row.keys(), dtype=np.intp, count=len(row))
            present[rank_idxs] = True
            alive &= present
            if not alive.any():
                break
            depth += alive
        return depth.astype(np.float64)

    def start(self):
        """
        Starts the main event loop for the coordinator.

        This method runs an infinite loop, continuously listening for incoming
        messages on the ZMQ ROUTER socket. It parses the message header to
        determine the message type and takes appropriate action, such as
        handling new client connections, forwarding requests, broadcasting
        control signals, or processing replies from the engines.
        """
        # Todo [Siddharth]: Make this more robust to handle invalid messages.
        known_clients = self.known_clients
        while True:
            # Messages are one or more frames. frames[0] is the metadata frame:
            # a header plus whatever the coordinator needs to route the message.
            # Any later frames are opaque payload bodies, forwarded without ever
            # being decoded here -- that is what keeps this loop's cost
            # independent of prompt length.
            sender_identity, *frames = self.router_socket.recv_multipart()

            # Allow for re-registration if connecting to a running coordinator.
            if frames[0] == b"":
                if sender_identity not in self.identities_of_data_parallel_ranks:
                    self.identities_of_data_parallel_ranks.append(sender_identity)
                    self._register_rank_identity(sender_identity)
                continue

            deserialized_payload = msgpack.unpackb(frames[0], raw=False)
            header = Headers(deserialized_payload[0])

            if header == Headers.CONNECT:
                if sender_identity in known_clients:
                    logging.info(
                        f"Client {sender_identity} sent a duplicate connect request. Ignoring .."
                    )
                    continue

                # print(f"New client connected: {sender_identity}")
                known_clients.add(sender_identity)
                self.router_socket.send_multipart(
                    [sender_identity, msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True)]
                )

            elif header == Headers.SUBMIT_REQUEST:
                # ToDo [Siddharth]: We might want to tokenize the prompt on the
                # assigned data parallel rank for this process instead
                # of the coordinator.

                # Message from a known client
                if sender_identity not in known_clients:
                    logging.info(
                        f"Received message from unknown client {sender_identity}. Ignoring."
                    )
                    continue
                # this is a message from a client.
                # route it to a data parallel rank
                if len(frames) != 4 or len(deserialized_payload) != 4:
                    logging.error("Coordinator: malformed framed SUBMIT_REQUEST")
                    continue
                client_request_id, sampling_params, media_meta = deserialized_payload[1:]
                if media_meta is not None:
                    raise ValueError("Multimodal submissions are not supported by this coordinator")
                # map client request_id to server request_id
                # necessary because multiple clients might have the same request_id.
                request_id = self.next_request_id
                self.next_request_id += 1
                self.request_id_to_client_id[request_id] = sender_identity
                self.request_id_to_client_request_id[request_id] = client_request_id
                self.client_request_to_request_id[(sender_identity, client_request_id)] = request_id

                payload = [
                    msgpack.packb(
                        [Headers.SUBMIT_REQUEST.value, request_id, sampling_params, None],
                        use_bin_type=True,
                    ),
                    frames[1],
                    frames[3],
                ]
                request_hashes = []
                if self.enable_prefix_caching and routes_on_prefix(
                    self.prefix_caching_coordinator_policy
                ):
                    request_hashes = msgpack.unpackb(frames[2], raw=False)
                    if request_hashes is None:
                        request_hashes = self.compute_request_hashes(
                            msgpack.unpackb(frames[1], raw=False)
                        )
                if (
                    self.prefix_caching_coordinator_policy
                    == PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
                ):
                    request_hashes = request_hashes[:1]

                # Account for the fact that some engines may have died.
                for _ in range(len(self.identities_of_data_parallel_ranks)):
                    next_identity = self.get_best_data_parallel_rank(request_hashes)
                    if self._send_to_engine(next_identity, payload):
                        break
                else:
                    # If all engines have died, we are in an abnormal state, and must exit cleanly.
                    logging.error("Coordinator: no reachable engines for request %d", request_id)
                    del self.request_id_to_client_id[request_id]
                    del self.request_id_to_client_request_id[request_id]
                    del self.client_request_to_request_id[(sender_identity, client_request_id)]
                    return

                self.request_id_to_rank[request_id] = next_identity
                self._pending_counts[self.identity_to_rank_index[next_identity]] += 1
                if request_hashes:
                    self._update_rank_hashes(next_identity, request_hashes)
                if self.schedule_records is not None:
                    self.schedule_records.append(
                        {
                            "request_id": request_id,
                            "rank_index": self.identity_to_rank_index[next_identity],
                            "num_hashes": len(request_hashes),
                        }
                    )

            elif header == Headers.SUBMIT_REQUEST_WITH_KV:
                if self._handle_submit_request_with_kv(
                    sender_identity, deserialized_payload, frames[1:]
                ):
                    return

            elif header == Headers.RELEASE_KV:
                self._handle_release_kv(sender_identity, deserialized_payload)

            elif header in (
                Headers.PAUSE,
                Headers.UNPAUSE,
                Headers.SUSPEND,
                Headers.RESUME,
                Headers.SET_GENERATION_EPOCH,
                Headers.STOP,
            ):
                # Start by checking the current state against the control signal.
                if sender_identity not in known_clients:
                    logging.warning("Coordinator: ignoring signal from unknown client.")
                    continue

                if header == Headers.PAUSE:
                    idem_states = (self.CoordinatorState.PAUSED, self.CoordinatorState.SUSPENDED)
                    if self.state == self.CoordinatorState.RUNNING:
                        self.state = self.CoordinatorState.PAUSED
                    elif self.state in idem_states:
                        # Already paused/suspended, ignore redundant PAUSE.
                        continue
                    else:
                        logging.warning("Coordinator: ignoring PAUSE in state %s", self.state)
                        continue
                elif header == Headers.UNPAUSE:
                    if self.state != self.CoordinatorState.PAUSED:
                        logging.warning("Coordinator: ignoring UNPAUSE in state %s", self.state)
                        continue
                    self.state = self.CoordinatorState.RUNNING
                elif header == Headers.SUSPEND:
                    if self.state != self.CoordinatorState.PAUSED:
                        logging.warning("Coordinator: ignoring SUSPEND in state %s", self.state)
                        continue
                    self.state = self.CoordinatorState.SUSPENDED
                elif header == Headers.RESUME:
                    if self.state != self.CoordinatorState.SUSPENDED:
                        logging.warning("Coordinator: ignoring RESUME in state %s", self.state)
                        continue
                    self.state = self.CoordinatorState.PAUSED
                elif header == Headers.STOP:
                    good_states = (self.CoordinatorState.PAUSED, self.CoordinatorState.SUSPENDED)
                    if self.state not in good_states:
                        logging.warning("Coordinator: ignoring STOP in state %s", self.state)
                        continue
                    self.state = self.CoordinatorState.STOPPING

                # Broadcast the control signal if we're in a good state.
                # Forward the full deserialized payload so that data-bearing
                # signals (e.g. SET_GENERATION_EPOCH) retain their arguments.
                self._broadcast_to_engines(deserialized_payload)

                # STOP affects engines; reset coordinator to RUNNING to allow future engines.
                if header == Headers.STOP:
                    self.state = self.CoordinatorState.RUNNING

            elif header == Headers.ENGINE_REPLY:
                self._handle_engine_reply(sender_identity, deserialized_payload, frames[1:])

            elif header == Headers.ENGINE_REPLY_PARTIAL:
                # Route token deltas without releasing request mappings or in-flight load.
                if sender_identity not in self.identities_of_data_parallel_ranks:
                    assert (
                        sender_identity in self.removed_engine_identities
                    ), f"ENGINE_REPLY_PARTIAL from never-connected sender {sender_identity!r}"
                    logging.warning(
                        "Coordinator: ENGINE_REPLY_PARTIAL from removed engine %r", sender_identity
                    )
                    continue
                if len(deserialized_payload[1]) != len(frames) - 1:
                    raise ValueError("ENGINE_REPLY_PARTIAL metadata/body count mismatch")
                for request_id, body in zip(deserialized_payload[1], frames[1:]):
                    client_identity = self.request_id_to_client_id[request_id]
                    client_request_id = self.request_id_to_client_request_id[request_id]
                    # Partial tokens are detokenized by the client-facing streaming layer.
                    self.router_socket.send_multipart(
                        [
                            client_identity,
                            msgpack.packb([header.value, client_request_id], use_bin_type=True),
                            body,
                        ]
                    )

            elif header == Headers.ABORT_REQUEST:
                if sender_identity not in known_clients:
                    logging.warning("Coordinator: ignoring abort from unknown client.")
                    continue
                client_request_id = int(deserialized_payload[1])
                request_id = self.client_request_to_request_id.get(
                    (sender_identity, client_request_id)
                )
                if request_id is None:
                    continue
                assigned_rank = self.request_id_to_rank.get(request_id)
                if assigned_rank is not None:
                    self._send_to_engine(
                        assigned_rank,
                        [
                            msgpack.packb(
                                [Headers.ABORT_REQUEST.value, request_id], use_bin_type=True
                            )
                        ],
                    )

            elif header == Headers.SHUTDOWN:
                if sender_identity not in known_clients:
                    logging.warning("Coordinator: ignoring signal from unknown client.")
                    continue
                break

            elif header == Headers.DISCONNECT:
                if sender_identity in self.identities_of_data_parallel_ranks:
                    self._remove_engine(sender_identity)

            else:
                raise UnknownHeaderError(header)

    def _broadcast_to_engines(self, payload):
        broadcast_payload = msgpack.packb(payload, use_bin_type=True)
        for identity in list(self.identities_of_data_parallel_ranks):
            self._send_to_engine(identity, [broadcast_payload])

    def _handle_submit_request_with_kv(self, sender_identity, payload, bodies):
        """Route a client-supplied KV handoff to a decode engine."""
        if sender_identity not in self.known_clients:
            logging.info(
                "Received SUBMIT_REQUEST_WITH_KV from unknown client %s; ignoring.", sender_identity
            )
            return
        if len(payload) != 4 or len(bodies) != 2:
            logging.error(
                "Coordinator: malformed SUBMIT_REQUEST_WITH_KV payload with %d fields",
                len(payload) - 1,
            )
            return

        client_request_id, sampling_params, kv_meta = payload[1:]
        request_id = self.next_request_id
        self.next_request_id += 1
        self.request_id_to_client_id[request_id] = sender_identity
        self.request_id_to_client_request_id[request_id] = client_request_id
        self.client_request_to_request_id[(sender_identity, client_request_id)] = request_id

        engine_payload = [
            msgpack.packb(
                [Headers.SUBMIT_REQUEST_WITH_KV.value, request_id, sampling_params, kv_meta],
                use_bin_type=True,
            ),
            *bodies,
        ]
        for _ in range(len(self.identities_of_data_parallel_ranks)):
            next_identity = self.get_least_loaded_data_parallel_rank()
            if self._send_to_engine(next_identity, engine_payload):
                break
        else:
            logging.error("Coordinator: no reachable engines for handoff request %d", request_id)
            del self.request_id_to_client_id[request_id]
            del self.request_id_to_client_request_id[request_id]
            del self.client_request_to_request_id[(sender_identity, client_request_id)]
            return True

        self.request_id_to_rank[request_id] = next_identity
        self._pending_counts[self.identity_to_rank_index[next_identity]] += 1

    def _handle_release_kv(self, sender_identity, payload):
        """Broadcast release of prefill blocks retained for a completed handoff."""
        if sender_identity not in self.known_clients:
            logging.warning("Coordinator: ignoring RELEASE_KV from unknown client.")
            return
        self._broadcast_to_engines([Headers.RELEASE_KV.value, int(payload[1])])

    def _handle_engine_reply(self, sender_identity, payload, bodies):
        """Deliver queued final replies, including those from a removed engine."""
        if sender_identity not in self.identities_of_data_parallel_ranks:
            assert (
                sender_identity in self.removed_engine_identities
            ), f"ENGINE_REPLY from never-connected sender {sender_identity!r}"
            logging.warning("Coordinator: ENGINE_REPLY from removed engine %r", sender_identity)
        if len(payload[1]) != len(bodies):
            raise ValueError("ENGINE_REPLY metadata/body count mismatch")
        for (fid, needs_detokenize), body in zip(payload[1], bodies):
            if needs_detokenize:
                finished_request = msgpack.unpackb(body, raw=False)
                self.detokenize(finished_request)
                body = msgpack.packb(finished_request, use_bin_type=True)
            client_identity = self.request_id_to_client_id.pop(fid)
            client_request_identity = self.request_id_to_client_request_id.pop(fid)
            del self.client_request_to_request_id[(client_identity, client_request_identity)]
            assigned_rank = self.request_id_to_rank.pop(fid, None)
            if assigned_rank is not None:
                idx = self.identity_to_rank_index.get(assigned_rank)
                if idx is not None:
                    assert self._pending_counts[idx] >= 1
                    self._pending_counts[idx] -= 1
            self.router_socket.send_multipart(
                [
                    client_identity,
                    msgpack.packb(
                        [Headers.ENGINE_REPLY.value, client_request_identity], use_bin_type=True
                    ),
                    body,
                ]
            )

    def detokenize(self, finished_request):
        """
        Detokenizes the generated tokens in the finished request.

        This method uses the coordinator's tokenizer to convert the list of
        generated token IDs back into human-readable text.

        Args:
            finished_request (dict): The serialized merged request containing the
                generated tokens to be detokenized. It is modified in place.
        """
        # Defaults to True, matching SamplingParams: params serialized before
        # this field existed still expect detokenization.
        if not (finished_request.get("sampling_params", {}) or {}).get(
            "detokenize_generations", True
        ):
            return

        if finished_request["prompt"] is None and finished_request.get("prompt_tokens") is not None:
            finished_request["prompt"] = TextGenerationController.detokenize(
                self.tokenizer, finished_request["prompt_tokens"][1], remove_EOD=False
            )
        detokenize_stop_sequence = (finished_request.get("sampling_params", {}) or {}).get(
            "detokenize_stop_sequence", False
        )
        finished_request["generated_text"] = TextGenerationController.detokenize(
            self.tokenizer,
            finished_request["generated_tokens"],
            remove_EOD=not detokenize_stop_sequence,
        )

    @classmethod
    def entrypoint(
        cls,
        pipe_connection: Connection,
        ready_event: Event,
        data_parallel_size: int,
        tokenizer,
        max_requests,
        inference_coordinator_port: int | None = None,
        deterministic_mode: bool = False,
        block_size_tokens: int | None = None,
        enable_prefix_caching: bool = False,
        prefix_caching_coordinator_policy: PrefixCachingCoordinatorPolicy = (
            PrefixCachingCoordinatorPolicy.LONGEST_PREFIX
        ),
        prefix_caching_routing_alpha: float = 1.0,
        prefix_cache_ttl_seconds: float = 300.0,
        schedule_output_path: str | None = None,
        hostname: str | None = None,
    ):
        """
        Class method to instantiate and run the coordinator, for use in a separate process.

        This method initializes the coordinator, signals a `ready_event` to indicate
        that it is fully initialized and listening, and then starts the main event loop.

        Args:
            pipe_connection (Connection): A connecting pipe to the parent process.
            ready_event (Event): A threading or multiprocessing event object that is set()
                once the coordinator is ready to accept connections.
            inference_coordinator_port (int): The port to bind to.
            data_parallel_size (int): The number of expected data parallel instances.
            deterministic_mode (bool): Whether to enable deterministic scheduling.
            block_size_tokens (Optional[int]): Token block size for prefix caching hashing.
            enable_prefix_caching (bool): Whether prefix caching is enabled.
            prefix_caching_coordinator_policy (PrefixCachingCoordinatorPolicy): Routing policy.
            schedule_output_path (Optional[str]): Path to write scheduling decisions JSON.
            prefix_caching_routing_alpha (float): Weight for prefix-aware routing score.
            max_requests (int): Max concurrent requests per rank.
        """
        coordinator = cls(
            pipe_connection,
            data_parallel_size,
            tokenizer,
            max_requests,
            inference_coordinator_port,
            deterministic_mode=deterministic_mode,
            block_size_tokens=block_size_tokens,
            enable_prefix_caching=enable_prefix_caching,
            prefix_caching_coordinator_policy=prefix_caching_coordinator_policy,
            prefix_caching_routing_alpha=prefix_caching_routing_alpha,
            prefix_cache_ttl_seconds=prefix_cache_ttl_seconds,
            schedule_output_path=schedule_output_path,
            hostname=hostname,
        )
        ready_event.set()
        try:
            coordinator.start()
        except KeyboardInterrupt:
            logging.info("Coordinator process interrupted. Exiting...")
        coordinator.stop()
        logging.info("Inference Coordinator: shut down successfully.")

    def stop(self):
        """
        Stops the inference coordinator, performing any necessary cleanup operations.
        """
        if self.schedule_output_path and self.schedule_records:
            schedule_data = {
                "policy": self.prefix_caching_coordinator_policy.value,
                "data_parallel_size": self.data_parallel_size,
                "num_requests": len(self.schedule_records),
                "records": self.schedule_records,
            }
            with open(self.schedule_output_path, "w") as f:
                json.dump(schedule_data, f, indent=2)
        self.router_socket.close()
        self.context.term()
