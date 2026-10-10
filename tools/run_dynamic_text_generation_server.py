# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import argparse
import asyncio
import logging
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    start_text_gen_server,
    stop_text_gen_server,
)
from megatron.core.utils import configure_nvtx_profiling, get_pg_size, trace_async_exceptions
from megatron.inference.utils import add_inference_args, get_dynamic_inference_engine
from megatron.post_training.arguments import add_modelopt_args
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron


def add_text_generation_server_args(parser: argparse.ArgumentParser):
    """Adds the required command line arguments for running the text generation server."""
    parser = add_modelopt_args(parser)
    parser = add_inference_args(parser)
    parser.add_argument("--port", type=int, default=5000, help="Port for Flask server to run on")
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Hostname or IP address to bind the server to. Defaults to 0.0.0.0 (all interfaces).",
    )
    parser.add_argument(
        "--parsers", type=str, nargs="+", default=[], help="Parsers to use for parsing the response"
    )
    parser.add_argument(
        "--frontend-replicas",
        type=int,
        default=-1,
        help="Number of HTTP frontend processes spawned per hosting rank. "
        "-1 (default) uses max(data parallel size, 4), or a flat 4 with "
        "--frontend-on-all-ranks, where capacity already scales with the "
        "number of ranks hosting a frontend.",
    )
    parser.add_argument(
        "--frontend-on-all-ranks",
        action="store_true",
        help="Run HTTP frontends on every rank instead of only rank 0, and "
        "return every rank's URL for the caller to spread requests over. "
        "Frontend work (chat template, detokenize, parsers, JSON) is "
        "CPU-bound and otherwise confined to the hosting rank's CPU "
        "allocation, which leaves the rest of the job's cores unused. "
        "Ranks still share one DP coordinator; only the HTTP tier is "
        "replicated.",
    )
    # NOTE: --chat-template is already declared by upstream's TrainingConfig
    # (megatron/training/config/training_config.py); we don't re-register it.
    # The chat_completions endpoint reads it from args.chat_template via
    # _load_chat_template, which accepts either a file path or an inline string.
    parser.add_argument(
        "--default-temperature",
        type=float,
        default=1.0,
        help="Default temperature sampling value when a request does not specify temperature.",
    )
    parser.add_argument(
        "--default-top-p",
        type=float,
        default=1.0,
        help="Default top-p sampling value when a request does not specify top_p.",
    )
    parser.add_argument(
        "--default-top-k",
        type=int,
        default=0,
        help="Default top-k sampling value when a request does not specify top_k.",
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help=(
            "Optimize defaults for pure serving. In chat requests, prevent_retokenization "
            "defaults to false so prompt token IDs are not returned."
        ),
    )
    return parser


@trace_async_exceptions
async def run_text_generation_server(
    engine: DynamicInferenceEngine,
    coordinator_port: int,
    server_port: int,
    hostname: str | None = None,
    chat_template: str | None = None,
    default_temperature: float = 1.0,
    default_top_p: float = 1.0,
    default_top_k: int = 0,
    eval_mode: bool = False,
):
    """
    Runs the text generation server from rank 0 and initializes the
    DynamicInferenceEngine on all ranks.

    Args:
        engine (DynamicInferenceEngine): The dynamic inference engine.
        coordinator_port (int): The network port for the dynamic inference DP coordinator.
        server_port (int): The network for port the frontend text generation server.
        hostname (str | None): Hostname or IP address for coordinator and HTTP traffic.
        chat_template (str | None): Inline chat template or contents loaded from a file.
        default_temperature (float): Sampling default when a request omits `temperature`.
        default_top_p (float): Sampling default when a request omits `top_p`.
        default_top_k (int): Sampling default when a request omits `top_k`.
        eval_mode (bool): Whether to use evaluation response defaults.
    """

    rank = torch.distributed.get_rank()

    coordinator_addr = await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=coordinator_port,
        launch_inference_coordinator=True,
        hostname=hostname,
    )

    num_replicas = getattr(args, 'frontend_replicas', -1)
    if num_replicas < 0:
        if getattr(args, 'frontend_on_all_ranks', False):
            # Capacity now scales with the number of ranks, so the per-rank
            # replica count stays flat rather than tracking DP size on top of it.
            num_replicas = 4
        else:
            # Each replica is a single event loop, so frontend capacity has to scale with
            # the number of engines it feeds. The floor of 4 preserves the previous default
            # for small deployments.
            num_replicas = max(get_pg_size(engine.pg_collection.dp), 4)
    if rank == 0:
        logging.info("Starting %d HTTP frontend replica(s) per hosting rank.", num_replicas)

    if getattr(args, 'frontend_on_all_ranks', False):
        # Only the DP coordinator rank learns the coordinator's address: the
        # engine broadcasts it over the DP group, which is a singleton when data
        # parallel size is 1. Every rank needs it here, since every rank's
        # frontend opens its own client.
        address = [coordinator_addr]
        torch.distributed.broadcast_object_list(address, src=0)
        coordinator_addr = address[0]
        assert coordinator_addr is not None, "no rank published a DP coordinator address"

    try:
        url = None
        if getattr(args, 'frontend_on_all_ranks', False) or rank == 0:
            url = start_text_gen_server(
                coordinator_addr=coordinator_addr,
                tokenizer=engine.controller.tokenizer,
                parsers=args.parsers,
                rank=rank,
                server_port=0 if getattr(args, 'frontend_on_all_ranks', False) else server_port,
                verbose=args.inference_text_gen_server_logging,
                num_replicas=num_replicas,
                hostname=hostname,
                chat_template=chat_template,
                multimodal_prompt_config=getattr(
                    engine.controller.inference_wrapped_model, "multimodal_prompt_config", None
                ),
                default_temperature=default_temperature,
                default_top_p=default_top_p,
                default_top_k=default_top_k,
                eval_mode=eval_mode,
                # Taken from the engine so the frontend hashes on the same block
                # boundaries the engine caches on; a mismatch would name blocks it
                # never held and every routing decision would miss.
                block_size_tokens=engine.context.block_size_tokens,
                prefix_caching_coordinator_policy=(
                    engine.context.prefix_caching_coordinator_policy
                ),
            )

        if getattr(args, 'frontend_on_all_ranks', False):
            # Unlike callers that already collect a URL per worker, this entry
            # point has to gather them itself before it can report the set.
            urls = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(urls, url)
            if rank == 0:
                for entry in [u for u in urls if u]:
                    logging.info("Frontend: %s", entry)
        elif rank == 0:
            logging.info("Frontend: %s", url)

        # Await the engine loop directly since the server is running in a separate process
        await engine.engine_loop_task

    finally:
        # Guarantee that the separate processes are terminated when the engine loop
        # stops or is interrupted. Every rank may now own frontend processes.
        stop_text_gen_server()


def _load_chat_template(value):
    """Resolve a --chat-template arg into the template string itself.

    If the value is a path to an existing file, read it. Otherwise treat the
    value as the inline template.
    """
    if value is None:
        return None
    if os.path.isfile(value):
        with open(value) as f:
            return f.read()
    return value


if __name__ == "__main__":
    with torch.inference_mode():
        initialize_megatron(
            extra_args_provider=add_text_generation_server_args,
            args_defaults={'no_load_rng': True, 'no_load_optim': True},
        )

        args = get_args()

        # Match training's NVTX gating (training.py only flips this when both
        # --profile and --nvtx-ranges are set). Otherwise the engine-side
        # nvtx_range_push labels (bookkeeping, Decode, _ep_establish_consensus,
        # etc.) are no-ops and the inter-step gap is unattributable in nsys.
        if args.profile and args.nvtx_ranges:
            configure_nvtx_profiling(True)

        # Enable return_log_probs to allow prompt logprobs computation for echo=True requests
        # This sets materialize_only_last_token_logits=False in the inference context,
        # which is required for lm-eval compatibility (loglikelihood evaluation tasks)
        args.return_log_probs = True

        engine = get_dynamic_inference_engine()
        chat_template = _load_chat_template(getattr(args, 'chat_template', None))

        try:
            asyncio.run(
                run_text_generation_server(
                    engine,
                    args.inference_coordinator_port,
                    args.port,
                    args.host,
                    chat_template=chat_template,
                    default_temperature=args.default_temperature,
                    default_top_p=args.default_top_p,
                    default_top_k=args.default_top_k,
                    eval_mode=args.eval_mode,
                )
            )
        except KeyboardInterrupt:
            # Catching at the top level ensures clean stdout without spamming the traceback
            print("Server process interrupted by user.")
        finally:
            # Clean up PyTorch distributed groups properly
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
