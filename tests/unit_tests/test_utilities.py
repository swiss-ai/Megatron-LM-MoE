# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import inspect
import os
from argparse import Namespace
from datetime import timedelta
from typing import Literal

import torch

from megatron.core.enums import ModelType
from megatron.training.config.container import PretrainConfigContainer
from megatron.training.training import (
    _build_legacy_dense_model_for_upcycling,
    _dense_model_config_for_upcycling,
    _normalize_pretrain_args,
    _normalize_setup_model_args,
    setup_model_and_optimizer,
)
from torch._C._distributed_c10d import PrefixStore
from torch.distributed import rendezvous

import megatron.core.parallel_state as ps
import megatron.training.training as training
from megatron.training.argument_utils import (
    gpt_config_from_args,
    hybrid_config_from_args,
    pretrain_cfg_container_from_args,
)




def test_pretrain_legacy_positional_orders():
    dataset = object()
    provider = lambda: None
    forward = lambda: None

    normalized = _normalize_pretrain_args(
        dataset, provider, ModelType.encoder_or_decoder, forward, None, None
    )
    assert normalized == (
        None,
        dataset,
        ModelType.encoder_or_decoder,
        forward,
        provider,
        None,
    )

    callback = lambda: None
    normalized = _normalize_pretrain_args(
        dataset, provider, ModelType.encoder_or_decoder, forward, callback, None
    )
    assert normalized == (
        None,
        dataset,
        ModelType.encoder_or_decoder,
        forward,
        provider,
        callback,
    )

    cfg = object.__new__(PretrainConfigContainer)
    normalized = _normalize_pretrain_args(
        cfg, dataset, provider, ModelType.encoder_or_decoder, forward, None
    )
    assert normalized == (
        cfg,
        dataset,
        ModelType.encoder_or_decoder,
        forward,
        provider,
        None,
    )


def test_setup_model_and_optimizer_positional_bindings():
    provider = object()
    model_type = ModelType.encoder_or_decoder
    checkpointing_context = object()
    pg_collection = object()
    signature = inspect.signature(setup_model_and_optimizer)

    old_bound = signature.bind(provider, model_type, checkpointing_context, pg_collection)
    assert old_bound.arguments["model_provider_func"] is model_type
    assert old_bound.arguments["model_type"] is provider
    assert old_bound.arguments["checkpointing_context"] is checkpointing_context
    assert old_bound.arguments["pg_collection"] is pg_collection
    normalized_type, normalized_provider = _normalize_setup_model_args(
        old_bound.arguments["model_type"], old_bound.arguments["model_provider_func"]
    )
    assert normalized_type is model_type
    assert normalized_provider is provider

    new_bound = signature.bind(
        model_type,
        provider,
        cfg_container=object(),
        pg_collection=pg_collection,
    )
    assert new_bound.arguments["model_type"] is model_type
    assert new_bound.arguments["model_provider_func"] is provider
    assert new_bound.arguments["pg_collection"] is pg_collection


def test_upcycling_dense_config_does_not_mutate_moe_config():
    transformer = Namespace(
        num_moe_experts=8,
        expert_model_parallel_size=2,
        ffn_hidden_size=256,
    )
    model_config = Namespace(transformer=transformer)

    dense = _dense_model_config_for_upcycling(model_config, 256, 2)

    assert (transformer.num_moe_experts, transformer.expert_model_parallel_size) == (8, 2)
    assert transformer.ffn_hidden_size == 256
    assert dense.transformer.num_moe_experts is None
    assert dense.transformer.expert_model_parallel_size == 1
    assert dense.transformer.ffn_hidden_size == 512



def test_upcycling_legacy_provider_build(monkeypatch):
    expected = object()
    calls = []

    def fake_get_model(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(training, "get_model", fake_get_model)
    provider = object()
    model_type = ModelType.encoder_or_decoder
    assert _build_legacy_dense_model_for_upcycling(provider, model_type) is expected
    assert calls == [
        (
            (provider, model_type),
            {},
        )
    ]

class TestModel(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        bias: bool,
        shared_embedding: bool = False,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, output_dim, bias) for _ in range(num_layers)]
        )
        if shared_embedding:
            self.layers[-1].weight.shared_embedding = True


class Utils:

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    inited = False
    store = None

    @staticmethod
    def initialize_distributed():

        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)

        if not torch.distributed.is_initialized() and Utils.rank >= 0:
            print(
                f'Initializing torch.distributed with rank: {Utils.rank}, '
                f'world_size: {Utils.world_size}'
            )
            torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
            init_method = 'tcp://'
            master_ip = os.getenv('MASTER_ADDR', 'localhost')
            master_port = os.getenv('MASTER_PORT', '23450')
            init_method += master_ip + ':' + master_port
            rendezvous_iterator = rendezvous(
                init_method, Utils.rank, Utils.world_size, timeout=timedelta(minutes=1)
            )
            store, rank, world_size = next(rendezvous_iterator)
            store.set_timeout(timedelta(minutes=1))

            # Use a PrefixStore to avoid accidental overrides of keys used by
            # different systems (e.g. RPC) in case the store is multi-tenant.
            store = PrefixStore("default_pg", store)
            Utils.store = store

            torch.distributed.init_process_group(
                backend='nccl', world_size=Utils.world_size, rank=Utils.rank, store=store
            )

            torch.distributed.barrier()
        Utils.inited = True

    @staticmethod
    def set_world_size(world_size=None, rank=None):
        Utils.world_size = torch.cuda.device_count() if world_size is None else world_size
        if (
            torch.distributed.is_initialized()
            and Utils.world_size != torch.distributed.get_world_size()
        ):
            torch.distributed.destroy_process_group()

        if rank is None:
            Utils.rank = int(os.environ['LOCAL_RANK'])
            if Utils.rank >= Utils.world_size:
                Utils.rank = -1
        else:
            Utils.rank = rank

    @staticmethod
    def destroy_model_parallel():
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        if not Utils.inited:
            return

        try:
            # Flush pending CUDA work before the barrier so slow ranks don't
            # time out while fast ranks tear down process groups.
            torch.cuda.synchronize()
            torch.distributed.barrier(timeout=timedelta(seconds=300))
        except Exception:
            Utils.inited = False
            return
        ps.destroy_model_parallel()
        Utils.inited = False

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        **kwargs,
    ):
        # Need to unset these variables to make sure previous
        # tests setting them doesn't interfere current test.
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)

        ps.destroy_model_parallel()
        Utils.initialize_distributed()
        ps.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            **kwargs,
        )
        Utils.inited = True

    @staticmethod
    def pretrain_config_from_global_args(args: Namespace, model_class: Literal["gpt", "hybrid"]):
        if model_class == "gpt":
            model_cfg = gpt_config_from_args(args)
        elif model_class == "hybrid":
            model_cfg = hybrid_config_from_args(args)
        else:
            raise ValueError(
                f"MCore model type {model_class} not supported. Choose one of 'gpt' or 'hybrid'."
            )

        return pretrain_cfg_container_from_args(args, model_cfg)

    @staticmethod
    def fake_initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        expert_model_parallel_size=1,
    ):
        """Used for layer-wise UT as a proxy for NeMo-style intialization."""
        ps.set_tensor_model_parallel_world_size(tensor_model_parallel_size)
        ps.set_tensor_model_parallel_rank(0)

        ps.set_expert_model_parallel_world_size(expert_model_parallel_size)
        ps.set_expert_model_parallel_rank(0)
        if virtual_pipeline_model_parallel_size is not None:
            ps.set_virtual_pipeline_model_parallel_world_size(virtual_pipeline_model_parallel_size)
        ps.set_virtual_pipeline_model_parallel_rank(0)

        ps.set_pipeline_model_parallel_world_size(pipeline_model_parallel_size)
