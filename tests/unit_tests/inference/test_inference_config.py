# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
from argparse import ArgumentParser
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from megatron.core.inference.config import (
    AsyncScheduleMode,
    ImageProcessingConfig,
    InferenceConfig,
    VideoProcessingConfig,
)
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.moe import InferenceGroupedGemmBackend
from megatron.core.inference.quantization.utils import resolve_mxfp8_backend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import inference_cfg_from_args
from megatron.training.arguments import _add_inference_args
from megatron.training.config.inference_config import InferenceSetupConfig


class TestInferenceConfig:

    @pytest.mark.parametrize("cache_bytes", [None, 0])
    def test_text_only_media_defaults(self, cache_bytes):
        config = InferenceConfig(
            image_preprocessing_config=None,
            video_preprocessing_config=None,
            vision_embedding_cache_max_bytes=cache_bytes,
            allow_stale_multimodal_embeddings=False,
        )
        assert config.image_preprocessing_config is None
        assert config.video_preprocessing_config is None
        assert config.vision_embedding_cache_max_bytes == cache_bytes
        assert config.allow_stale_multimodal_embeddings is False
        assert dataclasses.replace(config) == config
        assert dataclasses.asdict(config)["image_preprocessing_config"] is None
        assert GPTInferenceWrapper.multimodal_prompt_config is None

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("image_preprocessing_config", {}),
            ("video_preprocessing_config", {}),
            ("vision_embedding_cache_max_bytes", 1),
            ("vision_embedding_cache_max_bytes", -1),
            ("allow_stale_multimodal_embeddings", True),
        ],
    )
    def test_media_settings_fail_explicitly(self, field, value):
        with pytest.raises(NotImplementedError, match=field):
            InferenceConfig(**{field: value})

    def test_image_config_rejects_nemo_constructor(self):
        with pytest.raises(NotImplementedError, match="Image preprocessing"):
            ImageProcessingConfig(
                patch_dim=16,
                dynamic_resolution=True,
                use_tiling=False,
                pixel_shuffle=True,
                spatial_merge_size=2,
                dynamic_resolution_min_patches=1,
                dynamic_resolution_max_patches=128,
                vision_model_type="radio",
                pixel_mean=[0.5, 0.5, 0.5],
                pixel_std=[0.5, 0.5, 0.5],
            )

    def test_video_config_rejects_nemo_constructor(self):
        with pytest.raises(NotImplementedError, match="Video preprocessing"):
            VideoProcessingConfig(
                image_config=cast(ImageProcessingConfig, None),
                num_frames=8,
                temporal_patch_size=2,
                frame_manifest_magic=b"frames",
                video_maintain_aspect_ratio=False,
            )

    @pytest.mark.parametrize(
        ("grouped_gemm_backend", "expected_backend"),
        [
            ("torch", "triton"),
            (InferenceGroupedGemmBackend.TORCH, "triton"),
            ("flashinfer", "flashinfer"),
            (InferenceGroupedGemmBackend.FLASHINFER, "flashinfer"),
        ],
    )
    def test_resolve_mxfp8_backend(self, grouped_gemm_backend, expected_backend):
        assert resolve_mxfp8_backend(grouped_gemm_backend) == expected_backend

    @pytest.mark.parametrize(
        "grouped_gemm_backend",
        [
            "te",
            InferenceGroupedGemmBackend.TE,
            "vllm",
            InferenceGroupedGemmBackend.VLLM,
            "unsupported-backend",
        ],
    )
    def test_resolve_mxfp8_backend_rejects_unsupported_backend(self, grouped_gemm_backend):
        with pytest.raises(ValueError, match="does not support inference_grouped_gemm_backend"):
            resolve_mxfp8_backend(grouped_gemm_backend)

    def test_mutual_exclusivity_with_transformer_config(self):
        """
        Ensure mutual exclusivity between fields in `InferenceConfig` and
        `TransformerConfig`.
        """
        dynamic_inference_config_fields = set(dataclasses.fields(InferenceConfig))
        transformer_config_fields = set(dataclasses.fields(TransformerConfig))
        assert len(dynamic_inference_config_fields.intersection(transformer_config_fields)) == 0

    def test_verbose_is_init_only_and_not_serialized_or_compared(self):
        quiet = InferenceConfig()
        verbose = InferenceConfig(verbose=True)

        assert verbose._verbose is True
        assert quiet._verbose is False
        field_names = {field.name for field in dataclasses.fields(InferenceConfig)}
        assert "verbose" not in field_names
        assert "_verbose" not in field_names
        assert "verbose" not in dataclasses.asdict(verbose)
        assert quiet == verbose

    def test_setup_config_passes_verbose_to_runtime_config(self):
        model = SimpleNamespace(
            position_embedding_type="rotary",
            max_sequence_length=2560,
            pg_collection=None,
            config=SimpleNamespace(params_dtype=torch.float16),
            decoder=SimpleNamespace(layer_type_list=None, layers=[]),
        )

        runtime_config = InferenceSetupConfig().to_inference_config(model, verbose=True)

        assert runtime_config._verbose is True

    @pytest.mark.parametrize(
        "async_sched_mode, expected",
        [
            (None, AsyncScheduleMode.LEGACY),
            ("legacy", AsyncScheduleMode.LEGACY),
            (AsyncScheduleMode.LEGACY, AsyncScheduleMode.LEGACY),
            ("async", AsyncScheduleMode.ASYNC),
            (AsyncScheduleMode.ASYNC, AsyncScheduleMode.ASYNC),
        ],
    )
    def test_async_sched_mode_default_and_coercion(self, async_sched_mode, expected):
        """Ensure async scheduling mode defaults to legacy and accepts strings."""
        kwargs = {} if async_sched_mode is None else {"async_sched_mode": async_sched_mode}
        assert InferenceConfig(**kwargs).async_sched_mode == expected

    @pytest.mark.parametrize("invalid_mode", ["serial", "overlap", "invalid"])
    def test_async_sched_mode_rejects_invalid_value(self, invalid_mode):
        """Ensure invalid async scheduling modes fail during config construction."""
        with pytest.raises(ValueError):
            InferenceConfig(async_sched_mode=invalid_mode)

    def test_routing_alpha_accepts_values_above_one(self):
        """Alpha penalizes relative load and is not a bounded blend weight."""
        assert InferenceConfig(prefix_caching_routing_alpha=5.0).prefix_caching_routing_alpha == 5.0

    def test_routing_alpha_must_be_non_negative(self):
        with pytest.raises(ValueError, match="prefix_caching_routing_alpha"):
            InferenceConfig(prefix_caching_routing_alpha=-0.1)

    def test_async_sched_argparse_plumbing(self):
        parser = _add_inference_args(ArgumentParser())
        assert parser.parse_args([]).inference_dynamic_batching_async_sched_mode == "legacy"
        args = parser.parse_args(["--inference-dynamic-batching-async-sched-mode", "async"])
        assert args.inference_dynamic_batching_async_sched_mode == "async"
        assert inference_cfg_from_args(args).inference_dynamic_batching_async_sched_mode == "async"

    @pytest.mark.parametrize("invalid_mode", ["serial", "overlap"])
    def test_async_sched_argparse_rejects_removed_modes(self, invalid_mode):
        """Ensure the CLI rejects removed async scheduling modes."""
        parser = _add_inference_args(ArgumentParser())
        with pytest.raises(SystemExit):
            parser.parse_args(["--inference-dynamic-batching-async-sched-mode", invalid_mode])

    def test_inference_setup_config_maps_async_sched_mode(self):
        model = SimpleNamespace(
            position_embedding_type="rotary",
            max_sequence_length=2560,
            pg_collection=None,
            config=SimpleNamespace(params_dtype=torch.float16),
            decoder=SimpleNamespace(layer_type_list=None, layers=[]),
        )
        setup_config = InferenceSetupConfig(inference_dynamic_batching_async_sched_mode="async")

        runtime_config = setup_config.to_inference_config(model, verbose=False)

        assert runtime_config.async_sched_mode == AsyncScheduleMode.ASYNC

    @pytest.mark.parametrize("backend", ["UCX", "UCCL"])
    def test_nixl_backend_argparse_to_runtime_config(self, backend):
        parser = _add_inference_args(ArgumentParser())
        assert parser.parse_args([]).inference_nixl_backend == "UCX"
        args = parser.parse_args(["--inference-nixl-backend", backend])
        setup_config = inference_cfg_from_args(args)
        model = SimpleNamespace(
            position_embedding_type="rotary",
            max_sequence_length=2560,
            pg_collection=None,
            config=SimpleNamespace(params_dtype=torch.float16),
            decoder=SimpleNamespace(layer_type_list=None, layers=[]),
        )

        runtime_config = setup_config.to_inference_config(model, verbose=False)

        assert setup_config.inference_nixl_backend == backend
        assert runtime_config.nixl_backend == backend
        assert InferenceConfig().nixl_backend == "UCX"

    def test_nixl_backend_rejects_auto_selection(self):
        parser = _add_inference_args(ArgumentParser())
        with pytest.raises(SystemExit):
            parser.parse_args(["--inference-nixl-backend", "auto"])
        with pytest.raises(ValueError, match="nixl_backend"):
            InferenceConfig(nixl_backend="auto")

    def test_offset_sampling_seed_argparse_plumbing(self):
        """Ensure the CLI can select a shared sampling seed across DP ranks."""
        parser = _add_inference_args(ArgumentParser())
        default_args = parser.parse_args([])
        assert default_args.offset_sampling_seed_by_dp_rank is True

        disabled_args = parser.parse_args(["--use-same-sampling-seed-across-dp-ranks"])
        assert disabled_args.offset_sampling_seed_by_dp_rank is False

    def test_inference_setup_config_maps_offset_sampling_seed_by_dp_rank(self):
        """Ensure declarative inference config maps DP seed offset to runtime config."""
        model = SimpleNamespace(
            position_embedding_type="rope",
            max_sequence_length=4096,
            pg_collection="pg",
            decoder=SimpleNamespace(layer_type_list=None),
        )
        # Match the decoder/config interface also inspected by KDA state discovery.
        model.decoder.layers = []
        model.config = SimpleNamespace()
        setup_config = InferenceSetupConfig(offset_sampling_seed_by_dp_rank=False)

        inference_config = setup_config.to_inference_config(
            model=model,
            kv_cache_management_mode="persist",
            static_kv_memory_pointers=False,
            enable_cuda_graphs=False,
            verbose=False,
        )

        assert inference_config.offset_sampling_seed_by_dp_rank is False
