# Copyright (c) 2026, ETH Zurich / Swiss AI Initiative.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
import megatron.core.ssm.kimi_delta_attention as kda_module
from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.config import InferenceConfig, KDAInferenceStateConfig
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.kimi_delta_attention import (
    HAVE_KDA,
    KimiDeltaAttention,
    fused_recurrent_kda_fwd,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(
    not HAVE_KDA or fused_recurrent_kda_fwd is None,
    reason="The installed FLA does not provide dynamic KDA kernels.",
)
@pytest.mark.internal
class TestKimiDeltaAttentionInference:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(123)

        config = TransformerConfig(
            hidden_size=128,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=2,
            activation_func=F.silu,
            bf16=True,
            experimental_attention_variant="kda",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        pg_collection = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )
        submodules = get_experimental_attention_variant_module_spec(config=config).submodules
        cls.kda = KimiDeltaAttention(
            config,
            submodules=submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=pg_collection,
        ).cuda().bfloat16()
        cls.kda.eval()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_packed_prefill_and_indexed_decode_match_full_sequence(self):
        """Packed prompts plus one recurrent step must match full-sequence KDA."""
        torch.manual_seed(123)
        device = torch.cuda.current_device()
        hidden_size = self.kda.config.hidden_size
        prompt_lengths = [5, 3]
        prompts = [
            torch.randn(length, 1, hidden_size, device=device, dtype=torch.bfloat16)
            for length in prompt_lengths
        ]
        next_tokens = torch.randn(2, 1, hidden_size, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            reference_prompt_outputs = [self.kda(prompt, None)[0] for prompt in prompts]
            reference_decode_outputs = [
                self.kda(torch.cat((prompt, next_tokens[i : i + 1]), dim=0), None)[0][-1]
                for i, prompt in enumerate(prompts)
            ]

            packed_prompt = torch.cat(prompts, dim=0)
            projected_prompt, _ = self.kda.in_proj(packed_prompt)
            cu_seqlens_list = [0, prompt_lengths[0], sum(prompt_lengths)]
            cu_seqlens = torch.tensor(
                cu_seqlens_list, dtype=torch.int32, device=device
            )
            seq_idx = torch.tensor(
                [0] * prompt_lengths[0] + [1] * prompt_lengths[1],
                dtype=torch.int32,
                device=device,
            )
            seq_start = torch.tensor(
                [0] * prompt_lengths[0] + [prompt_lengths[0]] * prompt_lengths[1],
                dtype=torch.int32,
                device=device,
            )
            batch_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
            metadata = SimpleNamespace(
                real_prefill_token_count=sum(prompt_lengths),
                batch_indices_prefill=batch_indices,
                cu_seqlens=cu_seqlens,
                cu_seqlens_list=cu_seqlens_list,
                conv_seq_idx=seq_idx,
                conv_seq_start=seq_start,
            )
            context = SimpleNamespace(
                kda_metadata=metadata,
                batch_dimensions=SimpleNamespace(prefill_req_count=2),
            )

            conv_shape, recurrent_shape = self.kda.kda_state_shapes_per_request()
            conv_state = torch.zeros(
                (3,) + conv_shape, dtype=torch.bfloat16, device=device
            )
            recurrent_state = torch.zeros(
                (3,) + recurrent_shape, dtype=torch.float32, device=device
            )

            packed_inner_output = self.kda._dynamic_inference_prefill(
                projected_prompt, context, conv_state, recurrent_state
            )
            packed_output, _ = self.kda.out_proj(packed_inner_output)

            start = 0
            for length, reference in zip(prompt_lengths, reference_prompt_outputs):
                torch.testing.assert_close(
                    packed_output[start : start + length], reference, atol=2e-2, rtol=2e-2
                )
                start += length

            padded_next_tokens = torch.cat(
                (
                    next_tokens,
                    torch.zeros(1, 1, hidden_size, dtype=torch.bfloat16, device=device),
                ),
                dim=0,
            )
            projected_decode, _ = self.kda.in_proj(padded_next_tokens)
            decode_indices = torch.tensor([0, 1, -1], dtype=torch.int32, device=device)
            decode_inner_output = self.kda._dynamic_inference_decode(
                projected_decode,
                conv_state,
                recurrent_state,
                decode_indices,
                dummy_state_idx=2,
            )
            decode_output, _ = self.kda.out_proj(decode_inner_output)

            for i, reference in enumerate(reference_decode_outputs):
                torch.testing.assert_close(
                    decode_output[i], reference, atol=3e-2, rtol=3e-2
                )

    def test_mixed_decode_and_packed_prefill_match_dense_sequences(self):
        """A decode request and a newly admitted prompt may share one engine step."""
        torch.manual_seed(321)
        device = torch.cuda.current_device()
        hidden_size = self.kda.config.hidden_size
        prompt_a = torch.randn(4, 1, hidden_size, device=device, dtype=torch.bfloat16)
        next_a = torch.randn(1, 1, hidden_size, device=device, dtype=torch.bfloat16)
        prompt_b = torch.randn(3, 1, hidden_size, device=device, dtype=torch.bfloat16)

        conv_shape, recurrent_shape = self.kda.kda_state_shapes_per_request()
        conv_state = torch.zeros((3,) + conv_shape, dtype=torch.bfloat16, device=device)
        recurrent_state = torch.zeros(
            (3,) + recurrent_shape, dtype=torch.float32, device=device
        )

        def prefill_metadata(length, state_idx):
            cu_seqlens = torch.tensor([0, length], dtype=torch.int32, device=device)
            metadata = SimpleNamespace(
                real_prefill_token_count=length,
                batch_indices_prefill=torch.tensor(
                    [state_idx], dtype=torch.int32, device=device
                ),
                cu_seqlens=cu_seqlens,
                cu_seqlens_list=[0, length],
                conv_seq_idx=torch.zeros(length, dtype=torch.int32, device=device),
                conv_seq_start=torch.zeros(length, dtype=torch.int32, device=device),
            )
            return SimpleNamespace(
                kda_metadata=metadata,
                batch_dimensions=SimpleNamespace(prefill_req_count=1),
            )

        with torch.inference_mode():
            projected_a, _ = self.kda.in_proj(prompt_a)
            self.kda._dynamic_inference_prefill(
                projected_a,
                prefill_metadata(len(prompt_a), 0),
                conv_state,
                recurrent_state,
            )

            reference_a = self.kda(torch.cat((prompt_a, next_a)), None)[0][-1]
            reference_b = self.kda(prompt_b, None)[0]

            mixed_hidden = torch.cat((next_a, prompt_b))
            metadata = prefill_metadata(len(prompt_b), 1).kda_metadata
            metadata.batch_indices_decode = torch.tensor(
                [0], dtype=torch.int32, device=device
            )
            metadata.device_decode_prefill = torch.tensor(
                [1, len(prompt_b)], dtype=torch.int32, device=device
            )
            context = SimpleNamespace(
                kda_metadata=metadata,
                kda_dummy_state_idx=2,
                padded_batch_dimensions=SimpleNamespace(
                    decode_req_count=1,
                    prefill_req_count=1,
                    token_count=len(mixed_hidden),
                ),
                batch_dimensions=SimpleNamespace(prefill_req_count=1),
                padding_slice=slice(len(mixed_hidden), len(mixed_hidden)),
                kda_states_cache=lambda _layer_number: (conv_state, recurrent_state),
            )
            output, _ = self.kda._dynamic_inference(mixed_hidden, context)

        torch.testing.assert_close(output[0], reference_a, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(output[1:], reference_b, atol=2e-2, rtol=2e-2)

    def test_short_continuation_prefill_preserves_conv_history(self):
        """A continuation shorter than the convolution window must retain prior inputs."""
        device = torch.cuda.current_device()
        hidden = torch.randn(
            7,
            1,
            self.kda.config.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        conv_shape, recurrent_shape = self.kda.kda_state_shapes_per_request()
        conv_state = torch.zeros((1,) + conv_shape, dtype=torch.bfloat16, device=device)
        recurrent_state = torch.zeros(
            (1,) + recurrent_shape, dtype=torch.float32, device=device
        )

        outputs = []
        with torch.no_grad():
            reference = self.kda(hidden, None)[0]
            for chunk in (hidden[:5], hidden[5:]):
                projected, _ = self.kda.in_proj(chunk)
                length = len(chunk)
                metadata = SimpleNamespace(
                    batch_indices_prefill=torch.zeros(1, dtype=torch.int32, device=device),
                    cu_seqlens=torch.tensor([0, length], dtype=torch.int32, device=device),
                    conv_seq_idx=torch.zeros(length, dtype=torch.int32, device=device),
                    conv_seq_start=torch.zeros(length, dtype=torch.int32, device=device),
                )
                inner = self.kda._dynamic_inference_prefill(
                    projected,
                    SimpleNamespace(kda_metadata=metadata),
                    conv_state,
                    recurrent_state,
                )
                outputs.append(self.kda.out_proj(inner)[0])

        torch.testing.assert_close(torch.cat(outputs), reference, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("use_upstream_state_extraction", [True, False])
    def test_packed_prefill_cuda_graph_replay(
        self, monkeypatch, use_upstream_state_extraction
    ):
        """Packed prefill reads replay-time boundaries and updates indexed states."""
        if use_upstream_state_extraction:
            if kda_module.causal_conv1d_varlen_states is None:
                pytest.skip("causal-conv1d varlen state extraction is unavailable")
        else:
            monkeypatch.setattr(kda_module, "causal_conv1d_varlen_states", None)
        device = torch.cuda.current_device()
        hidden_size = self.kda.config.hidden_size
        padded_token_count = 12
        real_token_count = 8

        projected = torch.zeros(
            padded_token_count,
            1,
            sum(self.kda._in_proj_sharded_split()[0]),
            dtype=torch.bfloat16,
            device=device,
        )
        batch_indices = torch.tensor([0, 1, -1], dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor([0, 5, 8, 8], dtype=torch.int32, device=device)
        conv_seq_idx = torch.zeros(padded_token_count, dtype=torch.int32, device=device)
        conv_seq_start = torch.zeros_like(conv_seq_idx)
        metadata = SimpleNamespace(
            batch_indices_prefill=batch_indices,
            cu_seqlens=cu_seqlens,
            conv_seq_idx=conv_seq_idx,
            conv_seq_start=conv_seq_start,
        )
        context = SimpleNamespace(kda_metadata=metadata)

        conv_shape, recurrent_shape = self.kda.kda_state_shapes_per_request()
        conv_state = torch.zeros((3,) + conv_shape, dtype=torch.bfloat16, device=device)
        recurrent_state = torch.zeros(
            (3,) + recurrent_shape, dtype=torch.float32, device=device
        )

        def set_inputs(prompts):
            packed = torch.cat(prompts)
            projected_real, _ = self.kda.in_proj(packed)
            projected.zero_()
            projected[:real_token_count].copy_(projected_real)
            lengths = [len(prompt) for prompt in prompts]
            cu_seqlens.copy_(
                torch.tensor([0, lengths[0], sum(lengths), sum(lengths)], device=device)
            )
            conv_seq_idx.zero_()
            conv_seq_idx[:real_token_count].copy_(
                torch.repeat_interleave(
                    torch.arange(2, dtype=torch.int32, device=device),
                    torch.tensor(lengths, device=device),
                )
            )
            conv_seq_start.zero_()
            conv_seq_start[lengths[0] : real_token_count] = lengths[0]

        warmup_prompts = [
            torch.randn(5, 1, hidden_size, dtype=torch.bfloat16, device=device),
            torch.randn(3, 1, hidden_size, dtype=torch.bfloat16, device=device),
        ]
        with torch.no_grad():
            set_inputs(warmup_prompts)
            self.kda._dynamic_inference_prefill(
                projected, context, conv_state, recurrent_state
            )
            conv_state.zero_()
            recurrent_state.zero_()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_output = self.kda._dynamic_inference_prefill(
                    projected, context, conv_state, recurrent_state
                )

            replay_prompts = [
                torch.randn(4, 1, hidden_size, dtype=torch.bfloat16, device=device),
                torch.randn(4, 1, hidden_size, dtype=torch.bfloat16, device=device),
            ]
            set_inputs(replay_prompts)
            conv_state.zero_()
            recurrent_state.zero_()
            graph.replay()
            replay_output, _ = self.kda.out_proj(captured_output)
            references = [self.kda(prompt, None)[0] for prompt in replay_prompts]

        torch.testing.assert_close(replay_output[:4], references[0], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(replay_output[4:8], references[1], atol=2e-2, rtol=2e-2)
        assert conv_state[2].count_nonzero() == 0
        assert recurrent_state[2].count_nonzero() == 0

    def test_dynamic_context_assigns_and_releases_kda_slots(self):
        """The scheduler must keep prompt padding separate from live KDA state slots."""
        conv_shape, recurrent_shape = self.kda.kda_state_shapes_per_request()
        detected_config = KDAInferenceStateConfig.from_model(
            SimpleNamespace(
                config=self.kda.config,
                decoder=SimpleNamespace(
                    layers=[
                        SimpleNamespace(self_attention=self.kda),
                        SimpleNamespace(self_attention=SimpleNamespace(layer_number=2)),
                    ]
                ),
            )
        )
        assert detected_config.kda_layer_map == {0: 0}
        assert detected_config.attention_layer_map == {1: 0}
        assert detected_config.conv_states_shape == conv_shape
        assert detected_config.recurrent_states_shape == recurrent_shape

        kda_config = KDAInferenceStateConfig(
            kda_layer_map={0: 0},
            attention_layer_map={1: 0},
            conv_states_shape=conv_shape,
            recurrent_states_shape=recurrent_shape,
            conv_states_dtype=torch.bfloat16,
            recurrent_states_dtype=torch.float32,
        )
        context = DynamicInferenceContext(
            model_config=TransformerConfig(
                params_dtype=torch.bfloat16,
                num_layers=2,
                kv_channels=16,
                num_attention_heads=2,
            ),
            inference_config=InferenceConfig(
                max_sequence_length=32,
                block_size_tokens=8,
                buffer_size_gb=0.01,
                paused_buffer_size_gb=0.002,
                max_tokens=64,
                max_requests=8,
                num_cuda_graphs=None,
                use_cuda_graphs_for_non_decode_steps=True,
                use_flashinfer_fused_rope=None,
                unified_memory_level=0,
                kda_inference_state_config=kda_config,
            ),
        )

        prompts = [
            torch.tensor([11, 12, 13, 14, 15], dtype=torch.long, device="cuda"),
            torch.tensor([21, 22, 23], dtype=torch.long, device="cuda"),
        ]
        for request_id, prompt in enumerate(prompts):
            context.add_request(
                DynamicInferenceRequest(
                    request_id=request_id,
                    prompt_tokens=prompt,
                    sampling_params=SamplingParams(num_tokens_to_generate=2),
                )
            )

        context.initialize_attention_state()
        slots = context.kda_metadata.request_to_mamba_state_idx[:2].clone()
        assert torch.all(slots >= 0)
        assert slots[0] != slots[1]
        torch.testing.assert_close(
            context.kda_metadata.batch_indices_prefill[:2], slots, rtol=0, atol=0
        )
        assert context.kda_metadata.cu_seqlens_list == [0, 5, 8]
        torch.testing.assert_close(
            context.token_to_input_ids[:8], torch.cat(prompts), rtol=0, atol=0
        )
        assert context.kda_metadata.batch_indices_prefill[2:].eq(-1).all()
        assert context.use_cuda_graphs_for_non_decode_steps
        assert context.kda_conv_states.shape[:2] == (1, context.max_requests + 1)
        assert context.kda_recurrent_states.shape[:2] == (1, context.max_requests + 1)

        hidden_states = torch.zeros(
            context.padded_active_token_count,
            1,
            self.kda.config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        hidden_states[:8].normal_()
        with torch.no_grad():
            reference_prefill = [
                self.kda(hidden_states[:5], None)[0],
                self.kda(hidden_states[5:8], None)[0],
            ]
            output, _ = self.kda(hidden_states, None, inference_context=context)
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output[:8]).all()
        torch.testing.assert_close(output[:5], reference_prefill[0], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(output[5:8], reference_prefill[1], atol=2e-2, rtol=2e-2)
        assert context.kda_conv_states[0, slots].abs().sum() > 0
        assert context.kda_recurrent_states[0, slots].abs().sum() > 0
        assert context.kda_conv_states[0, context.kda_dummy_state_idx].count_nonzero() == 0
        assert context.kda_recurrent_states[0, context.kda_dummy_state_idx].count_nonzero() == 0

        next_hidden = torch.randn(
            2, 1, self.kda.config.hidden_size, dtype=torch.bfloat16, device="cuda"
        )
        with torch.no_grad():
            reference_decode = [
                self.kda(torch.cat((hidden_states[:5], next_hidden[:1])), None)[0][-1],
                self.kda(torch.cat((hidden_states[5:8], next_hidden[1:2])), None)[0][-1],
            ]
        context.update_requests(
            active_requests_mask=torch.ones(2, dtype=torch.int32, device="cuda"),
            new_tokens=torch.tensor([31, 32], dtype=torch.long, device="cuda"),
        )
        context.initialize_attention_state()
        torch.testing.assert_close(
            context.kda_metadata.batch_indices_decode[:2], slots, rtol=0, atol=0
        )
        assert context.kda_metadata.batch_indices_decode[2:].eq(-1).all()

        padded_decode_hidden = torch.zeros(
            context.padded_active_token_count,
            1,
            self.kda.config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        padded_decode_hidden[:2] = next_hidden
        with torch.no_grad():
            decode_output, _ = self.kda(
                padded_decode_hidden, None, inference_context=context
            )
        torch.testing.assert_close(decode_output[0], reference_decode[0], atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(decode_output[1], reference_decode[1], atol=3e-2, rtol=3e-2)

        context.update_requests(
            active_requests_mask=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            new_tokens=torch.tensor([41, 42], dtype=torch.long, device="cuda"),
        )
        assert context.total_request_count == 1
        assert context.kda_metadata.request_to_mamba_state_idx[0] == slots[1]
        assert context.kda_metadata.request_to_mamba_state_idx[1] == -1

        context.release_memory_blocks_from_request_indexes(
            torch.tensor([0], dtype=torch.long, device="cuda")
        )
        assert context.kda_metadata.request_to_mamba_state_idx[0] == -1
        assert context.kda_metadata.mamba_state_free_slot_count == context.max_requests

        context.reset()
        assert context.kda_metadata.mamba_state_free_slot_count == context.max_requests

        context.cuda_graph_batch_dimensions_list = [
            InferenceBatchDimensions(token_count=2, decode_req_count=2)
        ]
        context.add_dummy_requests_for_expert_parallel_step()
        dummy_slots = context.kda_metadata.request_to_mamba_state_idx[:2]
        assert context.mamba_metadata is None
        assert (dummy_slots >= 0).all()
        assert dummy_slots.unique().numel() == 2

    def test_dynamic_context_rejects_kda_context_parallelism(self):
        conv_shape, recurrent_shape = self.kda.kda_state_shapes_per_request()
        with pytest.raises(AssertionError, match="requires CP=1"):
            DynamicInferenceContext(
                model_config=TransformerConfig(
                    params_dtype=torch.bfloat16,
                    num_layers=1,
                    kv_channels=16,
                    num_attention_heads=2,
                    context_parallel_size=2,
                ),
                inference_config=InferenceConfig(
                    max_sequence_length=32,
                    block_size_tokens=8,
                    buffer_size_gb=0.01,
                    kda_inference_state_config=KDAInferenceStateConfig(
                        kda_layer_map={0: 0},
                        attention_layer_map={},
                        conv_states_shape=conv_shape,
                        recurrent_states_shape=recurrent_shape,
                        conv_states_dtype=torch.bfloat16,
                        recurrent_states_dtype=torch.float32,
                    ),
                ),
            )

    def test_indexed_decode_cuda_graph_replay(self):
        """Decode graphs must read updated request-to-state indices at replay time."""
        device = torch.cuda.current_device()
        hidden_states = torch.randn(
            3,
            1,
            self.kda.config.hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        with torch.no_grad():
            projected, _ = self.kda.in_proj(hidden_states)

        conv_shape, recurrent_shape = self.kda.kda_state_shapes_per_request()
        graph_conv_state = torch.zeros(
            (3,) + conv_shape, dtype=torch.bfloat16, device=device
        )
        graph_recurrent_state = torch.zeros(
            (3,) + recurrent_shape, dtype=torch.float32, device=device
        )
        batch_indices = torch.tensor([1, 0, -1], dtype=torch.int32, device=device)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream), torch.no_grad():
            for _ in range(2):
                graph_conv_state.zero_()
                graph_recurrent_state.zero_()
                self.kda._dynamic_inference_decode(
                    projected,
                    graph_conv_state,
                    graph_recurrent_state,
                    batch_indices,
                    dummy_state_idx=2,
                )
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph_conv_state.zero_()
        graph_recurrent_state.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            graph_output = self.kda._dynamic_inference_decode(
                projected,
                graph_conv_state,
                graph_recurrent_state,
                batch_indices,
                dummy_state_idx=2,
            )

        # Change the mapping after capture, as DynamicInferenceContext does between steps.
        batch_indices.copy_(torch.tensor([0, 1, -1], dtype=torch.int32, device=device))
        eager_conv_state = torch.zeros_like(graph_conv_state)
        eager_recurrent_state = torch.zeros_like(graph_recurrent_state)
        with torch.no_grad():
            eager_output = self.kda._dynamic_inference_decode(
                projected,
                eager_conv_state,
                eager_recurrent_state,
                batch_indices,
                dummy_state_idx=2,
            )

        graph_conv_state.zero_()
        graph_recurrent_state.zero_()
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_output, eager_output, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(graph_conv_state, eager_conv_state, atol=0, rtol=0)
        torch.testing.assert_close(
            graph_recurrent_state, eager_recurrent_state, atol=2e-3, rtol=2e-3
        )
