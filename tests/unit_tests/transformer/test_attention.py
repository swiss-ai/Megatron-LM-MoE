# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import copy
import types
from unittest import mock

import pytest
import torch
from packaging import version

import megatron.core.parallel_state as parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank as get_tensor_on_this_cp_rank,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.attention import HAVE_FA3, Attention, SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.utils import is_te_min_version, unwrap_model
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from megatron.training.training import get_model
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
)
from tests.unit_tests.test_utilities import Utils

try:
    from transformer_engine.pytorch.attention.rope import apply_fused_qkv_rotary_pos_emb

    HAVE_FUSED_QKV_ROPE = True
except ImportError:
    HAVE_FUSED_QKV_ROPE = False


def test_materialize_paged_kv_cache_in_varlen_order():
    kv_cache = torch.arange(8).reshape(4, 2, 1)
    block_table = torch.tensor([[2, 0, -1], [3, 1, -1]])
    sequence_lengths = torch.tensor([3, 4])

    materialized = Attention._materialize_paged_kv_cache(
        kv_cache, block_table, sequence_lengths
    )

    assert torch.equal(
        materialized.squeeze(-1), torch.tensor([4, 5, 0, 6, 7, 2, 3])
    )


def test_dynamic_flash_attention_2_prefill_forwards_window_size():
    attention = mock.Mock()
    attention.training = False
    attention.batch_invariant_mode = False
    attention.softmax_scale = None
    attention._get_flash_attention_window_size.return_value = (1024, 0)
    q = torch.zeros(2, 1, 1, 4)
    k = v = torch.zeros(2, 1, 4)

    with mock.patch("megatron.core.transformer.attention.HAVE_FA3", False), mock.patch(
        "megatron.core.transformer.attention.flash_attn_varlen_func",
        return_value=torch.zeros(2, 1, 4),
    ) as flash_attention:
        Attention.flash_decode_and_prefill(
            attention,
            q,
            k,
            v,
            2,
            2,
            torch.tensor([0, 2]),
            torch.tensor([0, 2]),
            torch.tensor([2]),
            torch.tensor([[0]]),
            False,
        )

    assert flash_attention.call_args.kwargs["window_size"] == (1024, 0)


@pytest.mark.parametrize(
    ("have_fa3", "kernel_name"),
    (
        (False, "flash_attn_with_kvcache"),
        (True, "flash_attn3_with_kvcache"),
    ),
    ids=("fa2", "fa3"),
)
def test_dynamic_flash_attention_decode_forwards_window_size(have_fa3, kernel_name):
    attention = mock.Mock()
    attention.training = False
    attention.batch_invariant_mode = False
    attention._get_flash_attention_window_size.return_value = (1024, 0)
    q = torch.zeros(1, 1, 1, 4)
    k = v = torch.zeros(1, 64, 1, 4)

    with mock.patch(
        "megatron.core.transformer.attention.HAVE_FA3", have_fa3
    ), mock.patch(
        f"megatron.core.transformer.attention.{kernel_name}",
        return_value=torch.zeros_like(q),
    ) as flash_attention:
        Attention.flash_decode_and_prefill(
            attention,
            q,
            k,
            v,
            1,
            2,
            torch.tensor([0, 1]),
            torch.tensor([0, 2]),
            torch.tensor([2]),
            torch.tensor([[0]]),
            True,
        )

    assert flash_attention.call_args.kwargs["window_size"] == (1024, 0)


@pytest.mark.skipif(not HAVE_FA3, reason="FlashAttention-3 is required")
@torch.inference_mode()
def test_mla_flash_attention_2_padding_matches_flash_attention_3():
    """FA2's padded MLA values must match FA3's native unequal-head-dim result."""
    pytest.importorskip("flash_attn")

    torch.manual_seed(1234)
    length = 257
    page_size = 64
    num_pages = (length + page_size - 1) // page_size
    num_heads = 4
    qk_dim = 192
    value_dim = 128
    dtype = torch.bfloat16
    device = torch.device("cuda")

    attention = mock.Mock()
    attention.training = False
    attention.batch_invariant_mode = False
    attention.softmax_scale = qk_dim**-0.5
    attention.config = MLATransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=num_heads,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=value_dim,
    )
    attention._get_flash_attention_window_size.return_value = (-1, -1)
    attention._flash_attention_3_forward_wrapper = types.MethodType(
        Attention._flash_attention_3_forward_wrapper, attention
    )

    query = torch.randn(length, 1, num_heads, qk_dim, device=device, dtype=dtype) * 0.1
    paged_key = (
        torch.randn(
            num_pages, page_size, num_heads, qk_dim, device=device, dtype=dtype
        )
        * 0.1
    )
    paged_value = (
        torch.randn(
            num_pages, page_size, num_heads, value_dim, device=device, dtype=dtype
        )
        * 0.1
    )
    contiguous_key = paged_key.flatten(0, 1)[:length]
    contiguous_value = paged_value.flatten(0, 1)[:length]
    cu_seqlens = torch.tensor([0, length], dtype=torch.int32, device=device)
    sequence_lengths = torch.tensor([length], dtype=torch.int32, device=device)
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)

    with mock.patch("megatron.core.transformer.attention.HAVE_FA3", False):
        fa2_output = Attention.flash_decode_and_prefill(
            attention,
            query,
            contiguous_key,
            contiguous_value,
            length,
            length,
            cu_seqlens,
            cu_seqlens,
            sequence_lengths,
            None,
            False,
        )

    with mock.patch("megatron.core.transformer.attention.HAVE_FA3", True):
        fa3_output = Attention.flash_decode_and_prefill(
            attention,
            query,
            paged_key,
            paged_value,
            length,
            length,
            cu_seqlens,
            cu_seqlens,
            sequence_lengths,
            block_table,
            False,
        )

    assert fa2_output.shape == fa3_output.shape == (length, 1, num_heads, value_dim)
    torch.testing.assert_close(
        fa2_output.float(),
        fa3_output.float(),
        atol=2e-3,
        rtol=2e-2,
    )


@pytest.mark.parametrize("output_gate", [False, True])
class TestParallelAttention:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, output_gate):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_output_gate=output_gate,
        )
        self.parallel_attention = SelfAttention(
            self.transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
        )

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.parallel_attention, SelfAttention)
        assert self.parallel_attention.layer_number == 1

        num_weights = sum([p.numel() for p in self.parallel_attention.parameters()])
        if self.transformer_config.attention_output_gate:
            assert num_weights == 82816
        else:
            assert num_weights == 66304

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self):

        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 2

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size),
            dtype=torch.bfloat16,
        )
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((micro_batch_size, 1, 1, sequence_length), dtype=bool).cuda()

        output, bias = self.parallel_attention(hidden_states, attention_mask)

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    @pytest.mark.skipif(not is_te_min_version("1.4.0"), reason="Fused RoPE requires TE >= 1.4.0")
    @pytest.mark.parametrize("rotary_interleaved", [True, False])
    @pytest.mark.parametrize("fused_qkv_rope", [True, False])
    def test_fused_rope_gpu_forward(self, rotary_interleaved, fused_qkv_rope):
        self.parallel_attention.config.apply_rope_fusion = True
        if rotary_interleaved and not is_te_min_version("2.3.0"):
            pytest.skip("Only TE >= 2.3.0 supports interleaved fused RoPE.")
        if fused_qkv_rope and self.parallel_attention.config.attention_output_gate:
            pytest.skip("Fused QKV RoPE does not support gated attention for now.")
        if fused_qkv_rope and not HAVE_FUSED_QKV_ROPE:
            pytest.skip("Fused QKV RoPE not available.")
        self.parallel_attention.config.rotary_interleaved = rotary_interleaved
        self.parallel_attention.config.fused_single_qkv_rope = fused_qkv_rope
        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 2

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size),
            dtype=torch.bfloat16,
        )
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((micro_batch_size, 1, 1, sequence_length), dtype=bool).cuda()
        rotary_pos_emb = torch.ones(
            sequence_length, 1, 1, self.parallel_attention.config.kv_channels
        ).cuda()
        output, bias = self.parallel_attention(
            hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb
        )

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size
        self.parallel_attention.config.apply_rope_fusion = False
        self.parallel_attention.config.rotary_interleaved = False

    def test_checkpointed_gpu_forward(self):
        transformer_config = self.transformer_config
        transformer_config.recompute_granularity = 'selective'
        checkpointed_parallel_attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
        )
        config = checkpointed_parallel_attention.config

        sequence_length = 32
        micro_batch_size = 2

        checkpointed_parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, checkpointed_parallel_attention.config.hidden_size),
            dtype=torch.bfloat16,
        )
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((micro_batch_size, 1, 1, sequence_length), dtype=bool).cuda()

        output, bias = checkpointed_parallel_attention(hidden_states, attention_mask)

        assert config.recompute_granularity == 'selective'
        assert "core_attn" in config.recompute_modules
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size


@pytest.mark.skipif(not is_te_min_version("2.9.0"), reason="QK clipping requires TE >= 2.9.0")
class TestClipQK:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_clip_qk_disabled_raises_error(self):
        """Test that clip_qk raises ValueError when qk_clip is not enabled."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_clip=False,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
        )

        with pytest.raises(ValueError, match="qk_clip option needs to be enabled"):
            attention.clip_qk()

    def test_clip_qk_none_logits_raises_error(self):
        """Test that clip_qk raises ValueError when current_max_attn_logits is None."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
        )

        with pytest.raises(ValueError, match="current_max_attn_logits is None"):
            attention.clip_qk()

    def test_clip_qk_below_threshold_no_update(self):
        """Test that weights are not updated when max logits are below threshold."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
        )
        attention.cuda()

        # Save original weights
        original_weight = attention.linear_qkv.weight.data.clone()

        # Set current_max_attn_logits below threshold
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [50.0, 60.0, 70.0, 80.0], device='cuda'
        )

        # Call clip_qk
        attention.clip_qk()

        # Weights should not be updated
        assert torch.equal(attention.linear_qkv.weight.data, original_weight)
        # current_max_attn_logits should be reset
        assert attention.core_attention.current_max_attn_logits is None

    def test_clip_qk_above_threshold_updates_weights(self):
        """Test that weights are updated when max logits exceed threshold."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
        )
        attention.cuda()

        # Save original weights
        original_weight = attention.linear_qkv.weight.data.clone()

        # Set current_max_attn_logits above threshold
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [150.0, 160.0, 170.0, 180.0], device='cuda'
        )

        # Call clip_qk
        attention.clip_qk()

        # Weights should be updated
        assert not torch.equal(attention.linear_qkv.weight.data, original_weight)
        # current_max_attn_logits should be reset
        assert attention.core_attention.current_max_attn_logits is None

    def test_clip_qk_gqa_configuration(self):
        """Test clip_qk with GQA (Grouped Query Attention) configuration."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            num_query_groups=4,  # GQA with 2 heads per group
            use_cpu_initialization=True,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
        )
        attention.cuda()

        # Save original weights
        original_weight = attention.linear_qkv.weight.data.clone()

        # Set current_max_attn_logits for all heads (8 heads)
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0], device='cuda'
        )

        # Call clip_qk
        attention.clip_qk()

        # Weights should be updated
        assert not torch.equal(attention.linear_qkv.weight.data, original_weight)
        # current_max_attn_logits should be reset
        assert attention.core_attention.current_max_attn_logits is None

    def test_clip_qk_mixed_logits(self):
        """Test clip_qk with mixed logits (some above, some below threshold)."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
        )
        attention.cuda()

        # Save original weights
        original_weight = attention.linear_qkv.weight.data.clone()

        # Set mixed current_max_attn_logits (some above, some below threshold)
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [80.0, 150.0, 90.0, 200.0], device='cuda'
        )

        # Call clip_qk
        attention.clip_qk()

        # Weights should be updated since at least one head exceeds threshold
        assert not torch.equal(attention.linear_qkv.weight.data, original_weight)
        # current_max_attn_logits should be reset
        assert attention.core_attention.current_max_attn_logits is None


@pytest.mark.parametrize("output_gate", [False, True])
class TestSelfAttention:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, output_gate):
        self.output_gate = output_gate
        Utils.destroy_model_parallel()

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def run_self_attention(self, pg_collection):
        tensor_model_parallel_size = torch.distributed.get_world_size(pg_collection.tp)
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            attention_output_gate=self.output_gate,
            tensor_model_parallel_size=tensor_model_parallel_size,
            use_cpu_initialization=False,
        )
        self.self_attention = SelfAttention(
            self.transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pg_collection,
        )

        config = self.self_attention.config
        sequence_length = 127
        micro_batch_size = 2

        self.self_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.self_attention.config.hidden_size),
            device='cuda',
        )
        hidden_states_ref = copy.deepcopy(hidden_states)

        output, bias = self.self_attention(hidden_states, None)
        assert config.recompute_granularity is None
        # Check if output and bias have the correct shape
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    @pytest.mark.internal
    def test_self_attention_mpu(self):

        tp_size = 4
        cp_size = 2
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        model_parallel_cuda_manual_seed(123)

        # Get TP and CP process groups from device mesh
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()

        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        self.run_self_attention(pg_collection)

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.internal
    def test_self_attention_independent_pg_smoke(self):

        tp_size = 4
        cp_size = 2
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        model_parallel_cuda_manual_seed(123)

        # Initialize torch.distributed if not already initialized
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')

        # Create HyperCommGrid with dimensions cp, tp (reversed from device mesh order)
        grid = HyperCommGrid([cp_size, tp_size], ["cp", "tp"])

        # Get TP and CP process groups from HyperCommGrid
        tp_group = grid.create_pg("tp")
        cp_group = grid.create_pg("cp")

        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        self.run_self_attention(pg_collection)


def _test_parallel_attention_correctness(
    transformer_config,
    transformer_layer_spec,
    tmp_path_dist_ckpt,
    atol,
    rtol,
    tp=1,
    sp=False,
    cp=1,
    seed=123,
    sequence_length=256,
    micro_batch_size=4,
):
    # Model initialization function
    def initialize_gpt_model(
        config, pre_process=True, post_process=True, vp_stage=None, pg_collection=None
    ):
        gpt_model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=128,
            max_sequence_length=sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )
        return gpt_model

    # Initialize baseline parallel state
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )

    # Initialize input hidden states
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    input_hidden_states = (
        torch.rand((sequence_length, micro_batch_size, transformer_config.hidden_size))
        .cuda()
        .bfloat16()
        .requires_grad_(True)
    )

    with TempNamedDir(tmp_path_dist_ckpt / 'test_parallel_attn', sync=True) as ckpt_dir:
        # Set argument
        mock_args = parse_args(ignore_unknown_args=True)
        set_args(mock_args)

        # Initialize baseline model
        init_basic_mock_args(mock_args, 1, 1, bf16=True)
        mock_args.context_parallel_size = 1
        mock_args.sequence_parallel = 1
        gpt_model = unwrap_model(get_model(initialize_gpt_model, config=transformer_config))

        # Initialize args and save checkpoint
        init_checkpointing_mock_args(mock_args, ckpt_dir, False)
        mock_args.no_save_optim = True
        mock_args.no_save_rng = True
        mock_args.no_load_optim = True
        mock_args.no_load_rng = True
        save_checkpoint(10, gpt_model, None, None, 0)

        # Calculate baseline output
        attention = gpt_model[0].decoder.layers[0].self_attention
        output_hidden_states_baseline, bias_hidden_states_baseline = attention(
            input_hidden_states, attention_mask=None
        )
        output_hidden_states_baseline.sum().backward()

        # Save baseline output
        input_grad_baseline = input_hidden_states.grad.detach()
        output_hidden_states_baseline = output_hidden_states_baseline.detach()
        bias_hidden_states_baseline = bias_hidden_states_baseline
        if bias_hidden_states_baseline is not None:
            bias_hidden_states_baseline = bias_hidden_states_baseline.detach()
            has_bias = True
        else:
            has_bias = False

        # Initialize parallel model
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=1, context_parallel_size=cp
        )
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        transformer_config.context_parallel_size = cp
        transformer_config.tensor_model_parallel_size = tp
        transformer_config.sequence_parallel = sp
        init_basic_mock_args(mock_args, tp, 1, bf16=True)
        mock_args.context_parallel_size = cp
        mock_args.sequence_parallel = sp
        gpt_model = unwrap_model(get_model(initialize_gpt_model, config=transformer_config))
        with mock.patch('megatron.training.checkpointing.check_checkpoint_args'):
            with mock.patch('megatron.training.checkpointing.update_num_microbatches'):
                load_checkpoint(gpt_model, None, None)

        # Function to get tensor on this tp and cp rank
        cp_group = parallel_state.get_context_parallel_group()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()

        def get_tensor_on_this_rank(tensor):
            if cp > 1:
                tensor = get_tensor_on_this_cp_rank(tensor, 0, cp_group)
            if tp > 1 and sp:
                sp_seg = sequence_length // tp // cp
                tensor = tensor[tp_rank * sp_seg : (tp_rank + 1) * sp_seg]
            return tensor

        # Calculate parallel model output
        input_hidden_states = get_tensor_on_this_rank(input_hidden_states)
        input_hidden_states = input_hidden_states.detach().requires_grad_(True)
        parallel_attention = gpt_model[0].decoder.layers[0].self_attention
        output_hidden_states_parallel, bias_hidden_states_parallel = parallel_attention(
            input_hidden_states, attention_mask=None
        )
        output_hidden_states_parallel.sum().backward()
        input_grad_parallel = input_hidden_states.grad.detach()

        # Check if the output is close
        output_hidden_states_baseline = get_tensor_on_this_rank(output_hidden_states_baseline)
        input_grad_baseline = get_tensor_on_this_rank(input_grad_baseline)

        assert torch.all(
            ~torch.isnan(output_hidden_states_baseline)
        ), "output_hidden_states_baseline contains nan"
        assert torch.all(
            ~torch.isinf(output_hidden_states_baseline)
        ), "output_hidden_states_baseline contains inf"
        assert torch.all(~torch.isnan(input_grad_baseline)), "input_grad_baseline contains nan"
        assert torch.all(~torch.isinf(input_grad_baseline)), "input_grad_baseline contains inf"
        assert torch.all(
            ~torch.isnan(output_hidden_states_parallel)
        ), "output_hidden_states_parallel contains nan"
        assert torch.all(
            ~torch.isinf(output_hidden_states_parallel)
        ), "output_hidden_states_parallel contains inf"
        assert torch.all(~torch.isnan(input_grad_parallel)), "input_grad_parallel contains nan"
        assert torch.all(~torch.isinf(input_grad_parallel)), "input_grad_parallel contains inf"
        if has_bias:
            assert torch.all(
                ~torch.isnan(bias_hidden_states_baseline)
            ), "bias_hidden_states_baseline contains nan"
            assert torch.all(
                ~torch.isinf(bias_hidden_states_baseline)
            ), "bias_hidden_states_baseline contains inf"
            assert torch.all(
                ~torch.isnan(bias_hidden_states_parallel)
            ), "bias_hidden_states_parallel contains nan"
            assert torch.all(
                ~torch.isinf(bias_hidden_states_parallel)
            ), "bias_hidden_states_parallel contains inf"

        torch.testing.assert_close(
            output_hidden_states_baseline,
            output_hidden_states_parallel,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"Mismatch in output_hidden_states: {msg}",
        )
        torch.testing.assert_close(
            input_grad_baseline,
            input_grad_parallel,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"Mismatch in input_grad: {msg}",
        )
        if has_bias:
            torch.testing.assert_close(
                bias_hidden_states_baseline,
                bias_hidden_states_parallel,
                atol=atol,
                rtol=rtol,
                msg=lambda msg: f"Mismatch in bias_hidden_states: {msg}",
            )

        Utils.destroy_model_parallel()


@pytest.mark.parametrize("apply_rope_fusion", [False, True])
@pytest.mark.parametrize(
    ("tp", "sp", "cp"),
    [
        (4, False, 1),  # TP w/o SP
        (4, True, 1),  # TP w/ SP
        (1, False, 4),  # CP
        (2, False, 2),  # CP + TP w/o SP
        (2, True, 2),  # CP + TP w/ SP
    ],
)
@pytest.mark.parametrize("qk_layernorm", [False, True])
@pytest.mark.parametrize("output_gate", [False, True])
def test_parallel_attention_correctness(
    tmp_path_dist_ckpt, apply_rope_fusion, tp, sp, cp, qk_layernorm, output_gate
):
    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        normalization="RMSNorm",
        bf16=True,
        qk_layernorm=qk_layernorm,
        apply_rope_fusion=apply_rope_fusion,
        attention_output_gate=output_gate,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=qk_layernorm)
    atol, rtol = 1e-2, 1e-2

    _test_parallel_attention_correctness(
        transformer_config,
        transformer_layer_spec,
        tmp_path_dist_ckpt,
        atol=atol,
        rtol=rtol,
        tp=tp,
        sp=sp,
        cp=cp,
        seed=123,
        sequence_length=256,
    )


@pytest.mark.parametrize("sp", [True, False])
@pytest.mark.parametrize("output_gate", [False, True])
def test_parallel_attention_correctness_num_query_groups_less_than_tp_size(
    tmp_path_dist_ckpt, sp, output_gate
):
    transformer_config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=8,
        num_query_groups=2,
        normalization="RMSNorm",
        bf16=True,
        attention_output_gate=output_gate,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
    atol, rtol = 1e-2, 1e-2

    _test_parallel_attention_correctness(
        transformer_config,
        transformer_layer_spec,
        tmp_path_dist_ckpt,
        atol=atol,
        rtol=rtol,
        tp=4,
        sp=sp,
        seed=123,
        sequence_length=256,
    )
