# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding
from megatron.core.utils import get_pg_size
from megatron.training.checkpointing import save_grads
from megatron.training.global_vars import set_args
from megatron.training.training import build_train_valid_test_data_iterators, get_model
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def mock_train_valid_test_datasets_provider(train_val_test_num_samples):
    return iter([1]), iter([2]), iter([3])


def create_test_args():
    # Set dummy values for the args.
    args = SimpleNamespace()
    args.iteration = 0
    args.train_samples = 1
    args.train_iters = 1
    args.eval_interval = 1
    args.eval_iters = 1
    args.global_batch_size = 1
    args.consumed_train_samples = 1
    args.consumed_valid_samples = 1
    args.dataloader_type = "external"
    args.skip_train = False
    args.full_validation = False
    args.multiple_validation_sets = False
    args.perform_rl_step = False
    args.phase_transition_iterations = None

    return args


class TestTraining:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        args = create_test_args()
        set_args(args)

    def test_build_train_valid_test_data_iterators(self):
        train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(
            mock_train_valid_test_datasets_provider
        )
        train_data = next(train_iter)
        valid_data = next(valid_iter)
        test_data = next(test_iter)
        assert (train_data, valid_data, test_data) == (1, 2, 3)

    def test_closed_formula_vocab_size_with_padding(self):
        def old_round_impl(after, multiple):
            while (after % multiple) != 0:
                after += 1
            return after

        args = SimpleNamespace()
        args.rank = 0
        args.tensor_model_parallel_size = 1

        for vocab in range(1, 600000, 1000):
            for mult in [1, 17, 32, 64, 128]:
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == vocab_size_with_padding(vocab, args, False), (
                    vocab,
                    mult,
                )

        for vocab in range(1, 10_000, 500):
            for mult in range(1, 1024 + 1):
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == vocab_size_with_padding(vocab, args, False), (
                    vocab,
                    mult,
                )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestLegacyLayoutProcessGroups:
    """Legacy DDP layout uses the MPU groups that DDP resolves with no collection."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1, num_distributed_optimizer_instances=2)

    def test_legacy_dist_opt_layout_uses_ddp_mpu_groups(self):
        """Legacy layout follows DDP's MPU groups, even if the DDP count is stale."""
        args = SimpleNamespace(
            init_model_with_meta_device=True,
            virtual_pipeline_model_parallel_size=None,
            use_torch_fsdp2=False,
            use_megatron_fsdp=False,
            fp16=False,
            bf16=False,
            use_distributed_optimizer=True,
            overlap_param_gather_with_optimizer_step=False,
            data_parallel_random_init=False,
        )
        set_args(args)

        external_pg = Mock()
        for name in ("dp", "cp", "tp", "pp"):
            getattr(external_pg, name).rank.return_value = 0
        external_pg.pp.size.return_value = 1
        external_pg.intra_dp_cp.size.return_value = 123
        external_pg.intra_expt_dp.size.return_value = 456
        external_pg.dp_cp.size.return_value = 789
        external_pg.expt_dp.size.return_value = 654

        stream_context = MagicMock()
        stream_context.__enter__.return_value = None
        stream_context.__exit__.return_value = False
        for num_instances in (1, 2):
            ddp_config = DistributedDataParallelConfig(
                use_distributed_optimizer=True,
                num_distributed_optimizer_instances=num_instances,
                overlap_grad_reduce=True,
                bucket_size=128,
            )
            with (
                patch("megatron.training.training.has_nvidia_modelopt", False),
                patch(
                    "megatron.training.training.get_megatron_ddp_config", return_value=ddp_config
                ),
                patch("megatron.training.training.get_model_config", return_value=Mock()),
                patch("megatron.training.training.get_pg_rank", return_value=0),
                patch("megatron.training.training.correct_amax_history_if_needed"),
                patch(
                    "megatron.training.training.to_empty_if_meta_device",
                    side_effect=lambda m, **_: m,
                ),
                patch(
                    "megatron.training.training.tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes"
                ),
                patch.object(
                    DistributedOptimizer, "compute_full_param_layout", return_value="LAYOUT"
                ) as layout,
                patch("megatron.training.training.DDP", return_value=Mock()) as ddp,
                patch("torch.cuda.Stream", return_value=Mock()),
                patch("torch.cuda.current_stream") as current_stream,
                patch("torch.cuda.stream", return_value=stream_context),
            ):
                current_stream.return_value.wait_stream = Mock()
                get_model(
                    lambda **_: torch.nn.Linear(2, 2), wrap_with_ddp=True, pg_collection=external_pg
                )

            resolved_groups = ProcessGroupCollection.setup_process_groups_for_ddp(
                None, SimpleNamespace(context_parallel_size=1), ddp_config
            )
            layout_call = layout.call_args
            if num_instances == 1:
                assert get_pg_size(resolved_groups["intra_dp_cp_group"]) < get_pg_size(
                    resolved_groups["dp_cp_group"]
                )
                assert get_pg_size(resolved_groups["intra_expt_dp_group"]) < get_pg_size(
                    resolved_groups["expt_dp_group"]
                )
            assert layout_call.args[2] == get_pg_size(resolved_groups["intra_dp_cp_group"])
            assert layout_call.kwargs["expert_data_parallel_world_size"] == get_pg_size(
                resolved_groups["intra_expt_dp_group"]
            )
            assert layout_call.args[2] != external_pg.intra_dp_cp.size.return_value
            assert (
                layout_call.kwargs["expert_data_parallel_world_size"]
                != external_pg.intra_expt_dp.size.return_value
            )
            assert ddp.call_args.kwargs["full_param_layout"] == "LAYOUT"
            assert "pg_collection" not in ddp.call_args.kwargs

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestSaveGrads:
    """Tests for the save_grads function."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_save_grads(self, tmp_path_dist_ckpt):
        """Test that save_grads creates the correct directory structure and saves
        state_dict correctly.

        With TP=1, PP=1 on 8 GPUs, we have 8 DP ranks. Only the rank with
        expert_data_parallel_rank==0 should save. All ranks verify the result.
        """
        save_dir = str(tmp_path_dist_ckpt / "test_save_grads")

        with TempNamedDir(save_dir, sync=True) as save_dir:
            # Create a mock state_dict with gradients (use deterministic values for reproducibility).
            state_dict = defaultdict(dict)
            state_dict["model_chunk0"]["layer.weight"] = torch.arange(16).reshape(4, 4).float()
            state_dict["model_chunk0"]["layer.bias"] = torch.arange(4).float()

            iteration = 100
            grad_label = "wgrads"

            # All ranks call save_grads, but only expert_data_parallel_rank==0 actually saves.
            save_grads(save_dir, dict(state_dict), iteration, grad_label)

            # Synchronize before checking results since only rank 0 saves.
            torch.distributed.barrier()

            # All ranks verify the file was created by rank 0.
            expected_dir = Path(save_dir) / grad_label / f"iter_{iteration:07d}"
            assert expected_dir.exists(), f"Expected directory {expected_dir} to exist"

            expected_file = expected_dir / "mp_rank_00.pth"
            assert expected_file.exists(), f"Expected file {expected_file} to exist"

            # Verify saved content.
            loaded = torch.load(expected_file)
            assert "model_chunk0" in loaded
            assert "layer.weight" in loaded["model_chunk0"]
            assert "layer.bias" in loaded["model_chunk0"]
            assert torch.equal(
                loaded["model_chunk0"]["layer.weight"], state_dict["model_chunk0"]["layer.weight"]
            )
            assert torch.equal(
                loaded["model_chunk0"]["layer.bias"], state_dict["model_chunk0"]["layer.bias"]
            )
