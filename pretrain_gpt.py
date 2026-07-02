# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT GPT."""

# Capture the true program start time BEFORE any heavy imports.
import time
_PROGRAM_START_TIME = time.time()

import json

# Suppress warnings on all ranks but rank 0.
import os
import warnings
rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from gpt_builders import gpt_builder
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import get_attr_wrapped_model, get_thd_batch_on_this_cp_rank, get_batch_on_this_hybrid_cp_rank, StragglerDetector
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank, get_mtp_ranks
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()

def _ensure_complete_batch_on_all_tp_ranks(batch, device):
    """Broadcast all batch fields from TP-rank-0 to sibling TP ranks.

    When PP > 1, get_batch_on_this_tp_rank leaves some fields as None on
    non-zero TP ranks (e.g. tokens on the last PP stage).  This ensures every TP
    rank has the full batch so they can all compute packed_seq_params
    independently.

    Collective: all TP ranks must call.  No-op when TP = 1.
    """
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return

    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_src = dist.get_global_rank(tp_group, 0)

    fields_and_dtypes = [
        ("tokens", torch.long), ("labels", torch.long),
        ("loss_mask", torch.float), ("position_ids", torch.long),
    ]

    # Step 1: broadcast shape from TP-rank-0 (which always has all fields).
    shape_buf = torch.zeros(2, device=device, dtype=torch.int64)
    if tp_rank == 0:
        ref = next(v for v in [batch.get(k) for k, _ in fields_and_dtypes] if v is not None)
        shape_buf[0] = ref.shape[0]
        shape_buf[1] = ref.shape[1] if ref.dim() > 1 else 1
    dist.broadcast(shape_buf, src=tp_src, group=tp_group)
    shape = (int(shape_buf[0].item()), int(shape_buf[1].item()))

    # Step 2: broadcast ALL fields unconditionally.  All TP ranks must
    # participate in every broadcast (it's a collective).
    for key, dtype in fields_and_dtypes:
        val = batch.get(key)
        if val is None:
            val = torch.empty(shape, device=device, dtype=dtype)
        if not val.is_contiguous():
            val = val.contiguous()
        dist.broadcast(val, src=tp_src, group=tp_group)
        batch[key] = val


def _packed_seq_params_from_batch(batch, args, tokenizer, cp_size, device):
    """Compute PackedSeqParams from a local batch, applying CP padding if needed.

    cu_seqlens are derived from EOD token boundaries plus fixed seq_length
    intervals.  When CP > 1, this modifies the batch in-place (padding tokens,
    labels, loss_mask, position_ids to lengths divisible by 2*CP*(TP if SP),
    then slicing for DualChunkSwap).

    Returns (packed_seq_params, modified_batch).
    """
    qkv_format = 'thd'

    # MBS>1: fold (mbs, seq) sequence tensors into one (1, mbs*seq) THD stream
    if batch["tokens"].shape[0] > 1:
        for _k in ("tokens", "labels", "loss_mask", "position_ids"):
            if batch.get(_k) is not None:
                batch[_k] = batch[_k].reshape(1, -1)

    tokens_full_flat = batch["tokens"].view(1, -1)
    total_tokens = tokens_full_flat.size(-1)

    # Step 1a: compute cu_seqlens from EOD boundaries + fixed seq_length intervals
    cu_seq, _ = torch.sort(torch.unique(torch.cat((
        torch.arange(0, total_tokens + args.seq_length, args.seq_length, device=device, dtype=torch.int32),
        (tokens_full_flat.flatten() == tokenizer.eod).nonzero()[:, 0].int() + 1)))
        )
    cu_seq = cu_seq[cu_seq <= total_tokens]

    # Step 1b: merge sub-sequences that are too short for CP
    if cp_size > 1:
        from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
            pad_thd_sequences_for_cp,
            generate_positional_ids_for_cp,
            get_batch_on_this_cp_rank as te_get_batch_on_this_cp_rank,
        )

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        sp_enabled = getattr(args, 'sequence_parallel', False)

        _divisibility = 2 * cp_size * (tp_size if sp_enabled else 1)
        _seq_lens = cu_seq[1:] - cu_seq[:-1]
        _keep = torch.cat([
            torch.tensor([True], device=device),
            _seq_lens >= _divisibility,
        ])
        cu_seq = cu_seq[_keep]

        divisibility = 2 * cp_size * (tp_size if sp_enabled else 1)

        cp_group = parallel_state.get_context_parallel_group()
        cp_rank = parallel_state.get_context_parallel_rank()

        # Step 2a. Pad every sub-sequence so its length is divisible by 2*cp_size(*tp)
        input_ids_padded, labels_padded, cu_seqlens_padded = pad_thd_sequences_for_cp(
            batch["tokens"].view(-1).cpu(),
            batch["labels"].view(-1).cpu(),
            cu_seq.cpu(),
            divisibility_factor=divisibility,
            padding_token_id=tokenizer.eod,
            padding_label_id=-100,
        )
        input_ids_padded = input_ids_padded.to(device)
        labels_padded = labels_padded.to(device)
        cu_seqlens_padded = cu_seqlens_padded.to(device=device, dtype=torch.int32)

        assert (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]).min() % divisibility == 0, \
            "Some padded sequence lengths are not divisible by 2*cp_size after merging."

        # Step 2b. Pad loss_mask (no TE utility exists for this)
        loss_mask_flat = batch["loss_mask"].view(-1)
        seqlens = cu_seq[1:] - cu_seq[:-1]
        padding_amounts = [
            ((l.item() + divisibility - 1) // divisibility) * divisibility - l.item()
            for l in seqlens
        ]
        loss_mask_seqs = [
            loss_mask_flat[cu_seq[i]:cu_seq[i + 1]]
            for i in range(len(cu_seq) - 1)
        ]
        loss_mask_padded = torch.cat([
            torch.cat([seq, torch.zeros(pad, dtype=seq.dtype, device=seq.device)])
            if pad > 0 else seq
            for seq, pad in zip(loss_mask_seqs, padding_amounts)
        ])

        # Step 2c. Generate position_ids for padded sequences
        position_ids_padded = generate_positional_ids_for_cp(
            cu_seq.cpu(), divisibility, dtype=batch["position_ids"].dtype
        ).to(device)

        # Step 3a: DualChunkSwap slicing
        input_ids_padded, labels_padded, position_ids_padded = (
            te_get_batch_on_this_cp_rank(
                cu_seqlens_padded,
                input_ids_padded,
                labels_padded,
                position_ids_padded,
                cp_group=cp_group,
                qvk_format='thd',
            )
        )

        # Step 3b: select loss_mask slices for this CP rank
        total_slices = 2 * cp_size
        slice_sizes = (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]) // total_slices
        cp_rank_indices = []
        for slice_size, seq_start in zip(slice_sizes, cu_seqlens_padded[:-1]):
            cp_rank_indices.append(torch.arange(
                seq_start + cp_rank * slice_size,
                seq_start + (cp_rank + 1) * slice_size,
                device=device,
            ))
            cp_rank_indices.append(torch.arange(
                seq_start + (total_slices - cp_rank - 1) * slice_size,
                seq_start + (total_slices - cp_rank) * slice_size,
                device=device,
            ))
        loss_mask_padded = loss_mask_padded.index_select(0, torch.cat(cp_rank_indices))

        batch["tokens"] = input_ids_padded.unsqueeze(0)
        batch["labels"] = labels_padded.unsqueeze(0)
        batch["loss_mask"] = loss_mask_padded.unsqueeze(0)
        batch["position_ids"] = position_ids_padded.unsqueeze(0)
        batch["attention_mask"] = None

        max_padded_len = int((cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]).max().item())
        assert max_padded_len % divisibility == 0, \
            f"max_padded_len {max_padded_len} not divisible by {divisibility}"

        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens_padded,
            cu_seqlens_kv=cu_seqlens_padded,
            max_seqlen_q=max_padded_len,
            max_seqlen_kv=max_padded_len,
            qkv_format=qkv_format,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
        )

    else:
        # CP = 1: use the already-computed cu_seq, no padding needed.
        max_len = (cu_seq[1:] - cu_seq[:-1]).max()
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seq,
            cu_seqlens_kv=cu_seq,
            max_seqlen_q=max_len,
            max_seqlen_kv=max_len,
            qkv_format=qkv_format,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
        )
        for key in ["tokens", "labels", "loss_mask", "position_ids"]:
            if batch.get(key) is not None:
                batch[key] = batch[key].view(1, -1)
        batch["attention_mask"] = None

    return packed_seq_params, batch


def _compute_and_broadcast_packed_seq_params(batch, args, device):
    """Compute packed_seq_params on TP-rank-0 and broadcast to sibling TP ranks.

    Used for middle PP stages where only TP-rank-0 has meaningful batch data and
    the batch itself is discarded (activations arrive via PP P2P).

    Returns packed_seq_params on all TP ranks.
    """
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    cp_size = parallel_state.get_context_parallel_world_size()
    tokenizer = get_tokenizer()

    # TP-rank-0 computes; others will receive via broadcast.
    packed_seq_params = None
    if tp_rank == 0:
        packed_seq_params, _ = _packed_seq_params_from_batch(
            batch, args, tokenizer, cp_size, device
        )

    # Broadcast packed_seq_params to other TP ranks.
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return packed_seq_params

    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_src_global = dist.get_global_rank(tp_group, 0)

    # Fixed-size buffer: [n_elements, max_seqlen, cu_seqlens...]
    buf_size = args.seq_length + 2
    buf = torch.zeros(buf_size, dtype=torch.int32, device=device)

    if tp_rank == 0:
        cu_sq = packed_seq_params.cu_seqlens_q
        msv = packed_seq_params.max_seqlen_q
        n = cu_sq.shape[0]
        buf[0] = n
        buf[1] = int(msv.item() if hasattr(msv, "item") else msv)
        buf[2:2 + n] = cu_sq.to(dtype=torch.int32, device=device)

    dist.broadcast(buf, src=tp_src_global, group=tp_group)

    if tp_rank != 0:
        n = int(buf[0].item())
        max_seqlen = int(buf[1].item())
        cu_seqlens = buf[2:2 + n].clone()
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format='thd',
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
        )

    return packed_seq_params


def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a batch.

    Two independent packed-sequence mechanisms are supported:

    1. ``--use-packed-seq-params`` (pre-training or post-training): cross-document
       (xdoc) attention masking where ``cu_seqlens`` is COMPUTED from EOD token
       boundaries inside this function (see ``_packed_seq_params_from_batch``).
       All PP stages build a dataloader on TP-rank-0 (see
       ``is_dataset_built_on_rank``); middle stages compute ``packed_seq_params``
       from their local dataloader and broadcast it to sibling TP ranks.

    2. ``--sft`` THD packing: the dataset emits ``cu_seqlens`` / ``max_seqlen``
       directly; CP slicing is delegated to ``get_thd_batch_on_this_cp_rank``.
       This is the original behaviour and is unchanged.

    Returns a 6-tuple
    ``(tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params)``.
    """
    args = get_args()
    config = core_transformer_config_from_args(args)

    use_xdoc = getattr(args, 'use_packed_seq_params', False)
    is_sft_packed = args.sft  # SFT always uses dataset-emitted THD packing
    is_mtp = mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
    is_first_last = is_first_or_last_pipeline_stage(vp_stage)

    # Path 1: xdoc packing via --use-packed-seq-params
    if use_xdoc:

        device = torch.cuda.current_device()

        # Middle PP stages (and not MTP): compute packed_seq_params from the
        # local dataloader on TP-rank-0 and broadcast to sibling TP ranks.
        if not is_first_last and not is_mtp:
            batch = get_batch_on_this_tp_rank(
                data_iterator,
                mtp_on_this_rank=is_mtp,
            )
            packed_seq_params = _compute_and_broadcast_packed_seq_params(batch, args, device)
            return None, None, None, None, None, packed_seq_params

        # First / last PP stages (and MTP ranks): all TP ranks compute the same
        # packed_seq_params from identical data.
        batch = get_batch_on_this_tp_rank(
            data_iterator,
            mtp_on_this_rank=is_mtp,
        )

        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if pp_size > 1 and tp_size > 1:
            _ensure_complete_batch_on_all_tp_ranks(batch, device)

        cp_size = parallel_state.get_context_parallel_world_size()
        tokenizer = get_tokenizer()
        packed_seq_params, batch = _packed_seq_params_from_batch(
            batch, args, tokenizer, cp_size, device
        )

        # Build the 6-tuple explicitly by key (do NOT rely on dict order).
        return (
            batch.get("tokens"),
            batch.get("labels"),
            batch.get("loss_mask"),
            batch.get("attention_mask"),
            batch.get("position_ids"),
            packed_seq_params,
        )

    # Path 2: original behaviour (SFT THD packing or plain dense batches)
    is_packed_sequence = is_sft_packed  # SFT always uses packed sequence
    if not is_first_last and not is_packed_sequence and not is_mtp:
        return None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(
        data_iterator,
        mtp_on_this_rank=is_mtp
        )

    cu_seqlens = batch.pop('cu_seqlens', None)
    cu_seqlens_padded = batch.pop('cu_seqlens_padded', None)
    max_seqlen = batch.pop('max_seqlen', None)
    local_cp_size = batch.pop('local_cp_size', None)
    if local_cp_size is not None:
        local_cp_size = int(local_cp_size.item())

    if cu_seqlens is not None:
        assert (
            cu_seqlens.dim() == 2 and cu_seqlens.shape[0] == 1
        ), "cu_seqlens must be (1, N) after flatten_batch_for_packed_sequences"
        cu_seqlens = cu_seqlens[0]
        assert max_seqlen.dim() == 1

    # For middle pipeline stages with packed sequences, only cu_seqlens and
    # max_seqlen are needed (for attention masking); skip the full batch.
    if not is_first_last and is_packed_sequence:
        return None, None, None, None, None, PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=int(max_seqlen[0].item()),
            max_seqlen_kv=int(max_seqlen[0].item()),
            qkv_format='thd',
        )

    if cu_seqlens is None and local_cp_size is None:
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)  # The implementation of this function is in MCore
        packed_seq_params = None
    elif local_cp_size is None:  # Packed THD format
        batch, packed_seq_params = get_thd_batch_on_this_cp_rank(batch, cu_seqlens, cu_seqlens_padded, max_seqlen)
    else: # Hybrid CP format
        batch, packed_seq_params = get_batch_on_this_hybrid_cp_rank(batch, local_cp_size)

    return (*batch.values(), packed_seq_params)


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()


    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator, vp_stage)
    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask,
                    packed_seq_params=packed_seq_params,
                )

                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, packed_seq_params=packed_seq_params
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None, is_packed_sequence=False):
    args = get_args()
    config = core_transformer_config_from_args(args)
    if parallel_state.get_tensor_model_parallel_rank() != 0:
        return False
    # When xdoc packing (--use-packed-seq-params) or SFT THD packing is enabled,
    # ALL PP stages build a dataloader on TP-rank-0.  Middle stages use it only
    # to compute cu_seqlens locally; they still receive activations via PP P2P.
    elif getattr(args, 'use_packed_seq_params', False) or is_packed_sequence:
        return True
    return (
        is_first_or_last_pipeline_stage(vp_stage)
        or mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
    )


def core_gpt_dataset_config_from_args(args):
    tokenizer = build_tokenizer(args)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    data_args = {
        "random_seed": args.seed,
        "sequence_length": args.seq_length,
        "blend": blend,
        "blend_per_split": blend_per_split,
        "split": args.split,
        "multiple_validation_sets": args.multiple_validation_sets,
        "full_validation": args.full_validation,
        "num_dataset_builder_threads": args.num_dataset_builder_threads,
        "path_to_cache": args.data_cache_path,
        "mmap_bin_files": args.mmap_bin_files,
        "tokenizer": tokenizer,
        "reset_position_ids": args.reset_position_ids,
        "reset_attention_mask": args.reset_attention_mask,
        "eod_mask_loss": args.eod_mask_loss,
        "create_attention_mask": args.create_attention_mask_in_dataloader,
        "object_storage_cache_path": args.object_storage_cache_path,
        "mid_level_dataset_surplus": args.mid_level_dataset_surplus,
        "allow_ambiguous_pad_tokens": args.allow_ambiguous_pad_tokens,
        "fast_cache_load": args.dataloader_fast_cache_load,
        "sequences_per_dataset": sequences_per_dataset,
        "defer_npy_index_mmap": args.dataloader_defer_npy_index_mmap,
        "context_parallel_size": args.context_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "sequence_parallel_size": args.tensor_model_parallel_size*args.sequence_parallel,
        "hybrid_context_parallel": args.hybrid_context_parallel,
        "pretraining_packing_strategy": args.pretraining_packing_strategy,
        "max_docs_per_bin": args.max_docs_per_bin,
    }

    # add FIM args to the config
    if args.fim_data:
        extra_tokens = {
            "prefix": args.fim_prefix_token,
            "middle": args.fim_middle_token,
            "suffix": args.fim_suffix_token,
            "pad": args.fim_pad_token,
            "eod": args.fim_eod_token,
        }
        data_args.update(
            {
                "fim_rate": args.fim_rate,
                "fim_spm_rate": args.fim_spm_rate,
                "fim_extra_tokens": extra_tokens,
                "fim_split_sample": args.fim_split_sample,
                "fim_fragment_rate": args.fim_fragment_rate,
                "fim_no_prefix": args.fim_no_prefix,
            }
        )
        return GPTFIMDatasetConfig(**data_args)

    return GPTDatasetConfig(**data_args)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)


    is_packed_sequence = False
    if args.sft:
        dataset_type = SFTDataset
        is_packed_sequence = True  # SFT always uses packed sequence
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        elif args.fim_data:
            dataset_type = GPTFIMDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    is_dataset_built = partial(is_dataset_built_on_rank, vp_stage=vp_stage, is_packed_sequence=is_packed_sequence)
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        config = core_transformer_config_from_args(args)
        mtp_ranks = get_mtp_ranks(pp_ranks, config)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )