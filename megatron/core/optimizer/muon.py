# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron muon optimizer wrapper to handle tensor-parallel."""

import logging
from typing import Any, Callable, Dict, List, Literal, Optional, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_size, log_single_rank

from . import HAVE_EMERGING_OPTIMIZERS, HAVE_EO_V02, _get_param_groups, get_megatron_optimizer
from .layer_wise_optimizer import LayerWiseDistributedOptimizer
from .muon_logging import _gain_log_family
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig, ParamKey

if HAVE_EMERGING_OPTIMIZERS:
    from emerging_optimizers.orthogonalized_optimizers import (
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz_tp
else:
    OrthogonalizedOptimizer = object

if HAVE_EO_V02:
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT


logger = logging.getLogger(__name__)


def get_supported_coefficient_types() -> tuple[str, ...]:
    """Return the coefficient types supported by the installed emerging_optimizers.

    Reads the members of the ``NSCoeffT`` Literal type so that new types
    added upstream are automatically available without code changes here.
    """
    assert (
        HAVE_EO_V02
    ), "emerging_optimizers >= 0.2 is required for NSCoeffT. Please install or upgrade it."
    return get_args(NSCoeffT)  # pylint: disable=possibly-used-before-assignment


def validate_coefficient_type(coefficient_type: str) -> None:
    """Raise ``ValueError`` if *coefficient_type* is not supported."""
    supported = get_supported_coefficient_types() if HAVE_EO_V02 else ("quintic",)
    if coefficient_type not in supported:
        raise ValueError(
            f"Unsupported muon coefficient type '{coefficient_type}'. "
            f"Supported types: {supported}"
        )


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        use_nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        split_fc1: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        is_kda_in_proj_fn: Callable[[torch.Tensor], bool] | None = None,
        is_kv_up_proj_fn: Callable[[torch.Tensor], bool] | None = None,
        kv_up_proj_split_shapes: tuple[int, int] | None = None,
        is_qkv_down_proj_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_down_proj_split_shapes: tuple[int, int] | None = None,
        split_mla_per_head: bool = False,
        is_q_up_proj_fn: Callable[[torch.Tensor], bool] | None = None,
        q_up_proj_head_dim: int | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")
        validate_coefficient_type(coefficient_type)

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, {coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            size = [grad.size(-2), grad.size(-1)]
            if partition_dim is not None:
                size[partition_dim] *= get_pg_size(tp_group)
            mode_value = "duplicated" if mode == "blockwise" else mode
            mode_kwarg = {"tp_mode": mode_value} if HAVE_EO_V02 else {"mode": mode_value}
            ns_kwargs = dict(
                steps=num_ns_steps, tp_group=tp_group, partition_dim=partition_dim, **mode_kwarg
            )
            ns_kwargs["coefficient_type"] = coefficient_type
            # pylint: disable-next=possibly-used-before-assignment
            orth_grad = newton_schulz_tp(grad, **ns_kwargs)
            # pylint: disable-next=possibly-used-before-assignment
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.mode = mode
        self.split_qkv = split_qkv
        self.split_fc1 = split_fc1
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes
        self.is_kda_in_proj_fn = is_kda_in_proj_fn
        self.is_kv_up_proj_fn = is_kv_up_proj_fn
        self.kv_up_proj_split_shapes = kv_up_proj_split_shapes
        self.is_qkv_down_proj_fn = is_qkv_down_proj_fn
        self.qkv_down_proj_split_shapes = qkv_down_proj_split_shapes
        self.split_mla_per_head = split_mla_per_head
        self.is_q_up_proj_fn = is_q_up_proj_fn
        self.q_up_proj_head_dim = q_up_proj_head_dim

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        nesterov_kwarg = (
            {"nesterov": use_nesterov} if HAVE_EO_V02 else {"use_nesterov": use_nesterov}
        )
        super().__init__(
            params,
            lr,
            momentum_beta,
            **nesterov_kwarg,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    def _split_partition_dim(self, p, partition_dim):
        """Return the TP partition dimension that still applies after splitting."""
        if self.split_mla_per_head:
            is_q_up_proj = self.is_q_up_proj_fn is not None and self.is_q_up_proj_fn(p)
            is_kv_up_proj = (
                self.is_kv_up_proj_fn is not None and self.is_kv_up_proj_fn(p)
            )
            if is_q_up_proj or is_kv_up_proj:
                return None
        return partition_dim

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to momentum
                because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            # emerging-optimizers use None instead of -1 to indicate no tensor parallel
            partition_dim = None

        if self.split_qkv and self.is_qkv_fn is not None and self.is_qkv_fn(p):
            # split grouped attention parameters (e.g., QKV, GQA, etc.)
            grad_shape = grad.shape
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, split shapes {self.qkv_split_shapes}',
            )
            assert self.qkv_split_shapes is not None
            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)
            qkv_grads = torch.split(
                grad.view(num_query_groups, sum(self.qkv_split_shapes), -1),
                self.qkv_split_shapes,
                dim=1,
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]

            # Apply Newton-Schulz and scales to each component, concat back
            qkv_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim).view(
                    num_query_groups, -1, grad_shape[-1]
                )
                for g in qkv_grads
            ]
            grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        elif self.split_fc1 and (glu_split_dim := getattr(p, 'glu_split_dim', None)) is not None:
            if grad.size(glu_split_dim) % 2 != 0:
                raise ValueError(
                    f"Fused GLU FC1 dimension {glu_split_dim} must be even, "
                    f"got shape {tuple(grad.shape)}"
                )
            fc1_grads = torch.chunk(grad, 2, dim=glu_split_dim)
            fc1_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim) for g in fc1_grads
            ]
            grad = torch.cat(fc1_grads, dim=glu_split_dim)
        elif (
            self.split_qkv
            and self.is_kda_in_proj_fn is not None
            and self.is_kda_in_proj_fn(p)
        ):
            split_shapes = getattr(p, "kda_split_shapes", None)
            if split_shapes is None:
                raise ValueError("KDA in_proj is missing kda_split_shapes metadata")
            split_shapes = self._local_dim0_split_shapes(
                p, grad, split_shapes, allow_groups=False
            )
            if sum(split_shapes) != grad.size(0):
                raise ValueError(
                    f"KDA split shapes {split_shapes} do not match gradient shape {grad.shape}"
                )
            kda_grads = torch.split(grad, split_shapes, dim=0)
            kda_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim) for g in kda_grads
            ]
            grad = torch.cat(kda_grads, dim=0)
        elif (
            self.split_qkv
            and self.is_qkv_down_proj_fn is not None
            and self.is_qkv_down_proj_fn(p)
        ):
            # MLA fused path: split linear_qkv_down_proj into Q and KV/RoPE blocks.
            # TODO: Validate whether this down-projection split is the right Muon
            # treatment for fused MLA.
            grad_shape = grad.shape
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv_down_proj split grad shape {grad_shape}, '
                f'split shapes {self.qkv_down_proj_split_shapes}',
            )
            assert self.qkv_down_proj_split_shapes is not None
            split_shapes = self._local_dim0_split_shapes(
                p, grad, self.qkv_down_proj_split_shapes, allow_groups=False
            )
            qkv_down_grads = torch.split(grad, split_shapes, dim=0)
            qkv_down_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim)
                for g in qkv_down_grads
            ]
            grad = torch.cat(qkv_down_grads, dim=0)
        elif (
            (self.split_qkv or self.split_mla_per_head)
            and self.is_kv_up_proj_fn is not None
            and self.is_kv_up_proj_fn(p)
        ):
            grad_shape = grad.shape
            log_single_rank(
                logger,
                logging.DEBUG,
                f'kv_up_proj split grad shape {grad_shape}, '
                f'split shapes {self.kv_up_proj_split_shapes}',
            )
            assert self.kv_up_proj_split_shapes is not None
            split_shapes = self._local_dim0_split_shapes(
                p, grad, self.kv_up_proj_split_shapes, allow_groups=True
            )
            if self.split_mla_per_head:
                num_heads = grad_shape[0] // sum(split_shapes)
                heads = grad.view(num_heads, sum(split_shapes), -1).unbind(0)
                kv_grads = [
                    block for head in heads for block in torch.split(head, split_shapes, dim=0)
                ]
                split_partition_dim = self._split_partition_dim(p, partition_dim)
                kv_grads = [
                    self.scaled_orthogonalize_fn(g, tp_group, split_partition_dim)
                    for g in kv_grads
                ]
                blocks_per_head = len(split_shapes)
                heads = [
                    torch.cat(kv_grads[i : i + blocks_per_head], dim=0)
                    for i in range(0, len(kv_grads), blocks_per_head)
                ]
                grad = torch.stack(heads, dim=0).view(grad_shape)
            else:
                num_groups = grad_shape[0] // sum(split_shapes)
                kv_grads = torch.split(
                    grad.view(num_groups, sum(split_shapes), -1),
                    split_shapes,
                    dim=1,
                )
                kv_grads = [g.reshape(-1, grad_shape[-1]) for g in kv_grads]
                kv_grads = [
                    self.scaled_orthogonalize_fn(g, tp_group, partition_dim).view(
                        num_groups, -1, grad_shape[-1]
                    )
                    for g in kv_grads
                ]
                grad = torch.cat(kv_grads, dim=1).view(grad_shape)
        elif (
            self.split_mla_per_head
            and self.is_q_up_proj_fn is not None
            and self.is_q_up_proj_fn(p)
        ):
            grad_shape = grad.shape
            log_single_rank(
                logger,
                logging.DEBUG,
                f'q_up_proj per-head split grad shape {grad_shape}, '
                f'head dim {self.q_up_proj_head_dim}',
            )
            assert self.q_up_proj_head_dim is not None
            num_heads = grad_shape[0] // self.q_up_proj_head_dim
            q_grads = grad.view(num_heads, self.q_up_proj_head_dim, -1).unbind(0)
            split_partition_dim = self._split_partition_dim(p, partition_dim)
            q_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, split_partition_dim) for g in q_grads
            ]
            grad = torch.stack(q_grads, dim=0).view(grad_shape)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad

    def _local_dim0_split_shapes(self, p, x, shapes, allow_groups: bool):
        shapes = tuple(shapes)
        total = sum(shapes)
        dim0 = x.size(0)
        if dim0 == total or (allow_groups and dim0 % total == 0):
            return shapes
        if getattr(p, "partition_dim", None) != 0:
            return shapes

        tp_size = 1
        if self.pg_collection is not None:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, "expert_tp", False)
                else self.pg_collection.tp
            )
            tp_size = get_pg_size(tp_group)
        if tp_size == 1 and total % dim0 == 0:
            tp_size = total // dim0
        if tp_size <= 1:
            raise ValueError(
                f"Cannot infer local split shapes for dim0-sharded projection parameter with "
                f"tensor dim0 {dim0} and global split shapes {shapes}"
            )
        if any(shape % tp_size != 0 for shape in shapes):
            raise ValueError(
                f"Cannot split dim0-sharded projection parameter with global split shapes {shapes} "
                f"over TP size {tp_size}"
            )

        local_shapes = tuple(shape // tp_size for shape in shapes)
        local_total = sum(local_shapes)
        if dim0 != local_total and not (allow_groups and dim0 % local_total == 0):
            raise ValueError(
                f"Local split shapes {local_shapes} do not match tensor dim0 {dim0}"
            )
        return local_shapes


def get_megatron_muon_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """This function is used to get the muon optimizer for the model chunks.
    It is used to get the muon optimizer for the model chunks.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        layer_wise_distributed_optimizer (bool): if true, use layer-wise distributed optimizer.
            Defaults to False.
    """
    # TODO: Mutating config.optimizer is a side effect; clean up after
    # https://github.com/NVIDIA/Megatron-LM/pull/3638 lands.
    # Set the nonlinear optimizer for muon (used for embeddings, biases, norms).
    config.optimizer = config.muon_scalar_optimizer

    if config.muon_scalar_optimizer == 'lion':
        assert HAVE_EO_V02, (
            "Lion optimizer requires emerging_optimizers >= 0.2. "
            "Please upgrade to use --muon-scalar-optimizer lion."
        )
    else:
        assert HAVE_EMERGING_OPTIMIZERS, "Emerging Optimizers is not installed."

    # Dist-opt is not supported due to strong coupling with how DDP init grad buffer
    # In theory we can change DDP to enable use muon and dist-opt-adam together
    if config.use_distributed_optimizer:
        raise Exception('muon with dist optimizer is not supported.')
    # only support bf16 w/o loss scale now
    if config.fp16:
        raise Exception('muon with fp16 is not supported.')

    # before this function receive properly created collection
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    log_single_rank(logger, logging.INFO, f'Setting up emerging optimizer with config {config}')

    # Needed for torch_dist ckpt_format, unlike torch ckpt_format
    # For other emerging optimizers, need to implement init_state_fn as well
    # TODO(boxiangw): Improve usability after optimizer refactor
    # TODO(boxiangw): support precision aware optimizer
    def muon_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    def adam_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    def lion_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data)

    nonlinear_init_state_fn = (
        lion_init_state_fn if config.muon_scalar_optimizer == 'lion' else adam_init_state_fn
    )

    optimizers = []
    # record list of non/linear params
    linear_params = []
    nonlinear_params = []
    for model_chunk in model_chunks:
        # use config to determine qkv split shapes.
        # no need to check tp since tp splits by head and this is per head(group) dimension
        num_attention_heads = model_chunk.config.num_attention_heads
        num_query_groups = model_chunk.config.num_query_groups
        kv_channels = model_chunk.config.kv_channels
        qkv_split_shapes = [
            num_attention_heads // num_query_groups * kv_channels,
            kv_channels,
            kv_channels,
        ]
        mla_config = model_chunk.config
        is_mla = getattr(mla_config, 'multi_latent_attention', False)
        if is_mla:
            # MLA views KV as [num_heads, qk_head_dim + v_head_dim], so these
            # shapes describe the K/V layout within each repeated head group.
            kv_up_proj_split_shapes = (mla_config.qk_head_dim, mla_config.v_head_dim)
            q_up_proj_head_dim = (
                mla_config.qk_head_dim + mla_config.qk_pos_emb_head_dim
                if config.muon_split_mla_per_head
                else None
            )
        else:
            kv_up_proj_split_shapes = None
            q_up_proj_head_dim = None

        qkv_down_proj_split_shapes = (
            mla_config.q_lora_rank,
            mla_config.kv_lora_rank + mla_config.qk_pos_emb_head_dim,
        ) if is_mla and getattr(mla_config, 'q_lora_rank', None) is not None else None

        named_modules = dict(model_chunk.named_modules())
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            # add flag for expert weight so optimizer can figure which tp group it uses
            # alternatively, create new param group and save tp_group. this require more
            # change in optimizer
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # add flags for grouped QKV and MLA projection parameters
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            if (
                getattr(model_chunk.config, 'gated_linear_unit', False)
                and param.ndim == 2
                and 'linear_fc1.weight' in name
            ):
                param.glu_split_dim = 0
            if 'linear_kv_up_proj.weight' in name and len(param.shape) == 2:
                param.is_kv_up_proj = True
            if 'linear_q_up_proj.weight' in name and len(param.shape) == 2:
                param.is_q_up_proj = True
            if 'linear_qkv_down_proj.weight' in name and len(param.shape) == 2:
                param.is_qkv_down_proj = True
            if len(param.shape) == 2 and name.endswith('router.weight'):
                param.is_router = True
            if len(param.shape) == 2 and ('linear_fc2' in name or 'linear_proj' in name):
                param.is_out_proj = True
            param.md_gain_log_family = _gain_log_family(name, param)
            if param.md_gain_log_family == "layernorm":
                param.md_layernorm_gain_offset = float(
                    getattr(model_chunk.config, "layernorm_zero_centered_gamma", False)
                )
            module_name = name.rpartition('.')[0]
            while module_name:
                module = named_modules.get(module_name)
                layer_number = getattr(module, 'layer_number', None)
                if layer_number is not None:
                    param.md_gain_log_layer = int(layer_number) - 1
                    break
                module_name = module_name.rpartition('.')[0]
            # TODO(deyuf): currently only allow 2D non-embedding weight to avoid breaking
            if (
                not getattr(param, 'is_embedding_or_output_parameter', False)
                and len(param.shape) == 2
                and not getattr(param, 'is_kda_decay_parameter', False)
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    # Separate LR for the Muon-managed matrices. Defaults to the base --lr (matrix_lr unset,
    # muon_lr_factor == 1.0); the chained scalar Adam/Lion below always stays on config.lr.
    matrix_lr = (config.matrix_lr if config.matrix_lr is not None
                 else config.muon_lr_factor * (config.lr or 0.0))

    muon_kwargs = {
        "lr": matrix_lr,
        "momentum_beta": config.muon_momentum,
        "use_nesterov": config.muon_use_nesterov,
        "weight_decay": config.weight_decay,
        "fp32_matmul_prec": config.muon_fp32_matmul_prec,
        "coefficient_type": config.muon_coefficient_type,
        "num_ns_steps": config.muon_num_ns_steps,
        "scale_mode": config.muon_scale_mode,
        "split_qkv": config.muon_split_qkv,
        "split_fc1": config.muon_split_fc1,
        "is_qkv_fn": lambda p: getattr(p, "is_qkv", False),
        "qkv_split_shapes": qkv_split_shapes,
        "is_kda_in_proj_fn": lambda p: getattr(p, "is_kda_in_proj", False),
        "is_kv_up_proj_fn": lambda p: getattr(p, "is_kv_up_proj", False),
        "kv_up_proj_split_shapes": kv_up_proj_split_shapes,
        "is_qkv_down_proj_fn": lambda p: getattr(p, "is_qkv_down_proj", False),
        "qkv_down_proj_split_shapes": qkv_down_proj_split_shapes,
        "split_mla_per_head": config.muon_split_mla_per_head,
        "is_q_up_proj_fn": lambda p: getattr(p, "is_q_up_proj", False),
        "q_up_proj_head_dim": q_up_proj_head_dim,
        "extra_scale_factor": config.muon_extra_scale_factor,
        "pg_collection": pg_collection,
        "mode": config.muon_tp_mode,
    }

    # freezing nonlinear params and get param groups for muon
    for param in nonlinear_params:
        param.requires_grad = False

    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides)
    # The LR scheduler overwrites each group's 'lr' every step from its 'max_lr'/'min_lr', so the
    # muon_kwargs["lr"] default above is not enough — set the matrix schedule on the groups too.
    # Only touch groups on the default LR schedule so explicit decoupled-LR overrides are kept.
    if config.matrix_lr is not None or config.muon_lr_factor != 1.0:
        # Per-group min_lr policy (mirrors md_decoupling): keep the decay curve well-formed by
        # deriving min_lr from this group's max_lr rather than the global config.min_lr.
        floor = config.min_lr if config.min_lr is not None else 0.0
        if config.min_lr_mode == 'absolute':
            matrix_min_lr = min(floor, matrix_lr)
        else:
            ratio = (floor / config.lr) if (config.lr and config.lr > 0) else 0.0
            matrix_min_lr = matrix_lr * ratio
        for group in linear_param_groups:
            if group.get('default_config', False):
                group['max_lr'] = matrix_lr
                group['min_lr'] = matrix_min_lr
                group['default_config'] = False

    # if layerwise distributed optimizer is not used, need to handle ep params separately
    expert_param_groups = []
    if not layer_wise_distributed_optimizer:
        for group in linear_param_groups:
            if group['is_expert_parallel']:
                expert_param_groups.append(group)
                linear_param_groups.remove(group)

    optimizer = TensorParallelMuon(linear_param_groups, **muon_kwargs)

    reset_config_bf16 = False
    if config.bf16:
        if layer_wise_distributed_optimizer:
            # creating master weight before layerwise sharding will lead to unnecessary master
            # weight so here we delay master weight creation into layer_wise unset config.bf16
            # will also result in all optimizers below(adam) to also not be wrapped
            config.bf16 = False
            reset_config_bf16 = True
        else:
            # if not using layer_wise wrapper, just create master weight here is fine
            optimizer = Float16OptimizerWithFloat16Params(
                optimizer, config, None, muon_init_state_fn
            )
    else:
        optimizer = FP32Optimizer(optimizer, config, muon_init_state_fn)

    optimizers.append(optimizer)

    # expert optimizer exists meaning layerwise distributed optimizer is not used
    if len(expert_param_groups) > 0:
        expert_optimizer = TensorParallelMuon(expert_param_groups, **muon_kwargs)
        if config.bf16:
            expert_optimizer = Float16OptimizerWithFloat16Params(
                expert_optimizer, config, None, muon_init_state_fn
            )
        else:
            expert_optimizer = FP32Optimizer(expert_optimizer, config, muon_init_state_fn)
        setattr(expert_optimizer, 'grad_stats_parallel_group', pg_collection.tp_ep_pp)
        optimizers.append(expert_optimizer)

    # done with muon, unfreeze nonlinear and freeze linear
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    # call original get. linear params will be skipped since they're freezed
    chained_adam = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )

    # unfreeze everything
    for param in linear_params:
        param.requires_grad = True

    # chain everything together
    init_fns = [muon_init_state_fn] + len(chained_adam.chained_optimizers) * [
        nonlinear_init_state_fn
    ]
    optimizers += chained_adam.chained_optimizers

    if layer_wise_distributed_optimizer:
        log_single_rank(logger, logging.INFO, 'Using LayerWiseDistributedOptimizer for Muon')
        if reset_config_bf16:
            config.bf16 = True
        return LayerWiseDistributedOptimizer(
            optimizers,
            config,
            pg_collection,
            init_state_fn_list=init_fns,
            model_chunks=model_chunks,
            async_allgather=config.overlap_param_gather,
        )
    return ChainedOptimizer(optimizers)
