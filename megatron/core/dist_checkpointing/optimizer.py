# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Helpers for defining sharding for optimizer states based on existing sharding
for model parameters.
"""

import logging
from copy import deepcopy
from dataclasses import replace
from typing import Callable, Dict, Iterable, Tuple, Union

logger = logging.getLogger(__name__)

import torch

from megatron.core.utils import log_single_rank, to_local_if_dtensor

from .dict_utils import nested_values
from .mapping import (
    LocalNonpersistentObject,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
    StateDict,
)
from .utils import extract_sharded_tensors_and_factories

KEEP_VARS_HINT = (
    " Make sure state dict contains original torch.nn.Parameters (not pure torch.Tensors)"
    " by passing `keep_vars=True` to `.state_dict()`. If any transformation of the original"
    " parameter is needed, use a ShardedTensorFactory."
)


def get_optim_param_to_id_map(optim_params_iter: Iterable[torch.nn.Parameter]) -> Dict[int, int]:
    """Generate mapping from optimizer param to optimizer state id."""
    param_mappings = {}
    for i, param in enumerate(optim_params_iter):
        param = to_local_if_dtensor(param)
        if id(param) not in param_mappings:
            param_mappings[id(param)] = i
    return param_mappings


def get_param_id_to_sharded_param_map(
    model_sharded_state_dict: ShardedStateDict, optim_params_iter: Iterable[torch.nn.Parameter]
) -> Dict[int, Union[ShardedTensor, ShardedTensorFactory]]:
    """Generate mapping from optimizer state ids to model sharded parameters.

    Args:
        model_sharded_state_dict: sharded state dict with all model sharded tensors
            (can have any structure)
        optim_params_iter: iterable which iterates over model parameters tracked by the optimizer.
            The iteration must be in the same order as in the optimizer parameters.

    Returns:
        Dict[int, Union[ShardedTensor, ShardedTensorFactory]]: mapping from optimizer state ids
            to model sharded parameters.
    """
    model_sharded_state_dict, _ = extract_sharded_tensors_and_factories(model_sharded_state_dict)
    id_to_sharded_param_map = {}
    param_to_id_map = get_optim_param_to_id_map(optim_params_iter)
    # If using PyTorch FSDP2 the values in model_sharded_state_dict would
    # have been converted to local tensors during initialization.
    # See the make_(tp)_sharded_tensor_for_checkpoint functions.
    for ten in nested_values(model_sharded_state_dict):
        if id(ten.data) in param_to_id_map:
            id_to_sharded_param_map[param_to_id_map[id(ten.data)]] = ten
        else:
            logger.debug(f'{ten} is not tracked by the optimizer')

    if not id_to_sharded_param_map:
        log_single_rank(
            logger,
            logging.WARNING,
            "Sharded parameters mapping is empty. It means tensors in model state dict"
            " do not correspond to tensors in optimizer parameters map."
            " Make sure to call state_dict with `keep_vars=True`.",
        )
    return id_to_sharded_param_map


def make_sharded_optimizer_tensor(
    model_param: Union[ShardedTensor, ShardedTensorFactory], optim_param: torch.Tensor, prefix: str
) -> Union[ShardedTensor, ShardedTensorFactory]:
    """Build a ShardedTensor or ShardedTensorFactory for optimizer param based on model param

    Args:
        model_param (Union[ShardedTensor, ShardedTensorFactory]): model param
        optim_param (torch.Tensor): corresponding optimizer param
        prefix (str): optimizer prefix for the ShardedTensor or ShardedTensorFactory

    Returns:
        Union[ShardedTensor, ShardedTensorFactory]: wrapped optimizer parameter
    """
    optim_param = to_local_if_dtensor(optim_param)
    if isinstance(model_param, ShardedTensorFactory):
        return replace(model_param, key=f'{prefix}.{model_param.key}', data=optim_param)

    assert tuple(optim_param.shape) == model_param.local_shape, (
        f'Optimizer shape ({tuple(optim_param.shape)}) does not match model shape '
        f'({model_param.local_shape}) for param key={getattr(model_param, "key", "?")!r} '
        f'(prefix={prefix!r})'
    )
    sh_ten = replace(
        model_param, key=f'{prefix}.{model_param.key}', data=optim_param, dtype=optim_param.dtype
    )
    sh_ten.validate_metadata_integrity()
    return sh_ten


def _project_replica_id(model_param: ShardedTensor, retained_axes: Tuple[int, ...]):
    """Turn fragmented model axes omitted from a projection into replica coordinates.

    Megatron's standard replica tuple is ``(PP, TP, DP)``. A TP-sharded model tensor has a
    zero TP replica coordinate because its TP ranks own distinct tensor shards. If an optimizer
    tensor projection drops that sharded axis, those ranks instead own the same projected shard,
    so its chunk coordinate becomes the TP replica coordinate.

    Non-standard replica identifiers are supported by appending the omitted chunk coordinates.
    The common ``(PP, TP, DP)`` case keeps its shape because layer-wise optimizer checkpointing
    relies on the DP coordinate remaining at index 2.
    """
    if model_param.axis_fragmentations is None:
        raise ValueError(
            f'Cannot project optimizer state from irregularly sharded tensor {model_param.key}'
        )

    retained_axes_set = set(retained_axes)
    prepend_axis_num = model_param.prepend_axis_num
    omitted_chunk_offsets = []
    for local_axis, local_size in enumerate(model_param.local_shape):
        global_axis = prepend_axis_num + local_axis # Map local axis to global axis
        if local_axis in retained_axes_set or model_param.axis_fragmentations[global_axis] == 1:
            continue
        global_offset = model_param.global_offset[global_axis]
        if local_size == 0 or global_offset % local_size != 0:
            raise ValueError(
                f'Cannot determine replica coordinate for axis {local_axis} of {model_param.key}'
            )
        omitted_chunk_offsets.append(global_offset // local_size)

    if not omitted_chunk_offsets:
        return model_param.replica_id

    replica_id = model_param.replica_id
    if isinstance(replica_id, tuple) and len(replica_id) == 3:
        if len(omitted_chunk_offsets) != 1:
            raise ValueError(
                f'Expected at most one omitted TP axis for {model_param.key}, got '
                f'{omitted_chunk_offsets}'
            )
        replica_id = list(replica_id)
        if replica_id[1] not in (0, omitted_chunk_offsets[0]):
            raise ValueError(
                f'Conflicting TP replica coordinates for {model_param.key}: '
                f'{replica_id[1]} and {omitted_chunk_offsets[0]}'
            )
        replica_id[1] = omitted_chunk_offsets[0]
        return tuple(replica_id)

    if isinstance(replica_id, int):
        if replica_id == 0 and len(omitted_chunk_offsets) == 1:
            return omitted_chunk_offsets[0]
        replica_id = (replica_id,)
    return (*replica_id, *omitted_chunk_offsets)


def make_sharded_optimizer_tensor_for_axes(
    model_param: ShardedTensor,
    optim_param: torch.Tensor,
    key: str,
    retained_axes: Tuple[int, ...],
    allow_shape_mismatch: bool = False,
) -> ShardedTensor:
    """Project model tensor sharding onto an optimizer tensor with fewer local axes.

    All prepended logical axes (for example layer and expert axes) are preserved. ``retained_axes``
    indexes the local model tensor axes represented by ``optim_param``. Fragmented model axes that
    are not retained become replica coordinates instead of tensor-sharding axes.
    ``allow_shape_mismatch`` should only be enabled when the projected tensor retains the
    model axis whose global shape may change.
    """
    optim_param = to_local_if_dtensor(optim_param)
    retained_axes = tuple(retained_axes)
    if len(set(retained_axes)) != len(retained_axes):
        raise ValueError(f'Duplicate retained axes for {key}: {retained_axes}')
    if any(axis < 0 or axis >= len(model_param.local_shape) for axis in retained_axes):
        raise ValueError(
            f'Invalid retained axes {retained_axes} for model shape {model_param.local_shape}'
        )
    if model_param.flattened_range is not None:
        raise ValueError(f'Cannot project flattened optimizer state for {model_param.key}')

    expected_local_shape = tuple(model_param.local_shape[axis] for axis in retained_axes)
    if tuple(optim_param.shape) != expected_local_shape:
        raise ValueError(
            f'Projected optimizer shape {tuple(optim_param.shape)} does not match axes '
            f'{retained_axes} of model shape {model_param.local_shape} '
            f'(expected {expected_local_shape})'
        )
    if model_param.axis_fragmentations is None:
        raise ValueError(
            f'Cannot project optimizer state from irregularly sharded tensor {model_param.key}'
        )

    prepend_axis_num = model_param.prepend_axis_num
    global_axes = (*range(prepend_axis_num), *(prepend_axis_num + a for a in retained_axes))
    sh_ten = ShardedTensor(
        key=key,
        data=optim_param,
        dtype=optim_param.dtype,
        local_shape=tuple(optim_param.shape),
        global_shape=tuple(model_param.global_shape[axis] for axis in global_axes),
        global_offset=tuple(model_param.global_offset[axis] for axis in global_axes),
        axis_fragmentations=tuple(
            model_param.axis_fragmentations[axis] for axis in global_axes
        ),
        replica_id=_project_replica_id(model_param, retained_axes),
        prepend_axis_num=prepend_axis_num,
        allow_shape_mismatch=allow_shape_mismatch,
    )
    return sh_ten


def optim_state_to_sharding_state(
    optim_state_dict: StateDict,
    id_to_sharded_param_map: Dict[int, Union[ShardedTensor, ShardedTensorFactory]],
    exclude_keys: Tuple[str, ...] = (),
    state_sharding_fn: Callable[
        [ShardedTensor | ShardedTensorFactory, torch.Tensor, str, str],
        ShardedTensor | ShardedTensorFactory | None,
    ] | None = None,
):
    """Turn optimizer state dict to sharded state dict based on model state dict *in-place*.

    Can be used to add sharding information to most common optimizer state dict.
    Creates separate ShardedTensors for each key in `optim_state_dict['state']`
    (e.g. for torch.optim.Adam there will be separate tensors for `exp_avg` and `exp_avg_sq`)

    Args:
        optim_state_dict (StateDict): optimizer state dict with
            state parameters under `state` key and group hyperparameters under
            `param_groups` -> `params` key.
        id_to_sharded_param_map: mapping from optimizer param ids to model sharded tensors or
            factories. Can be generated with `get_param_id_to_sharded_param_map`.
        exclude_keys (Tuple[str]): optimizer state keys to exclude from the final state dict.
        state_sharding_fn (Callable, optional): state-specific sharding builder. It receives the
            model sharding, optimizer value, state key, and optimizer key prefix. Returning None
            falls back to the standard model-shaped optimizer tensor path.

    Returns:
        None: state dict is modified in place
    """
    if isinstance(exclude_keys, str):
        exclude_keys = (exclude_keys,)

    sharded_state = {}
    for param_id, param_state in optim_state_dict['state'].items():
        sharded_state[param_id] = {}
        for state_key, param in param_state.items():
            if state_key in exclude_keys:
                continue
            if param_id in id_to_sharded_param_map:
                model_param = id_to_sharded_param_map[param_id]
                prefix = f'optimizer.state.{state_key}'
                custom_sharded_state = None
                if state_sharding_fn is not None:
                    custom_sharded_state = state_sharding_fn(
                        model_param, param, state_key, prefix
                    )
                if custom_sharded_state is None:
                    custom_sharded_state = make_sharded_optimizer_tensor(
                        model_param, param, prefix=prefix
                    )
                sharded_state[param_id][state_key] = custom_sharded_state
            else:
                raise ValueError(f'Param id {param_id} does not match any model sharded param')

    optim_state_dict['param_groups'] = deepcopy(optim_state_dict['param_groups'])
    for group in optim_state_dict['param_groups']:
        group['params'] = LocalNonpersistentObject(group['params'])
    optim_state_dict['state'] = sharded_state
