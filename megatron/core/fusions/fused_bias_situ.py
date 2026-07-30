# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


# pylint: disable=missing-function-docstring, missing-class-docstring

import torch

from megatron.core.jit import jit_fuser
from megatron.core.utils import nvtx_decorator

###### BIAS SITU FUSION/ NO AUTOGRAD ################
# SiTU is a gated linear unit whose gate is a sigmoid-times-tanh:
#   gate(x)  = sigmoid(x) * tanh(x)   (== situ_act in activations.py)
#   SiTU(y_1, y_2) = gate(y_1) * y_2
# Like RLGLU/SSSGLU (and unlike SwiGLU/SSGLU) the gate is NOT of the form ``x * squash(x)`` -- it is
# the gate itself, applied directly, so there is no separate SiLU-style helper. With s = sigmoid(y1)
# and t = tanh(y1), the gate derivative is
#   gate'(x) = s*(1-s)*t + s*(1-t**2)
# which reuses s and t (already computed for the gate). Built exactly like the
# SwiGLU/SSGLU/RLGLU/SSSGLU fusions (fused_bias_swiglu.py / ... / fused_bias_sssglu.py): @jit_fuser
# forward/backward pairs wrapped in torch.autograd.Function, since the gate has no cross-feature
# reduction. The gate math is inlined here (rather than importing situ_act) to keep the jit_fuser
# scripting self-contained; it must stay in sync with situ_act in activations.py.


@jit_fuser
def situ(y):
    """Performs SiTU (Sigmoid-Tanh-gated linear Unit) activation function.

    Args:
        y (torch.Tensor): Input tensor to be split into two halves along the last dimension.

    Returns:
        torch.Tensor: Result of SiTU activation: gate(y1) * y2, where y1, y2 are the split
            halves and gate(x) = sigmoid(x) * tanh(x).
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return (torch.sigmoid(y_1) * torch.tanh(y_1)) * y_2


@jit_fuser
def bias_situ(y, bias):
    """Performs SiTU activation with bias addition.

    Args:
        y (torch.Tensor): Input tensor.
        bias (torch.Tensor): Bias tensor to be added to input.

    Returns:
        torch.Tensor: Result of bias addition followed by SiTU activation.
    """
    y = y + bias
    return situ(y)


@jit_fuser
def weighted_situ(y, weights):
    dtype = y.dtype
    res = situ(y) * weights
    return res.to(dtype)


@jit_fuser
def situ_back(g, y):
    """Computes the gradient for the SiTU activation function.

    With gate(x) = sigmoid(x) * tanh(x), s = sigmoid(x), t = tanh(x), the gate derivative is
        gate'(x) = s*(1-s)*t + s*(1-t**2).
    So d/dy1 [gate(y1) * y2] = gate'(y1) * y2 and d/dy2 [gate(y1) * y2] = gate(y1).

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    s = torch.sigmoid(y_1)
    t = torch.tanh(y_1)
    gate = s * t
    gate_prime = s * (1 - s) * t + s * (1 - t * t)
    return torch.cat((g * gate_prime * y_2, g * gate), -1)


@jit_fuser
def bias_situ_back(g, y, bias):
    """Computes the gradient for the biased SiTU activation function.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.
        bias (torch.Tensor): Bias tensor that was added in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor, computed after
            applying the bias addition.
    """
    y = y + bias
    return situ_back(g, y)


@jit_fuser
def weighted_situ_back(g, y, weights):
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = situ_back(g * weights, y)
    # precison of w may be higher than y and g, so we need to cast g to w_dtype
    weights_grad = situ(y) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


class BiasSiTUFunction(torch.autograd.Function):
    """Custom autograd function for SiTU activation with bias support."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, bias, fp8_input_store, cpu_offload_input):
        """Forward pass of biased SiTU activation."""
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
            bias.activation_offloading = True
        ctx.save_for_backward(input_for_backward, bias)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return bias_situ(input, bias)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of biased SiTU activation."""
        input, bias = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = bias_situ_back(grad_output, input, bias)
        return tmp, tmp, None, None


class SiTUFunction(torch.autograd.Function):
    """Custom autograd function for SiTU activation without bias."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, fp8_input_store, cpu_offload_input):
        """Forward pass of SiTU activation."""
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return situ(input)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of SiTU activation."""
        input = ctx.saved_tensors[0]
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = situ_back(grad_output, input)
        return tmp, None, None


class WeightedSiTUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weights, fp8_input_store):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return weighted_situ(input, weights)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp, wgrad = weighted_situ_back(grad_output, input, weights)
        return tmp, wgrad, None


def bias_situ_impl(input, bias, fp8_input_store=False, cpu_offload_input=False):
    """Implementation of biased SiTU that handles different input shapes.

    This function reshapes the input if necessary, applies the SiTU activation
    (with or without bias), and restores the original shape.

    Args:
        input (torch.Tensor): Input tensor to apply SiTU activation.
        bias (torch.Tensor, optional): Bias tensor to be added to input. If None,
            uses the bias-free SiTU variant.
        fp8_input_store (bool, optional): Whether to store intermediate values in FP8 format.
            Defaults to False.

    Returns:
        torch.Tensor: Result of biased SiTU activation.

    Raises:
        AssertionError: If input tensor does not have 2 or 3 dimensions.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasSiTUFunction.apply(input, bias, fp8_input_store, cpu_offload_input)
    else:
        output = SiTUFunction.apply(input, fp8_input_store, cpu_offload_input)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def weighted_bias_situ_impl(input, bias, weights, fp8_input_store=False):
    """
    Token-wise-weighted bias situ fusion.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted situ fusion")
    else:
        output = WeightedSiTUFunction.apply(input, weights, fp8_input_store)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)
