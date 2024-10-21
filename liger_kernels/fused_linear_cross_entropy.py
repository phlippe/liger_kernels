"""
Liger Fused Linear Cross Entropy Kernel

This module provides a fused linear layer and cross entropy loss kernel for training Language Models. The kernel is
implemented using Triton and JAX, and is designed to be memory efficient by chunking the gradient calculation.

Adapted from the original Liger kernel implementation:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py
"""

import logging
from typing import Literal

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton

from cross_entropy import liger_cross_entropy_kernel
from utils import element_mul_kernel, get_stride

# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576
# https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19  # noqa E501,W505
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2

LOGGER = logging.getLogger(__name__)


def fused_linear_cross_entropy_forward(
    _input: jax.Array,
    weight: jax.Array,
    target: jax.Array,
    bias: jax.Array | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
    dtype: jnp.dtype = None,
    add_trivial_operations: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """
    Compute the forward pass of a fused linear layer and cross entropy loss.

    Args:
        _input: The input tensor of shape (BT, H).
        weight: The weight tensor of shape (H, V).
        target: The target tensor of shape (BT,).
        bias: The bias tensor of shape (V,). If None, no bias is added.
        ignore_index: The index to ignore in the target tensor.
        label_smoothing: The label smoothing factor.
        reduction: The reduction method for the loss.
        dtype: The dtype of the input tensor.
        add_trivial_operations: Whether to add trivial operations in an attempt to make the XLA compiler schedule
            the triton kernel correctly. Does not work in all cases at the moment.

    Returns:
        A tuple containing the loss, the gradient of the input, the gradient of the weight, and the gradient of the
        bias.
    """
    if dtype is None:
        dtype = _input.dtype

    # Cast weights and biases to the same dtype as the input.
    weight = weight.astype(dtype)
    if bias is not None:
        bias = bias.astype(dtype)

    # inputs have shape: BT x H
    # materialized activations will have shape: BT x V
    # the increase in memory = BT x V
    # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
    # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
    # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
    # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048
    BT, H = _input.shape
    V = weight.shape[1]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
    chunk_size = triton.next_power_of_2(
        triton.cdiv(BT, inc_factor)
    )  # (BT + inc_factor - 1) // inc_factor
    if BT % chunk_size != 0:
        raise NotImplementedError(
            "The current version of the kernel only supports batch sizes divisible by chunk size, "
            f"but got BT={BT} and chunk_size={chunk_size}."
        )
    num_chunks = BT // chunk_size

    total_n_non_ignore = (target != ignore_index).sum().astype(jnp.float32).clip(min=1)

    # We wrap the loop over chunks in a scan to accumulate the weight/bias gradients and stack the input gradients.
    def _scan_fn(
        carry: tuple[jax.Array, jax.Array], xs: tuple[jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        grad_weight, grad_bias = carry
        _input_chunk, target_chunk = xs

        # when doing matmul, use the original precision
        logits_chunk = _input_chunk @ weight  # chunk_size x V
        if bias is not None:
            logits_chunk = logits_chunk + bias

        n_rows = logits_chunk.shape[0]

        # unreduced loss
        loss_1d_slice = jax.ShapeDtypeStruct(shape=(chunk_size,), dtype=jnp.float32)
        n_non_ignore = (
            (target_chunk != ignore_index).sum().astype(jnp.float32).clip(min=1)
        )

        # when doing CE, use the upcasted precision
        logits_chunk = logits_chunk.astype(jnp.float32)

        # Here we calculate the gradient of logits_chunk in place so we can save memory.
        (loss_1d_slice,) = jt.triton_call(
            logits_chunk,
            target_chunk,
            n_non_ignore,
            out_shape=(loss_1d_slice,),
            X_stride=get_stride(logits_chunk, 0),
            Y_stride=get_stride(target_chunk, 0),  # always 1
            loss_stride=get_stride(loss_1d_slice, 0),  # always 1
            n_cols=V,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
            grid=(n_rows,),
            kernel=liger_cross_entropy_kernel,
        )

        # NOTE: while we perform in-place operations on logits_chunk in the triton kernel, the XLA compiler
        # seems to be not aware of this and may schedule future logits_chunk computations before the kernel
        # completes. This leads to incorrect results, as the actual logits are used for gradient computation
        # instead of the gradients of the logits. For the test case, this can be corrected by adding a trivial
        # operation that forces the XLA compiler to wait for the triton kernel to complete. However, even this
        # showed not sufficient when running on a LLM model.
        if add_trivial_operations:
            triv_loss_scalar = jax.lax.stop_gradient(loss_1d_slice.sum()).astype(
                logits_chunk.dtype
            )
            logits_chunk = logits_chunk + triv_loss_scalar
            logits_chunk = logits_chunk - triv_loss_scalar

        # gradient of logits_chunk is computed in-place by the above triton kernel.
        # Following HuggingFace model source code, we do the forward and backward
        # w.r.t. logits in fp32 for numerical stability especially as the num classes (vocab size) is huge.
        # (reference: https://github.com/huggingface/transformers/blob/v4.42.4/src/transformers/models/llama/modeling_llama.py#L1194)  # noqa E501,W505
        # Propagating to lm_head's backward, we'll switch back to the original dtype.
        logits_chunk = logits_chunk.astype(dtype)

        # gradient of logits_chunk is computed in-place by the above triton kernel and is of shape: chunk_size x V
        # thus grad_input[start_idx: end_idx] should be of shape: chunk_size x H
        # additionally, since we are chunking the inputs, observe that the loss and gradients are calculated only
        # on `n_non_ignore` tokens. However, the gradient of the input should be calculated for all tokens.
        # Thus, we need an additional scaling factor of (n_non_ignore/total_n_non_ignore) to scale the gradients.

        if reduction == "mean":
            alpha = n_non_ignore / total_n_non_ignore
        else:
            alpha = 1.0

        loss_1d_slice = loss_1d_slice * alpha.astype(loss_1d_slice.dtype)
        alpha = alpha.astype(logits_chunk.dtype)
        grad_logits_chunk = logits_chunk * alpha  # chunk_size x V
        grad_input = grad_logits_chunk @ weight.T  # chunk_size x H

        if grad_weight is not None:
            grad_weight = grad_weight + alpha * (_input_chunk.T @ logits_chunk)

        if bias is not None:
            grad_bias = grad_bias + alpha * logits_chunk.sum(axis=0)

        return (grad_weight, grad_bias), (loss_1d_slice, grad_input)

    grad_weight = jnp.zeros_like(weight)
    grad_bias = jnp.zeros_like(bias) if bias is not None else None

    (grad_weight, grad_bias), (loss_1d, grad_input) = jax.lax.scan(
        _scan_fn,
        init=(grad_weight, grad_bias),
        xs=(
            _input.reshape(num_chunks, chunk_size, H),
            target.reshape(num_chunks, chunk_size),
        ),
        length=num_chunks,
    )

    grad_input = grad_input.reshape(BT, H)
    loss = jnp.sum(loss_1d)
    return loss, grad_input, grad_weight, grad_bias


def fused_linear_cross_entropy_backward(
    grad_output: jax.Array,
    grad_input: jax.Array,
    grad_weight: jax.Array | None,
    grad_bias: jax.Array | None,
    last_layer: bool = False,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
    """
    Compute the backward pass of a fused linear layer and cross entropy loss.

    Args:
        grad_output: The gradient of the loss.
        grad_input: The gradient of the input tensor.
        grad_weight: The gradient of the weight tensor.
        grad_bias: The gradient of the bias tensor.
        last_layer: Whether the cross entropy loss is the last layer.

    Returns:
        A tuple containing the gradient of the input, the gradient of the weight, and the gradient of the bias.
    """

    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    if not last_layer:
        # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
        # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        jt.triton_call(
            grad_input,
            grad_output,
            out_shape=(),
            X_stride=get_stride(grad_input, 0),
            n_cols=H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
            grid=(n_rows,),
            kernel=element_mul_kernel,
        )

        # handle grad_weight
        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V

            jt.triton_call(
                grad_weight,
                grad_output,
                out_shape=(),
                X_stride=get_stride(grad_weight, 0),
                n_cols=H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32,
                grid=(n_rows,),
                kernel=element_mul_kernel,
            )

        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V

            jt.triton_call(
                grad_bias,
                grad_output,
                out_shape=(),
                X_stride=get_stride(grad_bias, 0),
                n_cols=1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32,
                grid=(n_rows,),
                kernel=element_mul_kernel,
            )
    return grad_input, grad_weight, grad_bias


def liger_fused_linear_cross_entropy(
    _input: jax.Array,
    weight: jax.Array,
    target: jax.Array,
    bias: jax.Array | None = None,
    target_mask: jax.Array | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
    last_layer: bool = False,
    dtype: jnp.dtype = None,
    add_trivial_operations: bool = False,
) -> jax.Array:
    """
    Compute the fused linear layer and cross entropy loss.

    NOTE: Using Liger fused linear cross entropy kernel may not be stable within jit compilation.
    The order of operations is important and may not be preserved when jitting the function.

    Args:
        _input: The input tensor of shape (..., hidden_size).
        weight: The weight tensor of shape (hidden_size, vocab_size).
        target: The target tensor of shape (...,). Must match the batch size of the input tensor.
        bias: The bias tensor of shape (vocab_size,). If None, no bias is added.
        target_mask: A mask tensor of shape (...,) to indicate which target values to keep (True) and which to
            ignore (False). If provided, sets targets where the mask is False to ignore_index.
            If None, no mask is applied.
        ignore_index: The index to ignore in the target tensor.
        label_smoothing: The label smoothing factor.
        reduction: The reduction method for the loss.
        last_layer: Whether the cross entropy loss is the last layer.
        dtype: The dtype to cast the logits in. If None, the dtype of the input tensor is used.
        add_trivial_operations: Whether to add trivial operations in an attempt to make the XLA compiler schedule
            the triton kernel correctly. Does not work in all cases at the moment.

    Returns:
        The cross entropy loss.
    """
    LOGGER.warning(
        "Using Liger fused linear cross entropy kernel may not be stable within jit compilation."
    )

    _input = _input.reshape(-1, _input.shape[-1])
    target = target.reshape(-1)
    assert (
        _input.shape[0] == target.shape[0]
    ), "Input and target must have the same batch size."
    assert (
        _input.shape[1] == weight.shape[0]
    ), "Input and weight must have the same hidden size."
    if bias is not None:
        assert (
            weight.shape[1] == bias.shape[0]
        ), "Weight and bias must have the same output size."
    if target_mask is not None:
        target_mask = target_mask.reshape(-1)
        assert (
            target.shape[0] == target_mask.shape[0]
        ), "Target and target mask must have the same shape."
        target = jnp.where(target_mask, target, ignore_index)

    @jax.custom_gradient
    def fwd(_input, weight, target, bias):
        loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input,
            weight,
            target,
            bias,
            ignore_index,
            label_smoothing,
            reduction,
            dtype,
            add_trivial_operations,
        )

        def backward(grad_output):
            n_grad_input, n_grad_weight, n_grad_bias = (
                fused_linear_cross_entropy_backward(
                    grad_output, grad_input, grad_weight, grad_bias, last_layer
                )
            )
            # Up-cast gradients for parameters.
            n_grad_input = n_grad_input.astype(_input.dtype)
            n_grad_weight = n_grad_weight.astype(weight.dtype)
            if n_grad_bias is not None:
                n_grad_bias = n_grad_bias.astype(bias.dtype)
            return n_grad_input, n_grad_weight, None, n_grad_bias

        return loss, backward

    return fwd(_input, weight, target, bias)
