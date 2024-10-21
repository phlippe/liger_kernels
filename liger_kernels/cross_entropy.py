"""
Liger Cross Entropy Kernel

This module contains the Liger Cross Entropy kernel for JAX. The kernel computes both the cross entropy loss and the
gradient of the input tensor. The kernel is optimized for performance using Triton.

Adapted from the original Liger kernel implementation:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py
"""

from typing import Literal

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
import triton.language as tl

from utils import element_mul_kernel, get_stride


@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    Y_ptr,
    n_non_ignore_ptr,
    loss_ptr,
    X_stride,
    Y_stride,
    loss_stride,
    n_cols,
    ignore_index,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,  # set it as constexpr since reduction is always known at compile time
    BLOCK_SIZE: tl.constexpr,
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now. Please refer to
    https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Args:
        X_ptr: Pointer to input tensor.
        Y_ptr: Pointer to target tensor.
        n_non_ignore: The number of non-ignored elements in the batch.
        loss_ptr: Pointer to tensor to store the loss.
        X_stride: The stride of the input tensor.
        Y_stride: The stride of the target tensor.
        loss_stride: The stride of the loss tensor.
        n_cols: The number of columns in the input tensor.
        ignore_index: The index to ignore in the target.
        label_smoothing: The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction: The string for the reduction to apply
        BLOCK_SIZE: The block size for Triton operations.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*V is too large, program_id * stride will overflow out of int32, so we convert to int64
    program_id = tl.program_id(0).to(tl.int64)

    # 1. Load Y_ptr first because if the target is ignore_index, we can return right away
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    # 2. locate the start index
    X_ptr += program_id * X_stride

    loss_ptr += program_id * loss_stride

    if y == ignore_index:
        # set all X_ptr as 0
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        tl.store(loss_ptr, 0.0)
        return

    # Load the number of non-ignored elements. Single value, so no need to shift.
    n_non_ignore = tl.load(n_non_ignore_ptr)

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # 3. [Online softmax] first pass: find max + sum
    m = float("-inf")  # m is the max value. use the notation from the paper
    d = 0.0  # d is the sum. use the notation from the paper
    ori_X_y = tl.load(
        X_ptr + y
    )  # we need to store the original value of X_y for the loss calculation

    # Label smoothing is a general case of normal cross entropy
    # See the full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issue-2503665310
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")
        )
        block_max = tl.max(X_block)
        if label_smoothing > 0:
            # scale X beforehand to avoid overflow
            scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    # 4. [Online Softmax] Second pass: compute gradients
    # For 'mean' reduction, gradients are normalized by number of non-ignored elements (N)
    # dx_y = (softmax(x_y) - 1) / N
    # dx_i = softmax(x_i) / N, i != y
    # For label smoothing:
    # dx_i = (softmax(x_y) - label_smoothing / V) / N, V = n_cols, i != y
    # dx_y = (softmax(x_y) - label_smoothing / V - (1 - label_smoothing)) / N
    #      = dx_i - (1 - label_smoothing) / N
    #
    # For 'sum' reduction, no normalization is applied:
    # dx_y = softmax(x_y) - 1
    # dx_i = softmax(x_i), for i ≠ y
    # For label smoothing:
    # dx_i = (softmax(x_y) - label_smoothing / V), V = n_cols, i != y
    # dx_y = (softmax(x_y) - label_smoothing / V - (1 - label_smoothing))
    #      = dx_i - (1 - label_smoothing)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf")
        )
        if reduction == "mean":
            X_block = (tl.exp(X_block - m) / d - eps) / (n_non_ignore)
        else:
            X_block = tl.exp(X_block - m) / d - eps

        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    # We need tl.debug_barrier() to ensure the new result of X_ptr is written as mentioned in
    # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34  # noqa: E501,W505
    tl.debug_barrier()

    # 5. Calculate the loss

    # loss = log (softmax(X_y)) = log ((e ^ (X_y - max(X)) / sum(e ^ (X - max(X))))
    #      = (X_y - max(X)) - log(sum(e ^ (X - max(X))))
    # sum(e ^ (X - max(X))) must >= 1 because the max term is e ^ 0 = 1
    # So we can safely calculate log (softmax(X_y)) without overflow
    loss = -(ori_X_y - m - tl.log(d))

    # Original loss = H(q, p),  with label smoothing regularization = H(q', p) and (label_smoothing / V) = eps
    # H(q', p) = (1 - label_smoothing) * H(q, p) + label_smoothing * H(u, p)
    #          = (1 - label_smoothing) * H(q, p) + eps * sum(logsoftmax(x_i))
    # By using m (global max of xi) and d (sum of e^(xi-m)), we can simplify as:
    #          = (1 - label_smoothing) * H(q, p) + (-sum(x_i * eps) + label_smoothing * (m + logd))
    # Refer to H(q', p) in section 7 of the paper: https://arxiv.org/pdf/1512.00567
    # pytorch: https://github.com/pytorch/pytorch/blob/2981534f54d49fa3a9755c9b0855e7929c2527f0/aten/src/ATen/native/LossNLL.cpp#L516  # noqa: E501,W505
    # See full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issuecomment-2333753087
    if label_smoothing > 0:
        smooth_loss = scaled_x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth_loss

    # Normalize the loss by the number of non-ignored elements if reduction is "mean"
    if reduction == "mean":
        loss = loss / n_non_ignore

    # 6. Specially handle the i==y case where `dx_y = (softmax(x_y) - (1 - label_smoothing) / N`
    X_y = tl.load(X_ptr + y)
    if reduction == "mean":
        X_y += -(1 - label_smoothing) / (n_non_ignore)
    else:
        X_y += -(1 - label_smoothing)

    tl.store(loss_ptr, loss)
    tl.store(X_ptr + y, X_y)


# The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576
# https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2  # the best size we found by manually tuning


def cross_entropy_forward(
    _input: jax.Array,
    target: jax.Array,
    ignore_index: int,
    label_smoothing: float,
    reduction: Literal["mean", "sum", "none"],
) -> tuple[jax.Array, jax.Array]:
    """
    Compute the cross entropy loss and the gradient of the input tensor.

    Note that the gradients are calculated in-place on the input tensor.

    Args:
        _input: The input tensor, shape (batch, vocab size).
        target: The target tensor, shape (batch,).
        ignore_index: The index to ignore in the target.
        label_smoothing: The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction: The reduction to apply to the loss. Can be "mean", "sum", or "none".

    Returns:
        The computed loss and the gradient of the input tensor.
    """
    BT, V = _input.shape
    n_rows = BT

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    # unreduced loss
    loss = jax.ShapeDtypeStruct(shape=(n_rows,), dtype=_input.dtype)

    n_non_ignore = (target != ignore_index).sum()

    # Here we use a trick to store X_ptr gradient in X_ptr so we can save memory
    (loss,) = jt.triton_call(
        _input,
        target,
        n_non_ignore,
        out_shape=(loss,),
        X_stride=get_stride(_input, 0),
        Y_stride=get_stride(target, 0),
        loss_stride=get_stride(loss, 0),
        n_cols=V,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
        grid=(n_rows,),
        kernel=liger_cross_entropy_kernel,
    )
    if reduction == "mean" or reduction == "sum":
        loss = loss.sum()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    return loss, _input


def cross_entropy_backward(
    _input: jax.Array, grad_output: jax.Array, last_layer: bool = False
) -> jax.Array:
    """
    Compute the gradient of the input tensor for the cross entropy loss.

    Note that this assumes :func:`cross_entropy_forward` was called before this function. This ensures that _input
    contains the gradients of the input tensor.

    Args:
        _input: The input tensor, shape (batch, vocab size).
        grad_output: The gradient of the loss. If the grad function has been created with respect to the loss of
            the :func:`cross_entropy_forward` function, this is 1.0.
        last_layer: Whether the cross entropy is the last layer in the model. If True, assumes that the incoming
            gradient is 1.0 and skips unnecessary operations in the backward.

    Returns:
        The gradient of the input tensor.
    """
    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
    # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
    if not last_layer:
        BT, V = _input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

        jt.triton_call(
            _input,
            grad_output,
            out_shape=(),
            X_stride=get_stride(_input, 0),
            n_cols=V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
            grid=(n_rows,),
            kernel=element_mul_kernel,
        )

    return _input


def liger_cross_entropy(
    _input: jax.Array,
    target: jax.Array,
    target_mask: jax.Array | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
    last_layer: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute the Liger Cross Entropy loss and its gradient.

    Note that the gradients are calculated in-place on the input tensor.

    Args:
        _input: The logits input tensor of shape (..., vocab size).
        target: The target tensor of shape (...,). Needs to match the size of _input up to the last axis.
        target_mask: A mask tensor of shape (...,) to indicate which target values to keep (True) and which to
            ignore (False). If provided, sets targets where the mask is False to ignore_index.
            If None, no mask is applied.
        ignore_index: The index to ignore in the target.
        label_smoothing: The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction: The reduction to apply to the loss. Can be "mean", "sum", or "none".
        last_layer: Whether the cross entropy is the last layer in the model. If True, assumes that the incoming
            gradient is 1.0 and skips unnecessary operations in the backward.

    Returns:
        The computed loss and the gradient of the input tensor.
    """
    _input = _input.reshape(-1, _input.shape[-1])
    target = target.reshape(-1)
    assert (
        _input.shape[0] == target.shape[0]
    ), "Input and target must have the same batch size."
    if target_mask is not None:
        target_mask = target_mask.reshape(-1)
        assert (
            target.shape[0] == target_mask.shape[0]
        ), "Target and target mask must have the same shape."
        target = jnp.where(target_mask, target, ignore_index)

    @jax.custom_gradient
    def fwd(x, y):
        loss, x = cross_entropy_forward(x, y, ignore_index, label_smoothing, reduction)

        def backward(grad_output):
            x_grad = cross_entropy_backward(x, grad_output, last_layer=last_layer)
            return x_grad, None

        return loss, backward

    return fwd(_input, target)
