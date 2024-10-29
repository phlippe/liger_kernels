import jax
import numpy as np
import triton
import triton.language as tl


@triton.jit
def element_mul_kernel(
    _,  # alias for X_ptr
    grad_output_ptr,
    X_ptr,
    X_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This function multiplies each element of the tensor pointed by X_ptr with the value pointed by grad_output_ptr.
    The multiplication is performed in-place on the tensor pointed by X_ptr.

    Parameters:
    X_ptr: Pointer to the input tensor.
    X_stride (int): The stride of the input tensor.
    grad_output_ptr: Pointer to the gradient output value.
    n_cols (int): The number of columns in the input tensor.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # Get the program ID and convert it to int64 to avoid overflow
    program_id = tl.program_id(0).to(tl.int64)

    # Locate the start index
    X_ptr += program_id * X_stride

    # Load the gradient output value
    grad_output = tl.load(grad_output_ptr)

    # Perform the element-wise multiplication
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)


def get_stride(array: jax.Array | jax.ShapeDtypeStruct, axis: int) -> int:
    """
    Returns the stride of a JAX array at a given axis.

    To calculate all strides, use get_strides.

    Args:
        array: JAX array or shape-dtype struct.
        axis: The axis at which to calculate the stride.

    Returns:
        The stride of the array at the given axis.
    """
    if axis < 0:
        axis += len(array.shape)
    shape = array.shape
    size = array.size
    stride = size // np.prod(shape[: axis + 1])
    return int(stride)
