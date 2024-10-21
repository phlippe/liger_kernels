# Transfering Liger Kernels to JAX

This repository implements the [Liger Kernels](https://github.com/linkedin/Liger-Kernel), originally implemented for PyTorch in Triton, to JAX using [jax-triton](https://github.com/jax-ml/jax-triton). 

At the moment, we focus on the fused linear cross entropy kernel, combining the linear output layer and the cross entropy loss computation in a single kernel.
A key part of this kernel is that the input tensor of the logits is overwritten in-place with the gradients for the logits. 
This is done to save memory, as the logits are not needed after the loss computation.
We use the in-place changed tensor to compute the gradients for the weights and bias.

However, this in-place operation may not be visible to the XLA compiler, and the weights and bias gradients may be computed using the original logits tensor instead of the gradients tensor.
A quick fix solution for some cases is to add trivial operations to the computation graph, which add dependencies between the kernel output (the loss) and the in-place changed tensor (the gradients).
While this solution works for the test cases presented in this repo, we found them not to be robust and fail for training a LLM like Llama on multi-GPU setups.

## Usage

To reproduce the issue, install the provided conda environment.
Note that some packages may not be needed as these kernels have been originally developed in a larger code base, but we leave in all packages to ensure that the environment is as close as possible to the original.

Then, run the following command:

```bash
py.test liger_kernels/
```

On a single H100, the following tests fail:

```bash

E           AssertionError: 
E           Not equal to tolerance rtol=0.001, atol=0.001
E           x does not match.
E           Mismatched elements: 64 / 64 (100%)
E           Max absolute difference: 8.978701
E           Max relative difference: 3.5081427
E            x: array([[[ 0.      ,  0.      ,  0.      ,  0.      ,  0.      ,
E                     0.      ,  0.      ,  0.      ,  0.      ,  0.      ,
E                     0.      ,  0.      ,  0.      ,  0.      ,  0.      ,...
E            y: array([[[ 5.581061,  1.939753, -8.217423, -3.024954,  1.263693,
E                    -0.965958, -2.374573, -3.700714,  1.441338,  5.212865,
E                     1.691312, -2.802538,  0.788894,  2.426594,  1.609066,...
...
E           AssertionError: 
E           Not equal to tolerance rtol=0.001, atol=0.001
E           bias does not match.
E           Mismatched elements: 50303 / 50304 (100%)
E           Max absolute difference: 1113.0697
E           Max relative difference: 1.021621
E            x: array([2.207514e-05, 6.792242e-06, 4.502681e-05, ..., 1.321617e-05,
E                  6.920744e-06, 6.079221e-06], dtype=float32)
E            y: array([ 155.37679 , -146.15147 ,  338.65143 , ...,   24.157772,
E                  -141.7925  , -174.98781 ], dtype=float32)
...
FAILED liger_kernels/test_liger_fused_linear_cross_entropy.py::test_liger_fused_linear_cross_entropy[False-float32-True-1-4-32-16] - AssertionError: 
FAILED liger_kernels/test_liger_fused_linear_cross_entropy.py::test_liger_fused_linear_cross_entropy[False-float32-True-4-2048-50304-1024] - AssertionError: 
FAILED liger_kernels/test_liger_fused_linear_cross_entropy.py::test_liger_fused_linear_cross_entropy[False-bfloat16-True-1-4-32-16] - AssertionError: 
FAILED liger_kernels/test_liger_fused_linear_cross_entropy.py::test_liger_fused_linear_cross_entropy[False-bfloat16-True-4-2048-50304-1024] - AssertionError: 
======================================================== 4 failed, 16 passed in 24.69s ========================================================
```

All tests pass if `add_trivial_operations=True`. 
If `add_trivial_operations=False`, the gradients are incorrectly computed when the bias is used (which seems to be by chance, ie that for this computation graph, the compiler changes the order).
The returned results above suggest that the logits have been used instead of the input gradients to calculate the gradients of the bias, as the gradients are much larger than expected. 
This suggests an incorrect scheduling of the operations in the compiled function.