from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fused_linear_cross_entropy import liger_fused_linear_cross_entropy


@pytest.mark.parametrize("batch_size,context_length,vocab_size,embed_dim", [(1, 4, 32, 16), (4, 2048, 50304, 1024)])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("add_trivial_operations", [True, False])
def test_liger_fused_linear_cross_entropy(
    batch_size: int, context_length: int, vocab_size: int, embed_dim: int, use_bias: bool, dtype: jnp.dtype, add_trivial_operations: bool
):
    """Test the liger fused linear cross entropy kernel."""
    # Generate random inputs
    input_tensor = jax.random.normal(jax.random.PRNGKey(0), (batch_size, context_length, embed_dim)).astype(
        dtype
    ) / np.sqrt(vocab_size)
    target_tensor = jax.random.randint(jax.random.PRNGKey(1), (batch_size, context_length), 0, vocab_size)
    target_mask = jax.random.bernoulli(jax.random.PRNGKey(2), 0.5, (batch_size, context_length)) == 1
    params = {"weight": jax.random.normal(jax.random.PRNGKey(3), (embed_dim, vocab_size))}
    if use_bias:
        params["bias"] = jax.random.normal(jax.random.PRNGKey(4), (vocab_size,))

    # Compute the cross entropy using JAX
    def jax_cross_entropy(x, params, y, mask):
        x = x.reshape(-1, embed_dim)
        logits = jnp.dot(x, params["weight"].astype(x.dtype))
        if "bias" in params:
            logits += params["bias"].astype(x.dtype)
        logits = logits.astype(jnp.float32)
        y = y.reshape(-1)
        mask = mask.reshape(-1).astype(logits.dtype)
        loss_ce = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        loss_ce = ((loss_ce * mask) / mask.sum()).sum()
        return loss_ce

    jax_grad_fn = jax.value_and_grad(jax_cross_entropy, argnums=(0, 1))
    jax_loss, (jax_x_grads, jax_param_grads) = jax_grad_fn(input_tensor, params, target_tensor, target_mask)
    jax_arrays = {"loss": jax_loss, "x": jax_x_grads, "weight": jax_param_grads["weight"]}
    if use_bias:
        jax_arrays["bias"] = jax_param_grads["bias"]
    else:
        jax_arrays["bias"] = None

    # Compute the cross entropy using Liger Kernels
    liger_grad_fn = jax.value_and_grad(
        partial(
            liger_fused_linear_cross_entropy, 
            reduction="mean", 
            last_layer=True, 
            add_trivial_operations=add_trivial_operations,
        ), 
        argnums=(0, 1, 3)
    )
    liger_arrays = {}
    liger_arrays["loss"], (liger_arrays["x"], liger_arrays["weight"], liger_arrays["bias"]) = liger_grad_fn(
        input_tensor, params["weight"], target_tensor, params.get("bias", None), target_mask
    )

    # Compare the results
    jax_arrays = jax.device_get(jax_arrays)
    liger_arrays = jax.device_get(liger_arrays)
    for key in jax_arrays:
        if jax_arrays[key] is not None:
            assert liger_arrays[key].shape == jax_arrays[key].shape, f"The shape of {key} does not match."
            assert liger_arrays[key].dtype == jax_arrays[key].dtype, f"The dtype of {key} does not match."
            assert jnp.all(jnp.isfinite(jax_arrays[key])), f"JAX {key} contains non-finite values: {jax_arrays[key]}."
            assert jnp.all(
                jnp.isfinite(liger_arrays[key])
            ), f"Liger {key} contains non-finite values: {liger_arrays[key]}."
            np.testing.assert_allclose(
                jax_arrays[key].astype(np.float32),
                liger_arrays[key].astype(np.float32),
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"{key} does not match.",
            )
        else:
            assert liger_arrays[key] is None, f"{key} was None in JAX but not Liger."
