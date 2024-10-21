from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from cross_entropy import liger_cross_entropy


@pytest.mark.parametrize(
    "batch_size,context_length,vocab_size", [(3, 4, 8), (4, 128, 50304)]
)
@pytest.mark.parametrize("mask_prob", [0.9, 0.5])
def test_liger_cross_entropy(
    batch_size: int, context_length: int, vocab_size: int, mask_prob: float
):
    """Test the liger cross entropy kernel."""
    # Generate random inputs
    input_tensor = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, context_length, vocab_size)
    )
    target_tensor = jax.random.randint(
        jax.random.PRNGKey(0), (batch_size, context_length), 0, vocab_size
    )
    target_mask = (
        jax.random.bernoulli(
            jax.random.PRNGKey(0), mask_prob, (batch_size, context_length)
        )
        == 1
    )

    # Compute the cross entropy using JAX
    def jax_cross_entropy(x, y, mask):
        x = x.reshape(-1, vocab_size)
        y = y.reshape(-1)
        mask = mask.reshape(-1).astype(x.dtype)
        loss_ce = optax.softmax_cross_entropy_with_integer_labels(x, y)
        loss_ce = (loss_ce * mask).sum() / mask.sum()
        return loss_ce

    jax_grad_fn = jax.value_and_grad(jax_cross_entropy)
    jax_loss, jax_grads = jax_grad_fn(input_tensor, target_tensor, target_mask)

    # Compute the cross entropy using Liger Kernels
    liger_grad_fn = jax.value_and_grad(
        partial(liger_cross_entropy, reduction="mean", last_layer=True)
    )
    liger_loss, liger_grads = liger_grad_fn(input_tensor, target_tensor, target_mask)

    # Check the shape
    assert liger_loss.shape == jax_loss.shape, "The loss shapes do not match."
    assert liger_grads.shape == jax_grads.shape, "The gradient shapes do not match."

    # Check the values
    assert jnp.all(jnp.isfinite(liger_loss))
    assert jnp.all(liger_loss >= 0.0)

    jax_loss, jax_grads = jax.device_get((jax_loss, jax_grads))
    liger_loss, liger_grads = jax.device_get((liger_loss, liger_grads))

    np.testing.assert_allclose(
        jax_loss, liger_loss, rtol=1e-5, atol=1e-5, err_msg="Losses do not match."
    )
    np.testing.assert_allclose(
        jax_grads, liger_grads, rtol=1e-5, atol=1e-5, err_msg="Gradients do not match."
    )
