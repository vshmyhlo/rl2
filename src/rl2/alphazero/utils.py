"""Shared AlphaZero array utilities."""

import jax
import jax.numpy as jnp


def masked_logits(logits: jax.Array, legal: jax.Array) -> jax.Array:
    """Shift logits by their row maximum and mask illegal actions.

    Args:
        logits: (..., A) floating-point logits over A actions.
        legal: (..., A) boolean mask indicating legal actions.

    Returns:
        (..., A) shifted logits with illegal actions set to the minimum finite
        float32 value. Rows with no legal actions contain only this finite value.
    """
    # Finite masking also keeps terminal nodes with no legal moves well-defined.
    return jnp.where(legal, logits - logits.max(axis=-1, keepdims=True), jnp.finfo(jnp.float32).min)
