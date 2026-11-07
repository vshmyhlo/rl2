"""Numerically bounded KL diagnostics, separate from training penalties."""

import jax
import jax.numpy as jnp


def k3_diagnostic(log_ratio: jax.Array) -> jax.Array:  # (...) -> (...)
    """(...) k3 estimates capped at 1e30 per token for logging only."""
    log_ratio = jax.lax.stop_gradient(log_ratio)  # (...)
    limit = jnp.asarray(1e30, dtype=log_ratio.dtype)  # ()
    # Bound before exp; use the bounded argument in both terms to preserve nonnegativity.
    bounded = jnp.minimum(log_ratio, jnp.log(limit))  # (...)
    return jnp.minimum(jnp.expm1(bounded) - bounded, limit)  # (...)
