"""Left-aligned, right-padded sequence masking for GDN-2."""

import chex
import jax
import jax.numpy as jnp


def prefix_mask(x_len: jax.Array, batch_size: int, sequence_length: int) -> jax.Array:
    """Validate prefix lengths and construct a mask for features or token IDs."""
    chex.assert_scalar_positive(batch_size)
    invalid = jnp.any((x_len < 0) | (x_len > sequence_length))

    def fail_if_invalid(value: jax.Array) -> None:
        if bool(value):
            raise ValueError("x_len must be between 0 and the input sequence length")

    def report_failure() -> None:
        # Under vmap both branches can run; the callback checks the predicate.
        jax.debug.callback(fail_if_invalid, invalid)

    def success() -> None:
        pass

    if isinstance(invalid, jax.core.Tracer):
        jax.lax.cond(invalid, report_failure, success)
    else:
        fail_if_invalid(invalid)
    valid = jnp.arange(sequence_length)[None, :] < x_len[:, None]
    return valid
