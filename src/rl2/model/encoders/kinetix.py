"""MLP encoder for Kinetix's normalized symbolic physics features."""

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn


class KinetixEncoder(nn.Module):
    """Preserve fixed body/joint slots and actuator bindings in the flat input."""

    hidden_size: int = 256
    embedding_size: int = 256
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:  # (B, F) -> (B, E)
        """Encode (B, F) floating-point physics features into (B, E) embeddings."""
        chex.assert_rank(obs, 2)
        chex.assert_type(obs, float)
        x = obs.astype(self.dtype)  # (B, F)
        for index, width in enumerate((self.hidden_size, self.embedding_size)):
            x = nn.Dense(width, dtype=self.dtype, name=f"dense_{index}")(x)  # (B, width)
            x = nn.silu(nn.LayerNorm(dtype=self.dtype, name=f"norm_{index}")(x))  # (B, width)
        return x  # (B, E)
