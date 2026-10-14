"""Spatial and non-spatial branches for Craftax symbolic observations."""

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.envs.observations import CraftaxObservation
from rl2.model.encoders.mlp import MLPEncoder
from rl2.model.encoders.rgb import ResidualBlock


class CraftaxEncoder(nn.Module):
    channels: int = 32
    blocks: int = 2
    hidden_size: int = 128
    embedding_size: int = 256
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, obs: CraftaxObservation) -> jax.Array:
        """Encode (B,H,W,C) grid and (B,F) feature leaves into (B,E) embeddings."""
        chex.assert_rank(obs.grid, 4)
        chex.assert_rank(obs.features, 2)
        chex.assert_equal_shape_prefix((obs.grid, obs.features), 1)
        chex.assert_type((obs.grid, obs.features), float)
        grid = obs.grid.astype(self.dtype)  # (B,H,W,C)
        grid = nn.Conv(self.channels, (3, 3), padding="SAME", dtype=self.dtype)(grid)  # (B,H,W,K)
        grid = nn.silu(nn.LayerNorm(dtype=self.dtype)(grid))  # (B,H,W,K)
        for index in range(self.blocks):
            grid = ResidualBlock(self.channels, dtype=self.dtype, name=f"block_{index}")(grid)  # (B,H,W,K)
        # Preserve positions relative to the player, rather than pooling the grid.
        grid = grid.reshape((grid.shape[0], -1))  # (B,H*W*K)
        grid = nn.silu(nn.Dense(self.hidden_size, dtype=self.dtype, name="grid_projection")(grid))  # (B,D)
        features = MLPEncoder(
            hidden_size=self.hidden_size, embedding_size=self.hidden_size, dtype=self.dtype, name="features"
        )(obs.features)  # (B,D)
        combined = jnp.concatenate((grid, features), axis=-1)  # (B,2D)
        combined = nn.Dense(self.embedding_size, dtype=self.dtype, name="projection")(combined)  # (B,E)
        return nn.silu(nn.LayerNorm(dtype=self.dtype, name="output_norm")(combined))  # (B,E)
