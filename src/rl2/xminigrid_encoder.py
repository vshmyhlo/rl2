"""Categorical, spatial encoder for XMiniGrid's egocentric tile/color view."""

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.observation_encoder import ResidualBlock


class XMiniGridEncoder(nn.Module):
    """Keep every grid cell; recurrence supplies history outside the current view."""

    token_size: int = 32
    channels: int = 64
    blocks: int = 2
    embedding_size: int = 256
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:  # (B, H, W, 2) -> (B, E)
        """Encode obs: (B, H, W, 2) integer tile/color IDs into (B, E) features."""
        from xminigrid.core.constants import NUM_COLORS, NUM_TILES

        chex.assert_shape(obs, (None, None, None, 2))
        chex.assert_type(obs, int)
        tiles = nn.Embed(NUM_TILES, self.token_size, dtype=self.dtype, name="tile_embedding")(obs[..., 0])  # (B,H,W,K)
        colors = nn.Embed(NUM_COLORS, self.token_size, dtype=self.dtype, name="color_embedding")(obs[..., 1])  # (B,H,W,K)
        x = jnp.concatenate((tiles, colors), axis=-1)  # (B, H, W, 2K)
        init = nn.initializers.variance_scaling(2.0, "fan_in", "truncated_normal")
        x = nn.Conv(self.channels, (3, 3), padding="SAME", kernel_init=init, dtype=self.dtype)(x)  # (B,H,W,C)
        x = nn.silu(nn.LayerNorm(dtype=self.dtype)(x))  # (B, H, W, C)
        for index in range(self.blocks):
            x = ResidualBlock(self.channels, dtype=self.dtype, name=f"block_{index}")(x)  # (B, H, W, C)
        # Flattening preserves location relative to the agent at bottom-center.
        x = x.reshape((x.shape[0], -1))  # (B, H*W*C)
        x = nn.Dense(self.embedding_size, kernel_init=init, dtype=self.dtype)(x)  # (B, E)
        return nn.silu(nn.LayerNorm(dtype=self.dtype)(x))  # (B, E)
