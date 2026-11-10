"""Spatial and entity encoding of the acting team's visible Lux observation."""

from math import prod

import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.envs.lux_env import LuxObservation
from rl2.model.encoders.conv2d import ResidualBlock


def _rasterize(
    positions: jax.Array,  # (B, N, 2), x/y coordinates
    mask: jax.Array,  # (B, N)
    values: jax.Array,  # (B, N, C)
    width: int,
    height: int,
) -> jax.Array:  # (B, W, H, C)
    """Sum visible entities on each tile, including stacked ships."""
    positions = jnp.clip(positions, 0, jnp.array([width - 1, height - 1]))  # (B, N, 2)
    values = jnp.where(mask[..., None], values, 0)  # (B, N, C)
    grid = jnp.zeros((positions.shape[0], width, height, values.shape[-1]), dtype=values.dtype)  # (B, W, H, C)
    return grid.at[jnp.arange(positions.shape[0])[:, None], positions[..., 0], positions[..., 1]].add(values)


class LuxEncoder(nn.Module):
    """CNN map features plus ship-to-map attention, with ordered own-ship slots.

    Opponent/relic slot permutations are immaterial; own-ship order identifies
    action components. All hidden fields are masked before feature extraction.
    """

    features: int
    channels: int = 16
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, observation: LuxObservation[jax.Array]) -> jax.Array:  # leaves: (*S, ...) -> (*S, F)
        batch_shape = observation.public_params.shape[:-1]
        batch_size = prod(batch_shape)
        obs = jax.tree.map(lambda leaf: leaf.reshape((batch_size, *leaf.shape[len(batch_shape) :])), observation)
        # obs leaves: (B, ...), B = prod(S), including unbatched B=1
        visible = obs.world.sensor_mask  # (B, W, H)
        width, height = visible.shape[-2:]
        unit_count = obs.self.units_mask.shape[-1]
        token_size = 2 * self.channels
        positions, energy, unit_mask = jax.tree.map(
            lambda own, opponent: jnp.concatenate((own, opponent), axis=1),
            (obs.self.units.position, obs.self.units.energy, obs.self.units_mask),
            (obs.opponent.units.position, obs.opponent.units.energy, obs.opponent.units_mask),
        )  # (B, 2U, 2); (B, 2U); (B, 2U)
        positions = jnp.where(unit_mask[..., None], positions, 0)  # (B, 2U, 2)
        energy = jnp.where(unit_mask, energy, 0).astype(jnp.float32) / 400.0  # (B, 2U)
        ship_values = jnp.stack((jnp.ones_like(energy), energy), axis=-1)  # (B, 2U, 2), count/total energy
        own_grid = _rasterize(
            positions[:, :unit_count], unit_mask[:, :unit_count], ship_values[:, :unit_count], width, height
        )  # (B, W, H, 2)
        opponent_grid = _rasterize(
            positions[:, unit_count:], unit_mask[:, unit_count:], ship_values[:, unit_count:], width, height
        )  # (B, W, H, 2)
        relic_grid = _rasterize(
            obs.world.relic_nodes,
            obs.world.relic_nodes_mask,
            jnp.ones((*obs.world.relic_nodes_mask.shape, 1), dtype=jnp.float32),
            width,
            height,
        )  # (B, W, H, 1), visible relic centers only, never hidden scoring tiles

        tile_ids = jnp.where(
            visible, obs.world.map_features.tile_type + 1, 0
        )  # (B, W, H), unknown/empty/nebula/asteroid
        tile_embedding = nn.Embed(num_embeddings=4, features=4, dtype=self.dtype, name="tile_embedding")(tile_ids)
        # tile_embedding: (B, W, H, 4)
        tile_energy = jnp.where(visible, obs.world.map_features.energy, 0).astype(jnp.float32) / 20.0  # (B, W, H)
        coordinates = jnp.stack(
            jnp.meshgrid(jnp.linspace(-1, 1, width), jnp.linspace(-1, 1, height), indexing="ij"), axis=-1
        )  # (W, H, 2), absolute coordinates preserve map orientation
        coordinates = jnp.broadcast_to(coordinates, (batch_size, width, height, 2))  # (B, W, H, 2)
        grid = jnp.concatenate(
            (
                tile_embedding,
                tile_energy[..., None],
                visible[..., None],
                coordinates,
                own_grid,
                opponent_grid,
                relic_grid,
            ),
            axis=-1,
        ).astype(self.dtype)  # (B, W, H, 13)
        grid = nn.Conv(features=self.channels, kernel_size=(3, 3), dtype=self.dtype, name="map_stem")(grid)
        grid = nn.silu(nn.LayerNorm(dtype=self.dtype, name="map_norm")(grid))  # (B, W, H, C)
        grid = ResidualBlock(channels=self.channels, dtype=self.dtype, name="map_local")(grid)  # (B, W, H, C)
        local = grid[jnp.arange(batch_size)[:, None], positions[..., 0], positions[..., 1]]  # (B, 2U, C)
        for index in range(2):
            grid = nn.Conv(
                features=token_size,
                kernel_size=(3, 3),
                strides=(2, 2),
                dtype=self.dtype,
                name=f"map_downsample_{index}",
            )(grid)  # (B, ceil(W/2**(i+1)), ceil(H/2**(i+1)), 2C)
            grid = nn.silu(nn.LayerNorm(dtype=self.dtype, name=f"map_downsample_norm_{index}")(grid))
        map_tokens = grid.reshape((batch_size, -1, token_size))  # (B, P, 2C), P=36 for a 24x24 map

        unit_coordinates = positions.astype(jnp.float32) / jnp.array([width - 1, height - 1]) * 2 - 1  # (B, 2U, 2)
        unit_features = jnp.concatenate((unit_coordinates, energy[..., None], local), axis=-1)  # (B, 2U, C+3)
        units = nn.Dense(features=token_size, dtype=self.dtype, name="ship_projection")(unit_features)  # (B, 2U, 2C)
        roles = jnp.repeat(jnp.arange(2), unit_count)  # (2U,), self/opponent
        role_embedding = nn.Embed(num_embeddings=2, features=token_size, dtype=self.dtype, name="team_embedding")(roles)
        units = nn.silu(units + role_embedding)  # (B, 2U, 2C)

        # Fixed physical scales, not observed batch statistics; keep relative score and progress explicit.
        scores = jnp.stack((obs.self.points, obs.opponent.points), axis=-1).astype(jnp.float32)  # (B, 2)
        context = jnp.concatenate(
            (
                jnp.log1p(scores) / 8.0,
                (scores[:, :1] - scores[:, 1:]) / 100.0,
                jnp.stack((obs.self.wins, obs.opponent.wins), axis=-1) / 5.0,
                obs.world.steps[:, None] / 505.0,
                obs.world.match_steps[:, None] / 100.0,
                obs.public_params / jnp.array([5.0, 50.0, 7.0, 7.0]),
                jnp.stack((obs.self.units_mask.sum(-1), obs.opponent.units_mask.sum(-1)), axis=-1) / unit_count,
                visible.mean(axis=(-2, -1))[:, None],
            ),
            axis=-1,
        )  # (B, 14)
        context = nn.silu(
            nn.Dense(features=token_size, dtype=self.dtype, name="context_projection")(context)
        )  # (B, 2C)
        queries = jnp.concatenate((units, context[:, None]), axis=1)  # (B, 2U+1, 2C)
        attended = nn.MultiHeadDotProductAttention(
            num_heads=4, dtype=self.dtype, force_fp32_for_softmax=True, name="map_attention"
        )(
            nn.LayerNorm(dtype=self.dtype, name="query_norm")(queries), inputs_k=map_tokens
        )  # (B, 2U+1, 2C); global map access without quadratic attention over all tiles
        queries = nn.LayerNorm(dtype=self.dtype, name="attention_norm")(queries + attended)  # (B, 2U+1, 2C)
        units = jnp.where(unit_mask[..., None], queries[:, :-1], 0)  # (B, 2U, 2C)
        own = units[:, :unit_count].reshape((batch_size, -1))  # (B, U*2C), retain command-slot identity
        opponents = units[:, unit_count:]  # (B, U, 2C)
        opponent_mean = opponents.sum(1) / jnp.maximum(obs.opponent.units_mask.sum(-1, keepdims=True), 1)  # (B, 2C)
        # Flatten the coarse map to retain absolute spatial layout for the team-level output head.
        spatial = nn.silu(
            nn.Dense(features=self.features, dtype=self.dtype, name="map_projection")(grid.reshape((batch_size, -1)))
        )  # (B, F)
        combined = jnp.concatenate((own, opponent_mean, queries[:, -1], spatial), axis=-1)  # (B, U*2C+4C+F)
        output = nn.Dense(features=self.features, dtype=self.dtype, name="output_projection")(combined)  # (B, F)
        output = nn.LayerNorm(dtype=self.dtype, name="output_norm")(output)  # (B, F)
        return output.reshape((*batch_shape, self.features))  # (*S, F)
