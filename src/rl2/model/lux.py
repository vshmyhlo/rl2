from math import isqrt, prod, sqrt
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import tokamax
from flax import linen as nn

from rl2.actions import HierarchicalMultiCategoricalPolicy, HierarchicalMultiDiscreteSpace
from rl2.envs.lux_env import LuxObservation
from rl2.lstm import LSTM, LSTMCarry
from rl2.model.actor_critic import ActorCriticOutput, BaseActorCritic
from rl2.model.config import LuxModelConfig
from rl2.model.encoders.conv2d import ResidualBlock

# Shape dimensions: B = local batch size (possibly flattened); T = time; *S = leading batch axes.
# U/V = own/opponent ship slots; N = U+V; R = relic slots; E = N+R+1 entity tokens.
# H/W = map height/width; P = downsampled map tokens; C = spatial CNN channels.
# Q/K = attention query/key counts; Nh = heads; Dh = head_dim; D = Nh*Dh token width.
# F = embedding size; Hg = global LSTM hidden size (equal to D).
# A = action logits per ship; As = sap targets, A = 7+As.
# M/Cv = rasterized entity count/value channels.


def _rasterize(
    positions: jax.Array,  # (B, M, 2), x/y coordinates
    positions_mask: jax.Array,  # (B, M)
    values: jax.Array,  # (B, M, Cv)
    height: int,
    width: int,
) -> jax.Array:  # (B, H, W, Cv)
    """Sum visible entities on each tile, including stacked ships."""
    positions = jnp.clip(positions, 0, jnp.array([width - 1, height - 1]))  # (B, M, 2)
    values = jnp.where(positions_mask[..., None], values, 0)  # (B, M, Cv)
    grid = jnp.zeros((positions.shape[0], height, width, values.shape[-1]), dtype=values.dtype)  # (B, H, W, Cv)
    return grid.at[jnp.arange(positions.shape[0])[:, None], positions[..., 1], positions[..., 0]].add(values)


def _global_context(observation: LuxObservation[jax.Array]) -> jax.Array:  # leaves: (B, ...) -> (B, 14)
    """Public scores, progress, game parameters, visible counts, and coverage."""
    # Fixed physical scales, not observed batch statistics; keep relative score and progress explicit.
    scores = jnp.stack((observation.self.points, observation.opponent.points), axis=-1).astype(jnp.float32)  # (B, 2)
    context = jnp.concatenate(
        (
            jnp.log1p(scores) / 8.0,
            (scores[:, :1] - scores[:, 1:]) / 100.0,
            jnp.stack((observation.self.wins, observation.opponent.wins), axis=-1) / 5.0,
            observation.world.steps[:, None] / 505.0,
            observation.world.match_steps[:, None] / 100.0,
            observation.public_params / jnp.array([5.0, 50.0, 7.0, 7.0]),
            jnp.stack((observation.self.units_mask.sum(-1), observation.opponent.units_mask.sum(-1)), axis=-1) / 16.0,
            observation.world.sensor_mask.mean(axis=(-2, -1))[:, None],
        ),
        axis=-1,
    )  # (B, 14)
    return context  # (B, 14)


class _Attention(nn.Module):
    """Noncausal attention; Triton pads narrow heads for tensor cores."""

    attention_implementation: Literal["triton", "xla"]
    heads: int = 8
    head_dim: int = 32
    dtype: jax.typing.DTypeLike | None = None

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,  # (B, Q, Dx), Dx is the input width
        context: jax.Array,  # (B, K, Dc), Dc is the context width
        context_mask: jax.Array | None = None,  # (B, 1, 1, K) bool; at least one valid key per example
    ) -> jax.Array:  # (B, Q, D)
        query = nn.DenseGeneral(features=(self.heads, self.head_dim), dtype=self.dtype, name="query")(
            tokens
        )  # (B, Q, Nh, Dh)
        key = nn.DenseGeneral(features=(self.heads, self.head_dim), dtype=self.dtype, name="key")(
            context
        )  # (B, K, Nh, Dh)
        value = nn.DenseGeneral(features=(self.heads, self.head_dim), dtype=self.dtype, name="value")(
            context
        )  # (B, K, Nh, Dh)
        # Normalize each logical head before padding; padding must not change its RMS.
        query = nn.RMSNorm(dtype=self.dtype, name="query_norm")(query)  # (B, Q, Nh, Dh)
        key = nn.RMSNorm(dtype=self.dtype, name="key_norm")(key)  # (B, K, Nh, Dh)
        if self.head_dim < 16:
            query, key, value = jax.tree.map(
                lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 16 - self.head_dim))), (query, key, value)
            )  # (B, Q/K, Nh, 16)
        attended = tokamax.dot_product_attention(
            query,
            key,
            value,
            mask=context_mask,
            is_causal=False,
            scale=self.head_dim**-0.5,
            implementation=self.attention_implementation,
        )[..., : self.head_dim]  # (B, Q, Nh, Dh)
        return nn.DenseGeneral(features=self.heads * self.head_dim, axis=(-2, -1), dtype=self.dtype, name="out")(
            attended
        )  # (B, Q, D)


class _SetAttentionBlock(nn.Module):
    """Bidirectional entity attention without slot embeddings or causal masks."""

    attention_implementation: Literal["triton", "xla"]
    heads: int = 8
    head_dim: int = 32
    dtype: jax.typing.DTypeLike | None = None

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,  # (B, E, D), E includes an always-valid global token
        tokens_mask: jax.Array,  # (B, E) bool
    ) -> jax.Array:  # (B, E, D)
        normalized = nn.LayerNorm(dtype=self.dtype, name="attention_norm")(tokens)  # (B, E, D)
        attended = _Attention(
            attention_implementation=self.attention_implementation,
            heads=self.heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            name="attention",
        )(
            normalized, normalized, tokens_mask[:, None, None, :]
        )  # (B, E, D); masking keys excludes padded entities in both directions
        tokens = tokens + attended  # (B, E, D)
        hidden = nn.LayerNorm(dtype=self.dtype, name="ffn_norm")(tokens)  # (B, E, D)
        hidden = nn.gelu(
            nn.Dense(features=4 * self.heads * self.head_dim, dtype=self.dtype, name="ffn_in")(hidden)
        )  # (B, E, 4D)
        hidden = nn.Dense(features=self.heads * self.head_dim, dtype=self.dtype, name="ffn_out")(hidden)  # (B, E, D)
        return jnp.where(tokens_mask[..., None], tokens + hidden, 0)  # (B, E, D)


class LuxFeatures(NamedTuple):
    global_: jax.Array  # (*S, F), invariant to ship slot ordering
    ships: jax.Array  # (*S, U, D), equivariant to own ship slot ordering
    spatial: jax.Array  # (*S, H, W, C), full-resolution map features


class LuxCarry(NamedTuple):
    """Global-token memory for each attention block, invariant to ship slots."""

    global_: tuple[LSTMCarry, ...]  # per-block cell/hidden: (B, Hg)


class _LuxTokens(NamedTuple):
    tokens: jax.Array  # (*S, E, D), global, own ships, opponent ships, relics
    tokens_mask: jax.Array  # (*S, E) bool
    token_types: jax.Array  # (E,) int32: global=0, own=1, opponent=2, relic=3
    map_tokens: jax.Array  # (*S, P, D)
    spatial: jax.Array  # (*S, H, W, C)


class _GlobalMemory(nn.Module):
    """Global-token LSTM with a residual projection and game-level resets."""

    features: int
    dtype: jax.typing.DTypeLike | None = None

    @nn.nowrap
    def _make_recurrent(self, *, parent: nn.Module | None = None) -> LSTM:
        return LSTM(features=self.features, dtype=self.dtype, name="lstm", parent=parent)

    @nn.nowrap
    def initial_carry(self, actor_count: int) -> LSTMCarry:
        """Return (B, Hg) float32 zero cell/hidden states."""
        return self._make_recurrent().initial_carry(actor_count)  # leaves: (B, Hg)

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,  # (B, T, D)
        carry: LSTMCarry,  # cell/hidden: (B, Hg)
        episode_start: jax.Array,  # (B, T) bool
    ) -> tuple[jax.Array, LSTMCarry]:  # tokens: (B, T, D); carry leaves: (B, Hg)
        inputs = nn.Dense(features=self.features, dtype=self.dtype, name="input")(nn.relu(tokens))  # (B, T, Hg)
        carry, memory = self._make_recurrent(parent=self)(
            inputs, carry, episode_start
        )  # carry leaves: (B, Hg); memory: (B, T, Hg)
        projected = nn.Dense(features=tokens.shape[-1], use_bias=False, dtype=self.dtype, name="output")(
            memory
        )  # (B, T, D)
        return tokens + projected, carry  # (B, T, D); leaves: (B, Hg)


class LuxAttentionBlock(nn.Module):
    """Global-token recurrence, type embeddings, map retrieval, then entity mixing."""

    attention_implementation: Literal["triton", "xla"]
    heads: int = 8
    head_dim: int = 32
    dtype: jax.typing.DTypeLike | None = None

    @nn.nowrap
    def _make_global_memory(self, *, parent: nn.Module | None = None) -> _GlobalMemory:
        return _GlobalMemory(features=self.heads * self.head_dim, dtype=self.dtype, name="global_memory", parent=parent)

    @nn.nowrap
    def initial_carry(self, actor_count: int) -> LSTMCarry:
        """Return (B, Hg) global cell/hidden states."""
        return self._make_global_memory().initial_carry(actor_count)  # leaves: (B, Hg)

    def setup(self) -> None:
        self.global_memory = self._make_global_memory(parent=self)
        self.token_type_embedding = nn.Embed(
            num_embeddings=4, features=self.heads * self.head_dim, dtype=self.dtype, name="token_type_embedding"
        )
        self.input_norm = nn.LayerNorm(dtype=self.dtype, name="input_norm")
        self.map_attention = _Attention(
            attention_implementation=self.attention_implementation,
            heads=self.heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            name="map_attention",
        )
        self.set_attention = _SetAttentionBlock(
            attention_implementation=self.attention_implementation,
            heads=self.heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            name="set_attention",
        )

    def __call__(
        self,
        tokens: jax.Array,  # (B, T, E, D)
        tokens_mask: jax.Array,  # (B, T, E) bool
        carry: LSTMCarry,  # cell/hidden: (B, Hg)
        episode_start: jax.Array,  # (B, T) bool
        map_tokens: jax.Array,  # (B, T, P, D)
        token_types: jax.Array,  # (E,) int32: global=0, own=1, opponent=2, relic=3
    ) -> tuple[jax.Array, LSTMCarry]:  # tokens: (B, T, E, D); carry leaves: (B, Hg)
        global_token, carry = self.global_memory(
            tokens[..., 0, :], carry, episode_start
        )  # token: (B, T, D); carry leaves: (B, Hg)
        tokens = tokens.at[..., 0, :].set(global_token)  # (B, T, E, D)
        tokens = tokens + self.token_type_embedding(token_types)  # (B, T, E, D)
        token_shape = tokens.shape
        batch_size = prod(tokens.shape[:-2])
        tokens = tokens.reshape((batch_size, *tokens.shape[-2:]))  # (B*T, E, D)
        map_tokens = map_tokens.reshape((batch_size, *map_tokens.shape[-2:]))  # (B*T, P, D)
        tokens_mask = tokens_mask.reshape((batch_size, tokens_mask.shape[-1]))  # (B*T, E)
        attended = self.map_attention(self.input_norm(tokens), map_tokens)  # (B*T, E, D)
        tokens = jnp.where(tokens_mask[..., None], tokens + attended, 0)  # (B*T, E, D)
        tokens = self.set_attention(tokens, tokens_mask).reshape(token_shape)  # (B, T, E, D)
        return tokens, carry  # (B, T, E, D); leaves: (B, Hg)


class LuxMapEncoder(nn.Module):
    """Encode visible map tiles and entities into spatial features and map tokens."""

    features: int
    channels: int = 64
    dtype: jax.typing.DTypeLike | None = None

    def setup(self) -> None:
        self.tile_embedding = nn.Embed(num_embeddings=4, features=4, dtype=self.dtype, name="tile_embedding")
        self.map_stem = nn.Conv(features=self.channels, kernel_size=(3, 3), dtype=self.dtype, name="map_stem")
        self.map_norm = nn.LayerNorm(dtype=self.dtype, name="map_norm")
        self.map_local = ResidualBlock(channels=self.channels, dtype=self.dtype, name="map_local")
        self.map_downsample = tuple(
            nn.Conv(
                features=self.features, kernel_size=(3, 3), strides=(2, 2), dtype=self.dtype, name=f"map_downsample_{i}"
            )
            for i in range(2)
        )
        self.map_downsample_norm = tuple(
            nn.LayerNorm(dtype=self.dtype, name=f"map_downsample_norm_{i}") for i in range(2)
        )

    def __call__(
        self,
        observation: LuxObservation[jax.Array],  # leaves: (B, ...)
        positions: jax.Array,  # (B, N, 2), masked positions replaced with zero
        positions_mask: jax.Array,  # (B, N), own ships followed by opponents
        energy: jax.Array,  # (B, N), normalized and zero for masked ships
    ) -> tuple[jax.Array, jax.Array]:  # spatial: (B, H, W, C); map tokens: (B, P, D)
        sensor_mask = observation.world.sensor_mask  # (B, H, W)
        batch_size, height, width = sensor_mask.shape
        own_ship_count = observation.self.units_mask.shape[-1]
        ship_values = jnp.stack((jnp.ones_like(energy), energy), axis=-1)  # (B, N, 2), count/total energy
        own_grid = _rasterize(
            positions[:, :own_ship_count],
            positions_mask[:, :own_ship_count],
            ship_values[:, :own_ship_count],
            height,
            width,
        )  # (B, H, W, 2)
        opponent_grid = _rasterize(
            positions[:, own_ship_count:],
            positions_mask[:, own_ship_count:],
            ship_values[:, own_ship_count:],
            height,
            width,
        )  # (B, H, W, 2)
        relic_grid = _rasterize(
            observation.world.relic_nodes,
            observation.world.relic_nodes_mask,
            jnp.ones((*observation.world.relic_nodes_mask.shape, 1), dtype=jnp.float32),
            height=height,
            width=width,
        )  # (B, H, W, 1), visible relic centers only, never hidden scoring tiles

        tile_ids = jnp.where(
            sensor_mask, observation.world.map_features.tile_type + 1, 0
        )  # (B, H, W), unknown/empty/nebula/asteroid
        tile_embedding = self.tile_embedding(tile_ids)  # (B, H, W, 4)
        tile_energy = (
            jnp.where(sensor_mask, observation.world.map_features.energy, 0).astype(jnp.float32) / 20.0
        )  # (B, H, W)
        coordinates = jnp.stack(
            jnp.meshgrid(jnp.linspace(-1, 1, width), jnp.linspace(-1, 1, height), indexing="xy"), axis=-1
        )  # (H, W, 2), x/y coordinates preserve map orientation
        coordinates = jnp.broadcast_to(coordinates, (batch_size, height, width, 2))  # (B, H, W, 2)
        grid = jnp.asarray(
            jnp.concatenate(
                (
                    tile_embedding,
                    tile_energy[..., None],
                    sensor_mask[..., None],
                    coordinates,
                    own_grid,
                    opponent_grid,
                    relic_grid,
                ),
                axis=-1,
            ),
            dtype=self.dtype,
        )  # (B, H, W, 13)
        grid = self.map_stem(grid)  # (B, H, W, C)
        grid = nn.silu(self.map_norm(grid))  # (B, H, W, C)
        grid = self.map_local(grid)  # (B, H, W, C)
        spatial = grid  # (B, H, W, C), retained for command target scoring
        for index in range(2):
            grid = self.map_downsample[index](grid)  # (B, ceil(H/2**(index+1)), ceil(W/2**(index+1)), D)
            grid = nn.silu(self.map_downsample_norm[index](grid))  # (B, ceil(H/2**(index+1)), ceil(W/2**(index+1)), D)
        map_tokens = grid.reshape((batch_size, -1, self.features))  # (B, P, D), P=36 for a 24x24 map

        return spatial, map_tokens  # (B, H, W, C); (B, P, D)


class LuxEncoder(nn.Module):
    """Spatial and typed entity features around the recurrent attention blocks.

    Tokens are ordered as global, own ships, opponent ships, and relics.
    Own-ship rows remain aligned with action slots; global features are invariant
    to each set's permutations and padding. Maps use H, W axes; positions and
    coordinate features remain x, y.
    """

    features: int
    token_size: int
    channels: int = 64
    dtype: jax.typing.DTypeLike | None = None

    def setup(self) -> None:
        token_size = self.token_size
        self.entity_norm = nn.LayerNorm(dtype=self.dtype, name="entity_norm")
        self.map_projection = nn.Dense(features=self.features, dtype=self.dtype, name="map_projection")
        self.output_projection = nn.Dense(features=self.features, dtype=self.dtype, name="output_projection")
        self.output_norm = nn.LayerNorm(dtype=self.dtype, name="output_norm")
        self.map_encoder = LuxMapEncoder(
            features=self.token_size, channels=self.channels, dtype=self.dtype, name="map_encoder"
        )
        self.ship_projection = nn.Dense(features=token_size, dtype=self.dtype, name="ship_projection")
        self.relic_projection = nn.Dense(features=token_size, dtype=self.dtype, name="relic_projection")
        self.entity_type_embedding = nn.Embed(
            num_embeddings=3, features=token_size, dtype=self.dtype, name="entity_type_embedding"
        )
        self.context_projection = nn.Dense(features=token_size, dtype=self.dtype, name="context_projection")

    def _encode_entities(
        self,
        observation: LuxObservation[jax.Array],  # leaves: (B, ...)
        positions: jax.Array,  # (B, N, 2), masked positions replaced with zero
        energy: jax.Array,  # (B, N), normalized and zero for masked ships
        spatial: jax.Array,  # (B, H, W, C)
    ) -> tuple[jax.Array, jax.Array]:  # ships: (B, N, D); relics: (B, R, D)
        batch_size, height, width, _ = spatial.shape
        own_ship_count = observation.self.units_mask.shape[-1]
        opponent_ship_count = observation.opponent.units_mask.shape[-1]
        local = spatial[jnp.arange(batch_size)[:, None], positions[..., 1], positions[..., 0]]  # (B, N, C)
        ship_coordinates = positions.astype(jnp.float32) / jnp.array([width - 1, height - 1]) * 2 - 1  # (B, N, 2)
        ship_features = jnp.concatenate((ship_coordinates, energy[..., None], local), axis=-1)  # (B, N, C+3)
        ships = self.ship_projection(ship_features)  # (B, N, D)
        roles = jnp.concatenate(
            (jnp.zeros(own_ship_count, jnp.int32), jnp.ones(opponent_ship_count, jnp.int32))
        )  # (N,)
        role_embedding = self.entity_type_embedding(roles)  # (N, D)
        ships = nn.silu(ships + role_embedding)  # (B, N, D)

        relic_positions = jnp.where(
            observation.world.relic_nodes_mask[..., None], observation.world.relic_nodes, 0
        )  # (B, R, 2)
        relic_local = spatial[
            jnp.arange(batch_size)[:, None], relic_positions[..., 1], relic_positions[..., 0]
        ]  # (B, R, C)
        relic_coordinates = (
            relic_positions.astype(jnp.float32) / jnp.array([width - 1, height - 1]) * 2 - 1
        )  # (B, R, 2)
        relic_features = jnp.concatenate((relic_coordinates, relic_local), axis=-1)  # (B, R, C+2)
        relics = nn.silu(
            self.relic_projection(relic_features) + self.entity_type_embedding(jnp.asarray(2))
        )  # (B, R, D)

        return ships, relics  # (B, N, D); (B, R, D)

    def prepare(self, observation: LuxObservation[jax.Array]) -> _LuxTokens:  # leaves: (*S, ...)
        batch_shape = observation.public_params.shape[:-1]
        batch_size = prod(batch_shape)
        observation = jax.tree.map(
            lambda leaf: leaf.reshape((batch_size, *leaf.shape[len(batch_shape) :])), observation
        )
        # observation leaves: (B, ...), B = prod(S), including unbatched B=1
        sensor_mask = observation.world.sensor_mask  # (B, H, W)
        height, width = sensor_mask.shape[-2:]
        token_size = self.token_size
        positions, energy, positions_mask = jax.tree.map(
            lambda own, opponent: jnp.concatenate((own, opponent), axis=1),
            (observation.self.units.position, observation.self.units.energy, observation.self.units_mask),
            (observation.opponent.units.position, observation.opponent.units.energy, observation.opponent.units_mask),
        )  # (B, N, 2); (B, N); (B, N)
        positions = jnp.where(positions_mask[..., None], positions, 0)  # (B, N, 2)
        energy = jnp.where(positions_mask, energy, 0).astype(jnp.float32) / 400.0  # (B, N)
        spatial, map_tokens = self.map_encoder(
            observation, positions, positions_mask, energy
        )  # (B, H, W, C); (B, P, D)
        ships, relics = self._encode_entities(observation, positions, energy, spatial)  # (B, N, D); (B, R, D)
        context = _global_context(observation)  # (B, 14)
        context = nn.silu(self.context_projection(context))  # (B, D)
        tokens = jnp.concatenate((context[:, None], ships, relics), axis=1)  # (B, E, D)
        tokens_mask = jnp.concatenate(
            (jnp.ones((batch_size, 1), bool), positions_mask, observation.world.relic_nodes_mask), axis=1
        )  # (B, E)
        tokens = jnp.where(tokens_mask[..., None], tokens, 0)  # (B, E, D)
        token_types = jnp.concatenate(
            (
                jnp.zeros(1, jnp.int32),
                jnp.ones(observation.self.units_mask.shape[-1], jnp.int32),
                jnp.full(observation.opponent.units_mask.shape[-1], 2, jnp.int32),
                jnp.full(observation.world.relic_nodes_mask.shape[-1], 3, jnp.int32),
            )
        )  # (E,)
        return _LuxTokens(
            tokens=tokens.reshape((*batch_shape, tokens.shape[1], token_size)),  # (*S, E, D)
            tokens_mask=tokens_mask.reshape((*batch_shape, tokens_mask.shape[1])),  # (*S, E)
            token_types=token_types,  # (E,)
            map_tokens=map_tokens.reshape((*batch_shape, map_tokens.shape[1], token_size)),  # (*S, P, D)
            spatial=spatial.reshape((*batch_shape, height, width, self.channels)),  # (*S, H, W, C)
        )

    def summarize(self, encoded: _LuxTokens, observation: LuxObservation[jax.Array]) -> LuxFeatures:
        # Batched leaves: (*S, ...); encoded.token_types: (E,).
        own_ship_count = observation.self.units_mask.shape[-1]
        total_ship_count = own_ship_count + observation.opponent.units_mask.shape[-1]
        tokens = jnp.where(encoded.tokens_mask[..., None], self.entity_norm(encoded.tokens), 0)  # (*S, E, D)
        own = tokens[..., 1 : 1 + own_ship_count, :]  # (*S, U, D)
        own_mean = own.sum(-2) / jnp.maximum(observation.self.units_mask.sum(-1, keepdims=True), 1)  # (*S, D)
        opponents = tokens[..., 1 + own_ship_count : 1 + total_ship_count, :]  # (*S, V, D)
        opponent_mean = opponents.sum(-2) / jnp.maximum(
            observation.opponent.units_mask.sum(-1, keepdims=True), 1
        )  # (*S, D)
        spatial = nn.silu(self.map_projection(encoded.map_tokens.mean(-2)))  # (*S, F)
        combined = jnp.concatenate((own_mean, opponent_mean, tokens[..., 0, :], spatial), axis=-1)  # (*S, 3D+F)
        output = self.output_norm(self.output_projection(combined))  # (*S, F)
        return LuxFeatures(global_=output, ships=own, spatial=encoded.spatial)

    def encode_features(self, observation: LuxObservation[jax.Array]) -> LuxFeatures:  # leaves: (*S, ...)
        """Return CNN/entity features without running the recurrent attention blocks."""
        return self.summarize(self.prepare(observation), observation)

    def __call__(self, observation: LuxObservation[jax.Array]) -> jax.Array:  # leaves: (*S, ...) -> (*S, F)
        return self.encode_features(observation).global_  # (*S, F)


class LuxModel(nn.Module):
    """Encode visible state and decode recurrent token features into commands and value."""

    embedding_size: int
    token_size: int
    channels: int
    action_space: HierarchicalMultiDiscreteSpace
    dtype: jax.typing.DTypeLike | None = None

    def setup(self) -> None:
        groups = self.action_space.group_sizes
        side = isqrt(groups[-1])
        if len(groups) != 3 or groups[:2] != (1, 4) or side < 3 or side % 2 != 1 or side * side != groups[-1]:
            raise ValueError("Lux requires stay/move/sap groups (1, 4, odd_square) with sap range >= 1")
        self.sap_range = side // 2
        self.encoder = LuxEncoder(
            features=self.embedding_size,
            token_size=self.token_size,
            channels=self.channels,
            dtype=self.dtype,
        )
        self.ship_hidden = nn.Dense(features=self.embedding_size, dtype=self.dtype)
        self.ship_norm = nn.LayerNorm(dtype=self.dtype)
        self.type_output = nn.Dense(features=3, kernel_init=nn.initializers.orthogonal(0.01), dtype=self.dtype)
        self.move_output = nn.Dense(features=4, kernel_init=nn.initializers.orthogonal(0.01), dtype=self.dtype)
        self.sap_query = nn.Dense(
            features=self.channels, kernel_init=nn.initializers.orthogonal(0.01), dtype=self.dtype
        )
        self.sap_key = nn.Dense(features=self.channels, dtype=self.dtype)
        self.sap_offset_hidden = nn.Dense(features=self.channels, dtype=self.dtype)
        self.sap_offset = nn.Dense(features=self.channels, use_bias=False, dtype=self.dtype)
        self.value_hidden = nn.Dense(features=self.embedding_size, dtype=self.dtype)
        self.value_norm = nn.LayerNorm(dtype=self.dtype)
        self.value_output = nn.Dense(features=1, kernel_init=nn.initializers.orthogonal(1.0), dtype=self.dtype)

    def encode(self, observation: LuxObservation[jax.Array]) -> LuxFeatures:  # leaves: (*S, ...)
        return self.encoder.encode_features(observation)  # leaves: (*S, ...)

    def _sap_logits(
        self,
        ships: jax.Array,  # (*S, U, F)
        spatial: jax.Array,  # (*S, H, W, C)
        observation: LuxObservation[jax.Array],  # leaves: (*S, ...)
    ) -> jax.Array:  # (*S, U, As)
        batch_shape = ships.shape[:-2]
        batch_size = prod(batch_shape)
        own_ship_count = ships.shape[-2]
        height, width = spatial.shape[-3:-1]
        # Match LuxEnv's x-major, y-minor relative target enumeration.
        axis = jnp.arange(-self.sap_range, self.sap_range + 1)  # (2*sap_range+1,)
        offsets = jnp.stack(jnp.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))  # (As, 2)
        positions = jnp.where(observation.self.units_mask[..., None], observation.self.units.position, 0)  # (*S, U, 2)
        targets = positions.reshape((batch_size, own_ship_count, 1, 2)) + offsets  # (B, U, As, 2)
        targets = jnp.clip(targets, 0, jnp.array([width - 1, height - 1]))  # (B, U, As, 2); off-map commands are masked
        keys = self.sap_key(spatial).reshape((batch_size, height, width, self.channels))  # (B, H, W, C)
        keys = keys[jnp.arange(batch_size)[:, None, None], targets[..., 1], targets[..., 0]]  # (B, U, As, C)
        normalized_offsets = jnp.asarray(offsets, dtype=self.dtype) / self.sap_range  # (As, 2)
        offset_features = jnp.concatenate((normalized_offsets, jnp.abs(normalized_offsets)), axis=-1)  # (As, 4)
        relative = self.sap_offset(nn.silu(self.sap_offset_hidden(offset_features)))  # (As, C)
        keys = keys + relative  # (B, U, As, C)
        queries = self.sap_query(ships).reshape((batch_size, own_ship_count, self.channels))  # (B, U, C)
        # Keep close target scores distinct instead of rounding the dot product to bf16.
        sap = jnp.einsum("bukc,buc->buk", keys, queries, preferred_element_type=jnp.float32) / sqrt(
            self.channels
        )  # (B, U, As)
        sap = sap.reshape((*batch_shape, own_ship_count, offsets.shape[0]))  # (*S, U, As)
        return sap  # (*S, U, As)

    def heads(
        self,
        features: LuxFeatures,  # global: (*S, F); ships: (*S, U, D); spatial: (*S, H, W, C)
        observation: LuxObservation[jax.Array],  # leaves: (*S, ...)
        actions_mask: jax.Array,  # (*S, U, A)
    ) -> tuple[HierarchicalMultiCategoricalPolicy, jax.Array]:  # logits: (*S, U, A); value: (*S,)
        own_ship_count = features.ships.shape[-2]
        context = jnp.broadcast_to(
            features.global_[..., None, :], (*features.global_.shape[:-1], own_ship_count, self.embedding_size)
        )  # (*S, U, F)
        ships = jnp.concatenate((features.ships, context), axis=-1)  # (*S, U, D+F)
        ships = nn.silu(self.ship_norm(self.ship_hidden(ships)))  # (*S, U, F)
        types = self.type_output(ships).astype(jnp.float32)  # (*S, U, 3), stay/move/sap
        moves = self.move_output(ships).astype(jnp.float32)  # (*S, U, 4), north/east/south/west

        sap = self._sap_logits(ships, features.spatial, observation)  # (*S, U, As)
        logits = jnp.concatenate((types, moves, sap), axis=-1)  # (*S, U, A)
        policy = HierarchicalMultiCategoricalPolicy(
            raw_logits=logits, actions_mask=actions_mask, group_sizes=self.action_space.group_sizes
        )
        critic = nn.silu(self.value_norm(self.value_hidden(features.global_)))  # (*S, F)
        value = self.value_output(critic).squeeze(-1).astype(jnp.float32)  # (*S,)
        return policy, value


class LuxActorCritic(BaseActorCritic[LuxObservation[jax.Array], LuxCarry, HierarchicalMultiCategoricalPolicy]):
    """Global-token memory in each attention block, retained across matches."""

    action_space: HierarchicalMultiDiscreteSpace
    model: LuxModelConfig
    dtype: jax.typing.DTypeLike | None = None

    @nn.nowrap
    def _make_block(self, index: int, *, parent: nn.Module | None = None) -> LuxAttentionBlock:
        return LuxAttentionBlock(
            attention_implementation=self.model.attention_implementation,
            heads=self.model.heads,
            head_dim=self.model.head_dim,
            dtype=self.dtype,
            name=f"block_{index}",
            parent=parent,
        )

    @nn.nowrap
    def initial_carry(self, actor_count: int) -> LuxCarry:
        return LuxCarry(
            global_=tuple(
                self._make_block(index).initial_carry(actor_count) for index in range(self.model.num_blocks)
            ),  # per-block leaves: (B, Hg)
        )

    def setup(self) -> None:
        self.lux = LuxModel(
            embedding_size=self.model.embedding_size,
            token_size=self.model.heads * self.model.head_dim,
            channels=self.model.channels,
            action_space=self.action_space,
            dtype=self.dtype,
            name="lux",
        )
        self.blocks = tuple(self._make_block(index, parent=self) for index in range(self.model.num_blocks))

    def _encode(
        self,
        observation: LuxObservation[jax.Array],  # leaves: (B, T, ...)
        carry: LuxCarry,  # leaves: (B, ...)
        episode_start: jax.Array,  # (B, T)
    ) -> tuple[LuxFeatures, LuxCarry]:  # features: (B, T, ...); carry: (B, ...)
        """Interleave global-token recurrence with within-observation attention."""
        encoded = self.lux.encoder.prepare(observation)  # batched leaves: (B, T, ...); token_types: (E,)
        global_carries: list[LSTMCarry] = []  # per-block leaves: (B, Hg)
        for block, global_carry in zip(self.blocks, carry.global_, strict=True):
            tokens, global_carry = block(
                tokens=encoded.tokens,
                tokens_mask=encoded.tokens_mask,
                carry=global_carry,  # leaves: (B, Hg)
                episode_start=episode_start,  # (B, T)
                map_tokens=encoded.map_tokens,
                token_types=encoded.token_types,
            )  # tokens: (B, T, E, D); carry leaves: (B, Hg)
            encoded = encoded._replace(tokens=tokens)  # batched leaves: (B, T, ...); token_types: (E,)
            global_carries.append(global_carry)
        features = self.lux.encoder.summarize(encoded, observation)  # leaves: (B, T, ...)
        carry = LuxCarry(global_=tuple(global_carries))  # per-block leaves: (B, Hg)
        return features, carry  # leaves: (B, T, ...); (B, ...)

    def __call__(
        self,
        observation: LuxObservation[jax.Array],  # leaves: (B, T, ...)
        model_carry: LuxCarry,  # per-block cell/hidden: (B, Hg)
        episode_start: jax.Array,  # (B, T)
        actions_mask: jax.Array,  # (B, T, U, A)
    ) -> tuple[ActorCriticOutput[HierarchicalMultiCategoricalPolicy], LuxCarry]:
        # logits: (B, T, U, A); value: (B, T); carry: (B, ...)
        """Scan global memory over time, resetting at game starts."""
        features, model_carry = self._encode(observation, model_carry, episode_start)  # (B, T, ...); (B, ...)
        policy, value = self.lux.heads(features, observation, actions_mask)  # logits: (B, T, U, A); value: (B, T)
        return ActorCriticOutput(policy=policy, value=value), model_carry

    def step(
        self,
        observation: LuxObservation[jax.Array],  # leaves: (B, ...)
        model_carry: LuxCarry,  # per-block cell/hidden: (B, Hg)
        episode_start: jax.Array,  # (B,)
        actions_mask: jax.Array,  # (B, U, A)
    ) -> tuple[ActorCriticOutput[HierarchicalMultiCategoricalPolicy], LuxCarry]:
        # logits: (B, U, A); value: (B,); carry: (B, ...)
        """Process one observation, retaining global history until the next game."""
        sequence, starts, masks = jax.tree.map(
            lambda leaf: leaf[:, None], (observation, episode_start, actions_mask)
        )  # observation: (B, 1, ...); starts: (B, 1); masks: (B, 1, U, A)
        output, model_carry = self(sequence, model_carry, starts, masks)
        policy, value = jax.tree.map(
            lambda leaf: leaf[:, 0], (output.policy, output.value)
        )  # logits: (B, U, A); value: (B,)
        return ActorCriticOutput(policy=policy, value=value), model_carry
