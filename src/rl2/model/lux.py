"""Lux set-attention policy with aligned ship decisions and spatial sap targets."""

from math import isqrt, prod, sqrt
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import tokamax
from flax import linen as nn

from rl2.actions import HierarchicalMultiCategoricalPolicy, HierarchicalMultiDiscreteSpace
from rl2.envs.lux_env import LuxObservation
from rl2.lstm import LSTMStack, LSTMStackCarry
from rl2.model.actor_critic import ActorCriticOutput, BaseActorCritic
from rl2.model.config import LuxModelConfig
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


class _Attention(nn.Module):
    """Four-head noncausal attention; Triton pads narrow heads for tensor cores."""

    features: int
    implementation: Literal["triton", "xla"]
    dtype: jax.typing.DTypeLike

    @nn.compact
    def __call__(
        self,
        queries: jax.Array,  # (B, Q, D)
        keys: jax.Array,  # (B, K, D)
        mask: jax.Array | None = None,  # (B, 1, 1, K) bool; at least one valid key per example
    ) -> jax.Array:  # (B, Q, D)
        head_size = self.features // 4
        query = nn.DenseGeneral(features=(4, head_size), dtype=self.dtype, name="query")(queries)  # (B, Q, 4, d)
        key = nn.DenseGeneral(features=(4, head_size), dtype=self.dtype, name="key")(keys)  # (B, K, 4, d)
        value = nn.DenseGeneral(features=(4, head_size), dtype=self.dtype, name="value")(keys)  # (B, K, 4, d)
        if head_size < 16:
            query, key, value = jax.tree.map(
                lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 16 - head_size))), (query, key, value)
            )  # (B, Q/K, 4, 16)
        attended = tokamax.dot_product_attention(
            query, key, value, mask=mask, is_causal=False, scale=head_size**-0.5, implementation=self.implementation
        )[..., :head_size]  # (B, Q, 4, d)
        return nn.DenseGeneral(features=self.features, axis=(-2, -1), dtype=self.dtype, name="out")(
            attended
        )  # (B, Q, D)


class _SetAttentionBlock(nn.Module):
    """Bidirectional entity attention without slot embeddings or causal masks."""

    features: int
    implementation: Literal["triton", "xla"]
    dtype: jax.typing.DTypeLike

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,  # (B, E, D), E includes an always-valid global token
        valid: jax.Array,  # (B, E) bool
    ) -> jax.Array:  # (B, E, D)
        normalized = nn.LayerNorm(dtype=self.dtype, name="attention_norm")(tokens)  # (B, E, D)
        attended = _Attention(
            features=self.features, implementation=self.implementation, dtype=self.dtype, name="attention"
        )(
            normalized, normalized, valid[:, None, None, :]
        )  # (B, E, D); masking keys excludes padded entities in both directions
        tokens = tokens + attended  # (B, E, D)
        hidden = nn.LayerNorm(dtype=self.dtype, name="ffn_norm")(tokens)  # (B, E, D)
        hidden = nn.gelu(nn.Dense(features=4 * self.features, dtype=self.dtype, name="ffn_in")(hidden))  # (B, E, 4D)
        hidden = nn.Dense(features=self.features, dtype=self.dtype, name="ffn_out")(hidden)  # (B, E, D)
        return jnp.where(valid[..., None], tokens + hidden, 0)  # (B, E, D)


class LuxFeatures(NamedTuple):
    global_features: jax.Array  # (*S, F), invariant to ship slot ordering
    ships: jax.Array  # (*S, U, 2C), equivariant to own ship slot ordering
    spatial: jax.Array  # (*S, W, H, C), full-resolution map features


class LuxEncoder(nn.Module):
    """Spatial map encoding and masked self-attention over all unordered entities.

    Own ships, opponent ships, and relics form one typed set plus a global token.
    Own-ship rows remain aligned with action slots; global features are invariant
    to each set's permutations and padding. Coordinates encode physical location.
    """

    features: int
    channels: int = 16
    attention_implementation: Literal["triton", "xla"] = "triton"
    dtype: jax.typing.DTypeLike = jnp.float32

    def __call__(self, observation: LuxObservation[jax.Array]) -> jax.Array:  # leaves: (*S, ...) -> (*S, F)
        return self.encode_features(observation).global_features  # (*S, F)

    @nn.compact
    def encode_features(self, observation: LuxObservation[jax.Array]) -> LuxFeatures:  # leaves: (*S, ...)
        batch_shape = observation.public_params.shape[:-1]
        batch_size = prod(batch_shape)
        obs = jax.tree.map(lambda leaf: leaf.reshape((batch_size, *leaf.shape[len(batch_shape) :])), observation)
        # obs leaves: (B, ...), B = prod(S), including unbatched B=1
        visible = obs.world.sensor_mask  # (B, W, H)
        width, height = visible.shape[-2:]
        unit_count = obs.self.units_mask.shape[-1]
        opponent_count = obs.opponent.units_mask.shape[-1]
        ship_count = unit_count + opponent_count  # N=U+V own/opponent slots
        token_size = 2 * self.channels
        positions, energy, unit_mask = jax.tree.map(
            lambda own, opponent: jnp.concatenate((own, opponent), axis=1),
            (obs.self.units.position, obs.self.units.energy, obs.self.units_mask),
            (obs.opponent.units.position, obs.opponent.units.energy, obs.opponent.units_mask),
        )  # (B, N, 2); (B, N); (B, N)
        positions = jnp.where(unit_mask[..., None], positions, 0)  # (B, N, 2)
        energy = jnp.where(unit_mask, energy, 0).astype(jnp.float32) / 400.0  # (B, N)
        ship_values = jnp.stack((jnp.ones_like(energy), energy), axis=-1)  # (B, N, 2), count/total energy
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
        spatial_features = grid  # (B, W, H, C), retained for command target scoring
        local = grid[jnp.arange(batch_size)[:, None], positions[..., 0], positions[..., 1]]  # (B, N, C)
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

        unit_coordinates = positions.astype(jnp.float32) / jnp.array([width - 1, height - 1]) * 2 - 1  # (B, N, 2)
        unit_features = jnp.concatenate((unit_coordinates, energy[..., None], local), axis=-1)  # (B, N, C+3)
        units = nn.Dense(features=token_size, dtype=self.dtype, name="ship_projection")(unit_features)  # (B, N, 2C)
        roles = jnp.concatenate((jnp.zeros(unit_count, jnp.int32), jnp.ones(opponent_count, jnp.int32)))  # (N,)
        type_embedding = nn.Embed(num_embeddings=3, features=token_size, dtype=self.dtype, name="entity_type_embedding")
        role_embedding = type_embedding(roles)  # (N, 2C)
        units = nn.silu(units + role_embedding)  # (B, N, 2C)

        relic_positions = jnp.where(obs.world.relic_nodes_mask[..., None], obs.world.relic_nodes, 0)  # (B, R, 2)
        relic_local = spatial_features[
            jnp.arange(batch_size)[:, None], relic_positions[..., 0], relic_positions[..., 1]
        ]  # (B, R, C)
        relic_coordinates = (
            relic_positions.astype(jnp.float32) / jnp.array([width - 1, height - 1]) * 2 - 1
        )  # (B, R, 2)
        relic_features = jnp.concatenate((relic_coordinates, relic_local), axis=-1)  # (B, R, C+2)
        relics = nn.silu(
            nn.Dense(features=token_size, dtype=self.dtype, name="relic_projection")(relic_features)
            + type_embedding(jnp.asarray(2))
        )  # (B, R, 2C)

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
                jnp.stack((obs.self.units_mask.sum(-1), obs.opponent.units_mask.sum(-1)), axis=-1) / 16.0,
                visible.mean(axis=(-2, -1))[:, None],
            ),
            axis=-1,
        )  # (B, 14)
        context = nn.silu(
            nn.Dense(features=token_size, dtype=self.dtype, name="context_projection")(context)
        )  # (B, 2C)
        tokens = jnp.concatenate((units, relics, context[:, None]), axis=1)  # (B, E=N+R+1, 2C)
        valid = jnp.concatenate(
            (unit_mask, obs.world.relic_nodes_mask, jnp.ones((batch_size, 1), bool)), axis=1
        )  # (B, E)
        tokens = jnp.where(valid[..., None], tokens, 0)  # (B, E, 2C)
        attended = _Attention(
            features=token_size, implementation=self.attention_implementation, dtype=self.dtype, name="map_attention"
        )(nn.LayerNorm(dtype=self.dtype, name="query_norm")(tokens), map_tokens)  # (B, E, 2C)
        tokens = jnp.where(valid[..., None], tokens + attended, 0)  # (B, E, 2C)
        for index in range(2):
            tokens = _SetAttentionBlock(
                features=token_size,
                implementation=self.attention_implementation,
                dtype=self.dtype,
                name=f"set_attention_{index}",
            )(tokens, valid)  # (B, E, 2C)
        tokens = jnp.where(
            valid[..., None], nn.LayerNorm(dtype=self.dtype, name="entity_norm")(tokens), 0
        )  # (B, E, 2C)
        own = tokens[:, :unit_count]  # (B, U, 2C), never pooled before the ship policy
        own_mean = own.sum(1) / jnp.maximum(obs.self.units_mask.sum(-1, keepdims=True), 1)  # (B, 2C)
        opponents = tokens[:, unit_count:ship_count]  # (B, V, 2C)
        opponent_mean = opponents.sum(1) / jnp.maximum(obs.opponent.units_mask.sum(-1, keepdims=True), 1)  # (B, 2C)
        # Coordinates and attention preserve spatial information without map-size-dependent parameters.
        spatial = nn.silu(
            nn.Dense(features=self.features, dtype=self.dtype, name="map_projection")(grid.mean(axis=(1, 2)))
        )  # (B, F)
        combined = jnp.concatenate((own_mean, opponent_mean, tokens[:, -1], spatial), axis=-1)  # (B, 6C+F)
        output = nn.Dense(features=self.features, dtype=self.dtype, name="output_projection")(combined)  # (B, F)
        output = nn.LayerNorm(dtype=self.dtype, name="output_norm")(output)  # (B, F)
        return LuxFeatures(
            global_features=output.reshape((*batch_shape, self.features)),  # (*S, F)
            ships=own.reshape((*batch_shape, unit_count, token_size)),  # (*S, U, 2C)
            spatial=spatial_features.reshape((*batch_shape, width, height, self.channels)),  # (*S, W, H, C)
        )


class LuxModel(nn.Module):
    """Encode visible state; decode team memory into ship commands and a team value."""

    embedding_size: int
    channels: int
    action_space: HierarchicalMultiDiscreteSpace
    attention_implementation: Literal["triton", "xla"] = "triton"
    dtype: jax.typing.DTypeLike = jnp.float32

    def setup(self) -> None:
        groups = self.action_space.group_sizes
        side = isqrt(groups[-1])
        if len(groups) != 3 or groups[:2] != (1, 4) or side < 3 or side % 2 != 1 or side * side != groups[-1]:
            raise ValueError("Lux requires stay/move/sap groups (1, 4, odd_square) with sap range >= 1")
        self.sap_range = side // 2
        self.encoder = LuxEncoder(
            features=self.embedding_size,
            channels=self.channels,
            attention_implementation=self.attention_implementation,
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
        self.sap_offset = nn.Dense(features=self.channels, use_bias=False, dtype=self.dtype)
        self.value_hidden = nn.Dense(features=self.embedding_size, dtype=self.dtype)
        self.value_norm = nn.LayerNorm(dtype=self.dtype)
        self.value_output = nn.Dense(features=1, kernel_init=nn.initializers.orthogonal(1.0), dtype=self.dtype)

    def encode(self, observation: LuxObservation[jax.Array]) -> LuxFeatures:  # leaves: (*S, ...)
        return self.encoder.encode_features(observation)  # leaves: (*S, ...)

    def heads(
        self,
        memory: jax.Array,  # (*S, H)
        features: LuxFeatures,  # global: (*S, F); ships: (*S, U, 2C); spatial: (*S, W, Hmap, C)
        observation: LuxObservation[jax.Array],  # leaves: (*S, ...)
        actions_mask: jax.Array,  # (*S, U, A)
    ) -> tuple[HierarchicalMultiCategoricalPolicy, jax.Array]:  # logits: (*S, U, A); value: (*S,)
        batch_shape = memory.shape[:-1]
        batch_size = prod(batch_shape)
        unit_count = features.ships.shape[-2]
        width, height = features.spatial.shape[-3:-1]
        team_memory = jnp.broadcast_to(memory[..., None, :], (*batch_shape, unit_count, memory.shape[-1]))  # (*S, U, H)
        ships = jnp.concatenate((features.ships, team_memory), axis=-1)  # (*S, U, 2C+H)
        ships = nn.silu(self.ship_norm(self.ship_hidden(ships)))  # (*S, U, F)
        types = self.type_output(ships).astype(jnp.float32)  # (*S, U, 3), stay/move/sap
        moves = self.move_output(ships).astype(jnp.float32)  # (*S, U, 4), north/east/south/west

        # Match LuxEnv's x-major, y-minor relative target enumeration.
        axis = jnp.arange(-self.sap_range, self.sap_range + 1)  # (2R+1,)
        offsets = jnp.stack(jnp.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))  # (K, 2)
        positions = jnp.where(observation.self.units_mask[..., None], observation.self.units.position, 0)  # (*S, U, 2)
        targets = positions.reshape((batch_size, unit_count, 1, 2)) + offsets  # (B, U, K, 2)
        targets = jnp.clip(targets, 0, jnp.array([width - 1, height - 1]))  # (B, U, K, 2); off-map commands are masked
        keys = self.sap_key(features.spatial).reshape((batch_size, width, height, self.channels))  # (B, W, Hmap, C)
        keys = keys[jnp.arange(batch_size)[:, None, None], targets[..., 0], targets[..., 1]]  # (B, U, K, C)
        relative = self.sap_offset(offsets.astype(self.dtype) / self.sap_range)  # (K, C)
        keys = keys + relative  # (B, U, K, C)
        queries = self.sap_query(ships).reshape((batch_size, unit_count, self.channels))  # (B, U, C)
        sap = jnp.einsum("bukc,buc->buk", keys, queries).astype(jnp.float32) / sqrt(self.channels)  # (B, U, K)
        sap = sap.reshape((*batch_shape, unit_count, offsets.shape[0]))  # (*S, U, K)
        logits = jnp.concatenate((types, moves, sap), axis=-1)  # (*S, U, 7+K)
        policy = HierarchicalMultiCategoricalPolicy(
            raw_logits=logits, actions_mask=actions_mask, group_sizes=self.action_space.group_sizes
        )
        context = jnp.concatenate((memory, features.global_features), axis=-1)  # (*S, H+F)
        critic = nn.silu(self.value_norm(self.value_hidden(context)))  # (*S, F)
        value = self.value_output(critic).squeeze(-1).astype(jnp.float32)  # (*S,)
        return policy, value


class LuxActorCritic(BaseActorCritic[LuxObservation[jax.Array], LSTMStackCarry, HierarchicalMultiCategoricalPolicy]):
    """Lux spatial/entity policy with recurrent team memory and shared ship heads."""

    action_space: HierarchicalMultiDiscreteSpace
    model: LuxModelConfig
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.nowrap
    def _make_recurrent(self, *, parent: nn.Module | None = None) -> LSTMStack:
        return LSTMStack(
            features=self.model.recurrent.hidden_size,
            num_layers=self.model.recurrent.num_layers,
            intermediate_size=self.model.recurrent.intermediate_size,
            dtype=self.dtype,
            name="lstm",
            parent=parent,
        )

    @nn.nowrap
    def initial_carry(self, actor_count: int) -> LSTMStackCarry:
        return self._make_recurrent().initial_carry(actor_count)  # leaves: (B, H)

    def setup(self) -> None:
        self.lux = LuxModel(
            embedding_size=self.model.embedding_size,
            channels=self.model.channels,
            action_space=self.action_space,
            attention_implementation=self.model.attention_implementation,
            dtype=self.dtype,
            name="lux",
        )
        self.recurrent = self._make_recurrent(parent=self)
        self.recurrent_input = nn.Dense(features=self.model.recurrent.hidden_size, dtype=self.dtype, name="lstm_input")

    def _encode(self, observation: LuxObservation[jax.Array]) -> tuple[jax.Array, LuxFeatures]:
        """Return (*S, H) recurrent inputs and Lux spatial/entity features."""
        features = self.lux.encode(observation)  # leaves: (*S, ...)
        x = self.recurrent_input(nn.relu(features.global_features)).astype(self.dtype)  # (*S, H)
        return x, features  # (*S, H); leaves: (*S, ...)

    def __call__(
        self,
        observation: LuxObservation[jax.Array],  # leaves: (B, T, ...)
        model_carry: LSTMStackCarry,  # leaves: (B, H)
        episode_start: jax.Array,  # (B, T)
        actions_mask: jax.Array,  # (B, T, *C, A)
    ) -> ActorCriticOutput[
        LSTMStackCarry, HierarchicalMultiCategoricalPolicy
    ]:  # carry leaves: (B, H); logits: (B, T, *C, A); value: (B, T)
        """Process observation sequences, resetting memory at episode starts."""
        x, features = self._encode(observation)  # (B, T, H)
        # The shared LSTM scans time-major inputs; MAPPO exposes batch-major arrays.
        recurrent_x, recurrent_starts = jax.tree.map(
            lambda leaf: jnp.swapaxes(leaf, 0, 1), (x, episode_start)
        )  # (T, B, H); (T, B)
        model_carry, recurrent_x = self.recurrent(
            recurrent_x, model_carry, recurrent_starts
        )  # model_carry leaves: (B, H); recurrent_x: (T, B, H)
        x = jnp.swapaxes(recurrent_x, 0, 1)  # (B, T, H)
        policy, value = self.lux.heads(x, features, observation, actions_mask)  # logits: (B, T, *C, A); (B, T)
        return ActorCriticOutput(carry=model_carry, policy=policy, value=value)

    def step(
        self,
        observation: LuxObservation[jax.Array],  # leaves: (B, ...)
        model_carry: LSTMStackCarry,  # leaves: (B, H)
        episode_start: jax.Array,  # (B,)
        actions_mask: jax.Array,  # (B, *C, A)
    ) -> ActorCriticOutput[
        LSTMStackCarry, HierarchicalMultiCategoricalPolicy
    ]:  # carry leaves: (B, H); logits: (B, *C, A); value: (B,)
        """Process one observation per sequence, resetting memory at episode starts."""
        x, features = self._encode(observation)  # (B, H)
        model_carry, x = self.recurrent.step(x, model_carry, episode_start)  # model_carry leaves: (B, H); x: (B, H)
        policy, value = self.lux.heads(x, features, observation, actions_mask)  # logits: (B, *C, A); (B,)
        return ActorCriticOutput(carry=model_carry, policy=policy, value=value)
