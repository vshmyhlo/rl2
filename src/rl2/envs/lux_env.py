"""Lux S3 team agents with joint ship commands and memory across matches."""

from typing import Annotated, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams, env_params_ranges
from luxai_s3.state import EnvObs, EnvState
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import MultiDiscreteSpace
from rl2.envs.multi_agent import AgentObservation, BaseMultiAgentEnv, MultiAgentStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LuxConfig:
    id: Literal["luxai_s3"] = "luxai_s3"
    max_units: Annotated[int, Field(ge=1, le=16, strict=True)] = 16
    max_steps_in_match: Annotated[int, Field(ge=2, lt=2**31, strict=True)] = 100
    match_count: Annotated[int, Field(ge=1, le=5, strict=True)] = 5
    randomize_params: bool = True

    def __post_init__(self) -> None:
        if (self.max_steps_in_match + 1) * self.match_count >= 2**31:
            raise ValueError("Game horizon must fit the native int32 counter")


class LuxState(NamedTuple):
    game: EnvState  # Array leaves: (...), full simulator state, never policy input.
    params: EnvParams  # Array leaves: (), sampled once per game.


class LuxEnv(BaseMultiAgentEnv[LuxState]):
    """Two agents; each chooses U categorical ship commands from A=230 choices.

    Observations flatten each native EnvObs pytree in JAX leaf order, followed by
    a two-element team ID and public move cost, sap cost/range, and sensor range.
    Values retain native units. Only the team's masked observation is exposed.
    """

    def __init__(self, config: LuxConfig) -> None:
        self.config = config
        self.num_agents = 2
        self.params = EnvParams(
            max_units=config.max_units,
            max_steps_in_match=config.max_steps_in_match,
            match_count_per_episode=config.match_count,
        )
        self.env = LuxAIS3Env(auto_reset=False, fixed_env_params=self.params)
        self.sap_range = max(env_params_ranges["unit_sap_range"])
        self.action_space = MultiDiscreteSpace(nvec=(5 + (2 * self.sap_range + 1) ** 2,) * config.max_units)
        offsets = jnp.arange((2 * self.sap_range + 1) ** 2, dtype=jnp.int16)  # (A-5,)
        self.sap_offsets = (
            jnp.stack((offsets // (2 * self.sap_range + 1), offsets % (2 * self.sap_range + 1)), axis=-1)
            - self.sap_range
        )  # (A-5, 2), dx then dy

    def _decode_action(self, action: jax.Array) -> jax.Array:  # (N, U) -> (N, U, 3)
        sap = action >= 5  # (N, U)
        offsets = self.sap_offsets[jnp.maximum(action - 5, 0)]  # (N, U, 2)
        return jnp.concatenate(
            (jnp.where(sap, 5, action)[..., None], jnp.where(sap[..., None], offsets, 0)), axis=-1
        ).astype(jnp.int16)  # (N, U, 3)

    def _observation(self, observation: dict[str, EnvObs], state: LuxState) -> AgentObservation:
        """Return (N, O) features and (N, U, A) masks from (...) state/observation leaves."""
        params, game = state.params, state.game
        public_params = jnp.stack(
            (params.unit_move_cost, params.unit_sap_cost, params.unit_sap_range, params.unit_sensor_range)
        )  # (4,)
        features = jnp.stack(
            [
                jnp.concatenate(
                    [leaf.reshape(-1).astype(jnp.float32) for leaf in jax.tree.leaves(observation[f"player_{team}"])]
                    + [jax.nn.one_hot(team, 2), public_params]
                )
                for team in range(2)
            ]
        )  # (N, O)
        energy = game.units.energy[..., 0]  # (N, U), own ship energy for each team
        active = game.units_mask & (energy >= 0) & (game.match_steps != 0)  # (N, U)
        targets = game.units.position[..., None, :] + self.sap_offsets  # (N, U, A-5, 2)
        in_bounds = ((targets >= 0) & (targets < jnp.array([params.map_width, params.map_height]))).all(-1)
        # in_bounds: (N, U, A-5); no hidden terrain or opponent information is used.
        sap = (
            active[..., None]
            & (energy >= params.unit_sap_cost)[..., None]
            & (jnp.abs(self.sap_offsets).max(-1) <= params.unit_sap_range)
            & in_bounds
        )  # (N, U, A-5)
        movement = jnp.broadcast_to(
            (active & (energy >= params.unit_move_cost))[..., None], (2, self.config.max_units, 4)
        )  # (N, U, 4), failed moves remain available as in native Lux
        masks = jnp.concatenate((jnp.ones((2, self.config.max_units, 1), dtype=bool), movement, sap), axis=-1)
        # masks: (N, U, A); stay is always available, including absent ships/boundary frames.
        return AgentObservation(observation=features, actions_mask=masks)

    def reset(self, key: jax.Array) -> tuple[AgentObservation, LuxState]:  # key: ()
        """Return (N, O)/(N, U, A) observation/masks and (...) state leaves."""
        params_key, map_key = jax.random.split(key)  # each ()
        params = self.params
        if self.config.randomize_params:
            keys = jax.random.split(params_key, len(env_params_ranges))  # (P,), P randomized parameters
            params = params.replace(
                **{
                    # Native ship energy/positions use int16; strong int32 costs promote state leaves.
                    name: jax.random.choice(
                        keys[i], jnp.asarray(values, dtype=jnp.int16 if isinstance(values[0], int) else jnp.float32)
                    )  # ()
                    for i, (name, values) in enumerate(env_params_ranges.items())
                }
            )
        params = jax.tree.map(jnp.asarray, params)  # scalar array leaves
        observation, game = self.env.reset_env(map_key, params)  # observation/game leaves: (...)
        state = LuxState(game=game, params=params)
        return self._observation(observation, state), state

    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: LuxState,  # leaves: (...)
        action: jax.Array,  # (N, U) integer command IDs
    ) -> MultiAgentStep[LuxState]:
        step_key, reset_key = jax.random.split(key)  # each ()
        commands = self._decode_action(action)  # (N, U, 3)
        observation, game, _, _, game_ended, _ = self.env.step_env(
            step_key, state.game, {f"player_{team}": commands[team] for team in range(2)}, state.params
        )  # observation/game leaves: (...); game_ended: ()
        final_state = LuxState(game=game, params=state.params)
        final_observation = self._observation(observation, final_state)  # (N, O)/(N, U, A)
        observation, next_state = jax.lax.cond(
            game_ended, lambda: self.reset(reset_key), lambda: (final_observation, final_state)
        )  # observation: (N, O)/(N, U, A); state leaves: (...)
        return MultiAgentStep(
            observation=observation,
            state=next_state,
            reward=(game.team_wins - state.game.team_wins).astype(jnp.float32),  # (N,)
            done=game_ended,  # ()
            agent_done=jnp.full(2, game_ended),  # (N,), retain team memory across matches/deaths
            agent_inactive=jnp.zeros(2, dtype=bool),  # (N,), teams always remain active
            terminated=game_ended,  # (), the full game is a task boundary, not a bootstrap timeout
            truncated=jnp.asarray(False),  # ()
            final_observation=final_observation,
            final_state=final_state,
        )

    def render(self, state: LuxState) -> NDArray[np.uint8]:  # leaves: (...) -> (H, W, 3)
        # A headless overview; rendering may show privileged state, unlike policy observations.
        game = jax.device_get(state.game)
        colors = np.array([[25, 30, 45], [70, 45, 100], [105, 105, 115]], dtype=np.uint8)  # (3, RGB)
        tiles = np.asarray(game.map_features.tile_type).T  # (H, W)
        image = np.repeat(np.repeat(colors[tiles], 12, axis=0), 12, axis=1)  # (H*12, W*12, 3)
        for position, visible in zip(game.relic_nodes, game.relic_nodes_mask):
            if visible:
                x, y = map(int, position)
                image[y * 12 + 2 : y * 12 + 10, x * 12 + 2 : x * 12 + 10] = (240, 205, 70)
        for team, color in enumerate(((70, 180, 255), (255, 95, 85))):
            for position, active, energy in zip(
                game.units.position[team], game.units_mask[team], game.units.energy[team]
            ):
                if active and energy[0] >= 0:
                    x, y = map(int, position)
                    image[y * 12 + 4 : y * 12 + 8, x * 12 + 4 : x * 12 + 8] = color
        return image  # (H*12, W*12, 3)
