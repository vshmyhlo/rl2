"""STORM n-player matrix games with flat symbolic observations and time limits."""

from typing import Annotated, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.storm.storm import InTheMatrix, State
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import DiscreteSpace
from rl2.envs.multi_agent import AgentObservation, BaseMultiAgentEnv, MultiAgentStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class STORMConfig:
    id: Literal["storm_np"] = "storm_np"
    game: Literal["coordination", "prisoners_dilemma"] = "prisoners_dilemma"
    num_agents: Annotated[int, Field(ge=2, strict=True)] = 2
    grid_size: Annotated[int, Field(ge=5, strict=True)] = 8
    num_coins: Annotated[int, Field(gt=0, strict=True)] = 6
    # The native inner horizon adds one to avoid its automatic reset.
    max_steps: Annotated[int, Field(gt=0, lt=2**31 - 1, strict=True)] = 152
    freeze_penalty: Annotated[int, Field(gt=0, strict=True)] = 5

    def __post_init__(self) -> None:
        if self.num_agents + 2 * self.num_coins > self.grid_size**2:
            raise ValueError("grid_size must fit all agents and both coin types")


class _InTheMatrix(InTheMatrix):
    def render_time(self, state: State, width_px: int) -> NDArray[np.uint8]:  # state leaves: (...)
        # Upstream constructs int8(255), which raises with NumPy 2.
        image = np.zeros((16, width_px, 3), dtype=np.uint8)  # (16, W, 3)
        progress = min(int(state.step), self.num_inner_steps - 1) / (self.num_inner_steps - 1)
        image[:, : round(width_px * progress)] = 255
        return image  # (16, W, 3)


class STORMEnv(BaseMultiAgentEnv[State, jax.Array]):
    def __init__(self, config: STORMConfig) -> None:
        self.config = config
        payoff = (
            [[[3, 1], [1, 0]], [[3, 1], [1, 0]]]
            if config.game == "coordination"
            else [[[3, 0], [5, 1]], [[3, 5], [0, 1]]]
        )
        self.payoff = jnp.asarray(payoff[0], dtype=jnp.float32)  # (2, 2), own vs partner coins
        self.env = _InTheMatrix(
            num_agents=config.num_agents,
            grid_size=config.grid_size,
            num_coins=config.num_coins,
            payoff_matrix=jnp.asarray(payoff, dtype=jnp.float32),  # (2, 2, 2)
            freeze_penalty=config.freeze_penalty,
            # Native inner resets discard the last reward and observation; truncate first.
            num_inner_steps=config.max_steps + 1,
            num_outer_steps=1,
            obs_size=5,
        )
        self.num_agents = config.num_agents
        self.action_space = DiscreteSpace(self.env.num_actions)

    def _observation(self, observation: jax.Array) -> AgentObservation[jax.Array]:  # (N, 5, 5, 14)
        return AgentObservation(
            observation=observation.reshape(self.num_agents, -1).astype(jnp.float32),  # (N, 350)
            actions_mask=jnp.ones((self.num_agents, self.action_space.n), dtype=bool),  # (N, 8)
        )

    def reset(self, key: jax.Array) -> tuple[AgentObservation[jax.Array], State]:  # key: ()
        """Return (N, 350)/(N, 8) observations/masks and (...) state leaves."""
        observation, state = self.env.reset(key)  # (N, 5, 5, 14); state leaves: (...)
        return self._observation(observation), state

    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: State,  # leaves: (...)
        action: jax.Array,  # (N,) int32
    ) -> MultiAgentStep[State, jax.Array]:
        step_key, reset_key = jax.random.split(key)  # each ()
        observation, final_state, _, _, _ = self.env.step_env(step_key, state, action)
        # observation: (N, 5, 5, 14); state leaves: (...)
        # Upstream's reward scatter can overwrite a successful zap with zero.
        # Newly frozen pairs identify accepted interactions, including simultaneous ones.
        interaction = (final_state.freeze > 0) & (state.freeze <= 0)  # (N, N)
        inventory = final_state.agent_invs.astype(jnp.float32)  # (N, 2)
        proportion = inventory / jnp.maximum(inventory.sum(axis=-1, keepdims=True), 1)  # (N, 2)
        reward = ((proportion @ self.payoff @ proportion.T) * interaction).sum(axis=-1)  # (N,)
        truncated = final_state.step >= self.config.max_steps  # ()
        final_state = final_state.replace(done=truncated)  # leaves: (...)
        final_observation = self._observation(observation)  # (N, 350)/(N, 8)
        observation, next_state = jax.lax.cond(
            truncated, lambda: self.reset(reset_key), lambda: (final_observation, final_state)
        )  # observation: (N, 350)/(N, 8); state leaves: (...)
        return MultiAgentStep(
            observation=observation,  # (N, 350)/(N, 8)
            state=next_state,  # leaves: (...)
            reward=reward.astype(jnp.float32),  # (N,)
            done=truncated,  # ()
            agent_done=jnp.full(self.num_agents, truncated),  # (N,)
            # Interactions, freezes and respawns remain part of the same social game.
            agent_inactive=jnp.zeros(self.num_agents, dtype=bool),  # (N,)
            terminated=jnp.asarray(False),  # ()
            truncated=truncated,  # ()
            final_observation=final_observation,  # (N, 350)/(N, 8)
            final_state=final_state,  # leaves: (...)
        )

    def render(self, state: State) -> NDArray[np.uint8]:  # state leaves: (...) -> (H, W, 3)
        return self.env.render(state)  # (H, W, 3)
