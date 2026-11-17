"""CrazyRL JAX tasks with continuous per-drone actions and same-step autoreset.

Observations and scalar rewards retain upstream semantics. Actions are bounded
xyz commands, scaled by upstream to at most 0.2 m per coordinate per step.
N is the number of configured initial_positions; O is 6 for hover/circle and
3*(N+1) for surround/escort/catch. Native episodes last at most 200 steps.
"""

from typing import Annotated, Literal

import jax
import jax.numpy as jnp
import numpy as np
from crazy_rl.multi_agent.jax.catch.catch import Catch
from crazy_rl.multi_agent.jax.catch.catch import State as CatchState
from crazy_rl.multi_agent.jax.circle.circle import Circle
from crazy_rl.multi_agent.jax.circle.circle import State as CircleState
from crazy_rl.multi_agent.jax.escort.escort import Escort
from crazy_rl.multi_agent.jax.escort.escort import State as EscortState
from crazy_rl.multi_agent.jax.hover.hover import Hover
from crazy_rl.multi_agent.jax.hover.hover import State as HoverState
from crazy_rl.multi_agent.jax.surround.surround import State as SurroundState
from crazy_rl.multi_agent.jax.surround.surround import Surround
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import ContinuousSpace
from rl2.envs.drone_render import render_drones
from rl2.envs.multi_agent import AgentObservation, BaseMultiAgentEnv, MultiAgentStep

type Position = tuple[float, float, float]
type CrazyRLState = CatchState | CircleState | EscortState | HoverState | SurroundState


@dataclass(frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False))
class CrazyRLConfig:
    id: Literal["CrazyRL"] = "CrazyRL"
    task: Literal["hover", "circle", "surround", "escort", "catch"] = "surround"
    initial_positions: tuple[Position, ...] = ((-1.0, 0.0, 1.0), (1.0, 0.0, 1.0))
    target_position: Position = (0.0, 0.0, 1.0)
    final_target_position: Position = (0.0, 1.0, 1.0)
    size: Annotated[int, Field(gt=0, strict=True)] = 2
    max_steps: Annotated[int, Field(gt=0, le=200, strict=True)] = 200
    num_intermediate_points: Annotated[int, Field(gt=0, strict=True)] = 100
    target_speed: Annotated[float, Field(ge=0)] = 0.05

    def __post_init__(self) -> None:
        minimum = 1 if self.task in ("hover", "circle") else 2
        if len(self.initial_positions) < minimum:
            raise ValueError(f"{self.task} requires at least {minimum} drones")
        for x, y, z in (*self.initial_positions, self.target_position, self.final_target_position):
            if not (-self.size <= x <= self.size and -self.size <= y <= self.size and 0 <= z <= 3):
                raise ValueError("positions must lie within [-size, size] in xy and [0, 3] in z")


class CrazyRLEnv(BaseMultiAgentEnv[CrazyRLState, jax.Array]):
    @property
    def supports_selfplay(self) -> bool:
        return False

    def __init__(self, config: CrazyRLConfig) -> None:
        self.config = config
        self.num_agents = len(config.initial_positions)
        positions = jnp.asarray(config.initial_positions, jnp.float32)  # (N, 3)
        self.target = jnp.asarray(config.target_position, jnp.float32)  # (3,)
        common = {"num_drones": self.num_agents, "init_flying_pos": positions, "size": config.size}
        self.env: Hover | Circle | Surround | Escort | Catch
        if config.task == "hover":
            self.env = Hover(**common)
        elif config.task == "circle":
            self.env = Circle(**common, num_intermediate_points=config.num_intermediate_points)
        elif config.task == "surround":
            self.env = Surround(**common, target_location=self.target, multi_obj=False)
        elif config.task == "escort":
            self.env = Escort(
                **common,
                init_target_location=self.target,
                final_target_location=jnp.asarray(config.final_target_position, jnp.float32),  # (3,)
                num_intermediate_points=config.num_intermediate_points,
                multi_obj=False,
            )
        else:
            self.env = Catch(
                **common, init_target_location=self.target, target_speed=config.target_speed, multi_obj=False
            )
        self.action_space = ContinuousSpace(shape=(3,), low=(-1.0,) * 3, high=(1.0,) * 3)

    def _observation(self, observation: jax.Array) -> AgentObservation[jax.Array]:  # (N, O)
        return AgentObservation(
            observation=observation.astype(jnp.float32),  # (N, O)
            actions_mask=jnp.empty((self.num_agents, 0), dtype=bool),  # (N, 0), continuous actions
        )

    def reset(self, key: jax.Array) -> tuple[AgentObservation[jax.Array], CrazyRLState]:  # key: ()
        """Return (N, O) observations, (N, 0) masks, and (...) native state leaves."""
        observation, _, state = self.env.reset(key)  # (N, O); state leaves: (...)
        return self._observation(observation), state

    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: CrazyRLState,  # leaves: (...)
        action: jax.Array,  # (N, 3)
    ) -> MultiAgentStep[CrazyRLState, jax.Array]:
        step_key, reset_key = jax.random.split(key)  # each ()
        observation, reward, terminations, truncations, _, final_state = self.env.step(
            state, jnp.clip(action, -1.0, 1.0), step_key
        )  # observations: (N, O); rewards/flags: (N,); state leaves: (...)
        terminated = jnp.any(terminations.astype(bool))  # ()
        truncated = jnp.any(truncations.astype(bool)) | (final_state.timestep >= self.config.max_steps)  # ()
        done = terminated | truncated  # ()
        final_observation = self._observation(observation)  # (N, O)/(N, 0)
        observation, next_state = jax.lax.cond(
            done, lambda: self.reset(reset_key), lambda: (final_observation, final_state)
        )  # (N, O)/(N, 0); state leaves: (...)
        return MultiAgentStep(
            observation=observation,
            state=next_state,
            reward=reward.astype(jnp.float32),  # (N,)
            done=done,  # ()
            agent_done=jnp.full(self.num_agents, done),  # (N,)
            agent_inactive=jnp.zeros(self.num_agents, dtype=bool),  # (N,), no individual removal
            terminated=terminated,  # ()
            truncated=truncated,  # ()
            final_observation=final_observation,
            final_state=final_state,
        )

    def render(self, state: CrazyRLState) -> NDArray[np.uint8]:  # leaves: (...) -> (400, 400, 3)
        if isinstance(state, (CircleState, EscortState, CatchState)):
            targets = state.target_location.reshape(-1, 3)  # (K, 3)
        elif self.config.task == "hover":
            targets = jnp.asarray(self.config.initial_positions, jnp.float32)  # (N, 3)
        else:
            targets = self.target[None]  # (1, 3)
        return render_drones(state.agents_locations, targets, extent=self.config.size)
