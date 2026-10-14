"""MPE Simple Spread and Reference with discrete actions and time-limit resets."""

from functools import partial
from typing import Annotated, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.mpe.simple import SimpleMPE, State
from jaxmarl.environments.mpe.simple_reference import SimpleReferenceMPE
from jaxmarl.environments.mpe.simple_spread import SimpleSpreadMPE
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import DiscreteSpace
from rl2.envs.multi_agent import AgentObservation, BaseMultiAgentEnv, MultiAgentStep

type MPEEnvId = Literal["MPE_simple_spread_v3", "MPE_simple_reference_v3"]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class MPEConfig:
    id: MPEEnvId = "MPE_simple_spread_v3"
    num_agents: Annotated[int, Field(gt=0, strict=True)] = 3
    num_landmarks: Annotated[int, Field(gt=0, strict=True)] = 3
    max_steps: Annotated[int, Field(gt=0, strict=True)] = 25
    local_ratio: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.5

    def __post_init__(self) -> None:
        if self.id == "MPE_simple_reference_v3" and (self.num_agents != 2 or self.num_landmarks != 3):
            raise ValueError("MPE Simple Reference requires 2 agents and 3 landmarks")


class MPEEnv(BaseMultiAgentEnv[State]):
    def __init__(self, config: MPEConfig) -> None:
        self.config = config
        env_class = SimpleSpreadMPE if config.id == "MPE_simple_spread_v3" else SimpleReferenceMPE
        self.env: SimpleMPE = env_class(
            num_agents=config.num_agents,
            num_landmarks=config.num_landmarks,
            max_steps=config.max_steps,
            local_ratio=config.local_ratio,
        )
        self.agents = tuple(self.env.agents)
        self.num_agents = config.num_agents
        self.action_space = DiscreteSpace(self.env.action_space(self.agents[0]).n)

    def _observation(self, observations: dict[str, jax.Array]) -> AgentObservation:  # agent leaves: (O,)
        return AgentObservation(
            observations=jnp.stack([observations[agent] for agent in self.agents]),  # (N, O)
            action_masks=jnp.ones((self.num_agents, self.action_space.n), dtype=bool),  # (N, A)
        )

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array) -> tuple[AgentObservation, State]:  # key: ()
        """Return (N, O)/(N, A) observations/masks and (...) state leaves."""
        observations, state = self.env.reset(key)  # agent leaves: (O,); state leaves: (...)
        return self._observation(observations), state

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: State,  # leaves: (...)
        actions: jax.Array,  # (N,) int32
    ) -> MultiAgentStep[State]:
        step_key, reset_key = jax.random.split(key)  # each ()
        observations, final_state, rewards, _, _ = self.env.step_env(
            step_key, state, {agent: actions[i] for i, agent in enumerate(self.agents)}
        )  # agent observations: (O,); rewards: (); state leaves: (...)
        # Native MPE checks the old step counter; enforce exactly max_steps transitions.
        truncated = final_state.step >= self.config.max_steps  # ()
        final_state = final_state.replace(done=truncated)  # leaves: (...)
        final_observation = self._observation(observations)  # (N, O)/(N, A)
        observation, next_state = jax.lax.cond(
            truncated, lambda: self.reset(reset_key), lambda: (final_observation, final_state)
        )  # observations: (N, O)/(N, A); state leaves: (...)
        return MultiAgentStep(
            observation=observation,  # (N, O)/(N, A)
            state=next_state,  # leaves: (...)
            rewards=jnp.stack([rewards[agent] for agent in self.agents]),  # (N,)
            done=truncated,  # ()
            agent_dones=jnp.full(self.num_agents, truncated),  # (N,)
            terminated=jnp.asarray(False),  # () Both tasks end only at the time limit.
            truncated=truncated,  # ()
            final_observation=final_observation,  # (N, O)/(N, A)
            final_state=final_state,  # leaves: (...)
        )

    def memory_resets(self, state: State) -> jax.Array:  # state leaves: (...) -> (N,)
        # All agents remain active when bootstrapping the final observation at a time limit.
        return jnp.zeros(self.num_agents, dtype=bool)  # (N,)

    def render(self, state: State) -> NDArray[np.uint8]:  # state leaves: (...) -> (H, W, 3)
        figure = Figure(figsize=(5, 5), dpi=80)
        canvas = FigureCanvasAgg(figure)
        axes = figure.subplots()
        positions, radii = jax.device_get((state.p_pos, self.env.rad))  # (N+L, 2), (N+L,); L landmarks
        extent = max(1.2, float(np.abs(positions).max()) + 0.2)
        axes.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        for i, (position, radius) in enumerate(zip(positions, radii)):
            color = tuple(channel / 255 for channel in self.env.colour[i])
            axes.add_patch(Circle(position, radius, color=color))
        canvas.draw()
        return np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()  # (H, W, 3)
