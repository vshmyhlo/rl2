"""JaxNav with discrete controls, local lidar observations, and explicit autoreset."""

from functools import partial
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav, State
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import DiscreteSpace
from rl2.envs.multi_agent import AgentObservation, MultiAgentStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class JaxNavConfig:
    num_agents: Annotated[int, Field(gt=0, strict=True)] = 4
    map_size: Annotated[int, Field(ge=5, strict=True)] = 11
    fill: Annotated[float, Field(ge=0, lt=1, allow_inf_nan=False)] = 0.3
    lidar_num_beams: Annotated[int, Field(gt=0, strict=True)] = 200
    max_steps: Annotated[int, Field(gt=0, strict=True)] = 500
    rew_lambda: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.5


class JaxNavEnv:
    def __init__(self, config: JaxNavConfig) -> None:
        self.config = config
        self.env = JaxNav(
            num_agents=config.num_agents,
            act_type="Discrete",
            map_id="Grid-Rand-Poly",
            map_params={"map_size": (config.map_size, config.map_size), "fill": config.fill},
            lidar_num_beams=config.lidar_num_beams,
            max_steps=config.max_steps,
            rew_lambda=config.rew_lambda,
            evaporating=False,
        )
        self.agents = tuple(self.env.agents)
        self.num_agents = config.num_agents
        self.action_space = DiscreteSpace(self.env.action_space(self.agents[0]).n)

    def _observation(self, observations: dict[str, jax.Array], state: State) -> AgentObservation:
        """Pack observations: (O,) agent leaves and state: (...) leaves in fixed order."""
        # JaxNav's get_avail_actions reports two continuous controls even in discrete mode.
        # Robots at their goals can still move to yield; collided robots can only stop (action 2).
        masks = ~state.move_term[:, None] | (jnp.arange(self.action_space.n) == 2)  # (N, A)
        return AgentObservation(
            jnp.stack([observations[agent] for agent in self.agents]),  # (N, O)
            masks,  # (N, A)
        )

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array) -> tuple[AgentObservation, State]:  # key: ()
        """Return (N, O)/(N, A) local observations/masks and (...) state leaves."""
        observations, state = self.env.reset(key)  # agent leaves: (O,); state leaves: (...)
        return self._observation(observations, state), state

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: State,  # leaves: (...)
        actions: jax.Array,  # (N,) int32
    ) -> MultiAgentStep[State]:
        step_key, reset_key = jax.random.split(key)  # each ()
        observations, final_state, rewards, dones, _ = self.env.step_env(
            step_key, state, {agent: actions[i] for i, agent in enumerate(self.agents)}
        )  # observations: agent (O,); rewards/dones: agent (); state leaves: (...)
        terminated = (final_state.goal_reached | final_state.move_term).all()  # ()
        truncated = final_state.step >= self.config.max_steps  # ()
        done = dones["__all__"]  # ()
        final_observation = self._observation(observations, final_state)
        observation, next_state = jax.lax.cond(
            done, lambda: self.reset(reset_key), lambda: (final_observation, final_state)
        )  # observations: (N, O)/(N, A); state leaves: (...)
        return MultiAgentStep(
            observation,
            next_state,
            jnp.stack([rewards[agent] for agent in self.agents]),  # (N,)
            done,  # ()
            self.memory_resets(final_state) | done,  # (N,)
            terminated,
            truncated,
            final_observation,
            final_state,
        )

    def memory_resets(self, state: State) -> jax.Array:  # state leaves: (...) -> (N,)
        return state.move_term  # (N,)

    def render(self, state: State) -> NDArray[np.uint8]:  # state leaves: (...) -> (H, W, 3)
        figure = Figure(figsize=(5, 5), dpi=80)
        canvas = FigureCanvasAgg(figure)
        self.env.init_render(figure.subplots(), state, lidar=False, colour_agents_by_idx=True)
        return np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()  # (H, W, 3)
