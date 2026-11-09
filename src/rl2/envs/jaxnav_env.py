"""JaxNav with discrete controls, local lidar observations, and explicit autoreset."""

from typing import Annotated, Literal

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
from rl2.envs.multi_agent import AgentObservation, BaseMultiAgentEnv, MultiAgentStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class JaxNavConfig:
    id: Literal["JaxNav"] = "JaxNav"
    num_agents: Annotated[int, Field(gt=0, strict=True)] = 4
    map_size: Annotated[int, Field(ge=5, strict=True)] = 11
    fill: Annotated[float, Field(ge=0, lt=1, allow_inf_nan=False)] = 0.3
    lidar_num_beams: Annotated[int, Field(gt=0, strict=True)] = 200
    max_steps: Annotated[int, Field(gt=0, lt=2**31, strict=True)] = 500
    rew_lambda: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.5


class JaxNavEnv(BaseMultiAgentEnv[State]):
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

    def _observation(self, observation: dict[str, jax.Array]) -> AgentObservation:
        """Pack observation: (O,) agent leaves in fixed order."""
        # JaxNav's get_avail_actions reports two continuous controls even in discrete mode.
        return AgentObservation(
            observation=jnp.stack([observation[agent] for agent in self.agents]),  # (N, O)
            actions_mask=jnp.ones((self.num_agents, self.action_space.n), dtype=bool),  # (N, A)
        )

    def reset(self, key: jax.Array) -> tuple[AgentObservation, State]:  # key: ()
        """Return (N, O)/(N, A) local observations/masks and (...) state leaves."""
        observation, state = self.env.reset(key)  # agent leaves: (O,); state leaves: (...)
        return self._observation(observation), state

    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: State,  # leaves: (...)
        action: jax.Array,  # (N,) int32
    ) -> MultiAgentStep[State]:
        step_key, reset_key = jax.random.split(key)  # each ()
        observation, final_state, reward, _, _ = self.env.step_env(
            step_key, state, {agent: action[i] for i, agent in enumerate(self.agents)}
        )  # observation: agent (O,); reward: agent (); state leaves: (...)
        terminated = (final_state.goal_reached | final_state.move_term).all()  # ()
        truncated = final_state.step >= self.config.max_steps  # ()
        done = terminated | truncated  # ()
        final_observation = self._observation(observation)  # (N, O)/(N, A)
        observation, next_state = jax.lax.cond(
            done, lambda: self.reset(reset_key), lambda: (final_observation, final_state)
        )  # observation: (N, O)/(N, A); state leaves: (...)
        return MultiAgentStep(
            observation=observation,  # (N, O)/(N, A)
            state=next_state,  # leaves: (...)
            reward=jnp.stack([reward[agent] for agent in self.agents]),  # (N,)
            done=done,  # ()
            agent_done=final_state.move_term | done,  # (N,)
            agent_inactive=final_state.move_term,  # (N,)
            terminated=terminated,  # ()
            truncated=truncated,  # ()
            final_observation=final_observation,  # (N, O)/(N, A)
            final_state=final_state,  # leaves: (...)
        )

    def render(self, state: State) -> NDArray[np.uint8]:  # state leaves: (...) -> (H, W, 3)
        figure = Figure(figsize=(5, 5), dpi=80)
        canvas = FigureCanvasAgg(figure)
        self.env.init_render(figure.subplots(), state, lidar=False, colour_agents_by_idx=True)
        return np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()  # (H, W, 3)
