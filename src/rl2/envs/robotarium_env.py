"""JaxRobotarium Material Transport with discrete waypoints and pure resets."""

from functools import partial
from typing import Annotated, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxrobotarium.constants import ROBOTARIUM_HEIGHT, ROBOTARIUM_WIDTH
from jaxrobotarium.robotarium_env import State
from jaxrobotarium.scenarios.material_transport import MaterialTransport
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from rps_jax.utilities.misc import generate_initial_conditions

from rl2.actions import DiscreteSpace
from rl2.envs.multi_agent import AgentObservation, BaseMultiAgentEnv, MultiAgentStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class RobotCapability:
    step_size: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.45
    capacity: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 5.0


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class RobotariumConfig:
    id: Literal["JaxRobotarium_material_transport"] = "JaxRobotarium_material_transport"
    # Native reset samples without replacement from 15 starting positions.
    robots: Annotated[tuple[RobotCapability, ...], Field(min_length=2, max_length=15)] = (
        RobotCapability(),
        RobotCapability(),
        RobotCapability(step_size=0.15, capacity=15),
        RobotCapability(step_size=0.15, capacity=15),
    )
    zone1_load: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 50.0
    zone2_load: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 50.0
    max_steps: Annotated[int, Field(gt=0, strict=True)] = 100
    update_frequency: Annotated[int, Field(gt=0, strict=True)] = 30
    collision_avoidance: bool = True


class RobotariumEnv(BaseMultiAgentEnv[State]):
    def __init__(self, config: RobotariumConfig) -> None:
        self.config = config
        self.num_agents = len(config.robots)
        self.env = MaterialTransport(
            num_agents=self.num_agents,
            max_steps=config.max_steps,
            update_frequency=config.update_frequency,
            zone1_dist={"mu": config.zone1_load, "sigma": 0.0},
            zone2_dist={"mu": config.zone2_load, "sigma": 0.0},
            heterogeneity={
                "type": "capability_set",
                "values": [[robot.step_size, robot.capacity] for robot in config.robots],
                "obs_type": "full_capability_set",
            },
            controller={
                "controller": "clf_uni_position",
                "barrier_fn": "robust_barriers" if config.collision_avoidance else None,
            },
        )
        self.agents = tuple(self.env.agents)
        self.action_space = DiscreteSpace(5)

    def _observation(self, observations: dict[str, jax.Array]) -> AgentObservation:  # agent leaves: (O,)
        return AgentObservation(
            observations=jnp.stack([observations[agent] for agent in self.agents]),  # (N, O=5N+2)
            action_masks=jnp.ones((self.num_agents, self.action_space.n), dtype=bool),  # (N, A=5)
        )

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array) -> tuple[AgentObservation, State]:  # key: ()
        """Return (N, O)/(N, A) observations/masks and (...) state leaves."""
        # Native reset stores traced poses/velocities on the simulator. Keep state explicit.
        _, pose_key = jax.random.split(key)  # each ()
        poses = generate_initial_conditions(
            self.num_agents, width=ROBOTARIUM_WIDTH / 4, height=ROBOTARIUM_HEIGHT, spacing=0.3, key=pose_key
        )  # (3, N)
        state = State(
            p_pos=poses.T,  # (N, 3): x, y, heading
            done=jnp.zeros(self.num_agents, dtype=bool),  # (N,)
            step=jnp.asarray(0, jnp.int32),  # ()
            het_rep=jnp.asarray([[r.step_size, r.capacity] for r in self.config.robots], jnp.float32),  # (N, 2)
            zone1_load=jnp.asarray([self.config.zone1_load], jnp.float32),  # (1,)
            zone2_load=jnp.asarray([self.config.zone2_load], jnp.float32),  # (1,)
            payload=jnp.zeros(self.num_agents, jnp.int32),  # (N,)
        )
        return self._observation(self.env.get_obs(state)), state

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
        # Native timeout checks the old counter; enforce exactly max_steps transitions.
        truncated = final_state.step >= self.config.max_steps  # ()
        final_state = final_state.replace(done=jnp.full(self.num_agents, truncated))  # leaves: (...)
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
            terminated=jnp.asarray(False),  # () Native success does not end the episode.
            truncated=truncated,  # ()
            final_observation=final_observation,  # (N, O)/(N, A)
            final_state=final_state,  # leaves: (...)
        )

    def memory_resets(self, state: State) -> jax.Array:  # state leaves: (...) -> (N,)
        return jnp.zeros(self.num_agents, dtype=bool)  # (N,)

    def render(self, state: State) -> NDArray[np.uint8]:  # state leaves: (...) -> (H, W, 3)
        figure = Figure(figsize=(6, 4), dpi=80)
        canvas = FigureCanvasAgg(figure)
        axes = figure.subplots()
        x, y, width, height = self.env.robotarium.boundaries
        axes.set(xlim=(x, x + width), ylim=(y, y + height), aspect="equal")
        axes.add_patch(Circle((0, 0), self.env.zone1_radius, color="orange", alpha=0.3))
        axes.add_patch(
            Rectangle((x + width - self.env.zone2_width, y), self.env.zone2_width, height, color="orange", alpha=0.3)
        )
        axes.add_patch(Rectangle((x, y), self.env.dropoff_width, height, color="green", alpha=0.3))
        positions, payload = jax.device_get((state.p_pos, state.payload))  # (N, 3), (N,)
        for i, ((px, py, heading), loaded) in enumerate(zip(positions, payload)):
            axes.add_patch(Circle((px, py), self.env.robotarium.robot_radius, color="red" if loaded else "blue"))
            axes.arrow(px, py, 0.1 * np.cos(heading), 0.1 * np.sin(heading), width=0.008, color="black")
            axes.text(px, py + 0.08, str(i), ha="center", fontsize=8)
        axes.set_title(f"Material remaining: {float(state.zone1_load[0]):g} / {float(state.zone2_load[0]):g}")
        canvas.draw()
        return np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()  # (H=320, W=480, 3)
