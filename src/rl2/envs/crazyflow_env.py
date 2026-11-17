"""Functional Crazyflow single-drone tasks for PPO.

Uses upstream dynamics, reset randomization, rewards and termination rules with
explicit simulation/goal state and same-step autoreset. Native attitude controls
are (roll, pitch, yaw, collective thrust); one environment step advances 500/freq
physics steps. Observations flatten pos, quat, vel, ang_vel and goal differences:
O=16, or O=13+3*n_samples for figure-eight tracking. Rendering shows positions.
"""

from functools import partial
from math import ceil
from typing import Annotated, Literal

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.control import Control
from crazyflow.drones import Drone
from crazyflow.dynamics import Dynamics
from crazyflow.envs.drone_env import DroneEnv, action_space
from crazyflow.envs.figure_8_env import FigureEightEnv
from crazyflow.envs.landing_env import LandingEnv
from crazyflow.envs.reach_pos_env import ReachPosEnv
from crazyflow.envs.reach_vel_env import ReachVelEnv
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData
from crazyflow.sim.functional import attitude_control
from crazyflow.sim.pipeline import append_fn
from flax import struct
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import ContinuousSpace
from rl2.envs.autoreset import reset_batch
from rl2.envs.base import BaseEnv, EnvStep
from rl2.envs.drone_render import render_drones


@dataclass(frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False))
class CrazyflowConfig:
    id: Literal["Crazyflow"] = "Crazyflow"
    task: Literal["reach_pos", "reach_vel", "landing", "figure_eight"] = "reach_pos"
    freq: Annotated[int, Field(gt=0, le=500, strict=True)] = 50
    max_steps: Annotated[int, Field(gt=0, lt=2**31, strict=True)] = 500
    n_samples: Annotated[int, Field(gt=0, strict=True)] = 10
    samples_dt: Annotated[float, Field(gt=0)] = 0.1
    trajectory_seconds: Annotated[float, Field(gt=0)] = 10.0

    def __post_init__(self) -> None:
        if 500 % self.freq:
            raise ValueError("freq must divide the 500 Hz physics frequency")
        if self.task == "figure_eight" and self.trajectory_seconds * self.freq < self.max_steps:
            raise ValueError("figure_eight trajectory must cover max_steps / freq seconds")


@struct.dataclass
class CrazyflowState:
    physics: SimData  # leaves: (...), internal world/drone axes are (1, 1)
    goal: jax.Array  # (1, 3), fixed position/velocity goal; unused for figure_eight
    step_count: jax.Array  # () int32


class CrazyflowEnv(BaseEnv[CrazyflowState, None]):
    params = None

    def __init__(self, config: CrazyflowConfig) -> None:
        self.config = config
        # Construction places constants on one device; training supplies its own batch mesh.
        with jax.set_mesh(None):
            sim = Sim(
                n_worlds=1,
                n_drones=1,
                drone=Drone.cf2x_L250,
                dynamics=Dynamics.so_rpy,
                control=Control.attitude,
                device=jax.default_backend(),
            )
            randomization = (
                FigureEightEnv._reset_randomization if config.task == "figure_eight" else DroneEnv._reset_randomization
            )
            append_fn(sim.reset_pipeline, randomization, name="randomization")
            self.reset_physics = sim.build_reset_fn()
            self.step_physics = sim.build_step_fn()
            self.default_data = sim.default_data  # leaves: (...)
            space = action_space(Control.attitude, Dynamics.so_rpy, Drone.cf2x_L250)
        self.action_space = ContinuousSpace(
            shape=(4,), low=tuple(float(x) for x in space.low), high=tuple(float(x) for x in space.high)
        )
        self.n_substeps = 500 // config.freq
        with jax.set_mesh(None):
            count = ceil(config.trajectory_seconds * config.freq)
            t = jnp.linspace(0.0, 2 * jnp.pi, count)  # (P,), P trajectory samples
            self.trajectory = jnp.stack((jnp.sin(t), jnp.zeros_like(t), 0.5 * jnp.sin(2 * t) + 1), axis=-1)  # (P, 3)
            self.sample_offsets = (jnp.arange(config.n_samples) * config.freq * config.samples_dt).astype(
                jnp.int32
            )  # (K,)

    def _targets(self, state: CrazyflowState) -> jax.Array:  # state leaves: (...) -> (K, 3)
        if self.config.task == "figure_eight":
            indices = (state.step_count - 1 + self.sample_offsets) % self.trajectory.shape[0]  # (K,)
            return self.trajectory[indices]  # (K, 3)
        return state.goal  # (1, 3)

    def _observation(self, state: CrazyflowState) -> jax.Array:  # state leaves: (...) -> (O,)
        physics = state.physics.states  # leaves: (1, 1, D)
        reference = physics.vel if self.config.task == "reach_vel" else physics.pos  # (1, 1, 3)
        difference = self._targets(state) - reference[0, 0]  # (K, 3)
        return jnp.concatenate(
            [leaf.reshape(-1) for leaf in (physics.pos, physics.quat, physics.vel, physics.ang_vel, difference)]
        ).astype(jnp.float32)  # (O,)

    def reset(self, key: jax.Array, params: None) -> tuple[jax.Array, CrazyflowState]:  # key: ()
        """Return (O,) float32 observation and (...) state leaves."""
        physics_key, goal_key = jax.random.split(key)  # each ()
        data = self.default_data.replace(core=self.default_data.core.replace(rng_key=physics_key))
        physics = self.reset_physics(data, self.default_data)  # leaves: (...)
        goal = jnp.zeros((1, 3), jnp.float32)  # (1, 3)
        if self.config.task == "reach_pos":
            goal = ReachPosEnv._sample_goal(goal_key, goal, None, None, None)  # (1, 3)
        elif self.config.task == "reach_vel":
            goal = ReachVelEnv._sample_goal_vel(goal_key, goal, None, None, None)  # (1, 3)
        elif self.config.task == "landing":
            goal = goal.at[:, 2].set(0.1)  # (1, 3)
        state = CrazyflowState(physics=physics, goal=goal, step_count=jnp.asarray(0, jnp.int32))
        return self._observation(state), state

    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: CrazyflowState,  # leaves: (...)
        action: jax.Array,  # (4,) attitude controls
        params: None,
        *,
        auto_reset: bool = True,
    ) -> EnvStep[CrazyflowState]:
        low, high = jnp.asarray((self.action_space.low, self.action_space.high), jnp.float32)  # each (4,)
        physics = attitude_control(state.physics, jnp.clip(action, low, high)[None, None])  # leaves: (...)
        physics = self.step_physics(physics, n_steps=self.n_substeps)  # leaves: (...)
        final_state = state.replace(physics=physics, step_count=state.step_count + 1)
        final_observation = self._observation(final_state)  # (O,)
        positions = physics.states.pos  # (1, 1, 3)
        if self.config.task == "figure_eight":
            terminations = FigureEightEnv._terminated(positions)  # (1,)
            reward = FigureEightEnv._reward(terminations, positions, self._targets(final_state)[:1])[0]  # ()
        else:
            terminations = DroneEnv._terminated(positions)  # (1,)
            if self.config.task == "reach_vel":
                reward = ReachVelEnv._reward(terminations, physics.states.vel, state.goal)[0]  # ()
            elif self.config.task == "landing":
                reward = LandingEnv._reward(terminations, physics.states, state.goal)[0]  # ()
            else:
                reward = ReachPosEnv._reward(terminations, positions, state.goal)[0]  # ()
        terminated = terminations[0]  # ()
        truncated = final_state.step_count >= self.config.max_steps  # ()
        done = terminated | truncated  # ()
        if auto_reset:
            observation, next_state = jax.lax.cond(
                done, lambda: self.reset(key, params), lambda: (final_observation, final_state)
            )  # (O,); leaves: (...)
        else:
            observation, next_state = final_observation, final_state  # (O,); leaves: (...)
        return EnvStep(
            observation=observation,  # (O,)
            state=next_state,
            reward=reward,  # ()
            done=done,  # ()
            info={
                "terminated": terminated,  # ()
                "truncated": truncated,  # ()
                "final_observation": final_observation,  # (O,)
                "final_state": final_state,  # leaves: (...)
            },
        )

    def step_batch(
        self,
        keys: jax.Array,  # (B,) typed PRNG keys
        state: CrazyflowState,  # leaves: (B, ...)
        actions: jax.Array,  # (B, 4)
        params: None,
    ) -> EnvStep[CrazyflowState]:
        result = jax.vmap(partial(self.step, auto_reset=False), in_axes=(0, 0, 0, None))(
            keys, state, actions, params
        )  # observation: (B, O); scalar leaves: (B,); state leaves: (B, ...)
        observation, state = reset_batch(
            partial(self.reset, params=params), keys, result.done, (result.observation, result.state)
        )  # (B, O); state leaves: (B, ...)
        return result._replace(observation=observation, state=state)

    def render(self, state: CrazyflowState) -> NDArray[np.uint8]:  # leaves: (...) -> (400, 400, 3)
        targets = self._targets(state)  # (K, 3)
        if self.config.task == "reach_vel":
            targets = state.physics.states.pos[0] + targets  # (1, 3), velocity arrow endpoint
        return render_drones(state.physics.states.pos[0], targets, extent=2.0)
