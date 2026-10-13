"""Single-environment XLand-MiniGrid tasks with categorical observations."""

from functools import partial
from typing import Annotated, Self

import jax
import jax.numpy as jnp
import numpy as np
import xminigrid
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass
from xminigrid.benchmarks import Benchmark
from xminigrid.environment import EnvParams
from xminigrid.rendering.rgb_render import render as render_grid
from xminigrid.types import TimeStep

from rl2.envs.base import BaseEnv, EnvStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class XMiniGridConfig:
    env_id: Annotated[str, Field(pattern=r"^(MiniGrid|XLand-MiniGrid)-")] = "MiniGrid-Empty-5x5"
    benchmark: Annotated[str, Field(min_length=1)] | None = None
    ruleset_id: Annotated[int, Field(ge=0, strict=True)] | None = None
    max_steps: Annotated[int, Field(gt=0, lt=2**31, strict=True)] | None = None

    @model_validator(mode="after")
    def validate_benchmark(self) -> Self:
        if self.env_id.startswith("XLand-") != (self.benchmark is not None):
            raise ValueError("XLand environments require a benchmark; MiniGrid environments do not use benchmarks")
        if self.ruleset_id is not None and self.benchmark is None:
            raise ValueError("ruleset_id requires a benchmark")
        return self


class XMiniGridEnv(BaseEnv[TimeStep, EnvParams]):
    def __init__(self, config: XMiniGridConfig) -> None:
        if config.env_id not in xminigrid.registered_environments():
            raise ValueError(f"Unknown XLand-MiniGrid environment: {config.env_id}")
        self.env, self.params = xminigrid.make(config.env_id)
        if config.max_steps is not None:
            self.params = self.params.replace(max_steps=config.max_steps)
        self.benchmark: Benchmark | None = None
        if config.benchmark is not None:
            if config.benchmark not in xminigrid.registered_benchmarks():
                raise ValueError(f"Unknown XLand-MiniGrid benchmark: {config.benchmark}")
            benchmark = xminigrid.load_benchmark(config.benchmark)
            if config.ruleset_id is None:
                self.benchmark = benchmark
            else:
                if config.ruleset_id >= benchmark.num_rulesets():
                    raise ValueError("ruleset_id must be smaller than the benchmark's number of rulesets")
                self.params = self.params.replace(ruleset=benchmark.get_ruleset(config.ruleset_id))
        self.num_actions = self.env.num_actions(self.params)

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, TimeStep]:  # key: ()
        """Return (H, W, 2) uint8 tile/color IDs and (...) timestep leaves."""
        if self.benchmark is not None:
            key, ruleset_key = jax.random.split(key)  # each () typed PRNG key
            params = params.replace(ruleset=self.benchmark.sample_ruleset(ruleset_key))
        timestep = self.env.reset(params, key)  # observation: (H, W, 2) tile/color IDs; state leaves: (...)
        return timestep.observation, timestep  # (H, W, 2); leaves: (...)

    @partial(jax.jit, static_argnames=("self", "auto_reset"))
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: TimeStep,  # leaves: (...)
        action: jax.Array,  # ()
        params: EnvParams,
        *,
        auto_reset: bool = True,
    ) -> EnvStep[TimeStep]:
        """Advance and autoreset; info retains (H, W, 2) final_observation tile/color IDs."""
        timestep = self.env.step(params, state, action)  # observation: (H, W, 2) tile/color IDs; state leaves: (...)
        terminated = timestep.last() & (timestep.discount == 0)  # ()
        truncated = timestep.state.step_num >= params.max_steps  # ()
        done = terminated | truncated  # ()

        def reset_done(_: None) -> TimeStep:
            return self.reset(key, params)[1]  # leaves: (...)

        def continue_episode(_: None) -> TimeStep:
            return timestep  # leaves: (...)

        state = jax.lax.cond(done, reset_done, continue_episode, None) if auto_reset else timestep  # leaves: (...)
        return EnvStep(
            state.observation,  # (H, W, 2)
            state,  # leaves: (...)
            timestep.reward,  # ()
            done,  # ()
            {
                "terminated": terminated,  # ()
                "truncated": truncated,  # ()
                "final_observation": timestep.observation,  # (H, W, 2)
                "final_state": timestep,  # timestep leaves: (...), before autoreset
            },
        )

    @partial(jax.jit, static_argnames="self")
    def step_batch(
        self,
        keys: jax.Array,  # (B,) typed PRNG keys
        state: TimeStep,  # leaves: (B, ...)
        actions: jax.Array,  # (B,)
        params: EnvParams,
    ) -> EnvStep[TimeStep]:
        """Skip batch reset generation when no episode ends; retain (B, ...) final info."""
        result = jax.vmap(partial(self.step, auto_reset=False), in_axes=(0, 0, 0, None))(
            keys, state, actions, params
        )  # observations: (B, H, W, 2); scalar leaves: (B,); state leaves: (B, ...)

        def reset_done(_: None) -> TimeStep:
            _, reset_state = jax.vmap(self.reset, in_axes=(0, None))(keys, params)  # (B, H, W, 2); leaves: (B, ...)

            def select(reset: jax.Array, current: jax.Array) -> jax.Array:  # each (B, ...) -> (B, ...)
                mask = result.done.reshape((-1,) + (1,) * (current.ndim - 1))  # (B, 1, ...)
                return jnp.where(mask, reset, current)  # (B, ...)

            return jax.tree.map(select, reset_state, result.state)  # leaves: (B, ...)

        state = jax.lax.cond(result.done.any(), reset_done, lambda _: result.state, None)  # leaves: (B, ...)
        return result._replace(observation=state.observation, state=state)

    def render(self, state: TimeStep) -> NDArray[np.uint8]:  # timestep leaves: (...) -> (P, Q, 3) pixels
        """Render the full grid, actual agent pose, and highlighted observation window."""
        return render_grid(np.asarray(state.state.grid), state.state.agent, self.params.view_size)  # (P, Q, 3)
