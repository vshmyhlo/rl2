"""Single-environment XLand-MiniGrid tasks with categorical or RGB observations."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Annotated, Self

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass

from rl2.envs.gymnax_env import EnvStep

if TYPE_CHECKING:
    from xminigrid.environment import EnvParams
    from xminigrid.types import TimeStep


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


class XMiniGridEnv:
    def __init__(self, config: XMiniGridConfig, *, rgb: bool = True) -> None:
        try:
            import xminigrid
        except ModuleNotFoundError as error:
            if error.name == "xminigrid":
                raise ImportError(
                    "XLand-MiniGrid requires the optional dependency: uv sync --extra xminigrid"
                ) from error
            raise

        if config.env_id not in xminigrid.registered_environments():
            raise ValueError(f"Unknown XLand-MiniGrid environment: {config.env_id}")
        self.env, self.params = xminigrid.make(config.env_id)
        self.rgb = rgb
        if rgb:
            from xminigrid.experimental.img_obs import RGBImgObservationWrapper

            self.env = RGBImgObservationWrapper(self.env)
        if config.max_steps is not None:
            self.params = self.params.replace(max_steps=config.max_steps)
        self.benchmark = None
        if config.benchmark is not None:
            if config.benchmark not in xminigrid.registered_benchmarks():
                raise ValueError(f"Unknown XLand-MiniGrid benchmark: {config.benchmark}")
            self.benchmark = xminigrid.load_benchmark(config.benchmark)
            if config.ruleset_id is not None:
                if config.ruleset_id >= self.benchmark.num_rulesets():
                    raise ValueError("ruleset_id must be smaller than the benchmark's number of rulesets")
                self.params = self.params.replace(ruleset=self.benchmark.get_ruleset(config.ruleset_id))
                self.benchmark = None
        self.num_actions = self.env.num_actions(self.params)

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, TimeStep]:  # key: ()
        """Return (*O,) uint8 observation and (...) timestep leaves; O is (3,H,W) or (H,W,2)."""
        if self.benchmark is not None:
            key, ruleset_key = jax.random.split(key)  # each () typed PRNG key
            params = params.replace(ruleset=self.benchmark.sample_ruleset(ruleset_key))
        timestep = self.env.reset(params, key)  # observation: (H,W,3) RGB or (H,W,2) IDs; state leaves: (...)
        observation = jnp.moveaxis(timestep.observation, -1, 0) if self.rgb else timestep.observation  # (*O,)
        return observation, timestep  # (*O,); leaves: (...)

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: TimeStep,  # leaves: (...)
        action: jax.Array,  # ()
        params: EnvParams,
    ) -> EnvStep[TimeStep]:
        """Advance and autoreset; info retains (*O,) final_observation in the selected format."""
        timestep = self.env.step(params, state, action)  # observation: (H,W,3) RGB or (H,W,2) IDs; state leaves: (...)
        final_observation = jnp.moveaxis(timestep.observation, -1, 0) if self.rgb else timestep.observation  # (*O,)
        terminated = timestep.last() & (timestep.discount == 0)  # ()
        truncated = timestep.state.step_num >= params.max_steps  # ()
        done = terminated | truncated  # ()

        def reset_done(_: None) -> tuple[jax.Array, TimeStep]:
            return self.reset(key, params)  # (*O,); leaves: (...)

        def continue_episode(_: None) -> tuple[jax.Array, TimeStep]:
            return final_observation, timestep  # (*O,); leaves: (...)

        observation, state = jax.lax.cond(done, reset_done, continue_episode, None)  # (*O,); leaves: (...)
        return (
            observation,  # (*O,)
            state,  # leaves: (...)
            timestep.reward,  # ()
            done,  # ()
            {
                "terminated": terminated,  # ()
                "truncated": truncated,  # ()
                "final_observation": final_observation,  # (*O,)
            },
        )

    def render_observation(self, observation: jax.Array) -> NDArray[np.uint8]:  # (*O,) -> (P, Q, 3) pixels
        if self.rgb:
            return np.moveaxis(np.asarray(observation), 0, -1)  # (H, W, 3)
        from xminigrid.rendering.rgb_render import render_tile

        grid = np.asarray(observation)  # (H, W, 2)
        height, width = grid.shape[:2]
        tile_size = 32
        image = np.empty((height * tile_size, width * tile_size, 3), dtype=np.uint8)  # (P, Q, 3)
        for y in range(height):
            for x in range(width):
                image[y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size] = render_tile(
                    tuple(grid[y, x]), agent_direction=0 if (y, x) == (height - 1, width // 2) else None
                )  # (32, 32, 3)
        return image  # (P, Q, 3)
