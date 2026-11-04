"""Pgx 2048 with native one-hot observations and same-step autoreset.

Native Pgx rules apply: an illegal move terminates the episode with reward -1.
Legal moves earn the sum of merged tile values; actions are left/up/right/down.
"""

from functools import partial
from typing import Annotated, Literal, cast

import jax
import numpy as np
from flax import struct
from numpy.typing import NDArray
from pgx.play2048 import Play2048, State
from PIL import Image, ImageDraw, ImageFont
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import DiscreteSpace
from rl2.envs.base import BaseEnv, EnvStep


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class PgxConfig:
    id: Literal["Pgx"] = "Pgx"
    game: Literal["2048"] = "2048"
    max_steps: Annotated[int, Field(gt=0, lt=2**31, strict=True)] = 10_000


@struct.dataclass
class PgxParams:
    max_steps: int = 10_000


class PgxEnv(BaseEnv[State, PgxParams]):
    def __init__(self, config: PgxConfig) -> None:
        self.env = Play2048()
        self.params = PgxParams(max_steps=config.max_steps)
        self.action_space = DiscreteSpace(n=4)

    @partial(jax.jit, static_argnames="self")
    def reset(self, key: jax.Array, params: PgxParams) -> tuple[jax.Array, State]:  # key: ()
        """Return (4, 4, 31) bool tile planes and (...) native state leaves."""
        state = cast(State, self.env.init(key))  # leaves: (...)
        return state.observation, state  # (4, 4, 31); leaves: (...)

    @partial(jax.jit, static_argnames="self")
    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: State,  # leaves: (...)
        action: jax.Array,  # () int32 action ID
        params: PgxParams,
    ) -> EnvStep[State]:
        """Advance; retain (4, 4, 31) final observation before autoreset."""
        step_key, reset_key = jax.random.split(key)  # each ()
        final_state = cast(State, self.env.step(state, action, step_key))  # leaves: (...)
        terminated = final_state.terminated  # ()
        truncated = final_state.truncated | (final_state._step_count >= params.max_steps)  # ()
        final_state = final_state.replace(truncated=truncated)  # leaves: (...)
        done = terminated | truncated  # ()

        def reset_done(_: None) -> State:
            return self.reset(reset_key, params)[1]  # leaves: (...)

        def continue_episode(_: None) -> State:
            return final_state  # leaves: (...)

        next_state = jax.lax.cond(done, reset_done, continue_episode, None)  # leaves: (...)
        return EnvStep(
            observation=next_state.observation,  # (4, 4, 31)
            state=next_state,  # leaves: (...)
            reward=final_state.rewards[0],  # () float32
            done=done,  # ()
            info={
                "terminated": terminated,  # ()
                "truncated": truncated,  # ()
                "final_observation": final_state.observation,  # (4, 4, 31)
                "final_state": final_state,  # leaves: (...)
            },
        )

    def render(self, state: State) -> NDArray[np.uint8]:  # leaves: (...) -> (256, 256, 3)
        """Return (256, 256, 3) uint8 board pixels with tile values."""
        board = np.asarray(state._board).reshape(4, 4)  # (4, 4) tile exponents
        font = ImageFont.load_default(size=18)
        with Image.new("RGB", (256, 256), "#bbada0") as frame:
            draw = ImageDraw.Draw(frame)
            for row in range(4):
                for col in range(4):
                    exponent = int(board[row, col])
                    x, y = col * 64, row * 64
                    color = "#cdc1b4" if exponent == 0 else (237, max(100, 228 - 12 * exponent), 130)
                    draw.rounded_rectangle((x + 3, y + 3, x + 61, y + 61), radius=4, fill=color)
                    if exponent:
                        draw.text((x + 32, y + 32), str(2**exponent), font=font, anchor="mm", fill="#403830")
            return np.array(frame, dtype=np.uint8)  # (256, 256, 3)
