from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rl2.playback import PlaybackState, advance_episode, finish_episode


def step(
    env: Any,
    apply_fn: Callable,
    params: optax.Params,
    state: PlaybackState,  # array leaves as in PlaybackState
) -> PlaybackState:
    length = state.length + 1  # ()
    return state._replace(
        env_state=jnp.where(length == 35, 0, length),  # () autoreset
        final_state=length,  # ()
        episode_return=state.episode_return + length,  # ()
        length=length,  # ()
        done=length == 35,  # ()
    )


class Env:
    def render(self, state: jax.Array | np.ndarray) -> np.ndarray:  # () -> (H, W, 3)
        return np.full((2, 3, 3), int(state), np.uint8)  # (H, W, 3)


@pytest.fixture(scope="module")
def env() -> Env:
    return Env()


def initial_state() -> PlaybackState:
    return PlaybackState(
        key=jax.random.key(0),  # ()
        observation=jnp.zeros(2),  # (O,)
        env_state=jnp.asarray(0),  # ()
        final_state=jnp.asarray(0),  # ()
        memory=jnp.zeros((1, 2)),  # (B, H)
        episode_starts=jnp.ones(1, dtype=bool),  # (B,)
        episode_return=jnp.asarray(0.0),  # ()
        length=jnp.asarray(0),  # ()
        done=jnp.asarray(False),  # ()
    )


@pytest.mark.parametrize(
    "record,cap,stop,length,frames",
    [
        (False, 1, True, 35, 0),
        (True, 1, True, 0, 1),
        (True, 34, True, 33, 34),
        (True, 34, False, 35, 34),
        (True, None, False, 35, 36),
    ],
    ids=["no-render", "initial-only", "cap-after-chunk", "finish-after-cap", "terminal-after-chunk"],
)
def test_compiled_playback(env: Env, record: bool, cap: int | None, stop: bool, length: int, frames: int) -> None:
    result = finish_episode(
        step,
        env,
        step,
        {},
        initial_state(),
        record=record,
        frame_limit=cap,
        stop_at_frame_limit=stop,
    )
    assert result.length == length
    assert result.episode_return == length * (length + 1) / 2
    assert len(result.frames) == frames
    for index, frame in enumerate(result.frames):  # (H, W, 3)
        assert frame.shape == (2, 3, 3) and frame.dtype == np.uint8
        np.testing.assert_array_equal(frame, index)


def test_compiled_chunk_shapes_and_terminal_padding(env: Env) -> None:
    state = initial_state()._replace(length=jnp.asarray(33), env_state=jnp.asarray(33), final_state=jnp.asarray(33))
    result, saved = advance_episode(step, env, step, {}, state, 100, record=True)  # saved: (32,)
    assert saved.shape == (32,)
    np.testing.assert_array_equal(saved, [34] + [35] * 31)
    assert result.length.shape == result.done.shape == result.episode_return.shape == ()
    assert result.observation.shape == (2,) and result.memory.shape == (1, 2)
    assert result.episode_starts.shape == (1,)
    assert int(result.length) == 35 and bool(result.done)
    assert int(result.env_state) == 0 and int(result.final_state) == 35
