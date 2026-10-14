"""Compiled episode playback with bounded state buffers for host rendering."""

from collections.abc import Callable
from functools import partial
from operator import itemgetter
from typing import Any, NamedTuple

import jax
import numpy as np
import optax
from numpy.typing import NDArray


class PlaybackState[Observation, State, Carry](NamedTuple):
    key: jax.Array  # () typed PRNG key
    observation: Observation  # leaves: (*O,)
    env_state: State  # leaves: (...)
    final_state: State  # leaves: (...), before autoreset
    memory: Carry  # leaves: (B, H)
    episode_starts: jax.Array  # (B,)
    episode_return: jax.Array  # ()
    length: jax.Array  # ()
    done: jax.Array  # ()


class EpisodeResult(NamedTuple):
    episode_return: float
    length: int
    frames: list[NDArray[np.uint8]]  # each (H, W, 3)


type Step = Callable[[Any, Callable, optax.Params, PlaybackState], PlaybackState]


@partial(jax.jit, static_argnames=("step", "env", "apply_fn", "record"))
def advance_episode(
    step: Step,
    env: Any,
    apply_fn: Callable,
    params: optax.Params,
    state: PlaybackState,  # array leaves as documented above
    limit: int,
    *,
    record: bool,
) -> tuple[PlaybackState, Any]:  # state leaves as above; saved states: (32, ...) or None
    def active(state: PlaybackState) -> jax.Array:  # state leaves as above; ()
        return ~state.done & (state.length < limit)  # ()

    def advance(state: PlaybackState) -> PlaybackState:
        return step(env, apply_fn, params, state)

    if not record:
        return jax.lax.while_loop(active, advance, state), None

    def body(state: PlaybackState, _: None) -> tuple[PlaybackState, Any]:
        state = jax.lax.cond(active(state), advance, lambda state: state, state)
        return state, state.final_state  # saved state leaves: (...)

    return jax.lax.scan(body, state, None, length=32)  # saved state leaves: (32, ...)


def finish_episode(
    step: Step,
    env: Any,
    apply_fn: Callable,
    params: optax.Params,
    state: PlaybackState,  # array leaves as documented above
    *,
    record: bool,
    frame_limit: int | None,
    stop_at_frame_limit: bool,
) -> EpisodeResult:
    frames = [np.asarray(env.render(state.env_state))] if record else []  # each (H, W, 3)
    limit = frame_limit - 1 if record and frame_limit is not None else 2**31 - 1
    length = 0
    done = False
    while record and length < limit and not done:
        state, saved = advance_episode(step, env, apply_fn, params, state, limit, record=True)
        new_length, done, saved = jax.device_get((state.length, state.done, saved))  # (); (); leaves: (32, ...)
        for index in range(int(new_length) - length):
            final_state = jax.tree.map(itemgetter(index), saved)  # leaves: (...)
            frames.append(np.asarray(env.render(final_state)))  # (H, W, 3)
        length = int(new_length)
    if not done and not (record and stop_at_frame_limit):
        state, _ = advance_episode(step, env, apply_fn, params, state, 2**31 - 1, record=False)
    episode_return, length = jax.device_get((state.episode_return, state.length))  # each ()
    return EpisodeResult(episode_return=float(episode_return), length=int(length), frames=frames)
