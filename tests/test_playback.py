from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2 import ppo
from rl2.envs.base import EnvStep
from rl2.envs.observations import CraftaxObservation


def policy(
    variables: dict[str, Any],
    observation: CraftaxObservation,  # grid: (B=1, 2, 2, 1); features: (B, 2)
    carry: ppo.RecurrentCarry,  # leaves: (B, H=1)
    starts: jax.Array,  # (B,)
    *,
    method: str,
) -> ppo.ActorCriticOutput:
    assert observation.grid.shape == (1, 2, 2, 1) and observation.grid.dtype == jnp.float32
    assert observation.features.shape == (1, 2) and observation.features.dtype == jnp.float32
    carry = jax.tree.map(lambda leaf: jnp.where(starts[:, None], 0, leaf) + 1, carry)  # leaves: (B, H)
    logits = jnp.where(jnp.arange(2) == carry[0][0] % 2, 0.0, -jnp.inf)  # (B, A=2)
    return ppo.ActorCriticOutput(carry=carry, policy=ppo.CategoricalPolicy(logits), values=jnp.zeros(1))


class Env:
    params = None

    def reset(self, key: jax.Array, params: None) -> tuple[CraftaxObservation, dict[str, jax.Array]]:  # key: ()
        observation = CraftaxObservation(grid=jnp.zeros((2, 2, 1)), features=jnp.zeros(2))  # (2, 2, 1); (2,)
        state = {"step": jnp.asarray(0), "action": jnp.asarray(0)}  # each ()
        return observation, state

    def step(
        self,
        key: jax.Array,  # () typed PRNG key
        state: dict[str, jax.Array],  # leaves: ()
        action: jax.Array,  # ()
        params: None,
    ) -> EnvStep:
        observation, reset = self.reset(key, params)  # (2, 2, 1)/(2,); state leaves: ()
        length = state["step"] + 1  # ()
        done = length == 35  # ()
        final = {"step": length, "action": action}  # leaves: ()
        return EnvStep(
            observation=observation,
            state=jax.tree.map(lambda fresh, old: jnp.where(done, fresh, old), reset, final),  # leaves: ()
            reward=length.astype(jnp.float32),  # ()
            done=done,  # ()
            info={
                "terminated": done,  # ()
                "truncated": jnp.asarray(False),  # ()
                "final_observation": observation,
                "final_state": final,  # leaves: ()
            },
        )


@pytest.fixture(scope="module")
def env() -> Env:
    return Env()


@pytest.mark.parametrize(
    "capacity,stop,length,frames",
    [(0, True, 35, 0), (1, True, 0, 1), (34, True, 33, 34), (34, False, 35, 34), (36, False, 35, 36)],
    ids=["no-recording", "initial-only", "capped-video", "finish-after-cap", "whole-episode"],
)
def test_compiled_playback(env: Env, capacity: int, stop: bool, length: int, frames: int) -> None:
    memory = ((jnp.full((1, 1), 9.0), jnp.full((1, 1), 9.0)),)  # leaves: (B=1, H=1)
    result = ppo.run_episode(
        env, policy, {}, jax.random.key(0), memory, recording_capacity=capacity, stop_at_frame_limit=stop
    )
    assert result.length.shape == result.episode_return.shape == ()
    assert int(result.length) == length
    assert float(result.episode_return) == length * (length + 1) / 2
    if capacity == 0:
        assert result.states is None
    else:
        assert result.states["step"].shape == result.states["action"].shape == (capacity,)
        # Include the initial and terminal states, before autoreset returns the counter to zero.
        np.testing.assert_array_equal(result.states["step"][:frames], np.arange(frames))
        # Actions expose memory: reset initially, then retain it across every step.
        np.testing.assert_array_equal(result.states["action"][:frames], np.arange(frames) % 2)
