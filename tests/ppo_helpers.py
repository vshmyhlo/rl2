"""Small transition fixtures for trainer orchestration tests with mocked policies."""

from typing import Any

import jax.numpy as jnp
from numpy.typing import ArrayLike, NDArray

from rl2.envs.base import EnvStep


def env_step(
    observation: NDArray[Any],  # (*O,)
    reward: ArrayLike,  # ()
    terminated: ArrayLike,  # ()
    truncated: ArrayLike,  # ()
    state: dict[str, Any],
) -> EnvStep[dict[str, Any]]:
    return EnvStep(
        jnp.asarray(observation),  # (*O,)
        state,
        jnp.asarray(reward, dtype=jnp.float32),  # ()
        jnp.asarray(terminated) | jnp.asarray(truncated),  # ()
        {
            "terminated": jnp.asarray(terminated),  # ()
            "truncated": jnp.asarray(truncated),  # ()
            "final_observation": jnp.asarray(observation),  # (*O,)
            "final_state": state,  # state pytree
        },
    )
