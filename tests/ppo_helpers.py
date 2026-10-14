"""Small transition fixtures for trainer orchestration tests with mocked policies."""

from typing import Any
from unittest.mock import Mock

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


def metric_history(writer: Mock) -> dict[str, list[tuple[float, int]]]:
    history: dict[str, list[tuple[float, int]]] = {}
    for call in writer.log_metrics.call_args_list:
        metrics, step = call.args
        for tag, value in metrics.items():
            history.setdefault(tag, []).append((value, step))
    return history
