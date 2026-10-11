from functools import partial
from typing import Any
from unittest.mock import Mock

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from rl2.jax_env import JaxVectorEnv


@pytest.mark.parametrize(
    "num_envs,observation_shape", [(1, ()), (2, (2, 3))], ids=["singleton-scalar", "batched-image"]
)
def test_jax_vector_env_array_contract(num_envs: int, observation_shape: tuple[int, ...]) -> None:
    shape = (num_envs, *observation_shape)  # (B, *O)
    observations = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)  # (B, *O)
    rewards = np.arange(num_envs, dtype=np.float64) + 0.5  # (B,)
    terminated = np.arange(num_envs) % 2 == 0  # (B,)
    truncated = ~terminated  # (B,)
    info = {"labels": np.full(num_envs, "game", dtype=object), "scores": rewards}  # each (B,)
    base = Mock(spec=gym.vector.VectorEnv)
    base.reset.return_value = (observations, info)
    base.step.return_value = (observations, rewards, terminated, truncated, info)
    env = JaxVectorEnv(base)
    mask = jnp.asarray(terminated)  # (B,)
    options = {"reset_mask": mask, "mode": "test"}
    reset_obs, reset_info = env.reset(seed=7, options=options)  # (B, *O), info
    assert isinstance(reset_obs, jax.Array)
    assert reset_obs.shape == shape
    assert reset_obs.dtype == jnp.uint8
    np.testing.assert_array_equal(reset_obs, observations)
    assert reset_info is info
    assert base.reset.call_args.kwargs["seed"] == 7
    host_options = base.reset.call_args.kwargs["options"]
    assert host_options is not options
    assert isinstance(host_options["reset_mask"], np.ndarray)
    np.testing.assert_array_equal(host_options["reset_mask"], mask)
    assert host_options["mode"] == "test"
    assert options["reset_mask"] is mask

    actions = jnp.arange(num_envs, dtype=jnp.int32)  # (B,)
    obs, reward, terminal, timeout, step_info = env.step(actions)  # (B, *O), 3 x (B,), info
    for actual, expected, dtype in (
        (obs, observations, jnp.uint8),
        (reward, rewards, jnp.float32),
        (terminal, terminated, jnp.bool_),
        (timeout, truncated, jnp.bool_),
    ):
        assert isinstance(actual, jax.Array)
        assert actual.shape == expected.shape
        assert actual.dtype == dtype
        np.testing.assert_array_equal(actual, expected)
    assert step_info is info
    host_actions = base.step.call_args.args[0]  # (B,)
    assert isinstance(host_actions, np.ndarray)
    assert host_actions.dtype == np.int32
    np.testing.assert_array_equal(host_actions, actions)
    # Gym can overwrite every output buffer after returning it.
    observations.fill(99)
    rewards.fill(99)
    terminated.fill(False)
    truncated.fill(False)
    np.testing.assert_array_equal(reset_obs, np.arange(np.prod(shape), dtype=np.uint8).reshape(shape))
    np.testing.assert_array_equal(obs, reset_obs)
    np.testing.assert_array_equal(reward, np.arange(num_envs) + 0.5)
    np.testing.assert_array_equal(terminal, np.arange(num_envs) % 2 == 0)
    np.testing.assert_array_equal(timeout, np.arange(num_envs) % 2 != 0)
    env.close()
    base.close.assert_called_once()


class CounterEnv(gym.Env):
    def __init__(self, limit: int) -> None:
        self.observation_space = gym.spaces.Box(0, 10, (1,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(2)
        self.limit = limit
        self.steps = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.int32], dict[str, Any]]:
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(1, dtype=np.int32), {}  # observation: (O=1,)

    def step(self, action: int) -> tuple[NDArray[np.int32], float, bool, bool, dict[str, Any]]:
        self.steps += 1
        return (
            np.array([self.steps], dtype=np.int32),  # (O=1,)
            float(action + 1),
            self.steps == self.limit,
            False,
            {},
        )


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async-shared-memory"])
def test_jax_vector_env_partial_reset_preserves_observations(asynchronous: bool) -> None:
    vector_cls = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv
    options = {"context": "spawn"} if asynchronous else {}
    base = vector_cls(
        [partial(CounterEnv, limit=1), partial(CounterEnv, limit=3)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
        copy=False,
        **options,
    )
    env = JaxVectorEnv(base)
    try:
        assert env.num_envs == 2
        assert env.single_action_space == base.single_action_space
        initial, _ = env.reset(seed=[3, 5])  # (B=2, O=1)
        obs, rewards, terminated, truncated, _ = env.step(jnp.array([1, 0], dtype=jnp.int32))  # (B, O), 3 x (B,)
        np.testing.assert_array_equal(obs, [[1], [1]])
        np.testing.assert_array_equal(rewards, [2, 1])
        np.testing.assert_array_equal(terminated, [True, False])
        np.testing.assert_array_equal(truncated, [False, False])
        reset_options = {"reset_mask": terminated}  # mask: (B,)
        reset_obs, _ = env.reset(options=reset_options)  # (B, O)
        np.testing.assert_array_equal(reset_obs, [[0], [1]])
        assert "reset_mask" in reset_options  # Gym pops this key from its host options.
        next_obs, *_ = env.step(jnp.array([0, 1], dtype=jnp.int32))  # (B, O)
        np.testing.assert_array_equal(next_obs, [[1], [2]])
        np.testing.assert_array_equal(initial, [[0], [0]])
        np.testing.assert_array_equal(obs, [[1], [1]])
        np.testing.assert_array_equal(reset_obs, [[0], [1]])
    finally:
        env.close()
    assert base.closed
