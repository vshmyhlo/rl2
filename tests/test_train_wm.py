import json
from pathlib import Path
from typing import Any, SupportsFloat

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import serialization
from flax.training.train_state import TrainState
from numpy.typing import NDArray
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rl2 import train_wm
from rl2.wm import MambaWorldModel


class ShortEpisodes(gym.Env[NDArray[np.uint8], int]):
    def __init__(self, timeout: bool = False) -> None:
        self.observation_space = gym.spaces.Box(0, 255, (1, 2, 2), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(3)
        self.timeout = timeout
        self.elapsed = 0
        self.closed = False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.uint8], dict[str, Any]]:
        super().reset(seed=seed)
        self.elapsed = 0
        return np.zeros(self.observation_space.shape, np.uint8), {}

    def step(self, action: int) -> tuple[NDArray[np.uint8], SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)
        self.elapsed += 1
        done = self.elapsed == 2
        observation = np.full(self.observation_space.shape, 10 * self.elapsed, np.uint8)
        return observation, float(action), done and not self.timeout, done and self.timeout, {}

    def close(self) -> None:
        self.closed = True


def test_random_collection_preserves_terminal_frames_and_reset_masks() -> None:
    envs = gym.vector.SyncVectorEnv(
        [ShortEpisodes, lambda: ShortEpisodes(timeout=True)], autoreset_mode=gym.vector.AutoresetMode.DISABLED
    )
    try:
        observation, _ = envs.reset(seed=3)
        start = np.ones(2, dtype=np.bool_)
        rng = np.random.default_rng(3)
        batch, observation, start = train_wm.collect_rollout(envs, observation, start, rng, 3)
        np.testing.assert_array_equal(batch.observations[:, :, 0, 0, 0], [[0, 0], [10, 10], [0, 0]])
        np.testing.assert_array_equal(batch.next_observations[:, :, 0, 0, 0], [[10, 10], [20, 20], [10, 10]])
        np.testing.assert_array_equal(batch.episode_starts, [[True, True], [False, False], [True, True]])
        np.testing.assert_array_equal(batch.terminated, [[False, False], [True, False], [False, False]])
        np.testing.assert_array_equal(batch.rewards, batch.actions)
        expected_rng = np.random.default_rng(3)
        expected_actions = np.stack([expected_rng.integers(3, size=2, dtype=np.int32) for _ in range(3)])
        np.testing.assert_array_equal(batch.actions, expected_actions)
        following, _, _ = train_wm.collect_rollout(envs, observation, start, rng, 2)
        np.testing.assert_array_equal(following.observations[0], batch.next_observations[-1])
        np.testing.assert_array_equal(following.episode_starts, [[False, False], [True, True]])
    finally:
        envs.close()


def test_update_targets_losses_carry_and_learning() -> None:
    model = MambaWorldModel((1, 2, 2), 3, d_model=8, d_state=4, headdim=4)
    batch = train_wm.Batch(
        observations=jnp.zeros((3, 2, 1, 2, 2), jnp.uint8),
        actions=jnp.arange(6, dtype=jnp.int32).reshape(3, 2) % 3,
        next_observations=jnp.full((3, 2, 1, 2, 2), 128, jnp.uint8),
        rewards=jnp.ones((3, 2), jnp.float32),
        terminated=jnp.array([[False, False], [True, False], [False, False]]),
        episode_starts=jnp.array([[True, True], [False, False], [True, True]]),
    )
    inputs = batch.observations.astype(jnp.float32) / 255.0
    variables = model.init(jax.random.key(0), inputs, batch.actions)
    state = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optax.adam(1e-3))
    carry = model.initial_carry(2)
    expected_carry, _, prediction = model.apply(
        variables, inputs, batch.actions, carry, batch.episode_starts, method=model.observe
    )
    state, final, metrics = train_wm.update(state, model, batch, carry)
    expected_observation = jnp.square(prediction.observation - 128 / 255.0).mean()
    expected_reward = jnp.square(prediction.reward - 1.0).mean()
    expected_terminal = optax.sigmoid_binary_cross_entropy(
        prediction.termination_logits, batch.terminated.astype(jnp.float32)
    ).mean()
    np.testing.assert_allclose(metrics["observation_loss"], expected_observation, rtol=1e-5)
    np.testing.assert_allclose(metrics["reward_loss"], expected_reward, rtol=1e-5)
    np.testing.assert_allclose(metrics["termination_loss"], expected_terminal, rtol=1e-5)
    np.testing.assert_allclose(metrics["loss"], expected_observation + expected_reward + expected_terminal, rtol=1e-5)
    chex.assert_trees_all_equal_shapes_and_dtypes(final, expected_carry)
    for actual, expected in zip(final, expected_carry):
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    first_loss = float(metrics["loss"])
    for _ in range(4):
        state, _, metrics = train_wm.update(state, model, batch, carry)
    assert int(state.step) == 5
    assert float(metrics["loss"]) < first_loss
    assert all(np.isfinite(value) for value in metrics.values())


def test_training_saves_checkpoint_logs_and_handles_short_final_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environments: list[ShortEpisodes] = []

    def make_env(env_id: str, atari_preprocessing: bool, observation_size: int) -> ShortEpisodes:
        assert env_id == "ALE/SpaceInvaders-v5"
        assert atari_preprocessing
        env = ShortEpisodes()
        environments.append(env)
        return env

    monkeypatch.setattr(train_wm, "make_env", make_env)
    config = train_wm.Config(
        total_steps=10, num_envs=2, num_steps=3, d_model=8, d_state=4, headdim=4, log_dir=str(tmp_path)
    )
    run_dir = train_wm.train(config)
    saved = serialization.msgpack_restore((run_dir / "checkpoint.msgpack").read_bytes())
    assert int(saved["step"]) == 2
    assert "mixer" in saved["params"]
    assert "opt_state" in saved
    metadata = json.loads((run_dir / "config.json").read_text())
    assert metadata["env_id"] == "ALE/SpaceInvaders-v5"
    assert metadata["observation_shape"] == [1, 2, 2]
    events = EventAccumulator(str(run_dir)).Reload()
    assert [event.step for event in events.Scalars("train/loss")] == [6, 10]
    assert all(env.closed for env in environments)
    assert not (run_dir / "checkpoint.msgpack.tmp").exists()


@pytest.mark.parametrize(
    "options",
    [{"num_steps": 0}, {"total_steps": 9, "num_envs": 2}, {"learning_rate": float("inf")}, {"d_state": 2}],
)
def test_invalid_config(options: dict[str, Any]) -> None:
    with pytest.raises((AssertionError, ValueError)):
        train_wm.Config(**options)
