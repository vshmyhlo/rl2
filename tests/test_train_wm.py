import json
import sys
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Any, SupportsFloat

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml
from flax import serialization
from flax.training.train_state import TrainState
from google.cloud import storage
from numpy.typing import NDArray
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rl2 import train_wm
from rl2.wm import MambaWorldModel


def training_config(**overrides: Any) -> train_wm.Config:
    path = Path(__file__).resolve().parents[1] / "configs/train_wm_atari.yaml"
    return replace(train_wm.load_config(path), **overrides)


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


@pytest.mark.parametrize("remote", [False, True])
def test_training_saves_checkpoint_logs_and_handles_short_final_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: bool
) -> None:
    environments: list[ShortEpisodes] = []
    uploads: dict[str, bytes] = {}
    if remote:
        client = storage.Client.create_anonymous_client()

        def make_client() -> storage.Client:
            return client

        def upload(blob: storage.Blob, data: bytes, **kwargs: Any) -> None:
            # TensorBoard may close again from __del__ after monkeypatch teardown.
            blob.upload_from_string = partial(upload, blob)
            uploads[f"gs://{blob.bucket.name}/{blob.name}"] = data

        monkeypatch.setattr(storage, "Client", make_client)
        monkeypatch.setattr(storage.Blob, "upload_from_string", upload)
        monkeypatch.chdir(tmp_path)

    def make_env(
        env_id: str, frame_stack: bool, atari_preprocessing: bool, observation_size: int | None
    ) -> ShortEpisodes:
        assert env_id == "ALE/SpaceInvaders-v5"
        assert not frame_stack
        assert atari_preprocessing
        env = ShortEpisodes()
        environments.append(env)
        return env

    monkeypatch.setattr(train_wm, "make_env", make_env)
    config = training_config(
        total_steps=10,
        num_envs=2,
        num_steps=3,
        d_model=8,
        d_state=4,
        headdim=4,
        log_dir="gs://test-bucket/rl2/" if remote else f"{tmp_path}/",
        checkpoint_every=1,
    )
    run_dir = train_wm.train(config)
    assert run_dir.startswith(f"{config.log_dir.rstrip('/')}/ALE_SpaceInvaders-v5_seed1_")
    local_dir = tmp_path / "download" if remote else Path(run_dir)
    if remote:
        assert uploads
        assert not (tmp_path / "gs:").exists()
        local_dir.mkdir()
        for path, data in uploads.items():
            assert path.startswith(f"{run_dir}/")
            (local_dir / path.rsplit("/", 1)[-1]).write_bytes(data)
    saved = serialization.msgpack_restore((local_dir / "checkpoint.msgpack").read_bytes())
    assert int(saved["step"]) == 2
    assert "mixer" in saved["params"]
    assert "opt_state" in saved
    metadata = json.loads((local_dir / "config.json").read_text())
    assert metadata["env_id"] == "ALE/SpaceInvaders-v5"
    assert metadata["observation_shape"] == [1, 2, 2]
    events = EventAccumulator(str(local_dir)).Reload()
    assert [event.step for event in events.Scalars("train/loss")] == [6, 10]
    config_text = events.Tensors("config/text_summary")[0].tensor_proto.string_val[0].decode()
    assert config_text.startswith("```yaml\n")
    assert config.log_dir in config_text
    assert events.Tensors("devices/text_summary")
    assert all(env.closed for env in environments)
    assert not (local_dir / "checkpoint.msgpack.tmp").exists()


@pytest.mark.parametrize(
    "options",
    [{"num_steps": 0}, {"total_steps": 9, "num_envs": 2}, {"learning_rate": float("inf")}, {"d_state": 2}],
)
def test_invalid_config(options: dict[str, Any]) -> None:
    with pytest.raises((AssertionError, ValueError)):
        training_config(**options)


def test_config_requires_all_training_settings(tmp_path: Path) -> None:
    settings = asdict(training_config())
    del settings["learning_rate"]
    path = tmp_path / "incomplete.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(TypeError, match="learning_rate"):
        train_wm.load_config(path)


def test_main_loads_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "train.yaml"
    expected = training_config(
        env_id="ALE/Pong-v5",
        num_envs=2,
        num_steps=7,
        learning_rate=0.001,
        observation_size=None,
        frame_stack=True,
        atari_preprocessing=False,
        vector_env="async",
        bf16=True,
    )
    path.write_text(yaml.safe_dump(asdict(expected)))
    calls: list[train_wm.Config] = []

    def train(config: train_wm.Config) -> str:
        calls.append(config)
        return str(tmp_path)

    monkeypatch.setattr(sys, "argv", ["train_wm", "--config", str(path)])
    monkeypatch.setattr(train_wm, "train", train)
    train_wm.main()
    assert calls == [expected]


def test_main_rejects_cli_training_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_wm", "--num-steps", "5"])
    with pytest.raises(SystemExit) as error:
        train_wm.main()
    assert error.value.code == 2


def test_main_uses_default_atari_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys, "argv", ["train_wm"])
    calls: list[train_wm.Config] = []

    def train(config: train_wm.Config) -> str:
        calls.append(config)
        return config.log_dir

    monkeypatch.setattr(train_wm, "train", train)
    train_wm.main()
    assert calls == [train_wm.load_config("configs/train_wm_atari.yaml")]
