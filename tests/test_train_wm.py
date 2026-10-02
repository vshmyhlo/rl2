import json
import sys
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Any, SupportsFloat
from unittest.mock import Mock

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
from tensorboardX import SummaryWriter

from rl2 import train_wm
from rl2.observation_encoder import ConvStage
from rl2.wm import MambaWorldModel, categorical_kl


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
    model = MambaWorldModel(
        (1, 2, 2),
        3,
        d_model=8,
        num_layers=2,
        d_state=4,
        headdim=4,
        encoder_stages=(ConvStage(4),),
        stochastic_size=4,
        stochastic_classes=4,
    )
    batch = train_wm.Batch(
        observations=jnp.zeros((3, 2, 1, 2, 2), jnp.uint8),
        actions=jnp.arange(6, dtype=jnp.int32).reshape(3, 2) % 3,
        next_observations=jnp.full((3, 2, 1, 2, 2), 128, jnp.uint8),
        rewards=jnp.ones((3, 2), jnp.float32),
        terminated=jnp.array([[False, False], [True, False], [False, False]]),
        episode_starts=jnp.array([[True, True], [False, False], [True, True]]),
    )
    inputs = batch.observations
    key = jax.random.key(17)
    keys = jax.random.split(key, batch.actions.shape[0])
    variables = model.init(jax.random.key(0), inputs, batch.actions, batch.next_observations, keys)
    state = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optax.adam(1e-3))
    carry = model.initial_carry(2)
    expected_carry, output = model.apply(
        variables,
        inputs,
        batch.actions,
        batch.next_observations,
        keys,
        carry,
        batch.episode_starts,
        method=model.observe,
    )
    prediction = output.prediction
    state, final, metrics = train_wm.update(state, model, batch, carry, key, 1.0, 0.1, 0.0)
    expected_observation = jnp.square(prediction.observation - 128 / 255.0).mean()
    expected_reward = jnp.square(prediction.reward - 1.0).mean()
    expected_terminal = optax.sigmoid_binary_cross_entropy(
        prediction.termination_logits, batch.terminated.astype(jnp.float32)
    ).mean()
    np.testing.assert_allclose(metrics["observation_loss"], expected_observation, rtol=1e-5)
    np.testing.assert_allclose(metrics["reward_loss"], expected_reward, rtol=1e-5)
    np.testing.assert_allclose(metrics["termination_loss"], expected_terminal, rtol=1e-5)
    expected_kl = categorical_kl(output.posterior_logits, output.prior_logits).mean()
    np.testing.assert_allclose(metrics["kl"], expected_kl, rtol=1e-5)
    np.testing.assert_allclose(
        metrics["loss"], 2 * expected_observation + expected_reward + expected_terminal + 1.1 * expected_kl, rtol=1e-5
    )
    chex.assert_trees_all_equal_shapes_and_dtypes(final, expected_carry)
    for actual, expected in zip(jax.tree.leaves(final), jax.tree.leaves(expected_carry)):
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    first_loss = float(metrics["loss"])
    for _ in range(4):
        state, _, metrics = train_wm.update(state, model, batch, carry, key, 1.0, 0.1, 0.0)
    assert int(state.step) == 5
    assert float(metrics["loss"]) < first_loss
    assert all(np.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("checkpoint_every", [1, 100])
def test_training_saves_checkpoint_logs_and_handles_short_final_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: bool, checkpoint_every: int
) -> None:
    environments: list[ShortEpisodes] = []
    flushes: list[str] = []

    def make_writer(logdir: str, flush_secs: int) -> SummaryWriter:
        assert flush_secs == config.log_flush_secs
        writer = SummaryWriter(logdir=logdir, flush_secs=flush_secs)
        original_flush = writer.flush
        original_close = writer.close

        def flush() -> None:
            flushes.append("flush")
            original_flush()

        def close() -> None:
            flushes.append("close")
            original_close()

        monkeypatch.setattr(writer, "flush", flush)
        monkeypatch.setattr(writer, "close", close)
        return writer

    monkeypatch.setattr(train_wm, "SummaryWriter", make_writer)
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
    checkpoint_spy = Mock(wraps=train_wm.save_checkpoint)
    monkeypatch.setattr(train_wm, "save_checkpoint", checkpoint_spy)
    config = training_config(
        total_steps=10,
        num_envs=2,
        num_steps=3,
        d_model=8,
        num_layers=2,
        encoder_stages=(ConvStage(4),),
        stochastic_size=4,
        stochastic_classes=4,
        d_state=4,
        headdim=4,
        atari_preprocessing=True,
        log_flush_secs=7,
        log_dir="gs://test-bucket/rl2/wm/" if remote else f"{tmp_path}/logs/",
        checkpoint_dir="gs://test-bucket/rl2_cp/wm/" if remote else f"{tmp_path}/checkpoints/",
        checkpoint_every=checkpoint_every,
        video_every_steps=5 if checkpoint_every == 1 else 0,
        video_num_steps=3,
    )
    run_dir = train_wm.train(config)
    assert flushes == ["flush"] * (4 if config.video_every_steps else 2) + ["close"]
    assert run_dir.startswith(f"{config.log_dir.rstrip('/')}/ALE_SpaceInvaders-v5_seed1_")
    run_name = run_dir.rsplit("/", 1)[-1]
    checkpoint_run_dir = f"{config.checkpoint_dir.rstrip('/')}/{run_name}"
    local_dir = tmp_path / "download" if remote else Path(run_dir)
    local_checkpoint_dir = tmp_path / "download_checkpoints" if remote else Path(checkpoint_run_dir)
    if remote:
        assert uploads
        assert not (tmp_path / "gs:").exists()
        local_dir.mkdir()
        local_checkpoint_dir.mkdir()
        for path, data in uploads.items():
            assert path.startswith((f"{run_dir}/", f"{checkpoint_run_dir}/"))
            destination = local_checkpoint_dir if path.startswith(f"{checkpoint_run_dir}/") else local_dir
            (destination / path.rsplit("/", 1)[-1]).write_bytes(data)
    assert checkpoint_spy.call_count == (2 if checkpoint_every == 1 else 1)
    assert checkpoint_spy.call_args.args[1] == checkpoint_run_dir
    assert not (local_dir / "checkpoint.msgpack").exists()
    saved = serialization.msgpack_restore((local_checkpoint_dir / "checkpoint.msgpack").read_bytes())
    assert int(saved["step"]) == 2
    assert "layers_1" in saved["params"]["dynamics"]
    assert "prior_head" in saved["params"]
    assert "posterior_head" in saved["params"]
    assert "opt_state" in saved
    metadata = json.loads((local_dir / "config.json").read_text())
    assert json.loads((local_checkpoint_dir / "config.json").read_text()) == metadata
    assert metadata["env_id"] == "ALE/SpaceInvaders-v5"
    assert metadata["observation_shape"] == [1, 2, 2]
    events = EventAccumulator(str(local_dir)).Reload()
    assert [event.step for event in events.Scalars("train/loss")] == [6, 10]
    assert [event.step for event in events.Scalars("train/kl")] == [6, 10]
    assert events.Scalars("train/prior_entropy")
    assert events.Scalars("train/posterior_entropy")
    config_text = events.Tensors("config/text_summary")[0].tensor_proto.string_val[0].decode()
    assert config_text.startswith("```yaml\n")
    assert config.log_dir in config_text
    assert events.Tensors("devices/text_summary")
    if config.video_every_steps:
        videos = events.Images("imagination/random_policy")
        assert [video.step for video in videos] == [6, 10]
        assert all(video.width == 2 and video.height == 2 for video in videos)
        assert all(video.encoded_image_string.startswith(b"GIF") for video in videos)
    else:
        assert "imagination/random_policy" not in events.Tags()["images"]
    assert all(env.closed for env in environments)
    assert not (local_checkpoint_dir / "checkpoint.msgpack.tmp").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"num_steps": 0},
        {"total_steps": 9, "num_envs": 2},
        {"learning_rate": float("inf")},
        {"d_state": 2},
        {"video_every_steps": -1},
        {"video_num_steps": 0},
        {"video_fps": 0},
        {"video_fps": float("inf")},
        {"log_flush_secs": 0},
        {"log_flush_secs": -1},
        {"encoder_stages": ()},
        {"encoder_stages": (ConvStage(4), ConvStage(8, project=False))},
        {"num_layers": 0},
        {"d_intermediate": -1},
        {"stochastic_size": 0},
        {"stochastic_classes": 1},
        {"unimix": -0.1},
        {"unimix": 1.0},
        {"unimix": float("nan")},
        {"dynamics_kl_scale": -1},
        {"representation_kl_scale": float("inf")},
        {"free_nats": -1},
    ],
)
def test_invalid_config(options: dict[str, Any]) -> None:
    with pytest.raises((AssertionError, ValueError)):
        training_config(**options)


def test_imagination_feeds_back_latents_and_resets_history() -> None:
    model = MambaWorldModel(
        (2, 3, 4),
        3,
        d_model=8,
        num_layers=2,
        d_state=4,
        headdim=4,
        encoder_stages=(ConvStage(4),),
        stochastic_size=4,
        stochastic_classes=4,
    )
    observation = jnp.arange(24, dtype=jnp.uint8).reshape((1, 2, 3, 4))
    normalized = observation.astype(jnp.float32) / 255.0
    actions = jnp.array([[0], [1], [2]], jnp.int32)
    key = jax.random.key(8)
    variables = model.init(key, observation, actions[0], observation, key)
    state = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optax.sgd(0.01))
    history, _ = model.apply(variables, observation, actions[0], observation, key, method=model.observe)
    frames = train_wm.imagine_frames(state, model, observation, history, jnp.zeros(1, jnp.bool_), actions, key)
    chex.assert_shape(frames, (4, 2, 3, 4))
    chex.assert_type(frames, jnp.float32)
    np.testing.assert_array_equal(frames[0], normalized[0])
    keys = jax.random.split(key, 4)
    history = model.apply(variables, observation, history, jnp.zeros(1, jnp.bool_), keys[0], method=model.condition)
    for index, action in enumerate(actions):
        history, prediction = model.apply(variables, history, action, keys[index + 1], method=model.imagine)
        np.testing.assert_allclose(frames[index + 1], prediction.observation[0], atol=2e-6)
    reset = train_wm.imagine_frames(state, model, observation, history, jnp.ones(1, jnp.bool_), actions, key)
    fresh = train_wm.imagine_frames(
        state, model, observation, model.initial_carry(1), jnp.zeros(1, jnp.bool_), actions, key
    )
    np.testing.assert_allclose(reset, fresh, atol=2e-6)


@pytest.mark.parametrize("rgb", [False, True])
def test_video_uses_newest_frame_clips_pixels_and_has_repeatable_actions(
    monkeypatch: pytest.MonkeyPatch, rgb: bool
) -> None:
    shape = (2, 3, 4, 3) if rgb else (2, 3, 4)
    model = MambaWorldModel(
        shape,
        3,
        d_model=8,
        num_layers=2,
        d_state=4,
        headdim=4,
        encoder_stages=(ConvStage(4),),
        stochastic_size=4,
        stochastic_classes=4,
    )
    config = training_config(video_num_steps=2, video_fps=12)
    frames = np.ones((3, *shape), np.float32)
    frames[:, -1] = np.array([-1, 0.5, 2], np.float32).reshape((3,) + (1,) * (len(shape) - 1))
    imagine = Mock(return_value=jnp.asarray(frames))
    monkeypatch.setattr(train_wm, "imagine_frames", imagine)
    writer = Mock()
    state = Mock(spec=TrainState)
    for steps in (10, 20):
        train_wm.log_video(
            state,
            model,
            config,
            writer,
            np.zeros((1, *shape), np.uint8),
            model.initial_carry(1),
            np.zeros(1, np.bool_),
            steps,
        )
    call = writer.add_video.call_args
    assert call.args[0] == "imagination/random_policy"
    assert call.args[2] == 20
    assert call.kwargs["fps"] == 12
    assert writer.flush.call_count == 2
    video = call.args[1]
    chex.assert_shape(video, (1, 3, 3 if rgb else 1, 3, 4))
    chex.assert_type(video, np.uint8)
    for frame, value in zip(video[0], (0, 128, 255)):
        np.testing.assert_array_equal(frame, np.full_like(frame, value))
    np.testing.assert_array_equal(imagine.call_args_list[0].args[-2], imagine.call_args_list[1].args[-2])
    np.testing.assert_array_equal(
        jax.random.key_data(imagine.call_args_list[0].args[-1]), jax.random.key_data(imagine.call_args_list[1].args[-1])
    )


def test_bf16_is_default_and_can_be_disabled(tmp_path: Path) -> None:
    assert training_config().bf16
    settings = asdict(training_config())
    del settings["bf16"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    assert train_wm.load_config(path).bf16
    settings["bf16"] = False
    path.write_text(yaml.safe_dump(settings))
    assert not train_wm.load_config(path).bf16


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
