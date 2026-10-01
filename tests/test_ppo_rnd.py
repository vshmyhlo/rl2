from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rl2 import ppo_rnd as rnd
from rl2.atari_eval import _action
from rl2.utils import RunningMeanStd


def config() -> rnd.Config:
    return rnd.load_config(Path(__file__).resolve().parents[1] / "configs/ppo_rnd_montezuma.yaml")


def test_running_moments_match_combined_samples() -> None:
    samples = np.arange(48, dtype=np.float32).reshape(6, 2, 4)
    combined, split = RunningMeanStd((2, 4)), RunningMeanStd((2, 4))
    combined.update(samples)
    split.update(samples[:2])
    split.update(samples[2:])
    np.testing.assert_allclose(split.mean, combined.mean)
    np.testing.assert_allclose(split.var, combined.var)
    np.testing.assert_allclose(combined.mean, samples.mean(axis=0), rtol=2e-5)
    assert split.count == combined.count
    normalized = rnd.normalize_rnd_observations(np.full((1, 2, 4), 255, dtype=np.uint8), split)
    assert normalized.dtype == np.float32
    np.testing.assert_array_equal(normalized, 5.0)


@pytest.mark.parametrize("rgb", (False, True))
def test_rnd_uses_newest_frame(rgb: bool) -> None:
    shape = (2, 4, 16, 16, 3) if rgb else (2, 4, 16, 16)
    obs = np.zeros(shape, dtype=np.uint8)
    obs[:, -1] = 20
    frames = rnd.rnd_frames(obs)
    assert frames.shape == (2, 16, 16, 3 if rgb else 1)
    np.testing.assert_array_equal(frames, 20)


def test_intrinsic_reward_scaling_preserves_filter_across_rollouts() -> None:
    moments = RunningMeanStd()
    rewards = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    normalized, discounted = rnd.normalize_intrinsic_rewards(rewards, np.zeros(2), moments, 0.5)
    np.testing.assert_allclose(discounted, [3.5, 5.0])
    np.testing.assert_allclose(normalized, rewards / np.sqrt(moments.var + 1e-8))
    _, continued = rnd.normalize_intrinsic_rewards(np.ones((1, 2), dtype=np.float32), discounted, moments, 0.5)
    np.testing.assert_allclose(continued, [2.75, 3.5])


def test_rnd_learns_without_changing_target() -> None:
    obs = jax.random.normal(jax.random.key(1), (4, 16, 16, 1))
    states = []
    for predictor, seed in ((True, 2), (False, 3)):
        model = rnd.RNDNetwork(predictor=predictor, channels=(4, 4, 4), feature_size=8)
        states.append(
            TrainState.create(
                apply_fn=model.apply, params=model.init(jax.random.key(seed), obs)["params"], tx=optax.adam(1e-3)
            )
        )
    predictor, target = states
    target_before = jax.tree.map(np.array, target)
    reward_before = rnd.rnd_reward(predictor, target, obs)
    expected = target.apply_fn({"params": target.params}, obs)
    predicted = predictor.apply_fn({"params": predictor.params}, obs)
    np.testing.assert_allclose(reward_before, jnp.square(predicted - expected).mean(-1), rtol=1e-5)
    for i in range(30):
        predictor, loss = rnd.update_rnd(predictor, target, obs, jax.random.key(i), 1.0)
        assert np.isfinite(loss)
    assert float(rnd.rnd_reward(predictor, target, obs).mean()) < float(reward_before.mean()) * 0.9
    for before, after in zip(jax.tree.leaves(target_before), jax.tree.leaves(target)):
        np.testing.assert_array_equal(before, after)
    # Empty random subsets must not apply Adam momentum from previous updates.
    skipped, loss = rnd.update_rnd(predictor, target, obs, jax.random.key(0), 0.0)
    assert float(loss) == 0
    for before, after in zip(jax.tree.leaves(predictor), jax.tree.leaves(skipped)):
        np.testing.assert_array_equal(before, after)


def test_two_value_outputs_recurrence_and_evaluation() -> None:
    model = rnd.ActorCritic(3, 8, encoder_channels=(4,), embedding_size=8)
    obs = jax.random.randint(jax.random.key(0), (3, 2, 4, 16, 16), 0, 256, dtype=jnp.uint8)
    starts = jnp.array([[True, True], [False, False], [True, False]])
    carry = rnd.initial_carry(2, 8)
    params = model.init(jax.random.key(1), obs, carry, starts)["params"]
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.sgd(0.01))
    final, logits, values = model.apply({"params": params}, obs, carry, starts)
    assert values.shape == (3, 2, 2)
    stepped = []
    for t in range(3):
        carry, _, prediction = model.apply({"params": params}, obs[t : t + 1], carry, starts[t : t + 1])
        stepped.append(prediction[0])
    np.testing.assert_allclose(jnp.stack(stepped), values, atol=5e-6)
    np.testing.assert_allclose(carry, final, atol=5e-6)
    # A game reset clears both critics' history for the resetting environment.
    _, _, fresh = model.apply({"params": params}, obs[2:], rnd.initial_carry(2, 8), starts[2:])
    np.testing.assert_allclose(fresh[:, 0], values[2:, 0], atol=5e-6)
    action, _ = _action(state, obs[0, 0], rnd.initial_carry(1, 8), True, jax.random.key(3), greedy=True)
    assert int(action) == int(logits[0, 0].argmax())
    batch = (
        obs,
        jnp.zeros((3, 2), dtype=jnp.int32),
        rnd.action_log_prob(logits, jnp.zeros((3, 2), dtype=jnp.int32)),
        jnp.arange(6.0).reshape(3, 2),
        values + jnp.array([1.0, 2.0]),
        rnd.initial_carry(2, 8),
        starts,
    )
    updated, metrics = rnd.update(state, batch, config())
    assert float(metrics[1]) == pytest.approx(0.5, abs=1e-5)
    assert float(metrics[5]) == pytest.approx(2.0, abs=1e-5)
    assert int(updated.step) == 1
    assert np.all(np.asarray(updated.params["value_output"]["bias"]) > 0)
    # Large old/current policy mismatch rejects the entire PPO optimizer update.
    rejected, metrics = rnd.update(state, (*batch[:2], batch[2] - 2, *batch[3:]), config())
    assert float(metrics[3]) > config().target_kl
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(rejected)):
        np.testing.assert_array_equal(before, after)


class ShortImageEnv(gym.Env):
    """Two-step episodes with distinct reset/final frames and staggered end types."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(0, 255, (4, 16, 16), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(3)
        self.steps = 0
        self.terminate = True

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.terminate = bool(seed % 2)
        self.steps = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        done = self.steps == 2
        return (
            np.full(self.observation_space.shape, self.steps * 20, dtype=np.uint8),
            2.0,
            done and self.terminate,
            done and not self.terminate,
            {},
        )


def short_env(env_id: str, frame_stack: bool, atari_preprocessing: bool, observation_size: int | None) -> gym.Env:
    return ShortImageEnv()


@pytest.mark.parametrize("stop_ppo", (False, True))
@pytest.mark.parametrize("minibatches", (1, 2))
def test_training_resets_returns_and_predictor_independence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_ppo: bool, minibatches: int
) -> None:
    cfg = replace(
        config(),
        num_envs=2,
        num_steps=3,
        num_minibatches=minibatches,
        total_steps=12,
        update_epochs=1,
        lstm_hidden_size=8,
        encoder_channels=(4,),
        embedding_size=8,
        bf16=False,
        vector_env="sync",
        log_dir=str(tmp_path),
        video_every_episodes=0,
        eval_every_minutes=0,
        rnd_warmup_steps=4,
        rnd_update_fraction=1.0,
        rnd_update_epochs=2,
    )
    gae_calls: list[tuple[np.ndarray, np.ndarray]] = []
    returns: list[np.ndarray] = []
    advantages: list[np.ndarray] = []
    reward_inputs: list[np.ndarray] = []
    update_inputs: list[np.ndarray] = []
    predictor_steps: list[int] = []
    raw_frames: list[np.ndarray] = []
    original_gae, original_update = rnd.gae, rnd.update
    original_reward, original_update_rnd, original_frames = rnd.rnd_reward, rnd.update_rnd, rnd.rnd_frames

    def checked_frames(obs: rnd.Array) -> np.ndarray:
        frames = original_frames(obs)
        raw_frames.append(frames.copy())
        return frames

    def checked_gae(
        rewards: rnd.Array, dones: rnd.Array, values: rnd.Array, next_value: rnd.Array, gamma: float, gae_lambda: float
    ) -> tuple[jax.Array, jax.Array]:
        gae_calls.append((np.array(rewards), np.array(dones)))
        result = original_gae(rewards, dones, values, next_value, gamma, gae_lambda)
        advantages.append(np.asarray(result[0]))
        returns.append(np.asarray(result[1]))
        return result

    def checked_update(state: TrainState, batch: rnd.PPOBatch, config: rnd.Config) -> tuple[TrainState, rnd.PPOMetrics]:
        # Identify shuffled environment columns by the critic returns.
        expected = np.stack(returns[-2:], axis=-1)
        order = [int(np.argmin(np.abs(expected[0, :, 0] - batch[4][0, i, 0]))) for i in range(batch[4].shape[1])]
        np.testing.assert_allclose(batch[4], expected[:, order])
        combined = config.extrinsic_coef * advantages[-2] + config.intrinsic_coef * advantages[-1]
        np.testing.assert_allclose(batch[3], combined[:, order])
        if stop_ppo:
            batch = (*batch[:2], batch[2] - 2, *batch[3:])
        return original_update(state, batch, config)

    def checked_reward(predictor: TrainState, target: TrainState, obs: rnd.Array) -> jax.Array:
        reward_inputs.append(np.array(obs))
        return original_reward(predictor, target, obs)

    def checked_rnd_update(
        predictor: TrainState, target: TrainState, obs: rnd.Array, key: jax.Array, update_fraction: float
    ) -> tuple[TrainState, jax.Array]:
        update_inputs.append(np.array(obs))
        result = original_update_rnd(predictor, target, obs, key, update_fraction)
        predictor_steps.append(int(result[0].step))
        return result

    monkeypatch.setattr(rnd, "RNDNetwork", partial(rnd.RNDNetwork, channels=(4, 4, 4), feature_size=8))
    monkeypatch.setattr(rnd, "make_env", short_env)
    monkeypatch.setattr(rnd, "gae", checked_gae)
    monkeypatch.setattr(rnd, "update", checked_update)
    monkeypatch.setattr(rnd, "rnd_reward", checked_reward)
    monkeypatch.setattr(rnd, "update_rnd", checked_rnd_update)
    monkeypatch.setattr(rnd, "rnd_frames", checked_frames)
    state = rnd.train(cfg)
    assert int(state.step) == (0 if stop_ppo else 2 * minibatches)
    assert predictor_steps == list(range(1, 2 * cfg.rnd_update_epochs * minibatches + 1))
    assert len(gae_calls) == 4
    np.testing.assert_array_equal(gae_calls[0][1][:, 0], [False, True, False])
    np.testing.assert_array_equal(gae_calls[2][1][:, 0], [True, False, True])
    for i in (1, 3):
        np.testing.assert_array_equal(gae_calls[i][1], False)
        assert np.all(gae_calls[i][0] >= 0)
    # True termination uses only game reward; timeouts add the extrinsic bootstrap.
    assert gae_calls[0][0][1, 0] == 1.0
    assert gae_calls[0][0][1, 1] != 1.0
    for rollout in range(2):
        reward_obs = np.concatenate(reward_inputs[rollout * minibatches : (rollout + 1) * minibatches])
        for epoch in range(cfg.rnd_update_epochs):
            offset = (rollout * cfg.rnd_update_epochs + epoch) * minibatches
            update_obs = np.concatenate(update_inputs[offset : offset + minibatches])
            # Environment order can shuffle, but every epoch uses the same normalized images.
            np.testing.assert_array_equal(np.sort(reward_obs, axis=0), np.sort(update_obs, axis=0))
    # The final six frame extractions are training successors, including final screens.
    np.testing.assert_array_equal([frames[0, 0, 0, 0] for frames in raw_frames[-6:]], [20, 40, 20, 40, 20, 40])
    (run_dir,) = tmp_path.iterdir()
    events = EventAccumulator(str(run_dir)).Reload()
    for tag in ("losses/rnd_predictor", "losses/intrinsic_value", "rnd/reward_normalized_mean"):
        assert [event.step for event in events.Scalars(tag)] == [6, 12]
        assert all(np.isfinite(event.value) for event in events.Scalars(tag))
    assert [event.value for event in events.Scalars("charts/return_mean_100")] == [4.0, 4.0]


@pytest.mark.parametrize(
    ("name", "invalid"),
    [
        ("intrinsic_gamma", 1.0),
        ("intrinsic_gamma", np.nan),
        ("intrinsic_coef", -1.0),
        ("extrinsic_coef", np.inf),
        ("rnd_learning_rate", 0),
        ("rnd_update_fraction", 0),
        ("rnd_update_fraction", 1.1),
        ("rnd_warmup_steps", -1),
        ("rnd_warmup_steps", True),
        ("rnd_update_epochs", 0),
        ("rnd_update_epochs", 1.5),
        ("encoder_channels", ()),
        ("encoder_channels", (32, 0)),
        ("embedding_size", 0),
    ],
)
def test_invalid_rnd_config_fails_before_creating_environments(name: str, invalid: Any) -> None:
    with patch.object(rnd, "make_env") as create_env, pytest.raises(ValueError, match=name):
        rnd.train(replace(config(), **{name: invalid}))
    create_env.assert_not_called()


@pytest.mark.parametrize("mode", ("sync", "async"))
def test_montezuma_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    cfg = replace(
        config(),
        num_envs=1,
        num_steps=4,
        num_minibatches=1,
        total_steps=4,
        update_epochs=1,
        lstm_hidden_size=8,
        encoder_channels=(4,),
        embedding_size=8,
        bf16=False,
        vector_env=mode,
        log_dir=str(tmp_path),
        observation_size=16,
        video_every_episodes=0,
        eval_every_minutes=0,
        rnd_warmup_steps=2,
        rnd_update_fraction=1.0,
        rnd_update_epochs=1,
    )
    monkeypatch.setattr(rnd, "RNDNetwork", partial(rnd.RNDNetwork, channels=(4, 4, 4), feature_size=8))
    state = rnd.train(cfg)
    assert int(state.step) == 1
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(state.params))
