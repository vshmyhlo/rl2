from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_rnd_learns_without_changing_target(dtype: jax.typing.DTypeLike) -> None:
    obs = jax.random.normal(jax.random.key(1), (4, 16, 16, 1))
    states = []
    for predictor, seed in ((True, 2), (False, 3)):
        model = rnd.RNDNetwork(predictor=predictor, channels=(4, 4, 4), feature_size=8, dtype=dtype)
        states.append(
            TrainState.create(
                apply_fn=model.apply, params=model.init(jax.random.key(seed), obs)["params"], tx=optax.adam(1e-3)
            )
        )
        features, captured = model.apply(
            {"params": states[-1].params}, obs, capture_intermediates=True, mutable=["intermediates"]
        )
        chex.assert_shape(features, (4, 8))
        chex.assert_type(features, jnp.float32)
        for name, intermediate in captured["intermediates"].items():
            if name.startswith(("Conv_", "Dense_")):
                chex.assert_type(intermediate["__call__"][0], dtype)
    predictor, target = states
    target_before = jax.tree.map(np.array, target)
    reward_before = rnd.rnd_reward(predictor, target, obs)
    expected = target.apply_fn({"params": target.params}, obs)
    predicted = predictor.apply_fn({"params": predictor.params}, obs)
    np.testing.assert_allclose(reward_before, jnp.square(predicted - expected).mean(-1), rtol=1e-5)
    for i in range(3):
        predictor, loss = rnd.update_rnd(predictor, target, obs, jax.random.key(i), 1.0)
        assert np.isfinite(loss)
        chex.assert_type(loss, jnp.float32)
    chex.assert_type(jax.tree.leaves(predictor.params), jnp.float32)
    for leaf in jax.tree.leaves(predictor.opt_state):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            chex.assert_type(leaf, jnp.float32)
    assert float(rnd.rnd_reward(predictor, target, obs).mean()) < float(reward_before.mean())
    for before, after in zip(jax.tree.leaves(target_before), jax.tree.leaves(target)):
        np.testing.assert_array_equal(before, after)
    # Empty random subsets must not apply Adam momentum from previous updates.
    skipped, loss = rnd.update_rnd(predictor, target, obs, jax.random.key(0), 0.0)
    assert float(loss) == 0
    for before, after in zip(jax.tree.leaves(predictor), jax.tree.leaves(skipped)):
        np.testing.assert_array_equal(before, after)


def test_bf16_policy_heads_and_update() -> None:
    model = rnd.ActorCritic(3, 8, dtype=jnp.bfloat16, encoder_channels=(4,), embedding_size=8)
    obs = jax.random.randint(jax.random.key(0), (3, 2, 4, 16, 16), 0, 256, dtype=jnp.uint8)
    carry = rnd.initial_carry(2, 8)
    starts = jnp.zeros((3, 2), dtype=jnp.bool_)
    params = model.init(jax.random.key(1), obs, carry, starts)["params"]
    (final, logits, values), captured = jax.jit(
        partial(model.apply, capture_intermediates=True, mutable=["intermediates"])
    )({"params": params}, obs, carry, starts)
    for name in ("Conv_0", "policy_hidden", "value_hidden", "policy_output", "value_output"):
        chex.assert_type(captured["intermediates"][name]["__call__"][0], jnp.bfloat16)
    chex.assert_type(jax.tree.leaves((final, logits, values, params)), jnp.float32)
    chex.assert_shape(logits, (3, 2, 3))
    chex.assert_shape(values, (3, 2, 2))
    actions = jnp.zeros((3, 2), dtype=jnp.int32)
    batch = (
        obs,
        actions,
        rnd.action_log_prob(logits, actions),
        jnp.arange(6.0).reshape(3, 2),
        values + 1,
        carry,
        starts,
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    updated, metrics = rnd.update(state, batch, replace(config(), target_kl=None))
    assert int(updated.step) == 1
    for leaf in jax.tree.leaves((updated.params, updated.opt_state, metrics)):
        assert np.isfinite(leaf).all()
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            chex.assert_type(leaf, jnp.float32)
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(updated.params)))


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


def test_warmup_collects_terminal_frames_then_resets() -> None:
    envs = gym.vector.SyncVectorEnv([ShortImageEnv, ShortImageEnv], autoreset_mode=gym.vector.AutoresetMode.DISABLED)
    try:
        cfg = replace(config(), num_envs=2, rnd_warmup_steps=2)
        moments = RunningMeanStd((16, 16, 1))
        observation = rnd.warmup_rnd_observations(envs, cfg, moments)
        np.testing.assert_array_equal(observation, 0)
        # Reset, live, and terminal frames are all included before resetting.
        expected = RunningMeanStd((16, 16, 1))
        expected.update(np.broadcast_to(np.array([0, 0, 20, 20, 40, 40])[:, None, None, None], (6, 16, 16, 1)))
        np.testing.assert_allclose(moments.mean, expected.mean)
        np.testing.assert_allclose(moments.var, expected.var)
        assert moments.count == expected.count
    finally:
        envs.close()


def test_gae_distinguishes_episodic_and_continuing_returns() -> None:
    rewards = jnp.array([[1.0], [2.0]])
    values = jnp.zeros_like(rewards)
    next_value = jnp.array([4.0])
    _, episodic = rnd.gae(rewards, jnp.array([[True], [False]]), values, next_value, 0.5, 1.0)
    _, continuing = rnd.gae(rewards, jnp.zeros((2, 1), dtype=jnp.bool_), values, next_value, 0.5, 1.0)
    np.testing.assert_allclose(episodic, [[1.0], [4.0]])
    np.testing.assert_allclose(continuing, [[3.0], [4.0]])
