from dataclasses import FrozenInstanceError, asdict, replace
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml
from flax import linen as nn
from flax.training.train_state import TrainState
from pydantic import ValidationError

from rl2 import ppo
from rl2.observation_encoder import ConvStage
from rl2.ppo import (
    ActorCritic,
    ResetLSTM,
    action_log_prob,
    explained_variance,
    gae,
    initial_carry,
    learning_rate_schedule,
    load_config,
    make_env,
    train,
    update,
    value,
)


def small_model_config(model_type: ppo.ModelType) -> ppo.ModelConfig:
    if model_type == "lstm":
        return ppo.LSTMConfig(hidden_size=8)
    return ppo.GDN2Config(hidden_size=8, num_heads=1, head_dim=4, intermediate_size=8)


@pytest.mark.parametrize("stacked", (False, True))
def test_resize_only_preserves_emulator_transitions(stacked: bool) -> None:
    raw = make_env("ALE/SpaceInvaders-v5", frame_stack=stacked)
    resized = make_env("ALE/SpaceInvaders-v5", frame_stack=stacked, observation_size=84)
    try:
        original, raw_info = raw.reset(seed=7)
        observation, resized_info = resized.reset(seed=7)
        assert raw_info == resized_info
        for action in (None, 0, 1, 2, 3, 0):
            if action is not None:
                original, *raw_transition = raw.step(action)
                observation, *resized_transition = resized.step(action)
                assert raw_transition == resized_transition
            expected = np.stack([cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) for frame in original])
            np.testing.assert_array_equal(observation, expected)
            assert observation.dtype == np.uint8
            assert resized.observation_space.contains(observation)
            assert raw.unwrapped.ale.getFrameNumber() == resized.unwrapped.ale.getFrameNumber()
    finally:
        raw.close()
        resized.close()


def test_default_model_parameter_budget_and_rgb_shapes() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")
    model = ActorCritic(6, config.model)
    # This architecture budget is specifically for 84x84 RGB, independent of the training config.
    obs = jax.ShapeDtypeStruct((1, 1, 1, 84, 84, 3), jnp.uint8)
    carry = initial_carry(1, config.model.hidden_size)
    starts = jnp.ones((1, 1), dtype=bool)
    variables = jax.eval_shape(model.init, jax.random.key(0), obs, carry, starts)
    count = sum(parameter.size for parameter in jax.tree.leaves(variables["params"]))
    assert variables["params"]["encoder"]["Dense_0"]["kernel"].shape == (1024, 768)
    assert 21_000_000 < count < 24_000_000
    final, logits, values = jax.eval_shape(model.apply, variables, obs, carry, starts)
    assert logits.shape == (1, 1, 6)
    assert values.shape == (1, 1)
    assert final[0].shape == (1, config.model.hidden_size)
    step_obs = jax.ShapeDtypeStruct(obs.shape[1:], obs.dtype)
    step_final, step_logits, step_values = jax.eval_shape(
        partial(model.apply, method=model.step), variables, step_obs, carry, starts[0]
    )
    assert step_logits.shape == (1, 6)
    assert step_values.shape == (1,)
    assert step_final[0].shape == final[0].shape


@pytest.mark.parametrize("frame_limit", (101, 102, 103, 104))
def test_atari_timeout_observation_is_current(frame_limit: int) -> None:
    # End on each offset within action repeat, including before pooling starts.
    gym_make = gym.make

    def limited_env(*args: Any, **kwargs: Any) -> gym.Env:
        return gym_make(*args, **kwargs, max_num_frames_per_episode=frame_limit)

    with patch("rl2.ppo.gym.make", new=limited_env):
        env = make_env("ALE/Pong-v5", atari_preprocessing=True)
    try:
        env.env.noop_max = 0
        obs, _ = env.reset(seed=1)
        for _ in range(26):
            obs, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        assert truncated
        assert not terminated
        expected = cv2.resize(env.unwrapped.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(obs[-1], expected)
    finally:
        env.close()


def test_clipped_policy_loss_and_gradient_direction() -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        target_kl=None,
        entropy_coef=0.0,
        value_coef=0.0,
    )
    probabilities = np.array([0.25, 0.75, 0.75, 0.25], dtype=np.float32).reshape(2, 2)
    logits = jnp.log(jnp.stack((probabilities, 1 - probabilities), axis=-1))
    state = TrainState.create(
        apply_fn=lambda variables, obs, carry, starts: (carry, variables["params"]["logits"], jnp.zeros((2, 2))),
        params={"logits": logits},
        tx=optax.sgd(0.1),
    )
    advantages = np.array([[-3.0, -1.0], [1.0, 3.0]])
    batch = (
        jnp.zeros((2, 2, 1)),
        jnp.zeros((2, 2), dtype=jnp.int32),
        jnp.full((2, 2), np.log(0.5)),
        advantages,
        jnp.ones((2, 2)),
        initial_carry(2, 1),
        jnp.zeros((2, 2), dtype=bool),
    )
    updated, metrics = update(state, batch, config)
    normalized = advantages / advantages.std()
    ratio = probabilities / 0.5
    expected = -np.minimum(
        ratio * normalized, np.clip(ratio, 1 - config.clip_coef, 1 + config.clip_coef) * normalized
    ).mean()
    assert float(metrics[0]) == pytest.approx(expected, rel=0, abs=5e-07)
    assert float(metrics[1]) == pytest.approx(0.5, rel=0, abs=5e-07)
    np.testing.assert_array_equal(updated.params["logits"][:, 0], logits[:, 0])  # Both clipped signs.
    assert float(updated.params["logits"][0, 1, 0]) < float(logits[0, 1, 0])
    assert float(updated.params["logits"][1, 1, 0]) > float(logits[1, 1, 0])


@pytest.mark.parametrize("model_type", ("lstm", "gdn2"))
def test_bf16_recurrent_training_keeps_float32_state_and_losses(model_type: ppo.ModelType) -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        target_kl=None,
        model=small_model_config(model_type),
    )
    assert config.bf16
    model = ActorCritic(
        3,
        small_model_config(model_type),
        dtype=jnp.bfloat16,
        encoder_stages=(ConvStage(4, blocks=1),),
        embedding_size=8,
    )
    obs = jax.random.randint(jax.random.key(2), (3, 2, 1, 8, 8), 0, 256, dtype=jnp.uint8)
    carry = model.initial_carry(2)
    starts = jnp.array([[True, True], [False, False], [True, False]])
    params = model.init(jax.random.key(1), obs, carry, starts)["params"]
    (final, logits, values), captured = jax.jit(
        partial(model.apply, capture_intermediates=True, mutable=["intermediates"])
    )({"params": params}, obs, carry, starts)
    assert captured["intermediates"]["encoder"]["stem"]["__call__"][0].dtype == jnp.bfloat16
    assert captured["intermediates"]["policy_hidden"]["__call__"][0].dtype == jnp.bfloat16
    assert captured["intermediates"]["policy_output"]["__call__"][0].dtype == jnp.bfloat16
    assert captured["intermediates"]["value_output"]["__call__"][0].dtype == jnp.bfloat16
    for array in (*jax.tree.leaves(final), logits, values, *jax.tree.leaves(params)):
        assert array.dtype == jnp.float32
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(0.001))
    replayed_log_probs = []
    for t in range(3):
        actions, log_probs, prediction, carry = ppo.act(state, obs[t], carry, starts[t], jax.random.key(t))
        # BF16 kernels may round differently for sequence and single-step batches.
        np.testing.assert_allclose(prediction, values[t], atol=0.02, rtol=0.02)
        np.testing.assert_allclose(log_probs, action_log_prob(logits[t], actions), atol=1e-3, rtol=0)
        replayed_log_probs.append(action_log_prob(logits[t], jnp.zeros(2, dtype=jnp.int32)))
    # Float32 carry still accumulates differences from the BF16 encoder.
    for actual, expected in zip(jax.tree.leaves(carry), jax.tree.leaves(final), strict=True):
        np.testing.assert_allclose(actual, expected, atol=0.03, rtol=0.02)
    batch = (
        obs,
        jnp.zeros((3, 2), dtype=jnp.int32),
        jnp.stack(replayed_log_probs),
        jnp.arange(6, dtype=jnp.float32).reshape(3, 2),
        jnp.ones((3, 2)),
        model.initial_carry(2),
        starts,
    )
    updated, metrics = update(state, batch, config)
    assert int(updated.step) == 1
    for array in (*metrics, *jax.tree.leaves((updated.params, updated.opt_state))):
        assert np.isfinite(array).all()
        if jnp.issubdtype(array.dtype, jnp.floating):
            assert array.dtype == jnp.float32
    assert any(
        (
            not np.array_equal(before, after)
            for before, after in zip(jax.tree.leaves(params), jax.tree.leaves(updated.params))
        )
    )


@pytest.mark.parametrize("model_type", ("lstm", "gdn2"))
@jax.default_matmul_precision("highest")
def test_recurrent_sequences_match_steps_and_reset_only_finished_env(model_type: ppo.ModelType) -> None:
    # Test sequence/reset semantics in float32, without GPU TF32 approximation.
    model = ActorCritic(
        3,
        small_model_config(model_type),
        encoder_stages=(ConvStage(4, blocks=1),),
        embedding_size=8,
    )
    obs = jax.random.randint(jax.random.key(2), (4, 2, 4, 8, 8), 0, 256, dtype=jnp.uint8)
    carry = model.initial_carry(2)
    starts = jnp.array([[True, True], [False, False], [True, False], [False, False]])
    params = model.init(jax.random.key(1), obs, carry, starts)
    apply = jax.jit(model.apply)
    step = jax.jit(partial(model.apply, method=model.step))
    final, logits, values = apply(params, obs, carry, starts)
    stepped_logits, stepped_values = [], []
    for t in range(4):
        carry, policy, critic = step(params, obs[t], carry, starts[t])
        stepped_logits.append(policy)
        stepped_values.append(critic)
    np.testing.assert_allclose(logits, jnp.stack(stepped_logits), atol=5e-6)
    np.testing.assert_allclose(values, jnp.stack(stepped_values), atol=5e-6)
    step_params = model.init(jax.random.key(1), obs[0], model.initial_carry(2), starts[0], method=model.step)
    for actual, expected in zip(jax.tree.leaves(step_params), jax.tree.leaves(params), strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(jax.tree.leaves(carry), jax.tree.leaves(final), strict=True):
        np.testing.assert_allclose(actual, expected, atol=5e-6)
    # Splitting a rollout preserves memory; bootstrapping must not consume it.
    prefix_carry, _, _ = apply(params, obs[:2], model.initial_carry(2), starts[:2])
    state = TrainState.create(apply_fn=model.apply, params=params["params"], tx=optax.sgd(0.0))
    peek = value(state, obs[2], prefix_carry, starts[2])
    np.testing.assert_allclose(peek, values[2], atol=5e-6)
    _, suffix_logits, suffix_values = apply(params, obs[2:], prefix_carry, starts[2:])
    np.testing.assert_allclose(suffix_values, values[2:], atol=5e-6)
    # A reset discards history for env 0; env 1 still depends on it.
    _, fresh_logits, fresh_values = apply(params, obs[2:], model.initial_carry(2), starts[2:])
    np.testing.assert_allclose(suffix_logits[:, 0], fresh_logits[:, 0], atol=5e-6)
    np.testing.assert_allclose(suffix_values[:, 0], fresh_values[:, 0], atol=5e-6)
    assert float(jnp.max(jnp.abs(suffix_values[:, 1] - fresh_values[:, 1]))) > 1e-05

    selected = jax.tree.map(lambda leaf: leaf[jnp.array([1])], prefix_carry)
    selected_final, selected_logits, selected_values = apply(params, obs[2:, 1:], selected, starts[2:, 1:])
    np.testing.assert_allclose(selected_logits, suffix_logits[:, 1:], atol=5e-6)
    np.testing.assert_allclose(selected_values, suffix_values[:, 1:], atol=5e-6)
    for actual, expected in zip(jax.tree.leaves(selected_final), jax.tree.leaves(final), strict=True):
        np.testing.assert_allclose(actual, expected[1:], atol=5e-6)


def test_lstm_gradients_follow_history_but_stop_at_episode_reset() -> None:
    cell = nn.scan(ResetLSTM, variable_broadcast="params", split_rngs={"params": False}, in_axes=0, out_axes=0)(8)
    inputs = jax.random.normal(jax.random.key(2), (4, 2, 5))
    starts = jnp.zeros((4, 2), dtype=bool).at[2, 0].set(True)
    carry = initial_carry(2, 8)
    params = cell.init(jax.random.key(1), carry, (inputs, starts))
    grads = jax.grad(lambda x: cell.apply(params, carry, (x, starts))[1][-1].sum())(inputs)
    np.testing.assert_array_equal(grads[:2, 0], 0)
    assert float(jnp.linalg.norm(grads[:2, 1])) > 0
    assert float(jnp.linalg.norm(grads[2:, 0])) > 0


def test_recurrent_minibatches_require_whole_environments() -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_envs=3,
        num_steps=4,
        num_minibatches=2,
    )
    with pytest.raises(ValueError, match="num_envs must be divisible"):
        train(config)


def test_learning_rate_schedule() -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_envs=2,
        num_steps=4,
        total_steps=19,
        num_minibatches=2,
        update_epochs=3,
        anneal_lr=True,
    )
    schedule = learning_rate_schedule(config)
    # Advance by rollouts, independently of how many optimizer updates were applied.
    expected = config.learning_rate * np.array([1.0, 0.5, 0.0, 0.0])
    np.testing.assert_allclose([schedule(i) for i in range(4)], expected, rtol=1e-6)
    constant = learning_rate_schedule(replace(config, anneal_lr=False))
    np.testing.assert_allclose([constant(i) for i in range(14)], config.learning_rate, rtol=1e-6)
    single = learning_rate_schedule(replace(config, total_steps=8))
    assert float(single(0)) == pytest.approx(config.learning_rate, rel=0, abs=5e-08)
    assert float(single(1)) == 0.0


def test_kl_rejects_update_without_changing_optimizer() -> None:
    config = replace(load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"), target_kl=0.01)
    state = TrainState.create(
        apply_fn=lambda variables, obs, carry, starts: (carry, variables["params"]["logits"], jnp.zeros(2)),
        params={"logits": jnp.log(jnp.array([[0.9, 0.1], [0.1, 0.9]]))},
        tx=optax.adam(0.01),
    )
    batch = (
        jnp.zeros((2, 1)),
        jnp.zeros(2, dtype=jnp.int32),
        jnp.full(2, np.log(0.5)),
        jnp.array([1.0, -1.0]),
        jnp.zeros(2),
        initial_carry(2, 1),
        jnp.zeros(2, dtype=bool),
    )
    stopped, metrics = update(state, batch, config)
    assert float(metrics[3]) > config.target_kl
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(stopped)):
        np.testing.assert_array_equal(before, after)
    continued, _ = update(state, batch, replace(config, target_kl=None))
    assert int(continued.step) == 1
    assert not np.array_equal(state.params["logits"], continued.params["logits"])


def test_policy_diagnostics() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")
    probabilities = jnp.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2], [0.52, 0.48]])
    state = TrainState.create(
        apply_fn=lambda variables, obs, carry, starts: (carry, variables["params"]["logits"], jnp.zeros(4)),
        params={"logits": jnp.log(probabilities)},
        tx=optax.sgd(0.0),
    )
    batch = (
        jnp.zeros((4, 1)),
        jnp.zeros(4, dtype=jnp.int32),
        jnp.full(4, np.log(0.5)),
        jnp.arange(4.0),
        jnp.arange(4.0),
        initial_carry(4, 1),
        jnp.zeros(4, dtype=bool),
    )
    _, metrics = update(state, batch, config)
    ratios = np.array([0.4, 1.0, 1.6, 1.04])
    assert float(metrics[3]) == pytest.approx(float(np.mean(ratios - 1 - np.log(ratios))), rel=0, abs=5e-07)
    assert float(metrics[4]) == 0.5
    same_policy_batch = (*batch[:2], jnp.log(probabilities[:, 0]), *batch[3:])
    _, metrics = update(state, same_policy_batch, config)
    assert float(metrics[3]) == pytest.approx(0.0, rel=0, abs=5e-07)
    assert float(metrics[4]) == 0.0


def test_explained_variance() -> None:
    returns = np.array([0.0, 1.0, 2.0])
    assert explained_variance(returns, returns) == 1.0
    assert explained_variance(np.zeros(3), returns) == 0.0
    assert explained_variance(-returns, returns) == -3.0
    assert np.isnan(explained_variance(returns, np.ones(3)))


def test_gae_stops_at_game_over() -> None:
    advantages, returns = gae(
        np.array([[1.0], [2.0]], dtype=np.float32),
        np.array([[True], [False]]),
        np.array([[0.5], [1.0]], dtype=np.float32),
        np.array([3.0], dtype=np.float32),
        0.9,
        0.8,
    )
    np.testing.assert_allclose(advantages, [[0.5], [3.7]], rtol=1e-6)
    np.testing.assert_allclose(returns, [[1.0], [4.7]], rtol=1e-6)


def test_gae_timeout_bootstrap_and_trace() -> None:
    # The final reward contains gamma * V(final observation).
    advantages, returns = gae(
        np.array([[1.0], [2.0 + 0.9 * 3.0]], dtype=np.float32),
        np.array([[False], [True]]),
        np.array([[0.5], [1.0]], dtype=np.float32),
        np.array([100.0], dtype=np.float32),
        0.9,
        0.8,
    )
    np.testing.assert_allclose(advantages, [[4.064], [3.7]], rtol=1e-6)
    np.testing.assert_allclose(returns, [[4.564], [4.7]], rtol=1e-6)


class SyntheticAtariEnv(gym.Env):
    """Expose a distinct screen on each raw step, including the final step."""

    def __init__(self, end_step: int, terminated: bool) -> None:
        self.observation_space = gym.spaces.Box(0, 255, (210, 160, 3), np.uint8)
        self.action_space = gym.spaces.Discrete(18)
        self._frameskip = 1
        self.ale = self
        self.end_step = end_step
        self.terminated = terminated
        self.steps = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(self.observation_space.shape, np.uint8), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.steps += 1
        done = self.steps >= self.end_step
        return (
            np.full(self.observation_space.shape, self.steps, np.uint8),
            0.0,
            done and self.terminated,
            done and not self.terminated,
            {},
        )

    def lives(self) -> int:
        return 1

    def getScreenGrayscale(self, output: np.ndarray) -> None:
        output.fill(self.steps)

    def getScreenRGB(self, output: np.ndarray) -> None:
        output.fill(self.steps)


# Every action-repeat offset matters; color and end flags do not require a product.
@pytest.mark.parametrize(
    "end_step,terminated,grayscale", [(1, False, False), (2, True, True), (3, False, True), (4, True, False)]
)
def test_preprocessing_returns_final_screen(end_step: int, terminated: bool, grayscale: bool) -> None:
    with ppo.AtariPreprocessing(SyntheticAtariEnv(end_step, terminated), noop_max=0, grayscale_obs=grayscale) as env:
        env.reset(seed=0)
        obs, _, actual_terminated, actual_truncated, _ = env.step(0)
        assert actual_terminated == terminated
        assert actual_truncated == (not terminated)
        np.testing.assert_array_equal(obs, np.full(obs.shape, end_step, np.uint8))


@pytest.mark.parametrize("frame_budget, model_type", ((1, "lstm"), (5, "lstm"), (5, "gdn2")))
def test_video_recording_stops_at_frame_budget(frame_budget: int, model_type: ppo.ModelType) -> None:
    from unittest.mock import Mock

    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        video_max_frames=frame_budget,
        model=small_model_config(model_type),
    )
    env = Mock()
    env.metadata = {"render_fps": 60}
    env.reset.return_value = (np.zeros((4, 84, 84), np.uint8), {})
    env.step.return_value = (np.zeros((4, 84, 84), np.uint8), 0.0, False, False, {})
    env.render.return_value = np.zeros((210, 160, 3), np.uint8)
    writer = Mock()
    carry = ppo.initial_model_carry(config, 1)
    with (
        patch("rl2.ppo.make_env", return_value=env),
        patch("rl2.ppo.act", return_value=(np.array([0]), np.array([0.0]), np.array([0.0]), carry)) as act,
    ):
        ppo.log_video(Mock(), config, writer, episode=1, steps=128)
    if frame_budget > 1:
        actual = act.call_args_list[0].args[2]
        assert jax.tree.structure(actual) == jax.tree.structure(carry)
        for leaf in jax.tree.leaves(actual):
            np.testing.assert_array_equal(leaf, 0)
    assert env.step.call_count == frame_budget - 1
    video = writer.add_video.call_args.args[1]
    assert video.shape == (1, frame_budget, 3, 210, 160)
    assert writer.add_video.call_args.kwargs["fps"] == 60
    env.close.assert_called_once()


@pytest.mark.parametrize("frame_budget", (0, -1, 1.5, True))
def test_invalid_video_frame_budget_fails_before_environment_creation(frame_budget: Any) -> None:
    with (
        patch("rl2.ppo.make_env") as create_env,
        pytest.raises(ValueError, match="video_max_frames"),
    ):
        config = replace(
            load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"), video_max_frames=frame_budget
        )
        train(config)
    create_env.assert_not_called()


@pytest.mark.parametrize("frame_budget,terminated", [(None, True), (10, False)])
def test_video_recording_stops_at_episode_end(frame_budget: int | None, terminated: bool) -> None:
    from unittest.mock import Mock

    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"), video_max_frames=frame_budget
    )
    env = Mock()
    env.metadata = {"render_fps": 60}
    obs = np.zeros((4, 84, 84), np.uint8)
    env.reset.return_value = (obs, {})
    env.step.side_effect = [(obs, 0.0, False, False, {})] * 6 + [(obs, 0.0, terminated, not terminated, {})]
    env.render.return_value = np.zeros((210, 160, 3), np.uint8)
    writer = Mock()
    carry = initial_carry(1, config.model.hidden_size)
    with (
        patch("rl2.ppo.make_env", return_value=env),
        patch("rl2.ppo.act", return_value=(np.array([0]), np.array([0.0]), np.array([0.0]), carry)),
    ):
        ppo.log_video(Mock(), config, writer, episode=1, steps=128)
    assert env.step.call_count == 7
    assert writer.add_video.call_args.args[1].shape == (1, 8, 3, 210, 160)
    env.close.assert_called_once()


@jax.default_matmul_precision("highest")
def test_gdn2_gradients_stop_at_episode_reset() -> None:
    config = ppo.GatedDeltaNet2Config(hidden_size=8, num_heads=1, head_dim=4)
    cell = nn.scan(ppo.ResetGDN2, variable_broadcast="params", split_rngs={"params": False}, in_axes=0, out_axes=0)(
        config, 2, 8
    )
    inputs = jax.random.normal(jax.random.key(2), (4, 2, 8))
    starts = jnp.zeros((4, 2), dtype=bool).at[2, 0].set(True)
    carry = ppo.GatedDeltaNet2Stack(config, 2, 8).initial_carry(2)
    params = cell.init(jax.random.key(1), carry, (inputs, starts))

    def loss(x: jax.Array) -> jax.Array:
        return cell.apply(params, carry, (x, starts))[1][-1].sum()

    grads = jax.jit(jax.grad(loss))(inputs)
    np.testing.assert_array_equal(grads[:2, 0], 0)
    assert float(jnp.linalg.norm(grads[:2, 1])) > 0
    assert float(jnp.linalg.norm(grads[2:, 0])) > 0


@pytest.mark.parametrize(
    "model_settings, message",
    [
        ({"type": "unknown"}, "union_tag_invalid"),
        ({"hidden_size": 8}, "union_tag_not_found"),
        ("lstm", "model"),
        ({"type": "lstm", "num_layers": 2}, "unexpected_keyword_argument"),
        ({"type": "gdn2", "lstm_hidden_size": 8}, "unexpected_keyword_argument"),
        ({"type": "lstm", "hidden_size": 0}, "greater_than"),
        ({"type": "lstm", "hidden_size": True}, "int_type"),
        ({"type": "gdn2", "hidden_size": 1.5}, "int_type"),
        *[
            ({"type": "gdn2", name: 0}, "greater_than")
            for name in ("hidden_size", "num_heads", "head_dim", "conv_size", "num_layers", "intermediate_size")
        ],
    ],
)
def test_invalid_model_config(model_settings: Any, message: str) -> None:
    settings = vars(load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")).copy()
    settings["model"] = model_settings
    with pytest.raises(ValidationError, match=message):
        ppo.Config(**settings)


@pytest.mark.parametrize("field", ("lstm_hidden_size", "gdn2_hidden_size"))
def test_flat_model_settings_are_rejected(field: str) -> None:
    settings = vars(load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")).copy()
    settings[field] = 8
    with pytest.raises(ValidationError, match=field):
        ppo.Config(**settings)


def test_model_config_loading_and_factory(tmp_path: Path) -> None:
    settings = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/ppo.yaml").read_text())
    settings.pop("model")
    path = tmp_path / "ppo.yaml"
    path.write_text(yaml.safe_dump(settings))
    default = load_config(path)
    assert ppo.make_model(default, 3).model == ppo.LSTMConfig()
    settings["model"] = {
        "type": "gdn2",
        "hidden_size": 8,
        "num_heads": 1,
        "head_dim": 4,
        "num_layers": 2,
        "intermediate_size": 8,
        "conv_size": 1,
    }
    path.write_text(yaml.safe_dump(settings))
    config = load_config(path)
    model = ppo.make_model(config, 3)
    assert isinstance(model.model, ppo.GDN2Config)
    assert model.dtype == jnp.bfloat16
    assert model.model.hidden_size == 8
    assert model.model.intermediate_size == 8
    assert asdict(config)["model"] == settings["model"]
    path.write_text(yaml.safe_dump(asdict(config)))
    round_trip = load_config(path)
    assert round_trip == config
    assert hash(round_trip) == hash(config)
    with pytest.raises(FrozenInstanceError):
        config.model.hidden_size = 16
    carry = ppo.initial_model_carry(config, 2)
    assert len(carry) == 2
    for layer in carry:
        assert layer.state.shape == (2, 1, 4, 4)
        assert layer.q.shape == layer.k.shape == layer.v.shape == (2, 0, 4)
        for leaf in layer:
            assert leaf.dtype == jnp.float32
            np.testing.assert_array_equal(leaf, 0)


@pytest.mark.parametrize("bad_input", ("observation_dtype", "reset_mask_shape"))
def test_step_rejects_invalid_array_metadata(bad_input: str) -> None:
    model = ActorCritic(3, ppo.LSTMConfig(hidden_size=8), encoder_stages=(ConvStage(4, blocks=1),), embedding_size=8)
    obs = jax.ShapeDtypeStruct((2, 1, 8, 8), jnp.float32 if bad_input == "observation_dtype" else jnp.uint8)
    starts = jnp.zeros((1, 2) if bad_input == "reset_mask_shape" else (2,), dtype=jnp.bool_)
    with pytest.raises(AssertionError):
        jax.eval_shape(partial(model.init, method=model.step), jax.random.key(0), obs, model.initial_carry(2), starts)
