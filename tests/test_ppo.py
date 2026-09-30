import contextlib
import io
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import cv2
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from flax.training.train_state import TrainState
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rl2 import ppo
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


class ShortEpisodes(gym.wrappers.TimeLimit):
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ppo.Array, dict[str, Any]]:
        # Seeded limits work in spawned workers and stagger resets across rollouts.
        if seed is not None:
            self._max_episode_steps = 3 if self.render_mode else 2 + 3 * ((seed - 1) % 2)
        return super().reset(seed=seed, options=options)


def short_env(
    env_id: str,
    render_mode: str | None = None,
    frame_stack: bool = False,
    atari_preprocessing: bool = False,
    observation_size: int | None = None,
) -> gym.Env:
    env = gym.wrappers.TransformReward(
        make_env(env_id, render_mode, frame_stack, atari_preprocessing, observation_size),
        lambda reward: 2.0,
    )
    return ShortEpisodes(env, max_episode_steps=3)


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
    model = ActorCritic(6, config.lstm_hidden_size)
    # This architecture budget is specifically for 84x84 RGB, independent of the training config.
    obs = jax.ShapeDtypeStruct((1, 1, 1, 84, 84, 3), jnp.uint8)
    carry = initial_carry(1, config.lstm_hidden_size)
    starts = jnp.ones((1, 1), dtype=bool)
    variables = jax.eval_shape(model.init, jax.random.key(0), obs, carry, starts)
    count = sum(parameter.size for parameter in jax.tree.leaves(variables["params"]))
    assert count > 48000000
    assert count < 52000000
    final, logits, values = jax.eval_shape(model.apply, variables, obs, carry, starts)
    assert logits.shape == (1, 1, 6)
    assert values.shape == (1, 1)
    assert final[0].shape == (1, config.lstm_hidden_size)


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


def test_bf16_recurrent_training_keeps_float32_state_and_losses() -> None:
    config = replace(load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"), target_kl=None)
    assert config.bf16
    model = ActorCritic(3, 8, dtype=jnp.bfloat16, encoder_channels=(8, 16, 16, 16), embedding_size=32)
    obs = jax.random.randint(jax.random.key(2), (3, 2, 1, 84, 84), 0, 256, dtype=jnp.uint8)
    carry = initial_carry(2, 8)
    starts = jnp.array([[True, True], [False, False], [True, False]])
    params = model.init(jax.random.key(1), obs, carry, starts)["params"]
    (final, logits, values), captured = jax.jit(
        partial(model.apply, capture_intermediates=True, mutable=["intermediates"])
    )({"params": params}, obs, carry, starts)
    assert captured["intermediates"]["Conv_0"]["__call__"][0].dtype == jnp.bfloat16
    assert captured["intermediates"]["policy_hidden"]["__call__"][0].dtype == jnp.bfloat16
    for array in (*final, logits, values, *jax.tree.leaves(params)):
        assert array.dtype == jnp.float32
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(0.001))
    replayed_log_probs = []
    for t in range(3):
        actions, log_probs, prediction, carry = ppo.act(state, obs[t], carry, starts[t], jax.random.key(t))
        # BF16 kernels may round differently for sequence and single-step batches.
        np.testing.assert_allclose(prediction, values[t], atol=0.02, rtol=0.02)
        np.testing.assert_allclose(log_probs, action_log_prob(logits[t], actions), atol=1e-3, rtol=0)
        replayed_log_probs.append(action_log_prob(logits[t], jnp.zeros(2, dtype=jnp.int32)))
    np.testing.assert_allclose(carry, final, atol=0.01, rtol=0.02)
    batch = (
        obs,
        jnp.zeros((3, 2), dtype=jnp.int32),
        jnp.stack(replayed_log_probs),
        jnp.arange(6, dtype=jnp.float32).reshape(3, 2),
        jnp.ones((3, 2)),
        initial_carry(2, 8),
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


def test_recurrent_sequences_match_steps_and_reset_only_finished_env() -> None:
    model = ActorCritic(3, 16, encoder_channels=(8, 16, 16, 16), embedding_size=32)
    obs = jax.random.randint(jax.random.key(2), (4, 2, 4, 84, 84), 0, 256, dtype=jnp.uint8)
    carry = initial_carry(2, 16)
    starts = jnp.array([[True, True], [False, False], [True, False], [False, False]])
    params = model.init(jax.random.key(1), obs, carry, starts)
    apply = jax.jit(model.apply)
    final, logits, values = apply(params, obs, carry, starts)
    stepped_logits, stepped_values = [], []
    for t in range(4):
        carry, policy, critic = apply(params, obs[t : t + 1], carry, starts[t : t + 1])
        stepped_logits.append(policy[0])
        stepped_values.append(critic[0])
    np.testing.assert_allclose(logits, jnp.stack(stepped_logits), atol=5e-6)
    np.testing.assert_allclose(values, jnp.stack(stepped_values), atol=5e-6)
    np.testing.assert_allclose(final, carry, atol=5e-6)
    # Splitting a rollout preserves memory; bootstrapping must not consume it.
    prefix_carry, _, _ = apply(params, obs[:2], initial_carry(2, 16), starts[:2])
    state = TrainState.create(apply_fn=model.apply, params=params["params"], tx=optax.sgd(0.0))
    peek = value(state, obs[2], prefix_carry, starts[2])
    np.testing.assert_allclose(peek, values[2], atol=5e-6)
    _, suffix_logits, suffix_values = apply(params, obs[2:], prefix_carry, starts[2:])
    np.testing.assert_allclose(suffix_values, values[2:], atol=5e-6)
    # A reset discards history for env 0; env 1 still depends on it.
    _, fresh_logits, fresh_values = apply(params, obs[2:], initial_carry(2, 16), starts[2:])
    np.testing.assert_allclose(suffix_logits[:, 0], fresh_logits[:, 0], atol=5e-6)
    np.testing.assert_allclose(suffix_values[:, 0], fresh_values[:, 0], atol=5e-6)
    assert float(jnp.max(jnp.abs(suffix_values[:, 1] - fresh_values[:, 1]))) > 1e-05


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


@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("frame_stack", (False, True))
@pytest.mark.parametrize("preprocessing", (False, True))
def test_atari_training_across_resets(mode: str, frame_stack: bool, preprocessing: bool) -> None:
    check_atari_training(mode, frame_stack=frame_stack, preprocessing=preprocessing)


def test_early_stop_resumes_next_rollout_and_anneals_lr() -> None:
    check_atari_training("sync", target_kl=1e-8, update_epochs=3)


def check_atari_training(
    mode: str,
    target_kl: float | None = None,
    update_epochs: int = 1,
    frame_stack: bool = False,
    preprocessing: bool = False,
) -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        lstm_hidden_size=16,
        bf16=False,
        total_steps=16,
        num_envs=2,
        num_steps=4,
        num_minibatches=2,
        update_epochs=update_epochs,
        video_every_episodes=2,
        eval_every_minutes=0,
        observation_size=84,
        vector_env=mode,
        target_kl=target_kl,
        frame_stack=frame_stack,
        atari_preprocessing=preprocessing,
    )
    output = io.StringIO()
    training_steps = 0
    checked_rollouts = set()
    previous_carry = initial_carry(config.num_envs, config.lstm_hidden_size)

    def checked_act(
        state: TrainState,
        obs: ppo.Array,
        carry: ppo.LSTMCarry,
        starts: ppo.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, ppo.LSTMCarry]:
        nonlocal training_steps, previous_carry
        image_shape = (84, 84) if preprocessing else (84, 84, 3)
        assert obs.shape[1:] == (4 if frame_stack else 1, *image_shape)
        if obs.shape[0] == config.num_envs:
            # Memory crosses rollout boundaries and video games cannot overwrite it.
            np.testing.assert_array_equal(carry, previous_carry)
            np.testing.assert_array_equal(starts, [training_steps % 2 == 0, training_steps % 5 == 0])
            result = ppo_act(state, obs, carry, starts, key)
            previous_carry = result[3]
            training_steps += 1
            return result
        return ppo_act(state, obs, carry, starts, key)

    def checked_update(state: TrainState, batch: ppo.PPOBatch, config: ppo.Config) -> tuple[TrainState, ppo.PPOMetrics]:
        assert batch[0].shape[:2] == (config.num_steps, config.num_envs // config.num_minibatches)
        if training_steps not in checked_rollouts:
            # Replay with unchanged parameters must recover rollout action probabilities.
            _, logits, _ = state.apply_fn({"params": state.params}, batch[0], batch[5], batch[6])
            np.testing.assert_allclose(action_log_prob(logits, batch[1]), batch[2], atol=5e-6)
            checked_rollouts.add(training_steps)
        return update(state, batch, config)

    ppo_act = ppo.act
    with TemporaryDirectory() as log_dir:
        with (
            patch("rl2.ppo.ActorCritic", new=partial(ActorCritic, encoder_channels=(8, 16, 16, 16), embedding_size=32)),
            patch("rl2.ppo.make_env", new=short_env),
            patch("rl2.ppo.act", new=checked_act),
            patch("rl2.ppo.update", new=checked_update),
            contextlib.redirect_stdout(output),
        ):
            state = train(replace(config, log_dir=log_dir))
        (run_dir,) = Path(log_dir).iterdir()
        events = EventAccumulator(str(run_dir)).Reload()
        for tag in (
            "losses/policy",
            "losses/value",
            "policy/entropy",
            "policy/approx_kl",
            "policy/clip_fraction",
            "value/explained_variance",
            "charts/learning_rate",
            "charts/updates_per_rollout",
            "policy/early_stop",
            "charts/steps_per_second",
            "charts/return_mean_100",
            "charts/episode_length_mean_100",
            "charts/total_episodes",
        ):
            scalars = events.Scalars(tag)
            assert [event.step for event in scalars] == [8, 16]
            assert all(np.isfinite(event.value) for event in scalars)
        np.testing.assert_allclose(
            [event.value for event in events.Scalars("charts/learning_rate")],
            [config.learning_rate, config.learning_rate / 2],
            rtol=1e-6,
        )
        assert [event.value for event in events.Scalars("charts/updates_per_rollout")] == (
            [1, 1] if target_kl is not None else [2, 2]
        )
        assert [event.value for event in events.Scalars("policy/early_stop")] == (
            [1, 1] if target_kl is not None else [0, 0]
        )
        assert [event.value for event in events.Scalars("charts/total_episodes")] == [2, 5]
        np.testing.assert_allclose(
            [event.value for event in events.Scalars("charts/episode_length_mean_100")],
            [2.0, 2.6],
        )
        np.testing.assert_allclose(
            [event.value for event in events.Scalars("charts/return_mean_100")],
            [4.0, 5.2],
        )
        config_text = events.Tensors("config/text_summary")[0].tensor_proto.string_val[0]
        assert f"env_id: {config.env_id}".encode() in config_text
        videos = events.Images("gameplay")
        assert [event.step for event in videos] == [8, 16]
        for video in videos:
            with Image.open(io.BytesIO(video.encoded_image_string)) as image:
                assert image.format == "GIF"
                assert image.size == (160, 210)
    assert int(state.step) == (2 if target_kl is not None else 4)
    assert checked_rollouts == {4, 8}
    for leaf in jax.tree.leaves(state.params):
        assert np.isfinite(leaf).all()
    assert "step=16" in output.getvalue()
    assert "step=16 episodes=5" in output.getvalue()
    assert "return=n/a" not in output.getvalue()
    assert "episode_length=2.6" in output.getvalue()
