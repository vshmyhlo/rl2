from dataclasses import FrozenInstanceError, asdict, replace
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import cv2
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml
from flax.training.train_state import TrainState
from pydantic import ValidationError

from rl2 import ppo
from rl2.observation_encoder import ConvStage
from rl2.ppo import (
    ActorCritic,
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
from rl2.shape_checker import ShapeChecker


def small_model_config(model_type: ppo.ModelType) -> ppo.ModelConfig:
    if model_type == "lstm":
        return ppo.LSTMConfig(hidden_size=8, num_layers=2, intermediate_size=8)
    if model_type == "gdn2":
        return ppo.GDN2Config(hidden_size=8, num_heads=1, head_dim=4, intermediate_size=8)
    return ppo.Mamba3Config(hidden_size=8, num_layers=2, intermediate_size=8, state_size=4, head_dim=4, mimo_rank=2)


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


def test_make_env_closes_emulator_when_wrapping_fails() -> None:
    env = Mock()
    with (
        patch.object(ppo.gym, "make", return_value=env),
        patch.object(ppo, "AtariPreprocessing", side_effect=ValueError("invalid preprocessing")),
        pytest.raises(ValueError, match="invalid preprocessing"),
    ):
        make_env("ALE/Pong-v5", atari_preprocessing=True)
    env.close.assert_called_once()


def test_default_model_parameter_budget_and_rgb_shapes() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")
    model = ActorCritic(6, config.model)
    # This architecture budget is specifically for 84x84 RGB, independent of the training config.
    obs = jax.ShapeDtypeStruct((1, 1, 1, 84, 84, 3), jnp.uint8)
    carry = model.initial_carry(1)
    starts = jnp.ones((1, 1), dtype=bool)
    variables = jax.eval_shape(model.init, jax.random.key(0), obs, carry, starts)
    count = sum(parameter.size for parameter in jax.tree.leaves(variables["params"]))
    assert variables["params"]["encoder"]["Dense_0"]["kernel"].shape == (1024, 768)
    assert "OptimizedLSTMCell_0" in variables["params"]["lstm"]["mixer_0"]
    assert 27_000_000 < count < 30_000_000
    final, logits, values = jax.eval_shape(model.apply, variables, obs, carry, starts)
    assert logits.shape == (1, 1, 6)
    assert values.shape == (1, 1)
    assert final[0][0].shape == (1, config.model.hidden_size)
    step_obs = jax.ShapeDtypeStruct(obs.shape[1:], obs.dtype)
    step_final, step_logits, step_values = jax.eval_shape(
        partial(model.apply, method=model.step), variables, step_obs, carry, starts[0]
    )
    assert step_logits.shape == (1, 6)
    assert step_values.shape == (1,)
    assert step_final[0][0].shape == final[0][0].shape


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


def test_single_transition_keeps_policy_gradient() -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        target_kl=None,
        entropy_coef=0.0,
        value_coef=0.0,
    )

    def apply(
        variables: optax.Params, obs: ppo.Array, carry: ppo.RecurrentCarry, starts: ppo.Array
    ) -> tuple[ppo.RecurrentCarry, jax.Array, jax.Array]:
        sc = ShapeChecker(T=1, B=1, F=1, H=1, W=1, A=2)
        sc.check(obs, "TBFHW", jnp.uint8)
        sc.check(starts, "TB", jnp.bool_)
        sc.check(variables["params"]["logits"], "TBA", jnp.float32)
        return carry, variables["params"]["logits"], jnp.zeros((1, 1))

    state = TrainState.create(apply_fn=apply, params={"logits": jnp.zeros((1, 1, 2))}, tx=optax.sgd(0.1))
    batch = (
        jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.uint8),
        jnp.zeros((1, 1), dtype=jnp.int32),
        jnp.full((1, 1), -np.log(2)),
        jnp.ones((1, 1)),
        jnp.zeros((1, 1)),
        initial_carry(1, 1),
        jnp.ones((1, 1), dtype=bool),
    )
    updated, metrics = update(state, batch, config)
    assert float(metrics[0]) == pytest.approx(-1.0)
    assert float(jax.nn.softmax(updated.params["logits"])[0, 0, 0]) > 0.5


@pytest.mark.parametrize(
    "field, invalid",
    [
        ("learning_rate", -0.1),
        ("learning_rate", float("nan")),
        ("gamma", -0.1),
        ("gamma", 1.1),
        ("gae_lambda", -0.1),
        ("gae_lambda", 1.1),
        ("clip_coef", -0.1),
        ("clip_coef", 1.0),
        ("target_kl", 0.0),
        ("target_kl", -0.1),
        ("target_kl", float("nan")),
        ("target_kl", float("inf")),
        ("entropy_coef", -0.1),
        ("value_coef", -0.1),
        ("max_grad_norm", 0.0),
        ("max_grad_norm", float("inf")),
    ],
)
def test_invalid_optimization_settings(field: str, invalid: float) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")
    with pytest.raises(ValueError, match=field):
        replace(config, **{field: invalid})


def test_optimization_settings_allow_disabled_terms_and_discount_boundaries() -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        learning_rate=0.0,
        gamma=0.0,
        gae_lambda=1.0,
        clip_coef=0.0,
        entropy_coef=0.0,
        value_coef=0.0,
    )
    replace(config, gamma=1.0, gae_lambda=0.0)


@pytest.mark.parametrize(
    "model_type,backend",
    [("lstm", "jax"), ("gdn2", "jax"), ("mamba3", "jax"), ("gdn2", "triton")],
)
def test_bf16_recurrent_training_keeps_float32_state_and_losses(
    model_type: ppo.ModelType, backend: ppo.GatedDeltaNet2Backend
) -> None:
    model_config = small_model_config(model_type)
    if backend == "triton":
        if not any("NVIDIA" in device.device_kind for device in jax.devices()):
            pytest.skip("requires NVIDIA GPU")
        pytest.importorskip("jax_triton")
        model_config = ppo.GDN2Config(
            hidden_size=8, num_layers=1, num_heads=1, head_dim=32, intermediate_size=8, backend=backend
        )
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        target_kl=None,
        model=model_config,
    )
    assert config.bf16
    model = ActorCritic(
        3,
        model_config,
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
    assert captured["intermediates"][f"{model_type}_input"]["__call__"][0].dtype == jnp.bfloat16
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


@pytest.mark.parametrize("model_type", ("lstm", "gdn2", "mamba3"))
@jax.default_matmul_precision("highest")
def test_recurrent_sequences_match_steps_and_reset_only_finished_env(model_type: ppo.ModelType) -> None:
    # Test sequence/reset semantics in float32, without GPU TF32 approximation.
    model = ActorCritic(
        3,
        small_model_config(model_type),
        encoder_stages=(ConvStage(4, blocks=1),),
        embedding_size=6,
    )
    obs = jax.random.randint(jax.random.key(2), (4, 2, 4, 8, 8), 0, 256, dtype=jnp.uint8)
    carry = model.initial_carry(2)
    starts = jnp.array([[True, True], [False, False], [True, False], [False, False]])
    params = model.init(jax.random.key(1), obs, carry, starts)
    encoded = model.apply(params, obs[0], method=model._encode)
    assert encoded.shape == (2, model.model.hidden_size)
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
    entropy = ppo.entropy_coef_schedule(config)
    assert float(entropy(0)) == config.entropy_coef
    assert float(entropy(14)) == config.entropy_coef


def test_cosine_schedules() -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_envs=2,
        num_steps=4,
        total_steps=35,  # Four full rollouts; the remaining steps are unused.
        num_minibatches=2,
        update_epochs=3,
        anneal_lr=True,
        lr_decay="cosine",
        entropy_decay="cosine",
    )
    # The quarter point distinguishes cosine from linear decay.
    expected = np.array([1.0, (1 + np.sqrt(0.5)) / 2, 0.5, 0.0, 0.0])
    for schedule_fn, initial in (
        (learning_rate_schedule, config.learning_rate),
        (ppo.entropy_coef_schedule, config.entropy_coef),
    ):
        schedule = schedule_fn(config)
        np.testing.assert_allclose([schedule(i) for i in (0, 1, 2, 4, 5)], initial * expected, rtol=1e-6)
        single = schedule_fn(replace(config, total_steps=8))
        assert float(single(0)) == pytest.approx(initial)
        assert float(single(1)) == 0.0
    constant = learning_rate_schedule(replace(config, anneal_lr=False))
    np.testing.assert_allclose([constant(i) for i in (0, 1, 4, 5)], config.learning_rate)


@pytest.mark.parametrize("field", ("lr_decay", "entropy_decay"))
def test_invalid_decay_config(field: str) -> None:
    settings = asdict(load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"))
    settings[field] = "unknown"
    with pytest.raises(ValidationError, match=field):
        ppo.Config(**settings)


def test_entropy_decay_scales_update_by_rollout() -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_envs=1,
        num_steps=1,
        total_steps=4,
        entropy_decay="cosine",
        target_kl=None,
    )

    def apply(
        variables: optax.Params, obs: ppo.Array, carry: ppo.RecurrentCarry, starts: ppo.Array
    ) -> tuple[ppo.RecurrentCarry, jax.Array, jax.Array]:
        return carry, variables["params"]["logits"], jnp.zeros_like(starts, dtype=jnp.float32)

    probabilities = np.array([[0.8, 0.2]], dtype=np.float32)
    logits = jnp.log(probabilities)
    state = TrainState.create(apply_fn=apply, params={"logits": logits}, tx=optax.sgd(1.0))
    state = state.replace(step=123)  # Optimizer step count must not drive decay.
    batch = (
        jnp.zeros((1, 1)),
        jnp.zeros(1, dtype=jnp.int32),
        logits[:, 0],
        jnp.zeros(1),
        jnp.zeros(1),
        initial_carry(1, 1),
        jnp.zeros(1, dtype=bool),
    )
    entropy = -(probabilities * np.log(probabilities)).sum()
    entropy_grad = -probabilities * (np.log(probabilities) + entropy)
    for iteration, factor in ((0, 1.0), (1, (1 + np.sqrt(0.5)) / 2), (4, 0.0)):
        updated, metrics = update(state, batch, config, iteration)
        np.testing.assert_allclose(
            updated.params["logits"], logits + config.entropy_coef * factor * entropy_grad, atol=1e-7
        )
        assert float(metrics[2]) == pytest.approx(entropy)
        assert int(updated.step) == 124


def test_train_logs_scheduled_coefficients_with_kl_stopping() -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=2,
        total_steps=2,
        vector_env="sync",
        anneal_lr=True,
        lr_decay="cosine",
        entropy_decay="cosine",
        target_kl=0.01,
        video_every_episodes=0,
        eval_every_minutes=0,
    )
    obs = np.zeros((1, 1), dtype=np.uint8)
    zeros = np.zeros(1, dtype=np.float32)
    dones = np.zeros(1, dtype=bool)
    carry = initial_carry(1, 1)
    envs = Mock()
    envs.single_action_space.n = 2
    envs.reset.return_value = (obs, {})
    envs.step.return_value = (obs, zeros, dones, dones, {})
    model = Mock()
    model.initial_carry.return_value = carry
    model.init.return_value = {"params": {"weight": jnp.zeros(1)}}
    writer = Mock()
    iterations: list[int] = []
    learning_rates: list[float] = []

    def reject_update(
        state: TrainState, batch: ppo.PPOBatch, config: ppo.Config, iteration: int
    ) -> tuple[TrainState, ppo.PPOMetrics]:
        iterations.append(iteration)
        learning_rates.append(float(state.opt_state.hyperparams["learning_rate"]))
        return state, tuple(jnp.asarray(v) for v in (0.0, 0.0, 0.5, 1.0, 0.0))

    with (
        patch("rl2.ppo.gym.vector.SyncVectorEnv", return_value=envs),
        patch("rl2.ppo.make_model", return_value=model),
        patch("rl2.ppo.SummaryWriter", return_value=writer),
        patch("rl2.ppo.act", return_value=(np.zeros(1, dtype=np.int32), zeros, zeros, carry)),
        patch("rl2.ppo.value", return_value=zeros),
        patch("rl2.ppo.update", side_effect=reject_update),
        patch("rl2.ppo.checkpoint_manager"),
        patch("rl2.ppo.restore_checkpoint", return_value=None),
        patch("rl2.ppo.save_checkpoint"),
    ):
        state = train(config)
    assert int(state.step) == 0
    assert iterations == [0, 1]
    np.testing.assert_allclose(learning_rates, config.learning_rate * np.array([1.0, 0.5]), rtol=1e-6)
    for tag, initial in (("charts/learning_rate", config.learning_rate), ("charts/entropy_coef", config.entropy_coef)):
        logged = [call.args[1:] for call in writer.add_scalar.call_args_list if call.args[0] == tag]
        np.testing.assert_allclose(logged, [(initial, 1), (initial / 2, 2)], rtol=1e-6)
    envs.close.assert_called_once()
    writer.close.assert_called_once()


@pytest.mark.parametrize(
    ("eval_every_minutes", "expected_steps"),
    [(0.0, []), (10.0, [4]), (1.0, [2, 4])],
    ids=["disabled", "final-before-timer", "periodic-and-final-without-duplicate"],
)
def test_train_evaluates_final_policy(
    eval_every_minutes: float, expected_steps: list[int], capsys: pytest.CaptureFixture[str]
) -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        env_id="ALE/Pong-v5",
        atari_preprocessing=True,
        num_envs=1,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        total_steps=5,  # Training ends after two complete rollouts, at step 4.
        vector_env="sync",
        video_every_episodes=0,
        eval_every_minutes=eval_every_minutes,
        target_kl=None,
    )
    obs = np.zeros((1, 1), dtype=np.uint8)
    zeros = np.zeros(1, dtype=np.float32)
    dones = np.ones(1, dtype=bool)
    carry = initial_carry(1, 1)
    envs = Mock()
    envs.single_action_space.n = 2
    envs.reset.return_value = (obs, {})
    envs.step.return_value = (obs, zeros, dones, ~dones, {})
    model = Mock()
    model.initial_carry.return_value = carry
    model.init.return_value = {"params": {"weight": jnp.zeros(1)}}
    clock = 0.0

    def advance_update(
        state: TrainState, batch: ppo.PPOBatch, config: ppo.Config, iteration: int
    ) -> tuple[TrainState, ppo.PPOMetrics]:
        nonlocal clock
        clock += 60.0
        return state.replace(step=iteration + 1), (jnp.asarray(0.0),) * 5

    def now() -> float:
        return clock

    def advance_evaluation(*args: object) -> None:
        nonlocal clock
        clock += 30.0

    with (
        patch.object(ppo.gym.vector, "SyncVectorEnv", return_value=envs),
        patch.object(ppo, "make_model", return_value=model),
        patch.object(ppo, "SummaryWriter") as writer,
        patch.object(ppo, "act", return_value=(np.zeros(1, dtype=np.int32), zeros, zeros, carry)),
        patch.object(ppo, "value", return_value=zeros),
        patch.object(ppo, "update", side_effect=advance_update),
        patch.object(ppo, "monotonic", side_effect=now),
        patch.object(ppo, "checkpoint_manager"),
        patch.object(ppo, "restore_checkpoint", return_value=None),
        patch.object(ppo, "save_checkpoint"),
        patch.object(ppo, "log_evaluation", side_effect=advance_evaluation) as evaluate,
    ):
        state = train(config)
    assert [call.args[4] for call in evaluate.call_args_list] == expected_steps
    for call in evaluate.call_args_list:
        evaluated_state, evaluated_config, evaluated_writer, episodes, steps = call.args
        assert int(evaluated_state.step) == steps // config.num_steps
        assert evaluated_config.eval_every_minutes == eval_every_minutes
        assert evaluated_writer is writer.return_value
        assert episodes == steps
    if expected_steps:
        assert evaluate.call_args.args[0] is state
    first_elapsed = 90.0 if 2 in expected_steps else 60.0
    final_elapsed = 120.0 + 30.0 * len(expected_steps)
    for tag, expected in (
        ("time/elapsed_seconds", [(first_elapsed, 2), (final_elapsed, 4)]),
        ("time/eta_seconds", [(first_elapsed, 2), (0.0, 4)]),
    ):
        logged = [call.args[1:] for call in writer.return_value.add_scalar.call_args_list if call.args[0] == tag]
        assert logged == expected
    output = capsys.readouterr().out
    assert f"elapsed=0:01:{int(first_elapsed % 60):02d} eta=0:01:{int(first_elapsed % 60):02d}" in output
    assert f"elapsed=0:{int(final_elapsed // 60):02d}:{int(final_elapsed % 60):02d} eta=0:00:00" in output
    envs.close.assert_called_once()
    writer.return_value.close.assert_called_once()


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
    # A stray singleton dimension would silently broadcast the loss to [B, B].
    with pytest.raises(AssertionError):
        update(state, (*batch[:2], batch[2][:, None], *batch[3:]), config)


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


def test_rollout_bootstraps_only_timeouts_before_partial_reset() -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_envs=4,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        total_steps=8,
        vector_env="sync",
        gamma=0.5,
        gae_lambda=1.0,
        target_kl=None,
        video_every_episodes=0,
        eval_every_minutes=0,
    )
    # Terminal, timeout, both flags, and continuing: each has distinct semantics.
    initial_obs = np.arange(1, 5, dtype=np.uint8)[:, None]
    final_obs = initial_obs + 10
    reset_obs = np.array([[21], [22], [23], [14]], dtype=np.uint8)
    next_obs = initial_obs + 30
    dones = np.array([True, True, True, False])
    envs = Mock()
    envs.single_action_space.n = 2
    envs.reset.side_effect = [(initial_obs, {}), (reset_obs, {})]
    envs.step.side_effect = [
        (
            final_obs,
            np.array([4.0, -2.0, 0.0, 4.0]),
            np.array([True, False, True, False]),
            np.array([False, True, True, False]),
            {},
        ),
        (next_obs, np.array([2.0, -2.0, 2.0, -2.0]), np.zeros(4, dtype=bool), np.zeros(4, dtype=bool), {}),
    ]
    carry = initial_carry(4, 1)
    advanced_carry = jax.tree.map(jnp.ones_like, carry)
    model = Mock()
    model.initial_carry.return_value = carry
    model.init.return_value = {"params": {"weight": jnp.zeros(1)}}

    def predict(state: TrainState, obs: ppo.Array, memory: ppo.RecurrentCarry, starts: ppo.Array) -> np.ndarray:
        sc = ShapeChecker(B=4, F=1)
        sc.check(obs, "BF", np.uint8)
        sc.check(starts, "B", np.bool_)
        prediction = np.asarray(obs[:, 0], dtype=np.float32)
        sc.check(prediction, "B", np.float32)
        return prediction

    def check_batch(
        state: TrainState, batch: ppo.PPOBatch, config: ppo.Config, iteration: int
    ) -> tuple[TrainState, ppo.PPOMetrics]:
        obs, _, _, advantages, returns, memory, starts = batch
        sc = ShapeChecker(T=2, B=4, F=1)
        sc.check(obs, "TBF", np.uint8)
        sc.check([advantages, returns], "TB", np.float32)
        sc.check(starts, "TB", np.bool_)
        # Minibatches permute whole environments; recover their original order.
        order = np.argsort(obs[0, :, 0])
        np.testing.assert_array_equal(obs[:, order], np.stack([initial_obs, reset_obs]))
        np.testing.assert_array_equal(starts[:, order], np.stack([np.ones(4, dtype=bool), dones]))
        expected = [[1.0, 5.0, 0.0, 9.0], [16.5, 15.0, 17.5, 16.0]]
        np.testing.assert_allclose(returns[:, order], expected)
        np.testing.assert_allclose(advantages[:, order], expected)
        for leaf in jax.tree.leaves(memory):
            np.testing.assert_array_equal(leaf, 0)
        return state, (jnp.asarray(0.0),) * 5

    with (
        patch.object(ppo.gym.vector, "SyncVectorEnv", return_value=envs) as vector_env,
        patch.object(ppo, "make_model", return_value=model),
        patch.object(ppo, "SummaryWriter") as writer,
        patch.object(
            ppo,
            "act",
            return_value=(
                np.zeros(4, dtype=np.int32),
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
                advanced_carry,
            ),
        ),
        patch.object(ppo, "value", side_effect=predict) as predict_value,
        patch.object(ppo, "update", side_effect=check_batch) as update_batch,
        patch.object(ppo, "checkpoint_manager"),
        patch.object(ppo, "restore_checkpoint", return_value=None),
        patch.object(ppo, "save_checkpoint"),
    ):
        train(config)
    assert vector_env.call_args.kwargs["autoreset_mode"] == gym.vector.AutoresetMode.DISABLED
    np.testing.assert_array_equal(envs.reset.call_args.kwargs["options"]["reset_mask"], dones)
    assert predict_value.call_count == 2
    timeout_call = predict_value.call_args_list[0]
    np.testing.assert_array_equal(timeout_call.args[1], final_obs)
    np.testing.assert_array_equal(timeout_call.args[3], False)
    for leaf in jax.tree.leaves(timeout_call.args[2]):
        np.testing.assert_array_equal(leaf, 1)
    update_batch.assert_called_once()
    scalars = {call.args[0]: call.args[1] for call in writer.return_value.add_scalar.call_args_list}
    assert scalars["charts/return_mean_100"] == pytest.approx(2 / 3)  # Raw rewards, without clipping or bootstrap.
    assert scalars["charts/total_episodes"] == 3
    envs.close.assert_called_once()


@pytest.mark.parametrize("invalid", ["broadcast_mask", "float_mask"])
def test_gae_rejects_invalid_mask(invalid: str) -> None:
    rewards = jnp.ones((2, 2), dtype=jnp.float32)
    dones = jnp.zeros((2, 1), dtype=jnp.bool_) if invalid == "broadcast_mask" else jnp.zeros((2, 2), dtype=jnp.float32)
    with pytest.raises(AssertionError):
        gae(rewards, dones, jnp.zeros_like(rewards), jnp.zeros(2), 0.9, 0.8)


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


@pytest.mark.parametrize("frame_budget, model_type", ((1, "lstm"), (5, "lstm"), (5, "gdn2"), (5, "mamba3")))
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
    cell = ppo.GatedDeltaNet2Recurrent(config, 2, 8)
    inputs = jax.random.normal(jax.random.key(2), (4, 2, 8))
    starts = jnp.zeros((4, 2), dtype=bool).at[2, 0].set(True)
    carry = cell.initial_carry(2)
    params = cell.init(jax.random.key(1), inputs, carry, starts)

    def loss(x: jax.Array) -> jax.Array:
        return cell.apply(params, x, carry, starts)[1][-1].sum()

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
        ({"type": "lstm", "num_layers": 0}, "greater_than"),
        ({"type": "lstm", "num_layers": True}, "int_type"),
        ({"type": "lstm", "intermediate_size": -1}, "greater_than_equal"),
        ({"type": "lstm", "intermediate_size": 1.5}, "int_type"),
        ({"type": "gdn2", "lstm_hidden_size": 8}, "unexpected_keyword_argument"),
        ({"type": "lstm", "hidden_size": 0}, "greater_than"),
        ({"type": "lstm", "hidden_size": True}, "int_type"),
        ({"type": "gdn2", "hidden_size": 1.5}, "int_type"),
        ({"type": "gdn2", "backend": "cudnn"}, "literal_error"),
        ({"type": "lstm", "backend": "triton"}, "unexpected_keyword_argument"),
        ({"type": "mamba3", "backend": "triton"}, "unexpected_keyword_argument"),
        ({"type": "mamba3", "conv_size": 4}, "unexpected_keyword_argument"),
        ({"type": "mamba3", "head_dim": 5}, "divisible by head_dim"),
        ({"type": "mamba3", "num_groups": 5}, "divisible by num_groups"),
        ({"type": "mamba3", "state_size": 3}, "state_size must be even"),
        ({"type": "mamba3", "state_size": 2}, "rotary pair"),
        ({"type": "mamba3", "rope_fraction": 0.25}, "literal_error"),
        ({"type": "mamba3", "intermediate_size": -1}, "greater_than_equal"),
        *[
            ({"type": "mamba3", name: 0}, "greater_than")
            for name in ("hidden_size", "num_layers", "state_size", "expand", "head_dim", "num_groups", "mimo_rank")
        ],
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


@pytest.mark.parametrize("backend", ("jax", "triton"))
def test_model_config_loading_and_factory(tmp_path: Path, backend: str) -> None:
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
        "backend": backend,
    }
    path.write_text(yaml.safe_dump(settings))
    config = load_config(path)
    model = ppo.make_model(config, 3)
    assert isinstance(model.model, ppo.GDN2Config)
    assert model.dtype == jnp.bfloat16
    assert model.model.hidden_size == 8
    assert model.model.intermediate_size == 8
    recurrent = model._make_recurrent()
    assert isinstance(recurrent, ppo.GatedDeltaNet2Recurrent)
    assert recurrent.backend == backend
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


def test_mamba3_config_loading_and_factory(tmp_path: Path) -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        model=small_model_config("mamba3"),
    )
    path = tmp_path / "ppo.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    restored = load_config(path)
    assert restored == config
    assert hash(restored) == hash(config)
    model = ppo.make_model(restored, 3)
    assert model.dtype == jnp.bfloat16
    carry = ppo.initial_model_carry(restored, 2)
    assert len(carry) == 2
    for layer in carry:
        assert layer.state.shape == (2, 4, 4, 4)
        assert layer.key.shape == (2, 4, 2, 4)
        assert layer.value.shape == (2, 4, 4)
        assert layer.angle.shape == (2, 4, 1)
        for leaf in layer:
            assert leaf.dtype == jnp.float32
            np.testing.assert_array_equal(leaf, 0)
    # The smallest rotary state works with full rotation; zero disables the MLP.
    boundary = ppo.Mamba3Config(state_size=2, rope_fraction=1.0, intermediate_size=0)
    assert boundary.intermediate_size == 0
    assert ppo.Mamba3Config().type == "mamba3"


def test_lstm_stack_config_loading_and_factory(tmp_path: Path) -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        model=small_model_config("lstm"),
    )
    path = tmp_path / "ppo.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    restored = load_config(path)
    assert restored == config
    model = ppo.make_model(restored, 3)
    recurrent = model._make_recurrent()
    assert isinstance(recurrent, ppo.LSTMStack)
    assert recurrent.num_layers == 2
    assert recurrent.intermediate_size == 8
    assert recurrent.dtype == jnp.bfloat16
    carry = ppo.initial_model_carry(restored, 2)
    assert len(carry) == 2
    for state in jax.tree.leaves(carry):
        assert state.shape == (2, 8)
        assert state.dtype == jnp.float32
        np.testing.assert_array_equal(state, 0)
