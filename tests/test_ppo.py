from dataclasses import FrozenInstanceError, asdict, fields, replace
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import yaml
from flax.training.train_state import TrainState
from ppo_helpers import env_step
from pydantic import ValidationError

from rl2 import ppo
from rl2.lstm import initial_carry
from rl2.model.encoders.rgb import ConvStage
from rl2.ppo import (
    ActorCritic,
    action_log_prob,
    learning_rate_schedule,
    load_config,
    train,
    update,
)


def small_model_config() -> ppo.ModelConfig:
    return ppo.ModelConfig(
        encoder=ppo.RGBEncoderConfig(stages=(ConvStage(4, blocks=1),), embedding_size=8),
        recurrent=ppo.LSTMConfig(hidden_size=8, num_layers=2, intermediate_size=8),
    )


@pytest.mark.parametrize(
    "num_steps,selected",
    [(1, (1,)), (3, (1, 0))],
    ids=["single-transition", "reordered-sequences"],
)
def test_batch_select_sequences_preserves_time_and_initial_memory(num_steps: int, selected: tuple[int, ...]) -> None:
    targets = jnp.arange(num_steps * 2).reshape(num_steps, 2)  # (T, B=2)
    memory = jnp.arange(4, dtype=jnp.float32).reshape(2, 2)  # (B=2, D=2)
    batch = ppo.PPOBatch(
        observations=targets[..., None, None, None],  # (T, B, 1, 1, 1)
        actions=targets % 2,  # (T, B)
        old_log_probs=-targets.astype(jnp.float32),  # (T, B)
        values=targets + 1.0,  # (T, B)
        advantages=targets + 2.0,  # (T, B)
        returns=targets + 3.0,  # (T, B)
        carry=((memory, memory + 1), (memory + 2, memory + 3)),  # leaves: (B, D)
        episode_starts=targets % 2 == 0,  # (T, B)
    )
    indices = jnp.asarray(selected)  # (M,)
    minibatch = batch.select_sequences(indices)  # sequences: (T, M, ...); memory: (M, D)
    for field in fields(batch):
        if field.name == "carry":
            continue
        actual = getattr(minibatch, field.name)  # (T, M, ...)
        original = getattr(batch, field.name)  # (T, B, ...)
        assert actual.shape == (num_steps, len(selected), *original.shape[2:])
        assert actual.dtype == original.dtype
        np.testing.assert_array_equal(actual, np.take(original, selected, axis=1))
    assert len(minibatch.carry) == 2
    for actual, original in zip(jax.tree.leaves(minibatch.carry), jax.tree.leaves(batch.carry), strict=True):
        assert actual.shape == (len(selected), 2)
        np.testing.assert_array_equal(actual, np.take(original, selected, axis=0))


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
    assert final[0][0].shape == (1, config.model.recurrent.hidden_size)
    step_obs = jax.ShapeDtypeStruct(obs.shape[1:], obs.dtype)
    step_final, step_logits, step_values = jax.eval_shape(
        partial(model.apply, method=model.step), variables, step_obs, carry, starts[0]
    )
    assert step_logits.shape == (1, 6)
    assert step_values.shape == (1,)
    assert step_final[0][0].shape == final[0][0].shape


@pytest.mark.parametrize("shape", [(2, 2), (4,)], ids=["recurrent", "step"])
def test_clipped_policy_loss_and_gradient_direction(shape: tuple[int, ...]) -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        target_kl=None,
        entropy_coef=0.0,
        value_coef=0.5,
    )
    probabilities = np.array([0.25, 0.75, 0.75, 0.25], dtype=np.float32).reshape(shape)  # (*S,)
    logits = jnp.log(jnp.stack((probabilities, 1 - probabilities), axis=-1))  # (*S, A)

    def apply(
        variables: optax.Params,  # logits: (*S, A); values: (*S,)
        obs: jax.Array,  # (*S, 1)
        carry: ppo.RecurrentCarry,  # leaves: (B, 1)
        starts: jax.Array,  # (*S,)
    ) -> tuple[ppo.RecurrentCarry, jax.Array, jax.Array]:
        return carry, variables["params"]["logits"], variables["params"]["values"]  # leaves: (B, 1); (*S, A); (*S,)

    state = TrainState.create(
        apply_fn=apply,
        params={"logits": logits, "values": jnp.zeros(shape)},  # (*S, A); (*S,)
        tx=optax.sgd(0.1),
    )
    advantages = jnp.array([-3.0, -1.0, 1.0, 3.0]).reshape(shape)  # (*S,)
    returns = jnp.array([-1.0, 0.0, 1.0, 2.0]).reshape(shape)  # (*S,)
    batch = ppo.PPOBatch(
        observations=jnp.zeros((*shape, 1)),  # (*S, *O)
        actions=jnp.zeros(shape, dtype=jnp.int32),  # (*S,)
        old_log_probs=jnp.full(shape, np.log(0.5)),  # (*S,)
        values=jnp.zeros(shape),  # (*S,)
        advantages=advantages,  # (*S,)
        returns=returns,  # (*S,)
        carry=initial_carry(shape[-1], 1),  # leaves: (B, ...)
        episode_starts=jnp.zeros(shape, dtype=bool),  # (*S,)
    )
    updated, metrics = update(state, batch, config)
    assert isinstance(metrics, ppo.PPOMetrics)
    chex.assert_shape(jax.tree.leaves(metrics), ())
    normalized = advantages / advantages.std()  # (*S,)
    ratio = probabilities / 0.5  # (*S,)
    expected = -np.minimum(
        ratio * normalized, np.clip(ratio, 1 - config.clip_coef, 1 + config.clip_coef) * normalized
    ).mean()  # ()
    assert float(metrics.policy_loss) == pytest.approx(expected, rel=0, abs=5e-07)
    assert float(metrics.value_loss) == pytest.approx(0.75, rel=0, abs=5e-07)
    assert updated.params["logits"].shape == (*shape, 2)
    assert updated.params["values"].shape == shape
    np.testing.assert_allclose(updated.params["values"], 0.1 * config.value_coef * returns / returns.size)
    before, after = jax.tree.map(jnp.ravel, (logits[..., 0], updated.params["logits"][..., 0]))  # (N,)
    np.testing.assert_array_equal(after[::2], before[::2])  # Both clipped signs.
    assert float(after[1]) < float(before[1])
    assert float(after[3]) > float(before[3])


def test_single_transition_keeps_policy_gradient() -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        target_kl=None,
        entropy_coef=0.0,
        value_coef=0.0,
    )

    def apply(
        variables: optax.Params, obs: jax.Array, carry: ppo.RecurrentCarry, starts: jax.Array
    ) -> tuple[ppo.RecurrentCarry, jax.Array, jax.Array]:
        chex.assert_shape(obs, (1, 1, 1, 1, 1))
        chex.assert_type(obs, jnp.uint8)
        chex.assert_shape(starts, (1, 1))
        chex.assert_type(starts, jnp.bool_)
        chex.assert_shape(variables["params"]["logits"], (1, 1, 2))
        chex.assert_type(variables["params"]["logits"], jnp.float32)
        return carry, variables["params"]["logits"], jnp.zeros((1, 1))

    state = TrainState.create(apply_fn=apply, params={"logits": jnp.zeros((1, 1, 2))}, tx=optax.sgd(0.1))
    batch = ppo.PPOBatch(
        observations=jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.uint8),  # (*S, *O)
        actions=jnp.zeros((1, 1), dtype=jnp.int32),  # (*S,)
        old_log_probs=jnp.full((1, 1), -np.log(2)),  # (*S,)
        values=jnp.zeros((1, 1)),  # (*S,)
        advantages=jnp.ones((1, 1)),  # (*S,)
        returns=jnp.zeros((1, 1)),  # (*S,)
        carry=initial_carry(1, 1),  # leaves: (B, ...)
        episode_starts=jnp.ones((1, 1), dtype=bool),  # (*S,)
    )
    updated, metrics = update(state, batch, config)
    assert float(metrics.policy_loss) == pytest.approx(-1.0)
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


def test_bf16_recurrent_training_keeps_float32_state_and_losses() -> None:
    model_config = small_model_config()
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
    )
    obs = jax.random.randint(jax.random.key(2), (3, 2, 1, 8, 8), 0, 256, dtype=jnp.uint8)
    carry = model.initial_carry(2)
    starts = jnp.array([[True, True], [False, False], [True, False]])
    params = model.init(jax.random.key(1), obs, carry, starts)["params"]
    (final, logits, values), captured = jax.jit(
        partial(model.apply, capture_intermediates=True, mutable=["intermediates"])
    )({"params": params}, obs, carry, starts)
    assert captured["intermediates"]["encoder"]["stem"]["__call__"][0].dtype == jnp.bfloat16
    assert captured["intermediates"]["lstm_input"]["__call__"][0].dtype == jnp.bfloat16
    assert captured["intermediates"]["policy_hidden"]["__call__"][0].dtype == jnp.bfloat16
    assert captured["intermediates"]["policy_output"]["__call__"][0].dtype == jnp.bfloat16
    assert captured["intermediates"]["value_output"]["__call__"][0].dtype == jnp.bfloat16
    for array in (*jax.tree.leaves(final), logits, values, *jax.tree.leaves(params)):
        assert array.dtype == jnp.float32
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(0.001))
    replayed_log_probs = []
    apply_step = jax.jit(state.apply_fn, static_argnames=("method",))
    for t in range(3):
        carry, step_logits, prediction = apply_step(
            {"params": state.params}, obs[t], carry, starts[t], method="step"
        )  # carry leaves: (B, ...); step_logits: (B, A); prediction: (B,)
        actions = jax.random.categorical(jax.random.key(t), step_logits)  # (B,)
        log_probs = action_log_prob(step_logits, actions)  # (B,)
        # BF16 kernels may round differently for sequence and single-step batches.
        np.testing.assert_allclose(prediction, values[t], atol=0.02, rtol=0.02)
        np.testing.assert_allclose(log_probs, action_log_prob(logits[t], actions), atol=1e-3, rtol=0)
        replayed_log_probs.append(action_log_prob(logits[t], jnp.zeros(2, dtype=jnp.int32)))
    # Float32 carry still accumulates differences from the BF16 encoder.
    for actual, expected in zip(jax.tree.leaves(carry), jax.tree.leaves(final), strict=True):
        np.testing.assert_allclose(actual, expected, atol=0.03, rtol=0.02)
    batch = ppo.PPOBatch(
        observations=obs,  # (*S, *O)
        actions=jnp.zeros((3, 2), dtype=jnp.int32),  # (*S,)
        old_log_probs=jnp.stack(replayed_log_probs),  # (*S,)
        values=values,  # (*S,)
        advantages=jnp.arange(6, dtype=jnp.float32).reshape(3, 2),  # (*S,)
        returns=jnp.ones((3, 2)),  # (*S,)
        carry=model.initial_carry(2),  # leaves: (B, ...)
        episode_starts=starts,  # (*S,)
    )
    updated, metrics = update(state, batch, config)
    assert int(updated.step) == 1
    for array in jax.tree.leaves((metrics, updated.params, updated.opt_state)):
        assert np.isfinite(array).all()
        if jnp.issubdtype(array.dtype, jnp.floating):
            assert array.dtype == jnp.float32
    assert any(
        (
            not np.array_equal(before, after)
            for before, after in zip(jax.tree.leaves(params), jax.tree.leaves(updated.params))
        )
    )


@jax.default_matmul_precision("highest")
def test_recurrent_sequences_match_steps_and_reset_only_finished_env() -> None:
    # Test sequence/reset semantics in float32, without GPU TF32 approximation.
    model = ActorCritic(
        3,
        replace(small_model_config(), encoder=ppo.RGBEncoderConfig(stages=(ConvStage(4, blocks=1),), embedding_size=6)),
    )
    obs = jax.random.randint(jax.random.key(2), (4, 2, 4, 8, 8), 0, 256, dtype=jnp.uint8)
    carry = model.initial_carry(2)
    starts = jnp.array([[True, True], [False, False], [True, False], [False, False]])
    params = model.init(jax.random.key(1), obs, carry, starts)
    encoded = model.apply(params, obs[0], method=model._encode)
    assert encoded.shape == (2, model.model.recurrent.hidden_size)
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
    peek = step(params, obs[2], prefix_carry, starts[2])[2]  # (B,)
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


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"num_envs": 3, "num_minibatches": 2}, "num_envs must be divisible"),
        ({"num_envs": 2, "num_steps": 4, "num_minibatches": 2, "total_steps": 7}, "total_steps must cover"),
    ],
)
def test_invalid_rollout_relationships_rejected_by_config(overrides: dict[str, int], message: str) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")
    with pytest.raises(ValidationError, match=message):
        replace(config, **overrides)


def test_config_accepts_exactly_one_rollout() -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_envs=2,
        num_steps=4,
        num_minibatches=2,
        total_steps=8,
    )
    assert config.total_steps == config.num_envs * config.num_steps


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("num_envs", 0),
        ("num_steps", 0),
        ("num_minibatches", 0),
        ("update_epochs", 0),
        ("video_fps", 0.0),
        ("video_fps", float("inf")),
        ("vector_env", "unknown"),
    ],
)
def test_invalid_rollout_settings_rejected_by_config(field: str, invalid: Any) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")
    with pytest.raises(ValidationError, match=field):
        replace(config, **{field: invalid})


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
        num_minibatches=1,
        total_steps=4,
        entropy_decay="cosine",
        target_kl=None,
    )

    def apply(
        variables: optax.Params, obs: jax.Array, carry: ppo.RecurrentCarry, starts: jax.Array
    ) -> tuple[ppo.RecurrentCarry, jax.Array, jax.Array]:
        return carry, variables["params"]["logits"], jnp.zeros_like(starts, dtype=jnp.float32)

    probabilities = np.array([[0.8, 0.2]], dtype=np.float32)
    logits = jnp.log(probabilities)
    state = TrainState.create(apply_fn=apply, params={"logits": logits}, tx=optax.sgd(1.0))
    state = state.replace(step=123)  # Optimizer step count must not drive decay.
    batch = ppo.PPOBatch(
        observations=jnp.zeros((1, 1)),  # (*S, *O)
        actions=jnp.zeros(1, dtype=jnp.int32),  # (*S,)
        old_log_probs=logits[:, 0],  # (*S,)
        values=jnp.zeros(1),  # (*S,)
        advantages=jnp.zeros(1),  # (*S,)
        returns=jnp.zeros(1),  # (*S,)
        carry=initial_carry(1, 1),  # leaves: (B, ...)
        episode_starts=jnp.zeros(1, dtype=bool),  # (*S,)
    )
    entropy = -(probabilities * np.log(probabilities)).sum()
    entropy_grad = -probabilities * (np.log(probabilities) + entropy)
    for iteration, factor in ((0, 1.0), (1, (1 + np.sqrt(0.5)) / 2), (4, 0.0)):
        updated, metrics = update(state, batch, config, iteration)
        np.testing.assert_allclose(
            updated.params["logits"], logits + config.entropy_coef * factor * entropy_grad, atol=1e-7
        )
        assert float(metrics.entropy) == pytest.approx(entropy)
        chex.assert_shape(jax.tree.leaves(metrics), ())
        assert float(metrics.entropy_loss) == pytest.approx(-config.entropy_coef * factor * entropy)
        assert int(updated.step) == 124


def test_train_logs_metric_windows_with_kl_stopping(capsys: pytest.CaptureFixture[str]) -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=2,
        total_steps=3,
        anneal_lr=True,
        lr_decay="cosine",
        entropy_decay="cosine",
        target_kl=0.01,
        video_interval=0,
        eval_interval=0,
        log_interval=2,
    )
    obs = np.zeros((1, 1), dtype=np.uint8)  # (B=1, O=1)
    zeros = np.zeros(1, dtype=np.float32)  # (B,)
    dones = np.zeros(1, dtype=bool)  # (B,)
    carry = initial_carry(1, 1)  # pair of (B, H=1)
    envs = Mock(num_actions=2, params={})
    envs.reset.return_value = (obs[0], {})
    # One episode spans both rollouts in the first window; the final window has none.
    envs.step.side_effect = [
        env_step(obs[0], np.array(reward), ~dones[0] if done else dones[0], dones[0], {})  # (O,), scalar reward/flags
        for reward, done in ((2.0, False), (6.0, True), (100.0, False))
    ]
    model = Mock()
    model.initial_carry.return_value = carry
    model.init.return_value = {"params": {"weight": jnp.zeros(1)}}  # weight: (1,)
    writer = Mock()
    iterations: list[int] = []
    learning_rates: list[float] = []
    losses = iter((1.0, 3.0, 8.0, 10.0, 14.0))
    clock = 0.0

    def fake_update(
        state: TrainState,
        batch: ppo.PPOBatch,  # obs: (T=1, B=1, O=1); targets/starts: (T, B); carry: pair of (B, H)
        config: ppo.Config,
        iteration: int,
    ) -> tuple[TrainState, ppo.PPOMetrics]:
        nonlocal clock
        chex.assert_shape(batch.observations, (1, 1, 1))
        chex.assert_shape((batch.actions, batch.old_log_probs, batch.advantages, batch.returns), (1, 1))
        chex.assert_shape(batch.episode_starts, (1, 1))
        iterations.append(iteration)
        learning_rates.append(float(state.opt_state.hyperparams["learning_rate"]))
        clock = (30.0, 60.0, 80.0)[iteration]
        loss = next(losses)
        kl = 0.02 if iteration == 1 else 0.0
        return state, ppo.PPOMetrics(
            policy_loss=jnp.asarray(loss),  # ()
            value_loss=jnp.asarray(loss * 2),  # ()
            entropy_loss=jnp.asarray(-config.entropy_coef * (1.0, 0.75, 0.25)[iteration] * loss * 3),  # ()
            entropy=jnp.asarray(loss * 3),  # ()
            approx_kl=jnp.asarray(kl),  # ()
            clip_fraction=jnp.asarray(loss / 100),  # ()
        )

    def now() -> float:
        return clock

    with (
        jax.disable_jit(),
        patch("rl2.ppo.GymnaxEnv", return_value=envs),
        patch("rl2.ppo.ActorCritic", return_value=model) as actor_critic,
        patch("rl2.ppo.SummaryWriter", return_value=writer),
        patch.object(
            model, "apply", return_value=(carry, jnp.array([[0.0, -jnp.inf]]), zeros)
        ),  # carry leaves: (B,H); (B,A); (B,)
        patch("rl2.ppo.update", side_effect=fake_update),
        patch("rl2.ppo.explained_variance", side_effect=(0.2, 0.4, 0.8)),
        patch("rl2.ppo.monotonic", side_effect=now),
        patch("rl2.ppo.checkpoint_manager"),
        patch("rl2.ppo.restore_checkpoint", return_value=None),
        patch("rl2.ppo.save_checkpoint"),
    ):
        train(config)
    actor_critic.assert_called_once_with(
        2,
        config.model,
        dtype=jnp.bfloat16 if config.bf16 else jnp.float32,
    )
    assert iterations == [0, 0, 1, 2, 2]
    np.testing.assert_allclose(learning_rates, config.learning_rate * np.array([1.0, 1.0, 0.75, 0.25, 0.25]))
    logged: dict[str, list[tuple[float, int]]] = {}
    for call in writer.add_scalar.call_args_list:
        logged.setdefault(call.args[0], []).append(call.args[1:])
    assert "time/window_seconds" not in logged
    assert "time/iteration_seconds" not in logged
    assert "charts/steps_per_second" not in logged
    assert not any("last_100" in tag for tag in logged)
    # Minibatch-weighted means: the rejected rollout contributes one sample, not two.
    for tag, expected in {
        "losses/policy": [(4.0, 2), (12.0, 3)],
        "losses/value": [(8.0, 2), (24.0, 3)],
        "losses/entropy": [(-12.0, 2), (-36.0, 3)],
        "losses/total": [
            (4.0 + config.value_coef * 8.0 - 10.0 * config.entropy_coef, 2),
            (12.0 + config.value_coef * 24.0 - 9.0 * config.entropy_coef, 3),
        ],
        "policy/entropy": [(12.0, 2), (36.0, 3)],
        "policy/approx_kl": [(0.02 / 3, 2), (0.0, 3)],
        "policy/clip_fraction": [(0.04, 2), (0.12, 3)],
        "value/explained_variance": [(0.3, 2), (0.8, 3)],
        "charts/updates_per_rollout": [(1.0, 2), (2.0, 3)],
        "policy/early_stop": [(0.5, 2), (0.0, 3)],
        "charts/iterations_per_second": [(2 / 60, 2), (1 / 20, 3)],
        "charts/observations_per_second": [(2 / 60, 2), (1 / 20, 3)],
        "charts/episodes_per_second": [(1 / 60, 2), (0.0, 3)],
        "charts/episodes": [(1.0, 2), (0.0, 3)],
        "charts/return_mean": [(8.0, 2)],
        "charts/episode_length_mean": [(2.0, 2)],
        "charts/total_return_mean": [(8.0, 2), (8.0, 3)],
        "charts/total_episode_length_mean": [(2.0, 2), (2.0, 3)],
        "charts/total_episodes": [(1.0, 2), (1.0, 3)],
        "charts/total_observations": [(2.0, 2), (3.0, 3)],
        "charts/training_progress": [(2 / 3, 2), (1.0, 3)],
        "time/elapsed_seconds": [(60.0, 2), (80.0, 3)],
    }.items():
        np.testing.assert_allclose(logged[tag], expected, err_msg=tag)
    for tag, initial in (("charts/learning_rate", config.learning_rate), ("charts/entropy_coef", config.entropy_coef)):
        np.testing.assert_allclose(logged[tag], [(initial * 0.75, 2), (initial * 0.25, 3)], rtol=1e-6)
    progress_lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("iteration=")]
    assert len(progress_lines) == 2
    assert "progress=0.667" in progress_lines[0]
    assert "progress=1.000" in progress_lines[1]
    assert all("s/iteration=" not in line for line in progress_lines)
    assert "iterations/s=0.033 observations/s=0.033 episodes/s=0.017" in progress_lines[0]
    assert "iterations/s=0.050 observations/s=0.050 episodes/s=0.000" in progress_lines[1]
    assert not {
        "time/rollout_seconds",
        "time/rollout_env_seconds",
        "time/rollout_model_seconds",
        "time/optimization_seconds",
        "time/eta_seconds",
    }.intersection(logged)
    assert "step=2 episodes=1 return=8.0 episode_length=2.0" in progress_lines[0]
    assert "step=3 episodes=1 return=n/a episode_length=n/a" in progress_lines[1]
    assert writer.flush.call_count == 2
    writer.close.assert_called_once()


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_invalid_log_interval(interval: float) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")
    with pytest.raises(ValueError, match="log_interval"):
        replace(config, log_interval=interval)


def test_video_interval_defaults_to_600_iterations() -> None:
    settings = vars(load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")).copy()
    settings.pop("video_interval")
    assert ppo.Config(**settings).video_interval == 600


@pytest.mark.parametrize("interval", [-1, 1.5, True])
def test_invalid_video_interval(interval: float) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")
    with pytest.raises(ValidationError, match="video_interval"):
        replace(config, video_interval=interval)


@pytest.mark.parametrize("interval,expected_steps", [(2, [2, 4]), (0, [])], ids=["periodic", "disabled"])
def test_video_schedule_uses_iterations(interval: int, expected_steps: list[int]) -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        total_steps=5,
        video_interval=interval,
        eval_interval=0,
        target_kl=None,
    )
    obs = np.zeros((1, 1), dtype=np.uint8)  # (B=1, O=1)
    zeros = np.zeros(1, dtype=np.float32)  # (B,)
    dones = np.zeros(1, dtype=bool)  # (B,); no episodes complete during training
    carry = initial_carry(1, 1)  # pair of (B, H=1)
    envs = Mock(num_actions=2, params={})
    envs.reset.return_value = (obs[0], {})
    envs.step.return_value = env_step(obs[0], zeros[0], dones[0], dones[0], {})
    model = Mock()
    model.initial_carry.return_value = carry
    model.init.return_value = {"params": {"weight": jnp.zeros(1)}}  # weight: (1,)
    clock = 0.0

    def advance_update(
        state: TrainState,
        batch: ppo.PPOBatch,  # observations: (T=1, B, O); targets/starts: (T, B); carry: pair of (B, H)
        config: ppo.Config,
        iteration: int,
    ) -> tuple[TrainState, ppo.PPOMetrics]:
        nonlocal clock
        # Uneven update times must not affect iteration-based recording.
        clock += (599.0, 1.0, 1.0, 599.0, 1.0)[iteration]
        return state.replace(step=iteration + 1), ppo.PPOMetrics(
            policy_loss=jnp.asarray(0.0),  # ()
            value_loss=jnp.asarray(0.0),  # ()
            entropy_loss=jnp.asarray(0.0),  # ()
            entropy=jnp.asarray(0.0),  # ()
            approx_kl=jnp.asarray(0.0),  # ()
            clip_fraction=jnp.asarray(0.0),  # ()
        )

    def now() -> float:
        return clock

    def record_video(*args: object, **kwargs: object) -> None:
        nonlocal clock
        clock += 1200.0  # Recording time must not affect the schedule.

    with (
        jax.disable_jit(),
        patch.object(ppo, "GymnaxEnv", return_value=envs),
        patch.object(ppo, "ActorCritic", return_value=model),
        patch.object(ppo, "SummaryWriter") as writer,
        patch.object(
            model, "apply", return_value=(carry, jnp.array([[0.0, -jnp.inf]]), zeros)
        ),  # carry leaves: (B,H); (B,A); (B,)
        patch.object(ppo, "update", side_effect=advance_update),
        patch.object(ppo, "monotonic", side_effect=now),
        patch.object(ppo, "checkpoint_manager"),
        patch.object(ppo, "restore_checkpoint", return_value=None),
        patch.object(ppo, "save_checkpoint"),
        patch.object(ppo, "log_video", side_effect=record_video) as video,
    ):
        train(config)
    assert [call.args[4] for call in video.call_args_list] == expected_steps
    for call in video.call_args_list:
        state, recorded_config, recorded_writer, episodes, steps = call.args
        assert int(state.step) == steps
        assert recorded_config.video_interval == interval
        assert recorded_writer is writer.return_value
        assert episodes == 0
        assert call.kwargs["initial_carry"] is carry
    writer.return_value.close.assert_called_once()


@pytest.mark.parametrize(
    ("eval_interval", "expected_steps"),
    [(0, []), (3, [4]), (1, [2, 4])],
    ids=["disabled", "final-before-interval", "periodic-and-final-without-duplicate"],
)
def test_train_evaluates_final_policy(
    eval_interval: int, expected_steps: list[int], capsys: pytest.CaptureFixture[str]
) -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        num_envs=1,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        total_steps=5,  # Training ends after two complete rollouts, at step 4.
        video_interval=0,
        eval_interval=eval_interval,
        log_interval=1,
        target_kl=None,
    )
    obs = np.zeros((1, 1), dtype=np.uint8)
    zeros = np.zeros(1, dtype=np.float32)
    dones = np.ones(1, dtype=bool)
    carry = initial_carry(1, 1)
    envs = Mock(num_actions=2, params={})
    envs.reset.return_value = (obs[0], {})
    # The first game spans two steps; the last unfinished game must not count.
    envs.step.side_effect = [
        env_step(obs[0], np.array(reward), dones[0] if done else ~dones[0], ~dones[0], {})
        for reward, done in ((5.0, False), (-2.0, True), (7.0, True), (1000.0, False))
    ]
    model = Mock()
    model.initial_carry.return_value = carry
    model.init.return_value = {"params": {"weight": jnp.zeros(1)}}
    clock = 0.0

    def advance_update(
        state: TrainState, batch: ppo.PPOBatch, config: ppo.Config, iteration: int
    ) -> tuple[TrainState, ppo.PPOMetrics]:
        nonlocal clock
        clock += 60.0
        return state.replace(step=iteration + 1), ppo.PPOMetrics(
            policy_loss=jnp.asarray(0.0),  # ()
            value_loss=jnp.asarray(0.0),  # ()
            entropy_loss=jnp.asarray(0.0),  # ()
            entropy=jnp.asarray(0.0),  # ()
            approx_kl=jnp.asarray(0.0),  # ()
            clip_fraction=jnp.asarray(0.0),  # ()
        )

    def now() -> float:
        return clock

    def advance_evaluation(*args: object, **kwargs: object) -> None:
        nonlocal clock
        clock += 30.0

    with (
        jax.disable_jit(),
        patch.object(ppo, "GymnaxEnv", return_value=envs),
        patch.object(ppo, "ActorCritic", return_value=model),
        patch.object(ppo, "SummaryWriter") as writer,
        patch.object(
            model, "apply", return_value=(carry, jnp.array([[0.0, -jnp.inf]]), zeros)
        ),  # carry leaves: (B,H); (B,A); (B,)
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
        assert evaluated_config.eval_interval == eval_interval
        assert evaluated_writer is writer.return_value
        assert call.kwargs["initial_carry"] is carry
        assert episodes == (1 if steps == 2 else 2)
    if expected_steps:
        assert evaluate.call_args.args[0] is state
    first_elapsed = 90.0 if 2 in expected_steps else 60.0
    final_elapsed = 120.0 + 30.0 * len(expected_steps)
    for tag, expected in (
        ("charts/total_episodes", [(1.0, 2), (2.0, 4)]),
        ("charts/total_return_mean", [(3.0, 2), (5.0, 4)]),
        ("charts/total_episode_length_mean", [(2.0, 2), (1.5, 4)]),
        ("time/elapsed_seconds", [(first_elapsed, 2), (final_elapsed, 4)]),
        ("charts/iterations_per_second", [(1 / first_elapsed, 2), (1 / (final_elapsed - first_elapsed), 4)]),
        ("charts/observations_per_second", [(2 / first_elapsed, 2), (2 / (final_elapsed - first_elapsed), 4)]),
        ("charts/episodes_per_second", [(1 / first_elapsed, 2), (1 / (final_elapsed - first_elapsed), 4)]),
    ):
        logged = [call.args[1:] for call in writer.return_value.add_scalar.call_args_list if call.args[0] == tag]
        assert logged == expected
    output = capsys.readouterr().out
    assert f"elapsed=0:01:{int(first_elapsed % 60):02d}" in output
    assert f"elapsed=0:{int(final_elapsed // 60):02d}:{int(final_elapsed % 60):02d}" in output
    assert "eta=" not in output
    writer.return_value.close.assert_called_once()


def test_kl_rejects_update_without_changing_optimizer() -> None:
    config = replace(load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"), target_kl=0.01)
    state = TrainState.create(
        apply_fn=lambda variables, obs, carry, starts: (carry, variables["params"]["logits"], jnp.zeros(2)),
        params={"logits": jnp.log(jnp.array([[0.9, 0.1], [0.1, 0.9]]))},
        tx=optax.adam(0.01),
    )
    batch = ppo.PPOBatch(
        observations=jnp.zeros((2, 1)),  # (*S, *O)
        actions=jnp.zeros(2, dtype=jnp.int32),  # (*S,)
        old_log_probs=jnp.full(2, np.log(0.5)),  # (*S,)
        values=jnp.zeros(2),  # (*S,)
        advantages=jnp.array([1.0, -1.0]),  # (*S,)
        returns=jnp.zeros(2),  # (*S,)
        carry=initial_carry(2, 1),  # leaves: (B, ...)
        episode_starts=jnp.zeros(2, dtype=bool),  # (*S,)
    )
    stopped, metrics = update(state, batch, config)
    assert float(metrics.approx_kl) > config.target_kl
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
    batch = ppo.PPOBatch(
        observations=jnp.zeros((4, 1)),  # (*S, *O)
        actions=jnp.zeros(4, dtype=jnp.int32),  # (*S,)
        old_log_probs=jnp.full(4, np.log(0.5)),  # (*S,)
        values=jnp.zeros(4),  # (*S,)
        advantages=jnp.arange(4.0),  # (*S,)
        returns=jnp.arange(4.0),  # (*S,)
        carry=initial_carry(4, 1),  # leaves: (B, ...)
        episode_starts=jnp.zeros(4, dtype=bool),  # (*S,)
    )
    _, metrics = update(state, batch, config)
    ratios = np.array([0.4, 1.0, 1.6, 1.04])
    assert float(metrics.approx_kl) == pytest.approx(float(np.mean(ratios - 1 - np.log(ratios))), rel=0, abs=5e-07)
    assert float(metrics.clip_fraction) == 0.5
    same_policy_batch = batch.replace(old_log_probs=jnp.log(probabilities[:, 0]))  # (B,)
    _, metrics = update(state, same_policy_batch, config)
    assert float(metrics.approx_kl) == pytest.approx(0.0, rel=0, abs=5e-07)
    assert float(metrics.clip_fraction) == 0.0


# Every action-repeat offset matters; color and end flags do not require a product.


@pytest.mark.parametrize("frame_budget", (0, -1, 1.5, True))
def test_invalid_video_frame_budget_rejected_by_config(frame_budget: Any) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")
    with pytest.raises(ValidationError, match="video_max_frames"):
        replace(config, video_max_frames=frame_budget)


@pytest.mark.parametrize(
    "model_settings, message",
    [
        ({"type": "unknown"}, "literal_error"),
        ({"type": "gdn2"}, "literal_error"),
        ({"type": "mamba3"}, "literal_error"),
        ("lstm", "model"),
        ({"type": "lstm", "num_layers": 0}, "greater_than"),
        ({"type": "lstm", "num_layers": True}, "int_type"),
        ({"type": "lstm", "intermediate_size": -1}, "greater_than_equal"),
        ({"type": "lstm", "intermediate_size": 1.5}, "int_type"),
        ({"type": "lstm", "hidden_size": 0}, "greater_than"),
        ({"type": "lstm", "hidden_size": True}, "int_type"),
        ({"type": "lstm", "backend": "triton"}, "unexpected_keyword_argument"),
    ],
)
def test_invalid_model_config(model_settings: Any, message: str) -> None:
    settings = vars(load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")).copy()
    settings["model"] = {"recurrent": model_settings}
    with pytest.raises(ValidationError, match=message):
        ppo.Config(**settings)


@pytest.mark.parametrize("field", ("lstm_hidden_size", "gdn2_hidden_size"))
def test_flat_model_settings_are_rejected(field: str) -> None:
    settings = vars(load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")).copy()
    settings[field] = 8
    with pytest.raises(ValidationError, match=field):
        ppo.Config(**settings)


def test_default_model_config(tmp_path: Path) -> None:
    settings = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/ppo.yaml").read_text())
    settings.pop("model")
    path = tmp_path / "ppo.yaml"
    path.write_text(yaml.safe_dump(settings))
    assert load_config(path).model == ppo.ModelConfig()
    settings["model"] = {"recurrent": {"hidden_size": 8}}
    path.write_text(yaml.safe_dump(settings))
    assert load_config(path).model == ppo.ModelConfig(recurrent=ppo.LSTMConfig(hidden_size=8))


def test_lstm_stack_config_loading_and_factory(tmp_path: Path) -> None:
    config = replace(
        load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
        model=small_model_config(),
    )
    path = tmp_path / "ppo.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    restored = load_config(path)
    assert restored == config
    assert hash(restored) == hash(config)
    with pytest.raises(FrozenInstanceError):
        config.model.recurrent.hidden_size = 16
    model = ppo.ActorCritic(3, restored.model, dtype=jnp.bfloat16 if restored.bf16 else jnp.float32)
    recurrent = model._make_recurrent()
    assert isinstance(recurrent, ppo.LSTMStack)
    assert recurrent.num_layers == 2
    assert recurrent.intermediate_size == 8
    assert recurrent.dtype == jnp.bfloat16
    carry = model.initial_carry(2)  # leaves: (2, ...)
    assert len(carry) == 2
    for state in jax.tree.leaves(carry):
        assert state.shape == (2, 8)
        assert state.dtype == jnp.float32
        np.testing.assert_array_equal(state, 0)
