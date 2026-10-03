from dataclasses import replace
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from rl2.grpo import (
    Config,
    GRPOBatch,
    act,
    action_log_prob,
    collect_rollout,
    group_advantages,
    learning_rate_schedule,
    load_config,
    objective,
    train,
    update,
)
from rl2.karel import TOKENS, KarelConfig, KarelProgramEnv, KarelTask, StepResult
from rl2.karel_model import KarelProgramModel


@pytest.fixture(scope="module")
def config() -> Config:
    return Config(
        total_updates=1,
        num_tasks=1,
        group_size=2,
        num_minibatches=1,
        update_epochs=1,
        d_model=8,
        num_layers=1,
        d_state=8,
        headdim=4,
        conv_channels=(4,),
        target_kl=None,
        env=KarelConfig(height=3, width=3, max_depth=0, max_statements=1, max_program_tokens=6),
    )


@pytest.fixture(scope="module")
def state(config: Config) -> TrainState:
    model = KarelProgramModel(
        d_model=config.d_model,
        num_layers=config.num_layers,
        d_state=config.d_state,
        headdim=config.headdim,
        conv_channels=config.conv_channels,
    )
    initial, target = KarelProgramEnv(config.env).reset(seed=42)
    params = model.init(jax.random.key(0), initial[None], target[None], jnp.empty((0, 1), jnp.int32))["params"]
    return TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(0.001))


def batch_from_logits(logits: jax.Array, mask: jax.Array, advantages: jax.Array) -> GRPOBatch:
    actions = jnp.where(mask, KarelProgramEnv.terminal_token_id, KarelProgramEnv.pad_token_id).astype(jnp.int32)
    initial = jnp.zeros((logits.shape[1], 3, 3, 6), dtype=jnp.int32)
    return GRPOBatch(initial, initial, actions, action_log_prob(logits, actions), mask, advantages)


def test_advantages_are_group_relative_and_constant_groups_are_zero() -> None:
    rewards = jnp.asarray([[0, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=jnp.float32)
    np.testing.assert_allclose(group_advantages(rewards), [[-1, 1, -1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])


@pytest.mark.parametrize("group_size", [8, 16, 32, 128])
def test_identical_fractional_rewards_have_exactly_zero_advantages(group_size: int) -> None:
    values = jnp.asarray([0.1, 1 / 3, -1 / 3, 0.7, -2.0], dtype=jnp.float32)
    rewards = jnp.broadcast_to(values[:, None], (len(values), group_size))
    np.testing.assert_array_equal(group_advantages(rewards), 0)


def test_fractional_advantages_match_float64_reference() -> None:
    rewards = np.asarray([[0.1, 0.1, 0.100001, 0.099999], [-2, -1, 1 / 3, 0.7]], dtype=np.float32)
    reference = rewards.astype(np.float64)
    expected = (reference - reference.mean(axis=1, keepdims=True)) / (reference.std(axis=1, keepdims=True) + 1e-8)
    np.testing.assert_allclose(group_advantages(rewards), expected, atol=2e-7, rtol=2e-7)


def test_equal_partial_rewards_are_not_logged_as_informative(
    config: Config, state: TrainState, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(config, group_size=128)

    def equal_reward_step(self: KarelProgramEnv, action: int) -> StepResult:
        # Isolate rollout statistics with identical fractional terminal rewards.
        assert self.action_space.contains(action)
        return (
            None,
            0.1,
            True,
            False,
            {
                "success": False,
                "error": "syntax_error",
                "reward_syntax": 0.1,
                "reward_runtime": 0.0,
                "reward_distance": 0.0,
                "reward_success": 0.0,
            },
        )

    monkeypatch.setattr(KarelProgramEnv, "step", equal_reward_step)
    envs = [KarelProgramEnv(config.env) for _ in range(config.group_size)]
    batch, rewards, diagnostics, _ = collect_rollout(state, envs, np.random.default_rng(0), jax.random.key(0), config)
    np.testing.assert_array_equal(rewards, np.float32(0.1))
    np.testing.assert_array_equal(batch.advantages, 0)
    assert diagnostics["charts/informative_group_fraction"] == 0.0


def test_objective_weights_programs_equally_and_masks_padding(config: Config) -> None:
    logits = jnp.zeros((4, 2, len(TOKENS)), dtype=jnp.float32)
    mask = jnp.asarray([[True, True], [False, True], [False, True], [False, True]])
    batch = batch_from_logits(logits, mask, jnp.asarray([1.0, -1.0]))
    loss, metrics = objective(logits, batch, config)
    np.testing.assert_allclose(loss, 0, atol=1e-6)
    np.testing.assert_allclose(metrics[1], np.log(len(TOKENS) - 1), rtol=1e-6)
    # Even extreme log-probabilities in padding must not affect loss or gradients.
    padded_batch = batch._replace(old_log_probs=jnp.where(mask, batch.old_log_probs, -1000.0))
    changed = logits.at[1:, 0, KarelProgramEnv.terminal_token_id].set(10)
    padded_loss, padded_metrics = objective(changed, padded_batch, config)
    np.testing.assert_allclose(padded_loss, loss, atol=1e-6)
    np.testing.assert_allclose(padded_metrics, metrics, atol=1e-6)

    def loss_fn(values: jax.Array) -> jax.Array:
        return objective(values, padded_batch, config)[0]

    gradients = jax.grad(loss_fn)(changed)
    assert np.isfinite(gradients).all()
    np.testing.assert_array_equal(gradients[1:, 0], 0)


def test_clipped_objective_and_reference_penalty(config: Config) -> None:
    logits = jnp.zeros((1, 2, len(TOKENS)), dtype=jnp.float32)
    batch = batch_from_logits(logits, jnp.ones((1, 2), dtype=jnp.bool_), jnp.asarray([1.0, -1.0]))
    ratios = jnp.asarray([[2.0, 0.5]], dtype=jnp.float32)
    batch = batch._replace(old_log_probs=batch.old_log_probs - jnp.log(ratios))
    loss, metrics = objective(logits, batch, config)
    np.testing.assert_allclose(loss, -config.clip_coef, atol=1e-6)
    np.testing.assert_allclose(metrics[3], 1.0)
    reference = action_log_prob(logits, batch.actions) - 1
    penalized, kl_metrics = objective(logits, batch, replace(config, kl_coef=0.1), reference)
    np.testing.assert_allclose(kl_metrics[4], np.exp(-1), atol=1e-6)
    np.testing.assert_allclose(penalized, loss + 0.1 * np.exp(-1), atol=1e-6)


@pytest.mark.parametrize("token", ["m)", "move"])
def test_collection_shares_pairs_and_handles_terminal_and_truncation(
    config: Config, state: TrainState, token: str
) -> None:
    config = replace(config, num_tasks=2)
    params = {
        **state.params,
        "head": {
            "kernel": jnp.zeros_like(state.params["head"]["kernel"]),
            "bias": jnp.full_like(state.params["head"]["bias"], -100).at[KarelProgramEnv.token_to_id[token]].set(100),
        },
    }
    state = state.replace(params=params)
    envs = [KarelProgramEnv(config.env) for _ in range(config.num_tasks * config.group_size)]
    batch, rewards, diagnostics, _ = collect_rollout(state, envs, np.random.default_rng(3), jax.random.key(3), config)
    for start in range(0, len(envs), config.group_size):
        np.testing.assert_array_equal(batch.initial[start], batch.initial[start + 1])
        np.testing.assert_array_equal(batch.target[start], batch.target[start + 1])
    np.testing.assert_array_equal(rewards, np.float32(0.2))
    np.testing.assert_array_equal(batch.advantages, 0)
    expected_length = 1 if token == "m)" else config.env.max_program_tokens
    np.testing.assert_array_equal(batch.mask.sum(axis=0), expected_length)
    np.testing.assert_array_equal(batch.actions[~batch.mask], KarelProgramEnv.pad_token_id)
    np.testing.assert_array_equal(batch.mask, batch.actions != KarelProgramEnv.pad_token_id)
    assert diagnostics["charts/truncation_rate"] == (token == "move")
    assert diagnostics["charts/syntax_error_rate"] == (token == "m)")
    assert diagnostics["charts/reward_mean"] == pytest.approx(0.2)
    assert diagnostics["charts/reward_syntax_mean"] == pytest.approx(0.2)
    for name in ("runtime", "distance", "success"):
        assert diagnostics[f"charts/reward_{name}_mean"] == 0.0
    assert diagnostics["charts/success_rate"] == diagnostics["charts/group_success_rate"] == 0.0
    _, logits = state.apply_fn({"params": state.params}, batch.initial, batch.target, batch.actions[:-1])
    recomputed = np.asarray(action_log_prob(logits, batch.actions))
    np.testing.assert_allclose(recomputed[batch.mask], batch.old_log_probs[batch.mask], atol=1e-6)


@pytest.mark.parametrize(
    "body,reward,success,error",
    [
        ("pickMarker pickMarker", 4.0, True, None),
        ("putMarker", 2.25, False, None),  # Regression still beats invalid syntax.
        ("pickMarker pickMarker pickMarker", 2.0, False, "runtime_error"),
    ],
)
def test_partial_rewards_are_not_logged_as_successes(
    config: Config,
    state: TrainState,
    monkeypatch: pytest.MonkeyPatch,
    body: str,
    reward: float,
    success: bool,
    error: str | None,
) -> None:
    config = replace(config, num_tasks=2, env=replace(config.env, max_program_tokens=8))
    initial = np.zeros((3, 3, 6), dtype=np.int32)
    initial[..., 4] = 1
    initial[1, 1, 4] = 0
    initial[1, 1, 0] = 1
    initial[1, 1, 5] = 2
    target = initial.copy()
    target[1, 1, 5] = 0

    def fixed_task(rng: np.random.Generator, config: KarelConfig) -> KarelTask:
        return KarelTask(initial, target, ("DEF", "run", "m(", "pickMarker", "pickMarker", "m)"))

    programs = [
        "DEF run m( pickMarker m)",
        "DEF run m( turnLeft turnRight m)",
        f"DEF run m( {body} m)",
        "m)",
    ]
    scripted = np.full((8, 4), KarelProgramEnv.pad_token_id, dtype=np.int32)
    for column, program in enumerate(programs):
        ids = [KarelProgramEnv.token_to_id[token] for token in program.split()]
        scripted[: len(ids), column] = ids
    steps = iter(scripted)

    def scripted_act(logits: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        chex.assert_shape(logits, (4, len(TOKENS)))
        chex.assert_type(logits, jnp.float32)
        chex.assert_shape(key, ())
        return jnp.asarray(next(steps)), jnp.zeros(4, dtype=jnp.float32)

    monkeypatch.setattr("rl2.karel.sample_task", fixed_task)
    monkeypatch.setattr("rl2.grpo.act", scripted_act)
    envs = [KarelProgramEnv(config.env) for _ in range(4)]
    batch, rewards, diagnostics, _ = collect_rollout(state, envs, np.random.default_rng(0), jax.random.key(0), config)
    np.testing.assert_array_equal(rewards, np.asarray([2.75, 2.5, reward, 0.2], dtype=np.float32))
    np.testing.assert_allclose(batch.advantages, [1.0, -1.0, 1.0, -1.0], atol=2e-7)
    assert diagnostics["charts/reward_mean"] == pytest.approx((reward + 5.45) / 4)
    assert diagnostics["charts/reward_syntax_mean"] == pytest.approx(0.8)
    assert diagnostics["charts/reward_runtime_mean"] == pytest.approx(0.75)
    third_distance = 0.0 if error else (1.0 if success else 0.25)
    assert diagnostics["charts/reward_distance_mean"] == pytest.approx((0.75 + 0.5 + third_distance) / 4)
    assert diagnostics["charts/reward_success_mean"] == float(success) / 4
    assert sum(
        diagnostics[f"charts/reward_{name}_mean"] for name in ("syntax", "runtime", "distance", "success")
    ) == pytest.approx(diagnostics["charts/reward_mean"])
    assert diagnostics["charts/success_rate"] == float(success) / 4
    assert diagnostics["charts/group_success_rate"] == float(success) / 2
    assert diagnostics["charts/syntax_error_rate"] == 0.25
    assert diagnostics["charts/runtime_error_rate"] == float(error == "runtime_error") / 4
    assert diagnostics["charts/informative_group_fraction"] == 1.0


def test_different_syntax_errors_provide_group_advantages(
    config: Config, state: TrainState, monkeypatch: pytest.MonkeyPatch
) -> None:
    programs = ["m)", "DEF run m( m)"]  # Four edits versus one insertion.
    scripted = np.full((4, 2), KarelProgramEnv.pad_token_id, dtype=np.int32)
    for column, program in enumerate(programs):
        ids = [KarelProgramEnv.token_to_id[token] for token in program.split()]
        scripted[: len(ids), column] = ids
    steps = iter(scripted)

    def scripted_act(logits: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        chex.assert_shape(logits, (2, len(TOKENS)))
        chex.assert_type(logits, jnp.float32)
        chex.assert_shape(key, ())
        return jnp.asarray(next(steps)), jnp.zeros(2, dtype=jnp.float32)

    monkeypatch.setattr("rl2.grpo.act", scripted_act)
    envs = [KarelProgramEnv(config.env) for _ in programs]
    batch, rewards, diagnostics, _ = collect_rollout(state, envs, np.random.default_rng(0), jax.random.key(0), config)
    np.testing.assert_allclose(rewards, [0.2, 0.5])
    np.testing.assert_allclose(batch.advantages, [-1, 1], atol=1e-6)
    assert diagnostics["charts/syntax_error_rate"] == 1.0
    assert diagnostics["charts/informative_group_fraction"] == 1.0
    assert diagnostics["charts/success_rate"] == 0.0


def test_update_changes_policy_and_kl_limit_skips_update(config: Config, state: TrainState) -> None:
    pair = KarelProgramEnv(config.env).reset(seed=3)
    initial = jnp.asarray(np.stack([pair.initial] * 2))
    target = jnp.asarray(np.stack([pair.target] * 2))
    actions = jnp.asarray([[1, 1], [2, 2], [3, 3], [22, 23], [4, 4]], dtype=jnp.int32)
    _, logits = state.apply_fn({"params": state.params}, initial, target, actions[:-1])
    batch = GRPOBatch(
        initial,
        target,
        actions,
        action_log_prob(logits, actions),
        jnp.ones_like(actions, dtype=jnp.bool_),
        jnp.asarray([1.0, -1.0]),
    )
    next_state, metrics = update(state, batch, replace(config, kl_coef=0.01), state.params)
    assert int(next_state.step) == 1
    assert all(np.isfinite(value) for value in metrics)
    assert any(
        not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(state.params), jax.tree.leaves(next_state.params))
    )
    np.testing.assert_allclose(metrics[2], 0, atol=1e-6)
    np.testing.assert_allclose(metrics[4], 0, atol=1e-6)
    rejected, _ = update(state, batch._replace(old_log_probs=batch.old_log_probs - 2), replace(config, target_kl=0.01))
    assert int(rejected.step) == 0
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(rejected)):
        np.testing.assert_array_equal(before, after)


def test_sampled_log_probs_match_teacher_forcing(config: Config, state: TrainState) -> None:
    # A zero head makes all probabilities identical, hiding shifts/state bugs.
    head = state.params["head"]
    nonzero_head = {**head, "kernel": jax.random.normal(jax.random.key(20), head["kernel"].shape) * 0.1}
    state = state.replace(params={**state.params, "head": nonzero_head})
    envs = [KarelProgramEnv(config.env) for _ in range(config.num_tasks * config.group_size)]
    batch, _, _, _ = collect_rollout(state, envs, np.random.default_rng(9), jax.random.key(9), config)
    _, logits = state.apply_fn({"params": state.params}, batch.initial, batch.target, batch.actions[:-1])
    recomputed = np.asarray(action_log_prob(logits, batch.actions))
    assert np.ptp(batch.old_log_probs[batch.mask]) > 0
    np.testing.assert_allclose(recomputed[batch.mask], batch.old_log_probs[batch.mask], rtol=3e-5, atol=3e-6)


def test_no_reward_variation_gives_no_policy_gradient(config: Config) -> None:
    logits = jax.random.normal(jax.random.key(0), (3, 2, len(TOKENS)))
    batch = batch_from_logits(logits, jnp.ones((3, 2), dtype=jnp.bool_), jnp.zeros(2, dtype=jnp.float32))

    def loss_fn(values: jax.Array) -> jax.Array:
        return objective(values, batch, config)[0]

    np.testing.assert_array_equal(jax.grad(loss_fn)(logits), 0)


def test_pad_is_never_sampled_even_with_largest_logit() -> None:
    logits = jnp.zeros((256, len(TOKENS)), dtype=jnp.float32).at[:, KarelProgramEnv.pad_token_id].set(1e6)
    actions, log_probs = act(logits, jax.random.key(19))
    assert not np.any(np.asarray(actions) == KarelProgramEnv.pad_token_id)
    np.testing.assert_allclose(log_probs, -np.log(len(TOKENS) - 1), rtol=1e-6)


def test_pad_labels_and_logits_have_no_loss_gradient(config: Config) -> None:
    config = replace(config, entropy_coef=0.1, kl_coef=0.1)
    logits = jax.random.normal(jax.random.key(7), (3, 2, len(TOKENS)))
    mask = jnp.asarray([[True, True], [False, True], [False, True]])
    batch = batch_from_logits(logits, mask, jnp.asarray([1.0, -1.0]))
    # PAD labels remain ignored even if a caller supplies an all-true mask.
    batch = batch._replace(mask=jnp.ones_like(mask))
    reference = action_log_prob(logits + 0.2, batch.actions)

    def loss_fn(values: jax.Array) -> jax.Array:
        chex.assert_equal_shape((values, logits))
        chex.assert_type(values, jnp.float32)
        return objective(values, batch, config, reference)[0]

    value, grads = jax.value_and_grad(loss_fn)(logits)
    assert np.isfinite(value) and np.isfinite(grads).all()
    np.testing.assert_array_equal(grads[1:, 0], 0)
    np.testing.assert_array_equal(grads[..., KarelProgramEnv.pad_token_id], 0)
    # The actual terminal m) prediction still contributes a policy gradient.
    assert float(grads[0, 0, KarelProgramEnv.terminal_token_id]) != 0
    changed = logits.at[..., KarelProgramEnv.pad_token_id].set(1e6)
    np.testing.assert_allclose(loss_fn(changed), value, atol=1e-6)


def test_config_and_lr_schedule(config: Config) -> None:
    loaded = load_config(Path(__file__).parents[1] / "configs/grpo_karel.yaml")
    assert isinstance(loaded.env, KarelConfig)
    assert isinstance(loaded.conv_channels, tuple)
    schedule = learning_rate_schedule(config)
    np.testing.assert_allclose(schedule(0), config.learning_rate)
    np.testing.assert_allclose(schedule(config.total_updates), 0)
    for options in ({"group_size": 1}, {"num_minibatches": 3}, {"total_updates": 0}, {"target_kl": -1}):
        with pytest.raises((AssertionError, ValueError)):
            replace(config, **options)


def test_one_rollout_training_smoke(config: Config, tmp_path: Path) -> None:
    result = train(replace(config, log_dir=str(tmp_path)))
    assert int(result.step) == 1
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(result.params))
    assert list(tmp_path.glob("karel_grpo_*/events.out.tfevents.*"))
