from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import serialization
from flax.training.train_state import TrainState

from rl2 import karel
from rl2 import train_karel_ast_grpo as grpo
from rl2.karel import REWARD_COMPONENTS, TOKEN_TO_ID, KarelConfig, KarelProgramEnv, _parse
from rl2.karel_ast import ACTION_ID, AST_ACTIONS, ASTFeatures, KarelAST, batch_features
from rl2.train_karel_ast_grpo import (
    Config,
    GRPOBatch,
    action_log_prob,
    collect_rollout,
    create_state,
    format_group_programs,
    group_advantages,
    learning_rate_schedule,
    load_config,
    load_model,
    objective,
    train,
    update,
)


@pytest.fixture(scope="module")
def config() -> Config:
    return Config(
        total_updates=1,
        num_tasks=1,
        group_size=2,
        num_minibatches=1,
        update_epochs=1,
        d_model=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        max_nodes=12,
        max_depth=8,
        entropy_coef=0.01,
        target_kl=None,
        log_interval=1,
        log_program_interval=1,
        env=KarelConfig(height=3, width=3, max_depth=0, max_statements=1, max_program_tokens=8),
    )


@pytest.fixture(scope="module")
def state(config: Config) -> TrainState:
    initial, target = KarelProgramEnv(config.env).reset(seed=42)
    state = create_state(config, initial[None], target[None])
    # Nonuniform logits expose replay/masking/gradient mistakes hidden by a zero head.
    params = dict(state.params)
    for name in ("constructor_head", "value_head"):
        params[name] = {
            **params[name],
            "kernel": 0.1 * jax.random.normal(jax.random.key(5), params[name]["kernel"].shape),
        }
    return state.replace(params=params)


@pytest.fixture(scope="module")
def rollout(config: Config, state: TrainState) -> tuple[GRPOBatch, np.ndarray, dict[str, float], jax.Array]:
    envs = [KarelProgramEnv(config.env) for _ in range(config.group_size)]
    return collect_rollout(state, envs, np.random.default_rng(4), jax.random.key(3), config)


def test_rollout_samples_task_and_distance_map_once_per_group(
    config: Config, state: TrainState, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(config, num_tasks=2, group_size=3)
    sampler = MagicMock(wraps=karel.sample_task)
    distance_map = MagicMock(wraps=karel.target_distance_map)
    monkeypatch.setattr(karel, "sample_task", sampler)
    monkeypatch.setattr(karel, "target_distance_map", distance_map)
    envs = [KarelProgramEnv(config.env) for _ in range(config.num_tasks * config.group_size)]
    batch, _, diagnostics, _ = collect_rollout(state, envs, np.random.default_rng(4), jax.random.key(3), config)
    assert sampler.call_count == distance_map.call_count == config.num_tasks
    stats = [env.sampling_stats for env in envs[:: config.group_size]]
    assert diagnostics["sampling/easy_fraction"] == 1
    assert diagnostics["sampling/navigation_fraction"] == 0
    assert "sampling/navigation_attempts_mean" not in diagnostics
    assert diagnostics["sampling/easy_attempts_mean"] == pytest.approx(np.mean([item.attempts for item in stats]))
    assert diagnostics["sampling/easy_acceptance_rate"] == pytest.approx(
        config.num_tasks / sum(item.attempts for item in stats)
    )
    assert diagnostics["sampling/easy_seconds_mean"] >= 0
    seeds = np.random.default_rng(4).integers(0, 2**31, size=config.num_tasks)
    for group, seed in enumerate(seeds):
        expected = KarelProgramEnv(config.env).reset(seed=int(seed))
        for index in range(group * config.group_size, (group + 1) * config.group_size):
            np.testing.assert_array_equal(batch.initial[index], expected.initial)
            np.testing.assert_array_equal(batch.target[index], expected.target)


def replay_logits(state: TrainState, batch: GRPOBatch, params: optax.Params | None = None) -> jax.Array:
    variables = {"params": state.params if params is None else params}

    def step(tree: ASTFeatures) -> jax.Array:
        return state.apply_fn(variables, batch.initial, batch.target, tree)

    return jax.vmap(step)(jax.tree.map(jnp.asarray, batch.tree))


def synthetic_batch(logits: jax.Array, mask: jax.Array, advantages: jax.Array) -> GRPOBatch:
    chex.assert_type(logits, jnp.float32)
    chex.assert_shape(mask, logits.shape[:-1])
    chex.assert_type(mask, jnp.bool_)
    chex.assert_shape(advantages, (logits.shape[1],))
    chex.assert_type(advantages, jnp.float32)
    actions = jnp.where(mask, ACTION_ID["move"], 0).astype(jnp.int32)
    initial = jnp.zeros((logits.shape[1], 3, 3, 6), jnp.int32)
    tree = batch_features(tuple(KarelAST.empty(4, 2).features() for _ in range(logits.shape[1])))
    snapshots = ASTFeatures(*(jnp.broadcast_to(x, (logits.shape[0], *x.shape)) for x in tree))
    return GRPOBatch(initial, initial, snapshots, actions, action_log_prob(logits, actions), mask, advantages)


def test_group_advantages() -> None:
    rewards = jnp.asarray([[0, 1, 0, 1], [0.1, 0.1, 0.1, 0.1]], jnp.float32)
    np.testing.assert_allclose(group_advantages(rewards), [[-1, 1, -1, 1], [0, 0, 0, 0]])


@pytest.mark.parametrize("group_size", [8, 16, 32, 128])
def test_fractional_equal_rewards_have_zero_advantages(group_size: int) -> None:
    rewards = jnp.broadcast_to(jnp.asarray([0.1, 1 / 3, -0.7], jnp.float32)[:, None], (3, group_size))
    np.testing.assert_array_equal(group_advantages(rewards), 0)


def test_objective_program_weighting_padding_entropy_and_clipping(config: Config) -> None:
    logits = jnp.zeros((4, 2, 4, len(AST_ACTIONS)), jnp.float32)
    mask = jnp.zeros((4, 2, 4), jnp.bool_).at[0, 0, 0].set(True)
    mask = mask.at[:, 1, 0].set(True).at[0, 1, 1].set(True)
    batch = synthetic_batch(logits, mask, jnp.asarray([1.0, -1.0], jnp.float32))
    config = replace(config, entropy_coef=0)
    loss, metrics = objective(logits, batch, config)
    np.testing.assert_allclose(loss, 0, atol=1e-6)
    np.testing.assert_allclose(metrics[1], np.log(len(AST_ACTIONS) - 1), rtol=1e-6)
    padded = batch._replace(old_log_probs=jnp.where(mask, batch.old_log_probs, -1000.0))
    changed = logits.at[1:, 0, :, ACTION_ID["move"]].set(30)

    def loss_fn(values: jax.Array) -> jax.Array:
        chex.assert_shape(values, logits.shape)
        chex.assert_type(values, jnp.float32)
        return objective(values, padded, config)[0]

    np.testing.assert_allclose(loss_fn(changed), loss, atol=1e-6)
    gradients = jax.grad(loss_fn)(changed)
    assert np.isfinite(gradients).all()
    np.testing.assert_array_equal(gradients[1:, 0], 0)
    ratios = jnp.asarray([[[2.0], [0.5]]], jnp.float32)
    clipped = batch._replace(old_log_probs=batch.old_log_probs - jnp.log(ratios))
    loss, metrics = objective(logits, clipped, config)
    np.testing.assert_allclose(loss, -config.clip_coef, atol=1e-6)
    np.testing.assert_allclose(metrics[3], 1.0, atol=1e-6)


def test_rollout_pairs_masks_replay_and_exact_environment_rewards(
    config: Config, state: TrainState, rollout: tuple
) -> None:
    batch, rewards, diagnostics, _ = rollout
    np.testing.assert_array_equal(batch.initial[0], batch.initial[1])
    np.testing.assert_array_equal(batch.target[0], batch.target[1])
    np.testing.assert_array_equal(batch.mask, batch.actions != 0)
    logits = replay_logits(state, batch)
    np.testing.assert_allclose(
        np.asarray(action_log_prob(logits, batch.actions))[batch.mask], batch.old_log_probs[batch.mask], atol=2e-6
    )
    assert (batch.mask.sum(axis=-1) > 1).any()
    assert diagnostics["charts/generation_rounds_mean"] < diagnostics["charts/episode_length_mean"]
    assert diagnostics["charts/syntax_error_rate"] == diagnostics["charts/truncation_rate"] == 0
    assert sum(diagnostics[f"charts/reward_{name}_mean"] for name in REWARD_COMPONENTS) == pytest.approx(
        diagnostics["charts/reward_mean"]
    )
    # Reproduce reset seeds and score the same printed sources through the existing env.
    seeds = np.repeat(np.random.default_rng(4).integers(0, 2**31, size=config.num_tasks), config.group_size)
    successes = []
    for column, seed in enumerate(seeds):
        tree = KarelAST.empty(config.max_nodes, config.max_depth, config.env.max_program_tokens)
        for t in np.flatnonzero(batch.mask[:, column].any(axis=-1)):
            for expected, stored in zip(tree.features(), batch.tree):
                np.testing.assert_array_equal(stored[t, column], expected)
            tree = tree.expand_round(batch.actions[t, column])
        assert tree.complete and len(tree.tokens()) <= config.env.max_program_tokens
        _parse(tree.tokens())
        env = KarelProgramEnv(config.env)
        env.reset(seed=int(seed))
        for token in tree.tokens():
            _, reward, terminated, truncated, info = env.step(TOKEN_TO_ID[token])
        assert terminated and not truncated
        np.testing.assert_allclose(rewards[column], reward, rtol=1e-6)
        successes.append(info["success"])
    assert diagnostics["charts/success_rate"] == np.mean(successes)
    assert diagnostics["charts/group_success_rate"] == float(any(successes))


def test_single_pass_update_matches_roundwise_gradient(config: Config, state: TrainState, rollout: tuple) -> None:
    batch = rollout[0]._replace(advantages=jnp.asarray([1.0, -1.0], jnp.float32))

    def full_loss(params: optax.Params) -> tuple[jax.Array, tuple]:
        return objective(replay_logits(state, batch, params), batch, config)

    (_, expected_metrics), grads = jax.jit(jax.value_and_grad(full_loss, has_aux=True))(state.params)
    expected = state.apply_gradients(grads=grads)
    calls: list[int] = []
    packed, _ = grpo.pack_replay(batch)
    rows = packed.actions.shape[1]

    def apply(variables: dict, initial: jax.Array, target: jax.Array, tree: ASTFeatures) -> jax.Array:
        chex.assert_shape(initial, (rows, *batch.initial.shape[1:]))
        chex.assert_equal_shape((initial, target))
        chex.assert_type((initial, target), jnp.int32)
        calls.append(initial.shape[0])
        return state.apply_fn(variables, initial, target, tree)

    actual, metrics = update(state.replace(apply_fn=apply), batch, config)
    assert calls == [rows]
    np.testing.assert_allclose(metrics, expected_metrics, atol=2e-6, rtol=2e-5)
    for a, b in zip(jax.tree.leaves(actual.params), jax.tree.leaves(expected.params)):
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)
    chex.assert_trees_all_close(actual.opt_state, expected.opt_state, atol=2e-6, rtol=2e-5)
    assert int(actual.step) == 1


@pytest.mark.parametrize(
    "required,capacity,expected",
    [
        (0, 1088, 272),
        (272, 1088, 272),
        (273, 1088, 544),
        (544, 1088, 544),
        (545, 1088, 816),
        (816, 1088, 816),
        (817, 1088, 1088),
        (1088, 1088, 1088),
        (0, 18, 5),
        (6, 18, 10),
        (11, 18, 15),
        (16, 18, 18),
        (18, 18, 18),
        (0, 1, 1),
        (1, 1, 1),
        (0, 3, 1),
        (3, 3, 3),
    ],
)
def test_bucket_boundaries(required: int, capacity: int, expected: int) -> None:
    assert grpo.bucket_size(required, capacity) == expected


@pytest.mark.parametrize("capacity", [1, 3, 5, 18, 40, 1088])
def test_row_buckets_use_fixed_step_and_cover_capacity(capacity: int) -> None:
    buckets = sorted({grpo.bucket_size(required, capacity) for required in range(capacity + 1)})
    step = (capacity + 3) // 4
    assert len(buckets) <= 4
    assert buckets[-1] == capacity
    assert buckets[:-1] == [step * index for index in range(1, len(buckets))]
    for required in range(capacity + 1):
        assert max(1, required) <= grpo.bucket_size(required, capacity) <= capacity


@pytest.mark.parametrize("required,capacity", [(-1, 16), (17, 16), (0, 0)])
def test_invalid_bucket_sizes(required: int, capacity: int) -> None:
    with pytest.raises(AssertionError):
        grpo.bucket_size(required, capacity)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_invalid_compile_logging_flag(config: Config, value: object) -> None:
    with pytest.raises(TypeError, match="log_compiles must be a boolean"):
        replace(config, log_compiles=value)


@pytest.mark.parametrize("empty", [False, True])
def test_pack_replay_removes_padding_and_preserves_program_weights(rollout: tuple, empty: bool) -> None:
    batch = rollout[0]
    rounds, programs, nodes = batch.actions.shape

    def pad(value: grpo.Array) -> np.ndarray:
        return np.pad(value, ((0, 32 - rounds), (0, 0), (0, 64 - nodes)) + ((0, 0),) * (value.ndim - 3))

    batch = batch._replace(
        tree=ASTFeatures(*(pad(field) for field in batch.tree)),
        actions=pad(batch.actions),
        old_log_probs=pad(batch.old_log_probs),
        mask=pad(batch.mask) if not empty else np.zeros((32, programs, 64), np.bool_),
        advantages=np.arange(programs, dtype=np.float32),
    )
    packed, weights = grpo.pack_replay(batch)
    live = np.flatnonzero(batch.mask.any(axis=-1).reshape(-1))
    assert packed.actions.shape == (1, grpo.bucket_size(len(live), 32 * programs), 32)
    assert packed.actions.shape[1] < 32 * programs
    assert not packed.mask[:, len(live) :].any()
    assert not weights[:, len(live) :].any()
    assert not packed.tree.node_mask[:, len(live) :].any()
    for position, index in enumerate(live):
        round_id, program_id = divmod(index, programs)
        np.testing.assert_array_equal(packed.actions[0, position], batch.actions[round_id, program_id, :32])
        np.testing.assert_array_equal(packed.initial[position], batch.initial[program_id])
        assert packed.advantages[position] == batch.advantages[program_id]
        expected = batch.mask[round_id, program_id, :32] / batch.mask[:, program_id].sum() / programs
        np.testing.assert_allclose(weights[0, position], expected)
    assert weights.sum() == pytest.approx(0 if empty else 1)


def test_empty_replay_has_zero_metrics_and_finite_update(config: Config, state: TrainState, rollout: tuple) -> None:
    batch = rollout[0]._replace(mask=np.zeros_like(rollout[0].mask))
    updated, metrics = update(state, batch, config)
    np.testing.assert_array_equal(metrics, 0)
    chex.assert_trees_all_close(updated.params, state.params)
    assert int(updated.step) == int(state.step) + 1


@pytest.mark.parametrize(
    "required,capacity,expected",
    [
        (0, 128, 32),
        (1, 128, 32),
        (32, 128, 32),
        (33, 128, 64),
        (64, 128, 64),
        (65, 128, 96),
        (96, 128, 96),
        (97, 128, 128),
        (128, 128, 128),
        (129, 160, 160),
        (1, 24, 24),
        (97, 100, 100),
    ],
)
def test_tree_bucket_keeps_resolved_context_nodes(required: int, capacity: int, expected: int) -> None:
    features = batch_features((KarelAST.empty(capacity, 8).features(),))
    present = np.zeros((1, capacity), np.bool_)
    present[:, :required] = True
    features = features._replace(node_mask=present)
    trimmed = grpo.bucket_tree(features)
    assert trimmed.node_mask.shape == (1, expected)
    np.testing.assert_array_equal(trimmed.node_mask, present[:, :expected])


def test_compilation_logs_only_on_new_update_trace(
    config: Config, state: TrainState, rollout: tuple, capsys: pytest.CaptureFixture[str]
) -> None:
    config = replace(config, log_compiles=True)
    original_apply = state.apply_fn

    # Unique apply function gives this test its own JIT cache entry.
    def apply(variables: dict, initial: jax.Array, target: jax.Array, tree: ASTFeatures) -> jax.Array:
        chex.assert_equal_shape((initial, target))
        chex.assert_type((initial, target), jnp.int32)
        return original_apply(variables, initial, target, tree)

    state = state.replace(apply_fn=apply)
    update(state, rollout[0], config)
    update(state, rollout[0], config)
    output = capsys.readouterr()
    assert output.out.count("JIT trace update: bucket_shape=") == 1
    assert not output.err
    empty = rollout[0]._replace(mask=np.zeros_like(rollout[0].mask))
    # Explicitly use another valid node bucket to exercise a shape cache miss.
    packed, weights = grpo.pack_replay(empty)
    packed = packed._replace(
        tree=ASTFeatures(*(field[:, :, :4] for field in packed.tree)),
        actions=packed.actions[:, :, :4],
        old_log_probs=packed.old_log_probs[:, :, :4],
        mask=packed.mask[:, :, :4],
    )
    grpo._update(state, packed, weights[:, :, :4], config)
    output = capsys.readouterr()
    assert output.out == f"JIT trace update: bucket_shape=({packed.actions.shape[1]}, 4)\n"
    assert not output.err
    grpo._update(state, packed, weights[:, :, :4], replace(config, log_compiles=False))
    assert not capsys.readouterr().out


def test_prediction_prints_bucket_shape_only_on_new_trace(
    state: TrainState, rollout: tuple, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = rollout[0]
    original_apply = state.apply_fn

    def apply(variables: dict, initial: jax.Array, target: jax.Array, tree: ASTFeatures) -> jax.Array:
        chex.assert_equal_shape((initial, target))
        chex.assert_type((initial, target), jnp.int32)
        return original_apply(variables, initial, target, tree)

    state = state.replace(apply_fn=apply)
    tree = ASTFeatures(*(field[0] for field in batch.tree))
    for _ in range(2):
        grpo.predict(state, batch.initial, batch.target, tree, log_compiles=True)
    output = capsys.readouterr()
    assert output.out == f"JIT trace predict: bucket_shape={tree.node_mask.shape}\n"
    assert not output.err
    trimmed = ASTFeatures(*(field[:, :4] for field in tree))
    grpo.predict(state, batch.initial, batch.target, trimmed, log_compiles=True)
    assert capsys.readouterr().out == f"JIT trace predict: bucket_shape={trimmed.node_mask.shape}\n"
    grpo.predict(state, batch.initial, batch.target, trimmed, log_compiles=False)
    assert not capsys.readouterr().out


def test_padding_and_rejected_kl_do_not_update_state(config: Config, state: TrainState, rollout: tuple) -> None:
    batch = rollout[0]
    padded = batch._replace(old_log_probs=np.where(batch.mask, batch.old_log_probs, -10000.0).astype(np.float32))
    one, m1 = update(state, batch, config)
    two, m2 = update(state, padded, config)
    chex.assert_trees_all_close(one, two)
    np.testing.assert_allclose(m1, m2)
    rejected = batch._replace(old_log_probs=batch.old_log_probs - 2.0)
    result, _ = update(state, rejected, replace(config, target_kl=0.001))
    chex.assert_trees_all_equal(result, state)


@pytest.mark.parametrize("interval", [0, -1, 1.5, True])
def test_invalid_log_interval(config: Config, interval: float | bool) -> None:
    with pytest.raises((AssertionError, TypeError)):
        replace(config, log_interval=interval)


def test_formatted_samples_replay_ast_source(config: Config, rollout: tuple) -> None:
    batch, rewards, _, _ = rollout
    output = format_group_programs(batch, rewards, config=config, group_size=2, group_index=0, count=2)
    assert "same initial/target pair" in output
    assert "actions=" in output and "tokens=" in output
    for column, source in enumerate(output.split("```text\n")[1:]):
        printed = tuple(source.split("```")[0].split())
        tree = KarelAST.empty(config.max_nodes, config.max_depth, config.env.max_program_tokens)
        for step in np.flatnonzero(batch.mask[:, column].any(axis=-1)):
            tree = tree.expand_round(batch.actions[step, column])
        assert printed == tree.tokens()
        assert "\n  " in source


def test_configuration_and_schedule(config: Config, tmp_path: Path) -> None:
    loaded = load_config(Path(__file__).resolve().parents[1] / "configs/karel_ast_grpo.yaml")
    assert loaded.max_nodes == 128
    assert not hasattr(loaded, "decision_batch_size")
    assert loaded.bf16 and loaded.attention_implementation == "cudnn"
    # Model size is a tunable experiment setting; require cuDNN-compatible
    # head dimensions rather than pinning the example to one architecture.
    assert loaded.d_model > 0 and loaded.num_layers > 0 and loaded.num_heads > 0
    assert loaded.d_model % (8 * loaded.num_heads) == 0
    assert loaded.env.success_weight == 1.0
    weighted = replace(
        loaded, env=replace(loaded.env, syntax_weight=0.25, runtime_weight=0, distance_weight=2.5, success_weight=2.0)
    )
    config_path = tmp_path / "weighted.yaml"
    config_path.write_text(grpo.yaml.safe_dump(grpo.asdict(weighted)))
    assert load_config(config_path) == weighted
    # Older saved run configs still load for inference/resume.
    config_path.write_text(grpo.yaml.safe_dump({**grpo.asdict(weighted), "decision_batch_size": 32}))
    assert load_config(config_path) == weighted
    schedule = learning_rate_schedule(config)
    np.testing.assert_allclose(schedule(0), config.learning_rate)
    np.testing.assert_allclose(schedule(config.total_updates), 0.0)


@pytest.mark.parametrize("success_weight", [0.0, 2.0])
def test_rollout_logs_weighted_reward_contributions(
    config: Config,
    state: TrainState,
    rollout: tuple[GRPOBatch, np.ndarray, dict[str, float], jax.Array],
    success_weight: float,
) -> None:
    config = replace(
        config,
        env=replace(
            config.env, syntax_weight=0.25, runtime_weight=0, distance_weight=2.5, success_weight=success_weight
        ),
    )
    envs = [KarelProgramEnv(config.env) for _ in range(config.group_size)]
    batch, rewards, diagnostics, _ = collect_rollout(state, envs, np.random.default_rng(4), jax.random.key(3), config)
    np.testing.assert_array_equal(batch.actions, rollout[0].actions)
    for name in REWARD_COMPONENTS:
        weight = {"syntax": 0.25, "runtime": 0, "distance": 2.5, "success": success_weight}.get(name, 1.0)
        tag = f"charts/reward_{name}_mean"
        assert diagnostics[tag] == pytest.approx(weight * rollout[2][tag])
    assert diagnostics["charts/success_rate"] == rollout[2]["charts/success_rate"]
    assert rewards.mean() == pytest.approx(sum(diagnostics[f"charts/reward_{name}_mean"] for name in REWARD_COMPONENTS))


@pytest.mark.skipif(not any(d.platform == "gpu" for d in jax.devices()), reason="cuDNN needs NVIDIA GPU")
def test_cudnn_ast_grpo(config: Config) -> None:
    config = replace(config, bf16=True, attention_implementation="cudnn")
    envs = [KarelProgramEnv(config.env) for _ in range(2)]
    initial, target = envs[0].reset(seed=1)
    state = create_state(config, initial[None], target[None])
    batch, _, _, _ = collect_rollout(state, envs, np.random.default_rng(1), jax.random.key(1), config)
    state, metrics = update(state, batch, config)
    assert all(np.isfinite(x) for x in metrics)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(state.params))


@pytest.mark.parametrize("run_id", ["", " ", ".", "..", "../run", "foo/bar", "foo\\bar", 1])
def test_invalid_run_id(config: Config, run_id: str | int) -> None:
    with pytest.raises((ValueError, TypeError)):
        replace(config, run_id=run_id)


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan")])
def test_invalid_checkpoint_interval(config: Config, interval: float) -> None:
    with pytest.raises((AssertionError, ValueError)):
        replace(config, checkpoint_interval_seconds=interval)


def test_legacy_weights_load_for_inference_but_cannot_resume(config: Config, state: TrainState, tmp_path: Path) -> None:
    config = replace(config, log_dir=str(tmp_path), run_id="legacy")
    directory = tmp_path / "legacy"
    directory.mkdir()
    (directory / "config.yaml").write_text(grpo.yaml.safe_dump(grpo.asdict(config)))
    (directory / "params.msgpack").write_bytes(serialization.to_bytes(state.params))
    _, loaded = load_model(directory)
    chex.assert_trees_all_equal(loaded.params, state.params)
    with pytest.raises(ValueError, match="only inference weights"):
        train(config)
    assert not (directory / "checkpoint.msgpack").exists()


def test_legacy_reference_checkpoint_and_config_load_without_reference(
    config: grpo.Config, state: TrainState, tmp_path: Path
) -> None:
    rng = np.random.default_rng(12)
    progress = grpo.TrainingProgress(state, jax.random.key(3), 2, 8, 1, 8)
    grpo._save_checkpoint(str(tmp_path), progress, rng)
    payload = serialization.msgpack_restore((tmp_path / "checkpoint.msgpack").read_bytes())
    assert "reference_params" not in payload
    assert "kl_coef" not in grpo.asdict(config)
    # Older checkpoints may contain a separate reference with different weights.
    payload["reference_params"] = {"obsolete_reference": np.zeros(1, np.float32)}
    restored_rng = np.random.default_rng(99)
    restored = grpo._restore_checkpoint(serialization.msgpack_serialize(payload), state, restored_rng)
    chex.assert_trees_all_equal(restored, progress)
    np.testing.assert_array_equal(restored_rng.integers(100, size=4), rng.integers(100, size=4))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(grpo.yaml.safe_dump({**grpo.asdict(config), "kl_coef": 0.25}))
    assert grpo.load_config(config_path) == config


def test_load_model_from_checkpoint(config: Config, state: TrainState, tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(grpo.yaml.safe_dump(grpo.asdict(config)))
    progress = grpo.TrainingProgress(state, jax.random.key(3), 2, 8, 1, 8)
    grpo._save_checkpoint(str(tmp_path), progress, np.random.default_rng(4))
    loaded_config, loaded = grpo.load_model(tmp_path, attention_implementation="xla")
    assert loaded_config == config
    chex.assert_trees_all_equal(loaded.params, state.params)
    assert int(loaded.step) == 0
