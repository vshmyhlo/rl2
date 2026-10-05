from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization
from flax.training.train_state import TrainState
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rl2 import karel
from rl2 import train_karel_grpo as grpo
from rl2.karel import REWARD_COMPONENTS, TOKEN_TO_ID, TOKENS, KarelConfig, KarelProgramEnv, _parse
from rl2.karel_grammar import mask_sequence_logits

type Rollout = tuple[grpo.GRPOBatch, np.ndarray, dict[str, float], jax.Array]


@pytest.fixture(scope="module")
def config() -> grpo.Config:
    return grpo.Config(
        total_updates=1,
        num_tasks=1,
        group_size=4,
        num_minibatches=2,
        update_epochs=1,
        d_model=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        entropy_coef=0.01,
        target_kl=None,
        log_interval=1,
        log_program_interval=1,
        env=KarelConfig(height=3, width=3, max_depth=0, max_statements=1, max_program_tokens=12),
    )


@pytest.fixture(scope="module")
def state(config: grpo.Config) -> TrainState:
    pair = KarelProgramEnv(config.env).reset(seed=42)
    state = grpo.create_state(config, pair.initial[None], pair.target[None])
    params = dict(state.params)
    params["head"] = {
        **params["head"],
        "kernel": 0.1 * jax.random.normal(jax.random.key(5), params["head"]["kernel"].shape),
    }
    return state.replace(params=params)


@pytest.fixture(scope="module")
def rollout(config: grpo.Config, state: TrainState) -> Rollout:
    return grpo.collect_rollout(
        state,
        [KarelProgramEnv(config.env) for _ in range(config.group_size)],
        np.random.default_rng(4),
        jax.random.key(3),
        config,
    )


def test_rollout_replay_and_environment_rewards(config: grpo.Config, state: TrainState, rollout: Rollout) -> None:
    batch, rewards, diagnostics, _ = rollout
    _, logits = state.apply_fn({"params": state.params}, batch.initial, batch.target, batch.actions[:-1])
    masked = mask_sequence_logits(logits, batch.actions, config.env.max_program_tokens)
    replay = grpo.action_log_prob(masked, batch.actions)
    np.testing.assert_allclose(replay, batch.old_log_probs, atol=2e-6)
    assert np.isfinite(batch.old_log_probs).all()
    np.testing.assert_array_equal(batch.old_log_probs[~batch.mask], 0)
    np.testing.assert_array_equal(batch.mask, batch.actions != 0)
    assert diagnostics["charts/syntax_error_rate"] == diagnostics["charts/truncation_rate"] == 0
    seeds = np.random.default_rng(4).integers(0, 2**31, size=config.num_tasks)
    for index in range(config.group_size):
        env = KarelProgramEnv(config.env)
        pair = env.reset(seed=int(seeds[0]))
        np.testing.assert_array_equal(batch.initial[index], pair.initial)
        np.testing.assert_array_equal(batch.target[index], pair.target)
        program = batch.actions[:, index][batch.mask[:, index]]
        assert program[-1] == KarelProgramEnv.terminal_token_id
        assert not batch.mask[len(program) :, index].any()
        _parse([TOKENS[token] for token in program])
        for token in program:
            _, reward, terminated, truncated, _ = env.step(token)
        assert terminated and not truncated
        assert rewards[index] == pytest.approx(reward)
    assert rewards.mean() == pytest.approx(sum(diagnostics[f"charts/reward_{name}_mean"] for name in REWARD_COMPONENTS))
    _, metrics = grpo.objective(logits, batch, config)
    assert float(metrics[2]) < 1e-6


def test_one_task_sample_per_group(config: grpo.Config, state: TrainState, monkeypatch: pytest.MonkeyPatch) -> None:
    config = replace(config, num_tasks=2)
    sampler = MagicMock(wraps=karel.sample_task)
    monkeypatch.setattr(karel, "sample_task", sampler)
    grpo.collect_rollout(
        state,
        [KarelProgramEnv(config.env) for _ in range(config.num_tasks * config.group_size)],
        np.random.default_rng(1),
        jax.random.key(1),
        config,
    )
    assert sampler.call_count == config.num_tasks
    with pytest.raises(ValueError, match="environments"):
        grpo.collect_rollout(state, [], np.random.default_rng(1), jax.random.key(1), config)


@pytest.mark.parametrize("budget", [5, 12, 32])
def test_generation_finishes_at_budget_and_is_deterministic(config: grpo.Config, budget: int) -> None:
    config = replace(config, env=replace(config.env, max_program_tokens=budget))
    grids = jnp.zeros((2, 3, 3, 6), jnp.int32)
    state = grpo.create_state(config, grids, grids)
    # Favor branching and continuation to exercise reservation of closing tokens.
    bias = state.params["head"]["bias"].at[TOKEN_TO_ID["REPEAT"]].set(5)
    bias = bias.at[TOKEN_TO_ID["m)"]].set(-20)
    state = state.replace(params={**state.params, "head": {**state.params["head"], "bias": bias}})
    first = grpo.generate(state, grids, grids, jax.random.key(10), budget)
    second = grpo.generate(state, grids, grids, jax.random.key(10), budget)
    chex.assert_trees_all_equal(first, second)
    assert not np.array_equal(jax.random.key_data(first[2]), jax.random.key_data(jax.random.key(10)))
    for column in np.asarray(first[0]).T:
        tokens = [TOKENS[token] for token in column if token]
        assert len(tokens) == budget
        assert tokens[-1] == "m)"
        _parse(tokens)
    _, logits = state.apply_fn({"params": state.params}, grids, grids, first[0][:-1])
    replay = grpo.action_log_prob(mask_sequence_logits(logits, first[0], budget), first[0])
    np.testing.assert_allclose(replay, first[1], atol=2e-6)


def objective_batch() -> tuple[jax.Array, grpo.GRPOBatch]:
    programs = ["DEF run m( move m)", "DEF run m( move turnLeft turnRight m)"]
    actions = np.zeros((8, 2), np.int32)
    for index, program in enumerate(programs):
        tokens = [TOKEN_TO_ID[token] for token in program.split()]
        actions[: len(tokens), index] = tokens
    actions = jnp.asarray(actions)
    grids = jnp.zeros((2, 3, 3, 6), jnp.int32)
    logits = jnp.zeros((*actions.shape, len(TOKENS)), jnp.float32)
    old = grpo.action_log_prob(mask_sequence_logits(logits, actions, 12), actions)
    return logits, grpo.GRPOBatch(grids, grids, actions, old, actions != 0, jnp.asarray([1.0, -1.0], jnp.float32))


def test_objective_program_weighting_clipping_and_padding(config: grpo.Config) -> None:
    config = replace(config, entropy_coef=0)
    logits, batch = objective_batch()
    loss, metrics = grpo.objective(logits, batch, config)
    np.testing.assert_allclose(loss, 0, atol=1e-6)
    assert float(metrics[1]) > 0
    ratios = jnp.asarray([[2.0, 0.5]], jnp.float32)
    changed = batch._replace(old_log_probs=batch.old_log_probs - jnp.log(ratios))
    loss, metrics = grpo.objective(logits, changed, config)
    np.testing.assert_allclose(loss, -0.2, atol=1e-6)
    np.testing.assert_allclose(metrics[3], 1.0)
    padded = batch._replace(old_log_probs=jnp.where(batch.mask, batch.old_log_probs, -1000.0))

    def loss_fn(values: jax.Array) -> jax.Array:
        chex.assert_shape(values, logits.shape)
        chex.assert_type(values, jnp.float32)
        return grpo.objective(values, padded, config)[0]

    altered = jnp.where(batch.mask[..., None], logits, 100.0)
    np.testing.assert_allclose(loss_fn(altered), 0, atol=1e-6)
    grads = jax.grad(loss_fn)(altered)
    assert np.isfinite(grads).all()
    np.testing.assert_array_equal(grads[~batch.mask], 0)
    # Forced prefix decisions and grammar-forbidden tokens receive zero gradient.
    np.testing.assert_array_equal(grads[:3], 0)
    np.testing.assert_array_equal(grads[..., TOKEN_TO_ID["<pad>"]], 0)


def test_update_and_old_policy_kl_early_stop(config: grpo.Config, state: TrainState, rollout: Rollout) -> None:
    batch = rollout[0]
    updated, metrics = grpo.update(state, batch, config)
    assert int(updated.step) == int(state.step) + 1
    assert all(np.isfinite(value) for value in metrics)
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(updated.params), jax.tree.leaves(state.params)))
    divergent = batch._replace(old_log_probs=batch.old_log_probs - 2.0)
    rejected, _ = grpo.update(state, divergent, replace(config, target_kl=0.001))
    chex.assert_trees_all_equal(rejected, state)


@pytest.mark.parametrize("bf16", [False, True])
def test_training_checkpoint_resume_and_logs(config: grpo.Config, tmp_path: Path, bf16: bool) -> None:
    config = replace(config, log_dir=str(tmp_path), run_id="test", bf16=bf16, anneal_lr=False)
    state = grpo.train(config)
    assert int(state.step) == config.num_minibatches
    assert all(np.isfinite(x).all() and x.dtype == jnp.float32 for x in jax.tree.leaves(state.params))
    directory = tmp_path / "test"
    loaded_config, loaded = grpo.load_model(directory, attention_implementation="xla")
    assert loaded_config == config
    chex.assert_trees_all_equal(loaded.params, state.params)
    checkpoint = directory / "checkpoint.msgpack"
    original = checkpoint.read_bytes()
    restored = grpo.train(config)
    chex.assert_trees_all_equal(restored.params, state.params)
    assert checkpoint.read_bytes() == original
    resumed = grpo.train(replace(config, total_updates=2))
    expected = grpo.train(replace(config, total_updates=2, run_id="continuous"))
    chex.assert_trees_all_equal((resumed.params, resumed.opt_state), (expected.params, expected.opt_state))
    saved = serialization.msgpack_restore(checkpoint.read_bytes())
    assert saved["iteration"] == 2
    assert "reference_params" not in saved
    assert saved["steps"] == saved["episodes"] == 2 * config.group_size
    events = EventAccumulator(str(directory)).Reload()
    assert [event.step for event in events.Scalars("charts/reward_mean")] == [4, 8]
    assert "samples/generated_programs/text_summary" in events.Tags()["tensors"]
    assert "policy/reference_kl" not in events.Tags()["scalars"]


def test_formatted_samples_preserve_tokens(config: grpo.Config, rollout: Rollout) -> None:
    batch, rewards, _, _ = rollout
    output = grpo.format_group_programs(batch, rewards, config=config, group_size=4, group_index=0, count=2)
    for index, source in enumerate(output.split("```text\n")[1:]):
        expected = [TOKENS[token] for token in batch.actions[:, index][batch.mask[:, index]]]
        assert source.split("```")[0].split() == expected


def test_example_config() -> None:
    config = grpo.load_config(Path(__file__).resolve().parents[1] / "configs/karel_grpo.yaml")
    assert config.bf16 and config.attention_implementation == "cudnn"
    assert not hasattr(config, "max_nodes")
    assert not hasattr(config, "max_depth")
    schedule = grpo.learning_rate_schedule(config)
    np.testing.assert_allclose(schedule(0), config.learning_rate)
    np.testing.assert_allclose(schedule(config.total_updates), 0.0)


@pytest.mark.parametrize(
    "settings",
    [
        {"group_size": 1},
        {"num_minibatches": 3},
        {"d_model": 15},
        {"num_kv_heads": 3},
        {"attention_implementation": "cudnn"},
        {"num_layers": 0},
        {"run_id": "../escape"},
    ],
)
def test_invalid_config(config: grpo.Config, settings: dict[str, Any]) -> None:
    with pytest.raises((ValueError, AssertionError)):
        replace(config, **settings)


@pytest.mark.skipif(not any(d.platform == "gpu" for d in jax.devices()), reason="cuDNN needs NVIDIA GPU")
def test_cudnn_training(config: grpo.Config, tmp_path: Path) -> None:
    state = grpo.train(replace(config, bf16=True, attention_implementation="cudnn", log_dir=str(tmp_path)))
    assert int(state.step) == config.num_minibatches
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(state.params))


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
