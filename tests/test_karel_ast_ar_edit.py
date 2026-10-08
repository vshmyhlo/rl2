from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from flax.training.train_state import TrainState

from rl2 import karel_ast_edit as editing
from rl2 import train_karel_ast_ar_edit as edit
from rl2 import train_karel_ppo_ast_ar_edit as ppo_edit
from rl2.edit_transformer import PAD_EVENT, EditTransformer
from rl2.karel import TOKEN_TO_ID, KarelConfig, KarelPair, KarelProgramEnv
from rl2.karel_ast import ACTION_ID, KarelAST, program_actions


@jax.jit
def predict(state: TrainState, history: edit.History) -> jax.Array:
    """Replay the complete history to verify rollout and training behavior."""
    _, logits = state.apply_fn({"params": state.params}, history)
    return logits


@pytest.fixture(scope="module")
def config() -> edit.Config:
    return edit.Config(
        total_updates=1,
        num_tasks=1,
        group_size=2,
        num_minibatches=1,
        update_epochs=1,
        d_model=16,
        num_layers=1,
        num_heads=2,
        max_nodes=8,
        max_depth=4,
        max_seq_len=14,
        target_kl=None,
        entropy_coef=0.01,
        log_interval=1,
        log_program_interval=1,
        env=KarelConfig(
            height=3, width=3, max_depth=0, max_statements=1, max_program_tokens=8, depth_penalty_weight=0.1
        ),
    )


@pytest.fixture(scope="module")
def state(config: edit.Config) -> TrainState:
    grids = np.zeros((1, 3, 3, 6), np.int32)
    state = edit.create_state(config, grids, grids)
    params = dict(state.params)
    # Nonuniform logits expose replay, cache and causality mistakes.
    params["head"] = {
        **params["head"],
        "kernel": 0.1 * jax.random.normal(jax.random.key(9), params["head"]["kernel"].shape),
    }
    return state.replace(params=params)


@pytest.fixture(scope="module")
def rollout(config: edit.Config, state: TrainState) -> edit.Rollout:
    return edit.collect_rollout(
        state,
        edit.KarelASTEditVectorEnv(config.edit_config, 2),
        np.random.default_rng(4),
        jax.random.key(7),
        config,
    )


def tree_from(source: str, config: edit.Config) -> KarelAST:
    tree = KarelAST.empty(config.max_nodes, config.max_depth, config.env.max_program_tokens)
    for action in program_actions(tuple(source.split())):
        tree = tree.expand(action)
    return tree


def test_single_task_repeats_across_rollouts(config: edit.Config) -> None:
    config = replace(config, max_unique_tasks=1, num_tasks=2)
    rng = np.random.default_rng(4)
    copy = KarelProgramEnv(config.env)
    first = edit.sample_task_groups(rng, config)
    expected = copy.reset_from(first[0])
    for tasks in (first, edit.sample_task_groups(rng, config)):
        assert len(tasks) == config.num_tasks * config.group_size
        for start in range(0, len(tasks), config.group_size):
            assert all(task is tasks[start] for task in tasks[start : start + config.group_size])
        for task in tasks:
            pair = copy.reset_from(task)
            np.testing.assert_array_equal(pair.initial, expected.initial)
            np.testing.assert_array_equal(pair.target, expected.target)
            assert task.reference_program == first[0].reference_program


def test_task_pool_sampling_and_rng_resume(config: edit.Config, monkeypatch: pytest.MonkeyPatch) -> None:
    factory = MagicMock()
    monkeypatch.setattr(edit, "KarelProgramEnv", factory)
    config = replace(config, num_tasks=4, max_unique_tasks=3)
    rng = np.random.default_rng(4)

    def sampled_seeds(settings: edit.Config, generator: np.random.Generator) -> tuple[int, ...]:
        factory.return_value.reset.reset_mock()
        edit.sample_task_groups(generator, settings)
        return tuple(call.kwargs["seed"] for call in factory.return_value.reset.call_args_list)

    first = sampled_seeds(config, rng)
    restored = np.random.default_rng()
    restored.bit_generator.state = rng.bit_generator.state
    second = sampled_seeds(config, rng)
    assert second == sampled_seeds(config, restored)
    assert 1 < len(set(first + second)) <= 3
    assert first == sampled_seeds(config, np.random.default_rng(4))
    assert first != sampled_seeds(replace(config, seed=config.seed + 1), np.random.default_rng(4))
    # Unlimited sampling retains the original RNG sequence and keeps drawing fresh seeds.
    unlimited = replace(config, max_unique_tasks=None)
    expected_rng = np.random.default_rng(4)
    rng = np.random.default_rng(4)
    for _ in range(2):
        assert sampled_seeds(unlimited, rng) == tuple(expected_rng.integers(0, 2**31, size=config.num_tasks))


@pytest.mark.parametrize("trainer", [edit, ppo_edit], ids=["grpo", "ppo"])
def test_collect_rollout_uses_fixed_task_pool(
    config: edit.Config, monkeypatch: pytest.MonkeyPatch, trainer: ModuleType
) -> None:
    config = replace(config, max_unique_tasks=1)
    sample = MagicMock(wraps=edit.sample_task_groups)
    monkeypatch.setattr(edit, "sample_task_groups", sample)
    # Stop at the model boundary: sampling wiring needs no policy execution.
    run = MagicMock(side_effect=RuntimeError("episodes reached"))
    monkeypatch.setattr(trainer, "run_episodes", run)
    with (
        edit.KarelASTEditVectorEnv(config.edit_config, config.num_tasks * config.group_size) as envs,
        pytest.raises(RuntimeError, match="episodes reached"),
    ):
        trainer.collect_rollout(MagicMock(), envs, np.random.default_rng(4), jax.random.key(7), config)
    sample.assert_called_once()
    assert sample.call_args.args[1].max_unique_tasks == 1
    assert len(run.call_args.args[1]) == config.num_tasks * config.group_size


@pytest.mark.parametrize(
    "value,error",
    [("0", ValueError), ("-1", ValueError), ("true", TypeError), ("1.5", TypeError)],
    ids=["zero", "negative", "boolean", "fractional"],
)
def test_load_config_rejects_invalid_task_limit(tmp_path: Path, value: str, error: type[Exception]) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(f"max_unique_tasks: {value}\n")
    with pytest.raises(error, match="max_unique_tasks"):
        edit.load_config(path)


def test_load_config_task_limit(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("max_unique_tasks: 1\n")
    config = edit.load_config(path)
    assert config.max_unique_tasks == 1
    path.write_text(yaml.safe_dump(asdict(config)))
    assert edit.load_config(path) == config
    for contents in ("max_unique_tasks: null\n", "{}\n"):
        path.write_text(contents)
        assert edit.load_config(path).max_unique_tasks is None


def test_replacement_grows_shrinks_and_preserves_context(config: edit.Config) -> None:
    tree = editing.seed_tree(config.edit_config)
    # Replace the final empty list by two statements; fixed prefix survives.
    position = next(p for p, i in enumerate(tree.preorder()) if tree.nodes[i].constructor == ACTION_ID["End"])
    partial = editing.open_subtree(tree, position)
    for name in ("Cons", "turnRight", "Cons", "putMarker", "End"):
        partial = partial.expand(ACTION_ID[name])
    assert partial.complete and len(partial.nodes) == config.max_nodes
    assert partial.tokens() == ("DEF", "run", "m(", "turnLeft", "turnRight", "putMarker", "m)")
    # Shrink the new list tail back to End. Detached descendants must not consume capacity.
    position = next(p for p, i in enumerate(partial.preorder()) if partial.nodes[i].constructor == ACTION_ID["Cons"])
    smaller = editing.open_subtree(partial, position).expand(ACTION_ID["End"])
    assert smaller.tokens() == tree.tokens()
    assert len(smaller.nodes) == len(tree.nodes)
    root = editing.open_subtree(smaller, 0)
    assert len(root.nodes) == 1 and root.nodes[0].is_hole
    with pytest.raises(ValueError, match="complete"):
        editing.open_subtree(root, 0)
    with pytest.raises(ValueError, match="outside"):
        editing.open_subtree(tree, len(tree.nodes))
    with pytest.raises(TypeError, match="integer"):
        editing.open_subtree(tree, True)


@pytest.mark.parametrize(
    "limit",
    [
        pytest.param("nodes", id="node-budget"),
        pytest.param("source", id="source-budget"),
        pytest.param("depth", id="depth-budget"),
    ],
)
def test_replacement_masks_reserve_completion(config: edit.Config, limit: str) -> None:
    changes = {
        "nodes": {"max_nodes": 4},
        "source": {"env": replace(config.env, max_program_tokens=5)},
        "depth": {"max_depth": 2},
    }[limit]
    limited = replace(config, **changes)
    tree = editing.open_subtree(editing.seed_tree(limited.edit_config), 0)
    # Pick the highest legal constructor/value ID to stress nontrivial completions.
    for _ in range(limited.max_nodes):
        legal = tree.allowed_actions()
        if tree.complete:
            break
        assert legal.any() and not legal[0]
        tree = tree.expand(int(np.flatnonzero(legal)[-1]))
    assert tree.complete
    assert len(tree.nodes) <= limited.max_nodes
    assert len(tree.tokens()) <= limited.env.max_program_tokens
    assert max(node.depth for node in tree.nodes) <= limited.max_depth


def test_execution_feedback_and_restart(config: edit.Config) -> None:
    task, env = KarelProgramEnv(config.env), KarelProgramEnv(config.env)
    pair = task.reset(seed=12)
    # A 1-cell enclosed world makes move fail deterministically after a valid turn.
    initial = np.zeros((3, 3, 6), np.int32)
    initial[..., 4] = 1
    initial[1, 1, 4] = 0
    initial[1, 1, 0] = 1
    target = initial.copy()
    target[1, 1, 0] = 0
    target[1, 1, 3] = 1
    task._task = replace(task._task, initial=initial, target=target, distance_map=None)
    from rl2.karel import target_distance_map

    task._distance_map = target_distance_map(target)
    pair = KarelPair(initial, target)
    bad = tree_from("DEF run m( turnLeft move m)", config)
    result = editing.evaluate(bad, task, env, initial)
    assert result.error == "runtime_error" and not result.success and result.ticks == 2
    np.testing.assert_array_equal(result.output, target)
    assert result.score == pytest.approx(sum(result.components.values()))
    observation = editing.observe(bad, pair, result, 1, config)
    assert observation.feedback[2] == 1 and observation.feedback[-1] == pytest.approx(1 / config.max_seq_len)
    # Failure does not poison the next execution or make it start from the partial state.
    good = editing.evaluate(editing.seed_tree(config.edit_config), task, env, initial)
    assert good.success and good.error is None and good.ticks == 1
    partial = editing.open_subtree(bad, 0)
    during = editing.observe(partial, pair, result, config.max_seq_len, config)
    np.testing.assert_array_equal(during.output, result.output)
    assert not during.legal[: 1 + config.max_nodes].any()
    assert during.legal[1 + config.max_nodes + ACTION_ID["Program"]]


def test_rollout_replay_rewards_and_minibatches(config: edit.Config, state: TrainState, rollout: edit.Rollout) -> None:
    batch, rewards, diagnostics, _, programs = rollout
    assert all(isinstance(leaf, (np.ndarray, jax.Array)) for leaf in jax.tree.leaves(batch))
    logits = edit.mask_logits(predict(state, batch.history), batch.legal)
    np.testing.assert_allclose(edit.action_log_prob(logits, batch.actions), batch.old_log_probs, atol=2e-6)
    assert np.take_along_axis(batch.legal, batch.actions[..., None], axis=-1).all()
    assert np.isfinite(batch.old_log_probs).all()
    np.testing.assert_allclose(batch.rewards.sum(axis=0), rewards)
    assert not batch.rewards[~batch.mask].any()
    expected_advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    np.testing.assert_allclose(batch.advantages, expected_advantages, atol=1e-6)
    assert np.all(batch.mask.sum(axis=0) <= config.max_seq_len)
    assert batch.actions.shape[0] <= config.max_seq_len
    assert not batch.mask[: len(program_actions(tuple(config.seed_program.split())))].any()
    seed_end = len(program_actions(tuple(config.seed_program.split()))) + 1
    np.testing.assert_array_equal(
        batch.history.events.output[:seed_end],
        np.broadcast_to(batch.history.events.output[0], batch.history.events.output[:seed_end].shape),
    )
    seed = int(np.random.default_rng(4).integers(0, 2**31, size=1)[0])
    env = KarelProgramEnv(config.env)
    for index, tree in enumerate(programs):
        pair = env.reset(seed=seed)
        np.testing.assert_array_equal(batch.history.initial[index], pair.initial)
        for token in tree.tokens():
            _, reward, terminated, truncated, info = env.step(TOKEN_TO_ID[token])
        assert terminated and not truncated
        reward -= float(info["reward_length"])
        reward -= config.env.length_penalty_weight * len(tree.nodes) / config.max_nodes
        reward -= config.env.depth_penalty_weight * max(node.depth for node in tree.nodes) / config.max_depth
        env.reset(seed=seed)
        seed_score = editing.evaluate(
            editing.seed_tree(config.edit_config),
            env,
            KarelProgramEnv(config.env),
            pair.initial,
        ).score
        assert rewards[index] == pytest.approx(reward - seed_score)
    assert diagnostics["charts/syntax_error_rate"] == diagnostics["charts/token_limit_rate"] == 0
    subset = edit.select_episodes(batch, np.asarray([1, 0], np.int64))
    np.testing.assert_array_equal(subset.actions, batch.actions[:, ::-1])
    np.testing.assert_array_equal(subset.rewards, batch.rewards[:, ::-1])
    np.testing.assert_array_equal(subset.advantages, batch.advantages[::-1])
    np.testing.assert_array_equal(subset.history.events.output, batch.history.events.output[:, ::-1])
    with pytest.raises(ValueError, match="unique"):
        edit.select_episodes(batch, np.asarray([0, 0], np.int64))
    with pytest.raises(ValueError, match="environments"):
        edit.collect_rollout(
            state,
            edit.KarelASTEditVectorEnv(config.edit_config, 1),
            np.random.default_rng(4),
            jax.random.key(7),
            config,
        )


@pytest.mark.parametrize("improvement", [0.0, 1e-8], ids=["equal-final-scores", "small-real-improvement"])
def test_advantages_use_final_scores_without_delta_roundoff(
    config: edit.Config,
    state: TrainState,
    rollout: edit.Rollout,
    monkeypatch: pytest.MonkeyPatch,
    improvement: float,
) -> None:
    # One episode cycles 0.3 -> 0.4 -> 0.1 -> 0.3; the other stops or improves.
    # Casting each delta to float32 before summing invents a difference on ties.
    rewards = np.zeros_like(rollout.batch.rewards)
    rewards[4:7, 0] = [0.1, -0.3, 0.2]
    rewards[4, 1] = improvement
    assert rewards[:, 0].sum() != 0
    results = [
        editing.Evaluation(
            np.zeros((3, 3, 6), np.int32), score, False, None, 0, dict.fromkeys(edit.EDIT_REWARD_COMPONENTS, 0.0)
        )
        for score in (0.3, 0.3 + improvement)
    ]
    monkeypatch.setattr(
        edit,
        "run_episodes",
        MagicMock(
            return_value=(
                rollout.batch._replace(rewards=rewards),
                results,
                np.full(2, 0.3, np.float32),
                np.asarray([3, 0], np.int32),
                rollout.key,
                rollout.programs,
            )
        ),
    )
    with edit.KarelASTEditVectorEnv(config.edit_config, 2) as envs:
        collected = edit.collect_rollout(state, envs, np.random.default_rng(4), rollout.key, config)
    expected = np.asarray([-improvement / 2, improvement / 2]) / (improvement / 2 + 1e-8)
    np.testing.assert_allclose(collected.batch.advantages, expected, atol=1e-7)
    assert collected.diagnostics["charts/reward_diverse_group_fraction"] == float(improvement > 0)


def test_stop_is_a_real_decision(config: edit.Config, state: TrainState) -> None:
    params = {**state.params, "head": {**state.params["head"], "bias": state.params["head"]["bias"].at[0].set(100)}}
    stopped = state.replace(params=params)
    batch, rewards, diagnostics, _, programs = edit.collect_rollout(
        stopped,
        edit.KarelASTEditVectorEnv(config.edit_config, 2),
        np.random.default_rng(4),
        jax.random.key(7),
        config,
    )
    np.testing.assert_array_equal(batch.actions[batch.mask], [0, 0])
    assert batch.mask.sum() == 2 and diagnostics["charts/edits_mean"] == 0
    assert diagnostics["charts/sequence_budget_exhausted_rate"] == 0
    np.testing.assert_array_equal(batch.advantages, 0)
    assert all(tree.tokens() == editing.seed_tree(config.edit_config).tokens() for tree in programs)
    assert rewards[0] == rewards[1]
    # The unchanged seed has Program, ConsNonEmpty, turnLeft, End: four nodes,
    # with both statement and End two edges from the root.
    assert diagnostics["charts/program_node_count_mean"] == 4
    assert diagnostics["charts/program_node_ratio_mean"] == 4 / config.max_nodes
    assert diagnostics["charts/program_depth_mean"] == 2
    assert diagnostics["charts/program_depth_ratio_mean"] == 2 / config.max_depth
    assert diagnostics["charts/reward_depth_mean"] == pytest.approx(
        -config.env.depth_penalty_weight * 2 / config.max_depth
    )
    assert diagnostics["charts/reward_length_mean"] == pytest.approx(
        -config.env.length_penalty_weight * 4 / config.max_nodes
    )


def test_compile_logs_name_dimensions(
    config: edit.Config,
    state: TrainState,
    rollout: edit.Rollout,
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch = rollout.batch
    history = batch.history._replace(events=jax.tree.map(lambda x: x[:4], batch.history.events))
    # Trace without compiling/executing extra model updates just to inspect their logs.
    carry, _ = jax.eval_shape(edit.prefill, state, history)
    event = jax.tree.map(lambda x: x[0], history.events)
    assert capsys.readouterr().out == ""
    for _ in range(2):
        edit.prefill.lower(state, history, log_compiles=True)
        edit.decode_step.lower(state, event, carry, log_compiles=True)
        edit.update.lower(state, batch, replace(config, log_compiles=True))
    assert capsys.readouterr().out == (
        "JIT trace edit prefill: b=2, t=4\n"
        "JIT trace edit step: b=2\n"
        f"JIT trace edit update: b=2, t={batch.actions.shape[0]}\n"
    )
    shorter = history._replace(events=jax.tree.map(lambda x: x[:3], history.events))
    edit.prefill.lower(state, shorter, log_compiles=True)
    assert capsys.readouterr().out == "JIT trace edit prefill: b=2, t=3\n"
    edit.prefill.lower(state, shorter)
    edit.decode_step.lower(state, event, carry)
    edit.update.lower(state, batch, config)
    assert capsys.readouterr().out == ""


def test_update_and_episode_normalization(
    config: edit.Config,
    state: TrainState,
    rollout: edit.Rollout,
) -> None:
    batch = rollout.batch
    assert np.any(batch.advantages != 0)
    updated, metrics = edit.update(state, batch, config)
    assert int(updated.step) == int(state.step) + 1
    assert np.isfinite(np.asarray(metrics)).all() and abs(float(metrics[2])) < 1e-6
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(state.params), jax.tree.leaves(updated.params)))
    assert not np.array_equal(state.params["grid_conv"]["kernel"], updated.params["grid_conv"]["kernel"])
    # Preserve equal episode weights even when their decision counts differ.
    weights = (batch.mask / batch.mask.sum(axis=0)[None] / 2).astype(np.float32)
    _, expected = edit.objective(
        edit.mask_logits(predict(state, batch.history), batch.legal),
        batch.actions,
        batch.old_log_probs,
        np.broadcast_to(batch.advantages, batch.actions.shape),
        weights,
        config,
    )
    np.testing.assert_allclose(metrics, expected, atol=2e-6)
    # Enabling the KL guard still accepts a fresh rollout and performs the same update.
    guarded_config = replace(config, target_kl=0.01)
    accepted, _ = edit.update(state, batch, guarded_config)
    chex.assert_trees_all_close(accepted, updated, atol=2e-6)
    # A stale rollout must reject the ENTIRE minibatch before applying gradients.
    stale = batch._replace(old_log_probs=batch.old_log_probs + np.float32(2))
    unchanged, metrics = edit.update(state, stale, guarded_config)
    assert float(metrics[2]) > 0.01 and int(unchanged.step) == int(state.step)
    chex.assert_trees_all_equal(unchanged, state)


@pytest.mark.parametrize("invalid", ["nonfinite-advantage", "no-decisions"])
def test_update_rejects_invalid_replay(
    config: edit.Config, state: TrainState, rollout: edit.Rollout, invalid: str
) -> None:
    # Rejection must work even when the optional KL threshold is disabled.
    assert config.target_kl is None
    batch = rollout.batch
    if invalid == "nonfinite-advantage":
        batch = batch._replace(advantages=np.full_like(batch.advantages, np.nan))
    else:
        mask = batch.mask.copy()
        mask[:, 0] = False
        batch = batch._replace(mask=mask)
    unchanged, metrics = edit.update(state, batch, config)
    if invalid == "nonfinite-advantage":
        assert not np.isfinite(np.asarray(metrics)).all()
    chex.assert_trees_all_equal(unchanged, state)


def test_config_and_checkpoint(config: edit.Config, state: TrainState, tmp_path: Path) -> None:
    supplied = edit.load_config("configs/karel_ast_ar_edit.yaml")
    assert supplied.edit_config.max_seq_len == supplied.max_seq_len
    assert supplied.edit_config.env == supplied.env
    # Test parsing independently of the tunable checked-in experiment settings.
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {"env": {"depth_penalty_weight": 0.1, "length_penalty_weight": 0.2, "execution_penalty_weight": 0.3}}
        )
    )
    parsed = edit.load_config(tmp_path / "config.yaml")
    assert parsed.env.depth_penalty_weight == 0.1
    assert parsed.env.length_penalty_weight == 0.2
    assert parsed.env.execution_penalty_weight == 0.3
    for changes, error in (
        ({"max_seq_len": 0}, AssertionError),
        ({"max_seq_len": True}, TypeError),
        ({"env_workers": -1}, AssertionError),
        ({"env_workers": True}, TypeError),
        ({"max_seq_len": 5}, ValueError),
        ({"seed_program": "DEF run m( m)"}, ValueError),
        ({"seed_program": "DEF run m( turnLeft turnRight putMarker move m)"}, ValueError),
    ):
        with pytest.raises(error):
            replace(config, **changes)
    # Runs created before accumulation was removed must still load for inference.
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({**asdict(config), "replay_batch_size": 1}))
    assert edit.load_config(tmp_path / "config.yaml") == config
    rng = np.random.default_rng(7)
    progress = edit.TrainingProgress(state, jax.random.key(5), 3, 6, 2, 6)
    edit._save_checkpoint(str(tmp_path), progress, rng)
    restored = edit._restore_checkpoint((tmp_path / "checkpoint.msgpack").read_bytes(), state, np.random.default_rng(8))
    assert restored.iteration == 3 and restored.steps == 6
    _, loaded = edit.load_model(tmp_path, attention_implementation="xla")
    for a, b in zip(jax.tree.leaves(loaded.params), jax.tree.leaves(state.params)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("rejected", [False, True], ids=["accepted", "nonfinite-rejection"])
def test_training_wiring(
    config: edit.Config,
    state: TrainState,
    rollout: edit.Rollout,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rejected: bool,
) -> None:
    # Exercise orchestration/checkpoint boundaries without another rollout or training update.
    monkeypatch.setattr(edit, "create_state", MagicMock(return_value=state))
    collect = MagicMock(return_value=rollout)
    monkeypatch.setattr(edit, "collect_rollout", collect)

    def fake_update(
        current: TrainState, batch: edit.EditBatch, settings: edit.Config
    ) -> tuple[TrainState, edit.Metrics]:
        chex.assert_shape(batch.advantages, (2,))
        chex.assert_type(batch.advantages, np.float32)
        assert settings.target_kl is None
        if rejected:
            return current, tuple(jnp.full(4, jnp.nan))
        return current.replace(step=current.step + 1), tuple(jnp.zeros(4))

    update = MagicMock(side_effect=fake_update)
    monkeypatch.setattr(edit, "update", update)
    writer = MagicMock()
    monkeypatch.setattr(edit, "SummaryWriter", MagicMock(return_value=writer))
    vector = edit.KarelASTEditVectorEnv(config.edit_config, config.num_tasks * config.group_size)
    vector_factory = MagicMock(return_value=vector)
    monkeypatch.setattr(edit, "KarelASTEditVectorEnv", vector_factory)
    config = replace(config, log_dir=str(tmp_path), run_id="test", update_epochs=2)
    edit.train(config)
    vector_factory.assert_called_once_with(config.edit_config, 2, workers=config.env_workers)
    assert vector.closed
    assert collect.call_count == 1
    assert update.call_count == (1 if rejected else 2)
    writer.add_scalar.assert_any_call("charts/updates_per_rollout", 0.0 if rejected else 2.0, 2)
    writer.add_scalar.assert_any_call("policy/early_stop", float(rejected), 2)
    output = capsys.readouterr().out
    depth_penalty = rollout.diagnostics["charts/reward_depth_mean"]
    assert f"depth_penalty={depth_penalty:.4f}" in output
    writer.add_scalar.assert_any_call("charts/reward_depth_mean", depth_penalty, 2)
    for name in (
        "decisions_mean",
        "sequence_length_mean",
        "edits_mean",
        "sequence_budget_exhausted_rate",
        "program_node_count_mean",
        "program_node_ratio_mean",
        "program_depth_mean",
        "program_depth_ratio_mean",
    ):
        value = rollout[2][f"charts/{name}"]
        assert f"{name}={value:.3f}" in output
        writer.add_scalar.assert_any_call(f"charts/{name}", value, 2)
    assert (tmp_path / "test" / "checkpoint.msgpack").is_file()
    writer.add_text.assert_any_call(
        "samples/generated_programs",
        edit.format_group_programs(rollout.programs, rollout.rewards, config=config, group_index=0),
        2,
    )
    edit.train(config)
    assert collect.call_count == 1  # Resume recognizes the completed rollout.
    writer.close.assert_called_once()
    if not rejected:
        # Different attention head layouts have identical parameter shapes but
        # change the meaning of the saved projections and rotary positions.
        saved_config = (tmp_path / "test" / "config.yaml").read_bytes()
        with pytest.raises(ValueError, match="num_heads"):
            edit.train(replace(config, num_heads=4, total_updates=2))
        assert collect.call_count == 1
        assert (tmp_path / "test" / "config.yaml").read_bytes() == saved_config


def test_resume_config_preserves_model_semantics(config: edit.Config) -> None:
    # Explicit KV heads equal to query heads are equivalent to the default.
    # Training duration, precision and episode budgets may change on resume.
    edit.check_resume_config(
        replace(config, num_kv_heads=config.num_heads, total_updates=2, bf16=True, max_seq_len=16), config
    )
    saved = replace(config, env=replace(config.env, height=3, width=6))
    # Equal cell counts preserve projection shapes, but change spatial positions.
    with pytest.raises(ValueError, match="env.height, env.width"):
        edit.check_resume_config(replace(saved, env=replace(saved.env, height=6, width=3)), saved)
    with pytest.raises(ValueError, match="env.max_markers"):
        edit.check_resume_config(replace(config, env=replace(config.env, max_markers=20)), config)


def test_generation_executes_atomic_replacements(
    config: edit.Config, state: TrainState, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A root replacement of a different size, then a statement edit. No execution
    # happens during the grammar decisions, and the sequence budget ends without STOP.
    grammar = program_actions(("DEF", "run", "m(", "turnRight", "turnLeft", "m)"))
    script = [1, *(1 + config.max_nodes + a for a in grammar), 3, 1 + config.max_nodes + ACTION_ID["putMarker"]]
    selected: list[int] = []
    original_step = edit.decode_step
    step = MagicMock(wraps=original_step)
    monkeypatch.setattr(edit, "decode_step", step)

    def act(logits: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        chex.assert_shape(logits, (None, 1 + config.max_nodes + len(edit.AST_ACTIONS)))
        chex.assert_type(logits, jnp.float32)
        chex.assert_shape(key, ())
        chex.assert_type(jax.random.key_data(key), jnp.uint32)
        action = script[len(selected)]
        assert np.isfinite(logits[0, action])
        selected.append(action)
        actions = jnp.asarray([action], jnp.int32)
        return actions, edit.action_log_prob(logits, actions)

    monkeypatch.setattr(edit, "act", act)
    evaluate = MagicMock(wraps=editing.evaluate)
    monkeypatch.setattr(editing, "evaluate", evaluate)
    task = KarelProgramEnv(config.env)
    task.reset(seed=12)
    result = edit.generate(state, task, jax.random.key(7), config)
    assert result.tokens() == ("DEF", "run", "m(", "putMarker", "turnLeft", "m)")
    assert selected == script and evaluate.call_count == 3  # Seed plus two complete edits.
    consumed = [call.args[1] for call in step.call_args_list]
    assert len(consumed) == len(script) - 1  # Every nonterminal action is decoded exactly once.
    assert all(event.kind[0] == edit.ACTION_EVENT for event in consumed)
    completion = consumed[len(grammar)]
    assert completion.feedback[0, -1] == pytest.approx(2 / config.max_seq_len)
    assert completion.feedback.shape[-1] == 8
    assert KarelProgramEnv(config.env).reset_from(task) is not None
    with pytest.raises(ValueError, match="limits"):
        edit.generate(state, task, jax.random.key(7), replace(config, env=replace(config.env, max_execution_steps=1)))


def test_execution_limit_retains_last_valid_grid(config: edit.Config) -> None:
    config = replace(config, env=replace(config.env, max_execution_steps=1))
    task, env = KarelProgramEnv(config.env), KarelProgramEnv(config.env)
    pair = task.reset(seed=42)
    seed = editing.evaluate(editing.seed_tree(config.edit_config), task, env, pair.initial)
    tree = tree_from("DEF run m( turnLeft turnLeft m)", config)
    failed = editing.evaluate(tree, task, env, pair.initial)
    assert failed.error == "execution_limit" and not failed.success and failed.ticks == 1
    np.testing.assert_array_equal(failed.output, seed.output)
    observation = editing.observe(tree, pair, failed, 0, config)
    assert observation.feedback[3] == observation.feedback[4] == 1
    assert observation.legal[0] and observation.legal.sum() == 1


def test_grid_encoder_mixes_all_three_images_per_cell(config: edit.Config, state: TrainState) -> None:
    # Perturb a different stacked image in each example at the same cell.
    initial, target, output = (np.zeros((3, 3, 3, 6), np.int32) for _ in range(3))
    initial[0, 1, 1, 0] = 1
    target[1, 1, 1, 0] = 1
    output[2, 1, 1, 0] = 1
    encoded = state.apply_fn({"params": state.params}, initial, target, output, method=EditTransformer.encode_grids)
    features = np.array(encoded).reshape(3, 3, 3, 32)
    assert np.all(np.any(features[:, 1, 1] != 0, axis=-1))
    assert not np.allclose(features[0, 1, 1], features[1, 1, 1])
    assert not np.allclose(features[1, 1, 1], features[2, 1, 1])
    features[:, 1, 1] = 0
    np.testing.assert_array_equal(features, 0)  # A 1x1 convolution cannot mix neighboring cells.
    # Every real event kind receives its current image, including seed and action tokens.
    events = edit.empty_events(3, 1, config)
    events.kind[:, 0] = [edit.SEED_EVENT, edit.ACTION_EVENT, edit.UPDATE_EVENT]
    original = state.apply_fn(
        {"params": state.params}, events, initial[:1], target[:1], method=EditTransformer.encode_events
    )
    events.output[:, 0, 1, 1, 5] = 1
    changed = state.apply_fn(
        {"params": state.params}, events, initial[:1], target[:1], method=EditTransformer.encode_events
    )
    assert np.all(np.any(np.abs(np.asarray(changed - original)) > 1e-6, axis=-1))


def test_history_is_causal_and_retains_action_and_update_context(
    config: edit.Config,
    state: TrainState,
    rollout: edit.Rollout,
) -> None:
    history = rollout[0].history
    logits = np.asarray(predict(state, history))
    assert logits.shape[:2] == history.events.kind.shape
    assert np.all(history.events.kind[0] == edit.SEED_EVENT)
    output = np.array(history.events.output)
    output[0, :, 1, 1, 5] += 1
    changed = history._replace(events=history.events._replace(output=output))
    seed_logits = np.asarray(predict(state, changed))
    assert not np.allclose(logits[0], seed_logits[0])
    # The initial execution report is a forced observation, not a prediction
    # target. Changing it must affect only predictions AFTER it is consumed.
    update_index = len(program_actions(tuple(config.seed_program.split())))
    feedback = np.array(history.events.feedback)
    feedback[update_index, :, 0] += 10
    changed = history._replace(events=history.events._replace(feedback=feedback))
    new_logits = np.asarray(predict(state, changed))
    np.testing.assert_array_equal(logits[:update_index], new_logits[:update_index])
    assert not np.allclose(logits[update_index], new_logits[update_index])
    # Changing one event's image cannot affect predictions preceding that event.
    output = np.array(history.events.output)
    output[update_index, :, 1, 1, 5] += 1
    changed = history._replace(events=history.events._replace(output=output))
    image_logits = np.asarray(predict(state, changed))
    np.testing.assert_array_equal(logits[:update_index], image_logits[:update_index])
    assert not np.allclose(logits[update_index], image_logits[update_index])
    # An earlier seed token remains in causal context after the initial report.
    values = np.array(history.events.action)
    values[2] = 1 + config.max_nodes + ACTION_ID["turnRight"]
    edited = history._replace(events=history.events._replace(action=values))
    edited_logits = np.asarray(predict(state, edited))
    np.testing.assert_array_equal(logits[:2], edited_logits[:2])
    assert not np.allclose(logits[update_index], edited_logits[update_index])
    # Trailing filler cannot influence any real decision.
    last = int(np.flatnonzero(rollout[0].mask.any(axis=1))[-1])
    kinds = np.array(history.events.kind)
    kinds[last + 1 :] = edit.UPDATE_EVENT
    padded = history._replace(events=history.events._replace(kind=kinds))
    np.testing.assert_allclose(predict(state, padded)[: last + 1], logits[: last + 1], atol=1e-6)


def test_action_feedback_and_finished_padding_replay_exactly(
    config: edit.Config, state: TrainState, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One member stops immediately; the other must consume execution feedback
    # on the completing action. Finished members contribute only PAD.
    grammar = program_actions(("DEF", "run", "m(", "turnRight", "turnLeft", "m)"))
    scripts = [[0], [1, *(1 + config.max_nodes + a for a in grammar), 3, 1 + config.max_nodes + ACTION_ID["putMarker"]]]
    offsets = [0, 0]

    def act(logits: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        chex.assert_shape(logits, (None, 1 + config.max_nodes + len(edit.AST_ACTIONS)))
        chex.assert_type(logits, jnp.float32)
        chex.assert_shape(key, ())
        chex.assert_type(jax.random.key_data(key), jnp.uint32)
        actions = np.zeros(2, np.int32)
        for index in range(2):
            if np.isfinite(logits[index, 0]) and np.isfinite(logits[index]).sum() == 1:
                continue
            actions[index] = scripts[index][offsets[index]]
            assert np.isfinite(logits[index, actions[index]])
            offsets[index] += 1
        actions = jnp.asarray(actions)
        return actions, edit.action_log_prob(logits, actions)

    monkeypatch.setattr(edit, "act", act)
    with edit.KarelASTEditVectorEnv(config.edit_config, 2, workers=2) as envs:
        batch, _, _, _, _ = edit.collect_rollout(state, envs, np.random.default_rng(4), jax.random.key(7), config)
    assert offsets == [len(script) for script in scripts]
    np.testing.assert_array_equal(batch.actions[batch.mask[:, 0], 0], scripts[0])
    np.testing.assert_array_equal(batch.actions[batch.mask[:, 1], 1], scripts[1])
    events = batch.history.events
    initial_update = len(program_actions(tuple(config.seed_program.split())))
    assert not np.any(events.kind[initial_update + 1 :] == edit.UPDATE_EVENT)
    completion = initial_update + 1 + len(grammar)
    assert events.kind[completion, 1] == edit.ACTION_EVENT
    assert batch.mask[completion - 1, 1] and batch.mask[completion, 1]
    # No missing decision positions between the seed report and terminal action.
    np.testing.assert_array_equal(
        np.flatnonzero(batch.mask[:, 1]), np.arange(initial_update, initial_update + len(scripts[1]))
    )
    assert events.feedback[completion, 1, -1] == pytest.approx(2 / config.max_seq_len)
    assert events.feedback[completion, 1, -2] == pytest.approx(batch.rewards[completion - 1, 1])
    # The image switches with the completing action and persists through partial edits.
    np.testing.assert_array_equal(
        events.output[:completion, 1],
        np.broadcast_to(events.output[0, 1], events.output[:completion, 1].shape),
    )
    last_decision = int(np.flatnonzero(batch.mask[:, 1])[-1])
    np.testing.assert_array_equal(
        events.output[completion:last_decision, 1],
        np.broadcast_to(events.output[completion, 1], events.output[completion:last_decision, 1].shape),
    )
    assert not np.array_equal(events.output[completion, 1], events.output[0, 1])
    np.testing.assert_array_equal(events.output[:, 0], np.broadcast_to(events.output[0, 0], events.output[:, 0].shape))
    assert np.all(events.kind[initial_update + 2 :, 0] == PAD_EVENT)
    logits = predict(state, batch.history)
    replay = edit.action_log_prob(edit.mask_logits(logits, batch.legal), batch.actions)
    np.testing.assert_allclose(replay, batch.old_log_probs, atol=2e-6)
    # An earlier action remains visible after a subsequent execution update.
    values = np.array(events.action)
    action_index = initial_update + 3  # First grammar action after location selection.
    values[action_index, 1] = 1 + config.max_nodes + ACTION_ID["turnLeft"]
    changed = batch.history._replace(events=events._replace(action=values))
    changed_logits = predict(state, changed)
    np.testing.assert_array_equal(changed_logits[:action_index], logits[:action_index])
    assert not np.allclose(changed_logits[completion, 1], logits[completion, 1])
    feedback = np.array(events.feedback)
    feedback[completion, 1, 0] += 10
    changed = batch.history._replace(events=events._replace(feedback=feedback))
    feedback_logits = predict(state, changed)
    np.testing.assert_array_equal(feedback_logits[:completion], logits[:completion])
    assert not np.allclose(feedback_logits[completion, 1], logits[completion, 1])


def test_sequence_budget_reserves_complete_replacements(config: edit.Config) -> None:
    task = KarelProgramEnv(config.env)
    pair = task.reset(seed=42)
    tree = editing.seed_tree(config.edit_config)
    result = editing.evaluate(tree, task, KarelProgramEnv(config.env), pair.initial)
    only_stop = editing.observe(tree, pair, result, 1, config).legal
    assert np.flatnonzero(only_stop).tolist() == [0]
    two_tokens = editing.observe(tree, pair, result, 2, config).legal
    assert two_tokens[3]  # Select the primitive statement, then replace it with a leaf.
    assert not two_tokens[1]  # Replacing Program needs more than one grammar action.
    assert editing.observe(tree, pair, result, 5, config).legal[1]  # Location + four-node completion.

    # Root replacement has two sibling holes after Program/ConsNonEmpty. Both
    # need a reserved action each; feedback costs no extra token.
    partial = editing.open_subtree(tree, 0).expand(ACTION_ID["Program"]).expand(ACTION_ID["ConsNonEmpty"])
    legal = editing.observe(partial, pair, result, 2, config).legal
    offset = 1 + config.max_nodes
    assert legal[offset + ACTION_ID["move"]]
    assert not legal[offset + ACTION_ID["REPEAT"]]
    partial = partial.expand(ACTION_ID["turnRight"])
    legal = editing.observe(partial, pair, result, 1, config).legal
    assert np.flatnonzero(legal).tolist() == [offset + ACTION_ID["End"]]
    assert partial.expand(ACTION_ID["End"]).complete


def test_disabled_stop_keeps_editing_despite_stop_biased_policy(config: edit.Config, state: TrainState) -> None:
    # Four location+leaf edits leave one token, too little for another edit.
    config = replace(config, allow_stop=False)
    replacement = 1 + config.max_nodes + ACTION_ID["turnRight"]
    bias = state.params["head"]["bias"].at[0].set(200).at[3].set(100).at[replacement].set(100)
    state = state.replace(params={**state.params, "head": {**state.params["head"], "bias": bias}})
    batch, _, metrics, _, programs = edit.collect_rollout(
        state,
        edit.KarelASTEditVectorEnv(config.edit_config, 2),
        np.random.default_rng(4),
        jax.random.key(7),
        config,
    )
    assert metrics["charts/edits_mean"] == 4
    np.testing.assert_array_equal(batch.mask.sum(axis=0), 8)
    assert metrics["charts/sequence_length_mean"] == config.max_seq_len - 1
    assert metrics["charts/sequence_budget_exhausted_rate"] == 1
    assert not batch.legal[..., 0][batch.mask].any()
    assert np.all(batch.actions[batch.mask] != 0)
    assert all(tree.complete and "turnRight" in tree.tokens() for tree in programs)
    for index in range(2):
        assert batch.actions[batch.mask[:, index], index][-1] == 1 + config.max_nodes + ACTION_ID["turnRight"]
    np.testing.assert_array_equal(batch.advantages, 0)  # Identical episode returns in this group.
    assert batch.history.events.kind.shape[0] <= config.max_seq_len
    replay = edit.action_log_prob(edit.mask_logits(predict(state, batch.history), batch.legal), batch.actions)
    np.testing.assert_allclose(replay, batch.old_log_probs, atol=2e-6)


def test_load_config_stop_option(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("allow_stop: false\n")
    config = edit.load_config(path)
    assert not config.allow_stop and not config.edit_config.allow_stop
    path.write_text("{}\n")
    assert edit.load_config(path).allow_stop  # Existing saved configurations retain STOP.
    path.write_text('allow_stop: "false"\n')
    with pytest.raises(TypeError, match="allow_stop must be a boolean"):
        edit.load_config(path)


@pytest.mark.parametrize(
    "changes,error,match",
    [
        ({"run_id": "../outside"}, ValueError, "run_id"),
        ({"group_size": 1}, ValueError, "group_size"),
        ({"num_minibatches": 3}, ValueError, "num_minibatches"),
        ({"attention_implementation": "cudnn", "bf16": False}, ValueError, "bf16"),
        ({"learning_rate": float("inf")}, ValueError, "finite"),
        ({"log_interval": True}, TypeError, "Logging"),
    ],
    ids=["run-path", "group-minimum", "minibatch-divisibility", "cudnn-dtype", "optimizer-finite", "logging-type"],
)
def test_config_validates_training_settings(
    config: edit.Config, changes: dict[str, object], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        replace(config, **changes)
