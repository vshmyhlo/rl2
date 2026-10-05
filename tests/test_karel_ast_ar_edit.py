from dataclasses import asdict, replace
from pathlib import Path
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
        env=KarelConfig(height=3, width=3, max_depth=0, max_statements=1, max_program_tokens=8),
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
            _, reward, terminated, truncated, _ = env.step(TOKEN_TO_ID[token])
        assert terminated and not truncated
        env.reset(seed=seed)
        seed_score = editing.evaluate(
            editing.seed_tree(config.edit_config), env, KarelProgramEnv(config.env), pair.initial
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
    np.testing.assert_array_equal(batch.advantages, 0)
    assert all(tree.tokens() == editing.seed_tree(config.edit_config).tokens() for tree in programs)
    assert rewards[0] == rewards[1]
    # The unchanged seed has Program, ConsNonEmpty, turnLeft, End: four nodes,
    # with both statement and End two edges from the root.
    assert diagnostics["charts/program_node_count_mean"] == 4
    assert diagnostics["charts/program_node_ratio_mean"] == 4 / config.max_nodes
    assert diagnostics["charts/program_depth_mean"] == 2
    assert diagnostics["charts/program_depth_ratio_mean"] == 2 / config.max_depth


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
    encoded = state.apply_fn(
        {"params": state.params}, initial, target, output, method=edit.EditTransformer.encode_grids
    )
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
        {"params": state.params}, events, initial[:1], target[:1], method=edit.EditTransformer.encode_events
    )
    events.output[:, 0, 1, 1, 5] = 1
    changed = state.apply_fn(
        {"params": state.params}, events, initial[:1], target[:1], method=edit.EditTransformer.encode_events
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
    values = np.array(history.events.value)
    values[2] = 1 + config.max_nodes + ACTION_ID["turnRight"]
    edited = history._replace(events=history.events._replace(value=values))
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
    assert np.all(events.kind[initial_update + 2 :, 0] == edit.PAD_EVENT)
    logits = predict(state, batch.history)
    replay = edit.action_log_prob(edit.mask_logits(logits, batch.legal), batch.actions)
    np.testing.assert_allclose(replay, batch.old_log_probs, atol=2e-6)
    # An earlier action remains visible after a subsequent execution update.
    values = np.array(events.value)
    action_index = initial_update + 3  # First grammar action after location selection.
    values[action_index, 1] = 1 + config.max_nodes + ACTION_ID["turnLeft"]
    changed = batch.history._replace(events=events._replace(value=values))
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


def test_more_than_three_edits_fit_sequence_budget(
    config: edit.Config, state: TrainState, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Four location+leaf edits fill the sequence after the five-token seed prefill.
    config = replace(config, max_seq_len=13)
    task = KarelProgramEnv(config.env)
    pair = task.reset(seed=4)
    state = edit.create_state(config, pair.initial[None], pair.target[None]).replace(params=state.params)

    def act(logits: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        chex.assert_shape(logits, (2, 1 + config.max_nodes + len(edit.AST_ACTIONS)))
        chex.assert_type(logits, jnp.float32)
        chex.assert_shape(key, ())
        chex.assert_type(jax.random.key_data(key), jnp.uint32)
        actions = np.zeros(2, np.int32)
        for index in range(2):
            allowed = np.isfinite(logits[index])
            if allowed[3]:
                actions[index] = 3  # Preorder statement location.
            elif allowed[1 + config.max_nodes + ACTION_ID["turnRight"]]:
                actions[index] = 1 + config.max_nodes + ACTION_ID["turnRight"]
            else:
                assert allowed[0]  # Final STOP.
        actions = jnp.asarray(actions)
        return actions, edit.action_log_prob(logits, actions)

    monkeypatch.setattr(edit, "act", act)
    batch, _, metrics, _, programs = edit.collect_rollout(
        state,
        edit.KarelASTEditVectorEnv(config.edit_config, 2),
        np.random.default_rng(4),
        jax.random.key(7),
        config,
    )
    assert metrics["charts/edits_mean"] == 4
    assert metrics["charts/sequence_budget_exhausted_rate"] == 1
    np.testing.assert_array_equal(batch.mask.sum(axis=0), 8)
    assert metrics["charts/sequence_length_mean"] == config.max_seq_len
    assert all(tree.complete and "turnRight" in tree.tokens() for tree in programs)
    for index in range(2):
        assert batch.actions[batch.mask[:, index], index][-1] == 1 + config.max_nodes + ACTION_ID["turnRight"]
    np.testing.assert_array_equal(batch.advantages, 0)  # Identical episode returns in this group.
    assert batch.history.events.kind.shape[0] <= config.max_seq_len
    replay = edit.action_log_prob(edit.mask_logits(predict(state, batch.history), batch.legal), batch.actions)
    np.testing.assert_allclose(replay, batch.old_log_probs, atol=2e-6)


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
