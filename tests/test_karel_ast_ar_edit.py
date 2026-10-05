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

from rl2 import train_karel_ast_ar_edit as edit
from rl2.karel import TOKEN_TO_ID, KarelConfig, KarelPair, KarelProgramEnv
from rl2.karel_ast import ACTION_ID, KarelAST, program_actions


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
        max_edits=2,
        replay_batch_size=1,
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
def rollout(config: edit.Config, state: TrainState) -> tuple[edit.GRPOBatch, np.ndarray, dict[str, float], jax.Array]:
    return edit.collect_rollout(
        state, [KarelProgramEnv(config.env) for _ in range(2)], np.random.default_rng(4), jax.random.key(7), config
    )


def tree_from(source: str, config: edit.Config) -> KarelAST:
    tree = KarelAST.empty(config.max_nodes, config.max_depth, config.env.max_program_tokens)
    for action in program_actions(tuple(source.split())):
        tree = tree.expand(action)
    return tree


def test_replacement_grows_shrinks_and_preserves_context(config: edit.Config) -> None:
    tree = edit.seed_tree(config)
    # Replace the final empty list by two statements; fixed prefix survives.
    position = next(p for p, i in enumerate(tree.preorder()) if tree.nodes[i].constructor == ACTION_ID["End"])
    partial = edit.open_subtree(tree, position)
    for name in ("Cons", "turnRight", "Cons", "putMarker", "End"):
        partial = partial.expand(ACTION_ID[name])
    assert partial.complete and len(partial.nodes) == config.max_nodes
    assert partial.tokens() == ("DEF", "run", "m(", "turnLeft", "turnRight", "putMarker", "m)")
    # Shrink the new list tail back to End. Detached descendants must not consume capacity.
    position = next(p for p, i in enumerate(partial.preorder()) if partial.nodes[i].constructor == ACTION_ID["Cons"])
    smaller = edit.open_subtree(partial, position).expand(ACTION_ID["End"])
    assert smaller.tokens() == tree.tokens()
    assert len(smaller.nodes) == len(tree.nodes)
    root = edit.open_subtree(smaller, 0)
    assert len(root.nodes) == 1 and root.nodes[0].is_hole
    with pytest.raises(ValueError, match="complete"):
        edit.open_subtree(root, 0)
    with pytest.raises(ValueError, match="outside"):
        edit.open_subtree(tree, len(tree.nodes))
    with pytest.raises(TypeError, match="integer"):
        edit.open_subtree(tree, True)


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
    tree = edit.open_subtree(edit.seed_tree(limited), 0)
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
    result = edit.evaluate(bad, task, env, initial)
    assert result.error == "runtime_error" and not result.success and result.ticks == 2
    np.testing.assert_array_equal(result.output, target)
    assert result.reward == pytest.approx(sum(result.components.values()))
    observation = edit.observe(bad, pair, result, 1, config)
    assert observation.feedback[2] == 1 and observation.feedback[-1] == 0.5
    # Failure does not poison the next execution or make it start from the partial state.
    good = edit.evaluate(edit.seed_tree(config), task, env, initial)
    assert good.success and good.error is None and good.ticks == 1
    partial = edit.open_subtree(bad, 0)
    during = edit.observe(partial, pair, result, 1, config)
    np.testing.assert_array_equal(during.output, result.output)
    assert not during.legal[: 1 + config.max_nodes].any()
    assert during.legal[1 + config.max_nodes + ACTION_ID["Program"]]


def test_rollout_replay_rewards_and_minibatches(
    config: edit.Config, state: TrainState, rollout: tuple[edit.GRPOBatch, np.ndarray, dict[str, float], jax.Array]
) -> None:
    batch, rewards, diagnostics, _ = rollout
    logits = edit.mask_logits(edit.predict(state, batch.history), batch.legal)
    np.testing.assert_allclose(edit.action_log_prob(logits, batch.actions), batch.old_log_probs, atol=2e-6)
    assert np.take_along_axis(batch.legal, batch.actions[..., None], axis=-1).all()
    assert np.isfinite(batch.old_log_probs).all()
    assert np.all(batch.mask.sum(axis=0) <= config.max_decisions)
    assert batch.actions.shape[0] <= config.max_seq_len
    assert not batch.mask[: len(program_actions(tuple(config.seed_program.split()))) + 1].any()
    seed = int(np.random.default_rng(4).integers(0, 2**31, size=1)[0])
    env = KarelProgramEnv(config.env)
    for index, tree in enumerate(batch.programs):
        pair = env.reset(seed=seed)
        np.testing.assert_array_equal(batch.history.initial[index], pair.initial)
        for token in tree.tokens():
            _, reward, terminated, truncated, _ = env.step(TOKEN_TO_ID[token])
        assert terminated and not truncated
        assert rewards[index] == pytest.approx(reward)
    assert diagnostics["charts/syntax_error_rate"] == diagnostics["charts/token_limit_rate"] == 0
    subset = edit.select_episodes(batch, np.asarray([1, 0], np.int64))
    np.testing.assert_array_equal(subset.actions, batch.actions[:, ::-1])
    np.testing.assert_array_equal(subset.history.events.output, batch.history.events.output[:, ::-1])
    assert subset.programs == tuple(reversed(batch.programs))
    with pytest.raises(ValueError, match="unique"):
        edit.select_episodes(batch, np.asarray([0, 0], np.int64))
    with pytest.raises(ValueError, match="environments"):
        edit.collect_rollout(state, [], np.random.default_rng(4), jax.random.key(7), config)


def test_stop_is_a_real_decision(config: edit.Config, state: TrainState) -> None:
    params = {**state.params, "head": {**state.params["head"], "bias": state.params["head"]["bias"].at[0].set(100)}}
    stopped = state.replace(params=params)
    batch, rewards, diagnostics, _ = edit.collect_rollout(
        stopped, [KarelProgramEnv(config.env) for _ in range(2)], np.random.default_rng(4), jax.random.key(7), config
    )
    np.testing.assert_array_equal(batch.actions[batch.mask], [0, 0])
    assert batch.mask.sum() == 2 and diagnostics["charts/edits_mean"] == 0
    np.testing.assert_array_equal(batch.advantages, 0)
    assert all(tree.tokens() == edit.seed_tree(config).tokens() for tree in batch.programs)
    assert rewards[0] == rewards[1]


def test_update_and_chunk_normalization(
    config: edit.Config, state: TrainState, rollout: tuple[edit.GRPOBatch, np.ndarray, dict[str, float], jax.Array]
) -> None:
    batch = rollout[0]._replace(advantages=np.asarray([-1, 1], np.float32))
    assert len(batch.programs) > config.replay_batch_size  # Accumulate complete episode sequences.
    updated, metrics = edit.update(state, batch, config)
    assert int(updated.step) == int(state.step) + 1
    assert np.isfinite(np.asarray(metrics)).all() and abs(float(metrics[2])) < 1e-6
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(state.params), jax.tree.leaves(updated.params)))
    # Compare summed chunks to one independently normalized objective.
    weights = (batch.mask / batch.mask.sum(axis=0)[None] / 2).astype(np.float32)
    _, expected = edit.objective(
        edit.mask_logits(edit.predict(state, batch.history), batch.legal),
        batch.actions,
        batch.old_log_probs,
        np.broadcast_to(batch.advantages, batch.actions.shape),
        weights,
        config,
    )
    np.testing.assert_allclose(metrics, expected, atol=2e-6)
    # A stale rollout must reject the ENTIRE minibatch before applying gradients.
    stale = batch._replace(old_log_probs=batch.old_log_probs + np.float32(2))
    unchanged, metrics = edit.update(state, stale, replace(config, target_kl=0.01))
    assert float(metrics[2]) > 0.01 and int(unchanged.step) == int(state.step)
    for a, b in zip(jax.tree.leaves(state.params), jax.tree.leaves(unchanged.params)):
        np.testing.assert_array_equal(a, b)


def test_config_and_checkpoint(config: edit.Config, state: TrainState, tmp_path: Path) -> None:
    supplied = edit.load_config("configs/karel_ast_ar_edit.yaml")
    assert supplied.max_seq_len == 2 + supplied.max_nodes + supplied.max_decisions + supplied.max_edits
    for changes, error in (
        ({"max_edits": 0}, AssertionError),
        ({"replay_batch_size": True}, TypeError),
        ({"seed_program": "DEF run m( m)"}, ValueError),
        ({"seed_program": "DEF run m( turnLeft turnRight putMarker move m)"}, ValueError),
    ):
        with pytest.raises(error):
            replace(config, **changes)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(asdict(config)))
    rng = np.random.default_rng(7)
    progress = edit.TrainingProgress(state, jax.random.key(5), 3, 6, 2, 6)
    edit._save_checkpoint(str(tmp_path), progress, rng)
    restored = edit._restore_checkpoint((tmp_path / "checkpoint.msgpack").read_bytes(), state, np.random.default_rng(8))
    assert restored.iteration == 3 and restored.steps == 6
    _, loaded = edit.load_model(tmp_path, attention_implementation="xla")
    for a, b in zip(jax.tree.leaves(loaded.params), jax.tree.leaves(state.params)):
        np.testing.assert_array_equal(a, b)


def test_training_wiring(
    config: edit.Config,
    state: TrainState,
    rollout: tuple[edit.GRPOBatch, np.ndarray, dict[str, float], jax.Array],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exercise orchestration/checkpoint boundaries without another rollout or training update.
    monkeypatch.setattr(edit, "create_state", MagicMock(return_value=state))
    collect = MagicMock(return_value=rollout)
    monkeypatch.setattr(edit, "collect_rollout", collect)
    update = MagicMock(return_value=(state, tuple(jnp.zeros(4))))
    monkeypatch.setattr(edit, "update", update)
    writer = MagicMock()
    monkeypatch.setattr(edit, "SummaryWriter", MagicMock(return_value=writer))
    config = replace(config, log_dir=str(tmp_path), run_id="test")
    edit.train(config)
    assert collect.call_count == update.call_count == 1
    assert (tmp_path / "test" / "checkpoint.msgpack").is_file()
    writer.add_text.assert_any_call(
        "samples/generated_programs",
        edit.format_group_programs(rollout[0], rollout[1], config=config, group_index=0),
        2,
    )
    edit.train(config)
    assert collect.call_count == 1  # Resume recognizes the completed rollout.
    writer.close.assert_called_once()


def test_generation_executes_atomic_replacements(
    config: edit.Config, state: TrainState, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A root replacement of a different size, then a statement edit. No execution
    # happens during the grammar decisions, and max_edits ends without STOP.
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
        # Forced UPDATE positions have only dummy STOP and are not policy actions.
        if np.isfinite(logits[0, 0]) and np.isfinite(logits).sum() == 1:
            action = 0
        else:
            action = script[len(selected)]
            assert np.isfinite(logits[0, action])
            selected.append(action)
        actions = jnp.asarray([action], jnp.int32)
        return actions, edit.action_log_prob(logits, actions)

    monkeypatch.setattr(edit, "act", act)
    evaluate = MagicMock(wraps=edit.evaluate)
    monkeypatch.setattr(edit, "evaluate", evaluate)
    task = KarelProgramEnv(config.env)
    task.reset(seed=12)
    result = edit.generate(state, task, jax.random.key(7), config)
    assert result.tokens() == ("DEF", "run", "m(", "putMarker", "turnLeft", "m)")
    assert selected == script and evaluate.call_count == 3  # Seed plus two complete edits.
    consumed = [call.args[1] for call in step.call_args_list]
    updates = [event for event in consumed if event.kind[0] == edit.UPDATE_EVENT]
    assert len(updates) == 1 and updates[0].feedback[0, -1] == 0.5
    assert len(consumed) == len(script)  # Nonterminal actions plus one intervening update.
    assert consumed[len(grammar) + 1].kind[0] == edit.UPDATE_EVENT
    assert KarelProgramEnv(config.env).reset_from(task) is not None
    with pytest.raises(ValueError, match="limits"):
        edit.generate(state, task, jax.random.key(7), replace(config, env=replace(config.env, max_execution_steps=1)))


def test_execution_limit_retains_last_valid_grid(config: edit.Config) -> None:
    config = replace(config, env=replace(config.env, max_execution_steps=1))
    task, env = KarelProgramEnv(config.env), KarelProgramEnv(config.env)
    pair = task.reset(seed=42)
    seed = edit.evaluate(edit.seed_tree(config), task, env, pair.initial)
    tree = tree_from("DEF run m( turnLeft turnLeft m)", config)
    failed = edit.evaluate(tree, task, env, pair.initial)
    assert failed.error == "execution_limit" and not failed.success and failed.ticks == 1
    np.testing.assert_array_equal(failed.output, seed.output)
    observation = edit.observe(tree, pair, failed, 0, config)
    assert observation.feedback[3] == observation.feedback[4] == 1
    assert observation.legal[0] and observation.legal.sum() == 1


def test_history_is_causal_and_retains_action_and_update_context(
    config: edit.Config,
    state: TrainState,
    rollout: tuple[edit.GRPOBatch, np.ndarray, dict[str, float], jax.Array],
) -> None:
    history = rollout[0].history
    logits = np.asarray(edit.predict(state, history))
    # The initial execution report is a forced observation, not a prediction
    # target. Changing it must affect only predictions AFTER it is consumed.
    update_index = len(program_actions(tuple(config.seed_program.split())))
    feedback = np.array(history.events.feedback)
    feedback[update_index, :, 0] += 10
    changed = history._replace(events=history.events._replace(feedback=feedback))
    new_logits = np.asarray(edit.predict(state, changed))
    np.testing.assert_array_equal(logits[: update_index + 1], new_logits[: update_index + 1])
    assert not np.allclose(logits[update_index + 1], new_logits[update_index + 1])
    # An earlier seed token remains in causal context after the initial report.
    values = np.array(history.events.value)
    values[2] = 1 + config.max_nodes + ACTION_ID["turnRight"]
    edited = history._replace(events=history.events._replace(value=values))
    edited_logits = np.asarray(edit.predict(state, edited))
    np.testing.assert_array_equal(logits[:3], edited_logits[:3])
    assert not np.allclose(logits[update_index + 1], edited_logits[update_index + 1])
    # Trailing filler cannot influence any real decision.
    last = int(np.flatnonzero(rollout[0].mask.any(axis=1))[-1])
    kinds = np.array(history.events.kind)
    kinds[last:] = edit.UPDATE_EVENT
    padded = history._replace(events=history.events._replace(kind=kinds))
    np.testing.assert_allclose(edit.predict(state, padded)[: last + 1], logits[: last + 1], atol=1e-6)


def test_asynchronous_updates_and_finished_padding_replay_exactly(
    config: edit.Config, state: TrainState, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One member stops immediately; the other must consume execution feedback
    # between edits. This exercises simultaneous action/update/PAD stream phases.
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
    batch, _, _, _ = edit.collect_rollout(
        state, [KarelProgramEnv(config.env) for _ in range(2)], np.random.default_rng(4), jax.random.key(7), config
    )
    assert offsets == [len(script) for script in scripts]
    np.testing.assert_array_equal(batch.actions[batch.mask[:, 0], 0], scripts[0])
    np.testing.assert_array_equal(batch.actions[batch.mask[:, 1], 1], scripts[1])
    events = batch.history.events
    initial_update = len(program_actions(tuple(config.seed_program.split())))
    later = np.flatnonzero(events.kind[initial_update + 1 :, 1] == edit.UPDATE_EVENT) + initial_update + 1
    assert len(later) == 1
    assert not batch.mask[later[0], 1] and batch.mask[later[0] + 1, 1]
    assert events.feedback[later[0], 1, -1] == 0.5
    assert np.all(events.kind[initial_update + 2 :, 0] == edit.PAD_EVENT)
    logits = edit.predict(state, batch.history)
    replay = edit.action_log_prob(edit.mask_logits(logits, batch.legal), batch.actions)
    np.testing.assert_allclose(replay, batch.old_log_probs, atol=2e-6)
    # An earlier action remains visible after a subsequent execution update.
    values = np.array(events.value)
    action_index = initial_update + 3  # First grammar action after location selection.
    values[action_index, 1] = 1 + config.max_nodes + ACTION_ID["turnLeft"]
    changed = batch.history._replace(events=events._replace(value=values))
    changed_logits = edit.predict(state, changed)
    np.testing.assert_array_equal(changed_logits[: action_index + 1], logits[: action_index + 1])
    assert not np.allclose(changed_logits[later[0] + 1, 1], logits[later[0] + 1, 1])
