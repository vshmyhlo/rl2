from dataclasses import replace
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

from rl2 import karel_ast_edit as editing
from rl2.karel import KarelConfig, KarelProgramEnv, target_distance_map
from rl2.karel_ast import ACTION_ID, program_actions


@pytest.fixture
def config() -> editing.EditConfig:
    return editing.EditConfig(
        max_nodes=8,
        max_depth=4,
        max_seq_len=12,
        env=KarelConfig(
            height=3,
            width=3,
            max_depth=0,
            max_statements=1,
            max_program_tokens=8,
            syntax_weight=0,
            runtime_weight=0,
            trajectory_weight=0,
        ),
    )


def turning_task(config: editing.EditConfig) -> KarelProgramEnv:
    """Create a deterministic enclosed task whose target is one right turn."""
    task = KarelProgramEnv(config.env)
    task.reset(seed=12)
    initial = np.zeros((3, 3, 6), np.int32)
    initial[..., 4] = 1
    initial[1, 1, 4] = 0
    initial[1, 1, 0] = 1
    target = initial.copy()
    target[1, 1, 0] = 0
    target[1, 1, 1] = 1
    task._task = replace(
        task._task,
        initial=initial,
        target=target,
        distance_map=target_distance_map(target),
    )
    task._distance_map = task._task.distance_map
    return task


def fill_program(env: editing.KarelASTEditEnv, source: str = "DEF run m( turnLeft m)") -> editing.EditStep:
    """Replace the entire program with a deterministic candidate."""
    env.step(1)
    for action in program_actions(tuple(source.split())):
        transition = env.step(1 + env.config.max_nodes + action)
    return transition


def test_improvement_regression_and_stop(config: editing.EditConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    env = editing.KarelASTEditEnv(config)
    evaluate = MagicMock(wraps=editing.evaluate)
    monkeypatch.setattr(editing, "evaluate", evaluate)
    initial = env.reset(task=turning_task(config))
    assert initial.kind == "executed"
    assert initial.feedback[-2] == 0 and evaluate.call_count == 1
    assert env.remaining == 7  # Four initial EDIT tokens and one FEEDBACK consumed.
    initial_score = env.result.score
    first = env.step(3)  # Select the primitive statement.
    assert first.reward == 0 and first.observation.kind == "editing"
    assert isinstance(first.observation, editing.EditingObservation)
    assert first.observation.action_mask[1 + config.max_nodes + ACTION_ID["turnRight"]]
    assert evaluate.call_count == 1
    improved = env.step(1 + config.max_nodes + ACTION_ID["turnRight"])
    assert improved.observation.kind == "executed" and improved.reward > 0
    assert isinstance(improved.observation, editing.ExecutedObservation)
    assert improved.reward == pytest.approx(env.result.score - initial_score)
    assert improved.observation.feedback[-2] == pytest.approx(improved.reward)
    assert env.result.success and not env.done  # Success permits further edits.
    assert evaluate.call_count == 2
    assert env.remaining == 4  # Location + replacement + feedback consume three tokens.
    assert env.step(3).reward == 0
    regressed = env.step(1 + config.max_nodes + ACTION_ID["turnLeft"])
    assert regressed.reward == pytest.approx(-improved.reward)
    assert regressed.reward < 0 and evaluate.call_count == 3
    stopped = env.step(0)
    assert stopped.terminated and not stopped.truncated and stopped.reward == 0
    assert env.remaining == 0
    assert stopped.observation.kind == "executed" and evaluate.call_count == 3
    assert isinstance(stopped.observation, editing.ExecutedObservation)
    assert stopped.observation.feedback[-2] == pytest.approx(regressed.reward)
    assert stopped.observation.feedback[-1] == 0
    np.testing.assert_array_equal(stopped.observation.output, initial.output)
    assert improved.reward + regressed.reward == pytest.approx(env.result.score - initial_score)
    assert env.completed_edits == 2 and env.tree.complete
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)
    reset = env.reset(task=turning_task(config))
    assert env.completed_edits == 0 and env.remaining == config.max_seq_len - config.prefill_length
    assert reset.kind == "executed" and reset.feedback[-2] == 0


def test_runtime_failure_at_budget_boundary(config: editing.EditConfig) -> None:
    config = replace(config, max_seq_len=8)
    env = editing.KarelASTEditEnv(config)
    task = turning_task(config)
    initial_score = env.reset(task=task).feedback[0]
    assert env.step(3).reward == 0
    transition = env.step(1 + config.max_nodes + ACTION_ID["move"])
    assert transition.observation.kind == "executed" and transition.truncated and not transition.terminated
    assert isinstance(transition.observation, editing.ExecutedObservation)
    assert env.result.error == "runtime_error"
    assert transition.observation.feedback[2] == 1
    np.testing.assert_array_equal(transition.observation.output, task._task.initial)
    assert transition.reward == pytest.approx(env.result.score - initial_score)
    assert env.tree.complete and env.remaining == 0
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)


def test_unchanged_replacement_still_returns_execution(config: editing.EditConfig) -> None:
    config = replace(config, max_seq_len=8)
    env = editing.KarelASTEditEnv(config)
    initial = env.reset(task=turning_task(config))
    assert env.step(3).observation.kind == "editing"
    transition = env.step(1 + config.max_nodes + ACTION_ID["turnLeft"])
    assert transition.observation.kind == "executed"
    assert isinstance(transition.observation, editing.ExecutedObservation)
    assert transition.reward == 0 and transition.truncated and not transition.terminated
    assert transition.observation.feedback[-2] == 0
    assert transition.observation.feedback[-1] == 0
    np.testing.assert_array_equal(transition.observation.output, initial.output)
    assert env.completed_edits == 1


def test_initial_prefill_leaves_only_stop(config: editing.EditConfig) -> None:
    config = replace(config, max_seq_len=6)
    env = editing.KarelASTEditEnv(config)
    observation = env.reset(task=turning_task(config))
    assert env.remaining == 1
    assert np.flatnonzero(observation.action_mask).tolist() == [0]
    transition = env.step(0)
    assert transition.terminated and env.remaining == 0
    assert env.tree.complete and env.completed_edits == 0


@pytest.mark.parametrize("max_seq_len", [11, 12], ids=["exact-budget", "one-unusable-token"])
def test_disabled_stop_continues_until_no_edit_fits(config: editing.EditConfig, max_seq_len: int) -> None:
    config = replace(config, allow_stop=False, max_seq_len=max_seq_len)
    env = editing.KarelASTEditEnv(config)
    observation = env.reset(task=turning_task(config))
    assert not observation.action_mask[0]
    with pytest.raises(gym.error.InvalidAction):
        env.step(0)
    assert env.step(3).reward == 0
    improved = env.step(1 + config.max_nodes + ACTION_ID["turnRight"])
    assert env.result is not None and env.result.success
    assert not improved.terminated and not improved.truncated
    assert not improved.observation.action_mask[0]
    assert env.step(3).reward == 0
    regressed = env.step(1 + config.max_nodes + ACTION_ID["turnLeft"])
    assert regressed.truncated and not regressed.terminated
    assert regressed.observation.kind == "executed" and regressed.reward == pytest.approx(-improved.reward)
    assert not regressed.observation.action_mask.any()
    assert env.done and env.tree.complete and env.completed_edits == 2
    assert env.remaining == max_seq_len - 11
    with pytest.raises(gym.error.ResetNeeded):
        env.step(3)


def test_disabled_stop_requires_budget_for_an_edit(config: editing.EditConfig) -> None:
    with pytest.raises(ValueError, match="complete edit"):
        replace(config, allow_stop=False, max_seq_len=config.prefill_length + 2)
    minimum = replace(config, allow_stop=False, max_seq_len=config.prefill_length + 3)
    env = editing.KarelASTEditEnv(minimum)
    assert env.reset(task=turning_task(minimum)).action_mask[3]
    env.step(3)
    assert env.step(1 + minimum.max_nodes + ACTION_ID["turnRight"]).truncated


@pytest.mark.parametrize(
    "source,nodes,depth,ticks,error",
    [
        pytest.param("turnRight", 4, 2, 1, None, id="successful-primitive"),
        pytest.param("turnRight move", 6, 3, 2, "runtime_error", id="failed-attempt-counts"),
        pytest.param("REPEAT R=2 r( turnRight r)", 8, 4, 3, None, id="ast-and-execution-limits"),
        pytest.param("REPEAT R=3 r( turnRight r)", 8, 4, 3, "execution_limit", id="exhausted-execution"),
    ],
)
def test_normalized_program_penalties(
    config: editing.EditConfig, source: str, nodes: int, depth: int, ticks: int, error: str | None
) -> None:
    config = replace(
        config,
        max_seq_len=16,
        env=replace(
            config.env,
            depth_penalty_weight=0.2,
            max_program_tokens=12,
            max_execution_steps=3,
            length_penalty_weight=0.3,
            execution_penalty_weight=0.4,
        ),
    )
    task = turning_task(config)
    env = editing.KarelASTEditEnv(config)
    env.reset(task=task)
    observation = fill_program(env, f"DEF run m( {source} m)").observation
    assert isinstance(observation, editing.ExecutedObservation)
    result = env.result
    assert result is not None
    assert result.error == error and result.ticks == ticks
    assert result.components["depth"] == pytest.approx(-0.2 * depth / 4)
    assert result.components["length"] == pytest.approx(-0.3 * nodes / 8)
    assert result.components["execution"] == pytest.approx(-0.4 * ticks / 3)
    assert result.score == pytest.approx(sum(result.components.values()))
    assert observation.feedback[0] == pytest.approx(result.score)
    assert env.step(0).reward == 0

    # Zero weights recover task-only scores, including on failed executions.
    unpenalized = replace(
        config,
        env=replace(config.env, depth_penalty_weight=0.0, length_penalty_weight=0.0, execution_penalty_weight=0.0),
    )
    baseline = editing.KarelASTEditEnv(unpenalized)
    baseline.reset(task=turning_task(unpenalized))
    fill_program(baseline, f"DEF run m( {source} m)")
    assert baseline.result is not None
    assert baseline.result.score == pytest.approx(
        sum(value for name, value in result.components.items() if name not in ("depth", "length", "execution"))
    )


def test_penalties_reward_smaller_equivalent_programs(config: editing.EditConfig) -> None:
    config = replace(
        config,
        max_seq_len=24,
        env=replace(config.env, depth_penalty_weight=0.1, length_penalty_weight=0.1, execution_penalty_weight=0.1),
    )
    env = editing.KarelASTEditEnv(config)
    initial = env.reset(task=turning_task(config))
    initial_score = env.result.score

    def replace_program(source: str) -> float:
        assert env.step(1).reward == 0  # Replace the root.
        actions = program_actions(tuple(source.split()))
        for index, action in enumerate(actions):
            transition = env.step(1 + config.max_nodes + action)
            if index < len(actions) - 1:
                assert transition.reward == 0 and transition.observation.kind == "editing"
        assert transition.observation.kind == "executed"
        assert isinstance(transition.observation, editing.ExecutedObservation)
        np.testing.assert_array_equal(transition.observation.output, initial.output)
        assert transition.observation.feedback[-2] == pytest.approx(transition.reward)
        return transition.reward

    grown = replace_program("DEF run m( turnRight turnRight turnRight m)")
    # Equivalent output; only the extra four nodes, two depth edges and two ticks cost reward.
    assert grown == pytest.approx(-0.1 * (4 / 8 + 2 / 4 + 2 / config.env.max_execution_steps))
    shrunk = replace_program("DEF run m( turnLeft m)")
    assert shrunk == pytest.approx(-grown)
    assert env.result is not None and env.result.score == pytest.approx(initial_score)
    assert env.step(0).reward == 0


@pytest.mark.parametrize("weight", [-0.1, float("inf"), float("nan")], ids=["negative", "infinite", "nan"])
def test_invalid_depth_penalty(config: editing.EditConfig, weight: float) -> None:
    with pytest.raises(ValueError, match="depth_penalty_weight"):
        replace(config.env, depth_penalty_weight=weight)


def test_invalid_actions_and_reset_contract(config: editing.EditConfig) -> None:
    env = editing.KarelASTEditEnv(config)
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)
    task = turning_task(config)
    with pytest.raises(ValueError, match="either"):
        env.reset(task=task, seed=1)
    with pytest.raises(ValueError, match="limits"):
        env.reset(task=KarelProgramEnv(replace(config.env, max_execution_steps=1)))
    env.reset(task=task)
    before = (env.tree, env.remaining, env.result)
    for action in (True, -1, env.action_space.n, 1 + config.max_nodes + ACTION_ID["move"]):
        with pytest.raises(gym.error.InvalidAction):
            env.step(action)
        assert env.tree is before[0] and env.remaining == before[1] and env.result is before[2]
    env.step(3)
    with pytest.raises(gym.error.InvalidAction):
        env.step(0)  # Cannot stop with an unfinished replacement.
    assert env.reset(seed=12).kind == "executed"
    stopped = env.step(0)
    assert stopped.reward == 0 and stopped.terminated and env.completed_edits == 0
