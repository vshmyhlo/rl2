from dataclasses import replace
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

from rl2 import karel_ast_edit as editing
from rl2.karel import KarelConfig, KarelProgramEnv, target_distance_map
from rl2.karel_ast import ACTION_ID


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


def test_improvement_regression_and_stop(config: editing.EditConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    env = editing.KarelASTEditEnv(config)
    evaluate = MagicMock(wraps=editing.evaluate)
    monkeypatch.setattr(editing, "evaluate", evaluate)
    initial = env.reset(task=turning_task(config))
    assert initial.feedback[-2] == 0 and evaluate.call_count == 1
    assert env.remaining == 7  # Four seed tokens and one initial UPDATE consumed.
    seed_score = env.seed_score
    first = env.step(3)  # Select the primitive statement.
    assert first.reward == 0 and not first.reevaluated
    assert evaluate.call_count == 1
    improved = env.step(1 + config.max_nodes + ACTION_ID["turnRight"])
    assert improved.reevaluated and improved.reward > 0
    assert improved.reward == pytest.approx(env.result.score - seed_score)
    assert improved.observation.feedback[-2] == pytest.approx(improved.reward)
    assert env.result.success and not env.done  # Success permits further edits.
    assert evaluate.call_count == 2
    assert env.remaining == 4  # Location + replacement + execution UPDATE.
    assert env.step(3).reward == 0
    regressed = env.step(1 + config.max_nodes + ACTION_ID["turnLeft"])
    assert regressed.reward == pytest.approx(-improved.reward)
    assert regressed.reward < 0 and evaluate.call_count == 3
    stopped = env.step(0)
    assert stopped.terminated and not stopped.truncated and stopped.reward == 0
    assert env.remaining == 0
    assert not stopped.reevaluated and evaluate.call_count == 3
    assert improved.reward + regressed.reward == pytest.approx(env.result.score - seed_score)
    assert env.completed_edits == 2 and env.tree.complete
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)
    env.reset(task=turning_task(config))
    assert env.completed_edits == 0 and env.remaining == config.max_seq_len - config.prefill_length
    assert env.observation.feedback[-2] == 0


def test_runtime_failure_at_budget_boundary(config: editing.EditConfig) -> None:
    config = replace(config, max_seq_len=8)
    env = editing.KarelASTEditEnv(config)
    task = turning_task(config)
    env.reset(task=task)
    assert env.step(3).reward == 0
    transition = env.step(1 + config.max_nodes + ACTION_ID["move"])
    assert transition.reevaluated and transition.truncated and not transition.terminated
    assert env.result.error == "runtime_error"
    assert transition.observation.feedback[2] == 1
    np.testing.assert_array_equal(transition.observation.output, task._task.initial)
    assert transition.reward == pytest.approx(env.result.score - env.seed_score)
    assert env.tree.complete and env.remaining == 0
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)


def test_seed_prefill_leaves_only_stop(config: editing.EditConfig) -> None:
    config = replace(config, max_seq_len=6)
    env = editing.KarelASTEditEnv(config)
    observation = env.reset(task=turning_task(config))
    assert env.remaining == 1
    assert np.flatnonzero(observation.legal).tolist() == [0]
    transition = env.step(0)
    assert transition.terminated and env.remaining == 0
    assert env.tree.complete and env.completed_edits == 0


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
    env.reset(seed=12)
    stopped = env.step(0)
    assert stopped.reward == 0 and stopped.terminated and env.completed_edits == 0
