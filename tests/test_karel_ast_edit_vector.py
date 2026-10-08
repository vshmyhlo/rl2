from multiprocessing import active_children

import gymnasium as gym
import numpy as np
import pytest

from rl2.karel import KarelConfig, KarelProgramEnv
from rl2.karel_ast import ACTION_ID
from rl2.karel_ast_edit import EditConfig, EditStep
from rl2.karel_ast_edit_vector import KarelASTEditVectorEnv


def assert_transition_equal(actual: EditStep | None, expected: EditStep | None) -> None:
    """Compare public transitions, including all array-valued observations."""
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    assert actual[1:] == expected[1:]
    assert type(actual.observation) is type(expected.observation)
    assert actual.observation.kind == expected.observation.kind
    for left, right in zip(actual.observation, expected.observation, strict=True):
        np.testing.assert_array_equal(left, right)


def test_parallel_matches_local_masked_steps_and_cleans_up() -> None:
    # Three environments split over two workers exercises unequal shard sizes.
    config = EditConfig(
        max_nodes=8,
        max_depth=4,
        max_seq_len=11,
        env=KarelConfig(
            height=3, width=3, max_depth=0, max_statements=1, max_program_tokens=8, depth_penalty_weight=0.1
        ),
    )
    tasks = [KarelProgramEnv(config.env) for _ in range(3)]
    for index, task in enumerate(tasks):
        task.reset(seed=12 + index)
    children_before = set(active_children())
    with KarelASTEditVectorEnv(config, 3, workers=2) as parallel, KarelASTEditVectorEnv(config, 3) as local:
        for actual, expected in zip(parallel.reset(tasks), local.reset(tasks), strict=True):
            for left, right in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(left, right)
        children = set(active_children()) - children_before
        assert len(children) == 2
        offset = 1 + config.max_nodes
        commands = [
            ([0, 3, -99], [True, True, False]),  # STOP, open hole, pause.
            ([-99, offset + ACTION_ID["turnRight"], 3], [False, True, True]),
            ([-99, -99, offset + ACTION_ID["move"]], [False, False, True]),
            ([-99, 3, 0], [False, True, True]),
            ([-99, offset + ACTION_ID["turnLeft"], -99], [False, True, False]),
            ([-99, -99, -99], [False, False, False]),  # Completed members never autoreset.
        ]
        for actions, mask in commands:
            action_array, mask_array = np.asarray(actions, np.int32), np.asarray(mask, np.bool_)
            for actual, expected in zip(
                parallel.step(action_array, mask_array), local.step(action_array, mask_array), strict=True
            ):
                assert_transition_equal(actual, expected)
        actual, expected = parallel.summaries(), local.summaries()
        assert [summary.completed_edits for summary in actual] == [0, 2, 1]
        assert [summary.remaining for summary in actual] == [5, 0, 2]
        for left, right in zip(actual, expected, strict=True):
            assert left.tree == right.tree
            assert left[2:] == right[2:]
            assert left.result.score == right.result.score
            assert left.result.error == right.result.error
            assert left.result.components == right.result.components
            assert left.result.components["depth"] == pytest.approx(-0.1 * 2 / 4)
            np.testing.assert_array_equal(left.result.output, right.result.output)
        # Persistent workers reset cleanly for the next rollout and propagate errors.
        parallel.reset(tasks)
        with pytest.raises(gym.error.InvalidAction):
            parallel.step(np.asarray([-1, 0, 0], np.int32), np.ones(3, np.bool_))
        assert parallel.closed
    assert all(not child.is_alive() for child in children)
    parallel.close()  # Idempotent cleanup after a worker exception.
    with pytest.raises(RuntimeError, match="closed"):
        parallel.reset(tasks)
    with pytest.raises(RuntimeError, match="closed"):
        parallel.step(np.zeros(3, np.int32), np.ones(3, np.bool_))


def test_vector_input_validation() -> None:
    config = EditConfig()
    with pytest.raises(TypeError, match="integers"):
        KarelASTEditVectorEnv(config, 1, workers=True)
    with pytest.raises(AssertionError):
        KarelASTEditVectorEnv(config, 0)
    with pytest.raises(AssertionError):
        KarelASTEditVectorEnv(config, 1, workers=-1)
    with KarelASTEditVectorEnv(config, 1) as envs:
        with pytest.raises(ValueError, match="matching task"):
            envs.reset([])
        with pytest.raises(AssertionError):
            envs.step(np.zeros(2, np.int32), np.ones(2, np.bool_))
