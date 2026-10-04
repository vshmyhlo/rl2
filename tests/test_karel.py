from collections.abc import Sequence
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from rl2.karel import (
    DistanceMap,
    ExecutionStats,
    KarelConfig,
    KarelProgramEnv,
    KarelProgramError,
    KarelTask,
    State,
    StepResult,
    execute_program,
    progress_reward,
    sample_task,
    state_distance,
    target_distance_map,
)


@pytest.fixture
def world() -> State:
    state = np.zeros((5, 6, 6), dtype=np.int32)
    state[[0, -1], :, 4] = 1
    state[:, [0, -1], 4] = 1
    state[2, 1, 1] = 1  # Facing east in a four-cell corridor.
    state[2, 1:5, 5] = 1
    return state


def execute(body: str, world: State, *, max_steps: int = 256) -> State:
    return execute_program(f"DEF run m( {body} m)".split(), world, max_steps=max_steps)


def submit(env: KarelProgramEnv, tokens: Sequence[str]) -> StepResult:
    result: StepResult = (None, 0.0, False, False, {})
    for token in tokens:
        result = env.step(env.token_to_id[token])
    return result


def expected_info(reward: float, success: bool = False, error: str | None = None) -> dict[str, object]:
    """Expected components for legacy reward cases with trajectory_weight=0."""
    if error in ("syntax_error", "token_limit"):
        syntax, runtime, distance = reward, 0.0, 0.0
    elif error in ("runtime_error", "execution_limit"):
        syntax, runtime, distance = 1.0, 0.0, reward - 1.0
    else:
        syntax = 1.0
        runtime = 1.0
        distance = reward - 2.0 - float(success)
    return {
        "success": success,
        "error": error,
        "reward_syntax": syntax,
        "reward_runtime": runtime,
        "reward_distance": distance,
        "reward_success": float(success),
        "reward_trajectory": 0.0,
        "reward_length": 0.0,
        "reward_execution": 0.0,
    }


@pytest.fixture
def fixed_env(world: State, monkeypatch: pytest.MonkeyPatch) -> KarelProgramEnv:
    target = world.copy()
    target[2, 1, 1] = 0
    target[2, 2, 1] = 1
    task = KarelTask(world, target, ("DEF", "run", "m(", "move", "m)"))

    def fixed_task(rng: np.random.Generator, config: KarelConfig) -> KarelTask:
        return task

    monkeypatch.setattr("rl2.karel.sample_task", fixed_task)
    return KarelProgramEnv(KarelConfig(length_penalty_weight=0.0, execution_penalty_weight=0.0, trajectory_weight=0.0))


def test_corridor_program_collects_markers_without_mutating_input(world: State) -> None:
    before = world.copy()
    result = execute("pickMarker WHILE c( frontIsClear c) w( move pickMarker w)", world)
    expected = world.copy()
    expected[..., :4] = 0
    expected[2, 4, 1] = 1
    expected[..., 5] = 0
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(world, before)


def test_repeat_branch_negation_and_marker_actions(world: State) -> None:
    result = execute(
        "REPEAT R=3 r( move r) "
        "IFELSE c( not c( frontIsClear c) c) i( pickMarker i) ELSE e( putMarker e) "
        "IF c( noMarkersPresent c) i( putMarker i) "
        "turnLeft turnRight",
        world,
    )
    expected = world.copy()
    expected[..., :4] = 0
    expected[2, 4, 1] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("heading", range(4))
@pytest.mark.parametrize("predicate,offset", [("frontIsClear", 0), ("leftIsClear", -1), ("rightIsClear", 1)])
def test_relative_sensors_for_every_heading(world: State, heading: int, predicate: str, offset: int) -> None:
    world[..., :4] = 0
    world[2, 2, heading] = 1
    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
    dr, dc = directions[(heading + offset) % 4]
    world[2 + dr, 2 + dc, 4] = 1
    world[2 + dr, 2 + dc, 5] = 0
    result = execute(f"IFELSE c( {predicate} c) i( putMarker i) ELSE e( pickMarker e)", world)
    assert result[2, 2, 5] == 0


@pytest.mark.parametrize(
    "body",
    [
        "WHILE c( markersPresent c) w( turnLeft w)",
        "WHILE c( markersPresent c) w( IF c( noMarkersPresent c) i( move i) w)",
        "WHILE c( markersPresent c) w( REPEAT R=0 r( move r) w)",
    ],
)
def test_nonterminating_programs_hit_execution_limit(world: State, body: str) -> None:
    with pytest.raises(KarelProgramError) as error:
        execute(body, world, max_steps=20)
    assert error.value.reason == "execution_limit"


@pytest.mark.parametrize("body", ["REPEAT R=4 r( move r)", "pickMarker pickMarker", "REPEAT R=10 r( putMarker r)"])
def test_invalid_robot_actions_fail(world: State, body: str) -> None:
    before = world.copy()
    with pytest.raises(KarelProgramError) as error:
        execute(body, world)
    assert error.value.reason == "runtime_error"
    np.testing.assert_array_equal(world, before)


@pytest.mark.parametrize(
    "prefix,suffix,budget,reason",
    [
        ("move turnLeft move", "move", 256, "runtime_error"),
        ("move pickMarker", "pickMarker", 256, "runtime_error"),
        ("move REPEAT R=9 r( putMarker r)", "putMarker", 256, "runtime_error"),
        ("move turnLeft putMarker", "move", 3, "execution_limit"),
        ("move", "WHILE c( frontIsClear c) w( move w)", 3, "execution_limit"),
    ],
)
def test_execution_error_exposes_last_valid_state(
    world: State, prefix: str, suffix: str, budget: int, reason: str
) -> None:
    before = world.copy()
    expected = execute(prefix, world)
    with pytest.raises(KarelProgramError) as error:
        execute(f"{prefix} {suffix}", world, max_steps=budget)
    assert error.value.reason == reason
    assert error.value.partial_state is not None
    np.testing.assert_array_equal(error.value.partial_state, expected)
    np.testing.assert_array_equal(world, before)
    error.value.partial_state.fill(0)
    np.testing.assert_array_equal(world, before)


@pytest.mark.parametrize(
    "program",
    [
        "",
        "DEF run m( m)",
        "DEF run m( move",
        "DEF run m( move m) move",
        "DEF run m( REPEAT R=20 r( move r) m)",
        "DEF run m( IF c( move c) i( move i) m)",
        "DEF run m( IFELSE c( frontIsClear c) i( move i) m)",
    ],
)
def test_invalid_syntax_is_rejected(world: State, program: str) -> None:
    with pytest.raises(KarelProgramError) as error:
        execute_program(program.split(), world)
    assert error.value.reason == "syntax_error"
    assert error.value.partial_state is None


def test_excessive_nesting_is_rejected_before_execution(world: State) -> None:
    body = "REPEAT R=1 r( " * 65 + "move " + "r) " * 65
    with pytest.raises(KarelProgramError, match="nesting"):
        execute(body, world)


def test_reset_pair_and_terminal_only_evaluation(fixed_env: KarelProgramEnv) -> None:
    initial, target = fixed_env.reset()
    assert initial.shape == target.shape == (5, 6, 6)
    assert initial.dtype == target.dtype == np.int32
    for token in ["DEF", "run", "m(", "move"]:
        assert fixed_env.step(fixed_env.token_to_id[token]) == (None, 0.0, False, False, {})
    assert fixed_env.step(fixed_env.terminal_token_id) == (None, 4.0, True, False, expected_info(4.0, success=True))


def test_equivalent_program_gets_full_reward(fixed_env: KarelProgramEnv) -> None:
    fixed_env.reset()
    result = submit(fixed_env, ["DEF", "run", "m(", "turnLeft", "turnRight", "move", "m)"])
    assert result == (None, 4.0, True, False, expected_info(4.0, success=True))


@pytest.mark.parametrize("body,reward", [("move turnLeft", 2.5), ("move pickMarker", 2.5), ("turnLeft", 2.0)])
def test_target_matching_checks_heading_markers_and_position(
    fixed_env: KarelProgramEnv, body: str, reward: float
) -> None:
    fixed_env.reset()
    result = submit(fixed_env, f"DEF run m( {body} m)".split())
    assert result == (None, reward, True, False, expected_info(reward))


@pytest.mark.parametrize(
    "program,reason,reward",
    [
        ("m)", "syntax_error", 0.2),  # Four missing tokens.
        ("DEF run m( WHILE c( frontIsClear c) w( move m)", "syntax_error", 0.5),  # Missing w).
        ("DEF run m( pickMarker pickMarker m)", "runtime_error", 1.0),
        ("DEF run m( WHILE c( markersPresent c) w( turnLeft w) m)", "execution_limit", 1.0),
    ],
)
def test_failed_program_rewards_distinguish_syntax(
    fixed_env: KarelProgramEnv, program: str, reason: str, reward: float
) -> None:
    fixed_env.reset()
    result = submit(fixed_env, program.split())
    assert result == (None, reward, True, False, expected_info(reward, error=reason))


def test_observations_cannot_mutate_private_task(fixed_env: KarelProgramEnv) -> None:
    pair = fixed_env.reset()
    pair.initial.fill(0)
    pair.target.fill(0)
    result = submit(fixed_env, ["DEF", "run", "m(", "move", "m)"])
    assert result[1] == 4.0


def test_reset_discards_partial_program(fixed_env: KarelProgramEnv) -> None:
    fixed_env.reset()
    submit(fixed_env, ["IF", "move"])
    fixed_env.reset()
    assert submit(fixed_env, ["DEF", "run", "m(", "move", "m)"])[1] == 4.0


def test_token_limit_and_terminal_boundary(fixed_env: KarelProgramEnv) -> None:
    # The fixed reference uses five tokens including m), exactly the limit.
    env = KarelProgramEnv(
        KarelConfig(
            length_penalty_weight=0.0, execution_penalty_weight=0.0, trajectory_weight=0.0, max_program_tokens=5
        )
    )
    env.reset()
    assert submit(env, ["DEF", "run", "m(", "move", "m)"])[1] == 4.0
    env.reset()
    assert submit(env, ["move"] * 5) == (None, 0.2, False, True, expected_info(0.2, error="token_limit"))
    with pytest.raises(gym.error.ResetNeeded):
        env.step(env.terminal_token_id)


def test_incomplete_program_gets_syntax_credit_without_execution(fixed_env: KarelProgramEnv) -> None:
    env = KarelProgramEnv(
        KarelConfig(
            length_penalty_weight=0.0, execution_penalty_weight=0.0, trajectory_weight=0.0, max_program_tokens=6
        )
    )
    env.reset()
    # Only m) is missing. The completed program would solve this task, but must
    # never be repaired and executed when determining the reward.
    result = submit(env, ["DEF", "run", "m(", "turnLeft", "turnRight", "move"])
    assert result == (None, 0.5, False, True, expected_info(0.5, error="token_limit"))


def test_episode_lifecycle_and_invalid_action(fixed_env: KarelProgramEnv) -> None:
    with pytest.raises(gym.error.ResetNeeded):
        fixed_env.step(0)
    with pytest.raises(gym.error.ResetNeeded):
        _ = fixed_env.reference_program
    fixed_env.reset()
    for invalid in (-1, len(fixed_env.tokens), 1.5, True, "move", fixed_env.pad_token_id):
        with pytest.raises(gym.error.InvalidAction):
            fixed_env.step(invalid)
    fixed_env.step(np.int64(fixed_env.terminal_token_id))
    with pytest.raises(gym.error.ResetNeeded):
        fixed_env.step(0)


def test_pad_is_reserved_and_cannot_change_program(fixed_env: KarelProgramEnv) -> None:
    assert "<eos>" not in fixed_env.tokens
    assert fixed_env.pad_token_id != fixed_env.terminal_token_id
    fixed_env.reset()
    assert fixed_env.reference_program[-1] == fixed_env.terminal_token_id
    with pytest.raises(gym.error.InvalidAction, match="PAD"):
        fixed_env.step(fixed_env.pad_token_id)
    assert submit(fixed_env, ["DEF", "run", "m(", "move", "m)"])[1] == 4.0


@pytest.mark.parametrize("seed", range(20))
def test_sampled_task_is_reachable_changed_and_within_limits(seed: int) -> None:
    config = KarelConfig(
        length_penalty_weight=0.0,
        execution_penalty_weight=0.0,
        trajectory_weight=0.0,
    )
    task = sample_task(np.random.default_rng(seed), config)
    assert len(task.program) <= config.max_program_tokens
    assert not np.array_equal(task.initial, task.target)
    np.testing.assert_array_equal(execute_program(task.program, task.initial), task.target)
    assert np.all(task.initial[[0, -1], :, 4] == 1)
    assert np.all(task.initial[:, [0, -1], 4] == 1)


def test_seed_reproduces_task_stream_and_reference_solves() -> None:
    first, second = (
        KarelProgramEnv(KarelConfig(length_penalty_weight=0.0, execution_penalty_weight=0.0, trajectory_weight=0.0)),
        KarelProgramEnv(KarelConfig(length_penalty_weight=0.0, execution_penalty_weight=0.0, trajectory_weight=0.0)),
    )
    pairs: set[bytes] = set()
    for episode in range(10):
        seed = 42 if episode == 0 else None
        pair_a, pair_b = first.reset(seed=seed), second.reset(seed=seed)
        np.testing.assert_array_equal(pair_a.initial, pair_b.initial)
        np.testing.assert_array_equal(pair_a.target, pair_b.target)
        assert first.reference_program == second.reference_program
        pairs.add(pair_a.initial.tobytes() + pair_a.target.tobytes())
        for token_id in first.reference_program:
            observation, reward, terminated, truncated, info = first.step(token_id)
            assert observation is None
        assert (reward, terminated, truncated, info) == (4.0, True, False, expected_info(4.0, success=True))
    assert len(pairs) == 10


def test_depth_zero_samples_only_primitive_actions() -> None:
    config = KarelConfig(length_penalty_weight=0.0, execution_penalty_weight=0.0, trajectory_weight=0.0, max_depth=0)
    for seed in range(10):
        task = sample_task(np.random.default_rng(seed), config)
        assert not {"WHILE", "REPEAT", "IF", "IFELSE"}.intersection(task.program)


def test_sampling_failure_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    def identity_program(rng: np.random.Generator, config: KarelConfig) -> tuple[str, ...]:
        return ("DEF", "run", "m(", "turnLeft", "turnRight", "m)")

    monkeypatch.setattr("rl2.karel._sample_program", identity_program)
    with pytest.raises(RuntimeError, match="max_sampling_attempts"):
        sample_task(
            np.random.default_rng(0),
            KarelConfig(
                length_penalty_weight=0.0, execution_penalty_weight=0.0, trajectory_weight=0.0, max_sampling_attempts=2
            ),
        )


@pytest.mark.parametrize(
    "options",
    [
        {"height": 2},
        {"width": 0},
        {"max_depth": -1},
        {"max_depth": 33},
        {"max_statements": 0},
        {"max_program_tokens": 4},
        {"max_execution_steps": 0},
        {"wall_probability": 1.1},
        {"marker_probability": -0.1},
        {"max_markers": 0},
        {"max_sampling_attempts": 0},
        {"max_depth": 1.5},
        {"height": True},
    ],
)
def test_config_validation(options: dict[str, int | float]) -> None:
    with pytest.raises((AssertionError, TypeError, ValueError)):
        KarelConfig(length_penalty_weight=0.0, execution_penalty_weight=0.0, trajectory_weight=0.0, **options)


def test_world_shape_dtype_and_robot_validation(world: State) -> None:
    with pytest.raises(AssertionError):
        execute("move", world.astype(np.float32))
    with pytest.raises(AssertionError):
        execute("move", world[..., :5])
    world[1, 1, 0] = 1
    with pytest.raises(ValueError, match="exactly one"):
        execute("move", world)


@pytest.mark.parametrize("heading,destination", [(0, (1, 2)), (1, (2, 3)), (2, (3, 2)), (3, (2, 1))])
def test_movement_and_marker_update_in_every_direction(
    world: State, heading: int, destination: tuple[int, int]
) -> None:
    world[..., :4] = 0
    world[2, 2, heading] = 1
    result = execute("move putMarker", world)
    expected = world.copy()
    expected[..., :4] = 0
    row, col = destination
    expected[row, col, heading] = 1
    expected[row, col, 5] += 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("heading", range(4))
def test_unwalled_array_edges_are_blocked(heading: int) -> None:
    world = np.zeros((1, 1, 6), dtype=np.int32)
    world[0, 0, heading] = 1
    result = execute("IFELSE c( frontIsClear c) i( move i) ELSE e( putMarker e)", world)
    assert result[0, 0, 5] == 1
    with pytest.raises(KarelProgramError, match="out of bounds"):
        execute("move", world)


@pytest.mark.parametrize(
    "body,required_steps",
    [
        ("move", 1),
        ("REPEAT R=2 r( move r)", 3),
        ("WHILE c( frontIsClear c) w( move w)", 8),
    ],
)
def test_exact_execution_budget(world: State, body: str, required_steps: int) -> None:
    expected = execute(body, world)
    np.testing.assert_array_equal(execute(body, world, max_steps=required_steps), expected)
    if required_steps > 1:
        with pytest.raises(KarelProgramError) as error:
            execute(body, world, max_steps=required_steps - 1)
        assert error.value.reason == "execution_limit"


def test_zero_repeat_and_false_branch_skip_invalid_actions(world: State) -> None:
    result = execute(
        "REPEAT R=0 r( REPEAT R=19 r( move r) r) "
        "IF c( noMarkersPresent c) i( REPEAT R=19 r( move r) i) "
        "pickMarker IF c( not c( markersPresent c) c) i( putMarker i)",
        world,
    )
    np.testing.assert_array_equal(result, world)


@pytest.mark.parametrize("seed", range(5))
def test_minimum_task_and_submission_budgets(seed: int) -> None:
    env = KarelProgramEnv(
        KarelConfig(
            length_penalty_weight=0.0,
            execution_penalty_weight=0.0,
            trajectory_weight=0.0,
            height=3,
            width=3,
            wall_probability=1.0,
            marker_probability=1.0,
            max_markers=1,
            max_program_tokens=5,
            max_execution_steps=1,
        )
    )
    initial, target = env.reset(seed=seed)
    assert not np.array_equal(initial, target)
    assert len(env.reference_program) == 5
    for token_id in env.reference_program:
        result = env.step(token_id)
    assert result == (None, 4.0, True, False, expected_info(4.0, success=True))


def test_failed_reset_invalidates_old_task_and_can_recover(monkeypatch: pytest.MonkeyPatch) -> None:
    env = KarelProgramEnv(KarelConfig(length_penalty_weight=0.0, execution_penalty_weight=0.0, trajectory_weight=0.0))
    env.reset(seed=1)
    env.step(env.token_to_id["DEF"])

    def fail_sampling(rng: np.random.Generator, config: KarelConfig) -> KarelTask:
        raise RuntimeError("Sampling failed")

    with monkeypatch.context() as patch:
        patch.setattr("rl2.karel.sample_task", fail_sampling)
        with pytest.raises(RuntimeError, match="Sampling failed"):
            env.reset()
    with pytest.raises(gym.error.ResetNeeded):
        env.step(env.terminal_token_id)
    with pytest.raises(gym.error.ResetNeeded):
        _ = env.reference_program
    env.reset(seed=2)
    for token_id in env.reference_program:
        result = env.step(token_id)
    assert result[1:4] == (4.0, True, False)


@pytest.mark.parametrize(
    "body,reward,success,error",
    [
        ("move", 2.75, False, None),
        ("turnLeft turnRight", 2.5, False, None),
        ("turnLeft", 2.25, False, None),
        ("turnLeft move putMarker", 2.0, False, None),
        ("move move", 4.0, True, None),
        ("move putMarker", 2.5, False, None),  # Undo progress by spoiling a correct cell.
        ("move move pickMarker", 2.75, False, None),
        ("move move move move", 1.75, False, "runtime_error"),  # Latest state, not the earlier exact target.
    ],
)
def test_terminal_progress_reward(
    world: State, monkeypatch: pytest.MonkeyPatch, body: str, reward: float, success: bool, error: str | None
) -> None:
    target = world.copy()
    target[..., :4] = 0
    target[2, 3, 1] = 1

    def fixed_task(rng: np.random.Generator, config: KarelConfig) -> KarelTask:
        return KarelTask(world, target, ("DEF", "run", "m(", "move", "move", "m)"))

    monkeypatch.setattr("rl2.karel.sample_task", fixed_task)
    env = KarelProgramEnv(KarelConfig(length_penalty_weight=0.0, execution_penalty_weight=0.0, trajectory_weight=0.0))
    env.reset()
    for token in f"DEF run m( {body}".split():
        assert env.step(env.token_to_id[token]) == (None, 0.0, False, False, {})
    assert env.step(env.terminal_token_id) == (None, reward, True, False, expected_info(reward, success, error))


@pytest.mark.parametrize(
    "action,markers,budget,reason",
    [
        ("pickMarker", 0, 256, "runtime_error"),
        ("putMarker", 10, 256, "runtime_error"),
        ("REPEAT R=0 r( move r) turnLeft", 1, 2, "execution_limit"),
    ],
)
def test_distance_score_uses_partial_progress_after_failure(
    world: State, monkeypatch: pytest.MonkeyPatch, action: str, markers: int, budget: int, reason: str
) -> None:
    world[2, 2, 5] = markers
    target = execute("move move", world)

    def fixed_task(rng: np.random.Generator, config: KarelConfig) -> KarelTask:
        return KarelTask(world, target, ("DEF", "run", "m(", "move", "move", "m)"))

    monkeypatch.setattr("rl2.karel.sample_task", fixed_task)
    env = KarelProgramEnv(
        KarelConfig(
            length_penalty_weight=0.0, execution_penalty_weight=0.0, trajectory_weight=0.0, max_execution_steps=budget
        )
    )
    env.reset()
    result = submit(env, f"DEF run m( move {action} m)".split())
    # Syntax=1, runtime=0, distance progress=0.75 on failure.
    assert result == (None, 1.75, True, False, expected_info(1.75, error=reason))


@pytest.mark.parametrize(
    "prefix,suffix,budget,reason",
    [("move turnLeft move", "move", 256, "runtime_error"), ("move", "turnLeft", 1, "execution_limit")],
)
def test_failure_at_exact_target_has_no_distance_or_success_bonus(
    world: State, monkeypatch: pytest.MonkeyPatch, prefix: str, suffix: str, budget: int, reason: str
) -> None:
    program = tuple(f"DEF run m( {prefix} m)".split())
    target = execute(prefix, world)

    def fixed_task(rng: np.random.Generator, config: KarelConfig) -> KarelTask:
        return KarelTask(world, target, program)

    monkeypatch.setattr("rl2.karel.sample_task", fixed_task)
    env = KarelProgramEnv(
        KarelConfig(
            length_penalty_weight=0.0, execution_penalty_weight=0.0, trajectory_weight=0.0, max_execution_steps=budget
        )
    )
    env.reset()
    assert submit(env, f"DEF run m( {prefix} {suffix} m)".split()) == (
        None,
        2.0,
        True,
        False,
        expected_info(2.0, error=reason),
    )
    env.reset()
    assert submit(env, program) == (None, 4.0, True, False, expected_info(4.0, success=True))


def test_distance_weights_and_marker_counts(world: State) -> None:
    target = world.copy()
    target[..., :4] = 0
    target[1, 3, 2] = 1
    target[2, 1, 5] = 4
    target[3, 3, 5] = 2
    config = KarelConfig(
        length_penalty_weight=0.0,
        execution_penalty_weight=0.0,
        trajectory_weight=0.0,
        position_weight=2.0,
        orientation_weight=3.0,
        marker_weight=4.0,
    )
    # Three free-cell moves, one heading mismatch, and five marker edits.
    assert state_distance(world, target, config) == 2 * 3 + 3 * 1 + 4 * 5
    assert progress_reward(world, world, target, config) == 0.5
    assert progress_reward(world, target, target, config) == 1.0
    final = world.copy()
    final[2, 1, 5] += 1
    assert progress_reward(world, final, target, config) == pytest.approx(33 / 58)


@pytest.mark.parametrize("error,score", [(0, 1.0), (1, 0.75), (2, 0.5), (3, 0.25), (4, 0.0), (5, 0.0)])
def test_normalized_progress_preserves_regressions(world: State, error: int, score: float) -> None:
    target = world.copy()
    target[2, 1, 5] += 2  # Initial distance is two markers.
    final = target.copy()
    final[2, 1, 5] += error
    assert (
        progress_reward(
            world,
            final,
            target,
            KarelConfig(
                length_penalty_weight=0.0,
                execution_penalty_weight=0.0,
                trajectory_weight=0.0,
            ),
        )
        == score
    )


@pytest.mark.parametrize("weight_name", ["position_weight", "orientation_weight", "marker_weight"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_reward_weights_must_be_positive_and_finite(weight_name: str, value: float) -> None:
    with pytest.raises((AssertionError, ValueError)):
        KarelConfig(
            length_penalty_weight=0.0, execution_penalty_weight=0.0, trajectory_weight=0.0, **{weight_name: value}
        )


def test_reward_rejects_zero_baseline_and_changed_walls(world: State) -> None:
    with pytest.raises(ValueError, match="non-identical"):
        progress_reward(
            world,
            world,
            world,
            KarelConfig(
                length_penalty_weight=0.0,
                execution_penalty_weight=0.0,
                trajectory_weight=0.0,
            ),
        )
    changed = world.copy()
    changed[1, 1, 4] = 1
    with pytest.raises(ValueError, match="identical walls"):
        state_distance(
            world,
            changed,
            KarelConfig(
                length_penalty_weight=0.0,
                execution_penalty_weight=0.0,
                trajectory_weight=0.0,
            ),
        )


@pytest.mark.parametrize(
    "suffix,budget,error", [("", 256, None), ("move", 256, "runtime_error"), ("move", 3, "execution_limit")]
)
def test_wall_detour_progress_and_reset_cache(
    world: State, monkeypatch: pytest.MonkeyPatch, suffix: str, budget: int, error: str | None
) -> None:
    initial = world.copy()
    initial[..., :4] = 0
    initial[1, 1, 1] = 1
    initial[1:3, 2, 4] = 1  # Must travel down to row 3 to cross this wall.
    initial[1:3, 2, 5] = 0
    target = initial.copy()
    target[..., :4] = 0
    target[1, 3, 1] = 1
    prefix = "turnRight move turnLeft"
    final = execute(prefix, initial)
    config = KarelConfig(
        length_penalty_weight=0.0, execution_penalty_weight=0.0, trajectory_weight=0.0, max_execution_steps=budget
    )
    distances = target_distance_map(target)
    assert distances[1, 1] == 6
    assert distances[2, 1] == 5
    assert distances[1, 3] == 0
    assert distances[1, 2] == -1
    assert state_distance(initial, target, config) == 6.0
    assert state_distance(final, target, config, distance_map=distances) == 5.0
    assert progress_reward(initial, final, target, config) == pytest.approx(7 / 12)

    # A new target on reset must replace the previous episode's cached map.
    targets = iter([target, final])
    map_calls: list[State] = []

    def fixed_task(rng: np.random.Generator, config: KarelConfig) -> KarelTask:
        return KarelTask(initial, next(targets), tuple(f"DEF run m( {prefix} m)".split()))

    def counted_map(target: State) -> DistanceMap:
        map_calls.append(target)
        return target_distance_map(target)

    monkeypatch.setattr("rl2.karel.sample_task", fixed_task)
    monkeypatch.setattr("rl2.karel.target_distance_map", counted_map)
    env = KarelProgramEnv(config)
    env.reset()
    assert len(map_calls) == 1
    result = submit(env, f"DEF run m( {prefix} {suffix} m)".split())
    expected_runtime = float(error is None)
    assert result[1] == pytest.approx(1 + expected_runtime + 7 / 12)
    assert result[2:4] == (True, False)
    assert result[4]["reward_runtime"] == expected_runtime
    assert result[4]["reward_distance"] == pytest.approx(7 / 12)
    assert result[4]["reward_success"] == 0.0
    assert result[4]["error"] == error
    assert len(map_calls) == 1  # Evaluation reuses reset's map for both states.
    env.reset()
    assert len(map_calls) == 2
    assert submit(env, f"DEF run m( {prefix} m)".split())[1] == 4.0
    assert len(map_calls) == 2


def test_distance_map_handles_open_edges() -> None:
    target = np.zeros((2, 3, 6), dtype=np.int32)
    target[0, 0, 0] = 1
    np.testing.assert_array_equal(target_distance_map(target), [[0, 1, 2], [1, 2, 3]])


def test_distance_rejects_unreachable_robot_and_wall_target(world: State) -> None:
    initial = world.copy()
    initial[1:4, 2, 4] = 1  # Completely separates robot and target.
    initial[1:4, 2, 5] = 0
    target = initial.copy()
    target[..., :4] = 0
    target[2, 3, 1] = 1
    assert target_distance_map(target)[2, 1] == -1
    with pytest.raises(ValueError, match="cannot reach"):
        state_distance(
            initial,
            target,
            KarelConfig(
                length_penalty_weight=0.0,
                execution_penalty_weight=0.0,
                trajectory_weight=0.0,
            ),
        )
    with pytest.raises(ValueError, match="cannot reach"):
        progress_reward(
            initial,
            target,
            target,
            KarelConfig(
                length_penalty_weight=0.0,
                execution_penalty_weight=0.0,
                trajectory_weight=0.0,
            ),
        )
    target[2, 3, 4] = 1
    with pytest.raises(ValueError, match="free cell"):
        target_distance_map(target)


@pytest.mark.parametrize(
    "body,budget,base,bonus,error",
    [
        ("pickMarker", 256, 2.75, 0.125, None),
        ("pickMarker pickMarker", 256, 4.0, 0.25, None),
        ("putMarker pickMarker pickMarker pickMarker", 256, 4.0, 1 / 6, None),
        ("pickMarker putMarker pickMarker pickMarker", 256, 4.0, 1 / 6, None),
        ("turnLeft pickMarker turnRight pickMarker", 256, 4.0, 1 / 6, None),
        ("pickMarker putMarker", 256, 2.5, 0.0, None),
        ("putMarker", 256, 2.25, 0.0, None),
        ("REPEAT R=0 r( pickMarker r)", 256, 2.5, 0.0, None),
        ("pickMarker pickMarker pickMarker", 256, 2.0, 0.25, "runtime_error"),
        ("pickMarker move", 256, 1.75, 0.125, "runtime_error"),
        ("pickMarker pickMarker", 1, 1.75, 0.125, "execution_limit"),
        ("pickMarker pickMarker putMarker", 2, 2.0, 0.25, "execution_limit"),
        ("", 256, 0.5, 0.0, "syntax_error"),
    ],
)
def test_trajectory_bonus_credits_net_progress_and_penalizes_reversals(
    monkeypatch: pytest.MonkeyPatch, body: str, budget: int, base: float, bonus: float, error: str | None
) -> None:
    initial = np.zeros((3, 3, 6), dtype=np.int32)
    initial[..., 4] = 1
    initial[1, 1, 4] = 0
    initial[1, 1, 0] = 1
    initial[1, 1, 5] = 2
    target = initial.copy()
    target[1, 1, 5] = 0

    def fixed_task(rng: np.random.Generator, config: KarelConfig) -> KarelTask:
        return KarelTask(initial, target, ("DEF", "run", "m(", "pickMarker", "pickMarker", "m)"))

    monkeypatch.setattr("rl2.karel.sample_task", fixed_task)
    env = KarelProgramEnv(
        KarelConfig(length_penalty_weight=0.0, execution_penalty_weight=0.0, max_execution_steps=budget)
    )
    env.reset()
    result = submit(env, f"DEF run m( {body} m)".split())
    assert result[1] == pytest.approx(base + bonus)
    assert result[4]["reward_trajectory"] == pytest.approx(bonus)
    assert result[4]["error"] == error
    assert result[4]["success"] == (base == 4.0)
    assert sum(value for key, value in result[4].items() if key.startswith("reward_")) == pytest.approx(result[1])
    # Disabling the bonus restores the original score on exactly the same task.
    disabled = KarelProgramEnv(
        KarelConfig(
            length_penalty_weight=0.0, execution_penalty_weight=0.0, max_execution_steps=budget, trajectory_weight=0.0
        )
    )
    disabled.reset()
    assert submit(disabled, f"DEF run m( {body} m)".split())[1] == pytest.approx(base)


def test_trajectory_token_limit_never_executes(fixed_env: KarelProgramEnv) -> None:
    from dataclasses import replace

    fixed_env.config = replace(fixed_env.config, trajectory_weight=0.25, max_program_tokens=5)
    fixed_env.reset()
    result = submit(fixed_env, ["DEF", "run", "m(", "move", "move"])
    assert result[3]
    assert result[4]["reward_trajectory"] == 0.0
    assert result[4]["error"] == "token_limit"


def test_action_observer_gets_independent_primitive_snapshots(world: State) -> None:
    snapshots: list[State] = []

    def observe(state: State) -> None:
        snapshots.append(state.copy())
        state.fill(0)  # Mutating a callback snapshot must not alter execution.

    program = "DEF run m( REPEAT R=2 r( move r) turnLeft m)"
    output = execute_program(program.split(), world, on_action=observe)
    assert len(snapshots) == 3
    for actual, body in zip(snapshots, ("move", "move move", "move move turnLeft")):
        np.testing.assert_array_equal(actual, execute(body, world))
    np.testing.assert_array_equal(output, snapshots[-1])


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_invalid_trajectory_weight(weight: float) -> None:
    with pytest.raises((ValueError, AssertionError)):
        KarelConfig(length_penalty_weight=0.0, execution_penalty_weight=0.0, trajectory_weight=weight)


@pytest.mark.parametrize(
    "program,budget,steps,error",
    [
        ("DEF run m( move m)", 256, 1, None),
        ("DEF run m( REPEAT R=0 r( move r) m)", 256, 1, None),
        ("DEF run m( WHILE c( frontIsClear c) w( move w) m)", 256, 8, None),
        ("DEF run m( pickMarker pickMarker m)", 256, 2, "runtime_error"),
        ("DEF run m( REPEAT R=9 r( turnLeft r) m)", 3, 3, "execution_limit"),
        ("m)", 256, 0, "syntax_error"),
    ],
)
def test_execution_stats_count_budget_ticks(
    world: State, program: str, budget: int, steps: int, error: str | None
) -> None:
    stats = ExecutionStats(steps=999)  # Reused stats must reset, including on syntax failure.
    if error is None:
        execute_program(program.split(), world, max_steps=budget, execution_stats=stats)
    else:
        with pytest.raises(KarelProgramError, match=".") as caught:
            execute_program(program.split(), world, max_steps=budget, execution_stats=stats)
        assert caught.value.reason == error
    assert stats.steps == steps


@pytest.mark.parametrize(
    "program,limit,steps,error",
    [
        ("DEF run m( move m)", 256, 1, None),
        ("DEF run m( REPEAT R=8 r( turnLeft r) move m)", 256, 10, None),
        ("DEF run m( turnLeft turnLeft turnLeft turnLeft turnLeft turnLeft turnLeft turnLeft move m)", 256, 9, None),
        ("DEF run m( pickMarker pickMarker m)", 256, 2, "runtime_error"),
        ("DEF run m( REPEAT R=9 r( turnLeft r) m)", 3, 3, "execution_limit"),
        ("m)", 256, 0, "syntax_error"),
    ],
)
def test_efficiency_penalties_are_separate_and_sum_into_reward(
    fixed_env: KarelProgramEnv, program: str, limit: int, steps: int, error: str | None
) -> None:
    from dataclasses import replace

    fixed_env.config = replace(fixed_env.config, max_execution_steps=limit)
    fixed_env.reset()
    baseline = submit(fixed_env, program.split())
    fixed_env.config = replace(fixed_env.config, length_penalty_weight=0.05, execution_penalty_weight=0.05)
    fixed_env.reset()
    result = submit(fixed_env, program.split())
    length_term = -0.05 * len(program.split()) / fixed_env.config.max_program_tokens
    execution_term = -0.05 * steps / limit
    assert result[4]["reward_length"] == pytest.approx(length_term)
    assert result[4]["reward_execution"] == pytest.approx(execution_term)
    assert result[1] == pytest.approx(baseline[1] + length_term + execution_term)
    assert result[4]["error"] == error
    assert result[4]["success"] == baseline[4]["success"]
    assert sum(value for key, value in result[4].items() if key.startswith("reward_")) == pytest.approx(result[1])


def test_token_limit_applies_full_length_penalty_without_execution(fixed_env: KarelProgramEnv) -> None:
    from dataclasses import replace

    fixed_env.config = replace(
        fixed_env.config, max_program_tokens=5, length_penalty_weight=0.05, execution_penalty_weight=0.05
    )
    fixed_env.reset()
    result = submit(fixed_env, ["DEF", "run", "m(", "move", "move"])
    assert result[3] and result[4]["error"] == "token_limit"
    assert result[4]["reward_length"] == -0.05
    assert result[4]["reward_execution"] == 0.0
    assert result[1] == pytest.approx(result[4]["reward_syntax"] - 0.05)


@pytest.mark.parametrize("field", ["length_penalty_weight", "execution_penalty_weight"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_efficiency_weights_must_be_finite_and_nonnegative(field: str, value: float) -> None:
    with pytest.raises((ValueError, AssertionError)):
        KarelConfig(**{field: value})


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("suffix", ["", "move", "REPEAT R=19 r( REPEAT R=19 r( turnLeft r) r)"])
def test_combined_default_rewards_match_recorded_execution(seed: int, suffix: str) -> None:
    config = KarelConfig(max_depth=1, max_statements=2, max_execution_steps=32)
    env = KarelProgramEnv(config)
    pair = env.reset(seed=seed)
    tokens = [env.tokens[index] for index in env.reference_program[:-1]] + suffix.split() + ["m)"]
    assert len(tokens) <= config.max_program_tokens
    trace: list[State] = [pair.initial]
    stats = ExecutionStats()
    error = None
    try:
        final = execute_program(
            tokens, pair.initial, max_steps=config.max_execution_steps, execution_stats=stats, on_action=trace.append
        )
    except KarelProgramError as exc:
        error = exc.reason
        assert exc.partial_state is not None
        final = exc.partial_state
    distances = np.asarray([state_distance(state, pair.target, config) for state in trace])
    final_distance = state_distance(final, pair.target, config)
    assert distances[-1] == final_distance
    backward = np.maximum(np.diff(distances), 0).sum()
    progress = float(np.clip(1 - 0.5 * final_distance / distances[0], 0, 1))
    success = error is None and np.array_equal(final, pair.target)
    expected = {
        "reward_syntax": 1.0,
        "reward_runtime": float(error is None),
        "reward_distance": progress,
        "reward_success": float(success),
        "reward_trajectory": config.trajectory_weight
        * max(0.0, distances[0] - final_distance)
        / (distances[0] + backward),
        "reward_length": -config.length_penalty_weight * len(tokens) / config.max_program_tokens,
        "reward_execution": -config.execution_penalty_weight * stats.steps / config.max_execution_steps,
    }
    result = submit(env, tokens)
    assert result[2:4] == (True, False)
    assert result[4]["success"] == success
    assert result[4]["error"] == error
    for name, value in expected.items():
        assert result[4][name] == pytest.approx(value)
    assert result[1] == pytest.approx(sum(expected.values()))


@pytest.mark.parametrize("field", ["length_penalty_weight", "execution_penalty_weight", "trajectory_weight"])
def test_large_finite_reward_weights_do_not_overflow(fixed_env: KarelProgramEnv, field: str) -> None:
    from dataclasses import replace

    fixed_env.config = replace(fixed_env.config, position_weight=4.0, **{field: 1e308})
    fixed_env.reset()
    # The two turns add a backward-distance unit before reaching the target.
    result = submit(fixed_env, ["DEF", "run", "m(", "turnLeft", "turnRight", "move", "m)"])
    expected = {
        "length_penalty_weight": -1e308 * (7 / fixed_env.config.max_program_tokens),
        "execution_penalty_weight": -1e308 * (3 / fixed_env.config.max_execution_steps),
        "trajectory_weight": 8e307,
    }
    assert np.isfinite(result[1])
    assert result[1] == pytest.approx(expected[field])


def test_large_distance_scale_keeps_trajectory_bonus(fixed_env: KarelProgramEnv) -> None:
    from dataclasses import replace

    fixed_env.config = replace(
        fixed_env.config,
        position_weight=1e308,
        orientation_weight=1e308,
        trajectory_weight=0.25,
    )
    fixed_env.reset()
    # Reach the target first: all distances remain finite, but D0+B would overflow.
    result = submit(fixed_env, ["DEF", "run", "m(", "move", "turnLeft", "turnRight", "m)"])
    assert result[4]["reward_trajectory"] == pytest.approx(0.125)
    assert result[1] == pytest.approx(4.125)


def test_distance_overflow_raises_instead_of_returning_nan(world: State) -> None:
    target = execute("move move", world)
    config = KarelConfig(position_weight=1e308)
    with pytest.raises(ValueError, match="overflowed"):
        progress_reward(world, target, target, config)


def test_total_reward_overflow_is_rejected(fixed_env: KarelProgramEnv) -> None:
    from dataclasses import replace

    fixed_env.config = replace(
        fixed_env.config,
        max_program_tokens=5,
        max_execution_steps=1,
        length_penalty_weight=1e308,
        execution_penalty_weight=1e308,
    )
    fixed_env.reset()
    with pytest.raises(ValueError, match="Total reward overflowed"):
        submit(fixed_env, ["DEF", "run", "m(", "move", "m)"])


@pytest.mark.parametrize("weights", [(1, 1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 0), (0.25, 2, 3)])
@pytest.mark.parametrize(
    "source,budget,token_limit,scores,error,success",
    [
        ("DEF run m( move m)", 256, 128, (1, 1, 1), None, True),
        ("DEF run m( turnLeft turnRight m)", 256, 128, (1, 1, 0.5), None, False),
        ("DEF run m( move pickMarker pickMarker m)", 256, 128, (1, 0, 0.5), "runtime_error", False),
        ("DEF run m( move turnLeft m)", 1, 128, (1, 0, 1), "execution_limit", False),
        ("DEF run m( m)", 256, 128, (0.5, 0, 0), "syntax_error", False),
        ("DEF run m( move move", 256, 5, (0.5, 0, 0), "token_limit", False),
    ],
)
def test_weighted_reward_terms(
    fixed_env: KarelProgramEnv,
    weights: tuple[float, float, float],
    source: str,
    budget: int,
    token_limit: int,
    scores: tuple[float, float, float],
    error: str | None,
    success: bool,
) -> None:
    fixed_env.config = replace(
        fixed_env.config,
        syntax_weight=weights[0],
        runtime_weight=weights[1],
        distance_weight=weights[2],
        max_execution_steps=budget,
        max_program_tokens=token_limit,
    )
    fixed_env.reset()
    _, reward, terminated, truncated, info = submit(fixed_env, source.split())
    assert info["error"] == error and info["success"] == success
    assert truncated == (error == "token_limit") and terminated == (not truncated)
    expected = dict(zip(("syntax", "runtime", "distance"), (weight * score for weight, score in zip(weights, scores))))
    for name, value in expected.items():
        assert info[f"reward_{name}"] == pytest.approx(value)
    assert info["reward_success"] == float(success)
    assert reward == pytest.approx(sum(expected.values()) + float(success))
    assert reward == pytest.approx(sum(value for key, value in info.items() if key.startswith("reward_")))


@pytest.mark.parametrize("weight_name", ["syntax_weight", "runtime_weight", "distance_weight", "success_weight"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_term_weights_must_be_finite_and_nonnegative(weight_name: str, value: float) -> None:
    with pytest.raises((AssertionError, ValueError)):
        KarelConfig(**{weight_name: value})


@pytest.mark.parametrize("success_weight", [0.0, 0.25, 1.0, 2.0])
@pytest.mark.parametrize(
    "body,budget,distance,success,error",
    [
        ("move", 256, 1.0, True, None),
        ("turnLeft turnRight", 256, 0.5, False, None),
        ("putMarker", 256, 0.5, False, "runtime_error"),  # Fails before any action succeeds.
        ("move putMarker", 256, 1.0, False, "runtime_error"),
        ("move turnLeft", 1, 1.0, False, "execution_limit"),
        ("", 256, 0.0, False, "syntax_error"),
    ],
)
def test_distance_only_reward_preserves_failed_execution_progress(
    fixed_env: KarelProgramEnv,
    body: str,
    budget: int,
    distance: float,
    success: bool,
    error: str | None,
    success_weight: float,
) -> None:
    fixed_env.config = replace(
        fixed_env.config,
        syntax_weight=0,
        runtime_weight=0,
        distance_weight=1,
        success_weight=success_weight,
        max_markers=1,
        max_execution_steps=budget,
    )
    fixed_env.reset()
    result = submit(fixed_env, f"DEF run m( {body} m)".split())
    assert result[1] == pytest.approx(distance + success_weight * float(success))
    assert result[4]["reward_success"] == pytest.approx(success_weight * float(success))
    assert result[4]["reward_runtime"] == 0
    assert result[4]["reward_distance"] == pytest.approx(distance)
    assert result[4]["success"] == success
    assert result[4]["error"] == error


@pytest.mark.parametrize("source", ["DEF run m( m)", "DEF run m( move move"])
def test_disabled_syntax_reward_skips_edit_distance(
    fixed_env: KarelProgramEnv, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    def unexpected_score(tokens: Sequence[str]) -> float:
        raise AssertionError("Disabled syntax reward must not compute edit distance")

    monkeypatch.setattr("rl2.karel.syntax_reward", unexpected_score)
    fixed_env.config = replace(fixed_env.config, syntax_weight=0, max_program_tokens=5)
    fixed_env.reset()
    result = submit(fixed_env, source.split())
    assert result[1] == 0 and result[4]["reward_syntax"] == 0
    assert result[4]["error"] in ("syntax_error", "token_limit")
