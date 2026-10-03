from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import pytest

from rl2.karel import (
    KarelConfig,
    KarelProgramEnv,
    KarelProgramError,
    KarelTask,
    State,
    StepResult,
    execute_program,
    sample_task,
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


@pytest.fixture
def fixed_env(world: State, monkeypatch: pytest.MonkeyPatch) -> KarelProgramEnv:
    target = world.copy()
    target[2, 1, 1] = 0
    target[2, 2, 1] = 1
    task = KarelTask(world, target, ("DEF", "run", "m(", "move", "m)"))

    def fixed_task(rng: np.random.Generator, config: KarelConfig) -> KarelTask:
        return task

    monkeypatch.setattr("rl2.karel.sample_task", fixed_task)
    return KarelProgramEnv()


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


def test_excessive_nesting_is_rejected_before_execution(world: State) -> None:
    body = "REPEAT R=1 r( " * 65 + "move " + "r) " * 65
    with pytest.raises(KarelProgramError, match="nesting"):
        execute(body, world)


def test_reset_pair_and_eos_only_evaluation(fixed_env: KarelProgramEnv) -> None:
    initial, target = fixed_env.reset()
    assert initial.shape == target.shape == (5, 6, 6)
    assert initial.dtype == target.dtype == np.int32
    for token in ["DEF", "run", "m(", "move", "m)"]:
        assert fixed_env.step(fixed_env.token_to_id[token]) == (None, 0.0, False, False, {})
    assert fixed_env.step(fixed_env.eos_token_id) == (None, 1.0, True, False, {"success": True, "error": None})


def test_equivalent_program_gets_full_reward(fixed_env: KarelProgramEnv) -> None:
    fixed_env.reset()
    result = submit(fixed_env, ["DEF", "run", "m(", "turnLeft", "turnRight", "move", "m)", "<eos>"])
    assert result == (None, 1.0, True, False, {"success": True, "error": None})


@pytest.mark.parametrize("body", ["move turnLeft", "move pickMarker", "turnLeft"])
def test_target_matching_checks_heading_markers_and_position(fixed_env: KarelProgramEnv, body: str) -> None:
    fixed_env.reset()
    result = submit(fixed_env, f"DEF run m( {body} m) <eos>".split())
    assert result == (None, 0.0, True, False, {"success": False, "error": None})


@pytest.mark.parametrize(
    "program,reason",
    [
        ("<eos>", "syntax_error"),
        ("DEF run m( move <eos>", "syntax_error"),
        ("DEF run m( pickMarker pickMarker m) <eos>", "runtime_error"),
        ("DEF run m( WHILE c( markersPresent c) w( turnLeft w) m) <eos>", "execution_limit"),
    ],
)
def test_failed_programs_terminate_with_zero_reward(fixed_env: KarelProgramEnv, program: str, reason: str) -> None:
    fixed_env.reset()
    result = submit(fixed_env, program.split())
    assert result == (None, 0.0, True, False, {"success": False, "error": reason})


def test_observations_cannot_mutate_private_task(fixed_env: KarelProgramEnv) -> None:
    pair = fixed_env.reset()
    pair.initial.fill(0)
    pair.target.fill(0)
    result = submit(fixed_env, ["DEF", "run", "m(", "move", "m)", "<eos>"])
    assert result[1] == 1.0


def test_reset_discards_partial_program(fixed_env: KarelProgramEnv) -> None:
    fixed_env.reset()
    submit(fixed_env, ["IF", "move"])
    fixed_env.reset()
    assert submit(fixed_env, ["DEF", "run", "m(", "move", "m)", "<eos>"])[1] == 1.0


def test_token_limit_and_eos_boundary(fixed_env: KarelProgramEnv) -> None:
    # The fixed reference uses five program tokens plus EOS, exactly the limit.
    env = KarelProgramEnv(KarelConfig(max_program_tokens=6))
    env.reset()
    assert submit(env, ["DEF", "run", "m(", "move", "m)", "<eos>"])[1] == 1.0
    env.reset()
    assert submit(env, ["move"] * 6) == (None, 0.0, False, True, {"success": False, "error": "token_limit"})
    with pytest.raises(gym.error.ResetNeeded):
        env.step(env.eos_token_id)


def test_episode_lifecycle_and_invalid_action(fixed_env: KarelProgramEnv) -> None:
    with pytest.raises(gym.error.ResetNeeded):
        fixed_env.step(0)
    with pytest.raises(gym.error.ResetNeeded):
        _ = fixed_env.reference_program
    fixed_env.reset()
    for invalid in (-1, len(fixed_env.tokens), 1.5, True, "move"):
        with pytest.raises(gym.error.InvalidAction):
            fixed_env.step(invalid)
    fixed_env.step(np.int64(fixed_env.eos_token_id))
    with pytest.raises(gym.error.ResetNeeded):
        fixed_env.step(0)


@pytest.mark.parametrize("seed", range(20))
def test_sampled_task_is_reachable_changed_and_within_limits(seed: int) -> None:
    config = KarelConfig()
    task = sample_task(np.random.default_rng(seed), config)
    assert len(task.program) + 1 <= config.max_program_tokens
    assert not np.array_equal(task.initial, task.target)
    np.testing.assert_array_equal(execute_program(task.program, task.initial), task.target)
    assert np.all(task.initial[[0, -1], :, 4] == 1)
    assert np.all(task.initial[:, [0, -1], 4] == 1)


def test_seed_reproduces_task_stream_and_reference_solves() -> None:
    first, second = KarelProgramEnv(), KarelProgramEnv()
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
        assert (reward, terminated, truncated, info) == (1.0, True, False, {"success": True, "error": None})
    assert len(pairs) == 10


def test_depth_zero_samples_only_primitive_actions() -> None:
    config = KarelConfig(max_depth=0)
    for seed in range(10):
        task = sample_task(np.random.default_rng(seed), config)
        assert not {"WHILE", "REPEAT", "IF", "IFELSE"}.intersection(task.program)


def test_sampling_failure_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    def identity_program(rng: np.random.Generator, config: KarelConfig) -> tuple[str, ...]:
        return ("DEF", "run", "m(", "turnLeft", "turnRight", "m)")

    monkeypatch.setattr("rl2.karel._sample_program", identity_program)
    with pytest.raises(RuntimeError, match="max_sampling_attempts"):
        sample_task(np.random.default_rng(0), KarelConfig(max_sampling_attempts=2))


@pytest.mark.parametrize(
    "options",
    [
        {"height": 2},
        {"width": 0},
        {"max_depth": -1},
        {"max_depth": 33},
        {"max_statements": 0},
        {"max_program_tokens": 5},
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
        KarelConfig(**options)


def test_world_shape_dtype_and_robot_validation(world: State) -> None:
    with pytest.raises(AssertionError):
        execute("move", world.astype(np.float32))
    with pytest.raises(AssertionError):
        execute("move", world[..., :5])
    world[1, 1, 0] = 1
    with pytest.raises(ValueError, match="exactly one"):
        execute("move", world)
