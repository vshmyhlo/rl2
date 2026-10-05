"""Karel program synthesis with an observation only at reset.

States are int32 arrays shaped (height, width, 6): four one-hot robot headings
(north, east, south, west), walls, and marker counts. Actions are DSL token IDs;
The outer closing token m) executes the program against the initial state. This is a plain
class, not a Gymnasium Env: step observations deliberately are always None.
Terminal reward sums weighted syntax, runtime, and distance scores, each originally in [0, 1],
plus weighted exact-success and trajectory bonuses. Terminal info exposes all seven components.
With default term weights, invalid syntax receives only 1/(1+d), where d is the minimum syntax edit count.
Execution failures receive 1 plus progress from the last valid state; completed
executions receive 2 plus final-state progress and the success bonus.
Terminal rewards deduct normalized program-length and execution-step costs.
Executable programs also receive a trajectory bonus for net progress discounted
by cumulative distance increases. PAD is reserved for batching.
Intermediate rewards are zero.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from time import perf_counter
from types import MappingProxyType
from typing import NamedTuple

import chex
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from rl2.karel_syntax import MAX_BLOCK_DEPTH, syntax_reward

type State = NDArray[np.int32]
type DistanceMap = NDArray[np.int32]
type StepResult = tuple[None, float, bool, bool, dict[str, object]]
REWARD_COMPONENTS = ("syntax", "runtime", "distance", "success", "trajectory", "length", "execution")
TASK_CATEGORIES = ("easy", "navigation", "marker", "combined")

ACTIONS = ("move", "turnLeft", "turnRight", "pickMarker", "putMarker")
PREDICATES = ("frontIsClear", "leftIsClear", "rightIsClear", "markersPresent", "noMarkersPresent")
TOKENS = (
    "<pad>",
    "DEF",
    "run",
    "m(",
    "m)",
    "WHILE",
    "w(",
    "w)",
    "IF",
    "IFELSE",
    "i(",
    "i)",
    "ELSE",
    "e(",
    "e)",
    "REPEAT",
    "r(",
    "r)",
    "c(",
    "c)",
    "not",
    *ACTIONS,
    *PREDICATES,
    *(f"R={count}" for count in range(20)),
)
TOKEN_TO_ID = MappingProxyType({token: index for index, token in enumerate(TOKENS)})
_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


class KarelPair(NamedTuple):
    initial: State
    target: State


@dataclass(frozen=True)
class KarelConfig:
    height: int = 8
    width: int = 8
    wall_probability: float = 0.1
    marker_probability: float = 0.2
    max_markers: int = 10
    max_depth: int = 2
    max_statements: int = 3
    max_program_tokens: int = 128  # Includes the terminal m); excludes padding.
    max_execution_steps: int = 256  # Statements and loop-condition checks.
    max_sampling_attempts: int = 1000
    # Relative category weights; defaults retain unrestricted sampling.
    task_easy_weight: float = 1.0
    task_navigation_weight: float = 0.0
    task_marker_weight: float = 0.0
    task_combined_weight: float = 0.0
    navigation_min_distance: int = 3
    marker_min_edits: int = 3
    combined_min_distance: int = 2
    combined_min_edits: int = 2
    position_weight: float = 1.0  # Weight per step of shortest-path robot-position error.
    orientation_weight: float = 1.0  # Weight for any incorrect robot heading.
    marker_weight: float = 1.0  # Weight per missing or extra marker, summed over all cells.
    syntax_weight: float = 1.0  # Scale the syntax reward contribution; zero disables.
    runtime_weight: float = 1.0  # Scale binary execution completion (1 completed, 0 failed); zero disables.
    distance_weight: float = 1.0  # Scale progress from the final or last valid execution state; zero disables.
    success_weight: float = 1.0  # Bonus for normal completion at the exact target; zero disables.
    trajectory_weight: float = 0.25  # Net progress discounted by cumulative distance increases; zero disables.
    length_penalty_weight: float = 0.05  # Maximum deduction at the generated-token limit.
    execution_penalty_weight: float = 0.05  # Maximum deduction at the execution-step limit.

    def __post_init__(self) -> None:
        for value in (
            self.height,
            self.width,
            self.max_markers,
            self.max_statements,
            self.max_program_tokens,
            self.max_execution_steps,
            self.max_sampling_attempts,
            self.navigation_min_distance,
            self.marker_min_edits,
            self.combined_min_distance,
            self.combined_min_edits,
        ):
            if type(value) is not int:
                raise TypeError("Karel sizes and limits must be integers")
            chex.assert_scalar_positive(value)
        if self.height < 3 or self.width < 3:
            raise ValueError("The grid must include an interior cell and its border walls")
        if self.max_program_tokens < 5:
            raise ValueError("At least five tokens are needed for DEF run m( action m)")
        if type(self.max_depth) is not int:
            raise TypeError("max_depth must be an integer")
        chex.assert_scalar_in(self.max_depth, 0, 32)
        chex.assert_scalar_in(self.max_markers, 1, np.iinfo(np.int32).max - 1)
        chex.assert_scalar_in(self.wall_probability, 0, 1)
        chex.assert_scalar_in(self.marker_probability, 0, 1)
        task_weights = [getattr(self, f"task_{category}_weight") for category in TASK_CATEGORIES]
        if any(not np.isfinite(weight) or weight < 0 for weight in task_weights) or not any(task_weights):
            raise ValueError("Task category weights must be finite, nonnegative, and not all zero")
        for weight in (self.position_weight, self.orientation_weight, self.marker_weight):
            chex.assert_scalar_positive(weight)
            if not np.isfinite(weight):
                raise ValueError("Reward distance weights must be finite and strictly positive")
        for name in (
            "syntax_weight",
            "runtime_weight",
            "distance_weight",
            "success_weight",
            "trajectory_weight",
            "length_penalty_weight",
            "execution_penalty_weight",
        ):
            weight = getattr(self, name)
            chex.assert_scalar_non_negative(weight)
            if not np.isfinite(weight):
                raise ValueError(f"{name} must be finite and nonnegative")


class TaskSamplingStats(NamedTuple):
    category: str
    attempts: int
    seconds: float


_DEFAULT_SAMPLING_STATS = TaskSamplingStats("easy", 1, 0.0)


@dataclass(frozen=True)
class KarelTask:
    initial: State
    target: State
    program: tuple[str, ...]  # Complete reference program, including terminal m).
    sampling: TaskSamplingStats = _DEFAULT_SAMPLING_STATS
    distance_map: DistanceMap | None = None  # Reuse acceptance-check BFS at reset.


def target_distance_map(target: State) -> DistanceMap:
    """BFS distances to the target robot through free cells; walls/unreachable = -1.

    Counts orthogonal moves, ignoring heading and markers. The map can be reused
    while the target robot position and wall layout stay fixed.
    """
    chex.assert_shape(target, (None, None, 6))
    chex.assert_type(target, np.int32)
    robot = np.argwhere(target[..., :4])
    chex.assert_shape(robot, (1, 3))
    row, col = map(int, robot[0, :2])
    if target[row, col, 4]:
        raise ValueError("Target robot must occupy a free cell")
    distances = np.full(target.shape[:2], -1, dtype=np.int32)
    distances[row, col] = 0
    queue = deque([(row, col)])
    height, width = distances.shape
    while queue:
        row, col = queue.popleft()
        for dr, dc in _DIRECTIONS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width and not target[nr, nc, 4] and distances[nr, nc] == -1:
                distances[nr, nc] = distances[row, col] + 1
                queue.append((nr, nc))
    return distances


def state_distance(
    state: State, target: State, config: KarelConfig, *, distance_map: DistanceMap | None = None
) -> float:
    """Weighted position, heading, and marker error between valid Karel states.

    D(s,t) = position_weight * shortest_path(robot_s, robot_t)
           + orientation_weight * (heading_s != heading_t)
           + marker_weight * sum_cells(abs(markers_s - markers_t)).

    Walls must be identical and constrain movement but are not scored directly.
    Position counts orthogonal moves through free cells, not turning costs. All wrong
    headings have the same cost; marker errors include initially correct cells.
    An optional distance_map must come from target_distance_map for this target's
    position and walls. Unreachable robot positions raise ValueError; sampled
    tasks and their executions always remain in the target's connected component.
    Weighted totals that overflow float64 also raise ValueError instead of
    allowing nonfinite progress rewards to reach the trainer.
    """
    chex.assert_shape(state, (None, None, 6))
    chex.assert_equal_shape((state, target))
    chex.assert_type((state, target), np.int32)
    if not np.array_equal(state[..., 4], target[..., 4]):
        raise ValueError("Reward distance requires identical walls")
    robot = np.argwhere(state[..., :4])
    target_robot = np.argwhere(target[..., :4])
    chex.assert_shape((robot, target_robot), (1, 3))
    if distance_map is None:
        distance_map = target_distance_map(target)
    chex.assert_shape(distance_map, target.shape[:2])
    chex.assert_type(distance_map, np.int32)
    if distance_map[tuple(target_robot[0, :2])] != 0:
        raise ValueError("Distance map must be rooted at the target robot")
    position_error = int(distance_map[tuple(robot[0, :2])])
    if position_error < 0:
        raise ValueError("Robot position cannot reach the target through free cells")
    orientation_error = int(robot[0, 2] != target_robot[0, 2])
    marker_error = int(np.abs(state[..., 5].astype(np.int64) - target[..., 5].astype(np.int64)).sum())
    distance = float(
        config.position_weight * position_error
        + config.orientation_weight * orientation_error
        + config.marker_weight * marker_error
    )
    if not np.isfinite(distance):
        raise ValueError("Weighted state distance overflowed; reduce distance weights")
    return distance


def progress_reward(
    initial: State, final: State, target: State, config: KarelConfig, *, distance_map: DistanceMap | None = None
) -> float:
    """Map signed progress from [-1, 1] to [0, 1] for a final or partial state.

    Score = clip(1 - D(final,t)/(2*D(initial,t)), 0, 1). Exact targets score 1,
    unchanged distance scores 0.5, and doubled or greater distance scores 0.
    Smaller regressions score between 0 and 0.5.
    The sampler excludes identical initial/target pairs;
    strictly positive weights therefore guarantee a positive denominator.
    This is the distance term on both normal completion and execution failure.
    The runtime term separately records normal completion; step() adds the
    remaining terms and applies their weights.
    """
    if distance_map is None:
        distance_map = target_distance_map(target)
    initial_distance = state_distance(initial, target, config, distance_map=distance_map)
    if initial_distance <= 0:
        raise ValueError("Progress reward requires a non-identical initial/target pair")
    final_distance = state_distance(final, target, config, distance_map=distance_map)
    return float(np.clip(1.0 - 0.5 * (final_distance / initial_distance), 0.0, 1.0))


class KarelProgramError(ValueError):
    """A failed program, with its last valid state when execution has started."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.partial_state: State | None = None


@dataclass(frozen=True)
class _Statement:
    kind: str
    condition: tuple[str, bool] = ("frontIsClear", False)
    body: tuple[_Statement, ...] = ()
    otherwise: tuple[_Statement, ...] = ()
    count: int = 0


def _parse(tokens: Sequence[str]) -> tuple[_Statement, ...]:
    position = 0

    def take(expected: str | None = None) -> str:
        nonlocal position
        if position == len(tokens):
            raise KarelProgramError("syntax_error", "Unexpected end of program")
        token = tokens[position]
        position += 1
        if expected is not None and token != expected:
            raise KarelProgramError("syntax_error", f"Expected {expected}, got {token}")
        return token

    def condition() -> tuple[str, bool]:
        take("c(")
        predicate = take()
        negate = predicate == "not"
        if negate:
            take("c(")
            predicate = take()
            take("c)")
        if predicate not in PREDICATES:
            raise KarelProgramError("syntax_error", f"Unknown predicate: {predicate}")
        take("c)")
        return predicate, negate

    def block(end: str, depth: int) -> tuple[_Statement, ...]:
        if depth > MAX_BLOCK_DEPTH:
            raise KarelProgramError("syntax_error", f"Program nesting exceeds {MAX_BLOCK_DEPTH} blocks")
        statements: list[_Statement] = []
        while position < len(tokens) and tokens[position] != end:
            kind = take()
            if kind in ACTIONS:
                statement = _Statement(kind)
            elif kind == "REPEAT":
                count_token = take()
                if count_token not in {f"R={count}" for count in range(20)}:
                    raise KarelProgramError("syntax_error", "Expected a repeat count R=0 through R=19")
                take("r(")
                statement = _Statement(kind, body=block("r)", depth + 1), count=int(count_token[2:]))
            elif kind in ("WHILE", "IF", "IFELSE"):
                test = condition()
                opening, closing = ("w(", "w)") if kind == "WHILE" else ("i(", "i)")
                take(opening)
                body = block(closing, depth + 1)
                otherwise: tuple[_Statement, ...] = ()
                if kind == "IFELSE":
                    take("ELSE")
                    take("e(")
                    otherwise = block("e)", depth + 1)
                statement = _Statement(kind, test, body, otherwise)
            else:
                raise KarelProgramError("syntax_error", f"Unexpected statement: {kind}")
            statements.append(statement)
        take(end)
        if not statements:
            raise KarelProgramError("syntax_error", "Empty blocks are not allowed")
        return tuple(statements)

    take("DEF")
    take("run")
    take("m(")
    program = block("m)", 0)
    if position != len(tokens):
        raise KarelProgramError("syntax_error", "Trailing tokens after the program")
    return program


@dataclass
class ExecutionStats:
    """Consumed statement/condition ticks, including failed primitive attempts."""

    steps: int = 0


def execute_program(
    tokens: Sequence[str],
    initial: State,
    *,
    max_steps: int = 256,
    max_markers: int = 10,
    on_action: Callable[[State], None] | None = None,
    execution_stats: ExecutionStats | None = None,
) -> State:
    """Execute a complete program ending in m), returning a new state or raising KarelProgramError.

    Every statement and while-condition check consumes execution budget, including
    loops whose bodies do nothing. Invalid moves/picks/puts stop before applying
    the offending action. Runtime/budget exceptions expose partial_state, with
    all earlier actions applied and the latest robot position and heading.
    If supplied, on_action receives an independent state snapshot after each
    successful primitive action, never for condition checks or failed actions.
    Optional execution_stats is reset on entry and updated even on failure.
    Syntax failures consume zero ticks; exceeding the budget adds no extra tick.
    """
    if execution_stats is not None:
        execution_stats.steps = 0
    chex.assert_shape(initial, (None, None, 6))
    chex.assert_type(initial, np.int32)
    for value in (max_steps, max_markers):
        if type(value) is not int:
            raise TypeError("Execution limits must be integers")
        chex.assert_scalar_positive(value)
    chex.assert_scalar_in(max_markers, 1, np.iinfo(np.int32).max - 1)
    if min(initial.shape[:2]) < 1:
        raise ValueError("The world cannot be empty")
    if not np.all((initial[..., :5] == 0) | (initial[..., :5] == 1)) or initial[..., :4].sum() != 1:
        raise ValueError("Expected binary walls and exactly one robot heading")
    if np.any(initial[..., 5] < 0) or np.any(initial[..., 5] > max_markers):
        raise ValueError("Marker counts are outside the allowed range")
    walls = initial[..., 4] != 0
    if np.any(initial[walls, :4]) or np.any(initial[walls, 5]):
        raise ValueError("Walls cannot contain the robot or markers")
    program = _parse(tokens)
    state = initial.copy()
    row, col, heading = (int(value) for value in np.argwhere(state[..., :4])[0])
    remaining = max_steps

    def tick() -> None:
        nonlocal remaining
        if remaining == 0:
            raise KarelProgramError("execution_limit", "Program exhausted its execution budget")
        remaining -= 1
        if execution_stats is not None:
            execution_stats.steps += 1

    def clear(direction: int) -> bool:
        dr, dc = _DIRECTIONS[direction % 4]
        r, c = row + dr, col + dc
        return 0 <= r < state.shape[0] and 0 <= c < state.shape[1] and not bool(state[r, c, 4])

    def check(test: tuple[str, bool]) -> bool:
        predicate, negate = test
        if predicate == "markersPresent":
            result = bool(state[row, col, 5] > 0)
        elif predicate == "noMarkersPresent":
            result = bool(state[row, col, 5] == 0)
        else:
            offset = {"frontIsClear": 0, "leftIsClear": -1, "rightIsClear": 1}[predicate]
            result = clear(heading + offset)
        return result != negate

    def run(statements: tuple[_Statement, ...]) -> None:
        nonlocal row, col, heading
        for statement in statements:
            tick()
            kind = statement.kind
            if kind == "WHILE":
                while True:
                    tick()
                    if not check(statement.condition):
                        break
                    run(statement.body)
            elif kind == "REPEAT":
                for _ in range(statement.count):
                    run(statement.body)
            elif kind in ("IF", "IFELSE"):
                run(statement.body if check(statement.condition) else statement.otherwise)
            elif kind == "move":
                if not clear(heading):
                    raise KarelProgramError("runtime_error", "Robot moved into a wall or out of bounds")
                dr, dc = _DIRECTIONS[heading]
                row, col = row + dr, col + dc
            elif kind == "turnLeft":
                heading = (heading - 1) % 4
            elif kind == "turnRight":
                heading = (heading + 1) % 4
            elif kind == "pickMarker":
                if state[row, col, 5] == 0:
                    raise KarelProgramError("runtime_error", "Robot picked from an empty cell")
                state[row, col, 5] -= 1
            elif kind == "putMarker":
                if state[row, col, 5] == max_markers:
                    raise KarelProgramError("runtime_error", "Robot exceeded the marker limit")
                state[row, col, 5] += 1
            if on_action is not None and kind in ACTIONS:
                snapshot = state.copy()
                snapshot[..., :4] = 0
                snapshot[row, col, heading] = 1
                on_action(snapshot)

    try:
        run(program)
    except KarelProgramError as exc:
        # Movement and turning live in local variables during execution. Commit
        # them before exposing the last valid state, including on budget failures.
        state[..., :4] = 0
        state[row, col, heading] = 1
        exc.partial_state = state
        raise
    state[..., :4] = 0
    state[row, col, heading] = 1
    return state


def _sample_program(rng: np.random.Generator, config: KarelConfig) -> tuple[str, ...]:
    tokens = ["DEF", "run", "m("]

    def emit(*parts: str) -> None:
        tokens.extend(parts)
        if len(tokens) > config.max_program_tokens:
            raise KarelProgramError("token_limit", "Sampled program exceeds the token limit")

    def block(depth: int) -> None:
        for _ in range(int(rng.integers(1, config.max_statements + 1))):
            # Half the choices terminate in a primitive; depth bounds control nesting.
            kind = (
                "action"
                if depth == config.max_depth or rng.random() < 0.5
                else str(rng.choice(("IF", "IFELSE", "WHILE", "REPEAT")))
            )
            if kind == "action":
                emit(str(rng.choice(ACTIONS)))
            elif kind == "REPEAT":
                emit(kind, f"R={rng.integers(0, 20)}", "r(")
                block(depth + 1)
                emit("r)")
            else:
                emit(kind, "c(")
                negate = bool(rng.integers(2))
                if negate:
                    emit("not", "c(")
                emit(str(rng.choice(PREDICATES)))
                if negate:
                    emit("c)")
                opening, closing = ("w(", "w)") if kind == "WHILE" else ("i(", "i)")
                emit("c)", opening)
                block(depth + 1)
                emit(closing)
                if kind == "IFELSE":
                    emit("ELSE", "e(")
                    block(depth + 1)
                    emit("e)")

    block(0)
    emit("m)")
    return tuple(tokens)


def sample_task(rng: np.random.Generator, config: KarelConfig | None = None) -> KarelTask:
    """Sample a reachable, changed target by executing a random grammar program.

    Rejection sampling discards invalid, over-budget, and identity executions.
    Select a category once, before retries, then filter by necessary movement
    and marker edits. Exhaustion raises rather than falling back to easier tasks.
    Categories may overlap. Sampling is with replacement; uniqueness is not guaranteed.
    """
    config = config if config is not None else KarelConfig()
    start = perf_counter()
    weights = np.asarray([getattr(config, f"task_{category}_weight") for category in TASK_CATEGORIES], np.float64)
    weights /= weights.max()  # Avoid overflow when normalizing large finite weights.
    enabled = np.flatnonzero(weights)
    # Preserve the old seeded task stream when only the easy category is enabled.
    category = TASK_CATEGORIES[int(enabled[0] if len(enabled) == 1 else rng.choice(4, p=weights / weights.sum()))]
    min_distance = {"navigation": config.navigation_min_distance, "combined": config.combined_min_distance}.get(
        category, 0
    )
    min_edits = {"marker": config.marker_min_edits, "combined": config.combined_min_edits}.get(category, 0)
    for attempt in range(1, config.max_sampling_attempts + 1):
        initial = np.zeros((config.height, config.width, 6), dtype=np.int32)
        initial[..., 4] = rng.random(initial.shape[:2]) < config.wall_probability
        initial[[0, -1], :, 4] = 1
        initial[:, [0, -1], 4] = 1
        row, col = int(rng.integers(1, config.height - 1)), int(rng.integers(1, config.width - 1))
        initial[row, col, 4] = 0
        initial[row, col, int(rng.integers(4))] = 1
        occupied = (rng.random(initial.shape[:2]) < config.marker_probability) & (initial[..., 4] == 0)
        initial[..., 5] = occupied * rng.integers(1, config.max_markers + 1, size=initial.shape[:2])
        try:
            program = _sample_program(rng, config)
            target = execute_program(
                program, initial, max_steps=config.max_execution_steps, max_markers=config.max_markers
            )
        except KarelProgramError:
            continue
        if np.array_equal(initial, target):
            continue
        marker_edits = int(np.abs(initial[..., 5].astype(np.int64) - target[..., 5]).sum())
        if marker_edits < min_edits:
            continue
        distance_map = target_distance_map(target)
        if int(distance_map[row, col]) < min_distance:
            continue
        return KarelTask(
            initial, target, program, TaskSamplingStats(category, attempt, perf_counter() - start), distance_map
        )
    raise RuntimeError(
        f"Could not sample a nontrivial Karel task in category {category!r} within max_sampling_attempts="
        f"{config.max_sampling_attempts} (minimum position distance={min_distance}, marker edits={min_edits})"
    )


class KarelProgramEnv:
    """Generate one program token per step, receiving the task only at reset.

    reset() returns KarelPair(initial, target), without an info wrapper.
    step() returns (None, reward, terminated, truncated, info). The token m) terminates
    and sums weighted syntax, runtime, and distance terms, plus a weighted exact-success bonus.
    Each term's weight defaults to 1; zero disables its reward contribution,
    without changing execution, error classification, or exact-success detection.
    The formulas below assume these default weights.
    Completed execution scores 2 + progress_reward(final) + float(success),
    before the trajectory bonus; execution failures score 1 +
    progress_reward(last_valid_state) + 0. Both receive trajectory_weight *
    max(0, D_initial - D_last) / (D_initial + backward_distance), where backward
    distance sums positive distance increases after primitive actions. The default
    upper bound before efficiency deductions is 4.25. Invalid syntax scores
    1/(1+d) with no trajectory bonus. All outcomes deduct length_penalty_weight *
    generated_tokens / max_program_tokens; executed programs also deduct
    execution_penalty_weight * consumed_steps / max_execution_steps.
    Exhausting max_program_tokens without m) truncates with the syntax score
    and no execution. PAD is not a valid environment action. Exact success
    requires both normal completion and target equality, even if a failed
    program visited or stopped at the target.
    """

    tokens = TOKENS
    token_to_id = TOKEN_TO_ID
    terminal_token_id = TOKEN_TO_ID["m)"]
    pad_token_id = TOKEN_TO_ID["<pad>"]

    def __init__(self, config: KarelConfig | None = None) -> None:
        self.config = config if config is not None else KarelConfig()
        self.action_space = gym.spaces.Discrete(len(TOKENS))
        self._rng = np.random.default_rng()
        self._task: KarelTask | None = None
        self._distance_map: DistanceMap | None = None
        self._program: list[str] = []
        self._needs_reset = True

    @property
    def reference_program(self) -> tuple[int, ...]:
        """Privileged supervision/debugging solution ending in m); never in observations."""
        if self._task is None:
            raise gym.error.ResetNeeded("Call reset() before requesting a reference program")
        return tuple(TOKEN_TO_ID[token] for token in self._task.program)

    @property
    def sampling_stats(self) -> TaskSamplingStats:
        """Generation diagnostics for the current task, excluded from observations."""
        if self._task is None:
            raise gym.error.ResetNeeded("Call reset() before requesting sampling statistics")
        return self._task.sampling

    def reset(self, *, seed: int | None = None) -> KarelPair:
        self._needs_reset = True
        self._task = None
        self._distance_map = None
        self._program.clear()
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
        self._task = sample_task(self._rng, self.config)
        self._distance_map = self._task.distance_map
        if self._distance_map is None:
            self._distance_map = target_distance_map(self._task.target)
        self._needs_reset = False
        # Callers may transform observations without changing the task we evaluate.
        return KarelPair(self._task.initial.copy(), self._task.target.copy())

    def reset_from(self, source: KarelProgramEnv) -> KarelPair:
        """Start an independent episode from a freshly reset matching environment.

        Reuse its sampled task and BFS map without recomputing either. Copy all
        mutable arrays and RNG state so execution, observation edits, and future
        resets cannot affect the source or other members of a sampling group.
        """
        if self.config != source.config:
            raise ValueError("Cannot copy a task between environments with different configs")
        if source._needs_reset or source._task is None or source._distance_map is None or source._program:
            raise ValueError("Source environment must be freshly reset")
        self._distance_map = source._distance_map.copy()
        self._task = replace(
            source._task,
            initial=source._task.initial.copy(),
            target=source._task.target.copy(),
            distance_map=self._distance_map,
        )
        self._rng = deepcopy(source._rng)
        self.action_space.np_random.bit_generator.state = deepcopy(source.action_space.np_random.bit_generator.state)
        self._program.clear()
        self._needs_reset = False
        return KarelPair(self._task.initial.copy(), self._task.target.copy())

    def _finish(
        self,
        syntax: float,
        runtime: float = 0.0,
        distance: float = 0.0,
        *,
        success: bool = False,
        trajectory: float = 0.0,
        execution_steps: int = 0,
        error: str | None = None,
        truncated: bool = False,
    ) -> StepResult:
        """Expose the exact terms summed into the terminal reward for diagnostics."""
        self._needs_reset = True
        # Normalize before multiplying: finite weights can overflow if multiplied
        # by raw counts, even when the final normalized penalty is representable.
        length_penalty = -self.config.length_penalty_weight * (len(self._program) / self.config.max_program_tokens)
        execution_penalty = -self.config.execution_penalty_weight * (execution_steps / self.config.max_execution_steps)
        components = dict(
            zip(
                REWARD_COMPONENTS,
                (
                    self.config.syntax_weight * syntax,
                    self.config.runtime_weight * runtime,
                    self.config.distance_weight * distance,
                    self.config.success_weight * float(success),
                    trajectory,
                    length_penalty,
                    execution_penalty,
                ),
            )
        )
        info: dict[str, object] = {"success": success, "error": error}
        info.update({f"reward_{name}": value for name, value in components.items()})
        reward = sum(components.values())
        if not np.isfinite(reward):
            raise ValueError("Total reward overflowed; reduce reward weights")
        return None, reward, not truncated, truncated, info

    def step(self, action: int | np.integer) -> StepResult:
        if self._needs_reset or self._task is None:
            raise gym.error.ResetNeeded("Call reset() before stepping a new episode")
        if (
            isinstance(action, (bool, np.bool_))
            or not isinstance(action, (int, np.integer))
            or not self.action_space.contains(action)
        ):
            raise gym.error.InvalidAction(f"Expected a token ID in [0, {len(TOKENS) - 1}], got {action!r}")
        if int(action) == self.pad_token_id:
            raise gym.error.InvalidAction("PAD is reserved for batching, not program generation")
        self._program.append(TOKENS[int(action)])
        if int(action) == self.terminal_token_id:
            self._needs_reset = True
            initial_distance: float | None = None
            previous_distance = 0.0
            # Normalize each increase before accumulation so D_initial + B need
            # not be representable when the distance weights are very large.
            backward_ratio = 0.0

            def observe_action(state: State) -> None:
                nonlocal initial_distance, previous_distance, backward_ratio
                # Only executable programs with an enabled trajectory term need
                # this baseline. Syntax failures must not evaluate state distance.
                if initial_distance is None:
                    initial_distance = state_distance(
                        self._task.initial, self._task.target, self.config, distance_map=self._distance_map
                    )
                    if initial_distance <= 0:
                        raise ValueError("Progress reward requires a non-identical initial/target pair")
                    previous_distance = initial_distance
                distance = state_distance(state, self._task.target, self.config, distance_map=self._distance_map)
                backward_ratio += max(0.0, distance - previous_distance) / initial_distance
                previous_distance = distance

            def trajectory_bonus() -> float:
                # A sum of positive changes would reward undo/redo loops. Instead
                # credit only net improvement, discounted by all backward steps.
                if initial_distance is None:
                    return 0.0
                return self.config.trajectory_weight * (
                    max(0.0, initial_distance - previous_distance) / initial_distance / (1.0 + backward_ratio)
                )

            execution_stats = ExecutionStats()
            try:
                output = execute_program(
                    self._program,
                    self._task.initial,
                    max_steps=self.config.max_execution_steps,
                    max_markers=self.config.max_markers,
                    on_action=observe_action if self.config.trajectory_weight else None,
                    execution_stats=execution_stats,
                )
            except KarelProgramError as exc:
                if exc.reason == "syntax_error":
                    score = syntax_reward(self._program) if self.config.syntax_weight else 0.0
                    return self._finish(score, error=exc.reason)
                else:
                    assert exc.partial_state is not None
                    distance = 0.0
                    if self.config.distance_weight:
                        distance = progress_reward(
                            self._task.initial,
                            exc.partial_state,
                            self._task.target,
                            self.config,
                            distance_map=self._distance_map,
                        )
                    return self._finish(
                        1.0,
                        runtime=0.0,
                        distance=distance,
                        error=exc.reason,
                        trajectory=trajectory_bonus(),
                        execution_steps=execution_stats.steps,
                    )
            success = bool(np.array_equal(output, self._task.target))
            distance = 0.0
            if self.config.distance_weight:
                distance = progress_reward(
                    self._task.initial, output, self._task.target, self.config, distance_map=self._distance_map
                )
            return self._finish(
                1.0,
                runtime=1.0,
                distance=distance,
                success=success,
                trajectory=trajectory_bonus(),
                execution_steps=execution_stats.steps,
            )
        if len(self._program) >= self.config.max_program_tokens:
            self._needs_reset = True
            score = syntax_reward(self._program) if self.config.syntax_weight else 0.0
            return self._finish(score, error="token_limit", truncated=True)
        return None, 0.0, False, False, {}
