"""Karel program synthesis with an observation only at reset.

States are int32 arrays shaped (height, width, 6): four one-hot robot headings
(north, east, south, west), walls, and marker counts. Actions are DSL token IDs;
The outer closing token m) executes the program against the initial state. This is a plain
class, not a Gymnasium Env: step observations deliberately are always None.
Terminal reward sums syntax, runtime, and distance scores, each in [0, 1],
plus a +1 exact-success bonus. Terminal info exposes all four components.
Invalid syntax receives only 1/(1+d), where d is the minimum syntax edit count.
Execution failures receive 1 plus progress from the last valid state; completed
executions receive 2 plus final-state progress and the success bonus. PAD is reserved for batching.
Intermediate rewards are zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

import chex
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from rl2.karel_syntax import MAX_BLOCK_DEPTH, syntax_reward

type State = NDArray[np.int32]
type StepResult = tuple[None, float, bool, bool, dict[str, object]]
REWARD_COMPONENTS = ("syntax", "runtime", "distance", "success")

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
    position_weight: float = 1.0  # Weight per cell of Manhattan robot-position error.
    orientation_weight: float = 1.0  # Weight for any incorrect robot heading.
    marker_weight: float = 1.0  # Weight per missing or extra marker, summed over all cells.

    def __post_init__(self) -> None:
        for value in (
            self.height,
            self.width,
            self.max_markers,
            self.max_statements,
            self.max_program_tokens,
            self.max_execution_steps,
            self.max_sampling_attempts,
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
        for weight in (self.position_weight, self.orientation_weight, self.marker_weight):
            chex.assert_scalar_positive(weight)
            if not np.isfinite(weight):
                raise ValueError("Reward distance weights must be finite and strictly positive")


@dataclass(frozen=True)
class KarelTask:
    initial: State
    target: State
    program: tuple[str, ...]  # Complete reference program, including terminal m).


def state_distance(state: State, target: State, config: KarelConfig) -> float:
    """Weighted position, heading, and marker error between valid Karel states.

    D(s,t) = position_weight * Manhattan(robot_s, robot_t)
           + orientation_weight * (heading_s != heading_t)
           + marker_weight * sum_cells(abs(markers_s - markers_t)).

    Walls must be identical and are not scored. Manhattan distance ignores walls
    and is a closeness heuristic, not a minimum-action solution cost. All wrong
    headings have the same cost; marker errors include initially correct cells.
    """
    chex.assert_shape(state, (None, None, 6))
    chex.assert_equal_shape((state, target))
    chex.assert_type((state, target), np.int32)
    if not np.array_equal(state[..., 4], target[..., 4]):
        raise ValueError("Reward distance requires identical walls")
    robot = np.argwhere(state[..., :4])
    target_robot = np.argwhere(target[..., :4])
    chex.assert_shape((robot, target_robot), (1, 3))
    position_error = int(np.abs(robot[0, :2] - target_robot[0, :2]).sum())
    orientation_error = int(robot[0, 2] != target_robot[0, 2])
    marker_error = int(np.abs(state[..., 5].astype(np.int64) - target[..., 5].astype(np.int64)).sum())
    return float(
        config.position_weight * position_error
        + config.orientation_weight * orientation_error
        + config.marker_weight * marker_error
    )


def progress_reward(initial: State, final: State, target: State, config: KarelConfig) -> float:
    """Map signed progress from [-1, 1] to [0, 1] for a final or partial state.

    Score = clip(1 - D(final,t)/(2*D(initial,t)), 0, 1). Exact targets score 1,
    unchanged distance scores 0.5, and doubled or greater distance scores 0.
    Smaller regressions score between 0 and 0.5.
    The sampler excludes identical initial/target pairs;
    strictly positive weights therefore guarantee a positive denominator.
    This is the runtime term on execution failure and the distance term on
    successful completion; the other reward terms are added by step().
    """
    initial_distance = state_distance(initial, target, config)
    if initial_distance <= 0:
        raise ValueError("Progress reward requires a non-identical initial/target pair")
    final_distance = state_distance(final, target, config)
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


def execute_program(tokens: Sequence[str], initial: State, *, max_steps: int = 256, max_markers: int = 10) -> State:
    """Execute a complete program ending in m), returning a new state or raising KarelProgramError.

    Every statement and while-condition check consumes execution budget, including
    loops whose bodies do nothing. Invalid moves/picks/puts stop before applying
    the offending action. Runtime/budget exceptions expose partial_state, with
    all earlier actions applied and the latest robot position and heading.
    """
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
    Sampling is with replacement; uniqueness across calls is not guaranteed.
    """
    config = config if config is not None else KarelConfig()
    for _ in range(config.max_sampling_attempts):
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
        if not np.array_equal(initial, target):
            return KarelTask(initial, target, program)
    raise RuntimeError("Could not sample a nontrivial Karel task within max_sampling_attempts")


class KarelProgramEnv:
    """Generate one program token per step, receiving the task only at reset.

    reset() returns KarelPair(initial, target), without an info wrapper.
    step() returns (None, reward, terminated, truncated, info). The token m) terminates
    and sums syntax, runtime, and distance terms, plus a +1 exact-success bonus.
    Completed execution scores 1 + 1 + progress_reward(final) + float(success),
    reaching 4 for an exact solution; execution failures score 1 +
    progress_reward(last_valid_state) + 0. Invalid syntax scores 1/(1+d) + 0 + 0.
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
        self._program: list[str] = []
        self._needs_reset = True

    @property
    def reference_program(self) -> tuple[int, ...]:
        """Privileged supervision/debugging solution ending in m); never in observations."""
        if self._task is None:
            raise gym.error.ResetNeeded("Call reset() before requesting a reference program")
        return tuple(TOKEN_TO_ID[token] for token in self._task.program)

    def reset(self, *, seed: int | None = None) -> KarelPair:
        self._needs_reset = True
        self._task = None
        self._program.clear()
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
        self._task = sample_task(self._rng, self.config)
        self._needs_reset = False
        # Callers may transform observations without changing the task we evaluate.
        return KarelPair(self._task.initial.copy(), self._task.target.copy())

    def _finish(
        self,
        syntax: float,
        runtime: float = 0.0,
        distance: float = 0.0,
        *,
        success: bool = False,
        error: str | None = None,
        truncated: bool = False,
    ) -> StepResult:
        """Expose the exact terms summed into the terminal reward for diagnostics."""
        self._needs_reset = True
        components = dict(zip(REWARD_COMPONENTS, (syntax, runtime, distance, float(success))))
        info: dict[str, object] = {"success": success, "error": error}
        info.update({f"reward_{name}": value for name, value in components.items()})
        return None, sum(components.values()), not truncated, truncated, info

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
            try:
                output = execute_program(
                    self._program,
                    self._task.initial,
                    max_steps=self.config.max_execution_steps,
                    max_markers=self.config.max_markers,
                )
            except KarelProgramError as exc:
                if exc.reason == "syntax_error":
                    return self._finish(syntax_reward(self._program), error=exc.reason)
                else:
                    assert exc.partial_state is not None
                    runtime_score = progress_reward(
                        self._task.initial, exc.partial_state, self._task.target, self.config
                    )
                    return self._finish(1.0, runtime_score, error=exc.reason)
            success = bool(np.array_equal(output, self._task.target))
            distance = progress_reward(self._task.initial, output, self._task.target, self.config)
            return self._finish(1.0, 1.0, distance, success=success)
        if len(self._program) >= self.config.max_program_tokens:
            self._needs_reset = True
            return self._finish(syntax_reward(self._program), error="token_limit", truncated=True)
        return None, 0.0, False, False, {}
