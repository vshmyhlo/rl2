"""Karel program synthesis with an observation only at reset.

States are int32 arrays shaped (height, width, 6): four one-hot robot headings
(north, east, south, west), walls, and marker counts. Actions are DSL token IDs;
EOS executes the complete program against the initial state. This is a plain
class, not a Gymnasium Env: step observations deliberately are always None.
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

type State = NDArray[np.int32]
type StepResult = tuple[None, float, bool, bool, dict[str, object]]

ACTIONS = ("move", "turnLeft", "turnRight", "pickMarker", "putMarker")
PREDICATES = ("frontIsClear", "leftIsClear", "rightIsClear", "markersPresent", "noMarkersPresent")
TOKENS = (
    "<eos>",
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
    max_program_tokens: int = 128  # Includes EOS.
    max_execution_steps: int = 256  # Statements and loop-condition checks.
    max_sampling_attempts: int = 1000

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
        if self.max_program_tokens < 6:
            raise ValueError("At least six tokens are needed for DEF run m( action m) <eos>")
        if type(self.max_depth) is not int:
            raise TypeError("max_depth must be an integer")
        chex.assert_scalar_in(self.max_depth, 0, 32)
        chex.assert_scalar_in(self.max_markers, 1, np.iinfo(np.int32).max - 1)
        chex.assert_scalar_in(self.wall_probability, 0, 1)
        chex.assert_scalar_in(self.marker_probability, 0, 1)


@dataclass(frozen=True)
class KarelTask:
    initial: State
    target: State
    program: tuple[str, ...]  # Reference program, without EOS.


class KarelProgramError(ValueError):
    """An invalid, unsafe, or nonterminating candidate program."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


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
        if depth > 64:
            raise KarelProgramError("syntax_error", "Program nesting exceeds 64 blocks")
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
    """Execute DSL tokens (without EOS), returning a new state or raising KarelProgramError.

    Every statement and while-condition check consumes execution budget, including
    loops whose bodies do nothing. Invalid moves/picks/puts fail the whole program.
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

    run(program)
    state[..., :4] = 0
    state[row, col, heading] = 1
    return state


def _sample_program(rng: np.random.Generator, config: KarelConfig) -> tuple[str, ...]:
    tokens = ["DEF", "run", "m("]

    def emit(*parts: str) -> None:
        tokens.extend(parts)
        if len(tokens) >= config.max_program_tokens:
            raise KarelProgramError("token_limit", "Sampled program is too long to submit with EOS")

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
    step() returns (None, reward, terminated, truncated, info). EOS terminates
    and scores exact state equality; invalid programs receive zero. Exhausting
    max_program_tokens without EOS truncates with zero reward and no evaluation.
    """

    tokens = TOKENS
    token_to_id = TOKEN_TO_ID
    eos_token_id = TOKEN_TO_ID["<eos>"]

    def __init__(self, config: KarelConfig | None = None) -> None:
        self.config = config if config is not None else KarelConfig()
        self.action_space = gym.spaces.Discrete(len(TOKENS))
        self._rng = np.random.default_rng()
        self._task: KarelTask | None = None
        self._program: list[str] = []
        self._needs_reset = True

    @property
    def reference_program(self) -> tuple[int, ...]:
        """Privileged supervision/debugging solution, including EOS; never in observations."""
        if self._task is None:
            raise gym.error.ResetNeeded("Call reset() before requesting a reference program")
        return tuple(TOKEN_TO_ID[token] for token in self._task.program) + (self.eos_token_id,)

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

    def step(self, action: int | np.integer) -> StepResult:
        if self._needs_reset or self._task is None:
            raise gym.error.ResetNeeded("Call reset() before stepping a new episode")
        if (
            isinstance(action, (bool, np.bool_))
            or not isinstance(action, (int, np.integer))
            or not self.action_space.contains(action)
        ):
            raise gym.error.InvalidAction(f"Expected a token ID in [0, {len(TOKENS) - 1}], got {action!r}")
        if int(action) == self.eos_token_id:
            self._needs_reset = True
            try:
                output = execute_program(
                    self._program,
                    self._task.initial,
                    max_steps=self.config.max_execution_steps,
                    max_markers=self.config.max_markers,
                )
            except KarelProgramError as exc:
                return None, 0.0, True, False, {"success": False, "error": exc.reason}
            success = bool(np.array_equal(output, self._task.target))
            return None, float(success), True, False, {"success": success, "error": None}
        self._program.append(TOKENS[int(action)])
        if len(self._program) >= self.config.max_program_tokens:
            self._needs_reset = True
            return None, 0.0, False, True, {"success": False, "error": "token_limit"}
        return None, 0.0, False, False, {}
