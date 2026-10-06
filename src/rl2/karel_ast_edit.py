"""Execution-guided Karel AST editing with rewards equal to score improvements.

A location action opens a typed hole; grammar actions fill it depth-first.
Only completed replacements execute, always from the original task input.
Their reward is new_score - previous_score. Other actions, including STOP,
receive zero. Undiscounted rewards sum to final_score - seed_score.
"""

from dataclasses import dataclass, field, replace
from typing import NamedTuple

import chex
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from rl2.karel import (
    REWARD_COMPONENTS,
    TOKEN_TO_ID,
    ExecutionStats,
    KarelConfig,
    KarelPair,
    KarelProgramEnv,
    KarelProgramError,
    execute_program,
)
from rl2.karel_ast import AST_ACTIONS, KarelAST, Node, program_actions
from rl2.shape_checker import ShapeChecker

FEEDBACK_SIZE = 8  # score, success, runtime error, execution limit, ticks, length, score delta, sequence tokens left
EDIT_REWARD_COMPONENTS = (*REWARD_COMPONENTS, "depth")


@dataclass(frozen=True)
class EditConfig:
    """Task distribution and budgets for one editing episode."""

    env: KarelConfig = field(default_factory=KarelConfig)
    max_nodes: int = 128
    max_depth: int = 64
    max_seq_len: int = 256
    seed_program: str = "DEF run m( turnLeft m)"

    def __post_init__(self) -> None:
        """Reject invalid sequence budgets and seeds that violate the AST limits."""
        if type(self.max_seq_len) is not int:
            raise TypeError("Sequence length must be an integer")
        chex.assert_scalar_positive(self.max_seq_len)
        if not isinstance(self.seed_program, str):
            raise TypeError("seed_program must be source text")
        seed_tree(self)
        if self.max_seq_len <= self.prefill_length:
            raise ValueError("max_seq_len must fit the seed, initial update, and at least one action")

    @property
    def prefill_length(self) -> int:
        """Count seed grammar tokens and the initial execution update."""
        return len(program_actions(tuple(self.seed_program.split()))) + 1


def seed_tree(config: EditConfig) -> KarelAST:
    """Build the episode's starting AST by replaying its source as grammar actions."""
    tree = KarelAST.empty(config.max_nodes, config.max_depth, config.env.max_program_tokens)
    for action in program_actions(tuple(config.seed_program.split())):
        tree = tree.expand(action)
    return tree


def open_subtree(tree: KarelAST, position: int) -> KarelAST:
    """Replace a complete subtree by its typed hole; compact unreachable nodes.

    Position is in the pre-edit preorder, not the append-only storage order.
    The untouched context keeps its original depth/field/child-index metadata.
    """
    if not tree.complete:
        raise ValueError("Select an edit location only on a complete tree")
    if type(position) is not int:
        raise TypeError("Location must be an integer preorder position")
    if not 0 <= position < len(tree.nodes):
        raise ValueError("Edit location is outside the tree")
    selected = tree.preorder()[position]
    nodes: list[Node] = []

    def copy(index: int) -> int:
        """Copy reachable nodes, replacing the selected subtree with one typed hole."""
        node = tree.nodes[index]
        destination = len(nodes)
        nodes.append(replace(node, constructor=0, value=0, children=()) if index == selected else node)
        if index != selected:
            children = tuple(copy(child) for child in node.children)
            nodes[destination] = replace(node, children=children)
        return destination

    copy(0)
    result = replace(tree, nodes=tuple(nodes))
    if not result.allowed_actions().any():
        raise ValueError("Selected subtree cannot be completed within the budgets")
    return result


class Observation(NamedTuple):
    """Host-side edit state returned with every action transition."""

    initial: NDArray[np.int32]
    target: NDArray[np.int32]
    output: NDArray[np.int32]
    feedback: NDArray[np.float32]
    legal: NDArray[np.bool_]


@dataclass(frozen=True)
class Evaluation:
    output: NDArray[np.int32]
    score: float
    success: bool
    error: str | None
    ticks: int
    components: dict[str, float]


def evaluate(
    tree: KarelAST,
    task: KarelProgramEnv,
    env: KarelProgramEnv,
    initial: NDArray[np.int32],
) -> Evaluation:
    """Score task performance minus normalized AST size, depth, and execution costs.

    Task remains freshly reset, allowing arbitrary successive candidates to use
    the same initial state and cached target distance map. No reference labels
    enter either the observation or the reward. Length counts all AST nodes;
    depth counts edges from the root, including lists. Execution counts consumed
    statement/condition ticks, including failed attempts, rather than wall time.
    """
    sc = ShapeChecker(H=task.config.height, W=task.config.width, C=6)
    sc.check(initial, "HWC", dtype=np.int32)
    tokens = tree.tokens()
    env.reset_from(task)
    info = {}
    for token in tokens:
        _, _, terminated, truncated, info = env.step(TOKEN_TO_ID[token])
        if terminated or truncated:
            break
    if info.get("error") in ("syntax_error", "token_limit"):
        raise RuntimeError("AST edit produced invalid or over-budget source")
    stats = ExecutionStats()
    try:
        output = execute_program(
            tokens,
            initial,
            max_steps=task.config.max_execution_steps,
            max_markers=task.config.max_markers,
            execution_stats=stats,
        )
    except KarelProgramError as error:
        if error.partial_state is None:
            raise
        output = error.partial_state
    sc.check(output, "HWC", dtype=np.int32)
    components = {name: float(info[f"reward_{name}"]) for name in REWARD_COMPONENTS}
    components["length"] = -task.config.length_penalty_weight * (len(tree.nodes) / tree.max_nodes)
    components["depth"] = -task.config.depth_penalty_weight * (max(node.depth for node in tree.nodes) / tree.max_depth)
    score = sum(components.values())
    if not np.isfinite(score):
        raise ValueError("Total reward overflowed; reduce reward weights")
    return Evaluation(
        output,
        score,
        bool(info["success"]),
        info["error"],
        stats.steps,
        components,
    )


def observe(
    tree: KarelAST,
    pair: KarelPair,
    result: Evaluation,
    tokens_left: int,
    config: EditConfig,
    score_delta: float = 0.0,
) -> Observation:
    """Build host feedback and action masks that reserve enough budget to finish edits."""
    chex.assert_scalar_in(tokens_left, 0, config.max_seq_len)
    legal = np.zeros(1 + config.max_nodes + len(AST_ACTIONS), np.bool_)
    if tree.complete:
        legal[0] = True
        # Reserve the location action plus the cheapest typed replacement.
        # In this grammar minimum-node completions also minimize source length.
        for position, index in enumerate(tree.preorder()):
            legal[1 + position] = 1 + int(tree._costs(index).min()) <= tokens_left
    else:
        # One grammar expansion resolves exactly one node. Reserve actions for
        # ALL remaining holes. Feedback travels with the completing action.
        frontier = tree.frontier
        assert frontier is not None
        costs = tree._costs(frontier)
        resolved = sum(not node.is_hole for node in tree.nodes)
        required = tree._minimum_completion() - resolved
        legal[1 + config.max_nodes :] = tree.allowed_actions() & (required - costs.min() + costs <= tokens_left)
    feedback = np.asarray(
        [
            result.score,
            result.success,
            result.error == "runtime_error",
            result.error == "execution_limit",
            result.ticks / config.env.max_execution_steps,
            # During replacement, length describes the current partial tree's minimum completion.
            tree._minimum_completion(source=True) / config.env.max_program_tokens,
            score_delta,
            tokens_left / config.max_seq_len,
        ],
        np.float32,
    )
    return Observation(pair.initial, pair.target, result.output, feedback, legal)


class EditStep(NamedTuple):
    """One policy transition; reevaluated marks a completed replacement."""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    reevaluated: bool


class KarelASTEditEnv:
    """Own an editing episode independently of the policy and its event history.

    reset() evaluates the seed but issues no reward. step() accepts STOP=0,
    preorder locations 1..max_nodes, or offset grammar IDs. A completed edit
    refreshes execution feedback and yields its signed score improvement.
    STOP terminates normally; completing an edit at the sequence limit truncates.
    Seed tokens, the initial report, and action/result tokens share max_seq_len.
    Every step costs one token; execution feedback is part of that action's
    resulting observation, with no additional token or pause.
    Both are terminal for this finite-budget optimization objective.
    """

    def __init__(self, config: EditConfig) -> None:
        """Initialize the task and scorer; reset is required before policy actions."""
        self.config = config
        self.task = KarelProgramEnv(config.env)
        self.scorer = KarelProgramEnv(config.env)
        self.tree = seed_tree(config)
        self.result: Evaluation | None = None
        self.observation: Observation | None = None
        self.remaining = config.max_seq_len - config.prefill_length
        self.completed_edits = 0
        self.seed_score = 0.0
        self.done = True
        self.action_space = gym.spaces.Discrete(1 + config.max_nodes + len(AST_ACTIONS))

    def reset(self, *, task: KarelProgramEnv | None = None, seed: int | None = None) -> Observation:
        """Start from the configured seed on a sampled task or a fresh supplied task."""
        if task is not None:
            if seed is not None:
                raise ValueError("Supply either task or seed, not both")
            if task.config != self.config.env:
                raise ValueError("Task must use the configured environment limits")
            self.pair = self.task.reset_from(task)
        else:
            self.pair = self.task.reset(seed=seed)
        self.tree = seed_tree(self.config)
        self.result = evaluate(self.tree, self.task, self.scorer, self.pair.initial)
        self.seed_score = self.result.score
        self.remaining = self.config.max_seq_len - self.config.prefill_length
        self.completed_edits = 0
        self.done = False
        self.observation = observe(self.tree, self.pair, self.result, self.remaining, self.config)
        return self.observation

    def step(self, action: int | np.integer) -> EditStep:
        """Apply one legal decision, awarding improvement only when an edit completes."""
        if self.done or self.observation is None or self.result is None:
            raise gym.error.ResetNeeded("Call reset() before stepping an editing episode")
        if (
            isinstance(action, (bool, np.bool_))
            or not isinstance(action, (int, np.integer))
            or not self.action_space.contains(action)
            or not self.observation.legal[int(action)]
        ):
            raise gym.error.InvalidAction(f"Illegal AST editing action: {action!r}")
        action = int(action)
        self.remaining -= 1
        reward = 0.0
        reevaluated = False
        terminated = self.tree.complete and action == 0
        if not terminated:
            if self.tree.complete:
                self.tree = open_subtree(self.tree, action - 1)
            else:
                self.tree = self.tree.expand(action - 1 - self.config.max_nodes)
                if self.tree.complete:
                    previous_score = self.result.score
                    self.result = evaluate(self.tree, self.task, self.scorer, self.pair.initial)
                    reward = self.result.score - previous_score
                    reevaluated = True
                    self.completed_edits += 1
        truncated = self.remaining == 0 and not terminated
        self.done = terminated or truncated
        if self.done and not self.tree.complete:
            raise RuntimeError("Action masks allowed an unfinished terminal program")
        # Incomplete edits retain the last execution's delta as feedback. Their
        # actual transition reward is still zero.
        delta = reward if reevaluated else float(self.observation.feedback[-2])
        self.observation = observe(self.tree, self.pair, self.result, self.remaining, self.config, delta)
        return EditStep(self.observation, reward, terminated, truncated, reevaluated)
