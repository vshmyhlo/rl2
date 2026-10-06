"""Karel AST construction with simultaneous expansion of current typed holes.

Actions are constructors or primitive values, not DSL tokens. Every constructor
creates all its child holes; new holes are expanded in the following round.
Single-hole DFS expansion remains available for source conversion utilities.
Lists use Cons/End, with a separate nonempty-list type for block bodies. Minimum
completion costs enforce node/depth and optional printed-source token budgets
without leaving unfinished trees.
AST depth includes list nodes and is distinct from interpreter control depth.
"""

from dataclasses import dataclass, replace
from enum import IntEnum
from functools import cached_property, lru_cache
from types import MappingProxyType
from typing import NamedTuple

import chex
import jax
import numpy as np
from numpy.typing import NDArray

from rl2.karel import ACTIONS, PREDICATES, _parse, _Statement
from rl2.karel_syntax import MAX_BLOCK_DEPTH
from rl2.shape_checker import ShapeChecker


class Hole(IntEnum):
    PROGRAM = 0
    NONEMPTY = 1
    LIST = 2
    STATEMENT = 3
    CONDITION = 4
    PREDICATE = 5
    COUNT = 6


class Field(IntEnum):
    ROOT = 0
    BODY = 1
    HEAD = 2
    TAIL = 3
    CONDITION = 4
    THEN = 5
    ELSE = 6
    PREDICATE = 7
    COUNT = 8


class Constructor(NamedTuple):
    name: str
    result: Hole
    fields: tuple[tuple[Field, Hole], ...]


CONSTRUCTORS = (
    Constructor("Program", Hole.PROGRAM, ((Field.BODY, Hole.NONEMPTY),)),
    Constructor("ConsNonEmpty", Hole.NONEMPTY, ((Field.HEAD, Hole.STATEMENT), (Field.TAIL, Hole.LIST))),
    Constructor("Cons", Hole.LIST, ((Field.HEAD, Hole.STATEMENT), (Field.TAIL, Hole.LIST))),
    Constructor("End", Hole.LIST, ()),
    *(Constructor(action, Hole.STATEMENT, ()) for action in ACTIONS),
    Constructor("IF", Hole.STATEMENT, ((Field.CONDITION, Hole.CONDITION), (Field.THEN, Hole.NONEMPTY))),
    Constructor(
        "IFELSE",
        Hole.STATEMENT,
        ((Field.CONDITION, Hole.CONDITION), (Field.THEN, Hole.NONEMPTY), (Field.ELSE, Hole.NONEMPTY)),
    ),
    Constructor("WHILE", Hole.STATEMENT, ((Field.CONDITION, Hole.CONDITION), (Field.BODY, Hole.NONEMPTY))),
    Constructor("REPEAT", Hole.STATEMENT, ((Field.COUNT, Hole.COUNT), (Field.BODY, Hole.NONEMPTY))),
    Constructor("Test", Hole.CONDITION, ((Field.PREDICATE, Hole.PREDICATE),)),
    Constructor("Not", Hole.CONDITION, ((Field.PREDICATE, Hole.PREDICATE),)),
)
VALUES = tuple((name, Hole.PREDICATE) for name in PREDICATES) + tuple((f"R={i}", Hole.COUNT) for i in range(20))
AST_ACTIONS = ("<pad>", *(c.name for c in CONSTRUCTORS), *(name for name, _ in VALUES))
ACTION_ID = MappingProxyType({name: index for index, name in enumerate(AST_ACTIONS)})
NUM_NODE_TYPES = 1 + len(CONSTRUCTORS) + len(Hole)  # 0 is padding; typed holes/value nodes follow constructors.
_INF = 1_000_000
_SOURCE_COST = {
    "Program": 4,
    "ConsNonEmpty": 0,
    "Cons": 0,
    "End": 0,
    **{action: 1 for action in ACTIONS},
    "IF": 3,
    "IFELSE": 6,
    "WHILE": 3,
    "REPEAT": 3,
    "Test": 2,
    "Not": 5,
}


def _limits(max_nodes: int, max_depth: int) -> None:
    if type(max_nodes) is not int or type(max_depth) is not int:
        raise TypeError("AST limits must be integers")
    chex.assert_scalar_positive(max_nodes)
    chex.assert_scalar_in(max_depth, 0, MAX_BLOCK_DEPTH)


@lru_cache(maxsize=2 * (MAX_BLOCK_DEPTH + 1))
def _minimum_sizes(max_depth: int, source: bool = False) -> NDArray[np.int32]:
    """Minimum node or printed-token cost by remaining depth and hole type."""
    sizes = np.full((max_depth + 1, len(Hole)), _INF, np.int32)
    for depth in range(max_depth + 1):
        for _, hole in VALUES:
            sizes[depth, hole] = 1
        for constructor in CONSTRUCTORS:
            if constructor.fields and depth == 0:
                continue
            local = _SOURCE_COST[constructor.name] if source else 1
            cost = local + sum(int(sizes[depth - 1, child]) for _, child in constructor.fields)
            sizes[depth, constructor.result] = min(sizes[depth, constructor.result], cost)
    sizes.flags.writeable = False
    return sizes


@lru_cache(maxsize=2 * (MAX_BLOCK_DEPTH + 1) * len(Hole))
def _action_costs(remaining_depth: int, hole: Hole, source: bool) -> NDArray[np.int64]:
    """Read-only grammar costs shared by nodes with the same type and depth budget."""
    sizes = _minimum_sizes(remaining_depth, source)
    costs = np.full(len(AST_ACTIONS), _INF, np.int64)
    for action, constructor in enumerate(CONSTRUCTORS, 1):
        if constructor.result != hole or (constructor.fields and remaining_depth == 0):
            continue
        local = _SOURCE_COST[constructor.name] if source else 1
        costs[action] = local + sum(int(sizes[remaining_depth - 1, child]) for _, child in constructor.fields)
    for value, (_, value_hole) in enumerate(VALUES, 1):
        if value_hole == hole:
            costs[len(CONSTRUCTORS) + value] = 1
    costs.flags.writeable = False
    return costs


@dataclass(frozen=True)
class Node:
    hole_type: Hole
    field: Field = Field.ROOT
    depth: int = 0
    child_index: int = 0
    constructor: int = 0  # One-based constructor ID; 0 until expanded or for value nodes.
    value: int = 0  # One-based index into VALUES, independent of action ID.
    children: tuple[int, ...] = ()

    @property
    def is_hole(self) -> bool:
        return self.constructor == 0 and self.value == 0


type FeatureArray = jax.Array | NDArray[np.int32] | NDArray[np.bool_]


class ASTFeatures(NamedTuple):
    """Fixed-capacity preorder representation of the current partial AST.

    Example partial AST with max_nodes=9, max_depth=4, and no source-token
    budget (? marks an unresolved hole)::

        Program
          BODY: ConsNonEmpty
            HEAD: REPEAT
              COUNT: R=2
              BODY: ConsNonEmpty
                HEAD: ?STATEMENT
                TAIL: ?LIST
            TAIL: ?LIST
        <padding>

    This represents a repeat block whose first statement and list endings
    are still undecided. The attribute examples below use this tree in
    preorder, displaying names instead of numeric IDs where applicable.

    N is max_nodes and A is len(AST_ACTIONS). Node arrays have shape [N]
    for one tree or [B, N] for a batch. Features contain only existing nodes
    and unresolved holes, with unused positions padded with zeros/False.

    Attributes:
        node_type: Int32 constructor IDs (1-based), or
            1 + len(CONSTRUCTORS) + Hole ID for holes and resolved value nodes.
            Zero denotes padding; is_hole distinguishes holes from filled values.
            Example: [Program, ConsNonEmpty, REPEAT, COUNT, ConsNonEmpty,
            STATEMENT, LIST, LIST, <padding>]. R=2 retains the COUNT type.
        field: Int32 Field IDs identifying each node's role in its parent
            (BODY, CONDITION, etc.); the root uses Field.ROOT.
            Example: [ROOT, BODY, HEAD, COUNT, BODY, HEAD, TAIL, TAIL,
            <padding>].
        depth: Int32 distance from the root in AST edges. The root is at 0;
            list nodes count toward depth, unlike control-block nesting depth.
            Example: [0, 1, 2, 3, 3, 4, 4, 2, 0]; the statement hole has
            AST depth 4 even though it is inside just one repeat block.
        child_index: Int32 zero-based position among the parent's children;
            the root uses 0.
            Example: REPEAT's COUNT is its first child and BODY is its
            second child; each ConsNonEmpty has HEAD first and TAIL second.
        value: Int32 1-based index into VALUES for resolved predicates/counts,
            otherwise 0. These indices are not AST action IDs.
            Example: [<unset>, <unset>, <unset>, R=2, <unset>, <unset>,
            <unset>, <unset>, <padding>]. Only the COUNT node has a value.
        is_hole: Computed bool mask for unresolved nodes: present nodes with
            a hole/value type and value=0. Not stored in the tuple.
            Example: [False, False, False, False, False, True, True, True,
            False]; the resolved count and padding are not holes.
        seq_len: Int32 scalar, or [B] for a batch, counting existing nodes,
            including holes. Nodes occupy the left-aligned valid segment
            [0, seq_len); the remaining positions are right padding excluded
            from attention keys. Example: 8.
        action_mask: Bool [N, A] or [B, N, A] array of legal expansions at
            each current hole. Shared completion budgets are allocated before
            sampling so any combination of allowed choices fits. Resolved nodes,
            padding, and finished trees have all-False masks; PAD is never legal.
            Sequential utility snapshots enable only the frontier row.
            Example: shape [9, 41], with True entries for move, turnLeft,
            turnRight, pickMarker, and putMarker at the STATEMENT hole,
            and for End at both LIST holes. All other entries are False.
            The depth/node budgets rule out larger expansions.
            With features(parallel=False), only the STATEMENT hole is enabled.
    """

    node_type: FeatureArray
    field: FeatureArray
    depth: FeatureArray
    child_index: FeatureArray
    value: FeatureArray
    seq_len: jax.Array | NDArray[np.int32]
    action_mask: FeatureArray  # [N,A]/[B,N,A]; inactive node rows are all false.

    @property
    def is_hole(self) -> jax.Array | NDArray[np.bool_]:
        """Infer unresolved nodes from their type/value, excluding padding.

        Preserves NumPy or JAX arrays and any batch/rollout leading dimensions.
        """
        sc = ShapeChecker()
        leading_dims = "abcdefghijklmnopqrstuvwxyz"[: self.seq_len.ndim]
        sc.check(self.seq_len, leading_dims, np.int32)
        sc.check((self.node_type, self.value), leading_dims + "N", np.int32)
        present = np.arange(self.node_type.shape[-1]) < self.seq_len[..., None]
        result = present & (self.node_type > len(CONSTRUCTORS)) & (self.value == 0)
        sc.check(result, leading_dims + "N", np.bool_)
        return result


@dataclass(frozen=True)
class KarelAST:
    nodes: tuple[Node, ...]
    max_nodes: int
    max_depth: int
    max_program_tokens: int | None = None

    @classmethod
    def empty(cls, max_nodes: int = 128, max_depth: int = 64, max_program_tokens: int | None = None) -> "KarelAST":
        _limits(max_nodes, max_depth)
        minimum = int(_minimum_sizes(max_depth)[max_depth, Hole.PROGRAM])
        if minimum >= _INF or minimum > max_nodes:
            raise ValueError("AST budgets cannot fit even a one-statement program")
        if max_program_tokens is not None:
            if type(max_program_tokens) is not int:
                raise TypeError("max_program_tokens must be an integer or None")
            if max_program_tokens < 5:
                raise ValueError("Source budget cannot fit a one-statement program")
        return cls((Node(Hole.PROGRAM),), max_nodes, max_depth, max_program_tokens)

    def preorder(self) -> tuple[int, ...]:
        return self._preorder

    @cached_property
    def _preorder(self) -> tuple[int, ...]:
        order: list[int] = []
        pending = [0]
        while pending:
            index = pending.pop()
            order.append(index)
            pending.extend(reversed(self.nodes[index].children))
        return tuple(order)

    @property
    def frontier(self) -> int | None:
        return next((index for index in self.preorder() if self.nodes[index].is_hole), None)

    @property
    def complete(self) -> bool:
        return self.frontier is None

    def _costs(self, index: int, *, source: bool = False) -> NDArray[np.int64]:
        """Minimum completed subtree costs for each grammar-legal action."""
        node = self.nodes[index]
        return _action_costs(self.max_depth - node.depth, node.hole_type, source)

    def _minimum_completion(self, *, source: bool = False) -> int:
        sizes = _minimum_sizes(self.max_depth, source)
        return sum(
            int(sizes[self.max_depth - node.depth, node.hole_type])
            if node.is_hole
            else _SOURCE_COST[CONSTRUCTORS[node.constructor - 1].name]
            if source and node.constructor
            else 1
            for node in self.nodes
        )

    def allowed_actions(self) -> NDArray[np.bool_]:
        """Legal actions for a single DFS-frontier expansion (utility API)."""
        index = self.frontier
        if index is None:
            return np.zeros(len(AST_ACTIONS), np.bool_)
        node = self.nodes[index]
        costs = self._costs(index)
        minimum = _minimum_sizes(self.max_depth)[self.max_depth - node.depth, node.hole_type]
        allowed = (costs < _INF) & (self._minimum_completion() - minimum + costs <= self.max_nodes)
        if self.max_program_tokens is not None:
            costs = self._costs(index, source=True)
            minimum = _minimum_sizes(self.max_depth, True)[self.max_depth - node.depth, node.hole_type]
            allowed &= self._minimum_completion(source=True) - minimum + costs <= self.max_program_tokens
        return allowed

    def parallel_action_mask(self) -> NDArray[np.bool_]:
        """Read-only mask cached for this immutable tree, including during validation.

        Expansion returns a new tree with a fresh cache, so features() and
        expand_round() reuse the same mask without accepting caller-supplied masks.
        """
        return self._parallel_action_mask

    @cached_property
    def _parallel_action_mask(self) -> NDArray[np.bool_]:
        """Allocate shared slack before sampling independent choices at every hole.

        Reserve the cheapest completion of every hole, then split extra capacity
        equally among holes able to use it (preorder breaks remainder ties).
        Node and source slack are allocated separately. Unspent shares become
        available next round. This conservative mask may exclude a choice that
        would fit if other holes volunteered more of their shares.
        """
        order = self.preorder()
        holes = [(position, index) for position, index in enumerate(order) if self.nodes[index].is_hole]
        result = np.zeros((self.max_nodes, len(AST_ACTIONS)), np.bool_)
        if not holes:
            result.flags.writeable = False
            return result
        extras = []
        slacks = []
        for source, limit in ((False, self.max_nodes), (True, self.max_program_tokens)):
            if limit is None:
                continue
            sizes = _minimum_sizes(self.max_depth, source)
            extras.append(
                np.stack(
                    [
                        self._costs(index, source=source)
                        - int(sizes[self.max_depth - self.nodes[index].depth, self.nodes[index].hole_type])
                        for _, index in holes
                    ]
                )
            )
            slacks.append(limit - self._minimum_completion(source=source))
        legal = np.stack([self._costs(index) < _INF for _, index in holes])
        for costs, slack in zip(extras, slacks):
            legal &= costs <= slack
        allocated = legal.copy()
        for costs, slack in zip(extras, slacks):
            eligible = np.flatnonzero(((costs > 0) & legal).any(axis=-1))
            shares = np.zeros(len(holes), np.int64)
            if len(eligible):
                quotient, remainder = divmod(slack, len(eligible))
                shares[eligible] = quotient
                shares[eligible[:remainder]] += 1
            allocated &= costs <= shares[:, None]
        for (position, _), allowed in zip(holes, allocated):
            result[position] = allowed
        result.flags.writeable = False
        return result

    def expand_round(self, actions: NDArray[np.int32]) -> "KarelAST":
        """Apply one action per current hole using pre-round preorder positions.

        All inactive positions must contain PAD=0. Validate the whole round
        before applying it; children created here wait until the next round.
        """
        sc = ShapeChecker(N=self.max_nodes, A=len(AST_ACTIONS))
        sc.check(actions, "N", dtype=np.int32)
        mask = self.parallel_action_mask()
        sc.check(mask, "NA", dtype=np.bool_)
        active = mask.any(axis=-1)
        if np.any(actions[~active] != 0):
            raise ValueError("Non-hole positions must use PAD")
        if np.any((actions[active] <= 0) | (actions[active] >= len(AST_ACTIONS))):
            raise ValueError("Invalid AST action in parallel round")
        if not mask[np.flatnonzero(active), actions[active]].all():
            raise ValueError("Invalid AST action for parallel grammar or budget allocation")
        tree = self
        for position, index in enumerate(self.preorder()):
            if active[position]:
                tree = tree._expand_at(index, int(actions[position]))
        return tree

    def expand(self, action: int) -> "KarelAST":
        if not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_)):
            raise TypeError("AST action must be an integer ID")
        if action < 0 or action >= len(AST_ACTIONS) or not self.allowed_actions()[action]:
            raise ValueError(f"Invalid AST action {action} for the frontier or remaining budgets")
        index = self.frontier
        assert index is not None
        return self._expand_at(index, action)

    def _expand_at(self, index: int, action: int) -> "KarelAST":
        """Apply a validated action at a stable storage index."""
        nodes = list(self.nodes)
        node = nodes[index]
        if action <= len(CONSTRUCTORS):
            constructor = CONSTRUCTORS[action - 1]
            children = tuple(range(len(nodes), len(nodes) + len(constructor.fields)))
            nodes[index] = replace(node, constructor=int(action), children=children)
            nodes.extend(
                Node(hole, field, node.depth + 1, child_index)
                for child_index, (field, hole) in enumerate(constructor.fields)
            )
        else:
            nodes[index] = replace(node, value=int(action) - len(CONSTRUCTORS))
        return replace(self, nodes=tuple(nodes))

    def features(self, *, parallel: bool = True) -> ASTFeatures:
        order = self.preorder()
        integers = np.zeros((5, self.max_nodes), np.int32)
        for position, index in enumerate(order):
            node = self.nodes[index]
            integers[:, position] = (
                node.constructor or 1 + len(CONSTRUCTORS) + int(node.hole_type),
                int(node.field),
                node.depth,
                node.child_index,
                node.value,
            )
        action_mask = (
            self.parallel_action_mask() if parallel else np.zeros((self.max_nodes, len(AST_ACTIONS)), np.bool_)
        )
        if not parallel:
            frontier = self.frontier
            if frontier is not None:
                action_mask[order.index(frontier)] = self.allowed_actions()
        return ASTFeatures(
            *integers,
            np.asarray(len(order), np.int32),
            action_mask,
        )

    def tokens(self) -> tuple[str, ...]:
        if not self.complete:
            raise ValueError("Cannot print an AST with unresolved holes")

        def render(index: int) -> tuple[str, ...]:
            node = self.nodes[index]
            if node.value:
                return (VALUES[node.value - 1][0],)
            kind = CONSTRUCTORS[node.constructor - 1].name
            children = [render(child) for child in node.children]
            if kind == "Program":
                return ("DEF", "run", "m(", *children[0], "m)")
            if kind in ("ConsNonEmpty", "Cons"):
                return (*children[0], *children[1])
            if kind == "End":
                return ()
            if kind == "Test":
                return ("c(", *children[0], "c)")
            if kind == "Not":
                return ("c(", "not", "c(", *children[0], "c)", "c)")
            if kind in ACTIONS:
                return (kind,)
            if kind == "REPEAT":
                return (kind, *children[0], "r(", *children[1], "r)")
            opening, closing = ("w(", "w)") if kind == "WHILE" else ("i(", "i)")
            result = (kind, *children[0], opening, *children[1], closing)
            return (*result, "ELSE", "e(", *children[2], "e)") if kind == "IFELSE" else result

        return render(0)

    def source(self) -> str:
        lines: list[str] = []
        current: list[str] = []
        depth = 0
        for token in self.tokens():
            if token in ("m)", "w)", "i)", "e)", "r)"):
                depth -= 1
            current.append(token)
            if token in ACTIONS or token in ("m(", "w(", "i(", "e(", "r(", "m)", "w)", "i)", "e)", "r)"):
                lines.append("  " * depth + " ".join(current))
                current.clear()
            if token in ("m(", "w(", "i(", "e(", "r("):
                depth += 1
        return "\n".join(lines)


def program_actions(tokens: tuple[str, ...]) -> tuple[int, ...]:
    """Parse source and emit its unique DFS constructor/value action sequence."""
    statements = _parse(tokens)
    actions = [ACTION_ID["Program"]]

    def body(statements: tuple[_Statement, ...]) -> None:
        for index, statement in enumerate(statements):
            actions.extend((ACTION_ID["ConsNonEmpty" if index == 0 else "Cons"], ACTION_ID[statement.kind]))
            if statement.kind == "REPEAT":
                actions.append(ACTION_ID[f"R={statement.count}"])
                body(statement.body)
            elif statement.kind not in ACTIONS:
                predicate, negate = statement.condition
                actions.extend((ACTION_ID["Not" if negate else "Test"], ACTION_ID[predicate]))
                body(statement.body)
                if statement.kind == "IFELSE":
                    body(statement.otherwise)
        actions.append(ACTION_ID["End"])

    body(statements)
    return tuple(actions)


def teacher_forcing(
    tokens: tuple[str, ...], max_nodes: int = 128, max_depth: int = 64
) -> tuple[tuple[ASTFeatures, ...], NDArray[np.int32]]:
    """Snapshot each partial tree BEFORE its target action; no future-node leakage."""
    tree = KarelAST.empty(max_nodes, max_depth)
    features: list[ASTFeatures] = []
    actions = program_actions(tokens)
    for action in actions:
        features.append(tree.features(parallel=False))
        tree = tree.expand(action)
    assert tree.complete
    return tuple(features), np.asarray(actions, np.int32)


def batch_features(features: tuple[ASTFeatures, ...]) -> ASTFeatures:
    """Stack unbatched snapshots with matching capacities and valid array metadata."""
    if not features:
        raise ValueError("Cannot batch zero ASTs")
    sc = ShapeChecker(B=len(features), A=len(AST_ACTIONS))
    for snapshot in features:
        sc.check(snapshot[:5], "N", dtype=np.int32)
        sc.check(snapshot.seq_len, "", dtype=np.int32)
        chex.assert_scalar_in(int(snapshot.seq_len), 0, snapshot.node_type.shape[-1])
        sc.check(snapshot.action_mask, "NA", dtype=np.bool_)
    result = ASTFeatures(*(np.stack(values) for values in zip(*features)))
    sc.check(result[:5], "BN", dtype=np.int32)
    sc.check(result.seq_len, "B", dtype=np.int32)
    sc.check(result.action_mask, "BNA", dtype=np.bool_)
    return result
