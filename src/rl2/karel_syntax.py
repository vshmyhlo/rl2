"""Exact token edit distance to the Karel grammar, without executing repairs.

This is minimum-cost context-free parsing. A chart entry stores the cheapest
repair of an input span into a nonterminal. Terminal spans allow substitution
and deletion; empty spans allow insertion. Binary rules combine spans, and a
precomputed weighted unary closure accounts for inserted sibling subtrees.
Cost is one per inserted, deleted, or replaced token. PAD is not a grammar token.

The usual 128-token case uses the compact recursive grammar. Longer inputs use
a depth-expanded grammar to match the interpreter's 64-block nesting limit.
Neither the task's reference program nor execution/token budgets constrain the
hypothetical repair. Only the submitted program is ever executed.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush

import numpy as np
from numpy.typing import NDArray

MAX_BLOCK_DEPTH = 64
_INF = 1_000_000_000
type IntArray = NDArray[np.int32]
type Production = tuple[str, tuple[str, ...]]


@dataclass(frozen=True)
class _Grammar:
    terminals: tuple[tuple[int, str], ...]
    binary: IntArray
    empty: IntArray
    closure: IntArray


@lru_cache(maxsize=2)
def _grammar(max_depth: int | None) -> _Grammar:
    # Imported lazily because karel uses this module for terminal rewards.
    from rl2.karel import ACTIONS, PREDICATES

    rules: list[Production] = []

    def add(lhs: str, *rhs: str) -> None:
        rules.append((lhs, rhs))

    add("Program", "DEF", "run", "m(", "Block0", "m)")
    for depth in range(max_depth + 1 if max_depth is not None else 1):
        block, statement = f"Block{depth}", f"Statement{depth}"
        child = f"Block{depth + 1}" if max_depth is not None else block
        add(block, statement)
        add(block, statement, block)
        add(statement, "Action")
        if depth == max_depth:
            continue
        add(statement, "REPEAT", "Count", "r(", child, "r)")
        add(statement, "WHILE", "Condition", "w(", child, "w)")
        add(statement, "IF", "Condition", "i(", child, "i)")
        add(statement, "IFELSE", "Condition", "i(", child, "i)", "ELSE", "e(", child, "e)")
    add("Condition", "c(", "Predicate", "c)")
    add("Condition", "c(", "not", "c(", "Predicate", "c)", "c)")
    for name, tokens in (("Action", ACTIONS), ("Predicate", PREDICATES), ("Count", tuple(f"R={i}" for i in range(20)))):
        for token in tokens:
            add(name, token)

    # Convert to terminal, unary, and binary productions, preserving the language.
    names = {lhs: index for index, lhs in enumerate(dict.fromkeys(lhs for lhs, _ in rules))}
    nonterminals = set(names)
    terminals: list[tuple[int, str]] = []
    unary: list[tuple[int, int]] = []
    binary: list[tuple[int, int, int]] = []

    def symbol(name: str) -> int:
        if name not in names:
            names[name] = len(names)
        return names[name]

    for rule_index, (lhs, rhs) in enumerate(rules):
        parent = symbol(lhs)
        if len(rhs) == 1:
            if rhs[0] in nonterminals:
                unary.append((parent, symbol(rhs[0])))
            else:
                terminals.append((parent, rhs[0]))
            continue
        children = []
        for token in rhs:
            if token in nonterminals:
                children.append(symbol(token))
            else:
                terminal = f"terminal:{token}"
                if terminal not in names:
                    terminals.append((symbol(terminal), token))
                children.append(symbol(terminal))
        for index, child in enumerate(children[:-2]):
            tail = symbol(f"tail:{rule_index}:{index}")
            binary.append((parent, child, tail))
            parent = tail
        binary.append((parent, *children[-2:]))

    binary_array = np.asarray(binary, dtype=np.int32)
    unary_array = np.asarray(unary, dtype=np.int32)
    parents, left, right = binary_array.T
    empty = np.full(len(names), _INF, dtype=np.int32)
    empty[[parent for parent, _ in terminals]] = 1
    while True:
        previous = empty.copy()
        np.minimum.at(empty, unary_array[:, 0], empty[unary_array[:, 1]])
        np.minimum.at(empty, parents, empty[left] + empty[right])
        if np.array_equal(previous, empty):
            break
    if np.any(empty == _INF):
        raise ValueError("Every grammar symbol must derive a finite string")

    # A binary rule can consume input on just one side, inserting the other.
    edges: list[list[tuple[int, int]]] = [[] for _ in names]
    for parent, child in unary:
        edges[parent].append((child, 0))
    for parent, a, b in binary:
        edges[parent].extend(((a, int(empty[b])), (b, int(empty[a]))))
    closure = np.full((len(names), len(names)), _INF, dtype=np.int32)
    for source in range(len(names)):
        distances = closure[source]
        distances[source] = 0
        queue = [(0, source)]
        while queue:
            cost, node = heappop(queue)
            if cost != distances[node]:
                continue
            for child, weight in edges[node]:
                candidate = cost + weight
                if candidate < distances[child]:
                    distances[child] = candidate
                    heappush(queue, (candidate, child))
    return _Grammar(tuple(terminals), binary_array, empty, closure)


def syntax_edit_distance(tokens: Sequence[str]) -> int:
    """Fewest token edits to any complete program accepted by the Karel grammar.

    All blocks are nonempty, counts are R=0..R=19, and conditions allow at most
    one `not`, exactly as in the interpreter. Unknown symbols count as ordinary
    erroneous tokens. The distance is independent of the task state.
    """
    tokens = tuple(tokens)
    if any(not isinstance(token, str) for token in tokens):
        raise TypeError("Expected a sequence of token strings")
    size = len(tokens)
    # A five-token primitive program is at most max(n, 5) edits away. Thus any
    # optimal repair is at most n + max(n, 5) tokens long. Nesting 65 controls
    # needs at least 4*65 + 5 tokens, so short inputs cannot need the depth bound.
    bounded = size + max(size, 5) >= 4 * (MAX_BLOCK_DEPTH + 1) + 5
    grammar = _grammar(MAX_BLOCK_DEPTH if bounded else None)
    if size == 0:
        return int(grammar.empty[0])
    parents, left, right = grammar.binary.T
    chart = np.full((len(grammar.empty), size + 1, size + 1), _INF, dtype=np.int32)
    for index in range(size + 1):
        chart[:, index, index] = grammar.empty
    # Prefix counts let a terminal keep any matching token and delete the rest.
    matches = {
        token: np.concatenate(([0], np.cumsum([value == token for value in tokens]))) for _, token in grammar.terminals
    }
    for length in range(1, size + 1):
        starts = np.arange(size - length + 1)
        ends = starts + length
        costs = np.broadcast_to(grammar.empty[:, None] + length, (len(grammar.empty), len(starts))).copy()
        for parent, token in grammar.terminals:
            terminal_cost = length - (matches[token][ends] > matches[token][starts])
            np.minimum(costs[parent], terminal_cost, out=costs[parent])
        if length > 1:
            splits = starts[:, None] + np.arange(1, length)
            # Chunk rules to bound temporary allocation for the expanded grammar.
            for offset in range(0, len(parents), 32):
                chunk = slice(offset, offset + 32)
                candidates = (
                    chart[left[chunk, None, None], starts[None, :, None], splits[None]]
                    + chart[right[chunk, None, None], splits[None], ends[None, :, None]]
                ).min(axis=-1)
                np.minimum.at(costs, parents[chunk], candidates)
        # Every nonempty split uses shorter spans, so only unary closure remains.
        for offset in range(0, len(grammar.empty), 32):
            chunk = slice(offset, offset + 32)
            chart[chunk, starts, ends] = (grammar.closure[chunk, :, None] + costs[None]).min(axis=1)
    return int(chart[0, 0, size])


def syntax_reward(tokens: Sequence[str]) -> float:
    """Score invalid/incomplete syntax in (-2, -1], without executing a repair.

    A zero distance is accepted for diagnostics and returns -1. The environment
    calls this only on syntax errors and incomplete submissions (distance >= 1),
    whose rewards are at most -1.5. Valid programs use execution rewards instead.
    """
    return -2.0 + 1.0 / (1.0 + syntax_edit_distance(tokens))
