from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.karel import ACTIONS, PREDICATES, _parse, execute_program, sample_task
from rl2.karel_ast import (
    ACTION_ID,
    AST_ACTIONS,
    ASTFeatures,
    Field,
    KarelAST,
    batch_features,
    program_actions,
    teacher_forcing,
)


def build(program: str, max_nodes: int = 256, max_depth: int = 64) -> KarelAST:
    tree = KarelAST.empty(max_nodes, max_depth)
    for action in program_actions(tuple(program.split())):
        assert tree.allowed_actions()[action]
        tree = tree.expand(action)
    return tree


def test_parallel_mask_cache_reused_and_invalidated_on_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    tree = KarelAST.empty(16, 4, 16)
    mask = tree.features().action_mask
    assert tree.parallel_action_mask() is mask
    assert tree.preorder() is tree.preorder()
    with pytest.raises(ValueError, match="read-only"):
        mask[0, ACTION_ID["move"]] = True
    original_costs = KarelAST._costs
    calls = []

    def costs(self: KarelAST, index: int, *, source: bool = False) -> np.ndarray:
        calls.append(index)
        return original_costs(self, index, source=source)

    monkeypatch.setattr(KarelAST, "_costs", costs)
    actions = np.zeros(16, np.int32)
    actions[0] = ACTION_ID["Program"]
    expanded = tree.expand_round(actions)
    assert not calls  # Validation reuses the mask that supplied the logits.
    new_mask = expanded.features().action_mask
    assert calls
    assert new_mask is not mask
    assert new_mask[1, ACTION_ID["ConsNonEmpty"]]
    assert not new_mask[0].any()
    actions[0] = ACTION_ID["move"]
    with pytest.raises(ValueError, match="Invalid AST action"):
        tree.expand_round(actions)


def test_all_karel_constructs_round_trip() -> None:
    bodies = [*ACTIONS, *(f"REPEAT R={i} r( move r)" for i in range(20))]
    for predicate in PREDICATES:
        for condition in (f"c( {predicate} c)", f"c( not c( {predicate} c) c)"):
            bodies.extend(
                (
                    f"IF {condition} i( move i)",
                    f"WHILE {condition} w( turnLeft w)",
                    f"IFELSE {condition} i( pickMarker i) ELSE e( putMarker e)",
                )
            )
    for body in bodies:
        program = f"DEF run m( {body} turnRight m)"
        tree = build(program)
        assert tree.complete
        assert tree.tokens() == tuple(program.split())
        assert tree.source().split() == program.split()
        assert len(program_actions(tree.tokens())) == len(tree.nodes)
        _parse(tree.tokens())


def test_reference_solutions_preserve_execution() -> None:
    rng = np.random.default_rng(15)
    for _ in range(3):
        task = sample_task(rng)
        tree = build(" ".join(task.program))
        assert tree.tokens() == task.program
        np.testing.assert_array_equal(execute_program(tree.tokens(), task.initial), task.target)


@pytest.mark.parametrize("nodes,depth", [(4, 2), (16, 8)])  # Minimum tree and room for branching.
def test_every_masked_expansion_can_finish_in_budget(nodes: int, depth: int) -> None:
    rng = np.random.default_rng(25)
    for _ in range(3):
        tree = KarelAST.empty(nodes, depth)
        steps = 0
        while not tree.complete:
            legal = np.flatnonzero(tree.allowed_actions())
            assert len(legal) > 0
            tree = tree.expand(int(rng.choice(legal)))
            steps += 1
            assert steps <= nodes
        assert not tree.allowed_actions().any()
        assert max(node.depth for node in tree.nodes) <= depth
        assert steps == len(tree.nodes)
        _parse(tree.tokens())


def test_typed_frontier_and_preorder_features() -> None:
    tree = KarelAST.empty(32, 16)
    assert np.flatnonzero(tree.allowed_actions()).tolist() == [ACTION_ID["Program"]]
    for name in ("Program", "ConsNonEmpty", "IFELSE"):
        tree = tree.expand(ACTION_ID[name])
    assert {AST_ACTIONS[a] for a in np.flatnonzero(tree.allowed_actions())} == {"Test", "Not"}
    features = tree.features()
    assert int(features.is_hole.argmax()) == 3
    # Full preorder includes the unexpanded branches and the outer list tail.
    np.testing.assert_array_equal(
        features.field[:7], [Field.ROOT, Field.BODY, Field.HEAD, Field.CONDITION, Field.THEN, Field.ELSE, Field.TAIL]
    )
    np.testing.assert_array_equal(features.depth[:7], [0, 1, 2, 3, 3, 3, 2])
    np.testing.assert_array_equal(features.child_index[:7], [0, 0, 0, 0, 1, 2, 1])
    np.testing.assert_array_equal(features.is_hole[:7], [False, False, False, True, True, True, True])
    assert features.node_mask.sum() == 7
    tree = tree.expand(ACTION_ID["Not"])
    assert {AST_ACTIONS[a] for a in np.flatnonzero(tree.allowed_actions())} == set(PREDICATES)
    tree = tree.expand(ACTION_ID["frontIsClear"])
    assert tree.nodes[tree.frontier].field == Field.THEN


def test_teacher_forcing_contains_only_prefix_information() -> None:
    programs = [tuple(f"DEF run m( {action} m)".split()) for action in ("move", "turnLeft")]
    examples = [teacher_forcing(program, max_nodes=8, max_depth=4) for program in programs]
    for index in range(3):
        for a, b in zip(examples[0][0][index], examples[1][0][index]):
            np.testing.assert_array_equal(a, b)
    assert examples[0][1][2] != examples[1][1][2]
    for program, (features, actions) in zip(programs, examples):
        tree = KarelAST.empty(8, 4)
        for snapshot, action in zip(features, actions):
            for actual, expected in zip(snapshot, tree.features(parallel=False)):
                np.testing.assert_array_equal(actual, expected)
            tree = tree.expand(int(action))
        assert tree.tokens() == program
    batched = batch_features(tuple(e[0][2] for e in examples))
    assert batched.node_type.shape == (2, 8)
    assert batched.action_mask.shape == (2, 8, len(AST_ACTIONS))


def test_invalid_actions_and_unfinished_printing_fail_without_mutating_tree() -> None:
    tree = KarelAST.empty(8, 4)
    for action in (-1, 0, len(AST_ACTIONS), ACTION_ID["move"]):
        with pytest.raises(ValueError, match="Invalid AST action"):
            tree.expand(action)
    with pytest.raises(TypeError):
        tree.expand(True)
    with pytest.raises(ValueError, match="unresolved holes"):
        tree.source()
    assert len(tree.nodes) == 1
    complete = build("DEF run m( move m)")
    with pytest.raises(ValueError):
        complete.expand(ACTION_ID["Program"])
    with pytest.raises(ValueError):
        batch_features(())


@pytest.mark.parametrize("nodes,depth", [(3, 64), (128, 1), (2_000_000, 0)])
def test_impossible_budgets_are_rejected(nodes: int, depth: int) -> None:
    with pytest.raises(ValueError, match="cannot fit"):
        KarelAST.empty(nodes, depth)


def test_reference_exceeding_ast_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid AST action"):
        teacher_forcing(("DEF", "run", "m(", "move", "turnLeft", "m)"), max_nodes=4, max_depth=2)


@pytest.mark.parametrize("budget", [5, 15])  # Minimum and branching; exact construct limits are tested below.
def test_source_budget_always_allows_completion(budget: int) -> None:
    rng = np.random.default_rng(53)
    for _ in range(3):
        tree = KarelAST.empty(64, 16, max_program_tokens=budget)
        for _ in range(64):
            if tree.complete:
                break
            actions = np.flatnonzero(tree.allowed_actions())
            assert len(actions)
            tree = tree.expand(int(rng.choice(actions)))
        assert tree.complete and len(tree.tokens()) <= budget
        _parse(tree.tokens())


@pytest.mark.parametrize(
    "body",
    [
        "move turnLeft putMarker",
        "REPEAT R=19 r( move r)",
        "WHILE c( not c( frontIsClear c) c) w( turnLeft w)",
        "IF c( markersPresent c) i( pickMarker i)",
        "IFELSE c( frontIsClear c) i( move i) ELSE e( REPEAT R=0 r( turnLeft r) e)",
    ],
)
def test_exact_source_budget_matches_printer(body: str) -> None:
    tokens = tuple(f"DEF run m( {body} m)".split())
    actions = program_actions(tokens)
    tree = KarelAST.empty(128, 64, len(tokens))
    for action in actions:
        tree = tree.expand(action)
    assert tree.tokens() == tokens
    tree = KarelAST.empty(128, 64, len(tokens) - 1)
    with pytest.raises(ValueError, match="Invalid AST action"):
        for action in actions:
            tree = tree.expand(action)


def test_invalid_source_budget() -> None:
    with pytest.raises(ValueError, match="Source budget"):
        KarelAST.empty(max_program_tokens=4)
    with pytest.raises(TypeError):
        KarelAST.empty(max_program_tokens=True)


def round_actions(tree: KarelAST, names: tuple[str, ...]) -> np.ndarray:
    """Place one named expansion at each current hole in preorder."""
    positions = np.flatnonzero(tree.features().is_hole)
    assert len(positions) == len(names)
    actions = np.zeros(tree.max_nodes, np.int32)
    actions[positions] = [ACTION_ID[name] for name in names]
    return actions


def test_parallel_round_expands_old_holes_only_and_preserves_positions() -> None:
    tree = KarelAST.empty(32, 16, 32)
    for names in (("Program",), ("ConsNonEmpty",), ("IFELSE", "Cons")):
        tree = tree.expand_round(round_actions(tree, names))
    # The sibling tail expanded in the same round as IFELSE, even though the
    # latter inserted new children before it in preorder. None were expanded yet.
    before = tree.features()
    positions = np.flatnonzero(before.is_hole)
    assert len(positions) == 5  # condition, then, else, next statement, next tail
    np.testing.assert_array_equal(
        before.field[positions], [Field.CONDITION, Field.THEN, Field.ELSE, Field.HEAD, Field.TAIL]
    )
    tree = tree.expand_round(round_actions(tree, ("Test", "ConsNonEmpty", "ConsNonEmpty", "turnLeft", "End")))
    tree = tree.expand_round(round_actions(tree, ("frontIsClear", "move", "End", "turnRight", "End")))
    assert tree.complete
    assert tree.tokens() == (
        "DEF",
        "run",
        "m(",
        "IFELSE",
        "c(",
        "frontIsClear",
        "c)",
        "i(",
        "move",
        "i)",
        "ELSE",
        "e(",
        "turnRight",
        "e)",
        "turnLeft",
        "m)",
    )
    assert before.is_hole.sum() == 5  # immutable pre-round snapshot
    assert len(tree.nodes) > 5  # five rounds resolved more than five nodes
    np.testing.assert_array_equal(tree.parallel_action_mask(), False)
    assert tree.expand_round(np.zeros(tree.max_nodes, np.int32)) == tree


@pytest.mark.parametrize("nodes,depth,tokens", [(4, 2, 5), (16, 8, 11), (16, 8, None)])
def test_parallel_samples_complete_within_all_budgets(nodes: int, depth: int, tokens: int | None) -> None:
    rng = np.random.default_rng(71)
    for _ in range(3):
        tree = KarelAST.empty(nodes, depth, tokens)
        decisions = 0
        for step in range(min(nodes, depth + 1)):
            features = tree.features()
            masks = features.action_mask
            np.testing.assert_array_equal(masks.any(axis=-1), features.is_hole & features.node_mask)
            actions = np.zeros(nodes, np.int32)
            positions = np.flatnonzero(features.is_hole)
            # Every hole created in the previous round is at this depth level.
            np.testing.assert_array_equal(features.depth[positions], step)
            for position in positions:
                actions[position] = rng.choice(np.flatnonzero(masks[position]))
            decisions += len(positions)
            tree = tree.expand_round(actions)
            assert len(tree.nodes) <= nodes
            if tree.complete:
                break
        assert tree.complete
        assert decisions == len(tree.nodes)
        assert tokens is None or len(tree.tokens()) <= tokens
        _parse(tree.tokens())


@pytest.mark.parametrize("nodes,tokens", [(8, 128), (128, 9)])
def test_parallel_masks_reserve_joint_budget_not_independent_full_budgets(nodes: int, tokens: int) -> None:
    tree = KarelAST.empty(nodes, 8, tokens)
    tree = tree.expand(ACTION_ID["Program"]).expand(ACTION_ID["ConsNonEmpty"])
    # A repeat fits alone, as does extending the tail, but together they need
    # ten nodes. The per-hole masks must exclude that simultaneous combination.
    assert tree.allowed_actions()[ACTION_ID["REPEAT"]]
    masks = tree.parallel_action_mask()
    assert not (masks[2, ACTION_ID["REPEAT"]] and masks[3, ACTION_ID["Cons"]])
    for left, right in product(np.flatnonzero(masks[2]), np.flatnonzero(masks[3])):
        actions = np.zeros(nodes, np.int32)
        actions[2:4] = left, right
        expanded = tree.expand_round(actions)
        while not expanded.complete:
            expanded = expanded.expand(int(np.flatnonzero(expanded.allowed_actions())[0]))
        assert len(expanded.nodes) <= nodes
        assert len(expanded.tokens()) <= tokens
        _parse(expanded.tokens())


@pytest.mark.parametrize("bad", [0, -1, len(AST_ACTIONS), ACTION_ID["turnLeft"]])
def test_invalid_parallel_actions_leave_original_tree_unchanged(bad: int) -> None:
    tree = KarelAST.empty(8, 4)
    actions = np.zeros(8, np.int32)
    actions[0] = bad
    with pytest.raises(ValueError, match="Invalid AST action"):
        tree.expand_round(actions)
    assert tree.frontier == 0 and len(tree.nodes) == 1
    actions[0] = ACTION_ID["Program"]
    actions[7] = ACTION_ID["End"]
    with pytest.raises(ValueError, match="Non-hole"):
        tree.expand_round(actions)
    with pytest.raises(AssertionError):
        tree.expand_round(np.zeros(7, np.int32))
    with pytest.raises(AssertionError):
        tree.expand_round(np.zeros(8, np.float32))


@pytest.mark.parametrize("leading_shape", [(), (3,), (2, 3)])
def test_is_hole_is_derived_for_numpy_and_jitted_jax(leading_shape: tuple[int, ...]) -> None:
    tree = KarelAST.empty(32, 16)
    # A resolved repeat count (including literal zero) coexists with open holes
    # and resolved constructor nodes in the same partial tree.
    for name in ("Program", "ConsNonEmpty", "REPEAT", "R=0"):
        tree = tree.expand(ACTION_ID[name])
    features = tree.features()
    expected = np.zeros(32, np.bool_)
    for position, index in enumerate(tree.preorder()):
        expected[position] = tree.nodes[index].is_hole
    # Padding remains inactive even if its type/value looks like a hole.
    features = features._replace(node_type=np.where(features.node_mask, features.node_type, 999).astype(np.int32))
    features = ASTFeatures(*(np.broadcast_to(array, (*leading_shape, *array.shape)) for array in features))
    assert isinstance(features.is_hole, np.ndarray)
    np.testing.assert_array_equal(features.is_hole, np.broadcast_to(expected, (*leading_shape, 32)))

    @jax.jit
    def infer(arrays: ASTFeatures) -> jax.Array:
        return arrays.is_hole

    actual = infer(jax.tree.map(jnp.asarray, features))
    assert actual.dtype == jnp.bool_
    np.testing.assert_array_equal(actual, features.is_hole)
    # The property follows value changes without a second mask to synchronize.
    count_position = int(np.flatnonzero(tree.features().value)[0])
    values = np.array(features.value)
    values[..., count_position] = 0
    assert features._replace(value=values).is_hole[..., count_position].all()
    assert not features.is_hole[..., count_position].any()


def test_resolved_predicate_is_not_a_hole() -> None:
    tree = KarelAST.empty(32, 16)
    for name in ("Program", "ConsNonEmpty", "IF", "Test"):
        tree = tree.expand(ACTION_ID[name])
    before = tree.features()
    position = int(before.is_hole.argmax())
    after = tree.expand(ACTION_ID["frontIsClear"]).features()
    assert before.node_type[position] == after.node_type[position]
    assert before.is_hole[position] and not after.is_hole[position]
