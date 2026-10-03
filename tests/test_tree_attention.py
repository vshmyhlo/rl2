import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.tree_attention import MAX_DISTANCE, MAX_RELATIVE_DEPTH, Relation, TreeAttentionBias, tree_relations


def _oracle(parents: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent parent-chain oracle; input parents precede their children."""
    paths: list[list[int]] = []
    for node, parent in enumerate(parents):
        paths.append((paths[parent] if parent >= 0 else []) + [node])
    size = len(parents)
    kind = np.zeros((size, size), np.int32)
    distance = np.zeros_like(kind)
    depth_delta = np.zeros_like(kind)
    for i, first in enumerate(paths):
        for j, second in enumerate(paths):
            common = sum(a == b for a, b in zip(first, second))
            up, down = len(first) - common, len(second) - common
            if i == j:
                relation = Relation.SELF
            elif parents[i] == j:
                relation = Relation.PARENT
            elif parents[j] == i:
                relation = Relation.CHILD
            elif j in first:
                relation = Relation.ANCESTOR
            elif i in second:
                relation = Relation.DESCENDANT
            elif parents[i] == parents[j]:
                relation = Relation.SIBLING
            else:
                relation = Relation.OTHER
            kind[i, j] = relation
            distance[i, j] = min(up + down, MAX_DISTANCE)
            depth_delta[i, j] = np.clip(len(second) - len(first), -MAX_RELATIVE_DEPTH, MAX_RELATIVE_DEPTH)
    return kind, distance, depth_delta + MAX_RELATIVE_DEPTH


@pytest.mark.parametrize("size", [1, 8, 40])
def test_relations_match_parent_chain_oracle(size: int) -> None:
    rng = np.random.default_rng(12)
    # Include a deep chain for saturation, a star, and varied branching trees.
    parent_sets = [[-1] + list(range(size - 1)), [-1] + [0] * (size - 1)]
    parent_sets += [[-1] + [int(rng.integers(i)) for i in range(1, size)] for _ in range(5)]
    for parents in parent_sets:
        order: list[int] = []

        def visit(node: int, parent_ids: list[int], result: list[int]) -> None:
            result.append(node)
            for child, parent in enumerate(parent_ids):
                if parent == node:
                    visit(child, parent_ids, result)

        visit(0, parents, order)
        reordered = [-1 if parents[node] < 0 else order.index(parents[node]) for node in order]
        depths: list[int] = []
        for parent in reordered:
            depths.append(0 if parent < 0 else depths[parent] + 1)
        relations = jax.jit(tree_relations)(jnp.asarray([depths], jnp.int32), jnp.ones((1, size), jnp.bool_))
        for actual, expected in zip(relations, _oracle(reordered)):
            np.testing.assert_array_equal(actual[0, 1:, 1:], expected)
        np.testing.assert_array_equal(relations.kind[0, 0, 1:], Relation.CONTEXT_TO_NODE)
        np.testing.assert_array_equal(relations.kind[0, 1:, 0], Relation.NODE_TO_CONTEXT)
        assert relations.kind[0, 0, 0] == Relation.CONTEXT_SELF


def test_relation_bias_ignores_padding_and_context_geometry() -> None:
    depth = jnp.asarray([[0, 1, 2, 1, 999999], [0, 999999, -999999, 999999, 999999]], jnp.int32)
    mask = jnp.asarray([[True, True, True, True, False], [True, False, False, False, False]])
    relations = tree_relations(depth, mask)
    clean = tree_relations(jnp.where(mask, depth, 0), mask)
    for actual, expected in zip(relations, clean):
        np.testing.assert_array_equal(actual, expected)
    model = TreeAttentionBias(2)
    params = model.init(jax.random.key(1), relations)["params"]
    np.testing.assert_array_equal(model.apply({"params": params}, relations), np.zeros((2, 2, 6, 6)))
    params = {name: jnp.ones_like(value) for name, value in params.items()}
    bias = model.apply({"params": params}, relations)
    expected = np.where(relations.kind == Relation.PADDING, 0, np.where(relations.kind <= Relation.OTHER, 3, 1))
    np.testing.assert_array_equal(bias, np.repeat(expected[:, None], 2, axis=1))


@pytest.mark.parametrize("bad", ["dtype", "mask_dtype", "shape", "rank", "empty"])
def test_relations_validate_inputs(bad: str) -> None:
    depth = jnp.zeros((2, 4), jnp.float32 if bad == "dtype" else jnp.int32)
    mask = jnp.ones((2, 3 if bad == "shape" else 4), jnp.int32 if bad == "mask_dtype" else jnp.bool_)
    if bad == "rank":
        depth, mask = depth[0], mask[0]
    elif bad == "empty":
        depth, mask = depth[:, :0], mask[:, :0]
    with pytest.raises(AssertionError):
        tree_relations(depth, mask)
