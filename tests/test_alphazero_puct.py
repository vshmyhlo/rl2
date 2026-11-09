"""PUCT selection and value-backup behavior."""

from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pgx
import pytest

from rl2.alphazero import train as az


@pytest.mark.parametrize(
    ("node", "legal", "expected"),
    [
        pytest.param(0, [True, False], 0, id="root"),
        pytest.param(1, [False, True], 1, id="child-mask-differs-from-root"),
        pytest.param(1, [False, False], 0, id="finished-node-fallback"),
    ],
)
def test_selection_uses_each_nodes_legal_actions(node: int, legal: list[bool], expected: int) -> None:
    state = (
        pgx.make("tic_tac_toe")
        .init(jax.random.PRNGKey(0))
        .replace(
            legal_action_mask=jnp.array([True, False]),  # (A,)
        )
    )
    root = az.SearchRoot(prior_logits=jnp.array([100.0, 0.0]), value=jnp.float32(0.0), env_state=state)
    tree = az.init_search_tree(root, 1)
    # Action 0 has the dominant score, but the child's own mask forbids it.
    tree = tree.replace(
        node_visits=jnp.ones(2, jnp.int32),  # (N,)
        node_prior_logits=jnp.tile(root.prior_logits, (2, 1)),  # (N, A)
        node_env_states=tree.node_env_states.replace(
            legal_action_mask=tree.node_env_states.legal_action_mask.at[node].set(jnp.array(legal)),  # (N, A)
            terminated=tree.node_env_states.terminated.at[node].set(not any(legal)),  # (N,)
        ),
    )
    action = jax.jit(az.select_action)(jax.random.PRNGKey(0), tree, jnp.int32(node))  # ()
    assert action.shape == ()
    assert action == expected


def test_backup_accumulates_returns_and_preserves_finished_rows() -> None:
    env_states = jax.vmap(pgx.make("tic_tac_toe").init)(jax.random.split(jax.random.PRNGKey(0), 3))  # Leaves: (B, ...)
    env_states = env_states.replace(legal_action_mask=jnp.ones((3, 2), bool))  # Leaves: (B, ...)
    root = az.SearchRoot(
        prior_logits=jnp.zeros((3, 2)),  # (B, A)
        value=jnp.zeros(3),  # (B,)
        env_state=env_states,  # Leaves: (B, ...)
    )
    tree = jax.vmap(partial(az.init_search_tree, num_simulations=3))(root)
    tree = tree.replace(
        parent_indices=jnp.tile(jnp.array([-1, 0, 1, 2]), (3, 1)),  # (B, N)
        parent_actions=jnp.tile(jnp.array([-1, 0, 1, 0]), (3, 1)),  # (B, N)
        node_visits=jnp.tile(jnp.array([4, 3, 2, 1]), (3, 1)),  # (B, N)
        node_values=jnp.tile(jnp.array([0.2, -0.3, 0.4, -0.5]), (3, 1)),  # (B, N)
        node_action_rewards=jnp.full((3, 4, 2), 0.1),  # (B, N, A)
        node_action_discounts=jnp.full((3, 4, 2), -1.0).at[2, 2, 0].set(0.0),  # (B, N, A)
    )
    # Paths of length zero, one, and three; the last stops bootstrapping at node 2.
    leaves = jnp.array([0, 1, 3], jnp.int32)  # (B,)
    actual = jax.jit(jax.vmap(az.backward))(tree, leaves)
    assert actual.node_values.shape == (3, 4)
    assert actual.children_values.shape == (3, 4, 2)
    np.testing.assert_array_equal(actual.node_visits, [[4, 3, 2, 1], [5, 3, 2, 1], [5, 4, 3, 1]])
    np.testing.assert_allclose(
        actual.node_values,
        [[0.2, -0.3, 0.4, -0.5], [0.24, -0.3, 0.4, -0.5], [0.18, -0.225, 0.3, -0.5]],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        actual.children_values,
        [[[0, 0]] * 4, [[-0.3, 0], [0, 0], [0, 0], [0, 0]], [[-0.225, 0], [0, 0.3], [-0.5, 0], [0, 0]]],
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        actual.node_action_visits,
        [[[0, 0]] * 4, [[1, 0], [0, 0], [0, 0], [0, 0]], [[1, 0], [0, 1], [1, 0], [0, 0]]],
    )

    def first_game(leaf: jax.Array) -> jax.Array:  # (B, ...) -> (...)
        return leaf[0]  # (...)

    chex.assert_trees_all_equal(jax.tree.map(first_game, actual), jax.tree.map(first_game, tree))

    def last_game(leaf: jax.Array) -> jax.Array:  # (B, ...) -> (...)
        return leaf[-1]  # (...)

    single = jax.jit(az.backward)(jax.tree.map(last_game, tree), leaves[-1])
    assert single.node_values.shape == (4,)
    assert single.children_values.shape == (4, 2)
    chex.assert_trees_all_close(single, jax.tree.map(last_game, actual), atol=1e-7)


@pytest.mark.parametrize("temperature", [0.0, 1.0], ids=["greedy", "sampling"])
def test_search_prefers_winning_actions_over_initial_priors(temperature: float) -> None:
    env_states = jax.vmap(pgx.make("tic_tac_toe").init)(jax.random.split(jax.random.PRNGKey(0), 2))  # Leaves: (B, ...)
    env_states = env_states.replace(
        current_player=jnp.arange(2, dtype=jnp.int32),  # (B,)
        legal_action_mask=jnp.tile(jnp.array([True, True, False]), (2, 1)),  # (B, A)
    )

    def predict(observations: jax.Array) -> tuple[jax.Array, jax.Array]:  # (*O,)
        return (
            jnp.array([0.0, 1.0, 100.0]),  # (A,)
            jnp.float32(0.0),  # ()
        )

    def recurrent(
        model: az.PolicyValueFn,
        action: jax.Array,  # ()
        state: pgx.State,  # Leaves: (...)
    ) -> az.SearchTransition:
        logits, value = model(state.observation)  # (A,), ()
        reward = jnp.where(action == state.current_player, 1.0, -1.0)  # ()
        return az.SearchTransition(
            reward=jnp.where(state.terminated, 0.0, reward),  # ()
            discount=jnp.zeros_like(value),  # ()
            prior_logits=az.masked_logits(logits, state.legal_action_mask),  # (A,)
            value=value,  # ()
            env_state=state.replace(terminated=jnp.ones_like(state.terminated)),  # Leaves: (...)
        )

    search = partial(az.puct_policy, predict, recurrent_fn=recurrent, num_simulations=16, temperature=temperature)
    result = jax.jit(jax.vmap(search))(jax.random.split(jax.random.PRNGKey(0), 2), env_states)
    assert result.action.shape == (2,)
    assert result.action_weights.shape == (2, 3)
    assert result.search_tree.children_values.shape == (2, 17, 3)
    np.testing.assert_array_equal(jnp.argmax(result.action_weights, axis=-1), [0, 1])
    np.testing.assert_array_equal(result.action_weights[:, 2], 0.0)
    np.testing.assert_allclose(result.action_weights.sum(axis=-1), 1.0)
    np.testing.assert_array_equal(result.search_tree.node_action_visits[:, 0].sum(axis=-1), 16)
    assert env_states.legal_action_mask[jnp.arange(2), result.action].all()
    if temperature == 0.0:
        np.testing.assert_array_equal(result.action, [0, 1])
