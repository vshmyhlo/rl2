"""Optional parity checks against Mctx; install mctx to run this reference suite."""

from dataclasses import fields
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pgx = pytest.importorskip("pgx")
mctx = pytest.importorskip("mctx")
from mctx._src import search as mctx_search

from rl2.alphazero import train as az
from rl2.alphazero.train import PolicyValueFn, backward, puct_policy

MCTX_TREE_FIELDS = {
    field.name: "embeddings" if field.name == "env_states" else field.name for field in fields(az.SearchTree)
}


def test_backup_preserves_finished_rows_and_matches_mctx() -> None:
    env_states = jax.vmap(pgx.make("tic_tac_toe").init)(jax.random.split(jax.random.PRNGKey(0), 3))  # Leaves: (B, ...).
    root = mctx.RootFnOutput(prior_logits=jnp.zeros((3, 2)), value=jnp.zeros(3), embedding=env_states)
    tree = mctx_search.instantiate_tree_from_root(root, 3, jnp.zeros((3, 2), bool), None)
    tree = tree.replace(
        parents=jnp.tile(jnp.array([-1, 0, 1, 2]), (3, 1)),
        action_from_parent=jnp.tile(jnp.array([-1, 0, 1, 0]), (3, 1)),
        node_visits=jnp.tile(jnp.array([4, 3, 2, 1]), (3, 1)),
        node_values=jnp.tile(jnp.array([0.2, -0.3, 0.4, -0.5]), (3, 1)),
        children_rewards=jnp.full((3, 4, 2), 0.1),
        children_discounts=jnp.full((3, 4, 2), -1.0).at[2, 2, 0].set(0.0),
    )
    # One game is already at the root; the others back up one and three edges.
    leaves = jnp.array([0, 1, 3], jnp.int32)
    expected = jax.jit(mctx_search.backward)(tree, leaves)
    local_tree = az.SearchTree(
        **{name: getattr(tree, reference_name) for name, reference_name in MCTX_TREE_FIELDS.items()}
    )
    actual = jax.jit(backward)(local_tree, leaves)
    for name, reference_name in MCTX_TREE_FIELDS.items():
        before = getattr(tree, reference_name)  # Leaves: (B, ...).
        got = getattr(actual, name)  # Leaves: (B, ...).
        want = getattr(expected, reference_name)  # Leaves: (B, ...).
        chex.assert_trees_all_close(got, want, rtol=1e-6, atol=1e-7)

        def first_game(leaf: jax.Array) -> jax.Array:  # (B, ...) -> (...)
            return leaf[0]  # (...)

        chex.assert_trees_all_equal(jax.tree.map(first_game, got), jax.tree.map(first_game, before))
    np.testing.assert_array_equal(actual.node_visits[:, 0], [4, 5, 5])


@pytest.mark.parametrize(
    ("batch_size", "simulations", "temperature"),
    [pytest.param(1, 1, 0.0, id="single-expansion-greedy"), pytest.param(2, 8, 1.0, id="batched-mixed-depth-sampling")],
)
def test_puct_matches_mctx_policy(batch_size: int, simulations: int, temperature: float) -> None:
    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.zeros((observation.shape[0], 4)), jnp.zeros(observation.shape[0])

    def recurrent(
        model: PolicyValueFn,
        key: jax.Array,  # (2,)
        actions: jax.Array,  # (B,)
        env_states: pgx.State,  # Leaves: (B, ...).
    ) -> tuple[mctx.RecurrentFnOutput, pgx.State]:
        depth = env_states._step_count  # (B,)
        limit = jnp.array([2, 4], jnp.int32)[:batch_size]  # (B,)
        done = depth >= limit  # (B,)
        next_depth = jnp.where(done, depth, depth + 1)  # (B,)
        terminated = next_depth >= limit  # (B,)
        return mctx.RecurrentFnOutput(
            reward=jnp.where(terminated & ~done, 1.0, 0.0),
            discount=jnp.where(terminated, 0.0, -1.0),
            prior_logits=jnp.tile(jnp.array([2.0, 1.0, 0.0, -1.0]), (batch_size, 1)),
            value=jnp.where(terminated, 0.0, 0.1 * actions),
        ), env_states.replace(_step_count=next_depth, terminated=terminated)  # Leaves: (B, ...).

    env_states = jax.vmap(pgx.make("tic_tac_toe").init)(
        jax.random.split(jax.random.PRNGKey(0), batch_size)
    )  # Leaves: (B, ...).
    root = mctx.RootFnOutput(
        prior_logits=jnp.array([[4.0, 0.0, -1.0, -2.0], [0.0, 0.0, 0.0, 0.0]])[:batch_size],
        value=jnp.zeros(batch_size),
        embedding=env_states,
    )
    invalid = jnp.array([[False, False, False, True], [False, False, True, False]])[:batch_size]
    key = jax.random.PRNGKey(0)
    kwargs = {
        "recurrent_fn": recurrent,
        "num_simulations": simulations,
        "invalid_actions": invalid,
        "temperature": temperature,
    }
    reference = jax.jit(partial(mctx.muzero_policy, predict, dirichlet_fraction=0.0, **kwargs))(key, root)

    def local_recurrent(
        model: PolicyValueFn,
        actions: jax.Array,  # (B,)
        env_states: pgx.State,  # Leaves: (B, ...).
    ) -> tuple[az.RecurrentFnOutput, pgx.State]:
        output, next_env_states = recurrent(model, key, actions, env_states)  # The reference ignores its RNG key.
        return az.RecurrentFnOutput(
            reward=output.reward, discount=output.discount, prior_logits=output.prior_logits, value=output.value
        ), next_env_states  # Leaves: (B, ...).

    local_root = az.RootFnOutput(prior_logits=root.prior_logits, value=root.value, env_states=env_states)
    actual = jax.jit(partial(puct_policy, predict, **{**kwargs, "recurrent_fn": local_recurrent}))(key, local_root)
    assert actual.action.shape == (batch_size,)
    assert actual.action_weights.shape == (batch_size, 4)
    assert actual.search_tree.children_values.shape == (batch_size, simulations + 1, 4)
    np.testing.assert_array_equal(actual.action, reference.action)
    np.testing.assert_allclose(actual.action_weights, reference.action_weights)
    for name, reference_name in MCTX_TREE_FIELDS.items():
        chex.assert_trees_all_close(
            getattr(actual.search_tree, name), getattr(reference.search_tree, reference_name), rtol=1e-6, atol=1e-7
        )
    np.testing.assert_array_equal(actual.action_weights[invalid], 0.0)
    np.testing.assert_allclose(actual.action_weights.sum(axis=-1), 1.0)
