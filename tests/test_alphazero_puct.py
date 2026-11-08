"""Optional parity checks against Mctx; install mctx to run this reference suite."""

from dataclasses import fields
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pgx")
mctx = pytest.importorskip("mctx")
from mctx._src import search as mctx_search

from rl2.alphazero import train as az
from rl2.alphazero.train import PolicyValueFn, _backward, puct_policy


def test_backup_preserves_finished_rows_and_matches_mctx() -> None:
    root = mctx.RootFnOutput(prior_logits=jnp.zeros((3, 2)), value=jnp.zeros(3), embedding=jnp.zeros((3, 1)))
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
    local_tree = az.SearchTree(**{field.name: getattr(tree, field.name) for field in fields(az.SearchTree)})
    actual = jax.jit(_backward)(local_tree, leaves)
    for field in fields(az.SearchTree):
        before, got, want = (getattr(item, field.name) for item in (tree, actual, expected))
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(got[0], before[0])
    np.testing.assert_array_equal(actual.node_visits[:, 0], [4, 5, 5])


@pytest.mark.parametrize(
    ("batch_size", "simulations", "temperature"),
    [pytest.param(1, 1, 0.0, id="single-expansion-greedy"), pytest.param(2, 8, 1.0, id="batched-mixed-depth-sampling")],
)
def test_puct_matches_mctx_policy(batch_size: int, simulations: int, temperature: float) -> None:
    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.zeros((observation.shape[0], 4)), jnp.zeros(observation.shape[0])

    def recurrent(
        model: PolicyValueFn, key: jax.Array, actions: jax.Array, state: jax.Array
    ) -> tuple[mctx.RecurrentFnOutput, jax.Array]:
        depth, limit = state[:, 0], state[:, 1]
        done = depth >= limit
        next_depth = jnp.where(done, depth, depth + 1)
        terminated = next_depth >= limit
        return mctx.RecurrentFnOutput(
            reward=jnp.where(terminated & ~done, 1.0, 0.0),
            discount=jnp.where(terminated, 0.0, -1.0),
            prior_logits=jnp.tile(jnp.array([2.0, 1.0, 0.0, -1.0]), (batch_size, 1)),
            value=jnp.where(terminated, 0.0, 0.1 * actions),
        ), jnp.stack((next_depth, limit), axis=-1)

    root = mctx.RootFnOutput(
        prior_logits=jnp.array([[4.0, 0.0, -1.0, -2.0], [0.0, 0.0, 0.0, 0.0]])[:batch_size],
        value=jnp.zeros(batch_size),
        embedding=jnp.array([[0, 2], [0, 4]], jnp.int32)[:batch_size],
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
        key: jax.Array,  # (2,)
        actions: jax.Array,  # (B,)
        state: jax.Array,  # (B, 2)
    ) -> tuple[az.RecurrentFnOutput, jax.Array]:
        output, state = recurrent(model, key, actions, state)
        return az.RecurrentFnOutput(
            reward=output.reward, discount=output.discount, prior_logits=output.prior_logits, value=output.value
        ), state  # Output leaves: (B, ...); state: (B, 2).

    local_root = az.RootFnOutput(prior_logits=root.prior_logits, value=root.value, embedding=root.embedding)
    actual = jax.jit(partial(puct_policy, predict, **{**kwargs, "recurrent_fn": local_recurrent}))(key, local_root)
    assert actual.action.shape == (batch_size,)
    assert actual.action_weights.shape == (batch_size, 4)
    assert actual.search_tree.children_values.shape == (batch_size, simulations + 1, 4)
    np.testing.assert_array_equal(actual.action, reference.action)
    np.testing.assert_allclose(actual.action_weights, reference.action_weights)
    for field in fields(az.SearchTree):
        got, want = (getattr(output.search_tree, field.name) for output in (actual, reference))
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(actual.action_weights[invalid], 0.0)
    np.testing.assert_allclose(actual.action_weights.sum(axis=-1), 1.0)
