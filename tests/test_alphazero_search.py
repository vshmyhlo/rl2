from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pgx = pytest.importorskip("pgx")

from rl2.alphazero import train as az
from rl2.alphazero.train import recurrent_step


def test_recurrent_step_finished_nodes_do_not_repeat_reward_or_bootstrap() -> None:
    env = pgx.make("tic_tac_toe")
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    states = states.replace(
        rewards=jnp.array([[1, -1], [-1, 1]], jnp.float32),
        terminated=jnp.array([True, False]),
        truncated=jnp.array([False, True]),
        # Truncation need not reset the legal mask. PGX still penalizes an
        # illegal action after checking whether the state has already ended.
        legal_action_mask=states.legal_action_mask.at[1, 0].set(False),
    )

    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        batch_size = observation.shape[0]
        return jnp.zeros((batch_size, 9), jnp.float32), jnp.full(batch_size, 0.5, jnp.float32)

    step = jax.jit(partial(recurrent_step, predict, env=env))
    output, next_states = step(jnp.zeros(2, jnp.int32), states)
    np.testing.assert_array_equal(output.reward, [0, 0])
    np.testing.assert_array_equal(output.discount, [0, 0])
    np.testing.assert_array_equal(output.value, [0, 0])
    np.testing.assert_array_equal(next_states.terminated, states.terminated)
    np.testing.assert_array_equal(next_states.truncated, states.truncated)
    np.testing.assert_array_equal(next_states.legal_action_mask, states.legal_action_mask)
    assert np.isfinite(jax.nn.softmax(output.prior_logits)).all()


@pytest.mark.parametrize(
    ("logits_shape", "value_shape"),
    [
        pytest.param((2, 9), (), id="scalar-value"),
        pytest.param((2, 9), (1,), id="broadcast-value"),
        pytest.param((2, 9), (2, 1), id="column-value"),
        pytest.param((1, 9), (2,), id="broadcast-policy"),
        pytest.param((2, 1), (2,), id="wrong-action-count"),
    ],
)
def test_recurrent_step_rejects_malformed_predictions(
    logits_shape: tuple[int, ...], value_shape: tuple[int, ...]
) -> None:
    env = pgx.make("tic_tac_toe")
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))

    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.zeros(logits_shape, jnp.float32), jnp.zeros(value_shape, jnp.float32)

    step = jax.jit(partial(recurrent_step, predict, env=env))
    with pytest.raises(AssertionError, match="shape"):
        step(jnp.zeros(2, jnp.int32), states)


@pytest.mark.parametrize(
    ("simulations", "temperature"),
    [
        pytest.param(0, 0.0, id="no-expansions"),
        pytest.param(4, 0.0, id="absorbing-terminal-branches"),
        pytest.param(4, 1e6, id="high-temperature-preserves-zero-probabilities"),
    ],
)
def test_local_puct_terminal_rewards_and_nested_state_shapes(simulations: int, temperature: float) -> None:
    # The first game has one legal terminal move; the second is already finished.
    env_states = jax.vmap(pgx.make("tic_tac_toe").init)(jax.random.split(jax.random.PRNGKey(0), 2))  # Leaves: (B, ...).
    env_states = env_states.replace(terminated=jnp.array([False, True]))  # Leaves: (B, ...).
    root = az.RootFnOutput(
        prior_logits=jnp.zeros((2, 2)),  # (B, A)
        value=jnp.array([0.25, 0.0]),  # (B,)
        env_states=env_states,  # Leaves: (B, ...).
    )
    invalid = jnp.array([[False, True], [True, True]])  # (B, A)

    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:  # (B, *O)
        return jnp.zeros((observation.shape[0], 2)), jnp.zeros(observation.shape[0])  # (B, A), (B,)

    def recurrent(
        model: az.PolicyValueFn,
        actions: jax.Array,  # (B,)
        env_states: pgx.State,  # Leaves: (B, ...)
    ) -> tuple[az.RecurrentFnOutput, pgx.State]:
        logits, value = model(env_states.observation)  # (B, A), (B,)
        return az.RecurrentFnOutput(
            reward=(~env_states.terminated).astype(jnp.float32),  # (B,)
            discount=jnp.zeros(2),  # (B,)
            prior_logits=logits,  # (B, A)
            value=value,  # (B,)
        ), env_states.replace(terminated=jnp.ones(2, bool))  # Leaves: (B, ...).

    search = jax.jit(
        partial(
            az.puct_policy,
            predict,
            recurrent_fn=recurrent,
            num_simulations=simulations,
            invalid_actions=invalid,
            temperature=temperature,
        )
    )
    result = search(jax.random.PRNGKey(0), root)
    tree = result.search_tree
    assert result.action.shape == (2,)
    assert result.action_weights.shape == (2, 2)
    assert tree.children_visits.shape == (2, simulations + 1, 2)
    assert tree.env_states.terminated.shape == (2, simulations + 1)
    assert tree.env_states.observation.shape == (2, simulations + 1, 3, 3, 2)
    np.testing.assert_array_equal(tree.node_visits[:, 0], simulations + 1)
    np.testing.assert_allclose(tree.node_values[:, 0], [(0.25 + simulations) / (simulations + 1), 0.0])
    np.testing.assert_array_equal(tree.children_visits[:, 0].sum(axis=-1), simulations)
    if simulations:
        np.testing.assert_array_equal(result.action, [0, 0])
        np.testing.assert_array_equal(result.action_weights, [[1, 0], [1, 0]])
        np.testing.assert_array_equal(tree.children_rewards[:, 0, 0], [1, 0])
        np.testing.assert_array_equal(tree.children_rewards[:, 1:], 0)
    else:
        # With no visits, use legal moves; finished rows still have a defined fallback.
        assert result.action[0] == 0
        np.testing.assert_array_equal(result.action_weights, [[1, 0], [0.5, 0.5]])
        np.testing.assert_array_equal(tree.parents, -1)


def test_search_expands_one_new_node_per_simulation() -> None:
    # One action forces the deepest possible tree: a chain through every allocated node.
    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:  # (B, *O)
        return jnp.zeros((observation.shape[0], 1)), jnp.zeros(observation.shape[0])  # (B, A), (B,)

    def recurrent(
        model: az.PolicyValueFn,
        actions: jax.Array,  # (B,)
        env_states: pgx.State,  # Leaves: (B, ...).
    ) -> tuple[az.RecurrentFnOutput, pgx.State]:
        next_env_states = env_states.replace(_step_count=env_states._step_count + 1)  # Leaves: (B, ...).
        logits, value = model(next_env_states.observation)  # (B, A), (B,)
        return az.RecurrentFnOutput(
            reward=jnp.zeros_like(value),  # (B,)
            discount=jnp.ones_like(value),  # (B,)
            prior_logits=logits,  # (B, A)
            value=value,  # (B,)
        ), next_env_states  # Leaves: (B, ...).

    env_states = jax.vmap(pgx.make("tic_tac_toe").init)(jax.random.split(jax.random.PRNGKey(0), 1))  # Leaves: (B, ...).
    root = az.RootFnOutput(
        prior_logits=jnp.zeros((1, 1)),  # (B, A)
        value=jnp.zeros(1),  # (B,)
        env_states=env_states,  # Leaves: (B, ...).
    )
    search = jax.jit(
        partial(
            az.puct_policy,
            predict,
            recurrent_fn=recurrent,
            num_simulations=4,
            invalid_actions=jnp.zeros((1, 1), bool),  # (B, A)
            temperature=0.0,
        )
    )
    result = search(jax.random.PRNGKey(0), root)
    tree = result.search_tree
    assert tree.children_index.shape == (1, 5, 1)
    assert tree.env_states._step_count.shape == (1, 5)
    np.testing.assert_array_equal(tree.parents, [[-1, 0, 1, 2, 3]])
    np.testing.assert_array_equal(tree.children_index, [[[1], [2], [3], [4], [-1]]])
    np.testing.assert_array_equal(tree.env_states._step_count, [[0, 1, 2, 3, 4]])
    np.testing.assert_array_equal(tree.node_visits, [[5, 4, 3, 2, 1]])
    np.testing.assert_array_equal(result.action_weights, [[1.0]])
