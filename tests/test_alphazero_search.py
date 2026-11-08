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


@pytest.mark.parametrize("simulations", [0, 4], ids=["no-expansions", "absorbing-terminal-branches"])
def test_local_puct_terminal_rewards_and_nested_state_shapes(simulations: int) -> None:
    # The first game has one legal terminal move; the second is already finished.
    root = az.RootFnOutput(
        prior_logits=jnp.zeros((2, 2)),  # (B, A)
        value=jnp.array([0.25, 0.0]),  # (B,)
        embedding={
            "done": jnp.array([False, True]),  # (B,)
            "board": jnp.zeros((2, 2, 2), jnp.int32),  # (B, H, W)
        },
    )
    invalid = jnp.array([[False, True], [True, True]])  # (B, A)

    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:  # (B, H, W)
        return jnp.zeros((observation.shape[0], 2)), jnp.zeros(observation.shape[0])  # (B, A), (B,)

    def recurrent(
        model: az.PolicyValueFn,
        key: jax.Array,  # (2,)
        actions: jax.Array,  # (B,)
        state: dict[str, jax.Array],  # Leaves: (B, ...)
    ) -> tuple[az.RecurrentFnOutput, dict[str, jax.Array]]:
        logits, value = model(state["board"])  # (B, A), (B,)
        return az.RecurrentFnOutput(
            reward=(~state["done"]).astype(jnp.float32),  # (B,)
            discount=jnp.zeros(2),  # (B,)
            prior_logits=logits,  # (B, A)
            value=value,  # (B,)
        ), {"done": jnp.ones(2, bool), "board": state["board"]}  # (B,), (B, H, W)

    search = jax.jit(
        partial(
            az.puct_policy,
            predict,
            recurrent_fn=recurrent,
            num_simulations=simulations,
            invalid_actions=invalid,
            temperature=0.0,
        )
    )
    result = search(jax.random.PRNGKey(0), root)
    tree = result.search_tree
    assert result.action.shape == (2,)
    assert result.action_weights.shape == (2, 2)
    assert tree.children_visits.shape == (2, simulations + 1, 2)
    assert tree.embeddings["done"].shape == (2, simulations + 1)
    assert tree.embeddings["board"].shape == (2, simulations + 1, 2, 2)
    np.testing.assert_array_equal(tree.node_visits[:, 0], simulations + 1)
    np.testing.assert_allclose(tree.node_values[:, 0], [(0.25 + simulations) / (simulations + 1), 0.0])
    np.testing.assert_array_equal(tree.children_visits[:, 0].sum(axis=-1), simulations)
    if simulations:
        np.testing.assert_array_equal(result.action, [0, 0])
        np.testing.assert_array_equal(result.action_weights, [[1, 0], [1, 0]])
        np.testing.assert_array_equal(tree.children_rewards[:, 0, 0], [1, 0])
        np.testing.assert_array_equal(tree.children_rewards[:, 1:], 0)
    else:
        # Match Mctx's uniform fallback when there are no visit counts.
        np.testing.assert_array_equal(result.action_weights, 0.5)
        np.testing.assert_array_equal(tree.parents, -1)
