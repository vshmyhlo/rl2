from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pgx = pytest.importorskip("pgx")

from rl2.alphazero import train as az
from rl2.alphazero.train import recurrent_step


@pytest.mark.parametrize("noise_fraction", [0.0, 1.0], ids=["network-priors", "noise-priors"])
def test_puct_prepares_root_predictions(noise_fraction: float) -> None:
    env = pgx.make("tic_tac_toe")
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 3))  # Leaves: (B, ...)
    states = states.replace(
        terminated=jnp.array([False, True, False]),  # (B,)
        truncated=jnp.array([False, False, True]),  # (B,)
        legal_action_mask=jnp.tile(
            jnp.array([True, True, False, False, False, False, False, False, False]), (3, 1)
        ),  # (B, A)
    )

    def predict(observations: jax.Array) -> tuple[jax.Array, jax.Array]:  # (*O,)
        return (
            jnp.array([0.0, -200.0, *([1000.0] * 7)]),  # (A,)
            jnp.float32(0.75),  # ()
        )

    search = partial(
        az.puct_policy,
        predict,
        recurrent_fn=partial(recurrent_step, env=env),
        num_simulations=0,
        temperature=0.0,
        dirichlet_alpha=0.3,
        dirichlet_fraction=noise_fraction,
    )
    result = jax.jit(jax.vmap(search))(jax.random.split(jax.random.PRNGKey(1), 3), states)
    tree = result.search_tree
    assert tree.node_prior_logits.shape == (3, 1, 9)
    assert tree.node_values.shape == (3, 1)
    np.testing.assert_array_equal(tree.node_values[:, 0], [0.75, 0.0, 0.0])
    np.testing.assert_array_equal(tree.node_env_states.legal_action_mask[:, 0], states.legal_action_mask)
    probabilities = jax.nn.softmax(tree.node_prior_logits[:, 0])  # (B, A)
    np.testing.assert_array_equal(probabilities[:, 2:], 0.0)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
    if noise_fraction == 0.0:
        # Preserve log-odds even when their probabilities underflow to zero.
        np.testing.assert_array_equal(tree.node_prior_logits[:, 0, :2], [[0.0, -200.0]] * 3)
        np.testing.assert_array_equal(probabilities[:, :2], [[1.0, 0.0]] * 3)
    else:
        assert not np.allclose(probabilities[:, :2], 0.5)
        assert not np.allclose(probabilities[0], probabilities[1])


@pytest.mark.parametrize(
    ("logits_shape", "value_shape"),
    [
        pytest.param((9,), (1,), id="batched-value"),
        pytest.param((1, 9), (), id="batched-policy"),
        pytest.param((1,), (), id="wrong-action-count"),
    ],
)
def test_puct_rejects_malformed_root_predictions(logits_shape: tuple[int, ...], value_shape: tuple[int, ...]) -> None:
    env = pgx.make("tic_tac_toe")
    state = env.init(jax.random.PRNGKey(0))  # Unbatched array leaves.

    def predict(observations: jax.Array) -> tuple[jax.Array, jax.Array]:  # (*O,)
        return jnp.zeros(logits_shape), jnp.zeros(value_shape)  # Intentionally invalid shapes.

    with pytest.raises(AssertionError, match="shape"):
        az.puct_policy(
            predict,
            jax.random.PRNGKey(0),
            state,
            partial(recurrent_step, env=env),
            0,
            temperature=0.0,
        )


def test_recurrent_step_finished_nodes_do_not_repeat_reward_or_bootstrap() -> None:
    env = pgx.make("tic_tac_toe")
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))  # Leaves: (B, ...)
    states = states.replace(
        rewards=jnp.array([[1, -1], [-1, 1]], jnp.float32),
        terminated=jnp.array([True, False]),
        truncated=jnp.array([False, True]),
        # Truncation need not reset the legal mask. PGX still penalizes an
        # illegal action after checking whether the state has already ended.
        legal_action_mask=states.legal_action_mask.at[1, 0].set(False),
    )

    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.zeros(9, jnp.float32), jnp.float32(0.5)  # (A,), ()

    step = jax.jit(jax.vmap(partial(recurrent_step, predict, env=env)))
    output = step(jnp.zeros(2, jnp.int32), states)
    next_states = output.env_state  # Leaves: (B, ...)
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
        pytest.param((9,), (1,), id="batched-value"),
        pytest.param((1, 9), (), id="batched-policy"),
        pytest.param((1,), (), id="wrong-action-count"),
    ],
)
def test_recurrent_step_rejects_malformed_predictions(
    logits_shape: tuple[int, ...], value_shape: tuple[int, ...]
) -> None:
    env = pgx.make("tic_tac_toe")
    state = env.init(jax.random.PRNGKey(0))

    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.zeros(logits_shape, jnp.float32), jnp.zeros(value_shape, jnp.float32)

    step = jax.jit(partial(recurrent_step, predict, env=env))
    with pytest.raises(AssertionError, match="shape"):
        step(jnp.int32(0), state)


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
    env_states = env_states.replace(
        terminated=jnp.array([False, True]),  # (B,)
        legal_action_mask=jnp.array([[True, False], [False, False]]),  # (B, A)
    )  # Leaves: (B, ...).

    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:  # (*O,)
        return jnp.zeros(2), jnp.float32(0.25)  # (A,), ()

    def recurrent(
        model: az.PolicyValueFn,
        action: jax.Array,  # ()
        env_state: pgx.State,  # Leaves: (...)
    ) -> az.SearchTransition:
        logits, value = model(env_state.observation)  # (A,), ()
        return az.SearchTransition(
            reward=(~env_state.terminated).astype(jnp.float32),  # ()
            discount=jnp.zeros_like(value),  # ()
            prior_logits=logits,  # (A,)
            value=value,  # ()
            env_state=env_state.replace(terminated=jnp.ones_like(env_state.terminated)),  # Leaves: (...)
        )

    search = partial(
        az.puct_policy,
        predict,
        recurrent_fn=recurrent,
        num_simulations=simulations,
        temperature=temperature,
    )
    result = jax.jit(jax.vmap(search))(jax.random.split(jax.random.PRNGKey(0), 2), env_states)
    tree = result.search_tree
    assert result.action.shape == (2,)
    assert result.action_weights.shape == (2, 2)
    assert tree.node_action_visits.shape == (2, simulations + 1, 2)
    assert tree.node_env_states.terminated.shape == (2, simulations + 1)
    assert tree.node_env_states.observation.shape == (2, simulations + 1, 3, 3, 2)
    np.testing.assert_array_equal(tree.node_visits[:, 0], simulations + 1)
    np.testing.assert_allclose(tree.node_values[:, 0], [(0.25 + simulations) / (simulations + 1), 0.0])
    np.testing.assert_array_equal(tree.node_action_visits[:, 0].sum(axis=-1), simulations)
    if simulations:
        np.testing.assert_array_equal(result.action, [0, 0])
        np.testing.assert_array_equal(result.action_weights, [[1, 0], [1, 0]])
        np.testing.assert_array_equal(tree.node_action_rewards[:, 0, 0], [1, 0])
        np.testing.assert_array_equal(tree.node_action_rewards[:, 1:], 0)
    else:
        # With no visits, use legal moves; finished rows still have a defined fallback.
        assert result.action[0] == 0
        np.testing.assert_array_equal(result.action_weights, [[1, 0], [0.5, 0.5]])
        np.testing.assert_array_equal(tree.parent_indices, -1)


def test_search_expands_one_new_node_per_simulation() -> None:
    # One action forces the deepest possible tree: a chain through every allocated node.
    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:  # (*O,)
        return jnp.zeros(1), jnp.float32(0.0)  # (A,), ()

    def recurrent(
        model: az.PolicyValueFn,
        action: jax.Array,  # ()
        env_state: pgx.State,  # Leaves: (...).
    ) -> az.SearchTransition:
        next_env_state = env_state.replace(_step_count=env_state._step_count + 1)  # Leaves: (...).
        logits, value = model(next_env_state.observation)  # (A,), ()
        return az.SearchTransition(
            reward=jnp.zeros_like(value),  # ()
            discount=jnp.ones_like(value),  # ()
            prior_logits=logits,  # (A,)
            value=value,  # ()
            env_state=next_env_state,  # Leaves: (...)
        )

    env_state = pgx.make("tic_tac_toe").init(jax.random.PRNGKey(0))  # Leaves: (...).
    env_state = env_state.replace(legal_action_mask=jnp.ones(1, bool))  # Leaves: (...).
    search = jax.jit(
        partial(
            az.puct_policy,
            predict,
            recurrent_fn=recurrent,
            num_simulations=4,
            temperature=0.0,
        )
    )
    result = search(jax.random.PRNGKey(0), env_state)
    assert result.action.shape == ()
    assert result.action_weights.shape == (1,)
    tree = result.search_tree
    assert tree.children_indices.shape == (5, 1)
    assert tree.node_env_states._step_count.shape == (5,)
    np.testing.assert_array_equal(tree.parent_indices, [-1, 0, 1, 2, 3])
    np.testing.assert_array_equal(tree.children_indices, [[1], [2], [3], [4], [-1]])
    np.testing.assert_array_equal(tree.node_env_states._step_count, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(tree.node_visits, [5, 4, 3, 2, 1])
    np.testing.assert_array_equal(result.action_weights, [1.0])
