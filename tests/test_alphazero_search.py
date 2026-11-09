from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pgx = pytest.importorskip("pgx")

from rl2.alphazero import train as az
from rl2.alphazero.train import transition_step


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
        transition_fn=partial(transition_step, env=env),
        num_simulations=0,
        temperature=0.0,
        dirichlet_alpha=0.3,
        dirichlet_fraction=noise_fraction,
    )
    result = jax.jit(jax.vmap(search))(jax.random.split(jax.random.PRNGKey(1), 3), states)
    tree = result.tree
    assert tree.node_logits.shape == (3, 1, 9)
    assert tree.node_values.shape == (3, 1)
    np.testing.assert_array_equal(tree.node_values[:, 0], [0.75, 0.0, 0.0])
    np.testing.assert_array_equal(tree.node_env_states.legal_action_mask[:, 0], states.legal_action_mask)
    probabilities = jax.nn.softmax(tree.node_logits[:, 0])  # (B, A)
    np.testing.assert_array_equal(probabilities[:, 2:], 0.0)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
    if noise_fraction == 0.0:
        # Preserve log-odds even when their probabilities underflow to zero.
        np.testing.assert_array_equal(tree.node_logits[:, 0, :2], [[0.0, -200.0]] * 3)
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
            partial(transition_step, env=env),
            0,
            temperature=0.0,
        )


def test_transition_step_finished_nodes_do_not_repeat_reward_or_bootstrap() -> None:
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

    step = jax.jit(jax.vmap(partial(transition_step, predict, env=env)))
    output = step(states, jnp.zeros(2, jnp.int32))
    child_states = output.child_env_state  # Leaves: (B, ...)
    np.testing.assert_array_equal(output.reward, [0, 0])
    np.testing.assert_array_equal(output.discount, [0, 0])
    np.testing.assert_array_equal(output.child_value, [0, 0])
    np.testing.assert_array_equal(child_states.terminated, states.terminated)
    np.testing.assert_array_equal(child_states.truncated, states.truncated)
    np.testing.assert_array_equal(child_states.legal_action_mask, states.legal_action_mask)
    assert np.isfinite(jax.nn.softmax(output.child_logits)).all()


@pytest.mark.parametrize(
    ("logits_shape", "value_shape"),
    [
        pytest.param((9,), (1,), id="batched-value"),
        pytest.param((1, 9), (), id="batched-policy"),
        pytest.param((1,), (), id="wrong-action-count"),
    ],
)
def test_transition_step_rejects_malformed_predictions(
    logits_shape: tuple[int, ...], value_shape: tuple[int, ...]
) -> None:
    env = pgx.make("tic_tac_toe")
    state = env.init(jax.random.PRNGKey(0))

    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.zeros(logits_shape, jnp.float32), jnp.zeros(value_shape, jnp.float32)

    step = jax.jit(partial(transition_step, predict, env=env))
    with pytest.raises(AssertionError, match="shape"):
        step(state, jnp.int32(0))


@pytest.mark.parametrize(
    ("simulations", "temperature", "truncated"),
    [
        pytest.param(0, 0.0, False, id="no-expansions"),
        pytest.param(4, 0.0, False, id="terminal-leaf-revisits"),
        pytest.param(4, 1e6, False, id="high-temperature-preserves-zero-probabilities"),
        pytest.param(4, 0.0, True, id="truncated-leaf-revisits"),
    ],
)
def test_local_puct_terminal_rewards_and_nested_state_shapes(
    simulations: int, temperature: float, truncated: bool
) -> None:
    # The first game has one legal terminal move; the second is already finished.
    env_states = jax.vmap(pgx.make("tic_tac_toe").init)(jax.random.split(jax.random.PRNGKey(0), 2))  # Leaves: (B, ...).
    env_states = env_states.replace(
        terminated=jnp.array([False, not truncated]),  # (B,)
        truncated=jnp.array([False, truncated]),  # (B,)
        legal_action_mask=jnp.array([[True, False], [False, False]]),  # (B, A)
    )  # Leaves: (B, ...).

    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:  # (*O,)
        return jnp.zeros(2), jnp.float32(0.25)  # (A,), ()

    def transition_fn(
        model: az.PolicyValueFn,
        env_state: pgx.State,  # Leaves: (...)
        action: jax.Array,  # ()
    ) -> az.SearchTransition:
        logits, value = model(env_state.observation)  # (A,), ()
        return az.SearchTransition(
            reward=(~(env_state.terminated | env_state.truncated)).astype(jnp.float32),  # ()
            discount=jnp.zeros_like(value),  # ()
            child_logits=logits,  # (A,)
            child_value=jnp.float32(0.0),  # ()
            child_env_state=env_state.replace(
                terminated=jnp.bool_(not truncated), truncated=jnp.bool_(truncated)
            ),  # Leaves: (...)
        )

    search = partial(
        az.puct_policy,
        predict,
        transition_fn=transition_fn,
        num_simulations=simulations,
        temperature=temperature,
    )
    result = jax.jit(jax.vmap(search))(jax.random.split(jax.random.PRNGKey(0), 2), env_states)
    tree = result.tree
    assert result.action.shape == (2,)
    assert result.action_weights.shape == (2, 2)
    assert tree.node_action_visits.shape == (2, simulations + 1, 2)
    assert tree.node_env_states.terminated.shape == (2, simulations + 1)
    assert tree.node_env_states.observation.shape == (2, simulations + 1, 3, 3, 2)
    np.testing.assert_array_equal(tree.node_visits[:, 0], simulations + 1)
    np.testing.assert_allclose(tree.node_values[:, 0], [(0.25 + simulations) / (simulations + 1), 0.0])
    np.testing.assert_array_equal(tree.node_action_visits[:, 0].sum(axis=-1), [simulations, 0])
    np.testing.assert_array_equal((tree.node_visits > 0).sum(axis=-1), [2 if simulations else 1, 1])
    np.testing.assert_array_equal(tree.children_indices[:, 1:], -1)
    np.testing.assert_array_equal(tree.node_action_visits[:, 1:], 0)
    np.testing.assert_array_equal(tree.parent_indices[1], -1)
    np.testing.assert_allclose(result.action_weights[1], [0.5, 0.5])
    np.testing.assert_allclose(result.action_weights.sum(axis=-1), 1.0)
    assert (result.action_weights >= 0).all()
    assert 0 <= result.action[1] < 2  # Either fallback action is valid for the finished game.
    if simulations:
        np.testing.assert_array_equal(tree.node_visits[:, 1], [simulations, 0])
        assert result.action[0] == 0
        np.testing.assert_array_equal(result.action_weights[0], [1, 0])
        np.testing.assert_array_equal(tree.node_action_rewards[:, 0, 0], [1, 0])
        np.testing.assert_array_equal(tree.node_action_rewards[:, 1:], 0)
    else:
        # With no visits, use legal moves; finished rows still have a defined fallback.
        assert result.action[0] == 0
        np.testing.assert_array_equal(result.action_weights, [[1, 0], [0.5, 0.5]])
        np.testing.assert_array_equal(tree.parent_indices, -1)


@pytest.mark.parametrize("terminal_depth", [None, 2], ids=["ongoing-chain", "terminal-grandchild"])
def test_search_expands_chain_until_finished(terminal_depth: int | None) -> None:
    # One action forces a chain; terminal revisits must still back up through every ancestor.
    def predict(observation: jax.Array) -> tuple[jax.Array, jax.Array]:  # (*O,)
        return jnp.zeros(1), jnp.float32(0.0)  # (A,), ()

    def transition_fn(
        model: az.PolicyValueFn,
        env_state: pgx.State,  # Leaves: (...).
        action: jax.Array,  # ()
    ) -> az.SearchTransition:
        child_depth = env_state._step_count + 1  # ()
        child_env_state = env_state.replace(
            _step_count=child_depth,  # ()
            terminated=jnp.bool_(False) if terminal_depth is None else child_depth >= terminal_depth,  # ()
        )  # Leaves: (...)
        logits, value = model(child_env_state.observation)  # (A,), ()
        return az.SearchTransition(
            reward=jnp.zeros_like(value),  # ()
            discount=jnp.where(child_env_state.terminated, 0.0, 1.0),  # ()
            child_logits=logits,  # (A,)
            child_value=value,  # ()
            child_env_state=child_env_state,  # Leaves: (...)
        )

    env_state = pgx.make("tic_tac_toe").init(jax.random.PRNGKey(0))  # Leaves: (...).
    env_state = env_state.replace(legal_action_mask=jnp.ones(1, bool))  # Leaves: (...).
    search = jax.jit(
        partial(
            az.puct_policy,
            predict,
            transition_fn=transition_fn,
            num_simulations=4,
            temperature=0.0,
        )
    )
    result = search(jax.random.PRNGKey(0), env_state)
    assert result.action.shape == ()
    assert result.action_weights.shape == (1,)
    tree = result.tree
    assert tree.children_indices.shape == (5, 1)
    assert tree.node_env_states._step_count.shape == (5,)
    depth = 4 if terminal_depth is None else terminal_depth
    np.testing.assert_array_equal(tree.parent_indices, [-1, *range(depth), *([-1] * (4 - depth))])
    np.testing.assert_array_equal(tree.children_indices[:, 0], [*range(1, depth + 1), *([-1] * (5 - depth))])
    np.testing.assert_array_equal(tree.node_env_states._step_count, [*range(depth + 1), *([0] * (4 - depth))])
    np.testing.assert_array_equal(tree.node_visits, [*(5 - i for i in range(depth + 1)), *([0] * (4 - depth))])
    np.testing.assert_array_equal(tree.node_action_visits[:, 0], [*(4 - i for i in range(depth)), *([0] * (5 - depth))])
    np.testing.assert_array_equal(result.action_weights, [1.0])
