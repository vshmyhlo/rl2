from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pgx = pytest.importorskip("pgx")
pytest.importorskip("mctx")

from rl2.alphazero.search import recurrent_step


def test_recurrent_step_finished_nodes_do_not_repeat_reward_or_bootstrap() -> None:
    env = pgx.make("tic_tac_toe")
    states = jax.vmap(env.init)(jax.random.split(jax.random.PRNGKey(0), 2))
    states = states.replace(
        rewards=jnp.array([[1, -1], [-1, 1]], jnp.float32),
        terminated=jnp.array([True, False]),
        truncated=jnp.array([False, True]),
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
    assert np.isfinite(jax.nn.softmax(output.prior_logits)).all()
