"""Shared policy masking and PGX transitions for self-play and evaluation."""

from typing import Any

import jax
import jax.numpy as jnp
import mctx
import pgx

from rl2.alphazero.model import PolicyValueNet
from rl2.shape_checker import ShapeChecker

type Parameters = dict[str, Any]


def masked_logits(logits: jax.Array, legal: jax.Array) -> jax.Array:
    sc = ShapeChecker()
    sc.check(logits, "BA", jnp.float32)
    sc.check(legal, "BA", jnp.bool_)
    # Finite masking also keeps terminal nodes with no legal moves well-defined.
    return jnp.where(legal, logits - logits.max(axis=-1, keepdims=True), jnp.finfo(jnp.float32).min)


def recurrent_step(
    params: Parameters,
    key: jax.Array,
    actions: jax.Array,
    states: pgx.State,
    *,
    env: pgx.Env,
    model: PolicyValueNet,
) -> tuple[mctx.RecurrentFnOutput, pgx.State]:
    """Expand real game states; back up the opponent's value with a minus sign."""
    sc = ShapeChecker(K=2, P=2)
    sc.check(key, "K", jnp.uint32)
    sc.check(actions, "B", jnp.int32)
    sc.check(states.current_player, "B", jnp.int32)
    next_states = jax.vmap(env.step)(states, actions)
    logits, value = model.apply({"params": params}, next_states.observation)
    sc.check(next_states.rewards, "BP", jnp.float32)
    sc.check([next_states.terminated, next_states.truncated], "B", jnp.bool_)
    reward = jnp.take_along_axis(next_states.rewards, states.current_player[:, None], axis=1)[:, 0]
    done = next_states.terminated | next_states.truncated
    output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.where(done, 0.0, -1.0),
        prior_logits=masked_logits(logits, next_states.legal_action_mask),
        value=jnp.where(done, 0.0, value),
    )
    sc.check([output.reward, output.discount, output.value], "B", jnp.float32)
    return output, next_states
