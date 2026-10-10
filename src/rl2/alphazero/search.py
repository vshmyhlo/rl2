"""Shared PGX search transitions for self-play and evaluation."""

from typing import Protocol

import jax
import jax.numpy as jnp
import mctx
import pgx

from rl2.alphazero.utils import masked_logits


class PolicyValueFn(Protocol):
    """Bound model mapping observations to policy logits and player-to-move values."""

    def __call__(self, observation: jax.Array, /) -> tuple[jax.Array, jax.Array]:
        """Map (B, H, W, C) observations to (B, A) float32 logits and (B,) float32 values."""
        ...


def recurrent_step(
    model: PolicyValueFn,
    actions: jax.Array,
    states: pgx.State,
    *,
    env: pgx.Env,
) -> tuple[mctx.RecurrentFnOutput, pgx.State]:
    """Expand batched Mctx nodes for deterministic, alternating-turn, two-player games.

    Args:
        model: Bound observation-to-(logits, value) callable, forwarded through
            Mctx's params argument. Values use the player-to-move perspective.
        actions: (B,) int32 action IDs.
        states: (B, ...) array leaves forming a batched PGX state pytree.
        env: Game rules and action/observation encodings.

    Returns:
        (output, next_states), where output contains:
            reward: (B,) float32 immediate reward for the parent state's acting player.
            discount: (B,) float32; -1 for ongoing games, 0 on termination or truncation.
            prior_logits: (B, A) float32 masked child policy logits, A = env.num_actions.
            value: (B,) float32 child player-to-move value, zero on termination or truncation.
        next_states: (B, ...) array leaves forming the child PGX state pytree.
        Backup uses reward + discount * child_value, negating the opponent's value.
    """
    next_states = jax.vmap(env.step)(states, actions)
    logits, value = model(next_states.observation)
    # Check before masking: where would silently broadcast a scalar or (1,) value.
    reward = jnp.take_along_axis(next_states.rewards, states.current_player[:, None], axis=1)[:, 0]
    done = next_states.terminated | next_states.truncated
    output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.where(done, 0.0, -1.0),
        prior_logits=masked_logits(logits, next_states.legal_action_mask),
        value=jnp.where(done, 0.0, value),
    )
    return output, next_states
