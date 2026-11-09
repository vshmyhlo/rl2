"""Classify final self-play states without changing environment termination rules."""

import jax
import jax.numpy as jnp
import pgx
from pgx._src.games.chess import MAX_TERMINATION_STEPS, has_insufficient_pieces
from pgx.chess import State as ChessState


def game_finish_counts(final: pgx.State) -> dict[str, jax.Array]:
    """Return (B,) int32 indicators for mutually exclusive finish reasons.

    final: (B, ...) state leaves after collection, including still-active games.
    Chess conditions use PGX's internal state because its public terminal action
    mask is replaced with all-true padding. First matching terminal reason wins.
    """
    decisive = (final.rewards != 0).any(axis=-1)  # (B,)
    if isinstance(final, ChessState):
        state = final._x  # Leaves: (B, ...)
        no_legal_moves = ~state.legal_action_mask.any(axis=-1)  # (B,)
        # PGX stores the current position's hash at history index zero.
        repetitions = (state.hash_history == state.hash_history[:, :1]).all(axis=-1).sum(axis=-1)  # (B,)
        conditions = {
            "checkmate": no_legal_moves & decisive,  # (B,)
            "stalemate": no_legal_moves & ~decisive,  # (B,)
            "fifty_move_rule": state.halfmove_count >= 100,  # (B,)
            "insufficient_material": jax.vmap(has_insufficient_pieces)(state),  # (B,)
            "threefold_repetition": repetitions >= 3,  # (B,)
            "environment_move_limit": state.step_count >= MAX_TERMINATION_STEPS,  # (B,)
            "other_terminal": jnp.ones_like(final.terminated),  # (B,)
        }
    else:
        conditions = {"win": decisive, "draw": ~decisive}  # Leaves: (B,)

    remaining = final.terminated  # (B,)
    counts: dict[str, jax.Array] = {}  # Leaves: (B,)
    for reason, condition in conditions.items():
        counts[reason] = (remaining & condition).astype(jnp.int32)  # (B,)
        remaining = remaining & ~condition  # (B,)
    counts["truncated"] = (~final.terminated & final.truncated).astype(jnp.int32)  # (B,)
    counts["max_moves"] = (~final.terminated & ~final.truncated).astype(jnp.int32)  # (B,)
    return counts
