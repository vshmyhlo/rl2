import jax
import jax.numpy as jnp
import numpy as np
import pgx
from pgx.experimental.chess import from_fen

from rl2.alphazero.finish_reasons import game_finish_counts


def test_chess_finish_reasons_and_precedence() -> None:
    initial = pgx.make("chess").init(jax.random.PRNGKey(0))  # Leaves: (...)
    mate = from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")  # Leaves: (...)
    stalemate = from_fen("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1")  # Leaves: (...)
    insufficient = from_fen("7k/8/5K2/8/8/8/8/8 b - - 0 1")  # Leaves: (...)
    repeated = initial._x.hash_history.at[1:3].set(initial._x.hash_history[0])  # (513, 2)
    twice = initial._x.hash_history.at[1].set(initial._x.hash_history[0])  # (513, 2)
    cases = [
        ("checkmate", mate),
        ("stalemate", stalemate),
        ("insufficient_material", insufficient),
        (
            "fifty_move_rule",
            initial.replace(terminated=jnp.bool_(True), _x=initial._x._replace(halfmove_count=jnp.int32(100))),
        ),
        (
            "threefold_repetition",
            initial.replace(terminated=jnp.bool_(True), _x=initial._x._replace(hash_history=repeated)),
        ),
        (
            "environment_move_limit",
            initial.replace(terminated=jnp.bool_(True), _x=initial._x._replace(step_count=jnp.int32(512))),
        ),
        # Terminal conditions win over the trainer limit and truncation.
        (
            "checkmate",
            mate.replace(
                truncated=jnp.bool_(True), _x=mate._x._replace(halfmove_count=jnp.int32(100), step_count=jnp.int32(512))
            ),
        ),
        (
            "fifty_move_rule",
            initial.replace(
                terminated=jnp.bool_(True),
                _x=initial._x._replace(halfmove_count=jnp.int32(100), hash_history=repeated, step_count=jnp.int32(512)),
            ),
        ),
        ("truncated", initial.replace(truncated=jnp.bool_(True))),
        # None of the chess thresholds has been reached; collection imposed the limit.
        (
            "max_moves",
            initial.replace(
                _x=initial._x._replace(halfmove_count=jnp.int32(99), hash_history=twice, step_count=jnp.int32(511))
            ),
        ),
        ("other_terminal", initial.replace(terminated=jnp.bool_(True))),
    ]

    def stack(*leaves: jax.Array) -> jax.Array:  # (...) -> (B, ...)
        return jnp.stack(leaves)  # (B, ...)

    final = jax.tree.map(stack, *(state for _, state in cases))  # Leaves: (B, ...)
    # PGX exposes all actions as legal after termination; use the internal mask.
    final = final.replace(legal_action_mask=jnp.ones_like(final.legal_action_mask))
    counts = jax.jit(game_finish_counts)(final)  # Leaves: (B,)
    assert set(counts) == {reason for reason, _ in cases}
    for reason, values in counts.items():
        assert values.shape == (len(cases),)
        assert values.dtype == jnp.int32
        np.testing.assert_array_equal(values, [int(expected == reason) for expected, _ in cases])
    np.testing.assert_array_equal(sum(counts.values()), np.ones(len(cases), dtype=np.int32))
