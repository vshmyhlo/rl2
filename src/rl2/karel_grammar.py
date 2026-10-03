"""Table-driven predictive Karel grammar masking, entirely in JAX at runtime.

A previous-token table cannot distinguish nested blocks or an IFELSE's pending
ELSE. A small symbol stack supplies that context. Each (top symbol, next token)
lookup consumes the token and replaces the top with its grammar continuation.
Minimum completion costs rule out choices that cannot finish within the token
budget. This enforces nonempty blocks and the interpreter's nesting limit.

No environment state is consulted: legal syntax can still fail at runtime.
Finished/invalid rows use a fixed dummy m) distribution, independent of model
logits, to keep padded minibatches numerically safe. They must not be submitted
to an environment.
"""

from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from rl2.karel import ACTIONS, PREDICATES, TOKEN_TO_ID, TOKENS
from rl2.karel_syntax import MAX_BLOCK_DEPTH

_VOCAB = len(TOKENS)
_CLOSE = ("m)", "w)", "i)", "e)", "r)")
_FIRST = {token: _VOCAB + index for index, token in enumerate(_CLOSE)}
_MORE = {token: _VOCAB + 5 + index for index, token in enumerate(_CLOSE)}
_CONDITION, _TEST, _PREDICATE, _COUNT = range(_VOCAB + 10, _VOCAB + 14)
_SYMBOLS = _VOCAB + 14
_MAX_PUSH = 7


def _tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct immutable lookup data once, outside JAX tracing."""
    legal = np.zeros((_SYMBOLS, _VOCAB), dtype=np.bool_)
    replacement = np.zeros((_SYMBOLS, _VOCAB, _MAX_PUSH), dtype=np.int32)
    lengths = np.zeros((_SYMBOLS, _VOCAB), dtype=np.int32)
    costs = np.ones(_SYMBOLS, dtype=np.int32)
    costs[list(_FIRST.values())] = 2  # At least one primitive, then its closer.
    costs[_CONDITION] = 3

    def rule(symbol: int, token: str, rest: tuple[int, ...] = ()) -> None:
        token_id = TOKEN_TO_ID[token]
        legal[symbol, token_id] = True
        lengths[symbol, token_id] = len(rest)
        replacement[symbol, token_id, : len(rest)] = rest[::-1]  # Stack top at the right.

    for token in TOKENS[1:]:
        rule(TOKEN_TO_ID[token], token)
    rule(_CONDITION, "c(", (_TEST, TOKEN_TO_ID["c)"]))
    for token in PREDICATES:
        rule(_TEST, token)
        rule(_PREDICATE, token)
    rule(_TEST, "not", (TOKEN_TO_ID["c("], _PREDICATE, TOKEN_TO_ID["c)"]))
    for count in range(20):
        rule(_COUNT, f"R={count}")
    for close in _CLOSE:
        tail = _MORE[close]
        for symbol in (_FIRST[close], tail):
            for token in ACTIONS:
                rule(symbol, token, (tail,))
            rule(symbol, "REPEAT", (_COUNT, TOKEN_TO_ID["r("], _FIRST["r)"], tail))
            for token, opening, closing in (("WHILE", "w(", "w)"), ("IF", "i(", "i)")):
                rule(symbol, token, (_CONDITION, TOKEN_TO_ID[opening], _FIRST[closing], tail))
            rule(
                symbol,
                "IFELSE",
                (
                    _CONDITION,
                    TOKEN_TO_ID["i("],
                    _FIRST["i)"],
                    TOKEN_TO_ID["ELSE"],
                    TOKEN_TO_ID["e("],
                    _FIRST["e)"],
                    tail,
                ),
            )
        rule(tail, close)
    replacement_cost = np.sum(
        np.where(np.arange(_MAX_PUSH) < lengths[..., None], costs[replacement], 0), axis=-1, dtype=np.int32
    )
    for array in (legal, replacement, lengths, costs, replacement_cost):
        array.flags.writeable = False
    return legal, replacement, lengths, costs, replacement_cost


_LEGAL, _REPLACEMENT, _LENGTHS, _COSTS, _REPLACEMENT_COST = _tables()
_OPEN_IDS = np.asarray([TOKEN_TO_ID[token] for token in ("m(", "w(", "i(", "e(", "r(")], np.int32)
_CLOSE_IDS = np.asarray([TOKEN_TO_ID[token] for token in _CLOSE], np.int32)
_CONTROL_IDS = np.asarray([TOKEN_TO_ID[token] for token in ("WHILE", "IF", "IFELSE", "REPEAT")], np.int32)


class GrammarState(NamedTuple):
    stack: jax.Array  # [B,capacity], top at size-1.
    size: jax.Array  # [B], zero after m).
    depth: jax.Array  # [B], open blocks including the outer main block.
    minimum: jax.Array  # [B], shortest possible completion in tokens.
    remaining: jax.Array  # [B], remaining generation budget.
    valid: jax.Array  # [B], false after an illegal input token.


def initial_grammar_state(batch_size: int, max_program_tokens: int) -> GrammarState:
    chex.assert_type((batch_size, max_program_tokens), int)
    chex.assert_scalar_positive(batch_size)
    chex.assert_scalar_non_negative(max_program_tokens - 5)
    # Every pending symbol costs >=1 token; budget feasibility bounds stack size.
    stack = jnp.zeros((batch_size, max_program_tokens + _MAX_PUSH), dtype=jnp.int32)
    prefix = jnp.asarray([_FIRST["m)"], TOKEN_TO_ID["m("], TOKEN_TO_ID["run"], TOKEN_TO_ID["DEF"]], jnp.int32)
    stack = stack.at[:, :4].set(prefix)
    return GrammarState(
        stack,
        jnp.full(batch_size, 4, jnp.int32),
        jnp.zeros(batch_size, jnp.int32),
        jnp.full(batch_size, 5, jnp.int32),
        jnp.full(batch_size, max_program_tokens, jnp.int32),
        jnp.ones(batch_size, jnp.bool_),
    )


def _validate(state: GrammarState) -> None:
    chex.assert_rank(state.stack, 2)
    chex.assert_shape((state.size, state.depth, state.minimum, state.remaining, state.valid), (state.stack.shape[0],))
    chex.assert_type((state.stack, state.size, state.depth, state.minimum, state.remaining), jnp.int32)
    chex.assert_type(state.valid, jnp.bool_)


@jax.jit
def allowed_tokens(state: GrammarState) -> jax.Array:
    """Return [B,V] legal next tokens with enough budget left to complete."""
    _validate(state)
    top = state.stack[jnp.arange(state.stack.shape[0]), jnp.maximum(0, state.size - 1)]
    next_cost = state.minimum[:, None] - jnp.asarray(_COSTS)[top, None] + jnp.asarray(_REPLACEMENT_COST)[top]
    allowed = jnp.asarray(_LEGAL)[top] & (next_cost <= state.remaining[:, None] - 1)
    controls = jnp.isin(jnp.arange(_VOCAB), jnp.asarray(_CONTROL_IDS))
    allowed &= ~(controls[None] & (state.depth[:, None] >= MAX_BLOCK_DEPTH + 1))
    return allowed & state.valid[:, None] & (state.size[:, None] > 0) & (state.remaining[:, None] > 0)


@jax.jit
def grammar_step(state: GrammarState, tokens: jax.Array) -> GrammarState:
    """Consume one token per row. PAD leaves the entire parser state unchanged."""
    _validate(state)
    chex.assert_shape(tokens, (state.stack.shape[0],))
    chex.assert_type(tokens, jnp.int32)
    rows = jnp.arange(tokens.shape[0])
    safe_tokens = jnp.clip(tokens, 0, _VOCAB - 1)
    legal = allowed_tokens(state)[rows, safe_tokens] & (tokens >= 0) & (tokens < _VOCAB)
    top = state.stack[rows, jnp.maximum(0, state.size - 1)]
    lengths = jnp.asarray(_LENGTHS)[top, safe_tokens]
    replacement = jnp.asarray(_REPLACEMENT)[top, safe_tokens]
    offsets = jnp.arange(_MAX_PUSH)[None]
    indices = jnp.clip(jnp.maximum(0, state.size - 1)[:, None] + offsets, 0, state.stack.shape[1] - 1)
    values = jnp.where(offsets < lengths[:, None], replacement, state.stack[rows[:, None], indices])
    stack = state.stack.at[rows[:, None], indices].set(values)
    next_state = GrammarState(
        stack,
        state.size - 1 + lengths,
        state.depth
        + jnp.isin(tokens, jnp.asarray(_OPEN_IDS)).astype(jnp.int32)
        - jnp.isin(tokens, jnp.asarray(_CLOSE_IDS)).astype(jnp.int32),
        state.minimum - jnp.asarray(_COSTS)[top] + jnp.asarray(_REPLACEMENT_COST)[top, safe_tokens],
        state.remaining - 1,
        state.valid & legal,
    )
    pad = tokens == TOKEN_TO_ID["<pad>"]

    # Illegal tokens invalidate the row without allowing out-of-bounds stack writes.
    def select(new: jax.Array, old: jax.Array) -> jax.Array:
        chex.assert_equal_shape((new, old))
        chex.assert_type((new, old), new.dtype)
        return jnp.where(legal.reshape((-1,) + (1,) * (new.ndim - 1)), new, old)

    result = jax.tree.map(select, next_state, state)
    return result._replace(valid=jnp.where(pad, state.valid, state.valid & legal))


@jax.jit
def mask_grammar_logits(logits: jax.Array, state: GrammarState) -> jax.Array:
    chex.assert_shape(logits, (state.stack.shape[0], _VOCAB))
    chex.assert_type(logits, jnp.float32)
    allowed = allowed_tokens(state)
    masked = jnp.where(allowed, logits, -jnp.inf)
    # Inactive rows take no action. Fix their dummy logit at zero: retaining the
    # model's m) logit can yield NaNs if it is nonfinite or was already masked.
    closing = TOKEN_TO_ID["m)"]
    return masked.at[:, closing].set(jnp.where(allowed.any(axis=-1), masked[:, closing], 0.0))


@partial(jax.jit, static_argnames="max_program_tokens")
def mask_sequence_logits(logits: jax.Array, actions: jax.Array, max_program_tokens: int) -> jax.Array:
    """Teacher-forcing masks: logits[t] sees actions[:t], never actions[t]."""
    chex.assert_shape(logits, (*actions.shape, _VOCAB))
    chex.assert_rank(actions, 2)
    chex.assert_type(logits, jnp.float32)
    chex.assert_type(actions, jnp.int32)

    def scan(state: GrammarState, inputs: tuple[jax.Array, jax.Array]) -> tuple[GrammarState, jax.Array]:
        prediction, token = inputs
        return grammar_step(state, token), mask_grammar_logits(prediction, state)

    _, masked = jax.lax.scan(scan, initial_grammar_state(actions.shape[1], max_program_tokens), (logits, actions))
    return masked
