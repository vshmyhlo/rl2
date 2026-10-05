import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.karel import ACTIONS, PREDICATES, TOKEN_TO_ID, TOKENS, KarelProgramError, _parse, execute_program, sample_task
from rl2.karel_grammar import (
    GrammarState,
    allowed_tokens,
    grammar_step,
    initial_grammar_state,
    mask_grammar_logits,
    mask_sequence_logits,
)
from rl2.karel_syntax import MAX_BLOCK_DEPTH


def prefix_state(prefix: str, budget: int = 128) -> GrammarState:
    state = initial_grammar_state(1, budget)
    for token in prefix.split():
        assert bool(allowed_tokens(state)[0, TOKEN_TO_ID[token]]), token
        state = grammar_step(state, jnp.asarray([TOKEN_TO_ID[token]], jnp.int32))
    assert bool(state.valid[0])
    return state


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("", {"DEF"}),
        ("DEF", {"run"}),
        ("DEF run", {"m("}),
        ("DEF run m(", {*ACTIONS, "IF", "IFELSE", "WHILE", "REPEAT"}),
        ("DEF run m( move", {*ACTIONS, "IF", "IFELSE", "WHILE", "REPEAT", "m)"}),
        ("DEF run m( WHILE", {"c("}),
        ("DEF run m( WHILE c(", {*PREDICATES, "not"}),
        ("DEF run m( WHILE c( not", {"c("}),
        ("DEF run m( WHILE c( not c(", set(PREDICATES)),
        ("DEF run m( WHILE c( not c( frontIsClear", {"c)"}),
        ("DEF run m( WHILE c( not c( frontIsClear c)", {"c)"}),
        ("DEF run m( WHILE c( not c( frontIsClear c) c)", {"w("}),
        ("DEF run m( WHILE c( frontIsClear c) w( move", {*ACTIONS, "IF", "IFELSE", "WHILE", "REPEAT", "w)"}),
        ("DEF run m( IFELSE c( markersPresent c) i( move i)", {"ELSE"}),
        ("DEF run m( IFELSE c( markersPresent c) i( move i) ELSE", {"e("}),
        ("DEF run m( IFELSE c( markersPresent c) i( move i) ELSE e(", {*ACTIONS, "IF", "IFELSE", "WHILE", "REPEAT"}),
        ("DEF run m( REPEAT", {f"R={i}" for i in range(20)}),
        ("DEF run m( REPEAT R=0", {"r("}),
    ],
)
def test_legal_token_sets_depend_on_full_prefix(prefix: str, expected: set[str]) -> None:
    mask = np.asarray(allowed_tokens(prefix_state(prefix)))[0]
    assert {TOKENS[index] for index in np.flatnonzero(mask)} == expected


def test_budget_reserves_nonempty_blocks_and_all_closers() -> None:
    state = prefix_state("DEF run m(", budget=5)
    assert {TOKENS[i] for i in np.flatnonzero(allowed_tokens(state)[0])} == set(ACTIONS)
    state = grammar_step(state, jnp.asarray([TOKEN_TO_ID["move"]], jnp.int32))
    assert {TOKENS[i] for i in np.flatnonzero(allowed_tokens(state)[0])} == {"m)"}
    # IFELSE's pending ELSE branch must be reserved, not just its current block.
    prefix = "DEF run m( IFELSE c( frontIsClear c) i( move i) ELSE e( move e) m)"
    prefix_state(prefix, budget=len(prefix.split()))
    state = prefix_state("DEF run m(", budget=len(prefix.split()) - 1)
    assert not bool(allowed_tokens(state)[0, TOKEN_TO_ID["IFELSE"]])


def test_nesting_matches_interpreter_limit() -> None:
    prefix = "DEF run m( " + "REPEAT R=1 r( " * MAX_BLOCK_DEPTH
    state = prefix_state(prefix, budget=4 * MAX_BLOCK_DEPTH + 20)
    for token in ("IF", "IFELSE", "WHILE", "REPEAT", "r)", "m)"):
        assert not bool(allowed_tokens(state)[0, TOKEN_TO_ID[token]])
    assert bool(allowed_tokens(state)[0, TOKEN_TO_ID["move"]])


@pytest.mark.parametrize(
    "opening,closing",
    [
        ("REPEAT R=19 r(", "r)"),
        ("WHILE c( not c( frontIsClear c) c) w(", "w)"),
        ("IF c( markersPresent c) i(", "i)"),
        ("IFELSE c( noMarkersPresent c) i(", "i) ELSE e( turnLeft e)"),
    ],
)
@pytest.mark.parametrize("depth", [MAX_BLOCK_DEPTH, MAX_BLOCK_DEPTH + 1])
def test_complete_nested_programs_agree_with_interpreter(opening: str, closing: str, depth: int) -> None:
    tokens = ("DEF run m( " + (opening + " ") * depth + "move " + (closing + " ") * depth + "m)").split()
    if depth <= MAX_BLOCK_DEPTH:
        _parse(tokens)
    else:
        with pytest.raises(KarelProgramError, match="nesting exceeds"):
            _parse(tokens)
    actions = jnp.asarray([[TOKEN_TO_ID[token]] for token in tokens], jnp.int32)
    masked = mask_sequence_logits(jnp.zeros((*actions.shape, len(TOKENS)), jnp.float32), actions, len(tokens))
    selected = jnp.take_along_axis(masked, actions[..., None], axis=-1)
    assert bool(jnp.isfinite(selected).all()) == (depth <= MAX_BLOCK_DEPTH)


def test_invalid_token_ids_are_sticky_and_do_not_affect_other_rows() -> None:
    state = initial_grammar_state(5, 8)
    tokens = jnp.asarray([TOKEN_TO_ID["<pad>"], TOKEN_TO_ID["DEF"], -1, len(TOKENS), 2**31 - 1], jnp.int32)
    updated = grammar_step(state, tokens)
    np.testing.assert_array_equal(updated.valid, [True, True, False, False, False])
    np.testing.assert_array_equal(allowed_tokens(updated)[0], allowed_tokens(state)[0])
    assert {TOKENS[i] for i in np.flatnonzero(allowed_tokens(updated)[1])} == {"run"}
    assert not np.asarray(allowed_tokens(updated)[2:]).any()
    updated = grammar_step(updated, jnp.full(5, TOKEN_TO_ID["DEF"], jnp.int32))
    np.testing.assert_array_equal(updated.valid, [True, False, False, False, False])
    assert int(updated.remaining[0]) == 7  # PAD did not consume the first row's budget.
    assert {TOKENS[i] for i in np.flatnonzero(allowed_tokens(updated)[0])} == {"run"}


def test_padding_done_and_invalid_prefixes_are_safe() -> None:
    state = prefix_state("DEF run m( move m)")
    assert int(state.size[0]) == 0
    assert not np.asarray(allowed_tokens(state)).any()
    padded = grammar_step(state, jnp.asarray([TOKEN_TO_ID["<pad>"]], jnp.int32))
    chex.assert_trees_all_equal(state, padded)
    logits = mask_grammar_logits(jnp.zeros((1, len(TOKENS)), jnp.float32), padded)
    assert np.isfinite(np.asarray(jax.nn.softmax(logits))).all()
    assert float(jax.nn.softmax(logits)[0, TOKEN_TO_ID["<pad>"]]) == 0.0
    invalid = grammar_step(initial_grammar_state(1, 5), jnp.asarray([TOKEN_TO_ID["move"]], jnp.int32))
    assert not bool(invalid.valid[0])
    assert not np.asarray(allowed_tokens(invalid)).any()


@pytest.mark.parametrize("finished", [False, True])
@pytest.mark.parametrize("value", [-jnp.inf, jnp.inf, jnp.nan])
def test_inactive_distribution_does_not_depend_on_model_logits(finished: bool, value: float) -> None:
    state = prefix_state("DEF run m( move m)") if finished else initial_grammar_state(1, 5)
    if not finished:
        state = grammar_step(state, jnp.asarray([TOKEN_TO_ID["move"]], jnp.int32))
    logits = jnp.full((1, len(TOKENS)), value, jnp.float32)
    expected = np.zeros((1, len(TOKENS)), np.float32)
    expected[0, TOKEN_TO_ID["m)"]] = 1
    np.testing.assert_array_equal(jax.nn.softmax(mask_grammar_logits(logits, state)), expected)

    def loss(prediction: jax.Array) -> jax.Array:
        chex.assert_shape(prediction, (1, len(TOKENS)))
        chex.assert_type(prediction, jnp.float32)
        return jax.nn.log_softmax(mask_grammar_logits(prediction, state))[0, TOKEN_TO_ID["m)"]]

    np.testing.assert_array_equal(jax.grad(loss)(logits), 0.0)


@pytest.mark.parametrize("budget", [5, 16, 128])
def test_random_masked_generation_always_parses_and_finishes(budget: int) -> None:
    batch = 4

    def generate(key: jax.Array) -> tuple[GrammarState, jax.Array]:
        chex.assert_shape(key, ())

        def step(
            carry: tuple[GrammarState, jax.Array], unused: None
        ) -> tuple[tuple[GrammarState, jax.Array], jax.Array]:
            state, key = carry
            key, sample_key = jax.random.split(key)
            logits = jnp.zeros((batch, len(TOKENS)), jnp.float32)
            logits = logits.at[:, jnp.asarray([TOKEN_TO_ID[t] for t in ("IF", "IFELSE", "REPEAT", "WHILE")])].set(2)
            tokens = jax.random.categorical(sample_key, mask_grammar_logits(logits, state)).astype(jnp.int32)
            tokens = jnp.where(state.size > 0, tokens, TOKEN_TO_ID["<pad>"])
            return (grammar_step(state, tokens), key), tokens

        (state, _), tokens = jax.lax.scan(step, (initial_grammar_state(batch, budget), key), None, length=budget)
        return state, tokens

    state, tokens = jax.jit(generate)(jax.random.key(42))
    assert np.asarray(state.valid).all()
    np.testing.assert_array_equal(state.size, 0)
    world = np.zeros((3, 3, 6), np.int32)
    world[..., 4] = 1
    world[1, 1, 4] = 0
    world[1, 1, 0] = 1
    for column in np.asarray(tokens).T:
        program = [TOKENS[index] for index in column if index != TOKEN_TO_ID["<pad>"]]
        assert program[-1] == "m)"
        try:
            execute_program(program, world, max_steps=32)
        except KarelProgramError as exc:
            assert exc.reason in ("runtime_error", "execution_limit")


def test_teacher_forcing_uses_only_preceding_tokens() -> None:
    program = "DEF run m( move turnLeft m)"
    actions = jnp.asarray([[TOKEN_TO_ID[t]] for t in program.split()] + [[0], [0]], jnp.int32)
    logits = jax.random.normal(jax.random.key(0), (len(actions), 1, len(TOKENS)))
    masked = mask_sequence_logits(logits, actions, max_program_tokens=8)
    state = initial_grammar_state(1, 8)
    for t in range(len(actions)):
        np.testing.assert_array_equal(masked[t], mask_grammar_logits(logits[t], state))
        state = grammar_step(state, actions[t])
    changed = actions.at[4, 0].set(TOKEN_TO_ID["REPEAT"])
    other = mask_sequence_logits(logits, changed, max_program_tokens=8)
    np.testing.assert_array_equal(masked[:5], other[:5])


def test_sampled_reference_solutions_remain_allowed() -> None:
    rng = np.random.default_rng(7)
    programs = [sample_task(rng).program for _ in range(4)]
    budget = max(map(len, programs))
    actions = np.zeros((budget, len(programs)), np.int32)
    for index, program in enumerate(programs):
        actions[: len(program), index] = [TOKEN_TO_ID[token] for token in program]
    masked = mask_sequence_logits(jnp.zeros((*actions.shape, len(TOKENS)), jnp.float32), jnp.asarray(actions), budget)
    selected = np.take_along_axis(np.asarray(masked), actions[..., None], axis=-1)[..., 0]
    assert np.isfinite(selected[actions != TOKEN_TO_ID["<pad>"]]).all()
