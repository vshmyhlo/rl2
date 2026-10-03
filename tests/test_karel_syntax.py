from itertools import product

import numpy as np
import pytest

from rl2.karel import ACTIONS, PREDICATES, KarelConfig, KarelProgramError, _parse, sample_task
from rl2.karel_syntax import syntax_edit_distance, syntax_reward


def token_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    """Independent ordinary Levenshtein distance for the exhaustive oracle."""
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


@pytest.mark.parametrize(
    "program,distance",
    [
        ("", 5),
        ("m)", 4),
        ("DEF run m( move m)", 0),
        ("DEF run m( move", 1),
        ("DEF run m( m)", 1),
        ("DEF run m( move ELSE m)", 1),
        ("DEF run m( move m) ELSE", 1),
        ("ELSE DEF run m( move m)", 1),
        ("DEF run m( unknown m)", 1),
        ("DEF run m( <pad> m)", 1),
        ("DEF run m( REPEAT R=20 r( move r) m)", 1),
        ("DEF run m( WHILE c( frontIsClear c) w( move m)", 1),
        ("DEF run m( IF c( move c) i( move i) m)", 1),
        ("DEF run m( IF c( frontIsClear c) i( i) m)", 1),
        ("DEF run m( IFELSE c( markersPresent c) i( move i) ELSE e( e) m)", 1),
        ("DEF run m( REPEAT R=0 r( r) m)", 1),
        ("DEF run m( turnLeft turnRight", 1),
        ("move move move move move", 4),
    ],
)
def test_known_minimum_repairs(program: str, distance: int) -> None:
    tokens = program.split()
    assert syntax_edit_distance(tokens) == distance
    if distance:
        assert -2 < syntax_reward(tokens) <= -1.5
        assert syntax_reward(tokens) == -2 + 1 / (1 + distance)
        with pytest.raises(KarelProgramError):
            _parse(tokens)
    else:
        _parse(tokens)


def test_short_inputs_match_exhaustive_nearest_program_oracle() -> None:
    # For <=2 input tokens, an optimal repair has <=7 tokens. All valid programs
    # that short are flat primitive sequences; the shortest control program has 9.
    valid = [("DEF", "run", "m(", *body, "m)") for length in range(1, 4) for body in product(ACTIONS, repeat=length)]
    alphabet = ("DEF", "run", "m(", "move", "m)", "REPEAT", "unknown")
    for length in range(3):
        for tokens in product(alphabet, repeat=length):
            assert syntax_edit_distance(tokens) == min(token_distance(tokens, candidate) for candidate in valid)


def test_all_controls_predicates_negations_and_counts_match_parser() -> None:
    bodies = [f"REPEAT R={count} r( move r)" for count in range(20)]
    for predicate in PREDICATES:
        for condition in (f"c( {predicate} c)", f"c( not c( {predicate} c) c)"):
            bodies.extend(
                (
                    f"IF {condition} i( move i)",
                    f"IFELSE {condition} i( move i) ELSE e( turnLeft e)",
                    f"WHILE {condition} w( move w)",
                )
            )
    for body in bodies:
        tokens = f"DEF run m( {body} m)".split()
        _parse(tokens)
        assert syntax_edit_distance(tokens) == 0


def test_generated_programs_and_single_edits_agree_with_parser() -> None:
    rng = np.random.default_rng(12)
    for _ in range(12):
        program = sample_task(rng, KarelConfig()).program
        assert syntax_edit_distance(program) == 0
        index = int(rng.integers(len(program)))
        for edited in (
            program[:index] + program[index + 1 :],
            program[:index] + ("unknown",) + program[index:],
            program[:index] + ("unknown",) + program[index + 1 :],
        ):
            try:
                _parse(edited)
            except KarelProgramError:
                expected = 1
            else:
                expected = 0
            assert syntax_edit_distance(edited) == expected


def test_depth_bounded_grammar_matches_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    # A small limit exercises the same expanded grammar without a 265-token chart.
    monkeypatch.setattr("rl2.karel_syntax.MAX_BLOCK_DEPTH", 1)
    monkeypatch.setattr("rl2.karel.MAX_BLOCK_DEPTH", 1)
    valid = ["DEF", "run", "m(", "REPEAT", "R=1", "r(", "move", "move", "move", "move", "move", "r)", "m)"]
    _parse(valid)
    assert syntax_edit_distance(valid) == 0
    too_deep = ["DEF", "run", "m(", "REPEAT", "R=1", "r(", "REPEAT", "R=1", "r(", "move", "r)", "r)", "m)"]
    with pytest.raises(KarelProgramError, match="nesting"):
        _parse(too_deep)
    # Insert `move r)` before the inner REPEAT, then delete the last r): the two
    # loops become siblings, which is cheaper than deleting an entire loop.
    repaired = ["DEF", "run", "m(", "REPEAT", "R=1", "r(", "move", "r)", "REPEAT", "R=1", "r(", "move", "r)", "m)"]
    _parse(repaired)
    assert token_distance(tuple(too_deep), tuple(repaired)) == 3
    assert syntax_edit_distance(too_deep) == 3


def test_max_length_primitive_program_and_incomplete_submission() -> None:
    program = ["DEF", "run", "m(", *(["move"] * 124), "m)"]
    assert len(program) == 128
    assert syntax_edit_distance(program) == 0
    assert syntax_edit_distance(program[:-1]) == 1


def test_non_string_tokens_are_rejected() -> None:
    with pytest.raises(TypeError, match="token strings"):
        syntax_edit_distance([1])
