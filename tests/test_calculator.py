import pytest

from rl2.calculator import calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("12 * 13", "156"),
        ("-(2 + 3) ** 2 + 10 // 3 + 10 % 3", "-21"),
        ("1 / 8", "0.125"),
        ("4 / 2", "2"),
        ("1000000000000001 / 1", "1000000000000001"),
        ("1.2345678901234567", "1.2345678901234567"),
        ("1 / 0", "Error:"),
        ("__import__('os').getcwd()", "Error:"),
        ("True + 1", "Error:"),
        ("1 +", "Error:"),
        ("2 ** 101", "Error:"),
        ("(-1) ** 0.5", "Error:"),
        ("1e101", "Error:"),
        ("1e100 * 10", "Error:"),
        ("1" * 257, "Error:"),
        ("+".join(["1"] * 30), "Error:"),
    ],
)
def test_calculate(expression: str, expected: str) -> None:
    actual = calculate(expression)
    assert actual.startswith(expected) if expected == "Error:" else actual == expected
