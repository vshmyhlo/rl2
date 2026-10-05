import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.shape_checker import ShapeChecker


def test_named_dimensions_bind_and_are_reused() -> None:
    checker = ShapeChecker(B=2)
    checker.check(np.zeros((2, 3), dtype=np.float32), "BT", dtype="float32")
    checker.check((np.zeros((3, 2)), np.ones((3, 2))), "TB")
    assert checker["BTB"] == (2, 3, 2)
    with pytest.raises(AssertionError, match="BT"):
        checker.check(np.zeros((2, 4)), "BT")
    assert checker["BT"] == (2, 3)


def test_repeated_names_require_equal_axes_without_leaking_bindings() -> None:
    checker = ShapeChecker()
    with pytest.raises(AssertionError, match="TT"):
        checker.check(np.zeros((2, 3)), "TT")
    with pytest.raises(KeyError):
        checker["T"]
    checker.check(np.zeros((3, 3)), "TT")
    assert checker["T"] == (3,)


@pytest.mark.parametrize("failure", ["shape", "rank", "dtype"])
def test_failed_sequence_check_preserves_previous_bindings(failure: str) -> None:
    checker = ShapeChecker(B=2)
    first = np.zeros((2, 3), dtype=np.float32)
    if failure == "shape":
        second = np.zeros((4, 3), dtype=np.float32)
    elif failure == "rank":
        second = np.zeros((2,), dtype=np.float32)
    else:
        second = np.zeros((2, 3), dtype=np.int32)
    with pytest.raises(AssertionError, match=r"array\[1\].*BT"):
        checker.check([first, second], "BT", dtype=np.float32)
    assert checker["B"] == (2,)
    with pytest.raises(KeyError):
        checker["T"]
    checker.check(np.zeros((2, 4)), "BT")


def test_dtype_is_exact_and_optional() -> None:
    checker = ShapeChecker()
    array = np.zeros((2,), dtype=np.float32)
    checker.check(array, "B", dtype=np.dtype("float32"))
    with pytest.raises(AssertionError, match="float64"):
        checker.check(array, "B", dtype=np.float64)
    checker.check(np.zeros((2,), dtype=np.int32), "B")


def test_scalars_zero_dimensions_and_empty_sequence() -> None:
    checker = ShapeChecker(B=0)
    checker.check(np.array(1, dtype=np.int32), "", dtype=np.int32)
    assert checker[""] == ()
    checker.check(np.zeros((0, 2)), "BT")
    assert checker["BT"] == (0, 2)
    checker.check([], "X")
    with pytest.raises(KeyError):
        checker["X"]
    with pytest.raises(AssertionError, match="rank"):
        checker.check(np.zeros((1,)), "")


@pytest.mark.parametrize("name", ["", "batch"])
def test_constructor_rejects_names_that_are_not_single_characters(name: str) -> None:
    with pytest.raises(ValueError, match="single character"):
        ShapeChecker(**{name: 2})


@pytest.mark.parametrize("size", [True, 2.0])
def test_constructor_rejects_non_integer_sizes(size: bool | float) -> None:
    with pytest.raises(TypeError, match="integer"):
        ShapeChecker(B=size)


def test_constructor_rejects_negative_sizes() -> None:
    with pytest.raises(AssertionError, match="non-negative"):
        ShapeChecker(B=-1)


def test_jit_checks_static_shapes_and_dtypes() -> None:
    @jax.jit
    def checked(array: jax.Array) -> jax.Array:
        checker = ShapeChecker(B=2)
        checker.check(array, "BT", dtype=jnp.bfloat16)
        return array.reshape(checker["TB"])

    result = checked(jnp.zeros((2, 3), dtype=jnp.bfloat16))
    chex.assert_shape(result, (3, 2))
    chex.assert_type(result, jnp.bfloat16)
    with pytest.raises(AssertionError, match="BT"):
        checked(jnp.zeros((2,), dtype=jnp.bfloat16))
