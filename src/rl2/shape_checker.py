from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import chex
import numpy as np
from flax.typing import Dtype


class Shaped(Protocol):
    """Array interface used by :class:`ShapeChecker`; values are never read."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Dtype: ...


class ShapeChecker:
    """Bind single-character dimension names and check array metadata.

    Each character in a shape string names one axis: ``"BTD"`` describes a
    rank-three array. Names are case-sensitive, with no separators, wildcards,
    or broadcasting. Repeated names require equal sizes (``"TT"`` is square),
    and ``""`` describes a scalar array.

    Keyword arguments pre-bind dimensions to non-negative integer sizes; their
    names must be single characters. Other dimensions bind on their first
    successful check and persist across calls. A failed check leaves all
    existing bindings unchanged and adds none. Use a fresh checker for each
    independent group of arrays.

    Example::

        import numpy as np
        from rl2.shape_checker import ShapeChecker

        checker = ShapeChecker(B=2)
        x = np.zeros((2, 3, 4), dtype=np.float32)
        checker.check(x, "BTD", dtype=np.float32)  # B=2, T=3, D=4
        checker.check([x, x.copy()], "BTD")
        checker.check(np.zeros((3, 2)), "TB")
        assert checker["DT"] == (4, 3)  # Shape tuple for reshape/allocation.

    NumPy arrays, JAX arrays, and objects exposing concrete ``shape`` and
    ``dtype`` metadata are supported. Inside ``jax.jit``, create the checker
    inside the traced function: checks run at tracing time, not on each compiled
    execution. Do not share a mutable checker across independent JIT traces.
    Symbolic dimensions are not supported. Checks use Chex assertions and are
    therefore subject to Chex's global assertion-disable setting.
    """

    def __init__(self, **dims: int) -> None:
        self._dims: dict[str, int] = {}
        for name, size in dims.items():
            if len(name) != 1:
                raise ValueError(f"Dimension name {name!r} must be a single character")
            if not isinstance(size, (int, np.integer)) or isinstance(size, bool):
                raise TypeError(f"Dimension {name!r} must have an integer size, got {size!r}")
            chex.assert_scalar_non_negative(int(size))
            self._dims[name] = int(size)

    def check(
        self,
        arrays: Shaped | Sequence[Shaped],
        names: str,
        dtype: Dtype | None = None,
    ) -> None:
        """Check one array or a flat sequence against the same shape and dtype.

        ``names`` must contain one dimension name per axis. An empty sequence
        binds nothing. If ``dtype`` is provided, it is normalized with ``np.dtype``
        and must match exactly; no casting occurs. For example, ``np.float32``
        and ``"float32"`` are equivalent, while Python ``float`` means NumPy's
        default float dtype, not any floating-point dtype. If omitted, dtypes
        are neither checked nor bound across calls.

        Raises ``AssertionError`` for rank, dimension-size, or dtype mismatches.
        New bindings are saved only after every array passes.
        """
        arrays = arrays if isinstance(arrays, Sequence) else [arrays]

        expected_dtype = None if dtype is None else np.dtype(dtype)
        dims = self._dims.copy()
        for i, array in enumerate(arrays):
            array_label = "array" if len(arrays) == 1 else f"array[{i}]"
            try:
                if expected_dtype is not None:
                    chex.assert_type(array, expected_dtype)
                chex.assert_rank(array, len(names))
                expected_shape = tuple(
                    dims.setdefault(name, size) for size, name in zip(array.shape, names, strict=True)
                )
                chex.assert_shape(array, expected_shape)
            except AssertionError as error:
                raise AssertionError(
                    f"{array_label} failed shape check against {names!r}: {error}; "
                    f"shape={array.shape}; bound dims {dims}"
                ) from error
        self._dims = dims

    def __getitem__(self, names: str) -> tuple[int, ...]:
        """Return sizes in name order (including repeats); raise ``KeyError`` for an unbound name."""
        return tuple(self._dims[n] for n in names)
