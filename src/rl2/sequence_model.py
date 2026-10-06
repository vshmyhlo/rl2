"""Abstract bases for autoregressive and bidirectional sequence models."""

from abc import ABC, abstractmethod

import jax


class BDSequenceModel(ABC):
    """Bidirectional sequence operation without carry or single-step decoding.

    Inputs are floating-point arrays with shape [batch,time,dim]. Required
    int32 ``x_len[batch]`` lies in [0, time] and denotes each example's
    left-aligned valid prefix, ``x[b, :x_len[b]]``. Remaining positions are
    right padding. Padding values must not affect valid outputs; padded
    outputs are zero. A zero length produces an all-zero output for that
    example.

    Each call returns an array with the same shape as its input. Valid outputs
    may depend on all valid tokens, including future tokens. There is no carry
    input or output, no step method, and no chunk-equivalence requirement.

    Output dtype and parameter storage are implementation-specific.
    Implementations validate shapes, dtypes, and length bounds, and must test
    padding isolation; abstract methods alone cannot enforce it. For Flax
    modules, invoke through ``init``/``apply``, on a bound module, or as a
    submodule.
    """

    @abstractmethod
    def __call__(self, x: jax.Array, x_len: jax.Array) -> jax.Array:
        """Process [batch,time,dim], returning same-shaped bidirectional output.

        Required int32 lengths have shape [batch] and lie in [0, time],
        delimiting left-aligned valid tokens. Right-padded outputs are zero.
        Implementations may require nonempty batch and time dimensions.
        """
        raise NotImplementedError


class ARSequenceModel[CarryT](ABC):
    """Autoregressive sequence and step operations with model-specific carry.

    Every implementation must make ``__call__`` equivalent to time-stacked
    ``step`` calls, up to floating-point numerical tolerance. For input
    ``x[batch,time,dim]``, lengths ``x_len[batch]``, and initial carry ``c``,
    calling ``self(x, x_len, c)`` must produce the same outputs and final carry
    as processing ``x[:, t]`` in increasing time order with ``step``, passing
    each returned carry to the next step. At time ``t``, the step length is
    ``(t < x_len).astype(int32)``; stack its outputs along axis 1.

    This invariant applies both from scratch (``c=None``) and from any valid
    supplied carry. Both methods must support either starting mode. Splitting
    a sequence into chunks and passing carry between sequence calls must also
    preserve outputs and final carry, using each chunk's valid prefix lengths.
    Outputs must not depend on future inputs or on how inputs are chunked.

    Inputs are floating-point arrays. Required int32 ``x_len[batch]`` denotes
    the length of each example's left-aligned valid prefix: ``x[b, :x_len[b]]``.
    The remaining positions are right padding. Padding values must not affect
    valid outputs or carry; padded outputs are zero and padded steps leave
    carry unchanged. A zero length preserves the supplied carry, or returns
    the model's fresh initial carry when none was supplied.

    The equivalence assumes identical model parameters and computation
    settings. Carry structure, capacity, and output dtype are model-specific.
    Gradients flow through supplied carry unless the caller detaches it.
    Implementations validate shapes, dtypes, and length bounds, and must test
    the equivalence invariant; abstract methods alone cannot enforce it.

    This base class does not prescribe parameter storage.
    For Flax modules, invoke these operations through ``init``/``apply``, on a
    bound module, or as a submodule.
    """

    @abstractmethod
    def __call__(
        self,
        x: jax.Array,
        x_len: jax.Array,
        carry: CarryT | None = None,
    ) -> tuple[CarryT, jax.Array]:
        """Process [batch,time,dim] equivalently to repeated ``step`` calls.

        Return (final carry, same-shaped output). Required int32 lengths have
        shape [batch] and lie in [0, time], delimiting left-aligned valid tokens.
        Omitted carry starts fresh; supplied carry continues prior history.
        Implementations may require nonempty batch and time dimensions.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        x: jax.Array,
        x_len: jax.Array,
        carry: CarryT | None = None,
    ) -> tuple[CarryT, jax.Array]:
        """Process [batch,dim], returning (updated carry, same-shaped output).

        Required int32 lengths have shape [batch] and must be 0 or 1. A zero
        length produces zero output and preserves that example's carry.
        Omitted carry starts fresh; supplied carry continues prior history.
        Equivalent to ``__call__`` with a singleton time axis, removed from
        the output. Repeated steps must satisfy the class equivalence invariant.
        """
        raise NotImplementedError
