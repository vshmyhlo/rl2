"""Abstract bases for autoregressive, bidirectional, and recurrent sequence models."""

from abc import ABC, abstractmethod

import jax


class RecurrentSequenceModel[CarryT](ABC):
    """Recurrent sequence and step operations with PPO-style episode resets.

    ``__call__`` is the time-stacked version of ``step``. It accepts a leading
    time dimension, ``x[time,batch,input_dim]``, while ``step`` accepts
    ``x[batch,input_dim]`` with no time dimension. Both methods require incoming
    model-specific carry and return updated outgoing carry. To start fresh,
    supply ``initial_carry(num_envs)``; otherwise supply carry from a previous
    call to continue prior history. Carry cannot be omitted or ``None``.

    Required boolean ``episode_starts`` masks have shape [time,batch] for
    ``__call__`` and [batch] for ``step``. Each true entry resets only that
    example's carry to the model's initial state before processing the current
    input; a false entry continues from its incoming carry. Every input is
    processed, including inputs that start a new episode.

    Calling ``self(x, carry, episode_starts)`` must be equivalent, up to
    floating-point numerical tolerance, to calling
    ``step(x[t], carry, episode_starts[t])`` in increasing time order and
    passing each returned carry to the next step. Stacking step outputs on
    axis 0 must reproduce the sequence output and yield the same final carry.
    This parity must hold both from the model's fresh initial state and from
    any valid carry containing prior history, including when resets occur
    within the sequence. It assumes identical parameters and computation settings.
    Splitting a sequence into chunks and passing carry between calls must
    also preserve outputs and final carry. Outputs cannot depend on future
    inputs or on history preceding the most recent episode reset.

    Both methods return (updated outgoing carry, output). Outputs have shape
    [time,batch,output_dim] or [batch,output_dim]; output_dim need not match
    input_dim. Carry structure, initial state, and input/output dtypes are
    implementation-specific. Gradients flow through supplied carry unless
    the caller detaches it; episode resets discard the prior history.

    Implementations validate shapes and dtypes and test sequence/step
    equivalence and reset isolation; abstract methods alone cannot enforce
    these properties. This base does not prescribe parameter storage.
    For Flax modules, invoke through ``init``/``apply``, on a bound module,
    or as a submodule. ``initial_carry`` also works on an unbound instance.
    """

    @abstractmethod
    def initial_carry(self, num_envs: int) -> CarryT:
        """Return fresh carry for a batch of ``num_envs`` independent examples.

        The result is suitable for both ``__call__`` and ``step`` and represents
        the same initial state used by episode resets. Carry structure and
        dtypes are implementation-specific. Implementations may require a
        nonempty batch. No parameter initialization or prior calls are required.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        x: jax.Array,
        carry: CarryT,
        episode_starts: jax.Array,
    ) -> tuple[CarryT, jax.Array]:
        """Process [time,batch,input_dim] with boolean resets [time,batch].

        Return (final carry, output [time,batch,output_dim]), equivalently to
        scanning ``step`` calls from the same required incoming carry.
        Supply the model's initial state to start fresh. True ``episode_starts`` entries
        reset carry before processing the corresponding inputs.
        Implementations may require nonempty batch and time dimensions.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        x: jax.Array,
        carry: CarryT,
        episode_starts: jax.Array,
    ) -> tuple[CarryT, jax.Array]:
        """Process [batch,input_dim] with boolean resets [batch].

        Return (updated carry, output [batch,output_dim]). Incoming carry is
        required; supply the model's initial state to start fresh. True
        ``episode_starts`` entries reset the corresponding examples' carry
        before processing their input. Equivalent to
        ``__call__`` with a singleton leading time axis, removing that axis
        from the output.
        """
        raise NotImplementedError


class BDSequenceModel(ABC):
    """Bidirectional sequence operation without carry or single-step decoding.

    Inputs are floating-point arrays with shape [batch,time,input_dim]. Required
    int32 ``x_len[batch]`` lies in [0, time] and denotes each example's
    left-aligned valid prefix, ``x[b, :x_len[b]]``. Remaining positions are
    right padding. Padding values must not affect valid outputs; padded
    outputs are zero. A zero length produces an all-zero output for that
    example.

    Each call returns an array with shape [batch,time,output_dim]. The output
    feature dimension need not match input_dim (for example, output logits
    may have one feature per vocabulary token). Valid outputs may depend on
    all valid tokens, including future tokens. There is no carry input or
    output, no step method, and no chunk-equivalence requirement.

    Output dtype and parameter storage are implementation-specific.
    Implementations validate shapes, dtypes, and length bounds, and must test
    padding isolation; abstract methods alone cannot enforce it. For Flax
    modules, invoke through ``init``/``apply``, on a bound module, or as a
    submodule.
    """

    @abstractmethod
    def __call__(self, x: jax.Array, x_len: jax.Array) -> jax.Array:
        """Process [batch,time,input_dim], returning [batch,time,output_dim].

        The output feature dimension need not match the input feature dimension.
        Required int32 lengths have shape [batch] and lie in [0, time],
        delimiting left-aligned valid tokens. Right-padded outputs are zero.
        Implementations may require nonempty batch and time dimensions.
        """
        raise NotImplementedError


class ARSequenceModel[CarryT](ABC):
    """Autoregressive sequence and step operations with model-specific carry.

    Every implementation must make ``__call__`` equivalent to time-stacked
    ``step`` calls, up to floating-point numerical tolerance. For input
    ``x[batch,time,input_dim]``, lengths ``x_len[batch]``, and initial carry ``c``,
    calling ``self(x, x_len, c)`` must produce the same outputs and final carry
    as processing ``x[:, t]`` in increasing time order with ``step``, passing
    each returned carry to the next step. At time ``t``, pass the boolean
    mask ``x_active=t < x_len``; stack the step outputs along axis 1.

    Sequence outputs have shape [batch,time,output_dim], and step outputs
    have shape [batch,output_dim]. The output feature dimension need not match
    input_dim (for example, output logits may have one feature per vocabulary
    token), but must agree between sequence and step operations.

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
        """Process [batch,time,input_dim] equivalently to repeated ``step`` calls.

        Return (final carry, output with shape [batch,time,output_dim]). The
        output feature dimension need not match the input feature dimension.
        Required int32 lengths have shape [batch] and lie in [0, time],
        delimiting left-aligned valid tokens.
        Omitted carry starts fresh; supplied carry continues prior history.
        Implementations may require nonempty batch and time dimensions.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        x: jax.Array,
        x_active: jax.Array,
        carry: CarryT | None = None,
    ) -> tuple[CarryT, jax.Array]:
        """Process [batch,input_dim], returning (updated carry, [batch,output_dim]).

        The output feature dimension need not match the input feature dimension.
        Required boolean x_active[batch] selects examples to advance.
        Inactive inputs are ignored, produce
        zero output, and preserve that example's carry. With no supplied carry,
        inactive examples retain the model's fresh initial carry.
        Omitted carry starts fresh; supplied carry continues prior history.
        Equivalent to ``__call__`` with a singleton time axis and int32 lengths
        obtained from x_active, removing that axis from the output.
        Repeated steps must satisfy the class equivalence invariant.
        """
        raise NotImplementedError
