"""Shared Flax interface for causal stacks with explicit, model-specific state."""

from abc import ABC, abstractmethod

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn


class BlockStack[CarryT](nn.Module, ABC):
    """Time-major sequence backbone, without embeddings or prediction heads.

    ``CarryT`` is the concrete stack's state pytree. Its array leaves have batch
    as their leading axis. Carry layouts and memory horizons are model-specific;
    a carry must only be reused with a compatible stack configuration.

    Sequence calls accept floating-point [time,batch,d_model] inputs and return
    (final carry, same-shaped outputs). Omitting carry starts fresh. Boolean
    ``episode_starts[time,batch]`` resets an example's history *before* its input.
    Calls are causal, and passing carry continues a preceding chunk. Empty
    sequences preserve supplied carry. Gradients flow through carry unless the
    caller explicitly detaches it with ``jax.lax.stop_gradient``.

    As with any Linen module, call through ``init``/``apply``, a bound module,
    or as a submodule. ``initial_carry`` also works on an unbound instance.
    Use ``model.apply(variables, x, carry, method=model.step)`` for a single step.
    """

    d_model: int
    num_layers: int

    @nn.nowrap
    @abstractmethod
    def initial_carry(self, batch_size: int) -> CarryT:
        """Allocate empty state for a positive batch size without parameter init."""
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        x: jax.Array,
        carry: CarryT | None = None,
        episode_starts: jax.Array | None = None,
    ) -> tuple[CarryT, jax.Array]:
        """Process [time,batch,d_model], validating inputs and concrete carry."""
        raise NotImplementedError

    def step(
        self,
        x: jax.Array,
        carry: CarryT | None = None,
        episode_starts: jax.Array | None = None,
    ) -> tuple[CarryT, jax.Array]:
        """Process [batch,d_model] with the sequence parameters and reset rules."""
        chex.assert_shape(x, (None, self.d_model))
        chex.assert_type(x, jnp.floating)
        if episode_starts is not None:
            chex.assert_shape(episode_starts, (x.shape[0],))
            chex.assert_type(episode_starts, jnp.bool_)
        starts = None if episode_starts is None else episode_starts[None]
        carry, y = self(x[None], carry, starts)
        return carry, y[0]
