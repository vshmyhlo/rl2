"""Running statistics for scalar pytrees."""

import chex
import jax
import jax.numpy as jnp


class RunningSum:
    """Pure running-sum operations with an explicit scalar-pytree state."""

    @staticmethod
    def init(values: chex.ArrayTree) -> chex.ArrayTree:  # input/output leaves: ()
        """Initialize () zero leaves matching values' structure and dtypes."""
        return jax.tree.map(jnp.zeros_like, values)  # scalar leaves: ()

    @staticmethod
    def update(
        state: chex.ArrayTree,  # scalar leaves: ()
        values: chex.ArrayTree,  # scalar leaves: ()
    ) -> chex.ArrayTree:  # scalar leaves: ()
        """Return () sum leaves; values must match state's structure and dtypes."""
        return jax.tree.map(jnp.add, state, values)  # scalar leaves: ()
