"""Basic Flax Linen interface for experimenting with latent world models."""

import jax
from flax import linen as nn
from flax import struct


@struct.dataclass
class Prediction:
    """Predicted observation and transition outcomes, stored as a JAX pytree.

    Observation shape matches the model's observation space. Reward and
    termination logits have only the leading batch dimensions; termination
    denotes an environment terminal state, excluding time-limit truncation.
    """

    observation: jax.Array
    reward: jax.Array
    termination_logits: jax.Array


class WorldModel(nn.Module):
    """Subclass and implement the encoder, latent dynamics, and prediction heads.

    Inputs share leading batch dimensions. Observation preprocessing, action
    encoding, and latent shape are defined by each implementation. Define
    layers in ``setup`` or use Linen's ``@nn.compact`` on overridden methods.
    Stochastic implementations can use ``self.make_rng("sample")`` with a
    caller-supplied ``rngs={"sample": key}`` in ``init`` and ``apply``.

    Initialize the full model with ``model.init(key, observation, action)``.
    For latent rollouts, call ``apply`` with ``method=model.transition`` and
    decode the resulting states with ``method=model.decode``.
    """

    def encode(self, observation: jax.Array) -> jax.Array:
        """Encode the current observation into a latent state."""
        raise NotImplementedError

    def transition(self, latent: jax.Array, action: jax.Array) -> jax.Array:
        """Predict the next latent state after taking an action."""
        raise NotImplementedError

    def decode(self, latent: jax.Array) -> Prediction:
        """Predict an observation and incoming reward/termination from a state."""
        raise NotImplementedError

    def __call__(self, observation: jax.Array, action: jax.Array) -> tuple[jax.Array, Prediction]:
        """Return the next latent state and predictions for this transition."""
        latent = self.transition(self.encode(observation), action)
        return latent, self.decode(latent)
