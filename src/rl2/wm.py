"""Flax Linen world-model interface and a deterministic Mamba-3 baseline."""

import math

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from rl2.mamba3 import Mamba3, Mamba3Carry


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


type RecurrentPrediction = tuple[Mamba3Carry, jax.Array, Prediction]


class MambaWorldModel(WorldModel):
    """MLP observation encoder/decoder around action-conditioned Mamba dynamics.

    Observations must be floating point, preprocessed by the caller, with shape
    ``[batch, *observation_shape]`` or ``[time, batch, *observation_shape]``.
    Actions are integer IDs in ``[0, num_actions)`` with matching batch/time
    dimensions. Predictions at index t target observation t+1, reward t, and
    termination of that transition. Observation and reward heads are unbounded.

    ``__call__`` and ``transition`` start with zero history. For recurrent use,
    ``observe`` encodes supplied observations (teacher forcing), while
    ``imagine`` accepts latents, including the previous predicted latent.
    Both return ``(carry, next_latent, prediction)`` and accept episode-start
    masks that clear history before the corresponding input. After a reset,
    use ``observe`` with the new initial observation to replace the old latent.
    A sequence returns all predicted latents but only the final carry.

    Example::

        model = MambaWorldModel(observation_shape=(4,), num_actions=2)
        obs, actions = jnp.zeros((8, 4)), jnp.zeros(8, dtype=jnp.int32)
        variables = model.init(jax.random.key(0), obs, actions)
        carry, latent, prediction = model.apply(variables, obs, actions, method=model.observe)
        carry, latent, prediction = model.apply(variables, latent, actions, carry, method=model.imagine)

    Parameters, returned latents, predictions, and carry remain float32;
    ``dtype`` controls the internal projection precision.
    """

    observation_shape: tuple[int, ...]
    num_actions: int
    d_model: int = 128
    d_state: int = 64
    expand: int = 2
    headdim: int = 32
    mimo_rank: int = 1
    dtype: jax.typing.DTypeLike = jnp.float32

    @nn.nowrap
    def _make_mixer(self) -> Mamba3:
        return Mamba3(
            d_model=self.d_model,
            d_state=self.d_state,
            expand=self.expand,
            headdim=self.headdim,
            mimo_rank=self.mimo_rank,
            dtype=self.dtype,
            parent=None,
        )

    def setup(self) -> None:
        if not isinstance(self.observation_shape, tuple) or not self.observation_shape:
            raise ValueError("observation_shape must be a nonempty tuple of positive integers")
        for size in (*self.observation_shape, self.num_actions):
            chex.assert_type(size, int)
            chex.assert_scalar_positive(size)
        self.encoder_hidden = nn.Dense(self.d_model, dtype=self.dtype)
        self.encoder_out = nn.Dense(self.d_model, dtype=self.dtype)
        self.action_embedding = nn.Embed(self.num_actions, self.d_model, dtype=self.dtype)
        self.input_projection = nn.Dense(self.d_model, dtype=self.dtype)
        self.pre_norm = nn.LayerNorm(dtype=self.dtype)
        self.mixer = self._make_mixer()
        self.post_norm = nn.LayerNorm(dtype=jnp.float32)
        self.decoder_hidden = nn.Dense(self.d_model, dtype=self.dtype)
        self.observation_head = nn.Dense(math.prod(self.observation_shape), dtype=jnp.float32)
        self.outcome_head = nn.Dense(2, dtype=jnp.float32)

    @nn.nowrap
    def initial_carry(self, batch_size: int) -> Mamba3Carry:
        """Allocate zero Mamba history without initializing model parameters."""
        chex.assert_type(batch_size, int)
        chex.assert_scalar_positive(batch_size)
        return self._make_mixer().initial_carry(batch_size)

    @nn.nowrap
    def _check_latent(self, latent: jax.Array) -> None:
        chex.assert_rank(latent, {2, 3})
        chex.assert_shape(latent, (*latent.shape[:-1], self.d_model))
        chex.assert_type(latent, float)
        chex.assert_scalar_positive(latent.shape[-2])

    def encode(self, observation: jax.Array) -> jax.Array:
        """Encode batched observations or time-major sequences into latents."""
        observation_rank = len(self.observation_shape)
        chex.assert_rank(observation, {observation_rank + 1, observation_rank + 2})
        chex.assert_type(observation, float)
        leading = observation.shape[:-observation_rank]
        chex.assert_shape(observation, (*leading, *self.observation_shape))
        chex.assert_scalar_positive(leading[-1])
        flat = observation.reshape((*leading, math.prod(self.observation_shape)))
        latent = self.encoder_out(nn.silu(self.encoder_hidden(flat))).astype(jnp.float32)
        self._check_latent(latent)
        return latent

    def _transition(
        self,
        latent: jax.Array,
        action: jax.Array,
        carry: Mamba3Carry | None = None,
        episode_starts: jax.Array | None = None,
    ) -> tuple[Mamba3Carry, jax.Array]:
        self._check_latent(latent)
        chex.assert_shape(action, latent.shape[:-1])
        chex.assert_type(action, int)
        if carry is not None:
            chex.assert_trees_all_equal_shapes_and_dtypes(carry, self.initial_carry(latent.shape[-2]))
        if episode_starts is not None:
            chex.assert_shape(episode_starts, action.shape)
            chex.assert_type(episode_starts, bool)
        action_features = self.action_embedding(action).astype(jnp.float32)
        inputs = self.input_projection(jnp.concatenate((latent, action_features), axis=-1)).astype(jnp.float32)
        normalized = self.pre_norm(inputs)
        if latent.ndim == 2:
            carry, mixed = self.mixer.step(normalized, carry, episode_starts)
        else:
            carry, mixed = self.mixer(normalized, carry, episode_starts)
        next_latent = self.post_norm(inputs + mixed.astype(jnp.float32))
        chex.assert_equal_shape((latent, next_latent))
        chex.assert_type(next_latent, jnp.float32)
        return carry, next_latent

    def transition(self, latent: jax.Array, action: jax.Array) -> jax.Array:
        """Predict next latents from zero history; use ``imagine`` to keep carry."""
        _, next_latent = self._transition(latent, action)
        return next_latent

    def decode(self, latent: jax.Array) -> Prediction:
        """Decode next-observation, reward, and termination-logit predictions."""
        self._check_latent(latent)
        hidden = nn.silu(self.decoder_hidden(latent))
        observation = self.observation_head(hidden).reshape((*latent.shape[:-1], *self.observation_shape))
        outcomes = self.outcome_head(hidden)
        chex.assert_shape(outcomes, (*latent.shape[:-1], 2))
        chex.assert_type((observation, outcomes), jnp.float32)
        return Prediction(observation, outcomes[..., 0], outcomes[..., 1])

    def observe(
        self,
        observation: jax.Array,
        action: jax.Array,
        carry: Mamba3Carry | None = None,
        episode_starts: jax.Array | None = None,
    ) -> RecurrentPrediction:
        """Predict from actual observations, preserving history across calls."""
        return self.imagine(self.encode(observation), action, carry, episode_starts)

    def imagine(
        self,
        latent: jax.Array,
        action: jax.Array,
        carry: Mamba3Carry | None = None,
        episode_starts: jax.Array | None = None,
    ) -> RecurrentPrediction:
        """Advance supplied latents without observations and decode predictions.

        For an autoregressive rollout, feed each returned latent and carry into
        the next call. Sequence inputs supply each step's latent explicitly.
        """
        carry, next_latent = self._transition(latent, action, carry, episode_starts)
        return carry, next_latent, self.decode(next_latent)
