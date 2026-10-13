"""Action spaces and JAX policies; S denotes arbitrary batch dimensions."""

from dataclasses import dataclass
from math import isfinite, prod

import chex
import distrax
import jax
import jax.numpy as jnp
from flax import struct
from tensorflow_probability.substrates import jax as tfp


@dataclass(frozen=True)
class DiscreteSpace:
    n: int

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("Discrete action count must be positive")


@dataclass(frozen=True)
class ContinuousSpace:
    """Finite Box bounds, flattened in row-major order; scalar actions use shape ()."""

    shape: tuple[int, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]

    def __post_init__(self) -> None:
        if any(size <= 0 for size in self.shape):
            raise ValueError("Continuous action dimensions must be positive")
        if len(self.low) != self.size or len(self.high) != self.size:
            raise ValueError("Continuous bounds must match the flattened action shape")
        if any(not (isfinite(low) and isfinite(high) and low < high) for low, high in zip(self.low, self.high)):
            raise ValueError("Continuous bounds must be finite with low < high")

    @property
    def size(self) -> int:
        return prod(self.shape)


type ActionSpace = DiscreteSpace | ContinuousSpace


@struct.dataclass
class CategoricalPolicy:
    logits: jax.Array  # (*S, A) float32

    def sample(
        self,
        *,
        seed: jax.Array,  # ()
        sample_shape: int | tuple[int, ...] = (),
    ) -> jax.Array:  # (*sample_shape, *S)
        return distrax.Categorical(logits=self.logits).sample(seed=seed, sample_shape=sample_shape)

    def sample_and_log_prob(
        self,
        *,
        seed: jax.Array,  # ()
        sample_shape: int | tuple[int, ...] = (),
    ) -> tuple[jax.Array, jax.Array]:  # each (*sample_shape, *S)
        return distrax.Categorical(logits=self.logits).sample_and_log_prob(seed=seed, sample_shape=sample_shape)

    def log_prob(self, actions: jax.Array) -> jax.Array:  # (*S,) -> (*S,)
        chex.assert_type(actions, int)
        return distrax.Categorical(logits=self.logits).log_prob(actions)  # (*S,)

    def entropy(self) -> jax.Array:  # (*S,)
        return distrax.Categorical(logits=self.logits).entropy()  # (*S,)


@struct.dataclass
class TruncatedGaussianPolicy:
    """Independent truncated normals; parameters describe the underlying Gaussian."""

    mean: jax.Array  # (*S, D) float32
    log_std: jax.Array  # (*S, D) float32
    action_space: ContinuousSpace = struct.field(pytree_node=False)

    @property
    def distribution(self) -> distrax.Independent:
        low, high = jnp.asarray((self.action_space.low, self.action_space.high), jnp.float32)  # each (D,)
        normal = tfp.distributions.TruncatedNormal(loc=self.mean, scale=jnp.exp(self.log_std), low=low, high=high)
        return distrax.Independent(normal, reinterpreted_batch_ndims=1)  # batch: (*S,); event: (D,)

    def sample(
        self,
        *,
        seed: jax.Array,  # ()
        sample_shape: int | tuple[int, ...] = (),
    ) -> jax.Array:  # (*sample_shape, *S, *action_shape)
        actions = self.distribution.sample(seed=seed, sample_shape=sample_shape)  # (*sample_shape, *S, D)
        return actions.reshape((*actions.shape[:-1], *self.action_space.shape))  # (*sample_shape, *S, *action_shape)

    def sample_and_log_prob(
        self,
        *,
        seed: jax.Array,  # ()
        sample_shape: int | tuple[int, ...] = (),
    ) -> tuple[jax.Array, jax.Array]:  # (*sample_shape, *S, *action_shape); (*sample_shape, *S)
        actions = self.sample(seed=seed, sample_shape=sample_shape)  # (*sample_shape, *S, *action_shape)
        # Evaluate the rounded action that PPO stores, even for tiny scales near a bound.
        actions = jax.lax.optimization_barrier(actions)  # (*sample_shape, *S, *action_shape)
        return actions, self.log_prob(actions)  # (*sample_shape, *S, *action_shape); (*sample_shape, *S)

    def log_prob(self, actions: jax.Array) -> jax.Array:  # (*S, *action_shape) -> (*S,)
        chex.assert_type(actions, float)
        batch_shape = actions.shape[: actions.ndim - len(self.action_space.shape)]
        flat_actions = actions.reshape((*batch_shape, self.action_space.size))  # (*S, D)
        return self.distribution.log_prob(flat_actions)  # (*S,)

    def entropy(self) -> jax.Array:  # (*S,)
        return self.distribution.entropy()  # (*S,)


type Policy = CategoricalPolicy | TruncatedGaussianPolicy
