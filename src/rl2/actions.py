"""Action spaces and JAX policies; S denotes batch dimensions, A categories, D flat action size.

Continuous log probabilities and differential entropies sum over the D action coordinates.
Multi-categorical log probabilities and entropies sum over discrete components.
Sampling prepends sample_shape to S; log_prob also accepts these leading sample dimensions.
"""

from abc import ABC, abstractmethod
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
class HierarchicalMultiDiscreteSpace:
    """Vector of two-level choices over contiguous command groups.

    Raw logits contain group logits followed by conditional logits for groups
    of size > 1. Every component uses the same group sizes.
    """

    nvec: tuple[int, ...]
    group_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.nvec or any(n <= 0 for n in self.nvec):
            raise ValueError("Multi-discrete action counts must be nonempty and positive")
        if not self.group_sizes or any(size <= 0 for size in self.group_sizes):
            raise ValueError("Action group sizes must be nonempty and positive")
        if any(n != sum(self.group_sizes) for n in self.nvec):
            raise ValueError("Action group sizes must sum to every component's action count")

    @property
    def logits_shape(self) -> tuple[int, int]:
        width = len(self.group_sizes) + sum(size for size in self.group_sizes if size > 1)
        return len(self.nvec), width


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
type MultiAgentActionSpace = DiscreteSpace | HierarchicalMultiDiscreteSpace


class Policy(ABC):
    """Action distribution; S denotes batch axes and E denotes event axes.

    Log probabilities and entropies reduce over the complete action event.
    Sampling prepends sample_shape; log_prob accepts the same leading axes.
    Concrete policies are Flax struct dataclasses registered as JAX pytrees.
    """

    @abstractmethod
    def sample(
        self,
        *,
        key: jax.Array,  # () typed PRNG key
        sample_shape: int | tuple[int, ...] = (),
    ) -> jax.Array:  # (*sample_shape, *S, *E)
        """Return (*sample_shape, *S, *E) sampled actions."""

    @abstractmethod
    def sample_and_log_prob(
        self,
        *,
        key: jax.Array,  # () typed PRNG key
        sample_shape: int | tuple[int, ...] = (),
    ) -> tuple[jax.Array, jax.Array]:  # (*sample_shape, *S, *E); (*sample_shape, *S)
        """Return (*sample_shape, *S, *E) actions and (*sample_shape, *S) log probabilities."""

    @abstractmethod
    def log_prob(self, actions: jax.Array) -> jax.Array:  # (*S, *E) -> (*S,)
        """Return (*S,) joint log probabilities for (*S, *E) actions."""

    @abstractmethod
    def entropy(self) -> jax.Array:  # (*S,)
        """Return (*S,) joint entropy, or differential entropy for continuous actions."""


def _masked_log_softmax(logits: jax.Array, mask: jax.Array) -> jax.Array:  # each (*S, A) -> (*S, A)
    # Normalize an empty branch safely before restoring its zero probability.
    safe_mask = mask | (~mask.any(axis=-1, keepdims=True) & (jnp.arange(mask.shape[-1]) == 0))  # (*S, A)
    log_prob = jax.nn.log_softmax(jnp.where(safe_mask, logits, -jnp.inf), axis=-1)  # (*S, A)
    return jnp.where(mask, log_prob, -jnp.inf)  # (*S, A)


def hierarchical_log_probs(
    logits: jax.Array,  # (*S, G + K), K = sum of non-singleton group sizes
    actions_mask: jax.Array,  # (*S, A), A = sum(group_sizes)
    group_sizes: tuple[int, ...],
) -> jax.Array:  # (*S, A)
    """Flatten masked group/conditional probabilities into command log probabilities.

    logits: (*S, G + K) group logits, then conditional logits in group order.
    actions_mask: (*S, A) valid commands, with at least one available per row.
    group_sizes: Positive sizes of G contiguous command groups.
    Returns: (*S, A) normalized log probabilities; unavailable commands are -inf.
    """
    masks: list[jax.Array] = []  # each (*S, group_size)
    offset = 0
    for size in group_sizes:
        masks.append(actions_mask[..., offset : offset + size])
        offset += size
    group_mask = jnp.stack([mask.any(axis=-1) for mask in masks], axis=-1)  # (*S, G)
    group_lp = _masked_log_softmax(logits[..., : len(group_sizes)], group_mask)  # (*S, G)
    commands: list[jax.Array] = []  # each (*S, group_size)
    offset = len(group_sizes)
    for group, (size, mask) in enumerate(zip(group_sizes, masks)):
        parent = group_lp[..., group : group + 1]  # (*S, 1)
        if size == 1:
            commands.append(parent)
        else:
            conditional = _masked_log_softmax(logits[..., offset : offset + size], mask)  # (*S, group_size)
            commands.append(parent + conditional)
            offset += size
    return jnp.concatenate(commands, axis=-1)  # (*S, A)


@struct.dataclass
class CategoricalPolicy(Policy):
    logits: jax.Array  # (*S, A) float32

    def sample(
        self,
        *,
        key: jax.Array,  # () typed PRNG key
        sample_shape: int | tuple[int, ...] = (),
    ) -> jax.Array:  # (*sample_shape, *S)
        return distrax.Categorical(logits=self.logits).sample(seed=key, sample_shape=sample_shape)

    def sample_and_log_prob(
        self,
        *,
        key: jax.Array,  # () typed PRNG key
        sample_shape: int | tuple[int, ...] = (),
    ) -> tuple[jax.Array, jax.Array]:  # each (*sample_shape, *S)
        return distrax.Categorical(logits=self.logits).sample_and_log_prob(seed=key, sample_shape=sample_shape)

    def log_prob(self, actions: jax.Array) -> jax.Array:  # (*S,) -> (*S,)
        chex.assert_type(actions, int)
        return distrax.Categorical(logits=self.logits).log_prob(actions)  # (*S,)

    def entropy(self) -> jax.Array:  # (*S,)
        return distrax.Categorical(logits=self.logits).entropy()  # (*S,)


@struct.dataclass
class MultiCategoricalPolicy(Policy):
    """One joint action with conditionally independent categorical components."""

    logits: jax.Array  # (*S, D, A) float32; unavailable/padded categories masked

    @property
    def distribution(self) -> distrax.Independent:
        return distrax.Independent(distrax.Categorical(logits=self.logits), reinterpreted_batch_ndims=1)

    def sample(
        self,
        *,
        key: jax.Array,  # () typed PRNG key
        sample_shape: int | tuple[int, ...] = (),
    ) -> jax.Array:  # (*sample_shape, *S, D)
        return self.distribution.sample(seed=key, sample_shape=sample_shape)

    def sample_and_log_prob(
        self,
        *,
        key: jax.Array,  # () typed PRNG key
        sample_shape: int | tuple[int, ...] = (),
    ) -> tuple[jax.Array, jax.Array]:  # (*sample_shape, *S, D); (*sample_shape, *S)
        return self.distribution.sample_and_log_prob(seed=key, sample_shape=sample_shape)

    def log_prob(self, actions: jax.Array) -> jax.Array:  # (*S, D) -> (*S,)
        chex.assert_type(actions, int)
        return self.distribution.log_prob(actions)  # (*S,)

    def entropy(self) -> jax.Array:  # (*S,)
        return self.distribution.entropy()  # (*S,)


@struct.dataclass
class HierarchicalMultiCategoricalPolicy(Policy):
    """Grouped choices exposed as a joint multi-categorical command distribution.

    Each component must have at least one valid command. Group sizes are static
    under JIT; logits retain the flattened command-space interface.
    """

    raw_logits: jax.Array  # (*S, D, G + K), K = sum of non-singleton group sizes
    actions_mask: jax.Array  # (*S, D, A), A = sum(group_sizes)
    group_sizes: tuple[int, ...] = struct.field(pytree_node=False)

    @property
    def logits(self) -> jax.Array:  # (*S, D, A)
        logits = hierarchical_log_probs(self.raw_logits, self.actions_mask, self.group_sizes)  # (*S, D, A)
        return jnp.where(self.actions_mask, logits, -1e10)  # (*S, D, A)

    @property
    def flat_policy(self) -> MultiCategoricalPolicy:
        return MultiCategoricalPolicy(logits=self.logits)

    def sample(
        self,
        *,
        key: jax.Array,  # () typed PRNG key
        sample_shape: int | tuple[int, ...] = (),
    ) -> jax.Array:  # (*sample_shape, *S, D)
        return self.flat_policy.sample(key=key, sample_shape=sample_shape)

    def sample_and_log_prob(
        self,
        *,
        key: jax.Array,  # () typed PRNG key
        sample_shape: int | tuple[int, ...] = (),
    ) -> tuple[jax.Array, jax.Array]:  # (*sample_shape, *S, D); (*sample_shape, *S)
        return self.flat_policy.sample_and_log_prob(key=key, sample_shape=sample_shape)

    def log_prob(self, actions: jax.Array) -> jax.Array:  # (*S, D) -> (*S,)
        return self.flat_policy.log_prob(actions)  # (*S,)

    def entropy(self) -> jax.Array:  # (*S,)
        return self.flat_policy.entropy()  # (*S,)


@struct.dataclass
class TruncatedGaussianPolicy(Policy):
    """Independent truncated normals; parameters describe the underlying Gaussian."""

    mean: jax.Array  # (*S, D) float32
    log_std: jax.Array  # (*S, D) float32
    action_space: ContinuousSpace = struct.field(pytree_node=False)

    @property
    def distribution(self) -> distrax.Independent:
        low, high = jnp.asarray((self.action_space.low, self.action_space.high), jnp.float32)  # each (D,)
        normal = tfp.distributions.TruncatedNormal(loc=self.mean, scale=jnp.exp(self.log_std), low=low, high=high)
        return distrax.Independent(distribution=normal, reinterpreted_batch_ndims=1)  # batch: (*S,); event: (D,)

    def sample(
        self,
        *,
        key: jax.Array,  # () typed PRNG key
        sample_shape: int | tuple[int, ...] = (),
    ) -> jax.Array:  # (*sample_shape, *S, *action_shape)
        actions = self.distribution.sample(seed=key, sample_shape=sample_shape)  # (*sample_shape, *S, D)
        return actions.reshape((*actions.shape[:-1], *self.action_space.shape))  # (*sample_shape, *S, *action_shape)

    def sample_and_log_prob(
        self,
        *,
        key: jax.Array,  # () typed PRNG key
        sample_shape: int | tuple[int, ...] = (),
    ) -> tuple[jax.Array, jax.Array]:  # (*sample_shape, *S, *action_shape); (*sample_shape, *S)
        actions = self.sample(key=key, sample_shape=sample_shape)  # (*sample_shape, *S, *action_shape)
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


@struct.dataclass
class BetaPolicy(Policy):
    """Independent Betas with positive alpha/beta, affinely scaled to Box bounds.

    Bounds must have representable float32 interior values. Sampling clips rounded
    endpoints into that interior; density and entropy describe the unclipped Beta.
    """

    alpha: jax.Array  # (*S, D) float32
    beta: jax.Array  # (*S, D) float32
    action_space: ContinuousSpace = struct.field(pytree_node=False)

    @property
    def distribution(self) -> distrax.Independent:
        low, high = jnp.asarray((self.action_space.low, self.action_space.high), jnp.float32)  # each (D,)
        beta = tfp.distributions.Beta(
            concentration1=self.alpha, concentration0=self.beta, force_probs_to_zero_outside_support=True
        )
        scaled = distrax.Transformed(distribution=beta, bijector=distrax.ScalarAffine(shift=low, scale=high - low))
        return distrax.Independent(distribution=scaled, reinterpreted_batch_ndims=1)  # batch: (*S,); event: (D,)

    def sample(
        self,
        *,
        key: jax.Array,  # () typed PRNG key
        sample_shape: int | tuple[int, ...] = (),
    ) -> jax.Array:  # (*sample_shape, *S, *action_shape)
        actions = self.distribution.sample(seed=key, sample_shape=sample_shape)  # (*sample_shape, *S, D)
        low, high = jnp.asarray((self.action_space.low, self.action_space.high), jnp.float32)  # each (D,)
        # Keep rounded samples away from endpoint singularities during PPO replay.
        margin = jnp.finfo(actions.dtype).eps * (high - low)  # (D,)
        actions = jnp.clip(
            actions, jnp.nextafter(low + margin, high), jnp.nextafter(high - margin, low)
        )  # (*sample_shape, *S, D)
        return actions.reshape((*actions.shape[:-1], *self.action_space.shape))  # (*sample_shape, *S, *action_shape)

    def sample_and_log_prob(
        self,
        *,
        key: jax.Array,  # () typed PRNG key
        sample_shape: int | tuple[int, ...] = (),
    ) -> tuple[jax.Array, jax.Array]:  # (*sample_shape, *S, *action_shape); (*sample_shape, *S)
        actions = self.sample(key=key, sample_shape=sample_shape)  # (*sample_shape, *S, *action_shape)
        actions = jax.lax.optimization_barrier(actions)  # (*sample_shape, *S, *action_shape)
        return actions, self.log_prob(actions)  # (*sample_shape, *S, *action_shape); (*sample_shape, *S)

    def log_prob(self, actions: jax.Array) -> jax.Array:  # (*S, *action_shape) -> (*S,)
        chex.assert_type(actions, float)
        batch_shape = actions.shape[: actions.ndim - len(self.action_space.shape)]
        flat_actions = actions.reshape((*batch_shape, self.action_space.size))  # (*S, D)
        low, high = jnp.asarray((self.action_space.low, self.action_space.high), jnp.float32)  # each (D,)
        # The affine inverse can round an out-of-bounds action back onto Beta's support.
        outside = jnp.any((flat_actions < low) | (flat_actions > high), axis=-1)  # (*S,)
        return jnp.where(outside, -jnp.inf, self.distribution.log_prob(flat_actions))  # (*S,)

    def entropy(self) -> jax.Array:  # (*S,)
        return self.distribution.entropy()  # (*S,)
