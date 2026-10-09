"""Categorical stochastic world model with observation-unaware Mamba dynamics."""

import math

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from rl2.mamba3 import Mamba3Stack, Mamba3StackCarry
from rl2.observation_decoder import ConvObservationDecoder
from rl2.observation_encoder import DEFAULT_STAGES, ConvObservationEncoder, ConvStages
from rl2.shape_checker import ShapeChecker


@struct.dataclass
class Prediction:
    """Decoded state: normalized pixels, incoming reward, and terminal logits.

    ``observation`` has shape ``[*leading, *observation_shape]`` and unbounded
    float32 values targeting [0, 1]. ``reward`` and ``termination_logits`` are
    float32 arrays with shape ``leading`` (batch, or time and batch). Terminal
    labels exclude time-limit truncations; apply sigmoid for probabilities.
    """

    observation: jax.Array
    reward: jax.Array
    termination_logits: jax.Array


@struct.dataclass
class WorldModelState:
    """History and sampled current state, aligned with the current observation.

    ``memory`` contains one Mamba carry per layer. ``deter`` is [B, D] and
    ``stoch`` is [B, S, C], containing one-hot samples of S categorical variables
    with C classes. Floating leaves are float32. ``initialized`` is bool [B];
    false means the next observe call must condition on its initial frame.
    """

    memory: Mamba3StackCarry
    deter: jax.Array
    stoch: jax.Array
    initialized: jax.Array


@struct.dataclass
class ObserveOutput:
    """Posterior reconstructions and distributions for each next observation.

    ``features`` is float32 [*leading, D + S*C]. Prior and posterior logits
    are float32 [*leading, S, C], with uniform mixing already applied.
    ``prediction`` decodes the posterior sample, not the prior mean.
    """

    features: jax.Array
    prior_logits: jax.Array
    posterior_logits: jax.Array
    prediction: Prediction


def check_keys(keys: jax.Array, leading: tuple[int, ...]) -> None:
    """Validate typed or legacy JAX keys with the specified leading axes."""
    data = jax.random.key_data(keys)
    chex.assert_shape(data, (*leading, None))
    chex.assert_type(data, jnp.uint32)


def categorical_kl(posterior_logits: jax.Array, prior_logits: jax.Array) -> jax.Array:
    """Return KL(q || p), summed over variables/classes, preserving leading axes.

    Args:
        posterior_logits: Float32 [*leading, stochastic_size, stochastic_classes].
        prior_logits: Matching float32 logits. Uniform mixing, if desired, must
            already be applied by the caller.

    Returns:
        Float32 KL in nats per batch/time element, shaped ``leading``.
    """
    chex.assert_equal_shape((posterior_logits, prior_logits))
    chex.assert_type((posterior_logits, prior_logits), jnp.float32)
    chex.assert_scalar_non_negative(posterior_logits.ndim - 2)
    log_q = jax.nn.log_softmax(posterior_logits, axis=-1)
    log_p = jax.nn.log_softmax(prior_logits, axis=-1)
    probs = jnp.exp(log_q)
    # A zero-mass category contributes zero, including when both log
    # probabilities are -inf. Mask operands before multiplication so the
    # backward pass does not encounter 0 * inf or -inf - -inf either.
    log_q = jnp.where(probs > 0, log_q, 0.0)
    log_p = jnp.where(probs > 0, log_p, 0.0)
    return jnp.sum(probs * (log_q - log_p), axis=(-2, -1))


def categorical_entropy(logits: jax.Array) -> jax.Array:
    """Return entropy in nats, summing variables/classes and retaining leading axes.

    Args:
        logits: Float32 [*leading, stochastic_size, stochastic_classes].
            Zero-probability categories may be represented by -inf logits.

    Returns:
        Float32 entropy shaped ``leading``. Each categorical variable must have
        at least one finite logit; an entirely invalid distribution remains NaN.
    """
    chex.assert_type(logits, jnp.float32)
    chex.assert_scalar_non_negative(logits.ndim - 2)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * jnp.where(probs > 0, log_probs, 0.0), axis=(-2, -1))


def latent_kl_losses(
    posterior_logits: jax.Array, prior_logits: jax.Array, free_nats: float
) -> tuple[jax.Array, jax.Array]:
    """Return scalar dynamics and representation KL losses with separate gradients.

    Logits are float32 [*leading, variables, classes]. The dynamics loss stops
    gradients into the posterior; representation stops gradients into the prior.
    A nonnegative free_nats floor is applied per transition before averaging.
    """
    chex.assert_shape(jnp.asarray(free_nats), ())
    dynamics = categorical_kl(jax.lax.stop_gradient(posterior_logits), prior_logits)
    representation = categorical_kl(posterior_logits, jax.lax.stop_gradient(prior_logits))
    return jnp.maximum(dynamics, free_nats).mean(), jnp.maximum(representation, free_nats).mean()


class WorldModel(nn.Module):
    """Flax interface for encoding images, advancing sampled states, and decoding.

    Stochastic methods take explicit JAX keys, separate from parameter RNGs.
    Implementations define their state structure and observation layout.
    """

    def encode(self, observation: jax.Array) -> jax.Array:
        """Return observation embeddings, preserving batch/time dimensions."""
        raise NotImplementedError

    def transition(self, state: WorldModelState, action: jax.Array, key: jax.Array) -> WorldModelState:
        """Sample the next state given the current state, batch of actions, and key."""
        raise NotImplementedError

    def decode(self, features: jax.Array) -> Prediction:
        """Decode state features into pixels and incoming transition outcomes."""
        raise NotImplementedError


# Inputs to the recurrent training scan: current embedding, action, next
# embedding, episode-start mask, and an independent random key for this step.
type ObserveInputs = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
type LatentOutputs = tuple[jax.Array, jax.Array, jax.Array]


class MambaWorldModel(WorldModel):
    """Dreamer-style categorical states with a Mamba stack as the history model.

    A posterior q(z_t | h_t, encode(o_t)) infers the state from real images.
    Mamba consumes sampled z_t and action a_t to produce h_{t+1}; a learned
    prior p(z_{t+1} | h_{t+1}) supplies samples during imagination. Both heads
    use the same categorical representation. Decoding [h, z] yields pixels,
    reward, and termination. Samples use straight-through probability gradients.

    ``observe`` consumes transitions (o_t, a_t, o_{t+1}) and reconstructs the
    next frame from its posterior sample. Its final state is aligned with the
    last o_{t+1}, ready for the following chunk. Episode-start masks discard all
    previous memory and infer a fresh initial posterior from o_t. Initial and
    reset observations are conditioned on, while next observations are trained
    reconstruction targets, including final frames before environment resets.

    ``condition`` seeds imagination from a real frame; ``imagine`` thereafter
    needs only states, actions, and keys. A sampled state feeds the next step,
    so samples can affect the entire trajectory. No pixel noise is added.

    Args:
        observation_shape: [frames, height, width], or [frames, height, width, 3].
        num_actions: Number of discrete actions.
        d_model: Deterministic history and image-embedding width.
        d_state: Even Mamba state width, at least four.
        expand: Mamba inner-width multiplier.
        headdim: Head width dividing expand*d_model.
        mimo_rank: Mamba rank; one selects SISO.
        encoder_stages: Explicit convolutional stages, mirrored by the decoder.
        dtype: Compute dtype; parameters, state, logits, and predictions are float32.
        num_layers: Number of independent Mamba layers.
        d_intermediate: Stack MLP width; None chooses its automatic width.
        stochastic_size: Number of independent categorical latent variables.
        stochastic_classes: Number of classes per variable, at least two.
        unimix: Uniform probability mixture in [0, 1), preventing zero mass.
    """

    observation_shape: tuple[int, ...]
    num_actions: int
    d_model: int = 128
    d_state: int = 64
    expand: int = 2
    headdim: int = 32
    mimo_rank: int = 1
    encoder_stages: ConvStages = DEFAULT_STAGES
    dtype: jax.typing.DTypeLike = jnp.float32
    num_layers: int = 4
    d_intermediate: int | None = None
    stochastic_size: int = 16
    stochastic_classes: int = 16
    unimix: float = 0.01

    @nn.nowrap
    def _make_dynamics(self) -> Mamba3Stack:
        """Return an unbound feature-processing stack with independent layers."""
        return Mamba3Stack(
            d_model=self.d_model,
            num_layers=self.num_layers,
            d_intermediate=self.d_intermediate,
            d_state=self.d_state,
            expand=self.expand,
            headdim=self.headdim,
            mimo_rank=self.mimo_rank,
            dtype=self.dtype,
            parent=None,
        )

    def setup(self) -> None:
        """Validate sizes and register shared convolution, dynamics, and latent heads."""
        if not isinstance(self.observation_shape, tuple) or not self.observation_shape:
            raise ValueError("observation_shape must be a nonempty tuple of positive integers")
        for size in (*self.observation_shape, self.num_actions, self.stochastic_size, self.stochastic_classes):
            chex.assert_type(size, int)
            chex.assert_scalar_positive(size)
        if len(self.observation_shape) not in (3, 4):
            raise ValueError("observation_shape must be [frames, height, width] or [frames, height, width, 3]")
        if len(self.observation_shape) == 4 and self.observation_shape[-1] != 3:
            raise ValueError("RGB observations must have exactly three color channels")
        chex.assert_scalar_in(self.unimix, 0.0, 1.0)
        if self.unimix == 1:
            raise ValueError("unimix must be less than one")
        if self.stochastic_classes < 2:
            raise ValueError("stochastic_classes must be at least two")
        self.encoder = ConvObservationEncoder(stages=self.encoder_stages, embedding_size=self.d_model, dtype=self.dtype)
        self.action_embedding = nn.Embed(self.num_actions, self.d_model, dtype=self.dtype)
        self.input_projection = nn.Dense(self.d_model, dtype=self.dtype)
        self.dynamics = self._make_dynamics()
        self.prior_hidden = nn.Dense(self.d_model, dtype=self.dtype)
        self.prior_head = nn.Dense(self.stochastic_size * self.stochastic_classes, dtype=self.dtype)
        self.posterior_hidden = nn.Dense(self.d_model, dtype=self.dtype)
        self.posterior_head = nn.Dense(self.stochastic_size * self.stochastic_classes, dtype=self.dtype)
        self.decoder_hidden = nn.Dense(self.d_model, dtype=self.dtype)
        self.observation_decoder = ConvObservationDecoder(
            self.observation_shape, stages=self.encoder_stages, dtype=self.dtype
        )
        self.outcome_head = nn.Dense(2, dtype=self.dtype)

    @nn.nowrap
    def initial_carry(self, batch_size: int) -> WorldModelState:
        """Return zero history and uninitialized categorical state for batch_size streams."""
        chex.assert_type(batch_size, int)
        chex.assert_scalar_positive(batch_size)
        return WorldModelState(
            self._make_dynamics().initial_carry(batch_size),
            jnp.zeros((batch_size, self.d_model), jnp.float32),
            jnp.zeros((batch_size, self.stochastic_size, self.stochastic_classes), jnp.float32),
            jnp.zeros(batch_size, jnp.bool_),
        )

    @nn.nowrap
    def _check_state(self, state: WorldModelState) -> None:
        chex.assert_rank(state.deter, 2)
        chex.assert_trees_all_equal_shapes_and_dtypes(state, self.initial_carry(state.deter.shape[0]))

    @nn.nowrap
    def features(self, state: WorldModelState) -> jax.Array:
        """Return float32 [batch, d_model + stochastic_size*stochastic_classes]."""
        self._check_state(state)
        return jnp.concatenate((state.deter, state.stoch.reshape((state.deter.shape[0], -1))), axis=-1)

    def encode(self, observation: jax.Array) -> jax.Array:
        """Encode uint8 [B, *image] or [T, B, *image] into float32 [..., d_model]."""
        rank = len(self.observation_shape)
        chex.assert_rank(observation, {rank + 1, rank + 2})
        chex.assert_type(observation, jnp.uint8)
        leading = observation.shape[:-rank]
        chex.assert_shape(observation, (*leading, *self.observation_shape))
        chex.assert_scalar_positive(leading[-1])
        images = observation.reshape((math.prod(leading), *self.observation_shape))
        return self.encoder(images).astype(jnp.float32).reshape((*leading, self.d_model))

    def _logits(self, raw: jax.Array) -> jax.Array:
        chex.assert_shape(raw, (None, self.stochastic_size * self.stochastic_classes))
        chex.assert_type(raw, self.dtype)
        raw = raw.astype(jnp.float32).reshape((-1, self.stochastic_size, self.stochastic_classes))
        log_probs = jax.nn.log_softmax(raw, axis=-1)
        if self.unimix == 0:
            return log_probs
        return jnp.logaddexp(log_probs + math.log1p(-self.unimix), math.log(self.unimix / self.stochastic_classes))

    def _prior(self, deter: jax.Array) -> jax.Array:
        chex.assert_shape(deter, (None, self.d_model))
        chex.assert_type(deter, jnp.float32)
        return self._logits(self.prior_head(nn.silu(self.prior_hidden(deter))))

    def _posterior(self, deter: jax.Array, embedding: jax.Array) -> jax.Array:
        chex.assert_shape((deter, embedding), (None, self.d_model))
        chex.assert_equal_shape((deter, embedding))
        chex.assert_type((deter, embedding), jnp.float32)
        return self._logits(
            self.posterior_head(nn.silu(self.posterior_hidden(jnp.concatenate((deter, embedding), -1))))
        )

    def _sample(self, logits: jax.Array, key: jax.Array) -> jax.Array:
        chex.assert_shape(logits, (None, self.stochastic_size, self.stochastic_classes))
        chex.assert_type(logits, jnp.float32)
        check_keys(key, ())
        probs = jax.nn.softmax(logits, axis=-1)
        sample = jax.nn.one_hot(
            jax.random.categorical(key, logits, axis=-1), self.stochastic_classes, dtype=jnp.float32
        )
        return sample + (probs - jax.lax.stop_gradient(probs))

    def _reset(self, state: WorldModelState, starts: jax.Array) -> WorldModelState:
        self._check_state(state)
        chex.assert_shape(starts, (state.deter.shape[0],))
        chex.assert_type(starts, jnp.bool_)
        return jax.tree.map(
            lambda x: jnp.where(starts.reshape((starts.shape[0],) + (1,) * (x.ndim - 1)), jnp.zeros_like(x), x), state
        )

    def condition(
        self, observation: jax.Array, state: WorldModelState, episode_starts: jax.Array, key: jax.Array
    ) -> WorldModelState:
        """Infer a posterior state from uint8 [B, *image], existing history, and key.

        Bool [B] episode_starts clears every state leaf before conditioning.
        Returns a state aligned with observation; no action or transition occurs.
        """
        state = self._reset(state, episode_starts)
        chex.assert_shape(observation, (state.deter.shape[0], *self.observation_shape))
        logits = self._posterior(state.deter, self.encode(observation))
        return state.replace(stoch=self._sample(logits, key), initialized=jnp.ones_like(state.initialized))

    def _advance(self, state: WorldModelState, action: jax.Array) -> WorldModelState:
        self._check_state(state)
        sc = ShapeChecker(B=state.deter.shape[0], D=self.d_model)
        sc.check(action, "B")
        chex.assert_type(action, int)
        inputs = jnp.concatenate((state.stoch.reshape((action.shape[0], -1)), self.action_embedding(action)), axis=-1)
        projected = self.input_projection(inputs).astype(jnp.float32)
        sc.check(projected, "BD", jnp.float32)
        starts = jnp.zeros_like(state.initialized)
        sc.check(starts, "B", jnp.bool_)
        memory, deter = self.dynamics.step(projected, state.memory, starts)
        sc.check(deter, "BD", self.dtype)
        return state.replace(
            memory=memory, deter=deter.astype(jnp.float32), initialized=jnp.ones_like(state.initialized)
        )

    def transition(self, state: WorldModelState, action: jax.Array, key: jax.Array) -> WorldModelState:
        """Sample the next prior state given current state, integer [B] actions, and key.

        Seed state with condition before the first action. Returned sampled state
        must be reused for the following action to preserve temporal dependence.
        """
        state = self._advance(state, action)
        return state.replace(stoch=self._sample(self._prior(state.deter), key))

    def decode(self, features: jax.Array) -> Prediction:
        """Decode float32 [B, F] or [T, B, F] features, F = d_model + S*C.

        Returns pixels with the model's image shape and reward/terminal logits
        with matching leading axes. Pixels/rewards are unbounded float32.
        """
        chex.assert_rank(features, {2, 3})
        chex.assert_shape(
            features, (*features.shape[:-1], self.d_model + self.stochastic_size * self.stochastic_classes)
        )
        chex.assert_type(features, jnp.float32)
        observation = self.observation_decoder(features.reshape((-1, features.shape[-1]))).astype(jnp.float32)
        observation = observation.reshape((*features.shape[:-1], *self.observation_shape))
        outcomes = self.outcome_head(nn.silu(self.decoder_hidden(features))).astype(jnp.float32)
        return Prediction(observation, outcomes[..., 0], outcomes[..., 1])

    def imagine(self, state: WorldModelState, action: jax.Array, key: jax.Array) -> tuple[WorldModelState, Prediction]:
        """Return next sampled prior state and its decoded frame/outcomes.

        Args are current state, integer actions [B], and a fresh scalar JAX key.
        No real observation or posterior is used; sampling affects future steps.
        """
        state = self.transition(state, action, key)
        return state, self.decode(self.features(state))

    def _observe_step(self, state: WorldModelState, inputs: ObserveInputs) -> tuple[WorldModelState, LatentOutputs]:
        embedding, action, next_embedding, starts, key = inputs
        state = self._reset(state, starts)
        initial_key, posterior_key = jax.random.split(key)
        initial = self._sample(self._posterior(state.deter, embedding), initial_key)
        state = state.replace(stoch=jnp.where(state.initialized[:, None, None], state.stoch, initial))
        state = self._advance(state, action)
        prior = self._prior(state.deter)
        posterior = self._posterior(state.deter, next_embedding)
        state = state.replace(stoch=self._sample(posterior, posterior_key))
        return state, (self.features(state), prior, posterior)

    def observe(
        self,
        observation: jax.Array,
        action: jax.Array,
        next_observation: jax.Array,
        sample_keys: jax.Array,
        carry: WorldModelState | None = None,
        episode_starts: jax.Array | None = None,
    ) -> tuple[WorldModelState, ObserveOutput]:
        """Infer posterior states and reconstruct the next frames of real transitions.

        Args:
            observation: Uint8 [B, *image] or time-major [T, B, *image].
            action: Integer [B] or [T, B], action taken from observation.
            next_observation: Same shape/dtype as observation; final frames before
                reset must be preserved rather than replaced with reset frames.
            sample_keys: Scalar JAX key for a single step, or T independent keys
                for a sequence. Supplying the same per-step keys makes chunked
                and unchunked processing equivalent.
            carry: State aligned with observation[0], or None for zero history.
                Pass the final state between contiguous chunks; it retains the
                posterior sample, avoiding resampling at chunk boundaries.
            episode_starts: Bool matching action; resets history before that step.

        Returns:
            Final WorldModelState and ObserveOutput with all leading batch/time
            axes. Gradients flow through recurrence unless callers detach carry.
        """
        chex.assert_equal_shape((observation, next_observation))
        chex.assert_type((observation, next_observation), jnp.uint8)
        embedding, next_embedding = self.encode(observation), self.encode(next_observation)
        chex.assert_shape(action, embedding.shape[:-1])
        chex.assert_type(action, int)
        single = embedding.ndim == 2
        check_keys(sample_keys, () if single else (action.shape[0],))
        if carry is None:
            carry = self.initial_carry(embedding.shape[-2])
        self._check_state(carry)
        chex.assert_shape(carry.deter, (embedding.shape[-2], self.d_model))
        starts = jnp.zeros(action.shape, jnp.bool_) if episode_starts is None else episode_starts
        chex.assert_shape(starts, action.shape)
        chex.assert_type(starts, jnp.bool_)
        inputs = (embedding, action, next_embedding, starts, sample_keys)
        if single:
            carry, (features, prior, posterior) = self._observe_step(carry, inputs)
        else:
            scan = nn.scan(
                MambaWorldModel._observe_step,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=0,
                out_axes=0,
            )
            carry, (features, prior, posterior) = scan(self, carry, inputs)
        return carry, ObserveOutput(features, prior, posterior, self.decode(features))

    def __call__(
        self,
        observation: jax.Array,
        action: jax.Array,
        next_observation: jax.Array,
        sample_keys: jax.Array,
    ) -> tuple[WorldModelState, ObserveOutput]:
        """Initialize/use the full model from real transitions; see observe for shapes."""
        return self.observe(observation, action, next_observation, sample_keys)
