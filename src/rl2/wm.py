"""Flax Linen world-model interface and a deterministic Mamba-3 baseline."""

import math

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from rl2.mamba3 import Mamba3, Mamba3Carry
from rl2.observation_encoder import ConvObservationEncoder


@struct.dataclass
class Prediction:
    """Predicted observation and transition outcomes, stored as a JAX pytree.

    Observation shape matches the model's observation space. Reward and
    termination logits have only the leading batch dimensions; termination
    denotes an environment terminal state, excluding time-limit truncation.

    Attributes:
        observation: Predicted observation with shape
            ``[*leading, *observation_shape]``, where ``leading`` is batch or
            time and batch. Values use the implementation's target scale and
            are not clipped. MambaWorldModel targets pixels normalized to [0, 1].
        reward: Reward for the transition into this observation, with shape
            ``[*leading]``.
        termination_logits: Unnormalized terminal-state scores with shape
            ``[*leading]``. Apply ``jax.nn.sigmoid`` to obtain probabilities.
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
        """Encode the current observation into a latent state.

        Args:
            observation: Preprocessed observations with leading batch
                dimensions and an implementation-defined observation shape.

        Returns:
            Latent states preserving the observation's leading dimensions;
            the latent representation and dtype are implementation-defined.
        """
        raise NotImplementedError

    def transition(self, latent: jax.Array, action: jax.Array) -> jax.Array:
        """Predict the next latent state after taking an action.

        Args:
            latent: Current latent states, as produced by ``encode`` or a
                previous transition.
            action: Actions sharing the latent states' leading dimensions.
                Action encoding and dtype are implementation-defined.

        Returns:
            Predicted next latent states with the same leading dimensions.
        """
        raise NotImplementedError

    def decode(self, latent: jax.Array) -> Prediction:
        """Predict an observation and incoming reward/termination from a state.

        Args:
            latent: Latent states representing predicted next observations.

        Returns:
            A ``Prediction`` containing observations, rewards, and termination
            logits with the same leading dimensions as ``latent``.
        """
        raise NotImplementedError

    def __call__(self, observation: jax.Array, action: jax.Array) -> tuple[jax.Array, Prediction]:
        """Return the next latent state and predictions for this transition.

        Args:
            observation: Current observations accepted by ``encode``.
            action: Actions taken from those observations, accepted by
                ``transition`` and aligned with the observations' leading axes.

        Returns:
            ``(next_latent, prediction)`` after encoding the observations,
            applying the actions, and decoding the resulting states. For
            ``MambaWorldModel``, this call starts with zero recurrent history;
            use ``observe`` or ``imagine`` to retain history across calls.
        """
        latent = self.transition(self.encode(observation), action)
        return latent, self.decode(latent)


type RecurrentPrediction = tuple[Mamba3Carry, jax.Array, Prediction]


class MambaWorldModel(WorldModel):
    """Convolutional observation encoder and MLP decoder around Mamba dynamics.

    Observations must be uint8 Atari frames, with shape
    ``[batch, *observation_shape]`` or ``[time, batch, *observation_shape]``.
    The shared convolutional encoder normalizes pixels and produces hidden
    features. Mamba processes only action-conditioned hidden features, with no
    knowledge of image layout or pixel normalization. Decoded observations are
    predictions on the normalized [0, 1] target scale, without clipping.
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

        model = MambaWorldModel(observation_shape=(1, 84, 84), num_actions=2)
        obs = jnp.zeros((8, 1, 84, 84), dtype=jnp.uint8)
        actions = jnp.zeros(8, dtype=jnp.int32)
        variables = model.init(jax.random.key(0), obs, actions)
        carry, latent, prediction = model.apply(variables, obs, actions, method=model.observe)
        carry, latent, prediction = model.apply(variables, latent, actions, carry, method=model.imagine)

    Parameters, returned latents, predictions, and carry remain float32;
    ``dtype`` controls the internal projection precision.

    Args:
        observation_shape: Positive dimensions ``[frames, height, width]`` for
            grayscale or ``[frames, height, width, 3]`` for RGB, excluding time
            and batch. Stacked frames become channels inside the encoder.
        num_actions: Positive number of discrete actions, indexed from zero.
        d_model: Positive latent width and Mamba input/output width.
        d_state: Mamba recurrent state width; must be even and at least four
            for the mixer's default rotary fraction.
        expand: Positive multiplier defining the Mamba inner width as
            ``expand * d_model``.
        headdim: Positive width of each Mamba head; must divide the inner width.
        mimo_rank: Positive Mamba projection rank. One selects SISO; larger
            values select MIMO.
        encoder_channels: Positive channel widths for the shared convolutional
            encoder's residual stages. Its output embedding has width d_model.
        dtype: Internal projection dtype: float32, bfloat16, or float16.
        encoder_max_flattened_size: Maximum encoder features before projection;
            None disables automatic extra downsampling stages.
    """

    observation_shape: tuple[int, ...]
    num_actions: int
    d_model: int = 128
    d_state: int = 64
    expand: int = 2
    headdim: int = 32
    mimo_rank: int = 1
    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)
    dtype: jax.typing.DTypeLike = jnp.float32
    encoder_max_flattened_size: int | None = 8192

    @nn.nowrap
    def _make_mixer(self) -> Mamba3:
        """Return an unbound Mamba-3 mixer configured from this model's fields."""
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
        """Validate observation/action dimensions and register the Linen layers."""
        if not isinstance(self.observation_shape, tuple) or not self.observation_shape:
            raise ValueError("observation_shape must be a nonempty tuple of positive integers")
        for size in (*self.observation_shape, self.num_actions):
            chex.assert_type(size, int)
            chex.assert_scalar_positive(size)
        if len(self.observation_shape) not in (3, 4):
            raise ValueError("observation_shape must be [frames, height, width] or [frames, height, width, 3]")
        if len(self.observation_shape) == 4 and self.observation_shape[-1] != 3:
            raise ValueError("RGB observations must have exactly three color channels")
        self.encoder = ConvObservationEncoder(
            encoder_channels=self.encoder_channels,
            embedding_size=self.d_model,
            dtype=self.dtype,
            max_flattened_size=self.encoder_max_flattened_size,
        )
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
        """Allocate zero Mamba history without initializing model parameters.

        Args:
            batch_size: Positive number of independent environments or streams.

        Returns:
            A zero-filled float32 ``Mamba3Carry`` containing the recurrent
            state, previous key/value, and rotary angle. Each leaf has leading
            dimension ``batch_size`` and trailing dimensions set by the mixer.
        """
        chex.assert_type(batch_size, int)
        chex.assert_scalar_positive(batch_size)
        return self._make_mixer().initial_carry(batch_size)

    @nn.nowrap
    def _check_latent(self, latent: jax.Array) -> None:
        """Validate latent shape and dtype without modifying the array.

        Args:
            latent: Floating-point array shaped ``[batch, d_model]`` or
                ``[time, batch, d_model]``, with a positive batch size.

        Returns:
            None. Chex raises ``AssertionError`` for invalid inputs.
        """
        chex.assert_rank(latent, {2, 3})
        chex.assert_shape(latent, (*latent.shape[:-1], self.d_model))
        chex.assert_type(latent, float)
        chex.assert_scalar_positive(latent.shape[-2])

    def encode(self, observation: jax.Array) -> jax.Array:
        """Encode batched observations or time-major sequences into latents.

        Args:
            observation: Uint8 pixel observations shaped
                ``[batch, *observation_shape]`` or
                ``[time, batch, *observation_shape]``. The convolutional encoder
                normalizes pixels internally; do not divide by 255 beforehand.

        Returns:
            Float32 latents shaped ``[batch, d_model]`` or
            ``[time, batch, d_model]``. Each observation is encoded independently
            without reading or updating recurrent history.
        """
        observation_rank = len(self.observation_shape)
        chex.assert_rank(observation, {observation_rank + 1, observation_rank + 2})
        chex.assert_type(observation, jnp.uint8)
        leading = observation.shape[:-observation_rank]
        chex.assert_shape(observation, (*leading, *self.observation_shape))
        chex.assert_scalar_positive(leading[-1])
        # Merge time and batch only; spatial dimensions stay intact for Conv.
        images = observation.reshape((math.prod(leading), *self.observation_shape))
        latent = self.encoder(images).astype(jnp.float32).reshape((*leading, self.d_model))
        self._check_latent(latent)
        return latent

    def _transition(
        self,
        latent: jax.Array,
        action: jax.Array,
        carry: Mamba3Carry | None = None,
        episode_starts: jax.Array | None = None,
    ) -> tuple[Mamba3Carry, jax.Array]:
        """Apply action-conditioned recurrent dynamics to supplied latents.

        Args:
            latent: Floating-point current states shaped ``[batch, d_model]``
                or ``[time, batch, d_model]``.
            action: Integer action IDs in ``[0, num_actions)``, shaped
                ``[batch]`` or ``[time, batch]`` to match ``latent``.
            carry: Float32 Mamba history matching the model and batch size.
                ``None`` initializes zero history. Gradients flow through a
                supplied carry unless the caller applies ``stop_gradient``.
            episode_starts: Boolean reset mask with the same shape as
                ``action``. True clears that stream's history before processing
                the input; it does not replace the supplied latent. ``None``
                means no resets.

        Returns:
            ``(final_carry, next_latent)`` with float32 leaves. The carry holds
            history after the final input; latents have the same shape as the
            input and include predictions for every supplied timestep.
        """
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
        """Predict next latents from zero history; use ``imagine`` to keep carry.

        Args:
            latent: Floating-point states shaped ``[batch, d_model]`` or
                ``[time, batch, d_model]``. Sequence inputs provide each
                timestep's state explicitly.
            action: Integer IDs in ``[0, num_actions)``, shaped ``[batch]`` or
                ``[time, batch]`` to match ``latent``.

        Returns:
            Float32 next latents with the same shape as ``latent``. History is
            propagated within a sequence but discarded at the end of the call.
        """
        _, next_latent = self._transition(latent, action)
        return next_latent

    def decode(self, latent: jax.Array) -> Prediction:
        """Decode next-observation, reward, and termination-logit predictions.

        Args:
            latent: Floating-point predicted states shaped ``[batch, d_model]``
                or ``[time, batch, d_model]``.

        Returns:
            A float32 ``Prediction``. Observations have shape
            ``[*leading, *observation_shape]``; rewards and termination logits
            have shape ``[*leading]``, where ``leading`` is ``[batch]`` or
            ``[time, batch]``. All heads are unbounded; termination probabilities
            require a sigmoid. Rewards and termination refer to the transition
            into the decoded observation.
        """
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
        """Predict from actual observations, preserving history across calls.

        Args:
            observation: Uint8 pixel observations shaped
                ``[batch, *observation_shape]`` or
                ``[time, batch, *observation_shape]``.
            action: Integer IDs in ``[0, num_actions)``, shaped ``[batch]`` or
                ``[time, batch]``. At index t, the action is taken from
                observation t, and the prediction targets observation t+1.
            carry: Float32 history matching this model and batch size, or
                ``None`` to start with zero history. It is not detached from
                the gradient graph automatically.
            episode_starts: Boolean mask matching ``action``; True resets
                history before processing that step's dynamics. Supply the new
                episode's initial observation at reset positions. ``None``
                means no resets.

        Returns:
            ``(final_carry, next_latent, prediction)`` with float32 leaves.
            Latents have shape ``[batch, d_model]`` or
            ``[time, batch, d_model]``; prediction shapes follow ``decode``.
            A sequence uses actual observations at every step (teacher forcing)
            and returns predictions for all steps plus only the final carry.
        """
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

        Args:
            latent: Floating-point current states shaped ``[batch, d_model]``
                or ``[time, batch, d_model]``. For a rollout, seed with
                ``encode`` and then use the previous predicted latent.
            action: Integer IDs in ``[0, num_actions)``, shaped ``[batch]`` or
                ``[time, batch]`` to match ``latent``.
            carry: Float32 history matching this model and batch size, or
                ``None`` for zero history. Use the previous returned carry to
                continue a rollout; detach it explicitly for truncated BPTT.
            episode_starts: Boolean mask matching ``action``; True clears
                history before that input. It does not replace ``latent``;
                use a newly encoded observation when starting an episode.
                ``None`` means no resets.

        Returns:
            ``(final_carry, next_latent, prediction)`` with float32 leaves.
            Next latents have the same shape as ``latent``; predictions follow
            ``decode`` and describe the observation and outcomes after each
            action. Predicted termination is an output only: it does not reset
            history or stop the rollout automatically.
        """
        carry, next_latent = self._transition(latent, action, carry, episode_starts)
        return carry, next_latent, self.decode(next_latent)
