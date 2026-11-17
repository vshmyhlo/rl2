"""Generic MAPPO actor-critic for flat observations.

Continuous policies use bounded truncated Gaussians, with log probabilities and
differential entropies summed over coordinates. Their action masks are (*S, 0);
policy leaves are (*S, D) instead of categorical logits (*S, *C, A).
"""

from math import prod, sqrt

import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.actions import (
    CategoricalPolicy,
    ContinuousSpace,
    DiscreteSpace,
    HierarchicalMultiCategoricalPolicy,
    HierarchicalMultiDiscreteSpace,
    MultiAgentActionSpace,
    TruncatedGaussianPolicy,
)
from rl2.lstm import LSTMStack, LSTMStackCarry
from rl2.model.actor_critic import ActorCriticOutput, BaseActorCritic
from rl2.model.config import GenericModelConfig

type GenericPolicy = CategoricalPolicy | HierarchicalMultiCategoricalPolicy | TruncatedGaussianPolicy


class GenericActorCritic(BaseActorCritic[jax.Array, LSTMStackCarry, GenericPolicy]):
    """Shared local observation encoder/LSTM with separate policy and value heads."""

    action_space: MultiAgentActionSpace
    model: GenericModelConfig
    dtype: jax.typing.DTypeLike | None = None

    @nn.nowrap
    def _make_recurrent(self, *, parent: nn.Module | None = None) -> LSTMStack:
        return LSTMStack(
            features=self.model.recurrent.hidden_size,
            num_layers=self.model.recurrent.num_layers,
            intermediate_size=self.model.recurrent.intermediate_size,
            dtype=self.dtype,
            name="lstm",
            parent=parent,
        )

    @nn.nowrap
    def initial_carry(self, actor_count: int) -> LSTMStackCarry:
        return self._make_recurrent().initial_carry(actor_count)  # leaves: (B, H)

    def setup(self) -> None:
        self.encoder = nn.Dense(features=self.model.encoder.embedding_size, dtype=self.dtype, name="encoder")
        self.recurrent = self._make_recurrent(parent=self)
        self.recurrent_input = nn.Dense(features=self.model.recurrent.hidden_size, dtype=self.dtype, name="lstm_input")
        init = nn.initializers.orthogonal(sqrt(2))
        self.policy_hidden = nn.Dense(features=512, kernel_init=init, name="policy_hidden", dtype=self.dtype)
        self.policy_norm = nn.LayerNorm(name="policy_norm", dtype=self.dtype)
        self.value_hidden = nn.Dense(features=512, kernel_init=init, name="value_hidden", dtype=self.dtype)
        self.value_norm = nn.LayerNorm(name="value_norm", dtype=self.dtype)
        if isinstance(self.action_space, ContinuousSpace):
            policy_size = self.action_space.size
            self.policy_log_std = self.param("policy_log_std", nn.initializers.zeros, (policy_size,))  # (D,)
        elif isinstance(self.action_space, DiscreteSpace):
            policy_size = self.action_space.n
        else:
            policy_size = prod(self.action_space.logits_shape)
        self.policy_output = nn.Dense(
            features=policy_size,
            kernel_init=nn.initializers.orthogonal(0.01),
            name="policy_output",
            dtype=self.dtype,
        )
        self.value_output = nn.Dense(
            features=1, kernel_init=nn.initializers.orthogonal(1.0), name="value_output", dtype=self.dtype
        )

    def _encode(self, observation: jax.Array) -> jax.Array:  # (*S, O) -> (*S, H)
        x = nn.relu(self.encoder(observation))  # (*S, F)
        return jnp.asarray(self.recurrent_input(x), dtype=self.dtype)  # (*S, H)

    def _heads(
        self,
        x: jax.Array,  # (*S, H)
        actions_mask: jax.Array,  # (*S, *C, A) discrete or (*S, 0) continuous
    ) -> tuple[GenericPolicy, jax.Array]:  # policy leaves: (*S, *C, A) or (*S, D); value: (*S,)
        x = jnp.asarray(x, dtype=self.dtype)  # (*S, H), S is (B,) or (B, T)
        policy = nn.relu(self.policy_norm(self.policy_hidden(x)))  # (*S, 512)
        critic = nn.relu(self.value_norm(self.value_hidden(x)))  # (*S, 512)
        # Heads use the compute dtype; float32 outputs keep PPO loss arithmetic precise.
        logits = self.policy_output(policy).astype(jnp.float32)  # (*S, D*L), (*S, A), or (*S, D) continuous means
        distribution: GenericPolicy
        if isinstance(self.action_space, ContinuousSpace):
            low, high = jnp.asarray((self.action_space.low, self.action_space.high), jnp.float32)  # each (D,)
            scale = high / 2 - low / 2  # (D,)
            mean = high / 2 + low / 2 + scale * jnp.tanh(logits)  # (*S, D)
            log_std = jnp.broadcast_to(
                jnp.clip(self.policy_log_std, -20.0, 2.0) + jnp.log(scale), logits.shape
            )  # (*S, D)
            distribution = TruncatedGaussianPolicy(mean=mean, log_std=log_std, action_space=self.action_space)
        elif isinstance(self.action_space, HierarchicalMultiDiscreteSpace):
            logits = logits.reshape((*x.shape[:-1], *self.action_space.logits_shape))  # (*S, D, L)
            distribution = HierarchicalMultiCategoricalPolicy(
                raw_logits=logits, actions_mask=actions_mask, group_sizes=self.action_space.group_sizes
            )
        else:
            distribution = CategoricalPolicy(logits=jnp.where(actions_mask, logits, -1e10))
        value = self.value_output(critic).squeeze(-1).astype(jnp.float32)  # (*S,)
        return distribution, value  # policy leaves: (*S, *C, A) or (*S, D); value: (*S,)

    def __call__(
        self,
        observation: jax.Array,  # (B, T, O)
        model_carry: LSTMStackCarry,  # leaves: (B, H)
        episode_start: jax.Array,  # (B, T)
        actions_mask: jax.Array,  # (B, T, *C, A) discrete or (B, T, 0) continuous
    ) -> tuple[
        ActorCriticOutput[GenericPolicy], LSTMStackCarry
    ]:  # carry leaves: (B, H); policy leaves: (B, T, *C, A) or (B, T, D); value: (B, T)
        """Process observation sequences, resetting memory at episode starts."""
        x = self._encode(observation)  # (B, T, H)
        model_carry, x = self.recurrent(x, model_carry, episode_start)  # (B, H) leaves; (B, T, H)
        policy, value = self._heads(x, actions_mask)  # policy leaves: (B, T, *C, A) or (B, T, D); (B, T)
        return ActorCriticOutput(policy=policy, value=value), model_carry

    def step(
        self,
        observation: jax.Array,  # (B, O)
        model_carry: LSTMStackCarry,  # leaves: (B, H)
        episode_start: jax.Array,  # (B,)
        actions_mask: jax.Array,  # (B, *C, A) discrete or (B, 0) continuous
    ) -> tuple[
        ActorCriticOutput[GenericPolicy], LSTMStackCarry
    ]:  # carry leaves: (B, H); policy leaves: (B, *C, A) or (B, D); value: (B,)
        """Process one observation per sequence, resetting memory at episode starts."""
        x = self._encode(observation)  # (B, H)
        model_carry, x = self.recurrent.step(x, model_carry, episode_start)  # model_carry leaves: (B, H); x: (B, H)
        policy, value = self._heads(x, actions_mask)  # policy leaves: (B, *C, A) or (B, D); (B,)
        return ActorCriticOutput(policy=policy, value=value), model_carry
