"""Generic MAPPO actor-critic for flat observations."""

from math import prod, sqrt

import jax
import jax.numpy as jnp
from flax import linen as nn

from rl2.actions import (
    CategoricalPolicy,
    DiscreteSpace,
    HierarchicalMultiCategoricalPolicy,
    HierarchicalMultiDiscreteSpace,
    MultiAgentActionSpace,
)
from rl2.lstm import LSTMStack, LSTMStackCarry
from rl2.model.actor_critic import ActorCriticOutput, BaseActorCritic
from rl2.model.config import GenericModelConfig

type GenericPolicy = CategoricalPolicy | HierarchicalMultiCategoricalPolicy


class GenericActorCritic(BaseActorCritic[jax.Array, LSTMStackCarry, GenericPolicy]):
    """Shared local observation encoder/LSTM with separate policy and value heads."""

    action_space: MultiAgentActionSpace
    model: GenericModelConfig
    dtype: jax.typing.DTypeLike = jnp.float32

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
        self.policy_output = nn.Dense(
            features=(
                self.action_space.n
                if isinstance(self.action_space, DiscreteSpace)
                else prod(self.action_space.logits_shape)
            ),
            kernel_init=nn.initializers.orthogonal(0.01),
            name="policy_output",
            dtype=self.dtype,
        )
        self.value_output = nn.Dense(
            features=1, kernel_init=nn.initializers.orthogonal(1.0), name="value_output", dtype=self.dtype
        )

    def _encode(self, observation: jax.Array) -> jax.Array:  # (*S, O) -> (*S, H)
        x = nn.relu(self.encoder(observation))  # (*S, F)
        return self.recurrent_input(x).astype(self.dtype)  # (*S, H)

    def _heads(
        self,
        x: jax.Array,  # (*S, H)
        actions_mask: jax.Array,  # (*S, *C, A)
    ) -> tuple[GenericPolicy, jax.Array]:  # logits: (*S, *C, A); value: (*S,)
        x = x.astype(self.dtype)  # (*S, H), S is (B,) or (B, T)
        policy = nn.relu(self.policy_norm(self.policy_hidden(x)))  # (*S, 512)
        critic = nn.relu(self.value_norm(self.value_hidden(x)))  # (*S, 512)
        # Heads use the compute dtype; float32 outputs keep PPO loss arithmetic precise.
        logits = self.policy_output(policy).astype(jnp.float32)  # (*S, D*L) or (*S, A); L is raw logits per component
        distribution: GenericPolicy
        if isinstance(self.action_space, HierarchicalMultiDiscreteSpace):
            logits = logits.reshape((*x.shape[:-1], *self.action_space.logits_shape))  # (*S, D, L)
            distribution = HierarchicalMultiCategoricalPolicy(
                raw_logits=logits, actions_mask=actions_mask, group_sizes=self.action_space.group_sizes
            )
        else:
            distribution = CategoricalPolicy(logits=jnp.where(actions_mask, logits, -1e10))
        value = self.value_output(critic).squeeze(-1).astype(jnp.float32)  # (*S,)
        return distribution, value  # logits: (*S, *C, A); value: (*S,)

    def __call__(
        self,
        observation: jax.Array,  # (B, T, O)
        model_carry: LSTMStackCarry,  # leaves: (B, H)
        episode_start: jax.Array,  # (B, T)
        actions_mask: jax.Array,  # (B, T, *C, A)
    ) -> ActorCriticOutput[LSTMStackCarry, GenericPolicy]:  # carry leaves: (B, H); logits: (B, T, *C, A); value: (B, T)
        """Process observation sequences, resetting memory at episode starts."""
        x = self._encode(observation)  # (B, T, H)
        # The shared LSTM scans time-major inputs; MAPPO exposes batch-major arrays.
        recurrent_x, recurrent_starts = jax.tree.map(
            lambda leaf: jnp.swapaxes(leaf, 0, 1), (x, episode_start)
        )  # (T, B, H); (T, B)
        model_carry, recurrent_x = self.recurrent(
            recurrent_x, model_carry, recurrent_starts
        )  # model_carry leaves: (B, H); recurrent_x: (T, B, H)
        x = jnp.swapaxes(recurrent_x, 0, 1)  # (B, T, H)
        policy, value = self._heads(x, actions_mask)  # logits: (B, T, *C, A); (B, T)
        return ActorCriticOutput(carry=model_carry, policy=policy, value=value)

    def step(
        self,
        observation: jax.Array,  # (B, O)
        model_carry: LSTMStackCarry,  # leaves: (B, H)
        episode_start: jax.Array,  # (B,)
        actions_mask: jax.Array,  # (B, *C, A)
    ) -> ActorCriticOutput[LSTMStackCarry, GenericPolicy]:  # carry leaves: (B, H); logits: (B, *C, A); value: (B,)
        """Process one observation per sequence, resetting memory at episode starts."""
        x = self._encode(observation)  # (B, H)
        model_carry, x = self.recurrent.step(x, model_carry, episode_start)  # model_carry leaves: (B, H); x: (B, H)
        policy, value = self._heads(x, actions_mask)  # logits: (B, *C, A); (B,)
        return ActorCriticOutput(carry=model_carry, policy=policy, value=value)
