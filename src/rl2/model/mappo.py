"""Tagged MAPPO models sharing team/agent recurrent sequence execution."""

from math import prod, sqrt
from typing import Annotated, Any, Literal, cast

import jax
import jax.numpy as jnp
from flax import linen as nn
from pydantic import BeforeValidator, ConfigDict, Field
from pydantic.dataclasses import dataclass

from rl2.actions import (
    CategoricalPolicy,
    DiscreteSpace,
    HierarchicalMultiCategoricalPolicy,
    HierarchicalMultiDiscreteSpace,
    MultiAgentActionSpace,
    Policy,
)
from rl2.envs.lux_env import LuxObservation
from rl2.lstm import LSTMStack, LSTMStackCarry
from rl2.model.actor_critic import ActorCriticOutput, BaseActorCritic
from rl2.model.lux import LuxFeatures, LuxModel

type RecurrentCarry = LSTMStackCarry
type Observation = jax.Array | LuxObservation[jax.Array]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LSTMConfig:
    type: Literal["lstm"] = "lstm"
    hidden_size: Annotated[int, Field(gt=0, strict=True)] = 128
    num_layers: Annotated[int, Field(gt=0, strict=True)] = 1
    intermediate_size: Annotated[int, Field(ge=0, strict=True)] = 0


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class MLPEncoderConfig:
    type: Literal["mlp"] = "mlp"
    embedding_size: Annotated[int, Field(gt=0, strict=True)] = 128


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class GenericModelConfig:
    type: Literal["generic"] = "generic"
    encoder: MLPEncoderConfig = MLPEncoderConfig()
    recurrent: LSTMConfig = LSTMConfig()


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LuxModelConfig:
    type: Literal["lux"] = "lux"
    embedding_size: Annotated[int, Field(gt=0, strict=True)] = 128
    channels: Annotated[int, Field(gt=0, multiple_of=2, strict=True)] = 16
    recurrent: LSTMConfig = LSTMConfig()


def _default_model_type(value: Any) -> Any:
    if isinstance(value, dict):
        return {"type": "generic", **value}
    return value


type ModelConfig = Annotated[
    Annotated[GenericModelConfig | LuxModelConfig, Field(discriminator="type")],
    BeforeValidator(_default_model_type),
]


class ActorCritic(BaseActorCritic[Observation, RecurrentCarry, Policy]):
    """Shared local observation encoder/LSTM with separate policy and value heads."""

    action_space: MultiAgentActionSpace
    model: ModelConfig
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
    def initial_carry(self, actor_count: int) -> RecurrentCarry:
        return self._make_recurrent().initial_carry(actor_count)  # leaves: (B, H)

    def setup(self) -> None:
        if isinstance(self.model, LuxModelConfig):
            self.lux = LuxModel(
                embedding_size=self.model.embedding_size,
                channels=self.model.channels,
                action_space=cast(HierarchicalMultiDiscreteSpace, self.action_space),
                dtype=self.dtype,
                name="lux",
            )
        else:
            self.encoder = nn.Dense(features=self.model.encoder.embedding_size, dtype=self.dtype, name="encoder")
        self.recurrent = self._make_recurrent(parent=self)
        self.recurrent_input = nn.Dense(features=self.model.recurrent.hidden_size, dtype=self.dtype, name="lstm_input")
        if isinstance(self.model, LuxModelConfig):
            return
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

    def _encode(self, observation: Observation) -> tuple[jax.Array, LuxFeatures | None]:
        """Return (*S, H) recurrent inputs and optional Lux spatial/entity features."""
        if isinstance(self.model, LuxModelConfig):
            features = self.lux.encode(cast(LuxObservation[jax.Array], observation))  # leaves: (*S, ...)
            x = nn.relu(features.global_features)  # (*S, F)
        else:
            features = None
            x = nn.relu(self.encoder(observation))  # (*S, F)
        return self.recurrent_input(x).astype(self.dtype), features  # (*S, H); leaves: (*S, ...)

    def _heads(
        self,
        x: jax.Array,  # (*S, H)
        actions_mask: jax.Array,  # (*S, *C, A)
        features: LuxFeatures | None,  # leaves: (*S, ...)
        observation: Observation,  # leaves: (*S, ...)
    ) -> tuple[Policy, jax.Array]:  # logits: (*S, *C, A); value: (*S,)
        if isinstance(self.model, LuxModelConfig):
            return self.lux.heads(
                x, cast(LuxFeatures, features), cast(LuxObservation[jax.Array], observation), actions_mask
            )
        x = x.astype(self.dtype)  # (*S, H), S is (B,) or (B, T)
        policy = nn.relu(self.policy_norm(self.policy_hidden(x)))  # (*S, 512)
        critic = nn.relu(self.value_norm(self.value_hidden(x)))  # (*S, 512)
        # Heads use the compute dtype; float32 outputs keep PPO loss arithmetic precise.
        logits = self.policy_output(policy).astype(jnp.float32)  # (*S, D*L) or (*S, A); L is raw logits per component
        distribution: Policy
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
        observation: Observation,  # leaves: (B, T, ...)
        model_carry: RecurrentCarry,  # leaves: (B, H)
        episode_start: jax.Array,  # (B, T)
        actions_mask: jax.Array,  # (B, T, *C, A)
    ) -> ActorCriticOutput[RecurrentCarry, Policy]:  # carry leaves: (B, H); logits: (B, T, *C, A); value: (B, T)
        """Process observation sequences, resetting memory at episode starts."""
        x, features = self._encode(observation)  # (B, T, H)
        # The shared LSTM scans time-major inputs; MAPPO exposes batch-major arrays.
        recurrent_x, recurrent_starts = jax.tree.map(
            lambda leaf: jnp.swapaxes(leaf, 0, 1), (x, episode_start)
        )  # (T, B, H); (T, B)
        model_carry, recurrent_x = self.recurrent(
            recurrent_x, model_carry, recurrent_starts
        )  # model_carry leaves: (B, H); recurrent_x: (T, B, H)
        x = jnp.swapaxes(recurrent_x, 0, 1)  # (B, T, H)
        policy, value = self._heads(x, actions_mask, features, observation)  # logits: (B, T, *C, A); (B, T)
        return ActorCriticOutput(carry=model_carry, policy=policy, value=value)

    def step(
        self,
        observation: Observation,  # leaves: (B, ...)
        model_carry: RecurrentCarry,  # leaves: (B, H)
        episode_start: jax.Array,  # (B,)
        actions_mask: jax.Array,  # (B, *C, A)
    ) -> ActorCriticOutput[RecurrentCarry, Policy]:  # carry leaves: (B, H); logits: (B, *C, A); value: (B,)
        """Process one observation per sequence, resetting memory at episode starts."""
        x, features = self._encode(observation)  # (B, H)
        model_carry, x = self.recurrent.step(x, model_carry, episode_start)  # model_carry leaves: (B, H); x: (B, H)
        policy, value = self._heads(x, actions_mask, features, observation)  # logits: (B, *C, A); (B,)
        return ActorCriticOutput(carry=model_carry, policy=policy, value=value)
