"""Typed Flax interface for recurrent actor-critic models."""

from abc import ABC, abstractmethod
from typing import NamedTuple

import jax
from flax import linen as nn

from rl2.actions import Policy


class ActorCriticOutput[CarryT, PolicyT: Policy](NamedTuple):
    """Outgoing memory, action distribution, and one value per observation."""

    carry: CarryT  # model-specific pytree; array leaves retain batch axis B
    policy: PolicyT  # batch shape (B,) for step or (B, T) for sequences
    value: jax.Array  # (B,) for step or (B, T) for sequences


class BaseActorCritic[ObservationT, CarryT, PolicyT: Policy](nn.Module, ABC):
    """Batch-major sequence and single-step policy/value evaluation.

    ObservationT and CarryT describe model-specific pytrees; PolicyT implements
    the action distribution. Use inherited Flax init/apply for parameterized
    methods. initial_carry must also work on an unbound, uninitialized module.
    B denotes batch size, T sequence length, C action component axes, A choices.

    True episode_start entries reset memory before the corresponding observation.
    Scanning step over time must reproduce __call__ outputs and final carry,
    including internal episode resets. Values and policies share batch axes;
    policies mask unavailable commands and reduce over the complete action event.
    """

    @nn.nowrap
    @abstractmethod
    def initial_carry(self, actor_count: int) -> CarryT:
        """Return initial memory for B=actor_count independent sequences."""

    @abstractmethod
    def __call__(
        self,
        observation: ObservationT,  # leaves: (B, T, ...)
        model_carry: CarryT,  # model-specific memory with batch axis B
        episode_start: jax.Array,  # (B, T) bool
        actions_mask: jax.Array,  # (B, T, *C, A) bool
    ) -> ActorCriticOutput[CarryT, PolicyT]:
        """Return final carry, (B, T) policy batch, and (B, T) values."""

    @abstractmethod
    def step(
        self,
        observation: ObservationT,  # leaves: (B, ...)
        model_carry: CarryT,  # model-specific memory with batch axis B
        episode_start: jax.Array,  # (B,) bool
        actions_mask: jax.Array,  # (B, *C, A) bool
    ) -> ActorCriticOutput[CarryT, PolicyT]:
        """Return updated carry, (B,) policy batch, and (B,) values."""
