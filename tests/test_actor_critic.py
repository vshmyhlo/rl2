"""Required actor-critic methods and concrete MAPPO implementations."""

import jax
import pytest

from rl2.actions import CategoricalPolicy, DiscreteSpace, HierarchicalMultiDiscreteSpace
from rl2.model.actor_critic import ActorCriticOutput, BaseActorCritic
from rl2.model.mappo import ActorCritic, GenericModelConfig, LuxModelConfig


class CarryOnly(BaseActorCritic[jax.Array, None, CategoricalPolicy]):
    def initial_carry(self, actor_count: int) -> None:
        return None


class SequenceOnly(BaseActorCritic[jax.Array, None, CategoricalPolicy]):
    def __call__(
        self,
        observation: jax.Array,  # (B, T, O)
        model_carry: None,
        episode_start: jax.Array,  # (B, T)
        actions_mask: jax.Array,  # (B, T, A)
    ) -> ActorCriticOutput[None, CategoricalPolicy]:
        raise NotImplementedError("Only abstract method completeness is exercised")


class StepOnly(BaseActorCritic[jax.Array, None, CategoricalPolicy]):
    def step(
        self,
        observation: jax.Array,  # (B, O)
        model_carry: None,
        episode_start: jax.Array,  # (B,)
        actions_mask: jax.Array,  # (B, A)
    ) -> ActorCriticOutput[None, CategoricalPolicy]:
        raise NotImplementedError("Only abstract method completeness is exercised")


class MissingCarry(SequenceOnly, StepOnly):
    pass


class MissingSequence(CarryOnly, StepOnly):
    pass


class MissingStep(CarryOnly, SequenceOnly):
    pass


@pytest.mark.parametrize(
    "model_type,missing",
    [(MissingCarry, "initial_carry"), (MissingSequence, "__call__"), (MissingStep, "step")],
)
def test_incomplete_actor_critics_cannot_be_instantiated(
    model_type: type[BaseActorCritic[jax.Array, None, CategoricalPolicy]], missing: str
) -> None:
    with pytest.raises(TypeError, match=missing):
        model_type()


@pytest.mark.parametrize(
    "model",
    [
        ActorCritic(action_space=DiscreteSpace(3), model=GenericModelConfig()),
        ActorCritic(
            action_space=HierarchicalMultiDiscreteSpace(nvec=(14,), group_sizes=(1, 4, 9)), model=LuxModelConfig()
        ),
    ],
    ids=["generic", "lux"],
)
def test_mappo_variants_implement_actor_critic_interface(model: ActorCritic) -> None:
    assert isinstance(model, BaseActorCritic)
    carry = model.initial_carry(2)  # leaves: (B=2, H=128), available without parameter initialization
    assert all(leaf.shape == (2, 128) for leaf in jax.tree.leaves(carry))
