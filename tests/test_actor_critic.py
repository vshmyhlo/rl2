"""Required actor-critic methods and concrete MAPPO implementations."""

import jax
import pytest

from rl2.actions import CategoricalPolicy, DiscreteSpace, HierarchicalMultiDiscreteSpace
from rl2.model.actor_critic import ActorCriticOutput, BaseActorCritic
from rl2.model.config import GenericModelConfig, LuxModelConfig
from rl2.model.generic import GenericActorCritic
from rl2.model.lux import LuxActorCritic
from rl2.train_mappo import make_actor_critic


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
        GenericActorCritic(action_space=DiscreteSpace(3), model=GenericModelConfig()),
        LuxActorCritic(
            action_space=HierarchicalMultiDiscreteSpace(nvec=(14,), group_sizes=(1, 4, 9)), model=LuxModelConfig()
        ),
    ],
    ids=["generic", "lux"],
)
def test_mappo_variants_implement_actor_critic_interface(model: GenericActorCritic | LuxActorCritic) -> None:
    assert isinstance(model, BaseActorCritic)
    carry = model.initial_carry(2)  # leaves: (B=2, H=128), available without parameter initialization
    assert all(leaf.shape == (2, 128) for leaf in jax.tree.leaves(carry))


@pytest.mark.parametrize(
    "action_space,config,expected_type",
    [
        (DiscreteSpace(3), GenericModelConfig(), GenericActorCritic),
        (HierarchicalMultiDiscreteSpace(nvec=(14,), group_sizes=(1, 4, 9)), LuxModelConfig(), LuxActorCritic),
    ],
    ids=["generic", "lux"],
)
def test_factory_selects_concrete_subclass(
    action_space: DiscreteSpace | HierarchicalMultiDiscreteSpace,
    config: GenericModelConfig | LuxModelConfig,
    expected_type: type[GenericActorCritic] | type[LuxActorCritic],
) -> None:
    model = make_actor_critic(action_space=action_space, model=config)
    assert type(model) is expected_type
    assert model.model == config and model.action_space == action_space


def test_lux_factory_rejects_scalar_action_space() -> None:
    with pytest.raises(TypeError, match="hierarchical multi-discrete"):
        make_actor_critic(action_space=DiscreteSpace(3), model=LuxModelConfig())
