import jax
import pytest

from rl2.sequence_model import ARSequenceModel, BDSequenceModel, RecurrentSequenceModel


class CallOnly(ARSequenceModel[None]):
    def __call__(self, x: jax.Array, x_len: jax.Array, carry: None = None) -> tuple[None, jax.Array]:
        raise NotImplementedError("Test stub; only method completeness is exercised")


class StepOnly(ARSequenceModel[None]):
    def step(self, x: jax.Array, x_active: jax.Array, carry: None = None) -> tuple[None, jax.Array]:
        raise NotImplementedError("Test stub; only method completeness is exercised")


@pytest.mark.parametrize("model_type", [ARSequenceModel, CallOnly, StepOnly])
def test_incomplete_sequence_models_cannot_be_instantiated(model_type: type[ARSequenceModel[None]]) -> None:
    with pytest.raises(TypeError, match="abstract"):
        model_type()


def test_subclass_implementing_both_methods_can_be_instantiated() -> None:
    class BothMethods(CallOnly, StepOnly):
        pass

    assert isinstance(BothMethods(), ARSequenceModel)


def test_bidirectional_base_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BDSequenceModel()


def test_bidirectional_subclass_only_requires_call() -> None:
    class CallOnlyBidirectional(BDSequenceModel):
        def __call__(self, x: jax.Array, x_len: jax.Array) -> jax.Array:
            raise NotImplementedError("Test stub; only method completeness is exercised")

    model = CallOnlyBidirectional()
    assert isinstance(model, BDSequenceModel)
    assert not hasattr(model, "step")


class RecurrentCallOnly(RecurrentSequenceModel[jax.Array]):
    def __call__(self, x: jax.Array, carry: jax.Array, episode_starts: jax.Array) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError("Test stub; only method completeness is exercised")


class RecurrentStepOnly(RecurrentSequenceModel[jax.Array]):
    def step(self, x: jax.Array, carry: jax.Array, episode_starts: jax.Array) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError("Test stub; only method completeness is exercised")


@pytest.mark.parametrize("model_type", [RecurrentSequenceModel, RecurrentCallOnly, RecurrentStepOnly])
def test_incomplete_recurrent_models_cannot_be_instantiated(
    model_type: type[RecurrentSequenceModel[jax.Array]],
) -> None:
    with pytest.raises(TypeError, match="abstract"):
        model_type()


def test_recurrent_subclass_without_initial_carry_cannot_be_instantiated() -> None:
    class BothMethods(RecurrentCallOnly, RecurrentStepOnly):
        pass

    with pytest.raises(TypeError, match="initial_carry"):
        BothMethods()


def test_recurrent_subclass_implementing_all_methods_can_be_instantiated() -> None:
    class AllMethods(RecurrentCallOnly, RecurrentStepOnly):
        def initial_carry(self, num_envs: int) -> jax.Array:
            raise NotImplementedError("Test stub; only method completeness is exercised")

    assert isinstance(AllMethods(), RecurrentSequenceModel)
