from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rl2.mamba3 import Mamba3Carry
from rl2.observation_encoder import ConvObservationEncoder
from rl2.wm import MambaWorldModel, Prediction


def assert_tree_close(actual: Any, expected: Any, *, atol: float = 3e-6) -> None:
    chex.assert_trees_all_equal_shapes_and_dtypes(actual, expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=atol)


@pytest.mark.parametrize("rank,rgb", [(1, False), (2, True)])
def test_sequence_steps_chunks_and_base_interface_agree(rank: int, rgb: bool) -> None:
    shape = (2, 4, 4, 3) if rgb else (2, 4, 4)
    model = MambaWorldModel(shape, 3, d_model=8, d_state=4, headdim=4, mimo_rank=rank, encoder_channels=(4,))
    obs = jax.random.randint(jax.random.key(0), (5, 2, *shape), 0, 256, dtype=jnp.uint8)
    actions = jnp.arange(10, dtype=jnp.int32).reshape(5, 2) % 3
    # Initializing the original interface creates every parameter used by the
    # recurrent methods, even when initialization only sees a single step.
    variables = model.init(jax.random.key(1), obs[0], actions[0])
    observe = jax.jit(partial(model.apply, variables, method=model.observe))
    final, latents, prediction = observe(obs, actions)
    chex.assert_shape(latents, (5, 2, 8))
    chex.assert_shape(prediction.observation, obs.shape)
    chex.assert_shape((prediction.reward, prediction.termination_logits), (5, 2))
    chex.assert_type(jax.tree.leaves((final, latents, prediction)), jnp.float32)

    def step(
        carry: Mamba3Carry, inputs: tuple[jax.Array, jax.Array]
    ) -> tuple[Mamba3Carry, tuple[jax.Array, Prediction]]:
        observation, action = inputs
        chex.assert_shape(observation, (2, *shape))
        chex.assert_type(observation, jnp.uint8)
        chex.assert_shape(action, (2,))
        chex.assert_type(action, jnp.int32)
        carry, latent, output = observe(observation, action, carry)
        return carry, (latent, output)

    stepped, (step_latents, step_prediction) = jax.lax.scan(step, model.initial_carry(2), (obs, actions))
    assert_tree_close((stepped, step_latents, step_prediction), (final, latents, prediction))
    carry, first_latents, first = observe(obs[:2], actions[:2])
    carry, last_latents, last = observe(obs[2:], actions[2:], carry)
    joined = jax.tree.map(lambda a, b: jnp.concatenate((a, b)), first, last)
    assert_tree_close((carry, jnp.concatenate((first_latents, last_latents)), joined), (final, latents, prediction))
    # Eager and fused JIT kernels can accumulate slightly different float32 rounding.
    assert_tree_close(model.apply(variables, obs, actions), (latents, prediction), atol=5e-6)
    encoded = model.apply(variables, obs, method=model.encode)
    # The shared encoder receives intact images; only time/batch are flattened.
    encoder = ConvObservationEncoder(encoder_channels=(4,), embedding_size=8)
    expected_encoding = encoder.apply({"params": variables["params"]["encoder"]}, obs.reshape((10, *shape)))
    assert_tree_close(encoded, expected_encoding.reshape((5, 2, 8)))
    transitioned = model.apply(variables, encoded, actions, method=model.transition)
    assert_tree_close(transitioned, latents)
    assert_tree_close(model.apply(variables, latents, method=model.decode), prediction)
    empty_carry, empty_latents, empty_prediction = observe(obs[:0], actions[:0], final)
    assert_tree_close(empty_carry, final)
    chex.assert_shape(empty_latents, (0, 2, 8))
    chex.assert_shape(empty_prediction.observation, (0, 2, *shape))


def test_episode_resets_and_causality() -> None:
    model = MambaWorldModel((1, 4, 4), 2, d_model=8, d_state=4, headdim=4, encoder_channels=(4,))
    obs = jax.random.randint(jax.random.key(2), (6, 2, 1, 4, 4), 0, 256, dtype=jnp.uint8)
    actions = jnp.arange(12, dtype=jnp.int32).reshape(6, 2) % 2
    variables = model.init(jax.random.key(3), obs, actions)
    observe = jax.jit(partial(model.apply, variables, method=model.observe))
    starts = jnp.zeros((6, 2), dtype=jnp.bool_).at[3, 0].set(True)
    final, latents, predictions = observe(obs, actions, episode_starts=starts)
    fresh = observe(obs[3:, :1], actions[3:, :1])
    continuous = observe(obs[:, 1:], actions[:, 1:])
    assert_tree_close(
        (jax.tree.map(lambda x: x[:1], final), latents[3:, :1], jax.tree.map(lambda x: x[3:, :1], predictions)),
        fresh,
    )
    assert_tree_close(
        (jax.tree.map(lambda x: x[1:], final), latents[:, 1:], jax.tree.map(lambda x: x[:, 1:], predictions)),
        continuous,
    )
    assert_tree_close(observe(obs[0], actions[0], final, jnp.ones(2, dtype=jnp.bool_)), observe(obs[0], actions[0]))
    _, baseline, _ = observe(obs, actions)
    _, changed, _ = observe(obs.at[3:].set(100), actions.at[3:].set(1 - actions[3:]))
    np.testing.assert_array_equal(changed[:3], baseline[:3])
    # Changing the action with the observation held fixed affects predictions.
    _, other_action, _ = observe(obs[0], 1 - actions[0])
    assert not np.allclose(other_action, baseline[0])
    # Incoming memory matters when there is no reset.
    _, continued, _ = observe(obs[0], actions[0], final)
    assert not np.allclose(continued, baseline[0])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_imagination_gradients_and_training(dtype: jax.typing.DTypeLike) -> None:
    model = MambaWorldModel((1, 4, 4), 2, d_model=8, d_state=4, headdim=4, dtype=dtype, encoder_channels=(4,))
    obs = jax.random.randint(jax.random.key(4), (4, 2, 1, 4, 4), 0, 256, dtype=jnp.uint8)
    actions = jnp.arange(8, dtype=jnp.int32).reshape(4, 2) % 2
    variables = model.init(jax.random.key(5), obs, actions)
    carry, latent, _ = model.apply(variables, obs[0], actions[0], method=model.observe)

    def rollout_step(
        state: tuple[Mamba3Carry, jax.Array], action: jax.Array
    ) -> tuple[tuple[Mamba3Carry, jax.Array], Prediction]:
        chex.assert_shape(action, (2,))
        chex.assert_type(action, jnp.int32)
        carry, latent = state
        chex.assert_shape(latent, (2, 8))
        chex.assert_type(latent, jnp.float32)
        carry, latent, prediction = model.apply(variables, latent, action, carry, method=model.imagine)
        return (carry, latent), prediction

    (_, imagined), predictions = jax.jit(partial(jax.lax.scan, rollout_step))((carry, latent), actions)
    chex.assert_shape(imagined, (2, 8))
    chex.assert_shape(predictions.observation, obs.shape)
    chex.assert_type(jax.tree.leaves(predictions), jnp.float32)
    assert all(leaf.dtype == jnp.float32 for leaf in carry)

    def loss_fn(params: optax.Params) -> jax.Array:
        _, _, output = model.apply({"params": params}, obs, actions, method=model.observe)
        observation_loss = jnp.mean(jnp.square(output.observation - 0.5))
        reward_loss = jnp.mean(jnp.square(output.reward - 1.0))
        terminal_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(output.termination_logits, jnp.zeros((4, 2))))
        return observation_loss + reward_loss + terminal_loss

    loss, gradients = jax.jit(jax.value_and_grad(loss_fn))(variables["params"])
    assert np.isfinite(loss)
    for component in gradients.values():
        assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(component))
        assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(component))
    updates = jax.tree.map(lambda gradient: -0.01 * gradient, gradients)
    updated = optax.apply_updates(variables["params"], updates)
    assert float(loss_fn(updated)) < float(loss)


def test_invalid_shapes_and_dtypes() -> None:
    model = MambaWorldModel((1, 4, 4), 2, d_model=8, d_state=4, headdim=4, encoder_channels=(4,))
    obs, actions = jnp.zeros((2, 1, 4, 4), jnp.uint8), jnp.zeros(2, dtype=jnp.int32)
    variables = model.init(jax.random.key(6), obs, actions)
    observe = partial(model.apply, variables, method=model.observe)
    with pytest.raises(AssertionError):
        observe(jnp.zeros((2, 1, 4, 5), jnp.uint8), actions)
    with pytest.raises(AssertionError):
        observe(obs.astype(jnp.float32), actions)
    with pytest.raises(AssertionError):
        observe(obs, actions.astype(jnp.float32))
    with pytest.raises(AssertionError):
        observe(obs, actions[:, None])
    with pytest.raises(AssertionError):
        observe(obs, actions, episode_starts=jnp.zeros(2))
    with pytest.raises(AssertionError):
        observe(obs, actions, episode_starts=jnp.zeros((1, 2), dtype=jnp.bool_))
    with pytest.raises(AssertionError):
        observe(obs, actions, model.initial_carry(1))
    with pytest.raises(AssertionError):
        observe(obs, actions, jax.tree.map(lambda x: x.astype(jnp.bfloat16), model.initial_carry(2)))
    with pytest.raises(AssertionError):
        model.apply(variables, jnp.zeros((2, 7)), actions, method=model.imagine)


@pytest.mark.parametrize("shape,num_actions", [((0, 4, 4), 2), ((1, 4, 4), 0), ((3.5, 4, 4), 2)])
def test_invalid_configuration(shape: tuple[int | float, ...], num_actions: int) -> None:
    model = MambaWorldModel(shape, num_actions, d_model=8, d_state=4, headdim=4, encoder_channels=(4,))
    with pytest.raises(AssertionError):
        model.init(jax.random.key(0), jnp.zeros((2, 1, 4, 4), jnp.uint8), jnp.zeros(2, dtype=jnp.int32))


@pytest.mark.parametrize("shape", [(3,), (4, 4), (1, 4, 4, 2)])
def test_invalid_image_layout(shape: tuple[int, ...]) -> None:
    model = MambaWorldModel(shape, 2, d_model=8, d_state=4, headdim=4, encoder_channels=(4,))
    with pytest.raises(ValueError):
        model.init(jax.random.key(0), jnp.zeros((2, *shape), jnp.uint8), jnp.zeros(2, jnp.int32))
