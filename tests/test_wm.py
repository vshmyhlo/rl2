from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn

from rl2.observation_encoder import ConvObservationEncoder, ConvStage
from rl2.wm import (
    LSTMWorldModel,
    MambaWorldModel,
    ObserveInputs,
    WorldModelState,
    categorical_entropy,
    categorical_kl,
    latent_kl_losses,
)


def small_model(model_class: type[MambaWorldModel] = MambaWorldModel, **kwargs: Any) -> MambaWorldModel:
    settings = {
        "observation_shape": (1, 4, 4),
        "num_actions": 3,
        "d_model": 8,
        "num_layers": 1,
        "d_state": 4,
        "headdim": 4,
        "encoder_stages": (ConvStage(4, blocks=1),),
        "stochastic_size": 4,
        "stochastic_classes": 4,
    }
    return model_class(**(settings | kwargs))


def assert_tree_close(actual: Any, expected: Any, *, atol: float = 3e-6) -> None:
    chex.assert_trees_all_equal_shapes_and_dtypes(actual, expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=atol)


@pytest.mark.parametrize(
    "model_class,rank,rgb,depth",
    [(MambaWorldModel, 1, False, 2), (MambaWorldModel, 2, True, 1), (LSTMWorldModel, 1, False, 2)],
)
def test_sequence_steps_chunks_and_initialization_agree(
    model_class: type[MambaWorldModel], rank: int, rgb: bool, depth: int
) -> None:
    shape = (2, 4, 4, 3) if rgb else (2, 4, 4)
    model = small_model(model_class, observation_shape=shape, num_layers=depth, mimo_rank=rank)
    obs = jax.random.randint(jax.random.key(0), (5, 2, *shape), 0, 256, dtype=jnp.uint8)
    actions = jnp.arange(8, dtype=jnp.int32).reshape(4, 2) % 3
    keys = jax.random.split(jax.random.key(1), 4)
    # Single-step initialization must create parameters used by all other APIs.
    variables = model.init(jax.random.key(2), obs[0], actions[0], obs[1], keys[0])
    observe = jax.jit(partial(model.apply, variables, method=model.observe))
    final, output = observe(obs[:-1], actions, obs[1:], keys)
    chex.assert_shape(output.features, (4, 2, 24))
    chex.assert_shape((output.prior_logits, output.posterior_logits), (4, 2, 4, 4))
    chex.assert_shape(output.prediction.observation, (4, 2, *shape))
    chex.assert_shape((output.prediction.reward, output.prediction.termination_logits), (4, 2))
    chex.assert_type(jax.tree.leaves(output), jnp.float32)
    assert len(final.memory) == depth
    np.testing.assert_array_equal(final.stoch.sum(-1), 1)
    assert set(np.unique(final.stoch)) <= {0, 1}
    for index in range(depth):
        if model_class is LSTMWorldModel:
            chex.assert_shape(final.memory[index], (2, 8))
        else:
            assert f"layers_{index}" in variables["params"]["dynamics"]
    current = model.initial_carry(2)
    outputs = []
    for t in range(4):
        current, result = observe(obs[t], actions[t], obs[t + 1], keys[t], current)
        outputs.append(result)
    stacked = jax.tree.map(lambda *x: jnp.stack(x), *outputs)
    assert_tree_close((current, stacked), (final, output))
    current, first = observe(obs[:2], actions[:2], obs[1:3], keys[:2])
    current, last = observe(obs[2:4], actions[2:], obs[3:], keys[2:], current)
    joined = jax.tree.map(lambda a, b: jnp.concatenate((a, b)), first, last)
    assert_tree_close((current, joined), (final, output))
    encoded = model.apply(variables, obs, method=model.encode)
    encoder = ConvObservationEncoder(stages=(ConvStage(4, blocks=1),), embedding_size=8)
    expected = encoder.apply({"params": variables["params"]["encoder"]}, obs.reshape((10, *shape)))
    assert_tree_close(encoded, expected.reshape((5, 2, 8)))
    assert_tree_close(model.apply(variables, output.features, method=model.decode), output.prediction)


@pytest.mark.parametrize("model_class", [MambaWorldModel, LSTMWorldModel])
def test_episode_resets_causality_and_no_future_leakage(model_class: type[MambaWorldModel]) -> None:
    model = small_model(model_class)
    obs = jax.random.randint(jax.random.key(3), (5, 2, 1, 4, 4), 0, 256, dtype=jnp.uint8)
    actions = jnp.arange(8, dtype=jnp.int32).reshape(4, 2) % 3
    keys = jax.random.split(jax.random.key(4), 4)
    variables = model.init(jax.random.key(5), obs[0], actions[0], obs[1], keys[0])
    observe = jax.jit(partial(model.apply, variables, method=model.observe))
    starts = jnp.zeros((4, 2), jnp.bool_).at[2].set(True)
    final, output = observe(obs[:-1], actions, obs[1:], keys, episode_starts=starts)
    fresh, fresh_output = observe(obs[2:-1], actions[2:], obs[3:], keys[2:])
    assert_tree_close((final, jax.tree.map(lambda x: x[2:], output)), (fresh, fresh_output))
    # Changing the target affects this step's posterior, but not its prior.
    _, changed = observe(obs[:-1], actions, obs[1:].at[2:].set(255), keys)
    _, baseline = observe(obs[:-1], actions, obs[1:], keys)
    np.testing.assert_allclose(changed.prior_logits[:3], baseline.prior_logits[:3], atol=1e-6)
    assert not np.allclose(changed.posterior_logits[2], baseline.posterior_logits[2])
    np.testing.assert_array_equal(changed.features[:2], baseline.features[:2])
    # Selective reset clears history only for the selected environment.
    mask = jnp.array([True, False])
    continued, _ = observe(obs[0], actions[0], obs[1], keys[0], final)
    reset, _ = observe(obs[0], actions[0], obs[1], keys[0], final, mask)
    zero, _ = observe(obs[0], actions[0], obs[1], keys[0])
    assert_tree_close(jax.tree.map(lambda x: x[:1], reset), jax.tree.map(lambda x: x[:1], zero))
    assert_tree_close(jax.tree.map(lambda x: x[1:], reset), jax.tree.map(lambda x: x[1:], continued))


@pytest.mark.parametrize("model_class", [MambaWorldModel, LSTMWorldModel])
def test_sampling_reproducibility_and_effect_on_future_dynamics(model_class: type[MambaWorldModel]) -> None:
    model = small_model(model_class)
    obs = jnp.full((2, 1, 4, 4), 120, jnp.uint8)
    actions = jnp.zeros(2, jnp.int32)
    key = jax.random.key(6)
    variables = model.init(key, obs, actions, obs, key)
    initial = model.apply(variables, obs, model.initial_carry(2), jnp.ones(2, jnp.bool_), key, method=model.condition)
    imagine = jax.jit(partial(model.apply, variables, method=model.imagine))
    first, image = imagine(initial, actions, jax.random.key(7))
    assert_tree_close((first, image), imagine(initial, actions, jax.random.key(7)))
    second, other = imagine(initial, actions, jax.random.key(8))
    assert not np.array_equal(first.stoch, second.stoch)
    assert not np.allclose(image.observation, other.observation)
    np.testing.assert_array_equal(first.deter, second.deter)
    future_a, _ = imagine(first, actions, key)
    future_b, _ = imagine(second, actions, key)
    assert not np.allclose(future_a.deter, future_b.deter)
    assert not np.allclose(imagine(initial, 1 + actions, key)[0].deter, first.deter)
    # Imagination cannot consult the posterior or encoder.
    poisoned = jax.tree.map(lambda x: x, variables)
    for name in ("posterior_hidden", "posterior_head", "encoder"):
        poisoned["params"][name] = jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), poisoned["params"][name])
    assert_tree_close(model.apply(poisoned, initial, actions, jax.random.key(7), method=model.imagine), (first, image))


@pytest.mark.parametrize(
    "model_class,dtype",
    [(MambaWorldModel, jnp.float32), (MambaWorldModel, jnp.bfloat16), (LSTMWorldModel, jnp.bfloat16)],
)
def test_straight_through_gradients_and_compute_dtypes(
    model_class: type[MambaWorldModel], dtype: jax.typing.DTypeLike
) -> None:
    model = small_model(model_class, dtype=dtype)
    obs = jax.random.randint(jax.random.key(9), (3, 2, 1, 4, 4), 0, 256, dtype=jnp.uint8)
    actions = jnp.array([[0, 1], [1, 2]], jnp.int32)
    keys = jax.random.split(jax.random.key(10), 2)
    variables = model.init(jax.random.key(11), obs[0], actions[0], obs[1], keys[0])

    def capture(module: nn.Module, method: str) -> bool:
        return method == "__call__" and isinstance(module, (nn.Conv, nn.Dense, nn.Embed))

    (_, out), intermediates = model.apply(
        variables, obs[0], actions[0], obs[1], keys[0], capture_intermediates=capture, mutable=["intermediates"]
    )
    chex.assert_type(jax.tree.leaves(intermediates), dtype)
    chex.assert_type(jax.tree.leaves((variables["params"], out)), jnp.float32)

    def loss_fn(params: optax.Params) -> jax.Array:
        _, result = model.apply({"params": params}, obs[:-1], actions, obs[1:], keys)
        prediction = result.prediction
        return (
            jnp.square(prediction.observation - obs[1:] / 255).mean()
            + jnp.square(prediction.reward - 1).mean()
            + optax.sigmoid_binary_cross_entropy(prediction.termination_logits, jnp.zeros((2, 2))).mean()
            + categorical_kl(result.posterior_logits, result.prior_logits).mean()
        )

    loss, gradients = jax.jit(jax.value_and_grad(loss_fn))(variables["params"])
    assert np.isfinite(loss)
    for name, component in gradients.items():
        assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(component)), name
        assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(component)), name
    for layer in gradients["dynamics"].values():
        if "mixer" in layer:
            assert np.any(np.asarray(layer["mixer"]["in_proj"]["kernel"]) != 0)


def test_kl_values_gradient_routing_and_free_nats() -> None:
    q = jnp.log(jnp.array([[[0.8, 0.2]], [[0.6, 0.4]]], jnp.float32))
    p = jnp.log(jnp.array([[[0.4, 0.6]], [[0.5, 0.5]]], jnp.float32))
    expected = np.sum(np.exp(q) * (q - p), axis=(-2, -1))
    np.testing.assert_allclose(categorical_kl(q, p), expected, rtol=1e-5)
    np.testing.assert_allclose(categorical_kl(q, q), 0, atol=1e-6)
    for index in (0, 1):
        grad_q, grad_p = jax.grad(lambda a, b, index=index: latent_kl_losses(a, b, 0.0)[index], argnums=(0, 1))(q, p)
        np.testing.assert_array_equal(grad_q if index == 0 else grad_p, 0)
        assert np.any(np.asarray(grad_p if index == 0 else grad_q) != 0)
    np.testing.assert_allclose(latent_kl_losses(q, p, 1.0), (1.0, 1.0))
    floored = jax.grad(lambda a: sum(latent_kl_losses(a, p, 1.0)))(q)
    np.testing.assert_array_equal(floored, 0)


def test_invalid_dtypes_and_keys() -> None:
    model = small_model()
    obs, actions = jnp.zeros((2, 1, 4, 4), jnp.uint8), jnp.zeros(2, jnp.int32)
    key = jax.random.key(12)
    variables = model.init(key, obs, actions, obs, key)
    observe = partial(model.apply, variables, method=model.observe)
    with pytest.raises(AssertionError):
        observe(obs.astype(jnp.float32), actions, obs, key)
    with pytest.raises(AssertionError):
        observe(obs, actions.astype(jnp.float32), obs, key)
    with pytest.raises(AssertionError):
        observe(obs, actions, obs, key, episode_starts=jnp.zeros(2))
    with pytest.raises(ValueError, match="single key"):
        observe(obs, actions, obs, jax.random.split(key, 2))


@pytest.mark.parametrize(
    "settings",
    [
        {"observation_shape": (0, 4, 4)},
        {"num_actions": 0},
        {"stochastic_size": 0},
        {"stochastic_classes": 1},
        {"unimix": -0.1},
        {"unimix": 1.0},
        {"unimix": float("nan")},
        {"observation_shape": (3,)},
        {"observation_shape": (1, 4, 4, 2)},
    ],
)
def test_invalid_configuration(settings: dict[str, Any]) -> None:
    model = small_model(**settings)
    key = jax.random.key(0)
    obs = jnp.zeros((2, 1, 4, 4), jnp.uint8)
    with pytest.raises((AssertionError, ValueError)):
        model.init(key, obs, jnp.zeros(2, jnp.int32), obs, key)


def test_gradients_cross_sampled_states_and_stop_at_episode_resets() -> None:
    model = small_model()
    key = jax.random.key(20)
    obs = jnp.ones((1, 1, 4, 4), jnp.uint8)
    variables = model.init(key, obs, jnp.zeros(1, jnp.int32), obs, key)
    keys = jax.random.split(key, 3)
    embeddings = jax.random.normal(key, (3, 1, 8))

    def last_hidden(module: MambaWorldModel, carry: WorldModelState, inputs: ObserveInputs) -> jax.Array:
        scan = nn.scan(MambaWorldModel._observe_step, variable_broadcast="params", split_rngs={"params": False})
        final, _ = scan(module, carry, inputs)
        return final.deter[0, 0]

    def loss_fn(next_embeddings: jax.Array, starts: jax.Array) -> jax.Array:
        chex.assert_shape(next_embeddings, (3, 1, 8))
        chex.assert_type(next_embeddings, jnp.float32)
        chex.assert_shape(starts, (3, 1))
        chex.assert_type(starts, jnp.bool_)
        inputs = (embeddings, jnp.zeros((3, 1), jnp.int32), next_embeddings, starts, keys)
        return model.apply(variables, model.initial_carry(1), inputs, method=last_hidden)

    gradients = jax.jit(jax.grad(loss_fn))(embeddings, jnp.zeros((3, 1), jnp.bool_))
    # Last hidden state depends on earlier posterior samples via straight-through
    # gradients and Mamba memory, but cannot see its own next-frame embedding.
    assert np.any(np.asarray(gradients[0]) != 0)
    assert np.any(np.asarray(gradients[1]) != 0)
    np.testing.assert_array_equal(gradients[2], 0)
    reset = jax.jit(jax.grad(loss_fn))(embeddings, jnp.array([[False], [False], [True]]))
    np.testing.assert_array_equal(reset, 0)


@pytest.mark.parametrize("zero_logit", [-jnp.inf, -1e30])
def test_zero_probability_categories_have_finite_losses_and_gradients(zero_logit: float) -> None:
    q = jnp.array([[[0.0, zero_logit], [0.0, 0.0]]], jnp.float32)
    p = jnp.zeros_like(q)
    np.testing.assert_allclose(categorical_kl(q, p), np.log(2), rtol=1e-6)
    np.testing.assert_allclose(categorical_kl(q, q), 0, atol=1e-6)
    np.testing.assert_allclose(categorical_entropy(q), np.log(2), rtol=1e-6)
    np.testing.assert_allclose(categorical_entropy(p), 2 * np.log(2), rtol=1e-6)
    for prior in (p, q):
        grad_q, grad_p = jax.grad(lambda a, b: categorical_kl(a, b).sum(), argnums=(0, 1))(q, prior)
        assert np.isfinite(grad_q).all()
        assert np.isfinite(grad_p).all()
    assert np.isfinite(jax.grad(lambda a: categorical_entropy(a).sum())(q)).all()
    # Disjoint support must remain infinite, and wholly invalid distributions
    # must remain NaN so training's finite-metric check can detect them.
    assert np.isinf(
        categorical_kl(jnp.array([[[0.0, -jnp.inf]]], jnp.float32), jnp.array([[[-jnp.inf, 0.0]]], jnp.float32))
    ).all()
    assert np.isnan(categorical_entropy(jnp.full_like(q, -jnp.inf))).all()


def test_straight_through_samples_stay_float32_with_x64_enabled() -> None:
    model = small_model()
    with jax.enable_x64():
        logits = jnp.zeros((2, 4, 4), jnp.float32)
        key = jax.random.key(24)
        sample = model.apply({}, logits, key, method=model._sample)
        legacy_sample = model.apply({}, logits, jax.random.key_data(key), method=model._sample)
        chex.assert_shape(sample, (2, 4, 4))
        np.testing.assert_array_equal(sample, legacy_sample)
        chex.assert_type(sample, jnp.float32)
        np.testing.assert_array_equal(sample.sum(-1), 1)
        assert set(np.unique(sample)) <= {0, 1}
        weights = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)

        def sample_loss(values: jax.Array) -> jax.Array:
            chex.assert_shape(values, (2, 4, 4))
            chex.assert_type(values, jnp.float32)
            return (model.apply({}, values, key, method=model._sample) * weights).sum()

        gradients = jax.grad(sample_loss)(logits)
        expected = jax.grad(lambda a: (jax.nn.softmax(a, axis=-1) * weights).sum())(logits)
        chex.assert_type(gradients, jnp.float32)
        np.testing.assert_allclose(gradients, expected)
