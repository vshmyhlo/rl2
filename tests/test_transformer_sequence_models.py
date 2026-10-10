from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2 import transformer
from rl2.sequence_model import ARSequenceModel, BDSequenceModel
from rl2.transformer import ARTransformer, BDTransformer, Transformer


def test_public_model_exports() -> None:
    assert set(transformer.__all__) == {
        "ARTransformer",
        "BDTransformer",
        "TransformerCarry",
        "TransformerStackCarry",
    }


def test_bidirectional_model_has_no_decoding_interface() -> None:
    model = BDTransformer(4, 1, num_heads=1, max_seq_len=3)
    assert not hasattr(model, "step")
    assert not hasattr(model, "initial_carry")


def test_autoregressive_sequence_chunks_and_steps_match_from_fresh_and_supplied_carry() -> None:
    model = ARTransformer(4, 2, num_heads=1, max_seq_len=5, initializer_range=0.2)
    assert isinstance(model, Transformer)
    assert isinstance(model, ARSequenceModel)
    x = jax.random.normal(jax.random.key(0), (3, 3, 4))
    x_len = jnp.array([0, 2, 3], jnp.int32)
    valid = jnp.arange(3)[None, :] < x_len[:, None]
    padded = jnp.where(valid[..., None], x, jnp.nan)
    variables = model.init(jax.random.key(1), x, x_len)
    prefix_len = jnp.array([2, 1, 0], jnp.int32)
    supplied, _ = model.apply(variables, x[:, :2], prefix_len)
    apply = jax.jit(model.apply)
    step = jax.jit(partial(model.apply, method=model.step))

    for initial in (None, supplied):
        final, expected = apply(variables, padded, x_len, initial)
        chex.assert_shape(expected, (3, 3, 4))
        chex.assert_type(expected, jnp.float32)
        state, clean = apply(variables, x, x_len, initial)
        chex.assert_trees_all_close((final, expected), (state, clean), atol=2e-6)
        np.testing.assert_array_equal(expected[~valid], 0)
        initial_state = model.initial_carry(3) if initial is None else initial
        for before, after in zip(jax.tree.leaves(initial_state), jax.tree.leaves(final)):
            np.testing.assert_array_equal(after[0], before[0])

        state, first = apply(variables, padded[:, :1], jnp.minimum(x_len, 1), initial)
        state, rest = apply(variables, padded[:, 1:], jnp.maximum(x_len - 1, 0), state)
        chex.assert_trees_all_close((state, jnp.concatenate((first, rest), 1)), (final, expected), atol=2e-6)

        state = initial
        outputs = []
        for t in range(3):
            state, output = step(variables, padded[:, t], t < x_len, state)
            chex.assert_shape(output, (3, 4))
            chex.assert_type(output, jnp.float32)
            outputs.append(output)
        chex.assert_trees_all_close((state, jnp.stack(outputs, 1)), (final, expected), atol=2e-6)

    # Continued batches must also match independently processed unpadded histories.
    for b in (1, 2):
        history = jnp.concatenate((x[b : b + 1, : prefix_len[b]], x[b : b + 1, : x_len[b]]), 1)
        reference, output = model.apply(variables, history, jnp.array([history.shape[1]], jnp.int32))
        np.testing.assert_allclose(expected[b, : x_len[b]], output[0, prefix_len[b] :], atol=2e-6)
        for actual_leaf, reference_leaf in zip(jax.tree.leaves(final), jax.tree.leaves(reference)):
            np.testing.assert_allclose(actual_leaf[b : b + 1], reference_leaf, atol=2e-6)

    _, perturbed = apply(variables, x.at[:, -1].add(10), x_len)
    _, baseline = apply(variables, x, x_len)
    np.testing.assert_allclose(perturbed[:, :-1], baseline[:, :-1], atol=2e-6)
    # An all-true mask advances everyone, including examples skipped previously.
    for initial in (None, supplied):
        actual = step(variables, x[:, 0], jnp.ones((3,), jnp.bool_), initial)
        expected_step = apply(variables, x[:, :1], jnp.ones((3,), jnp.int32), initial)
        chex.assert_trees_all_close(actual, (expected_step[0], expected_step[1][:, 0]), atol=2e-6)
    skipped, zero = step(variables, padded[:, 0], jnp.zeros((3,), jnp.bool_), final)
    chex.assert_trees_all_equal(skipped, final)
    np.testing.assert_array_equal(zero, 0)


def test_bidirectional_valid_prefixes_and_gradients_ignore_padding() -> None:
    model = BDTransformer(4, 2, num_heads=1, max_seq_len=3, initializer_range=0.2)
    assert isinstance(model, BDSequenceModel)
    x = jax.random.normal(jax.random.key(2), (3, 3, 4))
    x_len = jnp.array([0, 2, 3], jnp.int32)
    valid = jnp.arange(3)[None, :] < x_len[:, None]
    padded = jnp.where(valid[..., None], x, jnp.nan)
    variables = model.init(jax.random.key(3), x, x_len)
    # The shared stack retains checkpoint parameter names and computation.
    reference_model = Transformer(4, 2, num_heads=1, max_seq_len=3, causal=False)
    _, reference_output = reference_model.apply(variables, x, x_len)
    output = jax.jit(model.apply)(variables, padded, x_len)
    np.testing.assert_allclose(output, reference_output, atol=2e-6)
    chex.assert_shape(output, (3, 3, 4))
    chex.assert_type(output, jnp.float32)
    np.testing.assert_array_equal(output[~valid], 0)
    for b in (1, 2):
        reference = model.apply(variables, x[b : b + 1, : x_len[b]], x_len[b : b + 1])
        np.testing.assert_allclose(output[b, : x_len[b]], reference[0], atol=2e-6)
    changed = model.apply(variables, x.at[:, -1].add(jnp.array([1, -2, 3, -4])), x_len)
    np.testing.assert_allclose(changed[:2], output[:2], atol=2e-6)
    assert not np.allclose(changed[2, 0], output[2, 0], atol=1e-5)

    def loss(inputs: jax.Array) -> jax.Array:
        chex.assert_shape(inputs, (3, 3, 4))
        chex.assert_type(inputs, jnp.float32)
        y = model.apply(variables, inputs, x_len)
        chex.assert_shape(y, (3, 3, 4))
        chex.assert_type(y, jnp.float32)
        return jnp.sum(y[..., 0])

    gradient = jax.jit(jax.grad(loss))(padded)
    chex.assert_shape(gradient, (3, 3, 4))
    chex.assert_type(gradient, jnp.float32)
    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(gradient[~valid], 0)
    assert np.linalg.norm(gradient[valid]) > 0
    for lengths in (jnp.array([-1, 2, 3], jnp.int32), jnp.array([0, 2, 4], jnp.int32)):
        with pytest.raises(ValueError, match="x_len must be between"):
            model.apply(variables, x, lengths)


@pytest.mark.parametrize("model_type", [ARTransformer, BDTransformer])
def test_specialized_stacks_validate_fixed_direction(model_type: type[ARTransformer] | type[BDTransformer]) -> None:
    model = model_type(4, 1, num_heads=1, max_seq_len=3)
    x = jnp.zeros((1, 3, 4), jnp.float32)
    with pytest.raises(ValueError, match=f"{model_type.__name__} requires causal="):
        model.clone(causal=not model.causal).apply({}, x, jnp.array([3], jnp.int32))
