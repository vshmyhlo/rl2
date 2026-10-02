from functools import partial
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict, unflatten_dict

from rl2.mamba3 import Mamba3Stack, Mamba3StackCarry


def assert_carry_close(actual: Mamba3StackCarry, expected: Mamba3StackCarry) -> None:
    chex.assert_trees_all_equal_shapes_and_dtypes(actual, expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize(
    "rank,rms_norm,width", [(1, True, 16), (2, True, 16), (1, False, 0), (4, True, 16), (2, False, 16)]
)
def test_matches_official_blocks_outputs_states_and_gradients(rank: int, rms_norm: bool, width: int) -> None:
    """Fixtures execute upstream Block, GatedMLP and Mamba3 with CPU kernels."""
    model = Mamba3Stack(
        8, 2, d_state=8, headdim=4, d_intermediate=width, mlp_multiple_of=1, mimo_rank=rank, rms_norm=rms_norm
    )
    prefix = f"r{rank}_rms{int(rms_norm)}_w{width}/"
    with np.load(Path(__file__).parent / "data" / "mamba3_stack_reference.npz", allow_pickle=False) as fixture:
        assert str(fixture["upstream_revision"]) == "e9594ce1c732d97440f0332fdc43170a2294dbfa"
        case = {key.removeprefix(prefix): jnp.asarray(fixture[key]) for key in fixture.files if key.startswith(prefix)}
    params = unflatten_dict(
        {key.removeprefix("params/"): value for key, value in case.items() if key.startswith("params/")}, sep="/"
    )
    carry, y = jax.jit(model.apply)({"params": params}, case["x"])
    np.testing.assert_allclose(y, case["y"], rtol=5e-5, atol=3e-6)
    for i, state in enumerate(carry):
        for name, leaf in zip(state._fields, state):
            np.testing.assert_allclose(leaf, case[f"carry/{i}/{name}"], rtol=5e-5, atol=3e-6)

    def loss(parameters: Any, x: jax.Array) -> jax.Array:
        chex.assert_shape(x, (6, 2, 8))
        chex.assert_type(x, jnp.float32)
        _, y = model.apply({"params": parameters}, x)
        return jnp.sum(y * case["probe"])

    grads, dx = jax.jit(jax.grad(loss, argnums=(0, 1)))(params, case["x"])
    np.testing.assert_allclose(dx, case["dx"], rtol=1e-4, atol=1e-5)
    for name, gradient in flatten_dict(grads, sep="/").items():
        np.testing.assert_allclose(gradient, case[f"grads/{name}"], rtol=1e-4, atol=1e-5, err_msg=name)


@pytest.mark.parametrize("depth,rank,width,final_norm", [(1, 1, 0, False), (3, 1, 16, True), (2, 2, 13, True)])
def test_sequence_chunk_step_and_empty_agree(depth: int, rank: int, width: int, final_norm: bool) -> None:
    model = Mamba3Stack(
        8,
        depth,
        d_state=8,
        headdim=4,
        d_intermediate=width,
        mlp_multiple_of=4,
        mimo_rank=rank,
        final_norm=final_norm,
        ngroups=2,
        rope_fraction=1.0,
        outproj_norm=True,
    )
    x = jax.random.normal(jax.random.key(1), (6, 2, 8))
    starts = jnp.zeros((6, 2), jnp.bool_).at[3, 0].set(True)
    variables = model.init(jax.random.key(2), x)
    final, expected = jax.jit(model.apply)(variables, x, None, starts)
    carry, first = model.apply(variables, x[:2], episode_starts=starts[:2])
    carry, second = model.apply(variables, x[2:], carry, starts[2:])
    np.testing.assert_allclose(jnp.concatenate((first, second)), expected, rtol=3e-5, atol=3e-6)
    assert_carry_close(carry, final)
    carry = model.initial_carry(2)
    outputs = []
    step = jax.jit(partial(model.apply, method=model.step))
    for token, reset in zip(x, starts):
        carry, y = step(variables, token, carry, reset)
        outputs.append(y)
    np.testing.assert_allclose(jnp.stack(outputs), expected, rtol=3e-5, atol=3e-6)
    assert_carry_close(carry, final)
    empty_carry, empty = jax.jit(model.apply)(variables, x[:0], final)
    assert empty.shape == (0, 2, 8)
    assert_carry_close(empty_carry, final)
    assert len(final) == depth
    # Every layer gets independent weights, rather than a tied recurrent depth.
    if depth > 1:
        params = variables["params"]
        assert not np.array_equal(
            params["layers_0"]["mixer"]["in_proj"]["kernel"], params["layers_1"]["mixer"]["in_proj"]["kernel"]
        )


@pytest.mark.parametrize("rank", [1, 2])
def test_reset_clears_every_layer_without_affecting_other_examples(rank: int) -> None:
    model = Mamba3Stack(8, 2, d_state=8, headdim=4, d_intermediate=16, mlp_multiple_of=1, mimo_rank=rank)
    x = jax.random.normal(jax.random.key(3), (6, 2, 8))
    variables = model.init(jax.random.key(4), x)
    starts = jnp.zeros((6, 2), jnp.bool_).at[3, 0].set(True)
    final, y = model.apply(variables, x, episode_starts=starts)
    fresh, fresh_y = model.apply(variables, x[3:, :1])
    continuous, continuous_y = model.apply(variables, x[:, 1:])
    np.testing.assert_allclose(y[3:, :1], fresh_y, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(y[:, 1:], continuous_y, rtol=3e-5, atol=3e-6)

    def select_example(leaf: jax.Array, index: int) -> jax.Array:
        chex.assert_shape(leaf, (2, ...))
        chex.assert_type(leaf, jnp.float32)
        return leaf[index : index + 1]

    assert_carry_close(jax.tree.map(partial(select_example, index=0), final), fresh)
    assert_carry_close(jax.tree.map(partial(select_example, index=1), final), continuous)
    reset, reset_y = model.apply(variables, x[0], final, jnp.ones(2, jnp.bool_), method=model.step)
    zero, zero_y = model.apply(variables, x[0], method=model.step)
    assert_carry_close(reset, zero)
    np.testing.assert_allclose(reset_y, zero_y, atol=3e-6)


@pytest.mark.parametrize("dtype,residual_fp32", [(jnp.float32, True), (jnp.bfloat16, True), (jnp.bfloat16, False)])
def test_precision_gradients_and_causality(dtype: jax.typing.DTypeLike, residual_fp32: bool) -> None:
    model = Mamba3Stack(
        8,
        2,
        d_state=8,
        headdim=4,
        d_intermediate=16,
        mlp_multiple_of=1,
        dtype=dtype,
        residual_in_fp32=residual_fp32,
    )
    x = jax.random.normal(jax.random.key(5), (4, 2, 8))
    params = model.init(jax.random.key(6), x)["params"]

    def loss(parameters: Any, inputs: jax.Array) -> jax.Array:
        chex.assert_shape(inputs, (4, 2, 8))
        chex.assert_type(inputs, jnp.float32)
        _, y = model.apply({"params": parameters}, inputs)
        return jnp.sum(y[:2].astype(jnp.float32) * jnp.arange(8))

    grads, dx = jax.jit(jax.grad(loss, argnums=(0, 1)))(params, x)
    for gradient in jax.tree.leaves(grads):
        assert np.isfinite(gradient).all()
        assert np.any(np.asarray(gradient) != 0)
    np.testing.assert_array_equal(dx[2:], 0)
    carry, y = model.apply({"params": params}, x)
    _, changed = model.apply({"params": params}, x.at[2:].set(100))
    np.testing.assert_array_equal(changed[:2], y[:2])
    assert y.dtype == dtype
    chex.assert_type(jax.tree.leaves((params, carry)), jnp.float32)


@pytest.mark.parametrize("width", [0, 13, None])
def test_width_rounding_and_depth_scaled_initialization(width: int | None) -> None:
    model = Mamba3Stack(8, 3, d_state=8, headdim=4, d_intermediate=width, mlp_multiple_of=8)
    x = jnp.ones((1, 1, 8), jnp.float32)
    key = jax.random.key(7)
    scaled = model.init(key, x)["params"]
    unscaled = model.clone(rescale_prenorm_residual=False).init(key, x)["params"]
    factor = (3 * (1 if width == 0 else 2)) ** -0.5
    for name, value in flatten_dict(scaled, sep="/").items():
        original = flatten_dict(unscaled, sep="/")[name]
        if name.endswith(("out_proj/kernel", "fc2/kernel")):
            np.testing.assert_allclose(value, original * factor, rtol=2e-6, atol=1e-8)
        else:
            np.testing.assert_array_equal(value, original)
    if width == 0:
        assert "fc1" not in scaled["layers_0"]
        assert "norm2" not in scaled["layers_0"]
    else:
        expected_width = 24 if width is None else 16
        assert scaled["layers_0"]["fc1"]["kernel"].shape == (8, 2 * expected_width)


@pytest.mark.parametrize(
    "options",
    [
        {"num_layers": 0},
        {"num_layers": True},
        {"num_layers": 1.5},
        {"d_intermediate": -1},
        {"d_intermediate": 1.5},
        {"mlp_multiple_of": 0},
        {"norm_epsilon": 0},
        {"norm_epsilon": float("nan")},
        {"d_state": 3},
        {"dtype": jnp.int32},
    ],
)
def test_invalid_configuration(options: dict[str, Any]) -> None:
    config = {"d_model": 8, "num_layers": 2, "d_state": 8, "headdim": 4, **options}
    with pytest.raises((AssertionError, TypeError, ValueError)):
        Mamba3Stack(**config).initial_carry(2)


def test_invalid_input_carry_and_reset() -> None:
    model = Mamba3Stack(8, 2, d_state=8, headdim=4, d_intermediate=0)
    x = jnp.ones((3, 2, 8), jnp.float32)
    variables = model.init(jax.random.key(8), x)
    for bad in (x[0], x.astype(jnp.int32), x[..., :7]):
        with pytest.raises(AssertionError):
            model.apply(variables, bad)
    carry = model.initial_carry(2)
    for bad in (carry[:1], (None, None), model.initial_carry(1), list(carry)):
        with pytest.raises((AssertionError, TypeError, ValueError)):
            model.apply(variables, x, bad)
    for bad in (jnp.zeros((3, 2)), jnp.zeros((2, 3), jnp.bool_)):
        with pytest.raises(AssertionError):
            model.apply(variables, x, episode_starts=bad)
    with pytest.raises(AssertionError):
        model.apply(variables, x[0], episode_starts=jnp.zeros((1, 2), jnp.bool_), method=model.step)


@pytest.mark.parametrize("final_norm", [False, True])
def test_layernorm_stack_is_stable_under_large_input_offsets(final_norm: bool) -> None:
    """A featurewise constant offset must not change any normalized branch.

    Exercise both prenorms and the final norm. Flax's default fast variance
    loses the small feature variance at this offset, unlike torch LayerNorm.
    Integer-valued inputs avoid introducing input quantization differences.
    """
    model = Mamba3Stack(
        8,
        2,
        d_state=8,
        headdim=4,
        d_intermediate=16,
        mlp_multiple_of=1,
        rms_norm=False,
        final_norm=final_norm,
    )
    x = jnp.arange(48, dtype=jnp.float32).reshape(3, 2, 8) % 11
    variables = model.init(jax.random.key(19), x)
    forward = jax.jit(model.apply)
    carry, y = forward(variables, x)
    shifted_carry, shifted_y = forward(variables, x + 10000)
    if not final_norm:
        shifted_y = shifted_y - 10000
    # Residual additions at magnitude 1e4 round to about 1e-3 in float32.
    np.testing.assert_allclose(shifted_y, y, rtol=1e-3, atol=2e-3)
    for a, b in zip(jax.tree.leaves(shifted_carry), jax.tree.leaves(carry)):
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=5e-4)
