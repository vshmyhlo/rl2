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
from rl2.sequence_model import RecurrentSequenceModel


def assert_carry_close(actual: Mamba3StackCarry, expected: Mamba3StackCarry) -> None:
    chex.assert_trees_all_equal_shapes_and_dtypes(actual, expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize(
    # RMSNorm, no MLP, and LayerNorm with MIMO/MLP wiring. Mixer rank coverage lives in test_mamba3.
    "rank,rms_norm,width",
    [(1, True, 16), (1, False, 0), (2, False, 16)],
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

    def loss(parameters: Any, x: jax.Array) -> tuple[jax.Array, tuple[Mamba3StackCarry, jax.Array]]:
        chex.assert_shape(x, (6, 2, 8))
        chex.assert_type(x, jnp.float32)
        chex.assert_trees_all_equal_shapes_and_dtypes(parameters, params)
        carry, y = model.apply(
            {"params": parameters},
            x,
            carry=model.initial_carry(num_envs=x.shape[-2]),
            episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_),
        )
        return jnp.sum(y * case["probe"]), (carry, y)

    (_, (carry, y)), (grads, dx) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(params, case["x"])
    np.testing.assert_allclose(y, case["y"], rtol=5e-5, atol=3e-6)
    for i, state in enumerate(carry):
        for name, leaf in zip(state._fields, state):
            np.testing.assert_allclose(leaf, case[f"carry/{i}/{name}"], rtol=5e-5, atol=3e-6)
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
    assert isinstance(model, RecurrentSequenceModel)
    initial = model.initial_carry(num_envs=2)
    variables = model.init(jax.random.key(2), x, initial, starts)
    warm, _ = model.apply(variables, x[:2], initial, starts[:2])
    step = jax.jit(partial(model.apply, method=model.step))
    for incoming in (initial, warm):
        final, expected = jax.jit(model.apply)(variables, x, incoming, starts)
        carry, first = model.apply(variables, x[:2], incoming, starts[:2])
        carry, second = model.apply(variables, x[2:], carry, starts[2:])
        np.testing.assert_allclose(jnp.concatenate((first, second)), expected, rtol=3e-5, atol=3e-6)
        assert_carry_close(carry, final)
        carry = incoming
        outputs = []
        for token, reset in zip(x, starts):
            carry, y = step(variables, token, carry, reset)
            outputs.append(y)
        np.testing.assert_allclose(jnp.stack(outputs), expected, rtol=3e-5, atol=3e-6)
        assert_carry_close(carry, final)
    empty_carry, empty = jax.jit(model.apply)(variables, x[:0], final, starts[:0])
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
    variables = model.init(
        jax.random.key(4),
        x,
        carry=model.initial_carry(num_envs=x.shape[-2]),
        episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_),
    )
    starts = jnp.zeros((6, 2), jnp.bool_).at[3, 0].set(True)
    final, y = model.apply(variables, x, episode_starts=starts, carry=model.initial_carry(num_envs=x.shape[-2]))
    fresh, fresh_y = model.apply(
        variables,
        x[3:, :1],
        carry=model.initial_carry(num_envs=x[3:, :1].shape[-2]),
        episode_starts=jnp.zeros(x[3:, :1].shape[:-1], jnp.bool_),
    )
    continuous, continuous_y = model.apply(
        variables,
        x[:, 1:],
        carry=model.initial_carry(num_envs=x[:, 1:].shape[-2]),
        episode_starts=jnp.zeros(x[:, 1:].shape[:-1], jnp.bool_),
    )
    np.testing.assert_allclose(y[3:, :1], fresh_y, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(y[:, 1:], continuous_y, rtol=3e-5, atol=3e-6)

    def select_example(leaf: jax.Array, index: int) -> jax.Array:
        chex.assert_shape(leaf, (2, ...))
        chex.assert_type(leaf, jnp.float32)
        return leaf[index : index + 1]

    assert_carry_close(jax.tree.map(partial(select_example, index=0), final), fresh)
    assert_carry_close(jax.tree.map(partial(select_example, index=1), final), continuous)
    reset, reset_y = model.apply(variables, x[0], final, jnp.ones(2, jnp.bool_), method=model.step)
    zero, zero_y = model.apply(
        variables,
        x[0],
        method=model.step,
        carry=model.initial_carry(num_envs=x[0].shape[-2]),
        episode_starts=jnp.zeros(x[0].shape[:-1], jnp.bool_),
    )
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
    params = model.init(
        jax.random.key(6),
        x,
        carry=model.initial_carry(num_envs=x.shape[-2]),
        episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_),
    )["params"]

    def loss(parameters: Any, inputs: jax.Array) -> jax.Array:
        chex.assert_shape(inputs, (4, 2, 8))
        chex.assert_type(inputs, jnp.float32)
        _, y = model.apply(
            {"params": parameters},
            inputs,
            carry=model.initial_carry(num_envs=inputs.shape[-2]),
            episode_starts=jnp.zeros(inputs.shape[:-1], jnp.bool_),
        )
        return jnp.sum(y[:2].astype(jnp.float32) * jnp.arange(8))

    grads, dx = jax.jit(jax.grad(loss, argnums=(0, 1)))(params, x)
    for gradient in jax.tree.leaves(grads):
        assert np.isfinite(gradient).all()
        assert np.any(np.asarray(gradient) != 0)
    np.testing.assert_array_equal(dx[2:], 0)
    carry, y = model.apply(
        {"params": params},
        x,
        carry=model.initial_carry(num_envs=x.shape[-2]),
        episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_),
    )
    _, changed = model.apply(
        {"params": params},
        x.at[2:].set(100),
        carry=model.initial_carry(num_envs=(x.at[2:].set(100)).shape[-2]),
        episode_starts=jnp.zeros((x.at[2:].set(100)).shape[:-1], jnp.bool_),
    )
    np.testing.assert_array_equal(changed[:2], y[:2])
    assert y.dtype == dtype
    chex.assert_type(jax.tree.leaves((params, carry)), jnp.float32)


@pytest.mark.parametrize("width", [0, 13, None])
def test_width_rounding_and_depth_scaled_initialization(width: int | None) -> None:
    model = Mamba3Stack(8, 3, d_state=8, headdim=4, d_intermediate=width, mlp_multiple_of=8)
    x = jnp.ones((1, 1, 8), jnp.float32)
    key = jax.random.key(7)
    scaled = model.init(
        key, x, carry=model.initial_carry(num_envs=x.shape[-2]), episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_)
    )["params"]
    unscaled = model.clone(rescale_prenorm_residual=False).init(
        key, x, carry=model.initial_carry(num_envs=x.shape[-2]), episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_)
    )["params"]
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


def test_rejects_nonfloating_inputs_and_incorrect_layer_count() -> None:
    model = Mamba3Stack(4, 2, d_state=4, headdim=2, d_intermediate=0)
    x = jnp.ones((1, 1, 4), jnp.float32)
    starts = jnp.zeros((1, 1), jnp.bool_)
    carry = model.initial_carry(1)
    with pytest.raises(AssertionError):
        model.init(jax.random.key(0), x.astype(jnp.int32), carry, starts)
    with pytest.raises(ValueError, match="one Mamba3Carry per layer"):
        model.init(jax.random.key(0), x, carry[:1], starts)


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
    variables = model.init(
        jax.random.key(19),
        x,
        carry=model.initial_carry(num_envs=x.shape[-2]),
        episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_),
    )
    forward = jax.jit(model.apply)
    carry, y = forward(
        variables,
        x,
        carry=model.initial_carry(num_envs=x.shape[-2]),
        episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_),
    )
    shifted_carry, shifted_y = forward(
        variables,
        x + 10000,
        carry=model.initial_carry(num_envs=(x + 10000).shape[-2]),
        episode_starts=jnp.zeros((x + 10000).shape[:-1], jnp.bool_),
    )
    if not final_norm:
        shifted_y = shifted_y - 10000
    # Residual additions at magnitude 1e4 round to about 1e-3 in float32.
    np.testing.assert_allclose(shifted_y, y, rtol=1e-3, atol=2e-3)
    for a, b in zip(jax.tree.leaves(shifted_carry), jax.tree.leaves(carry)):
        np.testing.assert_allclose(a, b, rtol=2e-3, atol=5e-4)
