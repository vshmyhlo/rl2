from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.shape_checker import ShapeChecker
from rl2.transformer import (
    TransformerBlock,
    TransformerCarry,
    TransformerStack,
    TransformerStackCarry,
    _rope,
)

type Carry = TransformerCarry | TransformerStackCarry


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_rope_matches_llama_adjacent_pairs_in_float32(dtype: jax.typing.DTypeLike) -> None:
    x = jnp.asarray([[[[1.25, -0.75, 0.5, 1.75]], [[1.25, -0.75, 0.5, 1.75]], [[0.25, 1.5, -1.25, -0.5]]]], dtype=dtype)
    positions = jnp.asarray([[0, 3, 17]], jnp.int32)
    # Independent complex64 oracle matching Meta's adjacent-pair rotation.
    sc = ShapeChecker(B=1, T=3, H=1, F=4)
    sc.check(x, "BTHF", dtype)
    sc.check(positions, "BT", jnp.int32)
    angles = np.asarray(positions, np.float32)[..., None, None] * np.asarray([1, 0.01], np.float32)
    values = np.asarray(x, np.float32)
    complex_values = values[..., ::2] + 1j * values[..., 1::2]
    rotated = complex_values * np.exp(1j * angles)
    expected = np.stack((rotated.real, rotated.imag), axis=-1).reshape(x.shape).astype(x.dtype)
    actual = _rope(x, positions, 10000.0)
    sc.check(actual, "BTHF", dtype)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:, 0], x[:, 0])
    compiled = jax.jit(_rope, static_argnums=2)(x, positions, 10000.0)
    np.testing.assert_allclose(
        compiled.astype(jnp.float32), expected.astype(np.float32), atol=0, rtol=2 * jnp.finfo(dtype).eps
    )


@pytest.mark.parametrize("input_dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
def test_block_preserves_residual_dtype(input_dtype: jax.typing.DTypeLike) -> None:
    compute_dtype = jnp.bfloat16 if input_dtype == jnp.float32 else input_dtype
    block = TransformerBlock(8, mlp_expansion=2.0, num_heads=2, max_seq_len=2, dtype=compute_dtype)
    x = jax.random.normal(jax.random.key(29), (1, 2, 8)).astype(input_dtype)
    variables = block.init(jax.random.key(30), x, None, None)
    carry, output = jax.jit(block.apply)(variables, x, None, None)
    chex.assert_shape(output, x.shape)
    chex.assert_type(output, input_dtype)
    chex.assert_type((carry.key, carry.value), compute_dtype)
    chex.assert_type(jax.tree.leaves(variables["params"]), jnp.float32)
    assert np.isfinite(output).all()


def assert_carry_close(actual: Carry, expected: Carry, tolerance: float = 5e-6) -> None:
    chex.assert_trees_all_equal_shapes_and_dtypes(actual, expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)


def reference_block(
    x: np.ndarray,
    params: dict[str, Any],
    starts: np.ndarray,
    heads: int,
    kv_heads: int,
    theta: float,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Float64 NumPy oracle with explicit per-episode, per-query attention."""
    sc = ShapeChecker(H=heads, K=kv_heads)
    sc.check(x, "BTD")
    chex.assert_type(x, np.floating)
    sc.check(starts, "BT", np.bool_)
    batch, steps, width = x.shape
    head_dim = width // heads
    residual = x.astype(np.float64)
    x = residual / np.sqrt(np.mean(residual**2, axis=-1, keepdims=True) + epsilon)
    x *= np.asarray(params["norm"]["scale"])
    projected = [x @ np.asarray(params[f"{name}_proj"]["kernel"]) for name in ("q", "k", "v")]
    query = projected[0].reshape(batch, steps, heads, head_dim)
    key, value = (v.reshape(batch, steps, kv_heads, head_dim) for v in projected[1:])
    sc.check(query, "BTHF", np.float64)
    sc.check((key, value), "BTKF", np.float64)
    output = np.zeros_like(query)
    for b in range(batch):
        start = 0
        for t in range(steps):
            if starts[b, t]:
                start = t
            for pair in range(head_dim // 2):
                angle = (t - start) / theta ** (2 * pair / head_dim)
                c, s = np.cos(angle), np.sin(angle)
                for tensor in (query, key):
                    real = tensor[b, t, :, 2 * pair].copy()
                    imag = tensor[b, t, :, 2 * pair + 1].copy()
                    tensor[b, t, :, 2 * pair] = real * c - imag * s
                    tensor[b, t, :, 2 * pair + 1] = real * s + imag * c
            left = start
            for h in range(heads):
                group = h // (heads // kv_heads)
                logits = key[b, left : t + 1, group] @ query[b, t, h] / np.sqrt(head_dim)
                weights = np.exp(logits - logits.max())
                output[b, t, h] = (weights / weights.sum()) @ value[b, left : t + 1, group]
    x = residual + output.reshape(x.shape) @ np.asarray(params["out_proj"]["kernel"])
    normalized = x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + epsilon)
    normalized *= np.asarray(params["norm2"]["scale"])
    gate = normalized @ np.asarray(params["gate_proj"]["kernel"])
    value = normalized @ np.asarray(params["up_proj"]["kernel"])
    sc.check((gate, value), "BTI", np.float64)
    result = x + ((gate / (1 + np.exp(-gate))) * value) @ np.asarray(params["down_proj"]["kernel"])
    sc.check(result, "BTD", np.float64)
    return result


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_attention_matches_numpy_reference(kv_heads: int) -> None:
    model = TransformerBlock(
        16, num_heads=4, num_kv_heads=kv_heads, max_seq_len=9, rope_theta=137.0, initializer_range=0.2
    )
    x = jax.random.normal(jax.random.key(1), (2, 9, 16))
    starts = jnp.zeros((2, 9), jnp.bool_).at[0, 4].set(True).at[1, 1].set(True).at[1, 7].set(True)
    variables = model.init(jax.random.key(2), x)
    # Larger weights expose QK normalization and RoPE pairing differences.
    variables["params"]["norm"]["scale"] = jnp.linspace(0.5, 1.5, 16)
    variables["params"]["norm2"]["scale"] = jnp.linspace(1.5, 0.5, 16)
    assert "q_norm" not in variables["params"]
    assert "k_norm" not in variables["params"]
    _, actual = jax.jit(model.apply)(variables, x, None, starts)
    expected = reference_block(np.asarray(x), variables["params"], np.asarray(starts), 4, kv_heads, 137.0)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    # Also exercise the native full causal path without a supplied mask.
    _, actual = model.apply(variables, x)
    expected = reference_block(np.asarray(x), variables["params"], np.zeros((2, 9), bool), 4, kv_heads, 137.0)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_norm_epsilon_and_full_history() -> None:
    model = TransformerBlock(8, num_heads=2, num_kv_heads=1, max_seq_len=4, norm_epsilon=1e-4)
    x = jax.random.normal(jax.random.key(23), (1, 4, 8)) * 0.01
    variables = model.init(jax.random.key(24), x)
    carry, actual = model.apply(variables, x[:, :3])
    final, last = model.apply(variables, x[:, 3:], carry)
    expected = reference_block(
        np.asarray(x), variables["params"], np.zeros((1, 4), bool), 2, 1, model.rope_theta, model.norm_epsilon
    )
    np.testing.assert_allclose(jnp.concatenate((actual, last), axis=1), expected, atol=1e-7, rtol=2e-5)
    np.testing.assert_array_equal(final.position, [4])
    # A uniform-attention probe makes the first token's contribution explicit.
    variables["params"]["q_proj"]["kernel"] = jnp.zeros_like(variables["params"]["q_proj"]["kernel"])
    variables["params"]["down_proj"]["kernel"] = jnp.zeros_like(variables["params"]["down_proj"]["kernel"])
    _, baseline = model.apply(variables, x)
    changed = x.at[:, 0].add(1)
    _, perturbed = model.apply(variables, changed)
    normalized = x / jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + model.norm_epsilon)
    normalized_changed = changed / jnp.sqrt(jnp.mean(changed**2, axis=-1, keepdims=True) + model.norm_epsilon)
    delta = (normalized_changed - normalized)[0, 0] * variables["params"]["norm"]["scale"]
    contribution = jnp.tile(delta @ variables["params"]["v_proj"]["kernel"], 2)
    contribution = contribution @ variables["params"]["out_proj"]["kernel"] / 4
    np.testing.assert_allclose(perturbed[0, -1] - baseline[0, -1], contribution, atol=1e-6)


@pytest.mark.parametrize("compiled", [False, True])
def test_cache_capacity_and_packed_episode_resets(compiled: bool) -> None:
    model = TransformerBlock(8, num_heads=2, max_seq_len=3)
    x = jax.random.normal(jax.random.key(25), (1, 6, 8))
    variables = model.init(jax.random.key(26), x[:, :1])
    apply = jax.jit(model.apply) if compiled else model.apply
    starts = jnp.zeros((1, 6), jnp.bool_).at[:, 3].set(True)
    carry, actual = apply(variables, x, episode_starts=starts)
    expected = reference_block(np.asarray(x), variables["params"], np.asarray(starts), 2, 2, model.rope_theta)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    fresh, expected = apply(variables, x[:, 3:])
    assert_carry_close(carry, fresh)
    # A full cache may be reused after an explicit reset.
    reset, output = apply(variables, x[:, :1], carry, jnp.ones((1, 1), jnp.bool_))
    fresh, expected = apply(variables, x[:, :1])
    assert_carry_close(reset, fresh)
    np.testing.assert_allclose(output, expected, atol=2e-6)
    if compiled:
        # Batched conditionals must not report overflow for valid episodes.
        mapped = jax.jit(jax.vmap(model.apply, in_axes=(None, 0)))
        _, outputs = mapped(variables, jnp.stack((x[:, :3], x[:, 3:])))
        np.testing.assert_allclose(outputs[1], actual[:, 3:], atol=2e-6, rtol=2e-5)
    error = jax.errors.JaxRuntimeError if compiled else ValueError
    with pytest.raises(error, match="max_seq_len cache capacity"):
        jax.block_until_ready(apply(variables, x[:, :4]))
    with pytest.raises(error, match="max_seq_len cache capacity"):
        jax.block_until_ready(apply(variables, x[:, :1], carry))
    # Overflow before a later reset must also fail.
    with pytest.raises(error, match="max_seq_len cache capacity"):
        jax.block_until_ready(apply(variables, x, episode_starts=starts.at[:, 3].set(False).at[:, 4].set(True)))


@pytest.mark.parametrize(
    "kwargs,width",
    [
        pytest.param({}, 16, id="default-expansion"),
        pytest.param({"mlp_expansion": 3.0}, 24, id="explicit-expansion"),
        pytest.param({"mlp_expansion": 0.125}, 1, id="minimum-width"),
        pytest.param({"mlp_expansion": 1.4}, 11, id="round-down"),
        pytest.param({"mlp_expansion": 1.45}, 12, id="round-up"),
    ],
)
def test_mlp_expansion_width(kwargs: dict[str, Any], width: int) -> None:
    block = TransformerBlock(8, num_heads=2, **kwargs)
    stack = TransformerStack(8, 1, num_heads=2, **kwargs)
    assert block._mlp_width() == stack._mlp_width() == stack._make_block()._mlp_width() == width


def test_mlp_shapes_and_norm_defaults() -> None:
    model = TransformerStack(8, 1, num_heads=2, max_seq_len=1, mlp_expansion=1.45)
    width = 12
    x = jnp.ones((1, 1, 8))
    variables = model.init(jax.random.key(27), x)
    layer = variables["params"]["layers_0"]
    assert layer["gate_proj"]["kernel"].shape == (8, width)
    assert layer["up_proj"]["kernel"].shape == (8, width)
    assert layer["down_proj"]["kernel"].shape == (width, 8)
    assert model.norm_epsilon == model._make_block().norm_epsilon == 1e-5
    _, output = model.apply(variables, x)
    assert np.isfinite(output).all()


def test_transformer_defaults() -> None:
    block = TransformerBlock(dim=8, num_heads=2)
    stack = TransformerStack(dim=8, num_layers=1, num_heads=2)
    assert block.dim == stack.dim == stack._make_block().dim == 8
    assert block.norm_epsilon == stack.norm_epsilon == 1e-5
    assert block.rope_theta == stack.rope_theta == stack._make_block().rope_theta == 10000.0
    assert block.mlp_expansion == stack.mlp_expansion == 2.0


def test_stack_matches_llama_reference_with_final_norm() -> None:
    model = TransformerStack(
        8,
        2,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=3,
        mlp_expansion=4.0,
        initializer_range=0.2,
    )
    x = jax.random.normal(jax.random.key(31), (1, 3, 8))
    sc = ShapeChecker(B=1, T=3, D=8)
    sc.check(x, "BTD", jnp.float32)
    variables = model.init(jax.random.key(32), x)
    params = variables["params"]
    params["norm_f"]["scale"] = jnp.linspace(0.5, 1.5, 8)
    expected = np.asarray(x)
    for index in range(2):
        layer = params[f"layers_{index}"]
        assert layer["gate_proj"]["kernel"].shape == (8, 32)
        assert set(layer) == {
            "norm",
            "norm2",
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        for name in ("q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"):
            assert set(layer[name]) == {"kernel"}
        expected = reference_block(expected, layer, np.zeros((1, 3), bool), 2, 1, model.rope_theta)
    expected /= np.sqrt(np.mean(expected**2, axis=-1, keepdims=True) + model.norm_epsilon)
    expected *= np.asarray(params["norm_f"]["scale"])
    _, actual = jax.jit(model.apply)(variables, x)
    sc.check(actual, "BTD", jnp.float32)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("expansion", [0.0, -1.0, float("inf"), float("nan"), 0.01])
def test_invalid_mlp_expansion(expansion: float) -> None:
    for model in (
        TransformerBlock(8, num_heads=2, mlp_expansion=expansion),
        TransformerStack(8, 1, num_heads=2, mlp_expansion=expansion),
    ):
        with pytest.raises((ValueError, AssertionError)):
            model._mlp_width()


def test_projection_initialization_is_normal_and_independent_of_depth() -> None:
    model = TransformerStack(32, 2, num_heads=4, num_kv_heads=2, max_seq_len=1)
    x = jnp.zeros((1, 0, 32))
    key = jax.random.key(28)
    params = model.init(key, x)["params"]
    shallow = model.clone(num_layers=1).init(key, x)["params"]
    chex.assert_trees_all_equal(params["layers_0"], shallow["layers_0"])
    wider = model.clone(initializer_range=0.04).init(key, x)["params"]
    assert model.initializer_range == 0.02
    for index in range(2):
        layer, scaled = params[f"layers_{index}"], wider[f"layers_{index}"]
        for name in ("q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj"):
            weights = np.asarray(layer[name]["kernel"])
            # Each projection has at least 512 samples; fixed seeds keep these
            # distribution checks deterministic while tolerating sampling noise.
            np.testing.assert_allclose(weights.std(), 0.02, rtol=0.12)
            assert abs(weights.mean()) < 0.003
            assert np.max(np.abs(weights)) > 2.5 * 0.02
            np.testing.assert_allclose(scaled[name]["kernel"], weights * 2, atol=0, rtol=0)
        for norm in (layer["norm"], layer["norm2"]):
            np.testing.assert_array_equal(norm["scale"], 1)
    np.testing.assert_array_equal(params["norm_f"]["scale"], 1)


@pytest.mark.parametrize("initializer_range", [0.0, -0.02, float("inf"), float("nan")])
def test_invalid_initializer_range(initializer_range: float) -> None:
    for model in (
        TransformerBlock(8, num_heads=2, initializer_range=initializer_range),
        TransformerStack(8, 1, num_heads=2, initializer_range=initializer_range),
    ):
        with pytest.raises(ValueError, match="initializer_range must be positive and finite"):
            model.initial_carry(1)


@pytest.mark.parametrize("stack,window,resets", [(False, 9, True), (True, 9, True), (True, 9, False), (True, 1, True)])
def test_full_chunks_and_scanned_steps_agree(stack: bool, window: int, resets: bool) -> None:
    kwargs = {"dim": 16, "num_heads": 4, "num_kv_heads": 2, "max_seq_len": window}
    model = TransformerStack(**kwargs, num_layers=2) if stack else TransformerBlock(**kwargs)
    x = jax.random.normal(jax.random.key(3), (2, 9, 16))
    starts = jnp.zeros((2, 9), jnp.bool_)
    if window == 1:
        starts = jnp.ones_like(starts)
    elif resets:
        starts = starts.at[0, 2].set(True).at[0, 3].set(True).at[1, 6].set(True)
    variables = model.init(jax.random.key(4), x, episode_starts=starts)
    final, expected = jax.jit(model.apply)(variables, x, None, starts if resets else None)
    carry, first = model.apply(variables, x[:, :2], episode_starts=starts[:, :2])
    carry, second = model.apply(variables, x[:, 2:7], carry, starts[:, 2:7])
    carry, third = model.apply(variables, x[:, 7:], carry, starts[:, 7:])
    np.testing.assert_allclose(jnp.concatenate((first, second, third), axis=1), expected, rtol=2e-5, atol=5e-6)
    assert_carry_close(carry, final)

    def step(state: Carry, inputs: tuple[jax.Array, jax.Array]) -> tuple[Carry, jax.Array]:
        token, reset = inputs
        chex.assert_shape(token, (2, 16))
        chex.assert_type(token, jnp.float32)
        chex.assert_shape(reset, (2,))
        chex.assert_type(reset, jnp.bool_)
        return model.apply(variables, token, state, reset, method=model.step)

    carry, actual = jax.jit(partial(jax.lax.scan, step))(model.initial_carry(2), (jnp.swapaxes(x, 0, 1), starts.T))
    actual = jnp.swapaxes(actual, 0, 1)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=5e-6)
    assert_carry_close(carry, final)
    empty_carry, empty = jax.jit(model.apply)(variables, x[:, :0], final)
    assert empty.shape == (2, 0, 16)
    assert_carry_close(empty_carry, final)
    fresh, single = model.apply(variables, x[:, 0], method=model.step)
    initial, sequence = model.apply(variables, x[:, :1])
    assert_carry_close(fresh, initial)
    np.testing.assert_allclose(single, sequence[:, 0])


def test_causality_episode_isolation_and_cache_clearing() -> None:
    model = TransformerStack(16, 2, num_heads=4, num_kv_heads=1, max_seq_len=8)
    x = jax.random.normal(jax.random.key(5), (2, 7, 16))
    variables = model.init(jax.random.key(6), x)
    _, baseline = model.apply(variables, x)
    _, perturbed = model.apply(variables, x.at[:, 4:].add(100))
    np.testing.assert_allclose(baseline[:, :4], perturbed[:, :4], atol=1e-6)
    starts = jnp.zeros((2, 7), jnp.bool_).at[0, 5].set(True)
    final, actual = model.apply(variables, x, episode_starts=starts)
    fresh, expected = model.apply(variables, x[:1, 5:])
    np.testing.assert_allclose(actual[:1, 5:], expected, rtol=2e-5, atol=5e-6)
    np.testing.assert_allclose(actual[1], baseline[1], rtol=2e-5, atol=5e-6)

    def first_example(leaf: jax.Array) -> jax.Array:
        chex.assert_axis_dimension(leaf, 0, 2)
        chex.assert_type(leaf, jnp.int32 if leaf.ndim == 1 else jnp.float32)
        return leaf[:1]

    for state, new_state in zip(final, fresh):
        np.testing.assert_array_equal(state.position, [2, 7])
        assert_carry_close(jax.tree.map(first_example, state), new_state)
        np.testing.assert_array_equal(state.key[0, 2:], 0)
        np.testing.assert_array_equal(state.value[0, 2:], 0)


def test_gradients_through_chunked_cache_match_full_sequence() -> None:
    model = TransformerStack(8, 2, num_heads=2, num_kv_heads=1, max_seq_len=7)
    x = jax.random.normal(jax.random.key(7), (2, 7, 8))
    starts = jnp.zeros((2, 7), jnp.bool_).at[0, 4].set(True)
    params = model.init(jax.random.key(8), x)["params"]
    probe = jax.random.normal(jax.random.key(9), x.shape)

    def loss(parameters: Any, inputs: jax.Array, chunked: bool) -> jax.Array:
        chex.assert_shape(inputs, (2, 7, 8))
        chex.assert_type(inputs, jnp.float32)
        if chunked:
            carry, first = model.apply({"params": parameters}, inputs[:, :3], episode_starts=starts[:, :3])
            _, second = model.apply({"params": parameters}, inputs[:, 3:], carry, starts[:, 3:])
            output = jnp.concatenate((first, second), axis=1)
        else:
            _, output = model.apply({"params": parameters}, inputs, episode_starts=starts)
        return jnp.sum(output * probe)

    full = jax.jit(jax.grad(partial(loss, chunked=False), argnums=(0, 1)))(params, x)
    chunks = jax.jit(jax.grad(partial(loss, chunked=True), argnums=(0, 1)))(params, x)
    for a, b in zip(jax.tree.leaves(full), jax.tree.leaves(chunks)):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, rtol=3e-4, atol=2e-5)
    # Numerical directional derivative also checks the shared backward path.
    direction = jax.random.normal(jax.random.key(10), x.shape)
    epsilon = 1e-3
    numerical = (loss(params, x + epsilon * direction, False) - loss(params, x - epsilon * direction, False)) / (
        2 * epsilon
    )
    np.testing.assert_allclose(jnp.sum(full[1] * direction), numerical, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
def test_precision_empty_initialization_and_parameter_independence(dtype: jax.typing.DTypeLike) -> None:
    model = TransformerStack(16, 2, num_heads=4, max_seq_len=4, dtype=dtype)
    x = jax.random.normal(jax.random.key(11), (2, 3, 16))
    variables = model.init(jax.random.key(12), x[:, :0])
    carry, y = jax.jit(model.apply)(variables, x)
    assert y.dtype == dtype
    assert np.isfinite(y).all()
    for state in carry:
        chex.assert_type((state.key, state.value), dtype)
        chex.assert_type(state.position, jnp.int32)
    chex.assert_type(jax.tree.leaves(variables["params"]), jnp.float32)
    layers = variables["params"]
    assert not np.array_equal(layers["layers_0"]["q_proj"]["kernel"], layers["layers_1"]["q_proj"]["kernel"])
    state, first = model.apply(variables, x[:, :1])
    state, rest = model.apply(variables, x[:, 1:], state)
    np.testing.assert_allclose(
        jnp.concatenate((first, rest), axis=1).astype(jnp.float32), y.astype(jnp.float32), atol=0.04, rtol=0.04
    )
    assert_carry_close(state, carry, tolerance=0.04)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 15},
        {"num_heads": 3},
        {"num_kv_heads": 3},
        {"max_seq_len": 0},
        {"norm_epsilon": 0},
        {"norm_epsilon": float("nan")},
        {"rope_theta": 0},
        {"rope_theta": float("inf")},
        {"dtype": jnp.int32},
        {"attention_implementation": "invalid"},
        {"attention_implementation": "cudnn"},
        {"num_heads": True},
        {"dim": 12, "num_heads": 4},
    ],
)
def test_invalid_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises((ValueError, TypeError, AssertionError)):
        TransformerBlock(**({"dim": 16, "num_heads": 4} | kwargs)).initial_carry(2)


def test_invalid_inputs_and_carries() -> None:
    model = TransformerBlock(16, num_heads=4, max_seq_len=4)
    x = jnp.zeros((1, 2, 16))
    variables = model.init(jax.random.key(13), x)
    carry = model.initial_carry(1)
    with pytest.raises(AssertionError):
        model.apply(variables, x.astype(jnp.int32))
    with pytest.raises(AssertionError):
        model.apply(variables, x, episode_starts=jnp.zeros((1, 2), jnp.int32))
    with pytest.raises(AssertionError):
        model.apply(variables, x, episode_starts=jnp.zeros((2, 1), jnp.bool_))
    with pytest.raises(AssertionError):
        model.apply(variables, x, model.initial_carry(2))
    with pytest.raises(AssertionError):
        model.apply(variables, x, carry._replace(position=jnp.zeros((1,), jnp.float32)))
    with pytest.raises(AssertionError):
        model.apply(variables, x, carry._replace(key=carry.key[:, :2]))
    with pytest.raises(AssertionError):
        model.apply(variables, x, carry._replace(value=carry.value.astype(jnp.bfloat16)))
    with pytest.raises(AssertionError):
        model.apply(variables, x[..., :8])
    with pytest.raises(AssertionError):
        model.apply(variables, x[:, 0])
    with pytest.raises(TypeError):
        model.apply(variables, x, tuple(carry))


@pytest.mark.parametrize("stack,step", [(False, True), (True, False), (True, True)])
def test_sequence_and_step_validate_inputs_before_projections(stack: bool, step: bool) -> None:
    model = TransformerStack(8, 1, num_heads=2) if stack else TransformerBlock(8, num_heads=2)
    shape = (1, 8) if step else (1, 2, 8)
    x = jnp.zeros(shape, jnp.float32)
    method = model.step if step else model.__call__
    # Empty variables ensure invalid metadata is rejected before any projection.
    with pytest.raises(AssertionError):
        model.apply({}, x[..., :4], method=method)
    with pytest.raises(AssertionError):
        model.apply({}, x[None], method=method)
    with pytest.raises(AssertionError):
        model.apply({}, x.astype(jnp.int32), method=method)
    with pytest.raises(AssertionError):
        model.apply({}, x, episode_starts=jnp.zeros(shape[:-1], jnp.int32), method=method)
    with pytest.raises(AssertionError):
        model.apply({}, x, episode_starts=jnp.zeros((2, *shape[1:-1]), jnp.bool_), method=method)


@pytest.mark.parametrize(
    "positions_shape,positions_dtype",
    [
        pytest.param((2,), jnp.int32, id="position-rank"),
        pytest.param((2, 1), jnp.int32, id="position-dimensions"),
        pytest.param((1, 2), jnp.float32, id="position-dtype"),
    ],
)
def test_rope_rejects_invalid_positions(
    positions_shape: tuple[int, ...], positions_dtype: jax.typing.DTypeLike
) -> None:
    with pytest.raises(AssertionError):
        _rope(jnp.zeros((1, 2, 1, 2), jnp.float32), jnp.zeros(positions_shape, positions_dtype), 10000.0)


@pytest.mark.parametrize("steps,window", [(1, 4), (3, 7)])
def test_cudnn_mask_padding_preserves_outputs_and_gradients(
    monkeypatch: pytest.MonkeyPatch, steps: int, window: int
) -> None:
    """Exercise backend routing/padding on CPU; real kernels are tested below."""
    model = TransformerBlock(16, num_heads=2, num_kv_heads=1, max_seq_len=window, dtype=jnp.bfloat16)
    fused = model.clone(attention_implementation="cudnn")
    x = jax.random.normal(jax.random.key(16), (2, steps, 16))
    variables = model.init(jax.random.key(17), x)
    carry = model.apply(variables, x)[0]
    starts = jnp.zeros((2, steps), jnp.bool_).at[0, 0].set(True)
    original_attention = jax.nn.dot_product_attention
    routed = []

    def portable_attention(
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        *,
        mask: jax.Array | None,
        is_causal: bool,
        local_window_size: tuple[int, int] | None,
        implementation: str,
    ) -> jax.Array:
        chex.assert_shape(query, (2, None, 2, 8))
        chex.assert_shape((key, value), (2, None, 1, 8))
        chex.assert_type((query, key, value), jnp.bfloat16)
        if mask is not None:
            chex.assert_shape(mask, (2, 1, query.shape[1], key.shape[1]))
            chex.assert_type(mask, jnp.bool_)
        if implementation == "cudnn":
            assert mask is not None
            assert query.shape[1] % 2 == key.shape[1] % 2 == 0
            routed.append(True)
        return original_attention(
            query,
            key,
            value,
            mask=mask,
            is_causal=is_causal,
            local_window_size=local_window_size,
            implementation="xla",
        )

    monkeypatch.setattr(jax.nn, "dot_product_attention", portable_attention)

    def loss(inputs: jax.Array, network: TransformerBlock) -> tuple[jax.Array, jax.Array]:
        chex.assert_shape(inputs, (2, steps, 16))
        chex.assert_type(inputs, jnp.float32)
        _, output = network.apply(variables, inputs, carry, starts)
        return jnp.sum(output.astype(jnp.float32) ** 2), output

    expected = jax.jit(jax.value_and_grad(partial(loss, network=model), has_aux=True))(x)
    actual = jax.jit(jax.value_and_grad(partial(loss, network=fused), has_aux=True))(x)
    assert routed
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(np.asarray(a, np.float32), np.asarray(b, np.float32), rtol=0.02, atol=0.02)


def assert_gradient_close(actual: jax.Array, expected: jax.Array) -> None:
    """Compare each gradient tensor relative to its own magnitude."""
    chex.assert_equal_shape((actual, expected))
    chex.assert_type((actual, expected), jnp.floating)
    actual_array, expected_array = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    assert np.isfinite(actual_array).all()
    assert np.isfinite(expected_array).all()
    reference_norm = np.linalg.norm(expected_array)
    assert reference_norm > 0, "the probe loss must exercise every gradient tensor"
    relative_error = np.linalg.norm(actual_array - expected_array) / reference_norm
    assert relative_error < 0.03, f"gradient relative L2 error {relative_error} exceeds 3%"


@pytest.mark.parametrize(
    "cached,window,implementation",
    [
        (True, 8, "xla"),
        *[
            pytest.param(
                cached,
                window,
                "cudnn",
                marks=pytest.mark.skipif(
                    not any(d.platform == "gpu" for d in jax.devices()),
                    reason="cuDNN attention requires an NVIDIA GPU",
                ),
            )
            for cached, window in ((False, 8), (False, 9), (True, 8), (True, 9))
        ],
    ],
)
def test_attention_backend_forward_backward_and_decode(cached: bool, window: int, implementation: str) -> None:
    model = TransformerStack(16, 1, num_heads=2, num_kv_heads=1, max_seq_len=window, dtype=jnp.bfloat16)
    backend = model.clone(attention_implementation=implementation)
    x = jax.random.normal(jax.random.key(14), (2, 5, 16))
    probe = jax.random.normal(jax.random.key(18), x.shape) / jnp.sqrt(x.size)
    starts = jnp.zeros((2, 5), jnp.bool_).at[0, 3].set(True) if cached else None
    params = model.init(jax.random.key(15), x)["params"]
    initial = model.apply({"params": params}, x[:, :2])[0] if cached else None

    def loss(
        parameters: Any, inputs: jax.Array, network: TransformerStack
    ) -> tuple[jax.Array, tuple[TransformerStackCarry, jax.Array]]:
        chex.assert_shape(inputs, (2, 5, 16))
        chex.assert_type(inputs, jnp.float32)
        state, y = network.apply({"params": parameters}, inputs, initial, starts)
        # A squared norm is nearly constant after RMSNorm and hides broken
        # attention gradients. A random projection exercises all directions.
        return jnp.sum(y.astype(jnp.float32) * probe), (state, y)

    expected, expected_grad = jax.jit(jax.value_and_grad(partial(loss, network=model), argnums=(0, 1), has_aux=True))(
        params, x
    )
    actual, actual_grad = jax.jit(jax.value_and_grad(partial(loss, network=backend), argnums=(0, 1), has_aux=True))(
        params, x
    )
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(np.asarray(a, np.float32), np.asarray(b, np.float32), rtol=0.05, atol=0.015)
    chex.assert_trees_all_equal_shapes_and_dtypes(actual_grad, expected_grad)
    for a, b in zip(jax.tree.leaves(actual_grad), jax.tree.leaves(expected_grad)):
        assert_gradient_close(a, b)
        # Run this negative control on CPU too, so a permissive comparison or
        # degenerate loss cannot quietly invalidate the GPU-only coverage.
        with pytest.raises(AssertionError, match="gradient relative L2 error"):
            assert_gradient_close(jnp.zeros_like(b), b)
    cache = actual[1][0]
    expected_state, expected_y = model.apply({"params": params}, x[:, 0], cache, method=model.step)
    actual_state, actual_y = jax.jit(partial(backend.apply, method=backend.step))({"params": params}, x[:, 0], cache)
    np.testing.assert_allclose(actual_y.astype(jnp.float32), expected_y.astype(jnp.float32), rtol=0.05, atol=0.015)
    assert_carry_close(actual_state, expected_state, tolerance=0.04)
