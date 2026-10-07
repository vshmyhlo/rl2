"""Small CPU tests of GDN-2 mathematics, upstream wiring, and streaming semantics."""

from dataclasses import replace
from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.test_util import check_grads

from rl2.gdn2 import (
    GatedDeltaNet2,
    GatedDeltaNet2Carry,
    GatedDeltaNet2Config,
    GatedDeltaNet2LM,
    GatedDeltaNet2Stack,
    delta_rule_step,
    gated_delta_rule,
)
from rl2.shape_checker import ShapeChecker

type Parameters = dict[str, Any]


def _assert_tree_close(actual: Any, expected: Any) -> None:
    chex.assert_trees_all_equal_shapes_and_dtypes(actual, expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=2e-6)


def _numpy_rule(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the full transition matrix, independently of the rank-one JAX code."""
    sc = ShapeChecker()
    sc.check([q, k, g, b], "TBHK", np.float32)
    sc.check([v, w], "TBHV", np.float32)
    sc.check(state, "BHKV", np.float32)
    state = state.copy()
    outputs = []
    for t in range(q.shape[0]):
        for batch in range(q.shape[1]):
            for head in range(q.shape[2]):
                kt = k[t, batch, head]
                transition = np.eye(k.shape[-1]) - np.outer(kt, b[t, batch, head] * kt)
                decay = np.diag(np.exp(g[t, batch, head]))
                state[batch, head] = transition @ decay @ state[batch, head] + np.outer(
                    kt, w[t, batch, head] * v[t, batch, head]
                )
        outputs.append(np.einsum("bhk,bhkv->bhv", q[t], state))
    return state, np.stack(outputs)


def test_recurrence_matches_dense_equation_and_chunk_continuation() -> None:
    rng = np.random.default_rng(4)
    q, k, g, b = (rng.normal(size=(4, 1, 2, 2)).astype(np.float32) for _ in range(4))
    v, w = (rng.normal(size=(4, 1, 2, 3)).astype(np.float32) for _ in range(2))
    k /= np.sqrt(np.sum(k**2, axis=-1, keepdims=True) + 1e-6)
    g = -np.abs(g)
    b, w = 1 / (1 + np.exp(-b)), 1 / (1 + np.exp(-w))
    state = rng.normal(size=(1, 2, 2, 3)).astype(np.float32)
    inputs = tuple(jnp.asarray(a) for a in (q, k, v, g, b, w))
    expected_state, expected_y = _numpy_rule(q, k, v, g, b, w, state)
    final, y = jax.jit(gated_delta_rule)(*inputs, jnp.asarray(state))
    np.testing.assert_allclose(final, expected_state, atol=1e-6)
    np.testing.assert_allclose(y, expected_y, atol=1e-6)
    middle, first = gated_delta_rule(*(a[:2] for a in inputs), jnp.asarray(state))
    end, second = gated_delta_rule(*(a[2:] for a in inputs), middle)
    _assert_tree_close(end, final)
    np.testing.assert_allclose(jnp.concatenate((first, second)), y, atol=1e-6)
    empty_state, empty = gated_delta_rule(*(a[:0] for a in inputs), final)
    _assert_tree_close(empty_state, final)
    assert empty.shape == (0, 1, 2, 3)


@pytest.mark.parametrize(
    "erase,write,expected",
    [(0.0, 0.0, 2.0), (1.0, 0.0, 0.0), (0.0, 1.0, 5.0), (2.0, 0.0, -2.0)],
    ids=["decay-only", "erase-only", "write-only", "negative-eigenvalue"],
)
def test_independent_erase_and_write(erase: float, write: float, expected: float) -> None:
    one = jnp.ones((1, 1, 1), jnp.float32)
    state, y = delta_rule_step(4 * one[..., None], one, one, 3 * one, -jnp.log(2.0) * one, erase * one, write * one)
    np.testing.assert_allclose(state, expected, atol=1e-6)
    np.testing.assert_allclose(y, expected, atol=1e-6)


def test_equal_scalar_gates_recover_kda_and_gradients() -> None:
    state = jnp.arange(6, dtype=jnp.float32).reshape(1, 1, 2, 3) / 10
    k = jnp.array([[[0.6, 0.8]]])
    q = k / jnp.sqrt(2.0)
    v = jnp.array([[[0.3, -0.2, 0.1]]])
    g = jnp.array([[[-0.1, -0.3]]])
    b, w = jnp.full_like(k, 0.4), jnp.full_like(v, 0.4)
    decayed = jnp.exp(g)[..., None] * state
    residual = v - jnp.einsum("bhk,bhkv->bhv", k, decayed)
    expected = decayed + 0.4 * k[..., None] * residual[..., None, :]
    actual, _ = delta_rule_step(state, q, k, v, g, b, w)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    # Check derivatives of every input, including incoming state, against finite differences.
    check_grads(
        delta_rule_step, (state, q, k, v, g, b, w), order=1, modes=["fwd", "rev"], atol=2e-3, rtol=2e-3, eps=1e-3
    )


def _numpy_mixer(
    config: GatedDeltaNet2Config,
    params: Parameters,
    x: np.ndarray,
) -> tuple[GatedDeltaNet2Carry, np.ndarray]:
    """Upstream projection/conv/gate/norm recipe with NumPy matrix recurrence."""
    c = config
    sc = ShapeChecker(D=c.hidden_size)
    sc.check(x, "TBD", np.float32)

    def linear(a: np.ndarray, name: str) -> np.ndarray:
        sc = ShapeChecker()
        sc.check(a, "TBI", np.float32)
        p = params[name]
        return a @ np.asarray(p["kernel"]) + np.asarray(p.get("bias", np.float32(0)))

    def silu(a: np.ndarray) -> np.ndarray:
        return a / (1 + np.exp(-a))

    projections, histories = [], []
    for name in ("q", "k", "v"):
        raw = linear(x, f"{name}_proj")
        history = c.conv_size - 1 if c.use_short_conv else 0
        padded = np.concatenate((np.zeros((history, *raw.shape[1:]), np.float32), raw))
        if c.use_short_conv:
            kernel = np.asarray(params[f"{name}_conv_kernel"])
            projected = np.stack([np.einsum("sbd,sd->bd", padded[t : t + c.conv_size], kernel) for t in range(len(x))])
            projected += np.asarray(params.get(f"{name}_conv_bias", np.float32(0)))
        else:
            projected = raw
        projections.append(silu(projected))
        histories.append(np.swapaxes(padded[len(x) :], 0, 1))
    q, k, v = projections
    key_shape = (*x.shape[:2], c.num_heads, c.head_dim)
    value_shape = (*x.shape[:2], c.value_heads, c.value_head_dim)
    q, k = q.reshape(key_shape), k.reshape(key_shape)
    q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6) * np.sqrt(np.float32(c.head_dim))
    k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    f = linear(linear(x, "f_proj_in"), "f_proj_out")
    g = -np.repeat(np.exp(np.asarray(params["A_log"])), c.head_dim) * np.logaddexp(0, f + np.asarray(params["dt_bias"]))
    b = 1 / (1 + np.exp(-linear(x, "b_proj")))
    w = 1 / (1 + np.exp(-linear(x, "w_proj")))
    if c.allow_neg_eigval:
        b *= 2
    q, k, g, b = (np.repeat(a.reshape(key_shape), c.value_heads // c.num_heads, axis=2) for a in (q, k, g, b))
    state, y = _numpy_rule(
        q,
        k,
        v.reshape(value_shape),
        g,
        b,
        w.reshape(value_shape),
        np.zeros((x.shape[1], c.value_heads, c.head_dim, c.value_head_dim), np.float32),
    )
    gate = linear(linear(x, "g_proj_in"), "g_proj_out").reshape(value_shape)
    y = y / np.sqrt(np.mean(y * y, axis=-1, keepdims=True) + c.norm_eps)
    y = y * np.asarray(params["o_norm_scale"]) * silu(gate)
    y = linear(y.reshape((*x.shape[:2], -1)), "o_proj")
    return GatedDeltaNet2Carry(state, *histories), y


@pytest.mark.parametrize(
    "config",
    [
        GatedDeltaNet2Config(
            hidden_size=4,
            head_dim=2,
            num_heads=2,
            num_v_heads=4,
            expand_v=1.5,
            conv_size=3,
            conv_bias=True,
            allow_neg_eigval=True,
        ),
        GatedDeltaNet2Config(hidden_size=4, head_dim=2, num_heads=2, use_short_conv=False),
        GatedDeltaNet2Config(hidden_size=4, head_dim=2, num_heads=2, conv_size=1),
    ],
    ids=["grouped-values-channel-gates", "without-convolution", "single-tap-convolution"],
)
def test_mixer_matches_numpy_reference(config: GatedDeltaNet2Config) -> None:
    model = GatedDeltaNet2(config)
    x = jax.random.normal(jax.random.key(2), (4, 1, 4))
    params = model.init(jax.random.key(3), x)["params"]
    # Nontrivial output norm weights/bias exercise behavior hidden by default initialization.
    params["o_norm_scale"] = jnp.linspace(0.8, 1.2, config.value_head_dim)
    params["g_proj_out"]["bias"] = jnp.linspace(-0.2, 0.3, config.value_heads * config.value_head_dim)
    expected_carry, expected = _numpy_mixer(config, params, np.asarray(x))
    carry, y = jax.jit(model.apply)({"params": params}, x)
    np.testing.assert_allclose(y, expected, rtol=3e-5, atol=1e-7)
    _assert_tree_close(carry, expected_carry)
    assert np.all((np.exp(params["A_log"]) >= 1) & (np.exp(params["A_log"]) <= 16))
    dt = jax.nn.softplus(params["dt_bias"])
    assert np.all((dt >= 0.001) & (dt <= 0.1))


def test_mixer_streaming_resets_padding_causality_and_empty_input() -> None:
    model = GatedDeltaNet2(GatedDeltaNet2Config(hidden_size=4, head_dim=2, num_heads=1, conv_size=3))
    x = jax.random.normal(jax.random.key(4), (4, 2, 4))
    variables = model.init(jax.random.key(5), x)
    starts = jnp.zeros((4, 2), jnp.bool_).at[2, 0].set(True).at[1, 1].set(True)
    mask = jnp.ones((4, 2), jnp.bool_).at[1, 1].set(False)
    padded_x = x.at[1, 1].set(jnp.nan)
    final, y = jax.jit(model.apply)(variables, padded_x, None, starts, mask)
    carry, first = model.apply(variables, padded_x[:2], episode_starts=starts[:2], mask=mask[:2])
    carry, second = model.apply(variables, padded_x[2:], carry, starts[2:], mask[2:])
    _assert_tree_close(carry, final)
    np.testing.assert_allclose(jnp.concatenate((first, second)), y, atol=1e-7)
    carry = model.initial_carry(2)
    outputs = []
    step = jax.jit(partial(model.apply, method=model.step))
    for i in range(len(x)):
        carry, output = step(variables, padded_x[i], carry, starts[i], mask[i])
        outputs.append(output)
    _assert_tree_close(carry, final)
    np.testing.assert_allclose(jnp.stack(outputs), y, atol=1e-7)
    _, reset_output = model.apply(variables, x[2:, :1])
    np.testing.assert_allclose(reset_output[:, 0], y[2:, 0], atol=1e-7)
    # Padding skips both the convolution and recurrence, including a masked reset.
    compressed_carry, compressed = model.apply(variables, x[jnp.array([0, 2, 3]), 1:])
    np.testing.assert_allclose(compressed[:, 0], y[jnp.array([0, 2, 3]), 1], atol=1e-7)
    _assert_tree_close(compressed_carry, GatedDeltaNet2Carry(*(leaf[1:] for leaf in final)))
    np.testing.assert_array_equal(y[1, 1], 0)
    _, changed = model.apply(variables, x.at[2:].add(100))
    _, original = model.apply(variables, x)
    np.testing.assert_array_equal(changed[:2], original[:2])
    empty_carry, empty = model.apply(variables, x[:0], final)
    _assert_tree_close(empty_carry, final)
    assert empty.shape == (0, 2, 4)
    unchanged, zeros = model.apply(variables, x, final, jnp.ones_like(mask), jnp.zeros_like(mask))
    _assert_tree_close(unchanged, final)
    np.testing.assert_array_equal(zeros, 0)


def test_stack_matches_explicit_residual_blocks() -> None:
    config = GatedDeltaNet2Config(hidden_size=4, head_dim=2, num_heads=1, conv_size=2)
    model = GatedDeltaNet2Stack(config, num_layers=1, intermediate_size=6)
    x = jax.random.normal(jax.random.key(7), (4, 1, 4))
    variables = model.init(jax.random.key(8), x)
    p = variables["params"]

    def norm(a: np.ndarray, name: str) -> np.ndarray:
        sc = ShapeChecker(D=4)
        sc.check(a, "TBD", np.float32)
        return a / np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + config.norm_eps) * np.asarray(p[name]["scale"])

    state, mixed = _numpy_mixer(config, p["mixer_0"], norm(np.asarray(x), "norm_mixer_0"))
    residual = np.asarray(x) + mixed
    normalized = norm(residual, "norm_mlp_0")
    gate = normalized @ np.asarray(p["mlp_gate_0"]["kernel"])
    value = normalized @ np.asarray(p["mlp_up_0"]["kernel"])
    hidden = gate / (1 + np.exp(-gate)) * value
    expected = norm(residual + hidden @ np.asarray(p["mlp_down_0"]["kernel"]), "final_norm")
    carry, y = model.apply(variables, x)
    np.testing.assert_allclose(y, expected, rtol=3e-5, atol=1e-6)
    _assert_tree_close(carry, (state,))
    step_carry, single = model.apply(variables, x[0], method=model.step)
    np.testing.assert_allclose(single, y[0], atol=1e-6)
    assert len(step_carry) == 1
    with pytest.raises(ValueError, match="one state per layer"):
        model.apply(variables, x, ())


def test_language_model_bfloat16_streaming_and_training() -> None:
    config = GatedDeltaNet2Config(hidden_size=4, head_dim=2, num_heads=1, conv_size=2, dtype=jnp.bfloat16)
    model = GatedDeltaNet2LM(config, num_layers=2, intermediate_size=6, vocab_size=7)
    tokens = jnp.array([[1], [2], [3], [4]], jnp.int32)
    starts = jnp.array([[False], [False], [True], [False]])
    variables = model.init(jax.random.key(6), tokens)
    final, logits = jax.jit(model.apply)(variables, tokens, None, starts)
    assert logits.shape == (4, 1, 7)
    chex.assert_type((logits, *jax.tree.leaves(final), *jax.tree.leaves(variables)), jnp.float32)
    carry = model.initial_carry(1)
    step = jax.jit(partial(model.apply, method=model.step))
    outputs = []
    for i in range(4):
        carry, output = step(variables, tokens[i], carry, starts[i])
        outputs.append(output)
    _assert_tree_close(carry, final)
    np.testing.assert_allclose(jnp.stack(outputs), logits, atol=1e-6)
    _, fresh = model.apply(variables, tokens[2:])
    np.testing.assert_allclose(fresh, logits[2:], atol=1e-6)
    empty_carry, empty = model.apply(variables, tokens[:0], final)
    _assert_tree_close(empty_carry, final)
    assert empty.shape == (0, 1, 7)
    # Padding is ignored even if external token IDs would be out of vocabulary.
    unchanged, zero = model.apply(variables, jnp.full_like(tokens, -1), final, mask=jnp.zeros(tokens.shape, jnp.bool_))
    _assert_tree_close(unchanged, final)
    np.testing.assert_array_equal(zero, 0)
    # Every layer must receive its own parameters.
    p = variables["params"]["backbone"]
    assert not np.array_equal(p["mixer_0"]["q_proj"]["kernel"], p["mixer_1"]["q_proj"]["kernel"])

    def loss(params: Parameters) -> jax.Array:
        _, predictions = model.apply({"params": params}, tokens, episode_starts=starts)
        return optax.softmax_cross_entropy_with_integer_labels(predictions[:-1], tokens[1:]).mean()

    value, gradients = jax.jit(jax.value_and_grad(loss))(variables["params"])
    assert np.isfinite(value)
    for gradient in jax.tree.leaves(gradients):
        assert np.all(np.isfinite(gradient))
    for layer in ("mixer_0", "mixer_1"):
        for gate in ("b_proj", "w_proj", "f_proj_in"):
            assert np.any(gradients["backbone"][layer][gate]["kernel"] != 0)
    optimizer = optax.sgd(1e-4)
    updates, _ = optimizer.update(gradients, optimizer.init(variables["params"]))
    assert loss(optax.apply_updates(variables["params"], updates)) < value


@pytest.mark.parametrize(
    "changes",
    [
        {"hidden_size": 0},
        {"head_dim": 0},
        {"num_heads": 0},
        {"conv_size": 0},
        {"num_v_heads": 1},
        {"num_v_heads": 3},
        {"expand_v": 0.7},
        {"expand_v": float("nan")},
        {"norm_eps": 0},
        {"dtype": jnp.int32},
    ],
    ids=[
        "hidden-width",
        "head-width",
        "head-count",
        "conv-width",
        "fewer-value-heads",
        "nondivisible-heads",
        "fractional-value-width",
        "nonfinite-expansion",
        "epsilon",
        "dtype",
    ],
)
def test_invalid_configuration(changes: dict[str, Any]) -> None:
    config = GatedDeltaNet2Config(hidden_size=4, head_dim=2, num_heads=2)
    with pytest.raises(ValueError):
        replace(config, **changes)


def test_array_shape_dtype_and_carry_validation() -> None:
    q = jnp.ones((1, 1, 2), jnp.float32)
    state = jnp.zeros((1, 1, 2, 2), jnp.float32)
    with pytest.raises(AssertionError):
        delta_rule_step(state, q, q[..., :1], q, q, q, q)
    with pytest.raises(AssertionError):
        delta_rule_step(state.astype(jnp.bfloat16), q, q, q, q, q, q)
    model = GatedDeltaNet2(GatedDeltaNet2Config(hidden_size=4, head_dim=2, num_heads=1))
    x = jnp.zeros((1, 1, 4), jnp.float32)
    variables = model.init(jax.random.key(1), x)
    with pytest.raises(AssertionError):
        model.apply(variables, x, mask=jnp.ones((1, 1), jnp.int32))
    with pytest.raises(AssertionError):
        model.apply(variables, x, model.initial_carry(2))
    with pytest.raises(AssertionError):
        model.apply(variables, x[..., :3])
