"""Optional real-GPU parity checks; skipped when CUDA or Triton is unavailable."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl2.gdn2.core import gated_delta_rule

pytestmark = pytest.mark.skipif(not any(d.platform == "gpu" for d in jax.devices()), reason="requires NVIDIA GPU")

type Arrays = tuple[jax.Array, ...]
type Rule = Callable[..., tuple[jax.Array, jax.Array]]


def _inputs(time: int = 65, key: int = 32, value: int = 32) -> Arrays:
    rng = np.random.default_rng(3)
    q, k = (jnp.asarray(rng.normal(size=(time, 2, 2, key)).astype(np.float32)) for _ in range(2))
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True) / key**0.5
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    v = jnp.asarray(rng.normal(size=(time, 2, 2, value)).astype(np.float32))
    g = jnp.asarray(-rng.uniform(0.005, 0.08, q.shape).astype(np.float32))
    b = jnp.asarray(rng.uniform(0, 1, q.shape).astype(np.float32))
    w = jnp.asarray(rng.uniform(0, 1, v.shape).astype(np.float32))
    state = jnp.asarray(rng.normal(size=(2, 2, key, value)).astype(np.float32) * 0.1)
    return q, k, v, g, b, w, state


def _loss(rule: Rule, *args: jax.Array) -> jax.Array:
    state, y = rule(*args)
    # Exercise both output cotangents, including the carried state between chunks.
    return jnp.sum(jnp.sin(y)) + jnp.sum(jnp.sin(state)) * 0.3


@pytest.mark.parametrize("head_dim", [32, 128], ids=["small", "checkpoint-head-shared-memory-regression"])
def test_chunk_forward_and_backward(head_dim: int) -> None:
    pytest.importorskip("jax_triton")
    from rl2.gdn2.triton_backend import chunk_gated_delta_rule

    args = _inputs(key=head_dim, value=head_dim)
    expected = jax.jit(gated_delta_rule)(*args)
    actual = jax.jit(chunk_gated_delta_rule)(*args)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, want, atol=3e-5, rtol=3e-4)

    def reference_loss(*a: jax.Array) -> jax.Array:
        return _loss(gated_delta_rule, *a)

    def triton_loss(*a: jax.Array) -> jax.Array:
        return _loss(chunk_gated_delta_rule, *a)

    expected_grads = jax.jit(jax.grad(reference_loss, argnums=tuple(range(7))))(*args)
    actual_grads = jax.jit(jax.grad(triton_loss, argnums=tuple(range(7))))(*args)
    for i, (got, want) in enumerate(zip(actual_grads, expected_grads, strict=True)):
        np.testing.assert_allclose(got, want, atol=5e-5, rtol=8e-4, err_msg=f"gradient {i}")


def test_fused_recurrence_and_streaming_gradients() -> None:
    pytest.importorskip("jax_triton")
    from rl2.gdn2.triton_backend import fused_recurrent_gated_delta_rule

    args = _inputs(time=3, key=128, value=128)
    expected = jax.jit(gated_delta_rule)(*args)
    actual = jax.jit(fused_recurrent_gated_delta_rule)(*args)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, want, atol=3e-6, rtol=3e-5)

    def streamed(*a: jax.Array) -> jax.Array:
        state, first = fused_recurrent_gated_delta_rule(*(x[:1] for x in a[:-1]), a[-1])
        state, rest = fused_recurrent_gated_delta_rule(*(x[1:] for x in a[:-1]), state)
        return jnp.sum(jnp.sin(jnp.concatenate((first, rest)))) + jnp.sum(jnp.sin(state)) * 0.3

    def reference(*a: jax.Array) -> jax.Array:
        return _loss(gated_delta_rule, *a)

    actual_grads = jax.jit(jax.grad(streamed, argnums=tuple(range(7))))(*args)
    expected_grads = jax.jit(jax.grad(reference, argnums=tuple(range(7))))(*args)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(got, want, atol=5e-5, rtol=8e-4)


@pytest.mark.parametrize("size", [1, 4], ids=["no-history", "history-longer-than-input"])
def test_short_conv_outputs_and_gradients(size: int) -> None:
    pytest.importorskip("jax_triton")
    from rl2.gdn2.triton_backend import short_conv

    rng = np.random.default_rng(18)
    args = tuple(
        jnp.asarray(rng.normal(size=s).astype(np.float32)) for s in [(2, 2, 7), (2, size - 1, 7), (size, 7), (7,)]
    )

    def reference(x: jax.Array, history: jax.Array, weight: jax.Array, bias: jax.Array) -> tuple[jax.Array, jax.Array]:
        joined = jnp.concatenate((history, x.swapaxes(0, 1)), axis=1)
        z = sum(joined[:, c : c + x.shape[0]] * weight[c] for c in range(size)) + bias
        return joined[:, x.shape[0] :], jax.nn.silu(z).swapaxes(0, 1)

    for got, want in zip(jax.jit(short_conv)(*args), reference(*args), strict=True):
        np.testing.assert_allclose(got, want, atol=3e-6, rtol=3e-5)

    def loss(rule: Rule, *a: jax.Array) -> jax.Array:
        state, y = rule(*a)
        return jnp.sum(jnp.sin(y)) + jnp.sum(jnp.sin(state))

    def actual_loss(*a: jax.Array) -> jax.Array:
        return loss(short_conv, *a)

    def expected_loss(*a: jax.Array) -> jax.Array:
        return loss(reference, *a)

    actual_grads = jax.jit(jax.grad(actual_loss, argnums=(0, 1, 2, 3)))(*args)
    expected_grads = jax.grad(expected_loss, argnums=(0, 1, 2, 3))(*args)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(got, want, atol=3e-6, rtol=3e-5)


def test_gated_norm_outputs_and_gradients() -> None:
    pytest.importorskip("jax_triton")
    from rl2.gdn2.triton_backend import gated_rms_norm

    rng = np.random.default_rng(19)
    args = tuple(jnp.asarray(rng.normal(size=s).astype(np.float32)) for s in [(3, 2, 2, 7), (3, 2, 2, 7), (7,)])

    def reference(x: jax.Array, gate: jax.Array, weight: jax.Array) -> jax.Array:
        return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-5) * weight * jax.nn.silu(gate)

    def actual(x: jax.Array, gate: jax.Array, weight: jax.Array) -> jax.Array:
        return gated_rms_norm(x, gate, weight, 1e-5)

    np.testing.assert_allclose(jax.jit(actual)(*args), reference(*args), atol=3e-6, rtol=3e-5)

    def actual_loss(*a: jax.Array) -> jax.Array:
        return jnp.sum(jnp.sin(actual(*a)))

    def expected_loss(*a: jax.Array) -> jax.Array:
        return jnp.sum(jnp.sin(reference(*a)))

    actual_grads = jax.jit(jax.grad(actual_loss, argnums=(0, 1, 2)))(*args)
    expected_grads = jax.grad(expected_loss, argnums=(0, 1, 2))(*args)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        np.testing.assert_allclose(got, want, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("mode", ["plain", "masked", "reset"])
def test_mixer_streaming_and_padding(mode: str) -> None:
    pytest.importorskip("jax_triton")
    from rl2.gdn2 import GatedDeltaNet2, GatedDeltaNet2Config

    config = GatedDeltaNet2Config(
        hidden_size=16,
        head_dim=16,
        num_heads=1,
        num_v_heads=2,
        expand_v=1.5,
        conv_size=3,
        conv_bias=True,
        allow_neg_eigval=True,
    )
    reference = GatedDeltaNet2(config)
    model = reference.clone(backend="triton")
    x = jnp.asarray(np.random.default_rng(11).normal(size=(5, 2, 16)).astype(np.float32))
    variables = reference.init(jax.random.key(12), x, method=reference._forward)
    mask = None if mode == "plain" else jnp.array([[1, 1], [0, 1], [1, 0], [1, 1], [0, 0]], jnp.bool_)
    starts = jnp.array([[0, 0], [1, 0], [0, 0], [1, 1], [1, 0]], jnp.bool_) if mode == "reset" else None
    if mask is not None:
        x = jnp.where(mask[..., None], x, jnp.nan)
    expected = jax.jit(reference.apply, static_argnames=("method",))(
        variables, x, episode_starts=starts, mask=mask, method=reference._forward
    )
    actual = jax.jit(model.apply, static_argnames=("method",))(
        variables, x, episode_starts=starts, mask=mask, method=model._forward
    )
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(got, want, atol=3e-6, rtol=5e-4)
    step = jax.jit(model.apply, static_argnames=("method",))
    state = model.initial_carry(2)
    outputs = []
    for t in range(x.shape[0]):
        state, y = step(
            variables,
            x[t : t + 1],
            state,
            None if starts is None else starts[t : t + 1],
            None if mask is None else mask[t : t + 1],
            method=model._forward,
        )
        outputs.append(y[0])
    streamed = state, jnp.stack(outputs)
    for got, want in zip(jax.tree.leaves(streamed), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(got, want, atol=3e-6, rtol=5e-4)

    if mode == "masked":
        lengths = jnp.array([3, 0], jnp.int32)
        sequence = jnp.nan_to_num(x).swapaxes(0, 1)
        sequence = jnp.where(jnp.arange(5)[None, :, None] < lengths[:, None, None], sequence, jnp.nan)
        expected = jax.jit(reference.apply)(variables, sequence, lengths)
        actual = jax.jit(model.apply)(variables, sequence, lengths)
        for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(got, want, atol=3e-6, rtol=5e-4)
        state = None
        outputs = []
        for t in range(sequence.shape[1]):
            state, y = step(variables, sequence[:, t], t < lengths, state, method=model.step)
            outputs.append(y)
        for got, want in zip(
            jax.tree.leaves((state, jnp.stack(outputs, axis=1))), jax.tree.leaves(actual), strict=True
        ):
            np.testing.assert_allclose(got, want, atol=3e-6, rtol=5e-4)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_lm_parameter_gradients(dtype: jax.typing.DTypeLike) -> None:
    pytest.importorskip("jax_triton")
    from rl2.gdn2 import GatedDeltaNet2Config, GatedDeltaNet2LM
    from rl2.gdn2.checkpoints import Parameters

    config = GatedDeltaNet2Config(hidden_size=16, head_dim=16, num_heads=1, dtype=dtype)
    reference = GatedDeltaNet2LM(config, num_layers=1, intermediate_size=24, vocab_size=19)
    model = reference.clone(backend="triton")
    tokens = jnp.array([[1], [2], [3]], jnp.int32)
    variables = reference.init(jax.random.key(5), tokens)

    def loss(module: GatedDeltaNet2LM, params: Parameters) -> jax.Array:
        state, logits = module.apply(params, tokens)
        return jnp.mean(jax.nn.log_softmax(logits)[..., 4]) + 0.1 * jnp.sum(state[0].state)

    def actual_loss(params: Parameters) -> jax.Array:
        return loss(model, params)

    def expected_loss(params: Parameters) -> jax.Array:
        return loss(reference, params)

    got = jax.jit(jax.value_and_grad(actual_loss))(variables)
    want = jax.jit(jax.value_and_grad(expected_loss))(variables)
    tolerance = 2e-4 if dtype == jnp.bfloat16 else 4e-6
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=2e-3)


def test_empty_inputs_and_dimension_validation() -> None:
    pytest.importorskip("jax_triton")
    from rl2.gdn2.triton_backend import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    args = _inputs(time=0)
    for rule in (chunk_gated_delta_rule, fused_recurrent_gated_delta_rule):
        state, output = rule(*args)
        np.testing.assert_array_equal(state, args[-1])
        assert output.shape == args[2].shape
        with pytest.raises(ValueError, match="head_dim <= 256"):
            rule(*_inputs(time=1, key=257))
