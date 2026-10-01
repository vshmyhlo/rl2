from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rl2.mamba3 import Mamba3, Mamba3Carry, _ssm_step


def assert_carry_close(actual: Mamba3Carry, expected: Mamba3Carry) -> None:
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("rank", [1, 3])
@pytest.mark.parametrize("trap", [0.0, 0.37, 1.0])
def test_recurrence_matches_independent_complex_ssm(rank: int, trap: float) -> None:
    """Compare rotary B/C implementation to an explicitly complex transition.

    This checks the actual discretization, not just scan/step agreement: both
    endpoints use the *current* dt and lambda, and the previous input rotates
    with the old state. Leave half the state unrotated to check partial RoPE.
    """
    rng = np.random.default_rng(42)
    model = Mamba3(4, d_state=4, headdim=2, mimo_rank=rank)
    carry = model.initial_carry(2)
    weights = rng.normal(size=(4, rank, 2)).astype(np.float32)
    state = np.zeros((2, 4, 2, 2), np.complex128)
    previous = np.zeros_like(state)
    for _ in range(5):
        x = rng.normal(size=(2, 4, 2)).astype(np.float32)
        b = rng.normal(size=(2, 4, rank, 4)).astype(np.float32)
        c = rng.normal(size=b.shape).astype(np.float32)
        dt = rng.uniform(0.01, 0.7, (2, 4)).astype(np.float32)
        a = -rng.uniform(0.1, 2, (2, 4)).astype(np.float32)
        delta = rng.uniform(-2, 2, (2, 4, 1)).astype(np.float32)
        starts = jnp.zeros((2,), dtype=jnp.bool_)
        inputs = (x, b, c, dt, a, np.full_like(dt, trap), delta, starts)
        carry, y = _ssm_step(carry, jax.tree.map(jnp.asarray, inputs), jnp.asarray(weights))
        # Adjacent real pairs represent one complex coordinate. Upstream's
        # positive B/C rotation corresponds to a negative local state rotation.
        complex_b = b[..., ::2].astype(np.float64) + 1j * b[..., 1::2]
        complex_c = c[..., ::2].astype(np.float64) + 1j * c[..., 1::2]
        drive = np.einsum("bhrn,bhp,hrp->bhpn", complex_b, x, weights)
        phase = np.concatenate((delta, np.zeros_like(delta)), axis=-1)
        transition = np.exp((dt * a)[..., None, None] - 1j * phase[..., None, :])
        state = transition * (state + ((1 - trap) * dt)[..., None, None] * previous)
        state += (trap * dt)[..., None, None] * drive
        expected = np.einsum("bhpn,bhrn->bhrp", state, complex_c.conj()).real
        np.testing.assert_allclose(y, expected, rtol=2e-5, atol=3e-6)
        previous = drive


@pytest.mark.parametrize("rank,groups,norm", [(1, 1, False), (3, 2, True)])
def test_sequence_chunk_step_and_jit_agree(rank: int, groups: int, norm: bool) -> None:
    model = Mamba3(8, d_state=8, headdim=4, mimo_rank=rank, ngroups=groups, outproj_norm=norm)
    x = jax.random.normal(jax.random.key(0), (7, 2, 8))
    variables = model.init(jax.random.key(1), x)
    final, expected = jax.jit(model.apply)(variables, x)
    carry, first = model.apply(variables, x[:3])
    carry, second = model.apply(variables, x[3:], carry)
    np.testing.assert_allclose(jnp.concatenate((first, second)), expected, rtol=2e-5, atol=2e-6)
    assert_carry_close(carry, final)
    carry = model.initial_carry(2)
    outputs = []
    for token in x:
        carry, y = model.apply(variables, token, carry, method=model.step)
        outputs.append(y)
    np.testing.assert_allclose(jnp.stack(outputs), expected, rtol=2e-5, atol=2e-6)
    assert_carry_close(carry, final)
    assert final.state.shape == (2, 4, 4, 8)
    # Empty chunks preserve history and the output contract under JIT.
    empty_carry, empty = jax.jit(model.apply)(variables, x[:0], final)
    assert empty.shape == (0, 2, 8)
    assert_carry_close(empty_carry, final)


@pytest.mark.parametrize("rank", [1, 2])
def test_resets_clear_all_history_without_affecting_other_examples(rank: int) -> None:
    model = Mamba3(8, d_state=8, headdim=4, mimo_rank=rank)
    x = jax.random.normal(jax.random.key(2), (6, 2, 8))
    variables = model.init(jax.random.key(3), x)
    starts = jnp.zeros((6, 2), dtype=jnp.bool_).at[3, 0].set(True)
    final, y = jax.jit(model.apply)(variables, x, None, starts)
    fresh, fresh_y = model.apply(variables, x[3:, :1])
    continuous, continuous_y = model.apply(variables, x[:, 1:])
    np.testing.assert_allclose(y[3:, :1], fresh_y, atol=2e-6)
    np.testing.assert_allclose(y[:, 1:], continuous_y, atol=2e-6)
    for actual, reset, kept in zip(final, fresh, continuous):
        np.testing.assert_allclose(actual[:1], reset, atol=2e-6)
        np.testing.assert_allclose(actual[1:], kept, atol=2e-6)
    # Reset an incoming nonzero carry, including previous key/value and phase.
    reset, reset_y = model.apply(variables, x[0], final, jnp.ones(2, dtype=jnp.bool_), method=model.step)
    zero, zero_y = model.apply(variables, x[0], method=model.step)
    assert_carry_close(reset, zero)
    np.testing.assert_allclose(reset_y, zero_y, atol=2e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_causal_gradients_and_training(dtype: jax.typing.DTypeLike) -> None:
    model = Mamba3(8, d_state=8, headdim=4, mimo_rank=2, outproj_norm=True, dtype=dtype)
    x = jax.random.normal(jax.random.key(4), (5, 2, 8))
    params = model.init(jax.random.key(5), x)["params"]

    def loss_fn(parameters: Any, inputs: jax.Array) -> jax.Array:
        _, y = model.apply({"params": parameters}, inputs)
        return jnp.mean(jnp.square(y.astype(jnp.float32) - 1))

    loss, (grads, input_grads) = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))(params, x)
    assert np.isfinite(loss)
    for grad in jax.tree.leaves((grads, input_grads)):
        assert np.isfinite(grad).all()
        assert np.any(np.asarray(grad) != 0)

    def gradient_step(grad: jax.Array) -> jax.Array:
        return -0.01 * grad

    updated = optax.apply_updates(params, jax.tree.map(gradient_step, grads))
    assert float(loss_fn(updated, x)) < float(loss)
    carry, y = model.apply({"params": params}, x)
    assert y.dtype == dtype
    assert all(leaf.dtype == jnp.float32 for leaf in carry)

    def prefix_loss(inputs: jax.Array) -> jax.Array:
        _, outputs = model.apply({"params": params}, inputs)
        return outputs[:2].astype(jnp.float32).sum()

    grad = jax.grad(prefix_loss)(x)
    np.testing.assert_array_equal(grad[2:], 0)
    _, changed = model.apply({"params": params}, x.at[2:].set(100))
    np.testing.assert_array_equal(changed[:2], y[:2])


@pytest.mark.parametrize(
    "options",
    [
        {"d_model": 7, "headdim": 4},
        {"d_state": 3},
        {"d_state": 2, "rope_fraction": 0.5},
        {"mimo_rank": 0},
        {"ngroups": 3},
        {"dt_min": -1},
        {"dt_max": 0.0001},
        {"a_floor": 0},
    ],
)
def test_invalid_configuration(options: dict[str, Any]) -> None:
    settings = {"d_model": 8, "d_state": 8, "headdim": 4, **options}
    with pytest.raises(ValueError):
        Mamba3(**settings).initial_carry(2)


def test_invalid_input_and_carry_shapes() -> None:
    model = Mamba3(8, d_state=8, headdim=4)
    x = jnp.zeros((3, 2, 8))
    variables = model.init(jax.random.key(0), x)
    with pytest.raises(ValueError, match="x must have shape"):
        model.apply(variables, x[0])
    with pytest.raises(ValueError, match="carry shapes"):
        model.apply(variables, x, model.initial_carry(1))
    with pytest.raises(ValueError, match="episode_starts"):
        model.apply(variables, x, episode_starts=jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="step episode_starts"):
        model.apply(variables, x[0], episode_starts=jnp.zeros((1, 2)), method=model.step)
