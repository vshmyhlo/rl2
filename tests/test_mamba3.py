from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.traverse_util import flatten_dict, unflatten_dict

from rl2.mamba3 import Mamba3, Mamba3Carry, Mamba3Stack, Mamba3StackCarry, _ssm_step
from rl2.sequence_model import RecurrentSequenceModel
from rl2.shape_checker import ShapeChecker


def assert_carry_close(actual: Mamba3Carry, expected: Mamba3Carry) -> None:
    chex.assert_trees_all_equal_shapes_and_dtypes(actual, expected)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "model",
    [
        Mamba3(4, d_state=4, expand=1, headdim=2),
        Mamba3Stack(4, 2, d_state=4, expand=1, headdim=2, d_intermediate=0),
    ],
    ids=["mixer", "stack"],
)
def test_incoming_carry_gradients_stop_at_resets(model: Mamba3 | Mamba3Stack) -> None:
    x = jax.random.normal(jax.random.key(30), (2, 2, 4))
    starts = jnp.array([[True, False], [False, False]])
    initial = model.initial_carry(num_envs=2)
    variables = model.init(jax.random.key(31), x, initial, starts)
    incoming, _ = model.apply(variables, x, initial, jnp.zeros_like(starts))

    def loss(carry: Mamba3Carry | Mamba3StackCarry) -> jax.Array:
        final, output = model.apply(variables, x, carry, starts)
        sc = ShapeChecker(T=2, B=2, D=4)
        sc.check(output, "TBD", jnp.float32)
        return output.sum() + sum(leaf.sum() for leaf in jax.tree.leaves(final))

    gradients = jax.jit(jax.grad(loss))(incoming)
    for gradient in jax.tree.leaves(gradients):
        assert np.isfinite(gradient).all()
        np.testing.assert_array_equal(gradient[0], 0)
        assert np.any(np.asarray(gradient[1]) != 0)


# Check both discretization endpoints once, and both rotation layouts in the interior.
@pytest.mark.parametrize("rank,trap", [(1, 0.0), (1, 1.0), (1, 0.37), (3, 0.37)])
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
        # SISO uses adjacent pairs, MIMO pairs corresponding half-vectors.
        # Positive B/C rotation corresponds to negative local state rotation.
        if rank == 1:
            complex_b = b[..., ::2].astype(np.float64) + 1j * b[..., 1::2]
            complex_c = c[..., ::2].astype(np.float64) + 1j * c[..., 1::2]
        else:
            complex_b = b[..., :2].astype(np.float64) + 1j * b[..., 2:]
            complex_c = c[..., :2].astype(np.float64) + 1j * c[..., 2:]
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
    assert isinstance(model, RecurrentSequenceModel)
    initial = model.initial_carry(num_envs=2)
    starts = jnp.zeros(x.shape[:2], jnp.bool_).at[0, 0].set(True).at[4, 1].set(True)
    variables = model.init(jax.random.key(1), x, initial, starts)
    warm, _ = model.apply(variables, x[:2], initial, starts[:2])
    for incoming in (initial, warm):
        final, expected = jax.jit(model.apply)(variables, x, incoming, starts)
        carry, first = model.apply(variables, x[:3], incoming, starts[:3])
        carry, second = model.apply(variables, x[3:], carry, starts[3:])
        np.testing.assert_allclose(jnp.concatenate((first, second)), expected, rtol=2e-5, atol=2e-6)
        assert_carry_close(carry, final)
        carry = incoming
        outputs = []
        for token, reset in zip(x, starts):
            carry, y = model.apply(variables, token, carry, reset, method=model.step)
            outputs.append(y)
        np.testing.assert_allclose(jnp.stack(outputs), expected, rtol=2e-5, atol=2e-6)
        assert_carry_close(carry, final)
    assert final.state.shape == (2, 4, 4, 8)
    # Empty chunks preserve history and the output contract under JIT.
    empty_carry, empty = jax.jit(model.apply)(variables, x[:0], final, starts[:0])
    assert empty.shape == (0, 2, 8)
    assert_carry_close(empty_carry, final)


@pytest.mark.parametrize("rank", [1, 2])
def test_resets_clear_all_history_without_affecting_other_examples(rank: int) -> None:
    model = Mamba3(8, d_state=8, headdim=4, mimo_rank=rank)
    x = jax.random.normal(jax.random.key(2), (6, 2, 8))
    variables = model.init(
        jax.random.key(3),
        x,
        carry=model.initial_carry(num_envs=x.shape[-2]),
        episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_),
    )
    starts = jnp.zeros((6, 2), dtype=jnp.bool_).at[3, 0].set(True)
    final, y = jax.jit(model.apply)(variables, x, model.initial_carry(num_envs=x.shape[-2]), starts)
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
    np.testing.assert_allclose(y[3:, :1], fresh_y, atol=2e-6)
    np.testing.assert_allclose(y[:, 1:], continuous_y, atol=2e-6)
    for actual, reset, kept in zip(final, fresh, continuous):
        np.testing.assert_allclose(actual[:1], reset, atol=2e-6)
        np.testing.assert_allclose(actual[1:], kept, atol=2e-6)
    # Reset an incoming nonzero carry, including previous key/value and phase.
    reset, reset_y = model.apply(variables, x[0], final, jnp.ones(2, dtype=jnp.bool_), method=model.step)
    zero, zero_y = model.apply(
        variables,
        x[0],
        method=model.step,
        carry=model.initial_carry(num_envs=x[0].shape[-2]),
        episode_starts=jnp.zeros(x[0].shape[:-1], jnp.bool_),
    )
    assert_carry_close(reset, zero)
    np.testing.assert_allclose(reset_y, zero_y, atol=2e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
def test_causal_gradients_and_training(dtype: jax.typing.DTypeLike) -> None:
    model = Mamba3(8, d_state=8, headdim=4, mimo_rank=2, outproj_norm=True, dtype=dtype)
    x = jax.random.normal(jax.random.key(4), (5, 2, 8))
    params = model.init(
        jax.random.key(5),
        x,
        carry=model.initial_carry(num_envs=x.shape[-2]),
        episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_),
    )["params"]

    def loss_fn(parameters: Any, inputs: jax.Array) -> jax.Array:
        chex.assert_shape(inputs, (5, 2, 8))
        chex.assert_type(inputs, jnp.float32)
        _, y = model.apply(
            {"params": parameters},
            inputs,
            carry=model.initial_carry(num_envs=inputs.shape[-2]),
            episode_starts=jnp.zeros(inputs.shape[:-1], jnp.bool_),
        )
        return jnp.mean(jnp.square(y.astype(jnp.float32) - 1))

    loss, (grads, input_grads) = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))(params, x)
    assert np.isfinite(loss)
    for grad in jax.tree.leaves((grads, input_grads)):
        assert np.isfinite(grad).all()
        assert np.any(np.asarray(grad) != 0)

    def gradient_step(grad: jax.Array) -> jax.Array:
        chex.assert_type(grad, jnp.float32)
        return -0.01 * grad

    updated = optax.apply_updates(params, jax.tree.map(gradient_step, grads))
    assert float(loss_fn(updated, x)) < float(loss)
    carry, y = model.apply(
        {"params": params},
        x,
        carry=model.initial_carry(num_envs=x.shape[-2]),
        episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_),
    )
    assert y.dtype == dtype
    assert all(leaf.dtype == jnp.float32 for leaf in carry)

    def prefix_loss(inputs: jax.Array) -> jax.Array:
        chex.assert_shape(inputs, (5, 2, 8))
        chex.assert_type(inputs, jnp.float32)
        _, outputs = model.apply(
            {"params": params},
            inputs,
            carry=model.initial_carry(num_envs=inputs.shape[-2]),
            episode_starts=jnp.zeros(inputs.shape[:-1], jnp.bool_),
        )
        return outputs[:2].astype(jnp.float32).sum()

    grad = jax.grad(prefix_loss)(x)
    np.testing.assert_array_equal(grad[2:], 0)
    _, changed = model.apply(
        {"params": params},
        x.at[2:].set(100),
        carry=model.initial_carry(num_envs=(x.at[2:].set(100)).shape[-2]),
        episode_starts=jnp.zeros((x.at[2:].set(100)).shape[:-1], jnp.bool_),
    )
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
    with pytest.raises((ValueError, AssertionError)):
        Mamba3(**settings).initial_carry(2)


def test_invalid_input_and_carry_shapes() -> None:
    model = Mamba3(8, d_state=8, headdim=4)
    x = jnp.zeros((3, 2, 8))
    variables = model.init(
        jax.random.key(0),
        x,
        carry=model.initial_carry(num_envs=x.shape[-2]),
        episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_),
    )
    with pytest.raises(AssertionError):
        model.apply(
            variables,
            x[0],
            carry=model.initial_carry(num_envs=x[0].shape[-2]),
            episode_starts=jnp.zeros(x[0].shape[:-1], jnp.bool_),
        )
    with pytest.raises(AssertionError):
        model.apply(variables, x, model.initial_carry(1), episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_))
    carry = model.initial_carry(num_envs=2)
    starts = jnp.zeros(x.shape[:2], jnp.bool_)
    for name, leaf in zip(carry._fields, carry):
        with pytest.raises(AssertionError):
            model.apply(variables, x, carry._replace(**{name: leaf[..., :-1]}), starts)
    with pytest.raises(AssertionError):
        model.apply(variables, x, carry._replace(state=carry.state.astype(jnp.bfloat16)), starts)
    with pytest.raises(AssertionError):
        model.apply(variables, x, episode_starts=jnp.zeros((2, 3)), carry=model.initial_carry(num_envs=x.shape[-2]))
    with pytest.raises(AssertionError):
        model.apply(
            variables,
            x[0],
            episode_starts=jnp.zeros((1, 2)),
            method=model.step,
            carry=model.initial_carry(num_envs=x[0].shape[-2]),
        )


# SISO and MIMO have different rotation layouts; larger MIMO ranks share a path.
# Cover both RoPE fractions, normalization modes, and full/chunked gradients.
@pytest.mark.parametrize(
    "rank,fraction,norm,chunk_size",
    [
        (1, 0.5, False, 6),
        (1, 1.0, True, 2),
        (2, 0.5, True, 6),
        (2, 1.0, False, 2),
    ],
)
def test_matches_official_module_outputs_states_and_gradients(
    rank: int, fraction: float, norm: bool, chunk_size: int
) -> None:
    """Fixtures execute upstream Mamba3 with its CPU reference kernels.

    See generate_mamba3_reference.py for pinned provenance and reproduction.
    PyTorch and the upstream repository are not needed for normal pytest runs.
    Chunked evaluation must preserve both history and its gradient graph.
    """
    model = Mamba3(
        8,
        d_state=8,
        headdim=4,
        mimo_rank=rank,
        rope_fraction=fraction,
        ngroups=2 if norm else 1,
        outproj_norm=norm,
    )
    prefix = f"r{rank}_f{int(fraction * 100)}_n{int(norm)}/"
    with np.load(Path(__file__).parent / "data" / "mamba3_reference.npz", allow_pickle=False) as fixture:
        assert str(fixture["upstream_revision"]) == "e9594ce1c732d97440f0332fdc43170a2294dbfa"
        case = {key.removeprefix(prefix): jnp.asarray(fixture[key]) for key in fixture.files if key.startswith(prefix)}
    params = unflatten_dict(
        {key.removeprefix("params/"): value for key, value in case.items() if key.startswith("params/")}, sep="/"
    )

    def forward(parameters: Any, inputs: jax.Array) -> tuple[Mamba3Carry, jax.Array]:
        chex.assert_shape(inputs, (6, 2, 8))
        chex.assert_type(inputs, jnp.float32)
        carry = model.initial_carry(inputs.shape[1])
        outputs = []
        for start in range(0, inputs.shape[0], chunk_size):
            carry, output = model.apply(
                {"params": parameters},
                inputs[start : start + chunk_size],
                carry,
                episode_starts=jnp.zeros(inputs[start : start + chunk_size].shape[:-1], jnp.bool_),
            )
            outputs.append(output)
        return carry, jnp.concatenate(outputs)

    def loss_fn(parameters: Any, x: jax.Array) -> tuple[jax.Array, tuple[Mamba3Carry, jax.Array]]:
        chex.assert_shape(x, (6, 2, 8))
        chex.assert_type(x, jnp.float32)
        chex.assert_trees_all_equal_shapes_and_dtypes(parameters, params)
        carry, output = forward(parameters, x)
        return jnp.sum(output * case["probe"]), (carry, output)

    (_, (carry, y)), (grads, dx) = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))(params, case["x"])
    np.testing.assert_allclose(y, case["y"], rtol=5e-5, atol=3e-6)
    for name, leaf in zip(carry._fields, carry):
        np.testing.assert_allclose(leaf, case[f"carry/{name}"], rtol=5e-5, atol=3e-6)
    np.testing.assert_allclose(dx, case["dx"], rtol=1e-4, atol=1e-5)
    for name, gradient in flatten_dict(grads, sep="/").items():
        np.testing.assert_allclose(gradient, case[f"grads/{name}"], rtol=1e-4, atol=1e-5, err_msg=name)


def test_official_initialization_and_bounded_phase() -> None:
    model = Mamba3(32, d_state=8, headdim=8, mimo_rank=2)
    x = jnp.ones((1, 1, 32), jnp.float32)
    params = model.init(
        jax.random.key(7),
        x,
        carry=model.initial_carry(num_envs=x.shape[-2]),
        episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_),
    )["params"]
    for name in ("in_proj", "out_proj"):
        kernel = np.asarray(params[name]["kernel"])
        bound = kernel.shape[0] ** -0.5
        assert np.max(np.abs(kernel)) <= bound
        assert np.var(kernel) == pytest.approx(1 / (3 * kernel.shape[0]), rel=0.1)
    dt = jax.nn.softplus(params["dt_bias"])
    assert np.all((dt >= model.dt_min) & (dt <= model.dt_max))
    for name, value in (("B_bias", 1), ("C_bias", 1), ("D", 1), ("mimo_x", 0.5), ("mimo_z", 1), ("mimo_o", 0.5)):
        np.testing.assert_array_equal(params[name], value)
    carry = model.initial_carry(1)
    carry = carry._replace(angle=jnp.full_like(carry.angle, 100 * jnp.pi))
    carry, _ = model.apply({"params": params}, x, carry, episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_))
    assert np.all((carry.angle >= 0) & (carry.angle < 2 * jnp.pi))


def test_rejects_nonfloating_inputs_and_nonboolean_resets() -> None:
    model = Mamba3(8, d_state=8, headdim=4)
    x = jnp.ones((3, 2, 8), jnp.float32)
    params = model.init(
        jax.random.key(0),
        x,
        carry=model.initial_carry(num_envs=x.shape[-2]),
        episode_starts=jnp.zeros(x.shape[:-1], jnp.bool_),
    )
    with pytest.raises(AssertionError):
        model.apply(
            params,
            x.astype(jnp.int32),
            carry=model.initial_carry(num_envs=(x.astype(jnp.int32)).shape[-2]),
            episode_starts=jnp.zeros((x.astype(jnp.int32)).shape[:-1], jnp.bool_),
        )
    with pytest.raises(AssertionError):
        model.apply(
            params, x, episode_starts=jnp.ones((3, 2), jnp.float32), carry=model.initial_carry(num_envs=x.shape[-2])
        )
