from functools import partial
from math import prod

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import beta as beta_distribution
from scipy.stats import truncnorm

from rl2.actions import (
    BetaPolicy,
    CategoricalPolicy,
    ContinuousSpace,
    DiscreteSpace,
    HierarchicalMultiCategoricalPolicy,
    HierarchicalMultiDiscreteSpace,
    MultiCategoricalPolicy,
    TruncatedGaussianPolicy,
    hierarchical_log_probs,
)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)], ids=["scalar", "batch", "sequence"])
def test_categorical_shapes_likelihood_and_masked_entropy(batch_shape: tuple[int, ...]) -> None:
    logits = jnp.broadcast_to(jnp.array([0.0, jnp.log(3.0), -jnp.inf]), (*batch_shape, 3))  # (*S, A=3)
    policy = CategoricalPolicy(logits)
    key = jax.random.key(0)  # ()
    actions = jax.jit(policy.sample)(key=key)  # (*S,)
    sampled, log_probs = jax.jit(policy.sample_and_log_prob)(key=key)  # each (*S,)
    chex.assert_shape((actions, sampled, log_probs), batch_shape)
    assert actions.dtype == jnp.int32
    assert bool(jnp.all(actions < 2))
    np.testing.assert_array_equal(actions, sampled)
    np.testing.assert_allclose(log_probs, policy.log_prob(actions), atol=1e-7)
    np.testing.assert_allclose(log_probs, jnp.log(jnp.where(actions == 0, 0.25, 0.75)), atol=1e-7)
    entropy = jax.jit(policy.entropy)()  # (*S,)
    assert entropy.shape == batch_shape
    np.testing.assert_allclose(entropy, -0.25 * np.log(0.25) - 0.75 * np.log(0.75), rtol=1e-6)

    def entropy_sum(parameters: jax.Array) -> jax.Array:  # (*S, A) -> ()
        return CategoricalPolicy(parameters).entropy().sum()  # ()

    assert bool(jnp.all(jnp.isfinite(jax.grad(entropy_sum)(logits))))


@pytest.mark.parametrize(
    "batch_shape,action_shape",
    [((), ()), ((2,), (1,)), ((2, 3), (2, 2))],
    ids=["scalar-box", "one-dimensional-batch", "matrix-actions-sequence"],
)
def test_gaussian_shapes_bounds_and_joint_density(batch_shape: tuple[int, ...], action_shape: tuple[int, ...]) -> None:
    size = prod(action_shape)
    action_space = ContinuousSpace(action_shape, (-0.3,) * size, (0.7,) * size)
    mean = jnp.full((*batch_shape, size), 0.2)  # (*S, D)
    log_std = jnp.full_like(mean, -1.0)  # (*S, D)
    policy = TruncatedGaussianPolicy(mean, log_std, action_space)
    key = jax.random.key(0)  # ()
    actions = jax.jit(policy.sample)(key=key)  # (*S, *action_shape)
    sampled, log_probs = jax.jit(policy.sample_and_log_prob)(key=key)  # (*S, *action_shape); (*S,)
    assert actions.shape == sampled.shape == (*batch_shape, *action_shape)
    assert log_probs.shape == batch_shape
    assert actions.dtype == log_probs.dtype == jnp.float32
    assert bool(jnp.all((actions >= -0.3) & (actions <= 0.7)))
    np.testing.assert_array_equal(actions, sampled)
    assert not np.array_equal(actions, policy.sample(key=jax.random.key(1)))
    np.testing.assert_allclose(log_probs, jax.jit(policy.log_prob)(actions), atol=2e-6)
    reference = truncnorm((-0.3 - 0.2) / np.exp(-1.0), (0.7 - 0.2) / np.exp(-1.0), loc=0.2, scale=np.exp(-1.0))
    flat_actions = np.asarray(actions).reshape((*batch_shape, size))  # (*S, D)
    expected = reference.logpdf(flat_actions).sum(axis=-1)  # (*S,)
    np.testing.assert_allclose(log_probs, expected, atol=2e-6)
    entropy = jax.jit(policy.entropy)()  # (*S,)
    assert entropy.shape == batch_shape
    np.testing.assert_allclose(entropy, size * reference.entropy(), atol=2e-6)
    assert bool(jnp.all(jnp.isneginf(policy.log_prob(jnp.full_like(actions, 0.8)))))


@pytest.mark.parametrize(
    "shape,low,high,match",
    [
        ((0,), (), (), "positive"),
        ((2,), (-1.0,), (1.0,), "shape"),
        ((1,), (0.0,), (0.0,), "low < high"),
        ((1,), (1.0,), (-1.0,), "low < high"),
        ((1,), (-float("inf"),), (1.0,), "finite"),
        ((1,), (-1.0,), (float("nan"),), "finite"),
    ],
    ids=["empty-axis", "bound-shape", "zero-width", "reversed", "unbounded", "nan"],
)
def test_invalid_continuous_spaces(
    shape: tuple[int, ...], low: tuple[float, ...], high: tuple[float, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ContinuousSpace(shape, low, high)


def test_discrete_space_requires_at_least_one_action() -> None:
    with pytest.raises(ValueError, match="positive"):
        DiscreteSpace(0)
    policy = CategoricalPolicy(jnp.zeros((1,)))  # logits: (A=1,)
    actions, log_probs = policy.sample_and_log_prob(key=jax.random.key(0))  # each ()
    assert int(actions) == 0
    assert float(log_probs) == float(policy.entropy()) == 0


def test_policy_action_dtype_contracts() -> None:
    categorical = CategoricalPolicy(jnp.zeros(2))  # logits: (A=2,)
    with pytest.raises(AssertionError):
        categorical.log_prob(jnp.array(0.0))
    gaussian = TruncatedGaussianPolicy(jnp.zeros(1), jnp.zeros(1), ContinuousSpace((1,), (-1.0,), (1.0,)))
    # Gaussian parameters: (D=1,)
    with pytest.raises(AssertionError):
        gaussian.log_prob(jnp.zeros(1, jnp.int32))


def test_leading_sample_dimensions_and_broadcast_log_prob() -> None:
    key = jax.random.key(0)  # ()
    categorical = CategoricalPolicy(jnp.zeros((2, 3)))  # logits: (B=2, A=3)
    actions, log_probs = jax.jit(categorical.sample_and_log_prob, static_argnames="sample_shape")(
        key=key, sample_shape=2
    )  # each (N=2, B=2)
    assert actions.shape == log_probs.shape == (2, 2)
    np.testing.assert_array_equal(actions, categorical.sample(key=key, sample_shape=2))
    np.testing.assert_allclose(categorical.log_prob(actions), log_probs)
    gaussian = TruncatedGaussianPolicy(
        jnp.zeros((2, 2)), jnp.full((2, 2), -1.0), ContinuousSpace((2, 1), (-2.0, 0.0), (4.0, 1.0))
    )  # mean/log_std: (B=2, D=2)
    actions, log_probs = jax.jit(gaussian.sample_and_log_prob, static_argnames="sample_shape")(
        key=key, sample_shape=(2, 1)
    )  # (N=2, M=1, B=2, 2, 1); (N, M, B)
    assert actions.shape == (2, 1, 2, 2, 1)
    assert log_probs.shape == (2, 1, 2)
    np.testing.assert_allclose(actions, gaussian.sample(key=key, sample_shape=(2, 1)), atol=1e-6)
    np.testing.assert_allclose(gaussian.log_prob(actions), log_probs, atol=2e-6)


@pytest.mark.parametrize(
    "mean,log_std",
    [(0.0, -16.0), (1.0, -16.0), (1.0, 0.0), (0.0, 2.0)],
    ids=["tiny-scale", "tiny-scale-at-bound", "center-at-bound", "wide-gaussian"],
)
def test_truncated_sampling_entropy_and_gradients_at_boundaries(mean: float, log_std: float) -> None:
    policy = TruncatedGaussianPolicy(
        jnp.array([mean]), jnp.array([log_std]), ContinuousSpace((1,), (-1.0,), (1.0,))
    )  # mean/log_std: (D=1,)
    actions, log_prob = jax.jit(policy.sample_and_log_prob)(key=jax.random.key(0))  # (D,); ()
    assert bool(jnp.all((actions >= -1) & (actions <= 1)))
    np.testing.assert_allclose(log_prob, policy.log_prob(actions), atol=1e-6)
    assert bool(jnp.isfinite(policy.entropy()))

    def objective(mean: jax.Array, log_std: jax.Array) -> jax.Array:  # each (D,) -> ()
        current = policy.replace(mean=mean, log_std=log_std)
        return current.log_prob(actions) + current.entropy()  # ()

    value, gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(policy.mean, policy.log_std)
    assert bool(jnp.isfinite(value))
    assert all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in gradients)


def test_truncated_entropy_gradient_moves_mean_toward_center() -> None:
    action_space = ContinuousSpace((1,), (-1.0,), (1.0,))
    mean = jnp.array([[-0.8], [0.0], [0.8]])  # (B=3, D=1)
    log_std = jnp.full_like(mean, -0.7)  # (B, D)

    def entropy(mean: jax.Array, log_std: jax.Array) -> jax.Array:  # each (B, D) -> ()
        return TruncatedGaussianPolicy(mean, log_std, action_space).entropy().sum()  # ()

    mean_grad, std_grad = jax.jit(jax.grad(entropy, argnums=(0, 1)))(mean, log_std)  # each (B, D)
    assert float(mean_grad[0, 0]) > 0
    np.testing.assert_allclose(mean_grad[1, 0], 0, atol=1e-6)
    assert float(mean_grad[2, 0]) < 0
    assert bool(jnp.all(std_grad > 0))


@pytest.mark.parametrize(
    "batch_shape,action_shape,sample_shape",
    [((), (), ()), ((2,), (1,), 2), ((2, 3), (2, 1), (2, 1))],
    ids=["scalar-box", "vector-batch-integer-samples", "matrix-actions-sequence-tuple-samples"],
)
def test_beta_shapes_bounds_density_and_entropy(
    batch_shape: tuple[int, ...], action_shape: tuple[int, ...], sample_shape: int | tuple[int, ...]
) -> None:
    size = prod(action_shape)
    space = ContinuousSpace(action_shape, (-2.0, 0.0)[:size], (4.0, 1.0)[:size])
    alpha, beta = jax.tree.map(partial(jnp.full, (*batch_shape, size)), (2.0, 3.0))  # each (*S, D)
    policy = BetaPolicy(alpha, beta, space)
    key = jax.random.key(0)  # ()
    actions, log_probs = jax.jit(policy.sample_and_log_prob, static_argnames="sample_shape")(
        key=key, sample_shape=sample_shape
    )  # (*sample_shape, *S, *action_shape); (*sample_shape, *S)
    samples = (sample_shape,) if isinstance(sample_shape, int) else sample_shape
    assert actions.shape == (*samples, *batch_shape, *action_shape)
    assert log_probs.shape == (*samples, *batch_shape)
    assert actions.dtype == log_probs.dtype == jnp.float32
    np.testing.assert_allclose(actions, policy.sample(key=key, sample_shape=sample_shape), atol=1e-6)
    np.testing.assert_allclose(log_probs, jax.jit(policy.log_prob)(actions), atol=2e-6)
    flat_actions = actions.reshape((*samples, *batch_shape, size))  # (*sample_shape, *S, D)
    low, high = jnp.asarray((space.low, space.high))  # each (D,)
    assert bool(jnp.all((flat_actions > low) & (flat_actions < high)))
    reference = beta_distribution(2.0, 3.0, loc=low, scale=high - low)
    np.testing.assert_allclose(log_probs, reference.logpdf(flat_actions).sum(axis=-1), atol=3e-6)
    entropy = jax.jit(policy.entropy)()  # (*S,)
    assert entropy.shape == batch_shape
    np.testing.assert_allclose(entropy, reference.entropy().sum(), atol=3e-6)
    for bound in (low - 1, high + 1):  # each (D,)
        outside = jnp.broadcast_to(bound, (*batch_shape, size)).reshape(
            (*batch_shape, *action_shape)
        )  # (*S, *action_shape)
        assert bool(jnp.all(jnp.isneginf(policy.log_prob(outside))))
    with pytest.raises(AssertionError):
        policy.log_prob(actions.astype(jnp.int32))


def test_beta_log_prob_respects_box_bounds_before_affine_rounding() -> None:
    policy = BetaPolicy(
        alpha=jnp.ones(1),  # (D=1,)
        beta=jnp.ones(1),  # (D=1,)
        action_space=ContinuousSpace(shape=(1,), low=(-1.0,), high=(1.0,)),
    )
    bounds = jnp.array([[-1.0], [1.0]])  # (N=2, D=1)
    outside = jnp.nextafter(bounds, bounds * 2)  # (N, D)
    log_probs = jax.jit(policy.log_prob)(outside)  # (N,)
    np.testing.assert_array_equal(log_probs, jnp.full(2, -jnp.inf))
    np.testing.assert_allclose(policy.log_prob(bounds), -jnp.log(2.0), atol=1e-6)


@pytest.mark.parametrize(
    "alpha,beta", [(1.0, 1.0), (0.01, 0.01), (1.0, 100.0)], ids=["uniform", "endpoint-singularities", "skewed"]
)
def test_beta_boundary_samples_and_gradients(alpha: float, beta: float) -> None:
    space = ContinuousSpace((2,), (-1.0, 5.0), (1.0, 5.001))
    policy = BetaPolicy(jnp.full(2, alpha), jnp.full(2, beta), space)  # alpha/beta: (D=2,)
    actions, log_probs = jax.jit(policy.sample_and_log_prob, static_argnames="sample_shape")(
        key=jax.random.key(0), sample_shape=8
    )  # (N=8, D=2); (N,)
    low, high = jnp.asarray((space.low, space.high))  # each (D,)
    assert bool(jnp.all((actions > low) & (actions < high)))
    assert bool(jnp.all(jnp.isfinite(log_probs)))
    np.testing.assert_allclose(log_probs, jax.jit(policy.log_prob)(actions), atol=1e-5)

    def objective(alpha: jax.Array, beta: jax.Array) -> jax.Array:  # each (D,) -> ()
        current = policy.replace(alpha=alpha, beta=beta)
        return current.log_prob(actions).mean() + current.entropy()  # ()

    value, gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(policy.alpha, policy.beta)
    assert bool(jnp.isfinite(value))
    assert all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in gradients)


@pytest.mark.parametrize("batch_shape", [(), (2, 3)], ids=["unbatched", "sequence"])
def test_multi_categorical_joint_probability_and_entropy(batch_shape: tuple[int, ...]) -> None:
    logits = jnp.broadcast_to(
        jnp.array([[0.0, jnp.log(3.0), -jnp.inf], [0.0, -jnp.inf, -jnp.inf]]), (*batch_shape, 2, 3)
    )  # (*S, D=2, A=3)
    policy = MultiCategoricalPolicy(logits=logits)
    actions, log_prob = jax.jit(policy.sample_and_log_prob)(key=jax.random.key(0))  # (*S, D); (*S,)
    assert actions.shape == (*batch_shape, 2) and log_prob.shape == batch_shape
    np.testing.assert_array_equal(actions[..., 1], 0)
    np.testing.assert_allclose(log_prob, jnp.log(jnp.where(actions[..., 0] == 0, 0.25, 0.75)), atol=1e-7)
    np.testing.assert_allclose(policy.log_prob(actions), log_prob, atol=1e-7)
    np.testing.assert_allclose(policy.entropy(), -0.25 * np.log(0.25) - 0.75 * np.log(0.75), rtol=1e-6)
    samples, probs = policy.sample_and_log_prob(key=jax.random.key(1), sample_shape=(2, 1))
    # samples: (2, 1, *S, D); probs: (2, 1, *S)
    assert samples.shape == (2, 1, *batch_shape, 2) and probs.shape == (2, 1, *batch_shape)
    np.testing.assert_allclose(probs, policy.log_prob(samples), atol=1e-7)
    np.testing.assert_array_equal(samples, policy.sample(key=jax.random.key(1), sample_shape=(2, 1)))
    with pytest.raises(AssertionError):
        policy.log_prob(actions.astype(jnp.float32))


def test_multi_categorical_sums_components_and_has_gradients() -> None:
    logits = jnp.array([[0.0, jnp.log(3.0)], [jnp.log(2.0), 0.0]])  # (D=2, A=2)
    actions = jnp.array([1, 0], dtype=jnp.int32)  # (D,)
    policy = MultiCategoricalPolicy(logits=logits)
    np.testing.assert_allclose(policy.log_prob(actions), np.log(0.75 * 2 / 3), atol=1e-7)
    expected = -0.25 * np.log(0.25) - 0.75 * np.log(0.75) - 2 / 3 * np.log(2 / 3) - 1 / 3 * np.log(1 / 3)
    np.testing.assert_allclose(policy.entropy(), expected, rtol=1e-6)

    def objective(values: jax.Array) -> jax.Array:  # (D, A) -> ()
        return MultiCategoricalPolicy(logits=values).log_prob(actions)  # ()

    gradient = jax.grad(objective)(logits)  # (D, A)
    np.testing.assert_allclose(gradient, [[-0.25, 0.25], [1 / 3, -1 / 3]], atol=1e-7)


@pytest.mark.parametrize("counts", [(), (0,), (2, -1)])
def test_invalid_hierarchical_multi_discrete_space(counts: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="positive"):
        HierarchicalMultiDiscreteSpace(nvec=counts, group_sizes=(1,))


def test_hierarchical_multi_discrete_logit_shapes() -> None:
    assert HierarchicalMultiDiscreteSpace(nvec=(3, 3), group_sizes=(1, 1, 1)).logits_shape == (2, 3)
    assert HierarchicalMultiDiscreteSpace(nvec=(3,), group_sizes=(3,)).logits_shape == (1, 4)


@pytest.mark.parametrize("batch_shape", [(), (2, 3)], ids=["unbatched", "batched-sequences"])
def test_hierarchical_uniform_types_and_joint_entropy(batch_shape: tuple[int, ...]) -> None:
    space = HierarchicalMultiDiscreteSpace(nvec=(230,), group_sizes=(1, 4, 225))
    assert space.logits_shape == (1, 232)
    logits = jnp.zeros((*batch_shape, 232))  # (*S, L)
    mask = jnp.ones((*batch_shape, 230), dtype=bool)  # (*S, A)
    flat = jax.jit(hierarchical_log_probs, static_argnames="group_sizes")(logits, mask, space.group_sizes)
    # flat: (*S, A), valid command log probabilities
    probs = jnp.exp(flat)  # (*S, A)
    assert probs.shape == (*batch_shape, 230)
    np.testing.assert_allclose(probs[..., 0], 1 / 3, rtol=1e-6)
    np.testing.assert_allclose(probs[..., 1:5].sum(-1), 1 / 3, rtol=1e-6)
    np.testing.assert_allclose(probs[..., 5:].sum(-1), 1 / 3, rtol=1e-6)
    policy = HierarchicalMultiCategoricalPolicy(
        raw_logits=jnp.broadcast_to(logits[..., None, :], (*batch_shape, 2, 232)),  # (*S, D=2, L)
        actions_mask=jnp.broadcast_to(mask[..., None, :], (*batch_shape, 2, 230)),  # (*S, D=2, A)
        group_sizes=space.group_sizes,
    )
    policy = jax.jit(lambda distribution: distribution)(policy)
    assert policy.logits.shape == (*batch_shape, 2, 230)
    np.testing.assert_allclose(policy.logits[..., 0, :], flat, atol=1e-6)
    np.testing.assert_allclose(policy.entropy(), 2 * (np.log(3) + (np.log(4) + np.log(225)) / 3), rtol=1e-6)
    key = jax.random.key(0)  # ()
    action, lp = jax.jit(policy.sample_and_log_prob, static_argnames="sample_shape")(key=key, sample_shape=2)
    assert action.shape == (2, *batch_shape, 2) and lp.shape == (2, *batch_shape)
    np.testing.assert_array_equal(action, policy.sample(key=key, sample_shape=2))
    np.testing.assert_allclose(lp, policy.log_prob(action), atol=1e-6)


def test_hierarchical_masks_and_gradients() -> None:
    groups = (1, 2, 3)
    logits = jnp.zeros((4, 8))  # (B, G + K)
    mask = jnp.array(
        [
            [True, True, False, True, False, True],  # All types; different numbers of valid children.
            [True, True, True, False, False, False],  # Sap unavailable.
            [True, False, False, False, False, False],  # Inactive ship: stay only.
            [False, False, False, True, False, True],
        ]  # Only a non-singleton group available.
    )  # (B, A)

    def objective(raw: jax.Array) -> jax.Array:  # (B, L) -> ()
        policy = HierarchicalMultiCategoricalPolicy(
            raw_logits=raw[:, None, :], actions_mask=mask[:, None, :], group_sizes=groups
        )  # logits: (B, D=1, A)
        return policy.entropy().sum()  # ()

    flat = hierarchical_log_probs(logits, mask, groups)  # (B, A)
    expected = [
        [1 / 3, 1 / 3, 0, 1 / 6, 0, 1 / 6],
        [1 / 2, 1 / 4, 1 / 4, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1 / 2, 0, 1 / 2],
    ]
    np.testing.assert_allclose(jnp.exp(flat), expected, rtol=1e-6)
    entropy, gradient = jax.jit(jax.value_and_grad(objective))(logits)  # (); (B, L)
    np.testing.assert_allclose(entropy, np.log(3) + np.log(2) / 3 + np.log(2) * 1.5 + np.log(2), rtol=1e-6)
    expected_gradient = np.zeros((4, 8))  # (B, L)
    expected_gradient[0, :3] = [-np.log(2) / 9, -np.log(2) / 9, 2 * np.log(2) / 9]
    expected_gradient[1, :3] = [-np.log(2) / 4, np.log(2) / 4, 0]
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-6)


def test_hierarchical_selected_branch_probability_and_gradient() -> None:
    logits = jnp.log(jnp.array([2.0, 3.0, 5.0, 1.0, 3.0, 2.0, 3.0, 5.0]))  # (L=8), groups=(1, 2, 3)
    mask = jnp.ones(6, dtype=bool)  # (A,)

    def log_probability(raw: jax.Array, command: jax.Array) -> jax.Array:  # (L,); () -> ()
        policy = HierarchicalMultiCategoricalPolicy(
            raw_logits=raw[None, :], actions_mask=mask[None, :], group_sizes=(1, 2, 3)
        )  # logits: (D=1, A)
        return policy.log_prob(command[None])  # ()

    stay, stay_grad = jax.value_and_grad(log_probability)(logits, jnp.array(0))  # (); (L,)
    move, move_grad = jax.value_and_grad(log_probability)(logits, jnp.array(2))  # (); (L,)
    sap, sap_grad = jax.value_and_grad(log_probability)(logits, jnp.array(5))  # (); (L,)
    np.testing.assert_allclose([stay, move, sap], np.log([0.2, 0.3 * 0.75, 0.5 * 0.5]), atol=1e-6)
    np.testing.assert_allclose(stay_grad[3:], 0, atol=1e-6)
    np.testing.assert_allclose(move_grad[5:], 0, atol=1e-6)
    np.testing.assert_allclose(sap_grad[3:5], 0, atol=1e-6)
    np.testing.assert_allclose(move_grad[:5], [-0.2, 0.7, -0.5, -0.25, 0.25], atol=1e-6)
    np.testing.assert_allclose(sap_grad, [-0.2, -0.3, 0.5, 0, 0, -0.2, -0.3, 0.5], atol=1e-6)


@pytest.mark.parametrize("counts,groups", [((6,), ()), ((6,), (1, 0, 5)), ((6,), (1, -1, 6)), ((6, 7), (1, 2, 3))])
def test_invalid_hierarchical_groups(counts: tuple[int, ...], groups: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="group sizes"):
        HierarchicalMultiDiscreteSpace(nvec=counts, group_sizes=groups)
