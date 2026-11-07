from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from pydantic import ValidationError

from rl2 import train_grpo_v2 as training
from rl2.grpo_v2 import cispo
from rl2.grpo_v2.kl import k3_diagnostic
from rl2.qwen3 import Qwen3Output, Variables


def test_k3_diagnostic_bounds_extremes_and_preserves_ordinary_values() -> None:
    log_ratio = jnp.array([[0.0, 1.0, -1.0], [100.0, 1e35, -1e35]])  # (B, T)
    diagnostic = jax.jit(k3_diagnostic)(log_ratio)  # (B, T)
    assert diagnostic.shape == log_ratio.shape
    np.testing.assert_allclose(diagnostic, [[0, np.expm1(1) - 1, np.expm1(-1) + 1], [1e30, 1e30, 1e30]], rtol=1e-6)


@pytest.mark.parametrize("lower", [0.0, 0.5], ids=["default-no-floor", "positive-floor"])
def test_detached_clipping_gradients_and_rollout_reduction(lower: float) -> None:
    scores = jnp.full((3, 3), -2.0)  # (B, T)
    ratios = jnp.array([[0.25, 1, 8], [0.5, 8, 4], [1, 1, 1]])  # (B, T) caps with both advantage signs.
    old = scores - jnp.log(ratios)  # (B, T)
    mask = jnp.array([[True, True, True], [True, True, False], [False, False, False]])  # (B, T)
    advantages = jnp.array([1.0, -2.0, 3.0])  # (B,)

    def loss(current: jax.Array, sampler: jax.Array) -> tuple[jax.Array, training.Metrics]:  # (B, T) -> scalars
        return cispo.objective(current, sampler, advantages, mask, cispo.CISPOConfig(clip_low_threshold=lower))

    (value, metrics), (gradient, sampler_gradient) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(
        scores, old
    )
    floor = max(0.25, lower)
    expected = jnp.array([[-floor / 9, -1 / 9, -4 / 9], [1 / 6, 8 / 6, 0], [0, 0, 0]])  # (B, T)
    assert value.shape == () and all(item.shape == () for item in metrics.values())
    assert gradient.shape == sampler_gradient.shape == (3, 3)
    np.testing.assert_allclose(gradient, expected, atol=1e-7)
    np.testing.assert_array_equal(sampler_gradient, 0)
    assert float(value) == pytest.approx(2 * ((floor + 5) / 3 - 4.5) / 3)
    assert float(metrics["train/rollout_importance_weight_mean"]) == pytest.approx(((floor + 5) / 3 + 2.25) / 3)
    assert float(metrics["train/rollout_importance_weight_clipped_fraction"]) == pytest.approx(
        (1 + int(lower > 0.25)) / 9 + 1 / 6
    )


@pytest.mark.parametrize("upper", [4.0, 1e10], ids=["default-cap", "cap-above-exp20"])
def test_extreme_ratios_preserve_configured_clipping(upper: float) -> None:
    scores = jnp.array([[-1.0, -40.0, -jnp.inf]])  # (B, T)
    old = jnp.array([[-100.0, -1.0, -jnp.inf]])  # (B, T)

    def loss(current: jax.Array) -> tuple[jax.Array, training.Metrics]:  # (B, T) -> (), scalar metrics.
        return cispo.objective(
            current, old, jnp.ones(1), jnp.array([[True, True, False]]), cispo.CISPOConfig(clip_high_threshold=upper)
        )

    (_, metrics), gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))(scores)  # Scalars; (B, T).
    np.testing.assert_allclose(gradient, [[-upper / 2, -np.exp(-39) / 2, 0]], rtol=1e-6, atol=0)
    assert float(metrics["train/sampler_kl"]) == pytest.approx(5e29)


@pytest.mark.parametrize(
    "settings",
    [
        {"clip_low_threshold": -0.1},
        {"clip_high_threshold": 0.0},
        {"clip_low_threshold": 5.0},
        {"clip_high_threshold": float("inf")},
    ],
    ids=["negative-lower", "nonpositive-upper", "reversed-bounds", "infinite-upper"],
)
def test_invalid_thresholds(settings: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        cispo.CISPOConfig(**settings)


def test_default_and_equal_thresholds() -> None:
    assert training.Config().loss == cispo.CISPOConfig(clip_low_threshold=0.0, clip_high_threshold=4.0)
    config = cispo.CISPOConfig(clip_low_threshold=1, clip_high_threshold=1)

    def loss(scores: jax.Array) -> tuple[jax.Array, training.Metrics]:  # (B, T) -> scalars
        return cispo.objective(scores, jnp.full((1, 3), -2.0), jnp.ones(1), jnp.ones((1, 3), bool), config)

    (_, metrics), gradient = jax.value_and_grad(loss, has_aux=True)(jnp.array([[-3.0, -2.0, -1.0]]))
    np.testing.assert_allclose(gradient, [[-1 / 3, -1 / 3, -1 / 3]], atol=1e-7)
    assert float(metrics["train/rollout_importance_weight_clipped_fraction"]) == pytest.approx(2 / 3)


def apply_logits(
    variables: Variables,
    tokens: jax.Array,  # (B, T)
    *,
    attention_mask: jax.Array,  # (B, T)
) -> Qwen3Output:
    return Qwen3Output(logits=variables["params"]["logits"], cache=None)  # (B, T, V)


@pytest.mark.usefixtures("ppo_mesh")
def test_default_learner_updates_without_anchor_or_topk_and_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    learner = training.Learner.__ray_metadata__.modified_class(
        actor=Mock(), batcher=Mock(), config=training.Config(temperature=1.0)
    )
    learner.mesh = jax.sharding.get_mesh()
    learner.state = training.TrainState.create(
        apply_fn=apply_logits, params={"logits": jnp.zeros((1, 2, 4))}, tx=optax.sgd(0.1)
    )  # Parameter leaf: (B, T-1, V).
    batch = training.GRPOBatch(
        token_ids=np.array([[0, 2, 1]], dtype=np.int32),  # (B, T)
        attention_mask=np.ones((1, 3), dtype=np.bool_),  # (B, T)
        loss_mask=np.array([[False, True, False]]),  # (B, T)
        old_log_probs=np.log(np.full((1, 3), 0.025, dtype=np.float32)),  # (B, T) ratio 10 clips to 4.
        advantages=np.ones(1, dtype=np.float32),  # (B,)
    )
    monkeypatch.setattr(training, "compute_anchor_log_probs", Mock(side_effect=AssertionError("anchor pass")))
    result = learner.optimize(batch, jax.tree.map(jnp.zeros_like, learner.state.params))
    assert result.has_signal
    assert set(result.metrics) == {"train/policy_loss", *cispo.DIAGNOSTICS}
    np.testing.assert_allclose(result.gradient_sum["logits"], [[[1, 1, -3, 1], [0, 0, 0, 0]]], atol=1e-6)
    learner.apply_update(result.gradient_sum, has_signal=result.has_signal)
    assert int(learner.state.step) == 1
    np.testing.assert_allclose(learner.state.params["logits"][0, 0], [-0.1, -0.1, 0.3, -0.1], atol=1e-7)

    for inactive in (
        batch._replace(advantages=np.zeros(1, dtype=np.float32)),
        batch._replace(loss_mask=np.zeros((1, 3), dtype=np.bool_)),
    ):
        before = jax.tree.map(jnp.copy, learner.state)  # Parameter/optimizer leaves, copied before donation.
        result = learner.optimize(inactive, jax.tree.map(jnp.zeros_like, learner.state.params))
        assert not result.has_signal and result.metrics == {"train/policy_loss": 0.0}
        learner.apply_update(result.gradient_sum, has_signal=result.has_signal)
        for actual, expected in zip(jax.tree.leaves(learner.state), jax.tree.leaves(before), strict=True):
            np.testing.assert_array_equal(actual, expected)
