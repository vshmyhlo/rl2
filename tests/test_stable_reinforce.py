from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock, Mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from pydantic import ValidationError

from rl2 import train_grpo_v2 as training
from rl2.grpo_v2 import stable_reinforce as stable
from rl2.qwen3 import Qwen3Output, Variables


@pytest.mark.parametrize("head_size", [2, 4], ids=["approximated-tail", "full-vocabulary"])
def test_centered_gradient_matches_sampler_expectation(head_size: int) -> None:
    logits = jnp.log(jnp.array([0.4, 0.3, 0.2, 0.1]))  # (V,)
    sampler = jnp.array([0.5, 0.25, 1 / 6, 1 / 12])  # (V,) tail proportional to learner.

    def loss(values: jax.Array) -> jax.Array:  # (V,) -> ()
        log_probs = jax.nn.log_softmax(values)  # (V,)
        correction = stable.score_correction(log_probs[:head_size], jnp.log(sampler[:head_size]))  # ()
        return -(log_probs[2] - correction)  # ()

    gradient = jax.jit(jax.grad(loss))(logits)  # (V,)
    assert gradient.shape == (4,)
    np.testing.assert_allclose(gradient, sampler - jax.nn.one_hot(2, 4), atol=2e-7)


def test_centering_is_noop_on_policy_and_handles_saturated_head() -> None:
    logits = jnp.log(jnp.array([[0.4, 0.3, 0.2, 0.1], [1.0, 1e-30, 1e-30, 1e-30]]))  # (B, V)
    sampler = jax.nn.log_softmax(logits, axis=-1)[:, :2]  # (B, K)

    def correction(values: jax.Array) -> jax.Array:  # (B, V) -> ()
        return stable.score_correction(jax.nn.log_softmax(values, axis=-1)[:, :2], sampler).sum()  # ()

    gradient = jax.jit(jax.grad(correction))(logits)  # (B, V)
    np.testing.assert_allclose(gradient, 0, atol=2e-7)


@pytest.mark.parametrize("estimator", ["k1", "k3"])
def test_kl_only_gradient_masking_and_reduction(estimator: Literal["k1", "k3"]) -> None:
    selected = jnp.log(jnp.array([[0.4, 0.3], [0.4, 0.3], [0.4, 0.3]]))  # (B, T)
    old = jnp.log(jnp.array([[0.2, 0.6], [0.2, 0.6], [0.2, 0.6]]))  # (B, T)
    mask = jnp.array([[True, True], [True, False], [False, False]])  # (B, T)
    head = jnp.broadcast_to(jnp.log(jnp.array([0.4, 0.3])), (3, 2, 2))  # (B, T, K)
    config = stable.StableReinforceConfig(top_k=2, kl_coeff=0.1, kl_estimator=estimator)

    def loss(
        scores: jax.Array, sampling_scores: jax.Array
    ) -> tuple[jax.Array, dict[str, jax.Array]]:  # (B, T) -> scalars
        return stable.objective(scores, head, head, sampling_scores, jnp.zeros(3), mask, config)

    (value, metrics), (gradient, sampling_gradient) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(
        selected, old
    )
    expected = -jnp.ones_like(selected) if estimator == "k1" else jnp.exp(selected - old) - 1  # (B, T)
    expected = 0.1 * jnp.where(mask, expected, 0) / jnp.maximum(mask.sum(axis=1, keepdims=True), 1) / 3  # (B, T)
    assert value.shape == () and all(item.shape == () for item in metrics.values())
    assert gradient.shape == sampling_gradient.shape == (3, 2)
    assert float(metrics["train/policy_loss"]) == 0
    assert float(value) == pytest.approx(0.1 * float(metrics["train/sampler_kl"]))
    np.testing.assert_allclose(gradient, expected, atol=1e-7)
    np.testing.assert_array_equal(sampling_gradient, 0)


def test_policy_masking_and_detached_sampler() -> None:
    logits = jnp.log(jnp.array([[[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]]))  # (B, T, V)
    sampler = jnp.log(jnp.array([[[0.5, 0.25], [0.5, 0.25]]]))  # (B, T, K)

    def loss(values: jax.Array, sampling: jax.Array) -> jax.Array:  # (B, T, V), (B, T, K) -> ()
        log_probs = jax.nn.log_softmax(values, axis=-1)  # (B, T, V)
        return stable.objective(
            log_probs[..., 2],
            log_probs[..., :2],
            sampling,
            jnp.zeros((1, 2)),
            jnp.array([-2.0]),
            jnp.array([[True, False]]),
            stable.StableReinforceConfig(top_k=2, kl_coeff=0),
        )[0]

    gradient, sampling_gradient = jax.jit(jax.grad(loss, argnums=(0, 1)))(logits, sampler)
    np.testing.assert_allclose(gradient[0, 0], -2 * jnp.array([0.5, 0.25, 1 / 6 - 1, 1 / 12]), atol=3e-7)
    np.testing.assert_array_equal(gradient[0, 1], 0)
    np.testing.assert_array_equal(sampling_gradient, 0)


def apply_logits(
    variables: Variables,
    tokens: jax.Array,  # (B, T)
    *,
    attention_mask: jax.Array,  # (B, T)
) -> Qwen3Output:
    return Qwen3Output(logits=variables["params"]["logits"], cache=None)  # (B, T, V)


@pytest.mark.usefixtures("ppo_mesh")
def test_learner_stable_reinforce_updates_and_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    learner = training.Learner.__ray_metadata__.modified_class(
        actor=Mock(),
        batcher=Mock(),
        config=training.Config(stable_reinforce=stable.StableReinforceConfig(top_k=2), temperature=0.7),
    )
    learner.mesh = jax.sharding.get_mesh()
    learner.state = training.TrainState.create(
        apply_fn=apply_logits,
        params={"logits": jnp.zeros((1, 2, 4))},
        tx=optax.sgd(0.1),
    )  # Parameter leaf: (B, T-1, V).
    batch = training.GRPOBatch(
        token_ids=np.array([[0, 2, 1]], dtype=np.int32),  # (B, T)
        attention_mask=np.ones((1, 3), dtype=np.bool_),  # (B, T)
        loss_mask=np.array([[False, True, False]]),  # (B, T)
        old_log_probs=np.log(np.full((1, 3), 0.25, dtype=np.float32)),  # (B, T)
        advantages=np.zeros(1, dtype=np.float32),  # (B,)
        sampler_topk_ids=np.array([[[0, 1], [0, 1], [0, 1]]], dtype=np.int32),  # (B, T, K)
        sampler_topk_log_probs=np.log(np.full((1, 3, 2), 0.25, dtype=np.float32)),  # (B, T, K)
    )
    monkeypatch.setattr(
        training, "compute_anchor_log_probs", Mock(side_effect=AssertionError("unexpected anchor pass"))
    )
    result = learner.optimize(batch, jax.tree.map(jnp.zeros_like, learner.state.params))
    assert result.has_signal  # Zero advantages still train with k1 KL.
    assert set(result.metrics) == {"train/policy_loss", "train/loss", *stable.DIAGNOSTICS}
    assert result.metrics["train/policy_loss"] == 0
    learner.apply_update(result.gradient_sum, has_signal=result.has_signal)
    assert int(learner.state.step) == 1
    assert float(learner.state.params["logits"][0, 0, 2]) > 0
    np.testing.assert_array_equal(learner.state.params["logits"][0, 1], 0)

    learner.config = replace(learner.config, stable_reinforce=stable.StableReinforceConfig(top_k=2, kl_coeff=0))
    result = learner.optimize(batch, jax.tree.map(jnp.zeros_like, learner.state.params))
    assert not result.has_signal and result.metrics == {"train/policy_loss": 0, "train/loss": 0}
    learner.apply_update(result.gradient_sum, has_signal=result.has_signal)
    assert int(learner.state.step) == 1

    learner.config = replace(learner.config, stable_reinforce=stable.StableReinforceConfig(top_k=2))
    result = learner.optimize(
        batch._replace(loss_mask=np.zeros((1, 3), dtype=np.bool_)), jax.tree.map(jnp.zeros_like, learner.state.params)
    )
    assert not result.has_signal and result.metrics == {"train/policy_loss": 0, "train/loss": 0}


@pytest.mark.parametrize("options", [{"top_k": 0}, {"kl_coeff": -0.1}, {"kl_estimator": "unknown"}])
def test_invalid_stable_config(options: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        stable.StableReinforceConfig(**options)


def test_opt_in_config_and_server_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "training.yaml"
    path.write_text("stable_reinforce: {}\n")
    config = training.load_config(path, training.Config)
    assert config.stable_reinforce == stable.StableReinforceConfig()
    assert training.Config().stable_reinforce is None
    launch = Mock(return_value=MagicMock())
    monkeypatch.setattr(training.subprocess, "Popen", launch)
    with training.serve_vllm(sampler_top_k=config.stable_reinforce.top_k):
        command = launch.call_args.args[0]
        assert command[command.index("--max-logprobs") + 1] == "33"
