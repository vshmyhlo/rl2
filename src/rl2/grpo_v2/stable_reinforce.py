"""Local StableReinforce-style loss: top-k score centering and sampler KL.

Uses https://arxiv.org/html/2609.20807v1#S4.SS1, not Tinker's private implementation.
The unrecorded sampler tail is approximated by the learner's conditional tail.
"""

from typing import Annotated, Literal

import jax
import jax.numpy as jnp
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class StableReinforceConfig:
    top_k: Annotated[int, Field(ge=1)] = 32
    kl_coeff: Annotated[float, Field(ge=0)] = 0.01
    kl_estimator: Literal["k1", "k3"] = "k1"


DIAGNOSTICS = (
    "train/score_correction_mean",
    "train/sampler_kl",
    "train/sampler_topk_fraction_mean",
    "train/rollout_log_prob_error_mean",
)


def score_correction(
    head_log_probs: jax.Array,  # (..., K) learner log-probabilities on the sampler's head.
    sampler_head_log_probs: jax.Array,  # (..., K) recorded sampler log-probabilities.
) -> jax.Array:  # (...,)
    """(...,) surrogate whose gradient is the approximate sampler-expected score."""
    head_probs = jnp.exp(head_log_probs)  # (..., K)
    sampler_probs = jnp.exp(sampler_head_log_probs)  # (..., K)
    sampler_tail = jnp.maximum(1 - sampler_probs.sum(axis=-1), 0)  # (...,)
    # The head sum can round to one in FP32; floor the denominator at machine epsilon.
    learner_tail = jnp.maximum(1 - head_probs.sum(axis=-1), jnp.finfo(jnp.float32).eps)  # (...,)
    residual = jax.lax.stop_gradient(sampler_probs - (sampler_tail / learner_tail)[..., None] * head_probs)  # (..., K)
    return (residual * head_log_probs).sum(axis=-1)  # (...,)


def objective(
    log_probs: jax.Array,  # (B, T) learner sampled-token log-probabilities, already shifted.
    head_log_probs: jax.Array,  # (B, T, K) learner scores for recorded sampler alternatives.
    sampler_head_log_probs: jax.Array,  # (B, T, K)
    old_log_probs: jax.Array,  # (B, T) recorded sampled-token log-probabilities.
    advantages: jax.Array,  # (B,)
    mask: jax.Array,  # (B, T) retained sampled tokens.
    config: StableReinforceConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:  # (), scalar metric leaves.
    """() loss and scalar diagnostics: token mean per rollout, then mean over all B rows."""
    log_probs = jnp.where(mask, log_probs, 0)  # (B, T)
    old_log_probs = jax.lax.stop_gradient(jnp.where(mask, old_log_probs, 0))  # (B, T)
    head_log_probs = jnp.where(mask[..., None], head_log_probs, 0)  # (B, T, K)
    sampler_head_log_probs = jax.lax.stop_gradient(
        jnp.where(mask[..., None], sampler_head_log_probs, -jnp.inf)
    )  # (B, T, K)
    correction = score_correction(head_log_probs, sampler_head_log_probs)  # (B, T)

    def average(values: jax.Array) -> jax.Array:  # (B, T) -> ()
        return (jnp.where(mask, values, 0).sum(axis=1) / jnp.maximum(mask.sum(axis=1), 1)).mean()  # ()

    policy_loss = -average(advantages[:, None] * (log_probs - correction))  # ()
    log_ratio = log_probs - old_log_probs  # (B, T)
    if config.kl_estimator == "k1":
        kl = -log_ratio  # (B, T)
    else:
        # Match the existing objective's exponentiation safety bound.
        bounded = jnp.clip(log_ratio, -20, 20)  # (B, T)
        kl = jnp.expm1(bounded) - bounded  # (B, T)
    sampler_kl = average(kl)  # ()
    loss = policy_loss + config.kl_coeff * sampler_kl  # ()
    return loss, {
        "train/policy_loss": policy_loss,  # ()
        "train/loss": loss,  # ()
        "train/score_correction_mean": average(correction),  # ()
        "train/sampler_kl": sampler_kl,  # ()
        "train/sampler_topk_fraction_mean": average(jnp.exp(sampler_head_log_probs).sum(axis=-1)),  # ()
        "train/rollout_log_prob_error_mean": average(jnp.abs(log_ratio)),  # ()
    }
