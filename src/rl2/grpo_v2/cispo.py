"""CISPO: detached clipped importance weights multiplying sampled-token log-probabilities."""

from typing import Annotated, Literal

import jax
import jax.numpy as jnp
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class CISPOConfig:
    kind: Literal["cispo"] = "cispo"
    clip_low_threshold: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    clip_high_threshold: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 4.0

    def __post_init__(self) -> None:
        if self.clip_low_threshold > self.clip_high_threshold:
            raise ValueError("clip_low_threshold must not exceed clip_high_threshold")


DIAGNOSTICS = (
    "train/sampler_kl",
    "train/rollout_log_prob_error_mean",
    "train/rollout_importance_weight_mean",
    "train/rollout_importance_weight_clipped_fraction",
)


def objective(
    log_probs: jax.Array,  # (B, T) learner sampled-token log-probabilities, already shifted.
    old_log_probs: jax.Array,  # (B, T) recorded sampler log-probabilities.
    advantages: jax.Array,  # (B,)
    mask: jax.Array,  # (B, T) retained sampled tokens.
    config: CISPOConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:  # (), scalar metric leaves.
    """() loss and diagnostics: token mean per rollout, then mean over all B rows."""
    log_probs = jnp.where(mask, log_probs, 0)  # (B, T)
    old_log_probs = jax.lax.stop_gradient(jnp.where(mask, old_log_probs, 0))  # (B, T)
    log_ratio = log_probs - old_log_probs  # (B, T)
    log_low = jnp.log(config.clip_low_threshold)  # (); -inf disables the lower bound.
    log_high = jnp.log(config.clip_high_threshold)  # ()
    # Apply the configured upper cap before exp to avoid overflow without adding a floor.
    ratios = jnp.exp(jnp.minimum(log_ratio, log_high))  # (B, T)
    weights = jax.lax.stop_gradient(jnp.clip(ratios, config.clip_low_threshold, config.clip_high_threshold))  # (B, T)

    def average(values: jax.Array) -> jax.Array:  # (B, T) -> ()
        return (jnp.where(mask, values, 0).sum(axis=1) / jnp.maximum(mask.sum(axis=1), 1)).mean()  # ()

    loss = -average(weights * advantages[:, None] * log_probs)  # ()
    return loss, {
        "train/policy_loss": loss,  # ()
        "train/sampler_kl": average(jnp.expm1(log_ratio) - log_ratio),  # ()
        "train/rollout_log_prob_error_mean": average(jnp.abs(log_ratio)),  # ()
        "train/rollout_importance_weight_mean": average(weights),  # ()
        "train/rollout_importance_weight_clipped_fraction": average(
            (log_ratio < log_low) | (log_ratio > log_high)
        ),  # ()
    }
