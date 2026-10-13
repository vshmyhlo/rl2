"""Seed reproducibility for initialization, action sampling, and a single update."""

from dataclasses import replace
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from rl2 import ppo
from rl2.model.encoders.rgb import ConvStage


def test_seeded_policy_and_update_are_reproducible() -> None:
    model = ppo.ActorCritic(
        ppo.DiscreteSpace(3),
        ppo.ModelConfig(
            encoder=ppo.RGBEncoderConfig(stages=(ConvStage(2, blocks=1),), embedding_size=4),
            recurrent=ppo.LSTMConfig(hidden_size=4),
        ),
    )
    obs = jnp.arange(2 * 1 * 8 * 8, dtype=jnp.uint8).reshape(2, 1, 8, 8)
    carry = model.initial_carry(2)
    starts = jnp.ones(2, dtype=jnp.bool_)
    config = replace(ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"), target_kl=None)
    optimizer = optax.adam(1e-3)

    def initialize(seed: int) -> TrainState:
        params = model.init(jax.random.key(seed), obs[None], carry, starts[None])["params"]
        return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    first, repeated, different = initialize(11), initialize(11), initialize(12)
    chex.assert_trees_all_equal(first, repeated)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(first.params), jax.tree.leaves(different.params), strict=True)
    )
    key = jax.random.key(13)
    apply_step = jax.jit(model.apply, static_argnames=("method",))
    memory, step_logits, values = apply_step(
        {"params": first.params}, obs, carry, starts, method="step"
    )  # memory leaves: (B, H); step_logits: (B, A); values: (B,)
    step_logits = step_logits.logits  # (..., A)
    chex.assert_trees_all_equal(
        ppo.ActorCriticOutput(memory, ppo.CategoricalPolicy(step_logits), values),
        apply_step({"params": repeated.params}, obs, carry, starts, method="step"),
    )
    action = jax.random.categorical(key, step_logits)  # (B,)
    log_prob = ppo.CategoricalPolicy(step_logits).log_prob(action)  # (B,)
    _, logits, _ = model.apply({"params": first.params}, obs[None], carry, starts[None])
    logits = logits.logits  # (..., A)
    np.testing.assert_array_equal(action, jax.random.categorical(key, logits[0]))
    batch = ppo.PPOBatch(
        observations=obs[None],  # (*S, *O)
        actions=action[None],  # (*S,)
        old_log_probs=log_prob[None],  # (*S,)
        values=values[None],  # (*S,)
        advantages=jnp.array([[1.0, -1.0]]),  # (*S,)
        returns=values[None] + 1,  # (*S,)
        model_carry=carry,  # leaves: (B, ...)
        episode_starts=starts[None],  # (*S,)
    )
    updated, metrics = ppo.update(first, batch, config)
    replayed, replayed_metrics = ppo.update(repeated, batch, config)
    chex.assert_trees_all_equal((updated, metrics), (replayed, replayed_metrics))
    assert int(updated.step) == 1
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves((updated, metrics)))
