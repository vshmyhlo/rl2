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
from rl2.observation_encoder import ConvStage


def test_seeded_policy_and_update_are_reproducible() -> None:
    model = ppo.ActorCritic(3, 4, encoder_stages=(ConvStage(2, blocks=1),), embedding_size=4)
    obs = jnp.arange(2 * 1 * 8 * 8, dtype=jnp.uint8).reshape(2, 1, 8, 8)
    carry = ppo.initial_carry(2, 4)
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
    action, log_prob, values, memory = ppo.act(first, obs, carry, starts, key)
    chex.assert_trees_all_equal((action, log_prob, values, memory), ppo.act(repeated, obs, carry, starts, key))
    _, logits, _ = model.apply({"params": first.params}, obs[None], carry, starts[None])
    np.testing.assert_array_equal(action, jax.random.categorical(key, logits[0]))
    batch = (
        obs[None],
        action[None],
        log_prob[None],
        jnp.array([[1.0, -1.0]]),
        values[None] + 1,
        carry,
        starts[None],
    )
    updated, metrics = ppo.update(first, batch, config)
    replayed, replayed_metrics = ppo.update(repeated, batch, config)
    chex.assert_trees_all_equal((updated, metrics), (replayed, replayed_metrics))
    assert int(updated.step) == 1
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves((updated, metrics)))
