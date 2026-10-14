from dataclasses import replace
from pathlib import Path
from typing import Literal, NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from rl2 import ppo
from rl2.actions import BetaPolicy, ContinuousSpace, TruncatedGaussianPolicy
from rl2.envs.base import BaseEnv, EnvStep
from rl2.model.encoders.rgb import ConvStage


class State(NamedTuple):
    action: jax.Array  # (2, 1)


class ContinuousEnv(BaseEnv[State, None]):
    action_space = ContinuousSpace((2, 1), (-2.0, 0.0), (4.0, 1.0))
    params = None

    def reset(self, key: jax.Array, params: None) -> tuple[jax.Array, State]:  # key: ()
        return jnp.zeros((1, 4, 4), jnp.uint8), State(jnp.zeros((2, 1)))  # (C,H,W); action: (2,1)

    def step(
        self, key: jax.Array, state: State, action: jax.Array, params: None
    ) -> EnvStep[State]:  # key: (); action: (2,1)
        observation = jnp.zeros((1, 4, 4), jnp.uint8)  # (C,H,W)
        state = State(action)
        return EnvStep(
            observation,
            state,
            action.sum(),
            jnp.array(False),
            {
                "terminated": jnp.array(False),  # ()
                "truncated": jnp.array(False),  # ()
                "final_observation": observation,  # (C,H,W)
                "final_state": state,  # action: (2,1)
            },
        )

    def render(self, state: State) -> jax.Array:  # action: (2,1) -> (H,W,3)
        return jnp.zeros((4, 4, 3), jnp.uint8)  # (H,W,3)


@pytest.mark.parametrize("continuous_policy", ["truncated_gaussian", "beta"])
def test_continuous_recurrent_training_replays_bounded_actions(
    continuous_policy: Literal["truncated_gaussian", "beta"],
) -> None:
    config = replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml", ppo.Config),
        num_envs=2,
        num_steps=2,
        num_minibatches=2,
        update_epochs=1,
        target_kl=None,
        entropy_coef=0.01,
        model=ppo.ModelConfig(
            continuous_policy=continuous_policy,
            encoder=ppo.RGBEncoderConfig(stages=(ConvStage(2, blocks=0),), embedding_size=2),
            recurrent=ppo.LSTMConfig(hidden_size=2),
        ),
    )
    env = ContinuousEnv()
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 2), env.params
    )  # (B=2,C=1,H=4,W=4); action: (B,2,1)
    model = ppo.ActorCritic(env.action_space, config.model, dtype=jnp.bfloat16)
    carry = model.initial_carry(2)  # leaves: (B,H=2)
    starts = jnp.ones(2, bool)  # (B,)
    params = model.init(jax.random.key(1), observations[None], carry, starts[None])["params"]
    # Extreme network outputs must still produce stable bounded distributions.
    params["policy_output"]["bias"] = jnp.array(
        [12.0, -12.0] * (2 if continuous_policy == "beta" else 1)
    )  # (D=2 or 2D=4,)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.inject_hyperparams(optax.adam)(learning_rate=config.learning_rate),
    )
    runner = ppo.RolloutCarry(env_state, observations, carry, starts, jnp.zeros(2), jnp.zeros(2, jnp.int32))
    key = jax.random.key(2)  # ()
    advanced, rollout = ppo.collect_rollout(runner, train_state, env, config, key)
    batch = rollout.batch
    assert batch.actions.shape == (2, 2, 2, 1)
    assert batch.actions.dtype == jnp.float32
    np.testing.assert_array_equal(advanced.env_state.action, batch.actions[-1])
    low, high = jnp.asarray((env.action_space.low, env.action_space.high)).reshape(2, 2, 1)  # each (2,1)
    assert bool(jnp.all((batch.actions >= low) & (batch.actions <= high)))
    output = jax.jit(model.apply)({"params": params}, batch.observations, carry, batch.episode_starts)
    assert isinstance(output, ppo.ActorCriticOutput)
    if continuous_policy == "beta":
        assert isinstance(output.policy, BetaPolicy)
        assert output.policy.alpha.shape == output.policy.beta.shape == (2, 2, 2)
        assert output.policy.alpha.dtype == output.policy.beta.dtype == jnp.float32
        assert bool(jnp.all(output.policy.alpha >= 1) & jnp.all(output.policy.beta >= 1))
        assert "policy_log_std" not in params
    else:
        assert isinstance(output.policy, TruncatedGaussianPolicy)
        assert output.policy.mean.shape == output.policy.log_std.shape == (2, 2, 2)
        assert output.policy.mean.dtype == output.policy.log_std.dtype == jnp.float32
    assert output.values.shape == (2, 2)
    np.testing.assert_allclose(output.policy.log_prob(batch.actions), batch.old_log_probs, atol=1e-5)
    selected = batch.select_sequences(jnp.array([1]))
    assert selected.actions.shape == (2, 1, 2, 1)
    np.testing.assert_array_equal(selected.actions, batch.actions[:, 1:])
    result = ppo.train_step(runner, train_state, env, config, key, ppo.init_metrics())
    chex.assert_trees_all_equal(result, ppo.train_step(runner, train_state, env, config, key, ppo.init_metrics()))
    assert int(result.train_state.step) == 2
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves((result.train_state.params, result.metrics)))
    if continuous_policy == "beta":
        assert not np.array_equal(result.train_state.params["policy_output"]["bias"], params["policy_output"]["bias"])
    else:
        assert not np.array_equal(result.train_state.params["policy_log_std"], params["policy_log_std"])
    # Direct deterministic updates match the trainer after the same minibatch shuffle.
    expected = train_state
    _, shuffle_key = jax.random.split(rollout.key)  # each ()
    permutation = jax.random.permutation(shuffle_key, 2)  # (B,)
    for index in range(2):
        expected, _ = ppo.update(
            expected,
            batch.select_sequences(permutation[index : index + 1]),
            config,
        )
    chex.assert_trees_all_close(expected.params, result.train_state.params, atol=1e-5)
