from dataclasses import replace
from operator import itemgetter
from pathlib import Path
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import struct
from flax.training.train_state import TrainState

from rl2 import ppo
from rl2.envs.craftax_env import CraftaxConfig, CraftaxEnv
from rl2.envs.observations import CraftaxObservation, Observation


@struct.dataclass
class Params:
    max_timesteps: int = 2


class State(NamedTuple):
    timestep: jax.Array  # ()
    terminal: jax.Array  # ()
    color: jax.Array  # ()


class Game:
    def get_obs(self, state: State) -> jax.Array:  # scalar state leaves -> (H=2, W=3, 3)
        return jnp.full((2, 3, 3), state.color)  # (H, W, 3)

    def reset(self, key: jax.Array, params: Params) -> tuple[jax.Array, State]:  # key: ()
        color = jax.random.uniform(key)  # ()
        return jnp.full((2, 3, 3), color), State(jnp.array(0), jnp.array(False), color)  # (H,W,3); scalar leaves

    def step_env(
        self,
        key: jax.Array,  # ()
        state: State,  # scalar leaves
        action: jax.Array,  # ()
        params: Params,
    ) -> tuple[jax.Array, State, jax.Array, jax.Array, dict[str, jax.Array]]:
        state = state._replace(timestep=state.timestep + 1, color=state.color / 2 + action)  # scalar leaves
        done = state.terminal | (state.timestep >= params.max_timesteps)  # ()
        return jnp.full((2, 3, 3), state.color), state, jnp.array(0.25), done, {}  # (H,W,3); scalar leaves


def test_render_observation_converts_only_video_pixels() -> None:
    pixels = (jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3) - 1) / 15  # (H=2, W=3, C=3)
    frame = CraftaxEnv.render_observation(pixels)  # (H, W, C)
    expected = np.rint(np.clip(np.asarray(pixels), 0, 1) * 255).astype(np.uint8)  # (H, W, C)
    np.testing.assert_array_equal(frame, expected)
    assert frame.dtype == jnp.uint8


def test_terminal_timeout_both_and_partial_autoreset() -> None:
    env = object.__new__(CraftaxEnv)
    env.grid_shape = None
    env.env = Game()
    env.render_env = env.env
    env.params = Params(max_timesteps=2)
    key = jax.random.key(0)  # ()
    observations, initial = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key, 4), env.params
    )  # (B=4,2,3,3); leaves: (B,)
    chex.assert_trees_all_equal(
        (observations, initial), jax.vmap(env.env.reset, in_axes=(0, None))(jax.random.split(key, 4), env.params)
    )
    assert np.unique(initial.color).size == 4
    state = initial._replace(timestep=jnp.array([0, 1, 1, 0]), terminal=jnp.array([True, False, True, False]))
    # state leaves: (B,), respectively terminal, timeout, both, continuing
    key = jax.random.key(1)  # ()
    actions = jnp.arange(4, dtype=jnp.int32)  # (B,) native action IDs
    next_observations, next_state, rewards, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(key, 4), state, actions, env.params
    )  # (B,H,W,C); leaves: (B,...); rewards/done/flags: (B,); final_observation: (B,H,W,C)
    np.testing.assert_array_equal(info["terminated"], [True, False, True, False])
    np.testing.assert_array_equal(info["truncated"], [False, True, True, False])
    np.testing.assert_array_equal(done, [True, True, True, False])
    np.testing.assert_array_equal(next_state.timestep, [0, 0, 0, 1])
    np.testing.assert_array_equal(info["final_state"].timestep, [1, 2, 2, 1])
    final = jax.tree.map(itemgetter(0), info["final_state"])  # scalar state leaves
    np.testing.assert_array_equal(env.render(final), env.render_observation(info["final_observation"][0]))
    np.testing.assert_array_equal(rewards, [0.25] * 4)
    assert next_observations.shape == info["final_observation"].shape == (4, 2, 3, 3)
    reset_keys = jax.vmap(jax.random.split)(jax.random.split(key, 4))[:, 1]  # (B,) typed keys
    reset_observations, reset_state = jax.vmap(env.reset, in_axes=(0, None))(
        reset_keys, env.params
    )  # (B,2,3,3); leaves: (B,)
    np.testing.assert_array_equal(next_observations[:3], reset_observations[:3])
    np.testing.assert_array_equal(next_state.color[:3], reset_state.color[:3])
    np.testing.assert_array_equal(next_observations[3], info["final_observation"][3])
    expected_final = jnp.broadcast_to((state.color / 2 + actions)[:, None, None, None], (4, 2, 3, 3))  # (B,H,W,3)
    np.testing.assert_array_equal(info["final_observation"], expected_final)
    next_observations, next_state, rewards, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        jax.random.split(key, 4), initial, jnp.zeros(4, jnp.int32), env.params
    )  # (B,H,W,C); leaves: (B,...); rewards/done/flags: (B,); final_observation: (B,H,W,C)
    np.testing.assert_array_equal(next_state.timestep, 1)
    np.testing.assert_array_equal(next_observations, info["final_observation"])


def flatten_observation(observation: Observation) -> jax.Array:
    """Reconstruct (D,) native symbolic values, or return (H,W,3) pixels."""
    if isinstance(observation, CraftaxObservation):
        return jnp.concatenate((observation.grid.reshape(-1), observation.features))  # (D,)
    return observation  # (H,W,3)


@pytest.mark.parametrize(
    "env_id,shape,frame_shape,num_actions",
    [
        ("Craftax-Classic-Pixels-v1", (63, 63, 3), (63, 63, 3), 17),
        ("Craftax-Pixels-v1", (130, 110, 3), (130, 110, 3), 43),
        ("Craftax-Classic-Symbolic-v1", (1345,), (63, 63, 3), 17),
        ("Craftax-Symbolic-v1", (8268,), (130, 110, 3), 43),
    ],
    ids=["classic-pixels", "full-pixels", "classic-symbolic", "full-symbolic"],
)
def test_real_craftax_reset_step_and_video(
    env_id: str, shape: tuple[int, ...], frame_shape: tuple[int, ...], num_actions: int
) -> None:
    pytest.importorskip("craftax")
    env = CraftaxEnv(CraftaxConfig(env_id=env_id, max_steps=1))
    key = jax.random.key(0)  # ()
    observations, state = env.reset(key, env.params)  # (*O,); leaves: (...)
    flat = flatten_observation(observations)  # (D,) or (H,W,3)
    assert flat.shape == shape and flat.dtype == jnp.float32
    if isinstance(observations, CraftaxObservation):
        grid_shape, features = ((7, 9, 21), 22) if "Classic" in env_id else ((9, 11, 83), 51)
        assert observations.grid.shape == grid_shape and observations.features.shape == (features,)
    assert env.action_space.n == num_actions
    chex.assert_trees_all_equal((flat, state), env.env.reset(key, env.params))
    next_observations, next_state, rewards, done, info = env.step(
        jax.random.key(1), state, jnp.array(0), env.params
    )  # (*O,); leaves: (...); scalars; final_observation: (*O,)
    assert flatten_observation(next_observations).shape == shape
    # Compiled rendering can differ from eager rendering by float32 rounding.
    np.testing.assert_allclose(
        flatten_observation(info["final_observation"]), env.env.get_obs(info["final_state"]), rtol=2e-6, atol=1e-7
    )
    chex.assert_shape((rewards, done, info["terminated"], info["truncated"]), ())
    np.testing.assert_array_equal(info["terminated"], False)
    np.testing.assert_array_equal(info["truncated"], True)
    np.testing.assert_array_equal(next_state.timestep, 0)
    frame = env.render(info["final_state"])  # (H,W,3)
    pixels = env.render_env.get_obs(info["final_state"])  # (H,W,3)
    np.testing.assert_array_equal(frame, env.render_observation(pixels))
    assert frame.shape == frame_shape and frame.dtype == jnp.uint8
    batched_obs, _ = jax.eval_shape(
        jax.vmap(env.reset, in_axes=(0, None)), jax.random.split(key, 2), env.params
    )  # (B=2,*O); state leaves: (B,...)
    for original, batched in zip(jax.tree.leaves(observations), jax.tree.leaves(batched_obs)):
        assert batched.shape == (2, *original.shape)


def test_symbolic_ppo_rollout_and_update() -> None:
    config = replace(
        ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo/craftax/symbolic.yaml", ppo.Config),
        env=CraftaxConfig(env_id="Craftax-Symbolic-v1", max_steps=2),
        num_envs=1,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        target_kl=None,
        model=ppo.ModelConfig(
            encoder=ppo.CraftaxEncoderConfig(channels=4, blocks=0, hidden_size=4, embedding_size=2),
            recurrent=ppo.LSTMConfig(hidden_size=2),
        ),
    )
    env = ppo.make_env(config)
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.key(0), 1), env.params
    )  # grid: (B=1,9,11,83), features: (B,51); state leaves: (B,...)
    model = ppo.ActorCritic(action_space=env.action_space, model=config.model, dtype=jnp.bfloat16)
    carry = model.initial_carry(1)  # leaves: (B,H=2)
    starts = jnp.ones(1, bool)  # (B,)
    params = model.init(jax.random.key(1), jax.tree.map(lambda x: x[None], observations), carry, starts[None])["params"]
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    runner = ppo.RolloutCarry(
        env_state=env_state,  # leaves: (B,...)
        observations=observations,  # grid: (B,9,11,83), features: (B,51)
        model_carry=carry,  # leaves: (B,H)
        episode_starts=starts,  # (B,)
        episode_returns=jnp.zeros(1),  # (B,)
        episode_step_counts=jnp.zeros(1, jnp.int32),  # (B,)
    )
    advanced, rollout = ppo.collect_rollout(runner, train_state, env, config, jax.random.key(2))
    batch = rollout.batch
    assert batch.observations.grid.shape == (2, 1, 9, 11, 83)
    assert batch.observations.features.shape == (2, 1, 51)
    selected = batch.select_sequences(jnp.array([0, 0]))
    chex.assert_shape(selected.observations.grid, (2, 2, 9, 11, 83))
    chex.assert_shape(selected.observations.features, (2, 2, 51))
    chex.assert_trees_all_equal(
        jax.tree.map(lambda x: x[:, 0], selected.observations), jax.tree.map(lambda x: x[:, 0], batch.observations)
    )
    assert batch.actions.shape == batch.returns.shape == (2, 1)
    assert batch.actions.dtype == jnp.int32
    assert bool(jnp.all((batch.actions >= 0) & (batch.actions < 43)))
    output = model.apply({"params": params}, batch.observations, carry, batch.episode_starts)
    np.testing.assert_allclose(output.policy.log_prob(batch.actions), batch.old_log_probs, atol=1e-5)
    assert int(rollout.episode_count) == 1 and int(rollout.episode_length_sum) == 2
    np.testing.assert_array_equal(advanced.env_state.timestep, 0)
    updated, _ = ppo.update(train_state, batch, config)
    assert int(updated.step) == 1
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(updated.params)))
    episode = ppo.play_episode(updated, config, jax.random.key(3), carry, record=False, env=env)
    assert episode.length == 2 and not episode.frames
