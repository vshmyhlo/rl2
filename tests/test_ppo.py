import contextlib
from dataclasses import replace
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import gymnasium as gym
import cv2
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rl2 import ppo
from rl2.ppo import (
    ActorCritic,
    ResetLSTM,
    action_log_prob,
    explained_variance,
    gae,
    initial_carry,
    learning_rate_schedule,
    load_config,
    make_env,
    train,
    update,
    value,
)


class ShortEpisodes(gym.wrappers.TimeLimit):
    def reset(self, *, seed=None, options=None):
        # Seeded limits work in spawned workers and stagger resets across rollouts.
        if seed is not None:
            self._max_episode_steps = 3 if self.render_mode else 2 + 3 * ((seed - 1) % 2)
        return super().reset(seed=seed, options=options)


def short_env(env_id, render_mode=None, frame_stack=False, atari_preprocessing=False):
    env = gym.wrappers.TransformReward(
        make_env(env_id, render_mode, frame_stack, atari_preprocessing),
        lambda reward: 2.0,
    )
    return ShortEpisodes(env, max_episode_steps=3)


class PPOTests(unittest.TestCase):
    def test_atari_timeout_observation_is_current(self):
        # End on each offset within action repeat, including before pooling starts.
        gym_make = gym.make
        for frame_limit in (101, 102, 103, 104):
            with self.subTest(frame_limit=frame_limit):

                def limited_env(*args, **kwargs):
                    return gym_make(*args, **kwargs, max_num_frames_per_episode=frame_limit)

                with patch("rl2.ppo.gym.make", new=limited_env):
                    env = make_env("ALE/Pong-v5", atari_preprocessing=True)
                try:
                    env.env.noop_max = 0
                    obs, _ = env.reset(seed=1)
                    for _ in range(26):
                        obs, _, terminated, truncated, _ = env.step(0)
                        if terminated or truncated:
                            break
                    self.assertTrue(truncated)
                    self.assertFalse(terminated)
                    expected = cv2.resize(
                        env.unwrapped.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_AREA
                    )
                    np.testing.assert_array_equal(obs[-1], expected)
                finally:
                    env.close()

    def test_clipped_policy_loss_and_gradient_direction(self):
        config = replace(
            load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
            target_kl=None,
            entropy_coef=0.0,
            value_coef=0.0,
        )
        probabilities = np.array([0.25, 0.75, 0.75, 0.25], dtype=np.float32).reshape(2, 2)
        logits = jnp.log(jnp.stack((probabilities, 1 - probabilities), axis=-1))
        state = TrainState.create(
            apply_fn=lambda variables, obs, carry, starts: (carry, variables["params"]["logits"], jnp.zeros((2, 2))),
            params={"logits": logits},
            tx=optax.sgd(0.1),
        )
        advantages = np.array([[-3.0, -1.0], [1.0, 3.0]])
        batch = (
            jnp.zeros((2, 2, 1)),
            jnp.zeros((2, 2), dtype=jnp.int32),
            jnp.full((2, 2), np.log(0.5)),
            advantages,
            jnp.ones((2, 2)),
            initial_carry(2, 1),
            jnp.zeros((2, 2), dtype=bool),
        )
        updated, metrics = update(state, batch, config)
        normalized = advantages / advantages.std()
        ratio = probabilities / 0.5
        expected = -np.minimum(
            ratio * normalized, np.clip(ratio, 1 - config.clip_coef, 1 + config.clip_coef) * normalized
        ).mean()
        self.assertAlmostEqual(float(metrics[0]), expected, places=6)
        self.assertAlmostEqual(float(metrics[1]), 0.5, places=6)
        np.testing.assert_array_equal(updated.params["logits"][:, 0], logits[:, 0])  # Both clipped signs.
        self.assertLess(float(updated.params["logits"][0, 1, 0]), float(logits[0, 1, 0]))
        self.assertGreater(float(updated.params["logits"][1, 1, 0]), float(logits[1, 1, 0]))

    def test_recurrent_sequences_match_steps_and_reset_only_finished_env(self):
        model = ActorCritic(3, 16)
        obs = jax.random.randint(jax.random.key(2), (4, 2, 4, 84, 84), 0, 256, dtype=jnp.uint8)
        carry = initial_carry(2, 16)
        starts = jnp.array([[True, True], [False, False], [True, False], [False, False]])
        params = model.init(jax.random.key(1), obs, carry, starts)
        apply = jax.jit(model.apply)
        final, logits, values = apply(params, obs, carry, starts)
        stepped_logits, stepped_values = [], []
        for t in range(4):
            carry, policy, critic = apply(params, obs[t : t + 1], carry, starts[t : t + 1])
            stepped_logits.append(policy[0])
            stepped_values.append(critic[0])
        np.testing.assert_allclose(logits, jnp.stack(stepped_logits), atol=1e-6)
        np.testing.assert_allclose(values, jnp.stack(stepped_values), atol=1e-6)
        np.testing.assert_allclose(final, carry, atol=1e-6)
        # Splitting a rollout preserves memory; bootstrapping must not consume it.
        prefix_carry, _, _ = apply(params, obs[:2], initial_carry(2, 16), starts[:2])
        state = TrainState.create(apply_fn=model.apply, params=params["params"], tx=optax.sgd(0.0))
        peek = value(state, obs[2], prefix_carry, starts[2])
        np.testing.assert_allclose(peek, values[2], atol=1e-6)
        _, suffix_logits, suffix_values = apply(params, obs[2:], prefix_carry, starts[2:])
        np.testing.assert_allclose(suffix_values, values[2:], atol=1e-6)
        # A reset discards history for env 0; env 1 still depends on it.
        _, fresh_logits, fresh_values = apply(params, obs[2:], initial_carry(2, 16), starts[2:])
        np.testing.assert_allclose(suffix_logits[:, 0], fresh_logits[:, 0], atol=1e-6)
        np.testing.assert_allclose(suffix_values[:, 0], fresh_values[:, 0], atol=1e-6)
        self.assertGreater(float(jnp.max(jnp.abs(suffix_values[:, 1] - fresh_values[:, 1]))), 1e-5)

    def test_lstm_gradients_follow_history_but_stop_at_episode_reset(self):
        cell = nn.scan(ResetLSTM, variable_broadcast="params", split_rngs={"params": False}, in_axes=0, out_axes=0)(8)
        inputs = jax.random.normal(jax.random.key(2), (4, 2, 5))
        starts = jnp.zeros((4, 2), dtype=bool).at[2, 0].set(True)
        carry = initial_carry(2, 8)
        params = cell.init(jax.random.key(1), carry, (inputs, starts))
        grads = jax.grad(lambda x: cell.apply(params, carry, (x, starts))[1][-1].sum())(inputs)
        np.testing.assert_array_equal(grads[:2, 0], 0)
        self.assertGreater(float(jnp.linalg.norm(grads[:2, 1])), 0)
        self.assertGreater(float(jnp.linalg.norm(grads[2:, 0])), 0)

    def test_recurrent_minibatches_require_whole_environments(self):
        config = replace(
            load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
            num_envs=3,
            num_steps=4,
            num_minibatches=2,
        )
        with self.assertRaisesRegex(ValueError, "num_envs must be divisible"):
            train(config)

    def test_learning_rate_schedule(self):
        config = replace(
            load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
            num_envs=2,
            num_steps=4,
            total_steps=19,
            num_minibatches=2,
            update_epochs=3,
            anneal_lr=True,
        )
        schedule = learning_rate_schedule(config)
        # Advance by rollouts, independently of how many optimizer updates were applied.
        expected = config.learning_rate * np.array([1.0, 0.5, 0.0, 0.0])
        np.testing.assert_allclose([schedule(i) for i in range(4)], expected, rtol=1e-6)
        constant = learning_rate_schedule(replace(config, anneal_lr=False))
        np.testing.assert_allclose([constant(i) for i in range(14)], config.learning_rate, rtol=1e-6)
        single = learning_rate_schedule(replace(config, total_steps=8))
        self.assertAlmostEqual(float(single(0)), config.learning_rate)
        self.assertEqual(float(single(1)), 0.0)

    def test_kl_rejects_update_without_changing_optimizer(self):
        config = replace(load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"), target_kl=0.01)
        state = TrainState.create(
            apply_fn=lambda variables, obs, carry, starts: (carry, variables["params"]["logits"], jnp.zeros(2)),
            params={"logits": jnp.log(jnp.array([[0.9, 0.1], [0.1, 0.9]]))},
            tx=optax.adam(0.01),
        )
        batch = (
            jnp.zeros((2, 1)),
            jnp.zeros(2, dtype=jnp.int32),
            jnp.full(2, np.log(0.5)),
            jnp.array([1.0, -1.0]),
            jnp.zeros(2),
            initial_carry(2, 1),
            jnp.zeros(2, dtype=bool),
        )
        stopped, metrics = update(state, batch, config)
        self.assertGreater(float(metrics[3]), config.target_kl)
        for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(stopped)):
            np.testing.assert_array_equal(before, after)
        continued, _ = update(state, batch, replace(config, target_kl=None))
        self.assertEqual(int(continued.step), 1)
        self.assertFalse(np.array_equal(state.params["logits"], continued.params["logits"]))

    def test_policy_diagnostics(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml")
        probabilities = jnp.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2], [0.52, 0.48]])
        state = TrainState.create(
            apply_fn=lambda variables, obs, carry, starts: (carry, variables["params"]["logits"], jnp.zeros(4)),
            params={"logits": jnp.log(probabilities)},
            tx=optax.sgd(0.0),
        )
        batch = (
            jnp.zeros((4, 1)),
            jnp.zeros(4, dtype=jnp.int32),
            jnp.full(4, np.log(0.5)),
            jnp.arange(4.0),
            jnp.arange(4.0),
            initial_carry(4, 1),
            jnp.zeros(4, dtype=bool),
        )
        _, metrics = update(state, batch, config)
        ratios = np.array([0.4, 1.0, 1.6, 1.04])
        self.assertAlmostEqual(float(metrics[3]), float(np.mean(ratios - 1 - np.log(ratios))), places=6)
        self.assertEqual(float(metrics[4]), 0.5)
        same_policy_batch = (*batch[:2], jnp.log(probabilities[:, 0]), *batch[3:])
        _, metrics = update(state, same_policy_batch, config)
        self.assertAlmostEqual(float(metrics[3]), 0.0, places=6)
        self.assertEqual(float(metrics[4]), 0.0)

    def test_explained_variance(self):
        returns = np.array([0.0, 1.0, 2.0])
        self.assertEqual(explained_variance(returns, returns), 1.0)
        self.assertEqual(explained_variance(np.zeros(3), returns), 0.0)
        self.assertEqual(explained_variance(-returns, returns), -3.0)
        self.assertTrue(np.isnan(explained_variance(returns, np.ones(3))))

    def test_gae_stops_at_game_over(self):
        advantages, returns = gae(
            np.array([[1.0], [2.0]], dtype=np.float32),
            np.array([[True], [False]]),
            np.array([[0.5], [1.0]], dtype=np.float32),
            np.array([3.0], dtype=np.float32),
            0.9,
            0.8,
        )
        np.testing.assert_allclose(advantages, [[0.5], [3.7]], rtol=1e-6)
        np.testing.assert_allclose(returns, [[1.0], [4.7]], rtol=1e-6)

    def test_gae_timeout_bootstrap_and_trace(self):
        # The final reward contains gamma * V(final observation).
        advantages, returns = gae(
            np.array([[1.0], [2.0 + 0.9 * 3.0]], dtype=np.float32),
            np.array([[False], [True]]),
            np.array([[0.5], [1.0]], dtype=np.float32),
            np.array([100.0], dtype=np.float32),
            0.9,
            0.8,
        )
        np.testing.assert_allclose(advantages, [[4.064], [3.7]], rtol=1e-6)
        np.testing.assert_allclose(returns, [[4.564], [4.7]], rtol=1e-6)

    def test_atari_training_across_resets(self):
        for mode in ("sync", "async"):
            for frame_stack in (False, True):
                for preprocessing in (False, True):
                    with self.subTest(vector_env=mode, frame_stack=frame_stack, preprocessing=preprocessing):
                        self.check_atari_training(mode, frame_stack=frame_stack, preprocessing=preprocessing)

    def test_early_stop_resumes_next_rollout_and_anneals_lr(self):
        self.check_atari_training("sync", target_kl=1e-8, update_epochs=3)

    def check_atari_training(
        self,
        mode: str,
        target_kl: float | None = None,
        update_epochs: int = 1,
        frame_stack: bool = False,
        preprocessing: bool = False,
    ) -> None:
        config = replace(
            load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
            total_steps=16,
            num_envs=2,
            num_steps=4,
            num_minibatches=2,
            update_epochs=update_epochs,
            video_every_episodes=2,
            vector_env=mode,
            target_kl=target_kl,
            frame_stack=frame_stack,
            atari_preprocessing=preprocessing,
        )
        output = io.StringIO()
        training_steps = 0
        checked_rollouts = set()
        previous_carry = initial_carry(config.num_envs, config.lstm_hidden_size)

        def checked_act(state, obs, carry, starts, key):
            nonlocal training_steps, previous_carry
            image_shape = (84, 84) if preprocessing else (210, 160, 3)
            self.assertEqual(obs.shape[1:], (4 if frame_stack else 1, *image_shape))
            if obs.shape[0] == config.num_envs:
                # Memory crosses rollout boundaries and video games cannot overwrite it.
                np.testing.assert_array_equal(carry, previous_carry)
                np.testing.assert_array_equal(starts, [training_steps % 2 == 0, training_steps % 5 == 0])
                result = ppo_act(state, obs, carry, starts, key)
                previous_carry = result[3]
                training_steps += 1
                return result
            return ppo_act(state, obs, carry, starts, key)

        def checked_update(state, batch, config):
            self.assertEqual(batch[0].shape[:2], (config.num_steps, config.num_envs // config.num_minibatches))
            if training_steps not in checked_rollouts:
                # Replay with unchanged parameters must recover rollout action probabilities.
                _, logits, _ = state.apply_fn({"params": state.params}, batch[0], batch[5], batch[6])
                np.testing.assert_allclose(action_log_prob(logits, batch[1]), batch[2], atol=1e-6)
                checked_rollouts.add(training_steps)
            return update(state, batch, config)

        ppo_act = ppo.act
        with TemporaryDirectory() as log_dir:
            with (
                patch("rl2.ppo.make_env", new=short_env),
                patch("rl2.ppo.act", new=checked_act),
                patch("rl2.ppo.update", new=checked_update),
                contextlib.redirect_stdout(output),
            ):
                state = train(replace(config, log_dir=log_dir))
            (run_dir,) = Path(log_dir).iterdir()
            events = EventAccumulator(str(run_dir)).Reload()
            for tag in (
                "losses/policy",
                "losses/value",
                "policy/entropy",
                "policy/approx_kl",
                "policy/clip_fraction",
                "value/explained_variance",
                "charts/learning_rate",
                "charts/updates_per_rollout",
                "policy/early_stop",
                "charts/steps_per_second",
                "charts/return_mean_100",
                "charts/episode_length_mean_100",
                "charts/total_episodes",
            ):
                scalars = events.Scalars(tag)
                self.assertEqual([event.step for event in scalars], [8, 16])
                self.assertTrue(all(np.isfinite(event.value) for event in scalars))
            np.testing.assert_allclose(
                [event.value for event in events.Scalars("charts/learning_rate")],
                [config.learning_rate, config.learning_rate / 2],
                rtol=1e-6,
            )
            self.assertEqual(
                [event.value for event in events.Scalars("charts/updates_per_rollout")],
                [1, 1] if target_kl is not None else [2, 2],
            )
            self.assertEqual(
                [event.value for event in events.Scalars("policy/early_stop")],
                [1, 1] if target_kl is not None else [0, 0],
            )
            self.assertEqual(
                [event.value for event in events.Scalars("charts/total_episodes")],
                [2, 5],
            )
            np.testing.assert_allclose(
                [event.value for event in events.Scalars("charts/episode_length_mean_100")],
                [2.0, 2.6],
            )
            np.testing.assert_allclose(
                [event.value for event in events.Scalars("charts/return_mean_100")],
                [4.0, 5.2],
            )
            config_text = events.Tensors("config/text_summary")[0].tensor_proto.string_val[0]
            self.assertIn(f"env_id: {config.env_id}".encode(), config_text)
            videos = events.Images("gameplay")
            self.assertEqual([event.step for event in videos], [8, 16])
            for video in videos:
                with Image.open(io.BytesIO(video.encoded_image_string)) as image:
                    self.assertEqual(image.format, "GIF")
                    self.assertEqual(image.size, (160, 210))
        self.assertEqual(int(state.step), 2 if target_kl is not None else 4)
        self.assertEqual(checked_rollouts, {4, 8})
        for leaf in jax.tree.leaves(state.params):
            self.assertTrue(np.isfinite(leaf).all())
        self.assertIn("step=16", output.getvalue())
        self.assertIn("step=16 episodes=5", output.getvalue())
        self.assertNotIn("return=n/a", output.getvalue())
        self.assertIn("episode_length=2.6", output.getvalue())


if __name__ == "__main__":
    unittest.main()
