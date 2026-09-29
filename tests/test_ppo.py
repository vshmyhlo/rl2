import contextlib
from dataclasses import replace
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import gymnasium as gym
import jax
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rl2.ppo import gae, load_config, make_env, train


class PPOTests(unittest.TestCase):
    def test_gae_stops_at_game_over(self):
        advantages, returns = gae(
            np.array([[1.0], [2.0]], dtype=np.float32),
            np.array([[True], [False]]),
            np.array([[0.5], [1.0]], dtype=np.float32),
            np.array([3.0], dtype=np.float32), 0.9, 0.8,
        )
        np.testing.assert_allclose(advantages, [[0.5], [3.7]], rtol=1e-6)
        np.testing.assert_allclose(returns, [[1.0], [4.7]], rtol=1e-6)

    def test_gae_timeout_bootstrap_and_trace(self):
        # The final reward contains gamma * V(final observation).
        advantages, returns = gae(
            np.array([[1.0], [2.0 + 0.9 * 3.0]], dtype=np.float32),
            np.array([[False], [True]]),
            np.array([[0.5], [1.0]], dtype=np.float32),
            np.array([100.0], dtype=np.float32), 0.9, 0.8,
        )
        np.testing.assert_allclose(advantages, [[4.064], [3.7]], rtol=1e-6)
        np.testing.assert_allclose(returns, [[4.564], [4.7]], rtol=1e-6)

    def test_atari_training_across_resets(self):
        created = []

        def short_env(env_id):
            # Stagger timeouts across rollouts; raw rewards differ from clipped rewards.
            env = gym.wrappers.TransformReward(make_env(env_id), lambda reward: 2.0)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=2 + 3 * len(created))
            created.append(env)
            return env

        config = replace(load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
                         total_steps=16, num_envs=2, num_steps=4,
                         num_minibatches=2, update_epochs=1)
        output = io.StringIO()
        with TemporaryDirectory() as log_dir:
            with patch("rl2.ppo.make_env", side_effect=short_env), contextlib.redirect_stdout(output):
                state = train(replace(config, log_dir=log_dir))
            run_dir, = Path(log_dir).iterdir()
            events = EventAccumulator(str(run_dir)).Reload()
            for tag in ("losses/policy", "losses/value", "policy/entropy",
                        "charts/steps_per_second", "charts/return_mean_100",
                        "charts/episode_length_mean_100"):
                scalars = events.Scalars(tag)
                self.assertEqual([event.step for event in scalars], [8, 16])
                self.assertTrue(all(np.isfinite(event.value) for event in scalars))
            np.testing.assert_allclose(
                [event.value for event in events.Scalars("charts/episode_length_mean_100")],
                [2.0, 2.6],
            )
            np.testing.assert_allclose(
                [event.value for event in events.Scalars("charts/return_mean_100")],
                [4.0, 5.2],
            )
            config_text = events.Tensors("config/text_summary")[0].tensor_proto.string_val[0]
            self.assertIn(b"env_id: ALE/Pong-v5", config_text)
        self.assertEqual(int(state.step), 4)
        for leaf in jax.tree.leaves(state.params):
            self.assertTrue(np.isfinite(leaf).all())
        self.assertIn("step=16", output.getvalue())
        self.assertNotIn("return=n/a", output.getvalue())
        self.assertIn("episode_length=2.6", output.getvalue())


if __name__ == "__main__":
    unittest.main()
