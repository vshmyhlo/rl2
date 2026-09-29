import contextlib
import io
import unittest
from unittest.mock import patch

import gymnasium as gym
import jax
import numpy as np

from rl2.ppo import Config, gae, make_env, train


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
            # Stagger timeouts to exercise selective vector resets.
            env = gym.wrappers.TimeLimit(make_env(env_id), max_episode_steps=2 + len(created))
            created.append(env)
            return env

        config = Config(total_steps=16, num_envs=2, num_steps=4,
                        num_minibatches=2, update_epochs=1)
        output = io.StringIO()
        with patch("rl2.ppo.make_env", side_effect=short_env), contextlib.redirect_stdout(output):
            state = train(config)
        self.assertEqual(int(state.step), 4)
        for leaf in jax.tree.leaves(state.params):
            self.assertTrue(np.isfinite(leaf).all())
        self.assertIn("step=16", output.getvalue())
        self.assertNotIn("return=n/a", output.getvalue())


if __name__ == "__main__":
    unittest.main()
