"""Compare exact training results from fresh Python processes."""

import contextlib
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import gymnasium as gym
import jax
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rl2 import ppo


class ShortGame(gym.wrappers.TimeLimit):
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._max_episode_steps = 7 if self.render_mode else 31 + 16 * (seed % 2)
        return super().reset(seed=seed, options=options)


make_atari = ppo.make_env


def short_game(env_id, render_mode=None, frame_stack=False, atari_preprocessing=False):
    return ShortGame(make_atari(env_id, render_mode, frame_stack, atari_preprocessing), max_episode_steps=31)


def digest(tree):
    result = hashlib.sha256()
    for leaf in jax.tree.leaves(jax.device_get(tree)):
        array = np.asarray(leaf)
        if not np.isfinite(array).all():
            raise AssertionError("Non-finite training state")
        result.update(str((array.dtype, array.shape)).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def snapshot(path, mode, seed, videos):
    config = replace(ppo.load_config(Path(__file__).resolve().parents[1] / "configs/ppo.yaml"),
                     seed=seed, vector_env=mode, total_steps=128, num_envs=2,
                     num_steps=16, num_minibatches=2, update_epochs=2,
                     video_every_episodes=2 if videos else 0)
    trajectory = []
    original_act = ppo.act

    def record_act(state, obs, carry, episode_starts, key):
        result = original_act(state, obs, carry, episode_starts, key)
        if obs.shape[0] == config.num_envs:  # Exclude the separate video game.
            trajectory.append(digest((obs, carry, episode_starts, result)))
        return result

    with TemporaryDirectory() as log_dir:
        with patch("rl2.ppo.make_env", new=short_game), patch("rl2.ppo.act", new=record_act):
            with contextlib.redirect_stdout(io.StringIO()):
                state = ppo.train(replace(config, log_dir=log_dir))
        run_dir, = Path(log_dir).iterdir()
        events = EventAccumulator(str(run_dir)).Reload()
        metrics = {tag: [(event.step, event.value) for event in events.Scalars(tag)]
                   for tag in events.Tags()["scalars"] if tag != "charts/steps_per_second"}
        result = {"state": digest((state.step, state.params, state.opt_state)),
                  "trajectory": trajectory, "metrics": metrics}
        if videos:
            assert events.Images("gameplay"), "Expected recorded games"
        Path(path).write_text(json.dumps(result))


class ReproducibilityTests(unittest.TestCase):
    def test_fresh_process_runs(self):
        with TemporaryDirectory() as directory:
            def run(mode, seed=11, videos=False):
                path = Path(directory) / "snapshot.json"
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--snapshot",
                     str(path), mode, str(seed), str(int(videos))],
                    capture_output=True, text=True, timeout=180,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return json.loads(path.read_text())

            baseline = run("sync")
            for mode, videos in [("sync", False), ("async", False),
                                 ("async", False), ("async", True)]:
                with self.subTest(vector_env=mode, videos=videos):
                    self.assertEqual(baseline, run(mode, videos=videos))
            self.assertNotEqual(baseline["state"], run("async", seed=12)["state"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--snapshot":
        snapshot(sys.argv[2], sys.argv[3], int(sys.argv[4]), bool(int(sys.argv[5])))
    else:
        unittest.main()
