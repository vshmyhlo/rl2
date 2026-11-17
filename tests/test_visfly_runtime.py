"""Real headless rendering checks; run inside the --visfly image on an NVIDIA GPU."""

import os
from contextlib import closing

import numpy as np
import pytest

from rl2.envs.visfly_env import VisFlyConfig, VisFlyEnv

pytestmark = pytest.mark.skipif(os.environ.get("RL2_VISFLY_TESTS") != "1", reason="requires the VisFly GPU image")


@pytest.mark.parametrize("num_envs", [pytest.param(2, id="batch"), pytest.param(1, id="evaluation")])
def test_rendered_reset_step_and_autoreset(num_envs: int) -> None:
    config = VisFlyConfig(
        kwargs={
            "device": "cpu",
            "visual": True,
            "max_episode_steps": 1,
            "scene_kwargs": {"path": os.environ["VISFLY_SCENE_PATH"], "scene_type": "json"},
            "dynamics_kwargs": {"action_type": "bodyrate", "dt": 0.03, "ctrl_dt": 0.03},
            "random_kwargs": {
                "state_generator": {
                    "class": "Uniform",
                    "kwargs": [{"position": {"mean": [1.0, 0.0, 1.5], "half": [0.0, 0.0, 0.0]}}],
                }
            },
            "sensor_kwargs": [{"sensor_type": "DEPTH", "uuid": "depth", "resolution": [16, 16]}],
        }
    )
    with closing(VisFlyEnv(config=config, num_envs=num_envs, seed=1, evaluation=num_envs == 1)) as env:
        observation = env.reset()  # leaves: (B, *O)
        assert isinstance(observation, dict)
        assert observation["depth"].shape == (num_envs, 16, 16, 1)
        assert observation["state"].shape == (num_envs, 13)
        assert observation["target"].shape == (num_envs, 3)
        depth = observation["depth"].copy()  # (B, H, W, 1)
        actions = np.zeros((num_envs, *env.action_space.shape), dtype=np.float32)  # (B, A)
        result = env.step(actions)
        assert result.rewards.shape == result.terminated.shape == result.truncated.shape == (num_envs,)
        assert result.truncated.all()
        assert not result.terminated.any()
        assert result.final_observations["depth"].shape == depth.shape
        assert result.observations["depth"].shape == depth.shape
        np.testing.assert_array_equal(observation["depth"], depth)
        assert (depth > 0).any()
        assert env.step(actions).truncated.all()
