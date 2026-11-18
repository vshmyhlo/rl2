"""Real headless rendering checks; run inside the --visfly image on an NVIDIA GPU."""

import os
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

# Exercise the trainer's native-library import order before constructing VisFly.
from rl2.train_ppo_gym import VisFlyConfig, VisFlyEnv

pytestmark = pytest.mark.skipif(os.environ.get("RL2_VISFLY_TESTS") != "1", reason="requires the VisFly GPU image")


@pytest.mark.parametrize(
    ("num_envs", "num_scenes", "scenes"),
    [
        pytest.param(
            4,
            2,
            ("garage_empty/garage_empty_0.scene_instance.json", "v3_sc0_staging_08.scene_instance.json"),
            id="shared_context_rotation",
        ),
        pytest.param(1, 1, ("v3_sc0_staging_08.scene_instance.json",), id="articulated-evaluation"),
    ],
)
def test_rendered_reset_step_and_autoreset(num_envs: int, num_scenes: int, scenes: tuple[str, ...]) -> None:
    config = VisFlyConfig(
        num_scenes=num_scenes,
        scene_rotation_env_steps=num_envs,
        kwargs={
            "device": "cpu",
            "visual": True,
            "max_episode_steps": 1,
            "scene_randomization": True,
            "scene_kwargs": {"path": os.environ["VISFLY_SCENE_PATH"], "scene_type": "json"},
            "dynamics_kwargs": {"action_type": "bodyrate", "dt": 0.03, "ctrl_dt": 0.03},
            "sensor_kwargs": [
                {"sensor_type": "COLOR", "uuid": "color", "resolution": [16, 16]},
                {"sensor_type": "DEPTH", "uuid": "depth", "resolution": [16, 16]},
            ],
        },
    )
    with closing(VisFlyEnv(config=config, num_envs=num_envs, seed=1, evaluation=num_envs == 1)) as env:
        loader = env.backend.envs.sceneManager._scene_loader
        loader.paths = [str(Path(os.environ["VISFLY_SCENE_PATH"]) / scene) for scene in scenes]
        loader._select_index = list(range(len(scenes)))
        observation = env.reset()  # leaves: (B, *O)
        assert isinstance(observation, dict)
        assert set(observation) == {"state", "target", "rgb", "depth"}
        assert observation["rgb"].dtype == np.uint8
        assert observation["rgb"].shape == (num_envs, 16, 16, 3)
        assert observation["state"].shape == (num_envs, 13)
        assert observation["target"].shape == (num_envs, 3)
        rgb = observation["rgb"].copy()  # (B, H, W, 3)
        depth = observation["depth"].copy()  # (B, H, W, 1)
        assert depth.shape == (num_envs, 16, 16, 1)
        assert depth.dtype == np.float32
        assert (depth > 0).all()
        actions = np.zeros((num_envs, *env.action_space.shape), dtype=np.float32)  # (B, A)
        result = env.step(actions)
        assert result.rewards.shape == result.terminated.shape == result.truncated.shape == (num_envs,)
        assert result.truncated.all()
        assert not result.terminated.any()
        assert result.final_observations["rgb"].shape == rgb.shape
        assert result.observations["rgb"].shape == rgb.shape
        np.testing.assert_array_equal(observation["rgb"], rgb)
        np.testing.assert_array_equal(observation["depth"], depth)
        assert result.final_observations["depth"].shape == result.observations["depth"].shape == depth.shape
        assert (rgb > 0).any()
        assert env.step(actions).truncated.all()
