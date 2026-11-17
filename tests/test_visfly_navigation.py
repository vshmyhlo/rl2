"""RGB backend behavior without loading Habitat's native renderer."""

import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl2.envs import visfly_env


class NavigationStub:
    def __init__(self, *, visual: bool, sensor_kwargs: list[dict[str, Any]], **kwargs: Any) -> None:
        self.kwargs = {"visual": visual, "sensor_kwargs": sensor_kwargs, **kwargs}
        self.device = torch.device("cpu")
        self.state = torch.ones((2, 13))  # (B, S)
        self.target = torch.zeros((2, 3))  # (B, 3)
        height, width = sensor_kwargs[0]["resolution"]
        self.sensor_obs = {"color": np.full((2, 3, height, width), 255, dtype=np.uint8)}  # (B, 3, H, W)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,)),
                "target": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                "color": gym.spaces.Box(low=0, high=255, shape=(3, height, width), dtype=np.uint8),
            }
        )


@pytest.fixture(scope="module")
def navigation_class() -> type[Any]:
    modules = {
        "habitat_sim": SimpleNamespace(SensorType=SimpleNamespace(COLOR=object())),
        "VisFly.envs.NavigationEnv": SimpleNamespace(NavigationEnv=NavigationStub),
        "VisFly.utils.type": SimpleNamespace(TensorDict=dict),
    }
    with patch.dict("sys.modules", modules):
        namespace = runpy.run_path(str(Path(visfly_env.__file__).parent / "visfly" / "navigation.py"))
    return namespace["RGBNavigationEnv"]


def test_rgb_backend_defaults_and_observation_space(navigation_class: type[Any]) -> None:
    env = navigation_class(num_scene=2, seed=7)
    assert env.kwargs["visual"] is True
    assert env.kwargs["num_scene"] == 2
    assert env.kwargs["seed"] == 7
    observation = env.get_observation()  # leaves: (B, *O)
    assert set(observation) == set(env.observation_space.spaces) == {"state", "target", "rgb"}
    assert observation["rgb"].shape == (2, 3, 64, 64)
    assert observation["rgb"].dtype == torch.uint8
    assert env.observation_space.contains({key: value[0].numpy() for key, value in observation.items()})


def test_rgb_terminal_image_survives_sensor_buffer_reuse(navigation_class: type[Any]) -> None:
    env = navigation_class(sensor_kwargs=[{"sensor_type": "COLOR", "uuid": "color", "resolution": [2, 4]}])
    observation = env.get_observation()  # leaves: (B, *O)
    terminal = observation["rgb"][0]  # (3, H, W): upstream retains a view before autoreset
    env.sensor_obs["color"][0] = 0
    after_reset = env.get_observation()  # leaves: (B, *O)
    assert after_reset["rgb"].shape == (2, 3, 2, 4)
    np.testing.assert_array_equal(terminal.numpy(), 255)
    np.testing.assert_array_equal(after_reset["rgb"][0].numpy(), 0)
    np.testing.assert_array_equal(after_reset["rgb"][1].numpy(), 255)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"visual": False}, id="nonvisual"),
        pytest.param({"sensor_kwargs": []}, id="no_camera"),
        pytest.param({"sensor_kwargs": [{"sensor_type": "DEPTH", "uuid": "color"}]}, id="depth"),
        pytest.param({"sensor_kwargs": [{"sensor_type": "COLOR", "uuid": "rgb"}]}, id="invalid_uuid"),
        pytest.param({"sensor_kwargs": [{"sensor_type": "COLOR", "uuid": "color"}] * 2}, id="multiple_cameras"),
    ],
)
def test_rgb_backend_rejects_incompatible_sensors(navigation_class: type[Any], kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="RGBNavigationEnv requires"):
        navigation_class(**kwargs)
