"""Abstract interface requirements for host-stepped environments."""

import subprocess
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

from rl2.envs import GymStyleEnv
from rl2.envs.gym_env import GymConfig, GymEnv
from rl2.envs.gym_style_env import convert_observation, host_observation, replace_observation


def test_gym_style_env_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        GymStyleEnv()


@pytest.mark.parametrize("missing", ["reset", "step", "render", "close"])
def test_incomplete_adapter_cannot_be_instantiated(missing: str) -> None:
    methods = {name: GymEnv.__dict__[name] for name in ("reset", "step", "render", "close")}
    del methods[missing]
    incomplete = type("IncompleteEnv", (GymStyleEnv,), methods)
    with pytest.raises(TypeError, match=missing):
        incomplete()


def test_gym_environment_runs_without_importing_jax() -> None:
    code = """
import sys
from contextlib import closing
import numpy as np
from rl2.envs.gym_env import GymConfig, GymEnv

with closing(GymEnv(config=GymConfig(), num_envs=1, seed=0)) as env:
    initial = env.reset()
    result = env.step(np.zeros(1, dtype=np.int32))
    assert isinstance(initial, np.ndarray)
    assert all(isinstance(leaf, np.ndarray) for leaf in result)
assert 'jax' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_gym_partial_construction_closes_created_instances() -> None:
    instance = Mock()
    with (
        patch("rl2.envs.gym_env.gym.make", side_effect=[instance, RuntimeError("construction failed")]),
        pytest.raises(RuntimeError, match="construction failed"),
    ):
        GymEnv(config=GymConfig(), num_envs=2, seed=0)
    instance.close.assert_called_once_with()


def test_nested_numpy_observations_preserve_layout_and_snapshots() -> None:
    image = np.arange(24, dtype=np.uint8).reshape(2, 1, 3, 4)  # (B, C, H, W)
    state = np.ones((2, 3), dtype=np.float64)  # (B, O)
    initial = host_observation({"sensors": (image, {"state": state})})  # nested (B, *O) leaves
    final = host_observation(initial)  # nested (B, *O) leaves
    terminal = {"sensors": (np.full((1, 3, 4), 99, dtype=np.uint8), {"state": np.zeros(3)})}  # (*O,) leaves
    replace_observation(final, terminal, index=1)
    converted = convert_observation(final, channels_first=True)  # nested (B, *O) leaves
    pixels, vectors = converted["sensors"]  # (B, H, W, C); {state: (B, O)}
    assert pixels.shape == (2, 3, 4, 1)
    assert pixels.dtype == np.uint8
    assert vectors["state"].shape == (2, 3)
    assert vectors["state"].dtype == np.float64
    np.testing.assert_array_equal(pixels[0, ..., 0], image[0, 0])
    np.testing.assert_array_equal(pixels[1], 99)
    np.testing.assert_array_equal(vectors["state"], [[1, 1, 1], [0, 0, 0]])
    image[:] = 0
    state[:] = 0
    np.testing.assert_array_equal(initial["sensors"][0], np.arange(24).reshape(2, 1, 3, 4))
    np.testing.assert_array_equal(initial["sensors"][1]["state"], 1)
    unchanged = convert_observation(initial, channels_first=False)  # nested (B, *O) leaves
    assert unchanged["sensors"][0].shape == (2, 1, 3, 4)
