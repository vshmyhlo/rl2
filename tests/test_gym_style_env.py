"""Abstract interface requirements for host-stepped environments."""

import subprocess
import sys
from contextlib import ExitStack, closing
from unittest.mock import Mock, patch

import numpy as np
import pytest

from rl2.envs import GymStyleEnv
from rl2.envs.gym_env import GymConfig, GymEnv


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


def test_gym_environment_returns_numpy_arrays() -> None:
    code = """
from contextlib import closing
import numpy as np
from rl2.envs.gym_env import GymConfig, GymEnv

with closing(GymEnv(config=GymConfig(), num_envs=1, seed=0)) as env:
    initial = env.reset()
    result = env.step(np.zeros(1, dtype=np.int32))
    assert isinstance(initial, np.ndarray)
    assert all(isinstance(leaf, np.ndarray) for leaf in result)
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


@pytest.mark.parametrize(
    ("num_envs", "frame_shape"),
    [
        pytest.param(1, (2, 3, 3), id="single_rgb"),
        pytest.param(2, (2, 3, 4), id="batch_rgba"),
        pytest.param(1, (1, 2, 3, 3), id="singleton_backend_batch"),
    ],
)
def test_gym_render_preserves_batch_and_environment_order(num_envs: int, frame_shape: tuple[int, ...]) -> None:
    with (
        closing(GymEnv(config=GymConfig(), num_envs=num_envs, seed=0)) as env,
        ExitStack() as patches,
    ):
        for index, instance in enumerate(env.backend.envs):
            frame = np.full(frame_shape, index + 1, dtype=np.float32)  # (H, W, C) or (1, H, W, C)
            patches.enter_context(patch.object(instance, "render", return_value=frame))
        frames = env.render()  # (B, H, W, 3)
        assert frames.shape == (num_envs, 2, 3, 3)
        assert frames.dtype == np.uint8
        for index in range(num_envs):
            np.testing.assert_array_equal(frames[index], index + 1)


@pytest.mark.parametrize("frame_shape", [(2, 3), (2, 3, 2), (2, 2, 3, 3)])
def test_gym_render_rejects_invalid_images(frame_shape: tuple[int, ...]) -> None:
    with (
        closing(GymEnv(config=GymConfig(), num_envs=1, seed=0)) as env,
        patch.object(env.backend.envs[0], "render", return_value=np.zeros(frame_shape)),
        pytest.raises(ValueError, match="RGB array per environment"),
    ):
        env.render()
