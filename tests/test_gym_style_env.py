"""Abstract interface requirements for host-stepped environments."""

import subprocess
import sys
from unittest.mock import Mock, patch

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
