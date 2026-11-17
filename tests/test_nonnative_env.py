"""Abstract interface requirements for host-stepped environments."""

import pytest

from rl2.envs import BaseNonnativeEnv
from rl2.envs.gym_env import GymEnv


def test_base_nonnative_env_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseNonnativeEnv()


@pytest.mark.parametrize("missing", ["reset", "step", "render", "close"])
def test_incomplete_adapter_cannot_be_instantiated(missing: str) -> None:
    methods = {name: GymEnv.__dict__[name] for name in ("reset", "step", "render", "close")}
    del methods[missing]
    incomplete = type("IncompleteEnv", (BaseNonnativeEnv,), methods)
    with pytest.raises(TypeError, match=missing):
        incomplete()
