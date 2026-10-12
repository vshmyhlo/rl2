from inspect import isabstract
from typing import Any

import pytest

from rl2.envs import BaseEnv
from rl2.envs.craftax_env import CraftaxEnv
from rl2.envs.gymnax_env import GymnaxEnv
from rl2.envs.kinetix_env import KinetixEnv
from rl2.envs.xminigrid_env import XMiniGridEnv


def test_base_env_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseEnv()


@pytest.mark.parametrize("missing", ["reset", "step", "render"])
def test_incomplete_adapter_cannot_be_instantiated(missing: str) -> None:
    methods = {name: GymnaxEnv.__dict__[name] for name in ("reset", "step", "render")}
    del methods[missing]
    incomplete = type("IncompleteEnv", (BaseEnv,), methods)
    with pytest.raises(TypeError, match=missing):
        incomplete()


@pytest.mark.parametrize("adapter", [GymnaxEnv, CraftaxEnv, KinetixEnv, XMiniGridEnv])
def test_adapters_implement_base_env(adapter: type[BaseEnv[Any, Any]]) -> None:
    assert issubclass(adapter, BaseEnv)
    assert not isabstract(adapter)
