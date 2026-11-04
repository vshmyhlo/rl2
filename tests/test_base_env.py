from inspect import isabstract
from typing import Any

import pytest
from pydantic import ValidationError

from rl2.envs import BaseEnv
from rl2.envs.craftax_env import CraftaxConfig, CraftaxEnv
from rl2.envs.gymnax_env import GymnaxConfig, GymnaxEnv
from rl2.envs.kinetix_env import KinetixConfig, KinetixEnv
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


@pytest.mark.parametrize("config_class", [GymnaxConfig, CraftaxConfig, KinetixConfig])
def test_timeout_limit_reserves_native_terminal_sentinel(
    config_class: type[GymnaxConfig | CraftaxConfig | KinetixConfig],
) -> None:
    assert config_class(max_steps=2**31 - 2).max_steps == 2**31 - 2
    with pytest.raises(ValidationError, match="max_steps"):
        config_class(max_steps=2**31 - 1)
