from inspect import isabstract
from typing import Any

import pytest

from rl2.envs import BaseMultiAgentEnv
from rl2.envs.jaxnav_env import JaxNavEnv
from rl2.envs.mpe_env import MPEEnv
from rl2.envs.smax_env import SMAXEnv


def test_base_multi_agent_env_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseMultiAgentEnv()


@pytest.mark.parametrize("missing", ["reset", "step", "memory_resets", "render"])
def test_incomplete_adapter_cannot_be_instantiated(missing: str) -> None:
    methods = {name: MPEEnv.__dict__[name] for name in ("reset", "step", "memory_resets", "render")}
    del methods[missing]
    incomplete = type("IncompleteEnv", (BaseMultiAgentEnv,), methods)
    with pytest.raises(TypeError, match=missing):
        incomplete()


@pytest.mark.parametrize("adapter", [MPEEnv, SMAXEnv, JaxNavEnv])
def test_adapters_implement_base_multi_agent_env(adapter: type[BaseMultiAgentEnv[Any]]) -> None:
    assert issubclass(adapter, BaseMultiAgentEnv)
    assert not isabstract(adapter)
