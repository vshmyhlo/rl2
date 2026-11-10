from inspect import isabstract
from typing import Any

import pytest
from pydantic import ValidationError

from rl2.envs import BaseMultiAgentEnv
from rl2.envs.jaxnav_env import JaxNavConfig, JaxNavEnv
from rl2.envs.lux_env import LuxConfig, LuxEnv
from rl2.envs.mpe_env import MPEConfig, MPEEnv
from rl2.envs.robotarium_env import RobotariumConfig, RobotariumEnv
from rl2.envs.smax_env import SMAXConfig, SMAXEnv
from rl2.envs.storm_env import STORMConfig, STORMEnv

type MultiAgentConfig = JaxNavConfig | LuxConfig | MPEConfig | RobotariumConfig | SMAXConfig | STORMConfig


def test_base_multi_agent_env_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseMultiAgentEnv()


@pytest.mark.parametrize("missing", ["supports_selfplay", "reset", "step", "render"])
def test_incomplete_adapter_cannot_be_instantiated(missing: str) -> None:
    methods = {name: MPEEnv.__dict__[name] for name in ("supports_selfplay", "reset", "step", "render")}
    del methods[missing]
    incomplete = type("IncompleteEnv", (BaseMultiAgentEnv,), methods)
    with pytest.raises(TypeError, match=missing):
        incomplete()


@pytest.mark.parametrize("adapter", [MPEEnv, SMAXEnv, JaxNavEnv, RobotariumEnv, STORMEnv, LuxEnv])
def test_adapters_implement_base_multi_agent_env(adapter: type[BaseMultiAgentEnv[Any, Any]]) -> None:
    assert issubclass(adapter, BaseMultiAgentEnv)
    assert not isabstract(adapter)


@pytest.mark.parametrize(
    "adapter,config,expected",
    [
        pytest.param(MPEEnv, MPEConfig(), False, id="mpe_spread"),
        pytest.param(MPEEnv, MPEConfig(id="MPE_simple_reference_v3", num_agents=2), False, id="mpe_reference"),
        pytest.param(SMAXEnv, SMAXConfig(), False, id="smax_heuristic_enemy"),
        pytest.param(SMAXEnv, SMAXConfig(selfplay=True), True, id="smax_selfplay"),
        pytest.param(JaxNavEnv, JaxNavConfig(), False, id="jaxnav"),
        pytest.param(RobotariumEnv, RobotariumConfig(), False, id="robotarium"),
        pytest.param(STORMEnv, STORMConfig(game="coordination"), False, id="storm_coordination"),
        pytest.param(STORMEnv, STORMConfig(game="prisoners_dilemma"), True, id="storm_prisoners_dilemma"),
        pytest.param(LuxEnv, LuxConfig(), True, id="lux"),
    ],
)
def test_supports_selfplay(
    adapter: type[BaseMultiAgentEnv[Any, Any]], config: MultiAgentConfig, expected: bool
) -> None:
    env = adapter(config)
    assert env.supports_selfplay is expected


@pytest.mark.parametrize(
    "config_class,upper_bound",
    [
        (JaxNavConfig, 2**31),
        (MPEConfig, 2**31),
        (RobotariumConfig, 2**31),
        (SMAXConfig, 2**31),
        (STORMConfig, 2**31 - 1),
    ],
)
def test_horizon_fits_native_int32_counter(config_class: type[MultiAgentConfig], upper_bound: int) -> None:
    assert config_class(max_steps=upper_bound - 1).max_steps == upper_bound - 1
    with pytest.raises(ValidationError, match="max_steps"):
        config_class(max_steps=upper_bound)
