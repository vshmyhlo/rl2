"""Active Gymnax presets and explicit rejection of historical Atari configs."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rl2.ppo import load_config


@pytest.mark.parametrize("filename", ["ppo.yaml", "ppo/minatar/breakout.yaml"])
def test_native_ppo_configs(filename: str) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / filename)
    assert config.env_id == "Breakout-MinAtar"
    assert config.num_envs % config.num_minibatches == 0
    assert config.video_interval_seconds == 600
    assert config.run_id


def test_historical_atari_configs_are_unsupported() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    for path in (*configs.glob("ppo_atari*.yaml"), configs / "ppo_montezuma.yaml"):
        with pytest.raises(ValidationError, match="env_id"):
            load_config(path)
