"""Active Gymnax presets and explicit rejection of historical Atari configs."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rl2.ppo import Config, load_config, make_env


@pytest.mark.parametrize(
    "filename,env_id,video_interval",
    [
        ("ppo.yaml", "Breakout-MinAtar", 245),
        ("ppo/minatar/breakout.yaml", "Breakout-MinAtar", 123),
        ("ppo/minatar/space_invaders.yaml", "SpaceInvaders-MinAtar", 123),
    ],
)
def test_native_ppo_configs(filename: str, env_id: str, video_interval: int) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / filename, Config)
    assert config.env.id == env_id
    assert config.num_envs % config.num_minibatches == 0
    assert config.video_interval == video_interval
    assert config.run_id
    if filename.startswith("ppo/minatar/"):
        assert config.total_steps == 10_000_000
        assert (
            config.video_max_frames is None
            or config.video_max_frames >= make_env(config).params.max_steps_in_episode + 1
        )
        assert config.num_envs % (config.num_minibatches * 2) == 0
        rollout_size = config.num_envs * config.num_steps
        assert rollout_size // config.num_minibatches // 2 == 512
        assert (config.video_interval - 1) * rollout_size < 1_000_000 <= config.video_interval * rollout_size
        assert config.eval_interval == config.checkpoint_interval
        assert (config.eval_interval - 1) * rollout_size < 2_000_000 <= config.eval_interval * rollout_size


def test_historical_atari_configs_are_unsupported() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    for path in (*configs.glob("ppo_atari*.yaml"), configs / "ppo_montezuma.yaml"):
        with pytest.raises(ValidationError, match="env_id"):
            load_config(path, Config)
