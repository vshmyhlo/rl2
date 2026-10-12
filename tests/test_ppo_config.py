from pathlib import Path

import pytest
import yaml
from omegaconf.errors import OmegaConfBaseException
from pydantic import ValidationError

from rl2.ppo import Config, load_config


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "ppo.yaml"
    source = Path(__file__).resolve().parents[1] / "configs/ppo.yaml"
    path.write_text(source.read_text())
    return path


@pytest.mark.parametrize("run_id", ["explicit-run", None, "missing"])
def test_load_config_returns_fully_resolved_config(config_path: Path, run_id: str | None) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings.update(
        env_id="Breakout-MinAtar",
        seed=7,
        log_dir="logs/${run_id}/${path_name:${env_id}}",
        num_minibatches="${num_envs}",
    )
    if run_id == "missing":
        settings.pop("run_id")
    else:
        settings["run_id"] = run_id
    config_path.write_text(yaml.safe_dump(settings))

    config = load_config(config_path)

    assert isinstance(config, Config)
    assert config.run_id is not None
    if run_id == "explicit-run":
        assert config.run_id == run_id
    else:
        assert config.run_id.startswith("Breakout-MinAtar_seed7_")
    assert config.log_dir == f"logs/{config.run_id}/Breakout-MinAtar"
    assert config.num_minibatches == config.num_envs


@pytest.mark.parametrize(
    "log_dir,error,match",
    [
        ("logs/${unknown}", OmegaConfBaseException, "unknown"),
        ("${seed}", ValidationError, "log_dir"),
    ],
    ids=["invalid-interpolation", "resolved-value-fails-validation"],
)
def test_load_config_rejects_invalid_settings(
    config_path: Path, log_dir: str, error: type[Exception], match: str
) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings["log_dir"] = log_dir
    config_path.write_text(yaml.safe_dump(settings))

    with pytest.raises(error, match=match):
        load_config(config_path)


@pytest.mark.parametrize("observation_size", [0, -1])
def test_load_config_rejects_nonpositive_observation_size(config_path: Path, observation_size: int) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings["observation_size"] = observation_size
    config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match="observation_size"):
        load_config(config_path)


@pytest.mark.parametrize(
    "option,value",
    [
        ("frame_stack", True),
        ("atari_preprocessing", False),
        ("vector_env", "sync"),
        ("observation_size", None),
        ("env_id", "ALE/Breakout-v5"),
        ("eval_episodes", 0),
        ("eval_seed", -1),
        ("eval_interval_seconds", -1),
    ],
)
def test_rejects_unsupported_environment_options(config_path: Path, option: str, value: object) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings[option] = value
    config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match=option):
        load_config(config_path)
