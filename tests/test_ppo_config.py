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
        env_id="ALE/Boxing-v5",
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
        assert config.run_id.startswith("ALE_Boxing-v5_seed7_")
    assert config.log_dir == f"logs/{config.run_id}/ALE_Boxing-v5"
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
