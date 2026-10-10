from pathlib import Path

import pytest
from omegaconf.errors import OmegaConfBaseException

from rl2.configuration import load_settings, resolve_settings


@pytest.mark.parametrize(
    "template,expected",
    [
        ("gs://bucket/${run_id}/${path_name:${env_id}}", "gs://bucket/run/ALE_Boxing-v5"),
        ("logs/${run_id}", "logs/run"),
        ("logs/${env_id}/${run_id}", "logs/ALE/Boxing-v5/run"),
        ("logs/", "logs/"),
    ],
    ids=["cloud-path-name", "run-only", "unmodified-env-id", "literal-path"],
)
def test_config_interpolation(template: str, expected: str) -> None:
    config = resolve_settings({"env_id": "ALE/Boxing-v5", "run_id": "run", "log_dir": template})
    assert config["log_dir"] == expected
    assert config["env_id"] == "ALE/Boxing-v5"


@pytest.mark.parametrize(
    "template",
    ["logs/${unknown}", "logs/${run_id", "${log_dir}"],
    ids=["unknown-field", "malformed", "cycle"],
)
def test_config_rejects_invalid_interpolation(template: str) -> None:
    with pytest.raises(OmegaConfBaseException):
        resolve_settings({"env_id": "chess", "run_id": "run", "log_dir": template})


def test_load_settings_resolves_defaults_and_generated_run_id(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("env_id: ALE/Boxing-v5\niterations: 3\nbatch_size: ${iterations}\n")
    config = load_settings(path, {"seed": 7, "run_id": None, "log_dir": "logs/${run_id}"})
    assert config["batch_size"] == 3
    assert isinstance(config["batch_size"], int)
    assert config["run_id"].startswith("ALE_Boxing-v5_seed7_")
    assert config["log_dir"] == f"logs/{config['run_id']}"
