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


@pytest.mark.parametrize("env_yaml", ["env_id: ALE/Boxing-v5", "env:\n  id: ALE/Boxing-v5"], ids=["flat", "tagged"])
def test_load_settings_resolves_defaults_and_generated_run_id(tmp_path: Path, env_yaml: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(f"{env_yaml}\niterations: 3\nbatch_size: ${{iterations}}\n")
    config = load_settings(path, {"seed": 7, "run_id": None, "log_dir": "logs/${run_id}"})
    assert config["batch_size"] == 3
    assert isinstance(config["batch_size"], int)
    assert config["run_id"].startswith("ALE_Boxing-v5_seed7_")
    assert config["log_dir"] == f"logs/{config['run_id']}"


def test_progress_logging_presets() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    paths = [
        configs / "ppo.yaml",
        *sorted((configs / "ppo").rglob("*.yaml")),
        *sorted((configs / "mappo").rglob("*.yaml")),
    ]
    for path in paths:
        config = load_settings(path)
        rollouts = config["total_steps"] // (config["num_envs"] * config["num_steps"])
        assert config["video_interval"] == (rollouts + 9) // 10, path
        assert config["eval_interval"] == (rollouts + 4) // 5, path

    config = load_settings(configs / "wm_atari.yaml")
    assert config["video_every_steps"] == config["total_steps"] // 10
