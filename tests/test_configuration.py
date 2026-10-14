from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf.errors import OmegaConfBaseException

from rl2 import configuration
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


def test_load_remote_settings_with_custom_run_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    read = Mock(return_value=b"run_id: null\nckpt_dir: gs://bucket/${run_id}\n")
    monkeypatch.setattr(configuration, "read_bytes", read)
    config = load_settings("gs://bucket/config.yaml", run_id_prefix="grpo_reasoning")
    read.assert_called_once_with("gs://bucket/config.yaml")
    assert config["run_id"].startswith("grpo_reasoning_")
    assert config["ckpt_dir"] == f"gs://bucket/{config['run_id']}"


@pytest.mark.parametrize("settings", [{"env_id": "test"}, {"env": {}, "run_id": None}])
def test_resolution_without_generated_run_id(settings: dict[str, object]) -> None:
    assert resolve_settings(settings, generate_run_id=False) == settings


def test_load_settings_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- seed: 1\n")
    with pytest.raises(TypeError, match="mapping"):
        load_settings(path)


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
