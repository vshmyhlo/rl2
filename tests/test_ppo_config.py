import builtins
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import pytest
import yaml
from omegaconf.errors import OmegaConfBaseException
from pydantic import ValidationError

from rl2 import ppo
from rl2.envs.kinetix_env import KinetixConfig
from rl2.ppo import Config, RGBEncoderConfig, XMiniGridEncoderConfig, load_config


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
        ("eval_interval", -1),
        ("eval_interval", 1.5),
        ("eval_interval", True),
    ],
)
def test_rejects_unsupported_environment_options(config_path: Path, option: str, value: object) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings[option] = value
    config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match=option):
        load_config(config_path)


def test_kinetix_config_loads_yaml_levels_as_hashable_tuple() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/ppo/kinetix.yaml")
    assert config.env_id == "Kinetix"
    assert config.kinetix.action_type == "discrete"
    assert config.kinetix.levels == ("s/h4_thrust_aim.json",)
    assert hash(config)


@pytest.mark.parametrize(
    "settings",
    [
        {"size": "xl"},
        {"action_type": "multi_discrete"},
        {"levels": []},
        {"levels": [""]},
        {"max_steps": 0},
        {"frame_skip": 0},
        {"downscale": 3},
        {"dense_reward_scale": float("nan")},
    ],
)
def test_kinetix_rejects_invalid_settings(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        KinetixConfig(**settings)


def test_kinetix_settings_require_kinetix_environment(config_path: Path) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings["kinetix"] = {"size": "m"}
    config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match="kinetix settings require"):
        load_config(config_path)


@pytest.mark.parametrize("name,env_id", [("xminigrid", "XLand-MiniGrid"), ("craftax", "Craftax")])
def test_optional_environment_configs_load(name: str, env_id: str) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / f"configs/ppo/{name}.yaml")
    assert config.env_id == env_id
    assert hash(config)


@pytest.mark.parametrize(
    "name,settings",
    [
        ("xminigrid", {"env_id": "unknown"}),
        ("xminigrid", {"env_id": "XLand-MiniGrid-R1-9x9"}),
        ("xminigrid", {"benchmark": "trivial-21k"}),
        ("xminigrid", {"ruleset_id": 0}),
        ("xminigrid", {"ruleset_id": -1}),
        ("xminigrid", {"max_steps": 0}),
        ("craftax", {"env_id": "Craftax-Symbolic-v1"}),
        ("craftax", {"max_steps": 0}),
    ],
    ids=[
        "unknown-grid",
        "missing-benchmark",
        "minigrid-benchmark",
        "missing-ruleset-benchmark",
        "negative-id",
        "grid-horizon",
        "symbolic-craftax",
        "craftax-horizon",
    ],
)
def test_optional_environments_reject_invalid_settings(name: str, settings: dict[str, object]) -> None:
    from rl2.envs.craftax_env import CraftaxConfig
    from rl2.envs.xminigrid_env import XMiniGridConfig

    cls = {"xminigrid": XMiniGridConfig, "craftax": CraftaxConfig}[name]
    with pytest.raises(ValidationError):
        cls(**settings)


@pytest.mark.parametrize("name", ["xminigrid", "craftax"])
def test_optional_settings_require_matching_environment(config_path: Path, name: str) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings[name] = {"max_steps": 1}
    config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match=f"{name} settings require"):
        load_config(config_path)


@pytest.mark.parametrize("name,env_id", [("xminigrid", "XLand-MiniGrid"), ("craftax", "Craftax")])
def test_optional_environment_factory(name: str, env_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / f"configs/ppo/{name}.yaml")
    assert config.env_id == env_id
    cls = ppo.XMiniGridEnv if name == "xminigrid" else ppo.CraftaxEnv
    constructor = Mock()
    monkeypatch.setattr(ppo, cls.__name__, constructor)
    assert ppo.make_env(config) is constructor.return_value
    constructor.assert_called_once_with(getattr(config, name))


def test_craftax_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs/ppo/craftax.yaml")
    original_import = builtins.__import__

    def without_dependency(module: str, *args: Any, **kwargs: Any) -> ModuleType:
        if module.split(".")[0] == "craftax":
            raise ModuleNotFoundError(name="craftax")
        return original_import(module, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_dependency)
    with pytest.raises(ImportError, match="uv sync --extra craftax"):
        ppo.make_env(config)


@pytest.mark.parametrize(
    "model,match",
    [
        ({"encoder": {"type": "rgb", "stages": []}}, "stages"),
        ({"encoder": {"type": "rgb", "stages": [{"channels": 0}]}}, "stages"),
        ({"encoder": {"type": "rgb", "embedding_size": 0}}, "embedding_size"),
        ({"encoder": {"type": "rgb", "embedding_size": True}}, "embedding_size"),
        ({"encoder": {"type": "rgb", "unknown": 1}}, "unknown"),
        ({"hidden_size": 8}, "hidden_size"),
        ({"encoder_stages": []}, "encoder_stages"),
    ],
)
def test_rejects_invalid_nested_model(config_path: Path, model: dict[str, object], match: str) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings["model"] = model
    config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match=match):
        load_config(config_path)


def test_rejects_top_level_encoder_stages(config_path: Path) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings["encoder_stages"] = [{"channels": 4}]
    config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match="encoder_stages"):
        load_config(config_path)


@pytest.mark.parametrize("kind", ["rgb", "xminigrid"])
def test_encoder_variant_round_trip(config_path: Path, kind: str) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings["env_id"] = "XLand-MiniGrid" if kind == "xminigrid" else "Breakout-MinAtar"
    settings["model"]["encoder"] = {"type": kind, "embedding_size": 8}
    config_path.write_text(yaml.safe_dump(settings))
    config = load_config(config_path)
    assert isinstance(config.model.encoder, RGBEncoderConfig if kind == "rgb" else XMiniGridEncoderConfig)
    assert config.model.encoder.embedding_size == 8
    config_path.write_text(yaml.safe_dump(asdict(config)))
    restored = load_config(config_path)
    assert restored == config and hash(restored) == hash(config)


@pytest.mark.parametrize(
    "encoder,match",
    [
        ({}, "union_tag_not_found"),
        ({"type": "unknown"}, "union_tag_invalid"),
        ({"type": "rgb", "token_size": 4}, "token_size"),
        ({"type": "xminigrid", "stages": []}, "stages"),
        ({"type": "xminigrid", "token_size": 0}, "token_size"),
        ({"type": "xminigrid", "channels": True}, "channels"),
        ({"type": "xminigrid", "blocks": -1}, "blocks"),
        ({"type": "xminigrid", "embedding_size": 0}, "embedding_size"),
    ],
)
def test_invalid_encoder_variants(config_path: Path, encoder: dict[str, object], match: str) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings["env_id"] = "XLand-MiniGrid"
    settings["model"]["encoder"] = encoder
    config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match=match):
        load_config(config_path)


def test_symbolic_encoder_requires_xminigrid(config_path: Path) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings["model"]["encoder"] = {"type": "xminigrid"}
    config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match="xminigrid encoder requires"):
        load_config(config_path)


@pytest.mark.parametrize("explicit_rgb", [True, False], ids=["explicit-rgb", "default-rgb"])
def test_xminigrid_requires_matching_encoder(config_path: Path, explicit_rgb: bool) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings["env_id"] = "XLand-MiniGrid"
    if explicit_rgb:
        settings["model"]["encoder"] = {"type": "rgb"}
    else:
        settings["model"].pop("encoder")
    config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match="XLand-MiniGrid requires model.encoder.type: xminigrid"):
        load_config(config_path)
