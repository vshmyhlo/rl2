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
from rl2.envs.craftax_env import CraftaxConfig
from rl2.envs.kinetix_env import KinetixConfig
from rl2.envs.xminigrid_env import XMiniGridConfig
from rl2.ppo import Config, KinetixEncoderConfig, RGBEncoderConfig, XMiniGridEncoderConfig, load_config


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "ppo.yaml"
    source = Path(__file__).resolve().parents[1] / "configs/ppo.yaml"
    path.write_text(source.read_text())
    return path


@pytest.mark.parametrize("field", ["model", "encoder"])
def test_requires_encoder(config_path: Path, field: str) -> None:
    settings = yaml.safe_load(config_path.read_text())
    (settings if field == "model" else settings["model"]).pop(field)
    config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError) as error:
        load_config(config_path)
    assert error.value.errors()[0]["loc"] == (("model",) if field == "model" else ("model", "encoder"))
    assert error.value.errors()[0]["type"] == "missing"


@pytest.mark.parametrize("run_id", ["explicit-run", None, "missing"])
def test_load_config_returns_fully_resolved_config(config_path: Path, run_id: str | None) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings.update(
        env={"id": "Breakout-MinAtar"},
        seed=7,
        ckpt_dir="logs/${run_id}/${path_name:${env.id}}",
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
    assert config.ckpt_dir == f"logs/{config.run_id}/Breakout-MinAtar"
    assert config.num_minibatches == config.num_envs


@pytest.mark.parametrize(
    "ckpt_dir,error,match",
    [
        ("logs/${unknown}", OmegaConfBaseException, "unknown"),
        ("${seed}", ValidationError, "ckpt_dir"),
    ],
    ids=["invalid-interpolation", "resolved-value-fails-validation"],
)
def test_load_config_rejects_invalid_settings(
    config_path: Path, ckpt_dir: str, error: type[Exception], match: str
) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings["ckpt_dir"] = ckpt_dir
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
    assert config.env.id == "Kinetix"
    assert config.env.action_type == "continuous"
    assert config.env.size == "l"
    assert config.model.encoder == KinetixEncoderConfig()
    assert config.env.levels == ("l/mjc_walker_easy.json",)
    assert hash(config)


def test_beta_policy_config(config_path: Path) -> None:
    assert load_config(config_path).model.continuous_policy == "truncated_gaussian"
    settings = yaml.safe_load(config_path.read_text())
    settings["model"]["continuous_policy"] = "beta"
    config_path.write_text(yaml.safe_dump(settings))
    assert load_config(config_path).model.continuous_policy == "beta"


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


@pytest.mark.parametrize("name", ["kinetix", "xminigrid", "craftax", "playground"])
def test_rejects_separate_environment_settings(config_path: Path, name: str) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings[name] = {}
    config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match=name):
        load_config(config_path)


@pytest.mark.parametrize(
    "env,match",
    [
        ({}, "union_tag_not_found"),
        ({"id": "unknown"}, "union_tag_invalid"),
        ({"id": "Breakout-MinAtar", "levels": None}, "levels"),
        ({"id": "Kinetix", "task": "PandaPickCube"}, "task"),
    ],
)
def test_rejects_invalid_environment_union(config_path: Path, env: dict[str, object], match: str) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings["env"] = env
    config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match=match):
        load_config(config_path)


@pytest.mark.parametrize("name,env_id", [("xminigrid", "XLand-MiniGrid"), ("craftax", "Craftax")])
def test_optional_environment_configs_load(name: str, env_id: str) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / f"configs/ppo/{name}.yaml")
    assert config.env.id == env_id
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
    cls = {"xminigrid": XMiniGridConfig, "craftax": CraftaxConfig}[name]
    with pytest.raises(ValidationError):
        cls(**settings)


@pytest.mark.parametrize(
    "filename,config_class,constructor_name",
    [
        ("minatar/breakout", ppo.GymnaxConfig, "GymnaxEnv"),
        ("kinetix", KinetixConfig, "KinetixEnv"),
        ("xminigrid", XMiniGridConfig, "XMiniGridEnv"),
        ("craftax", CraftaxConfig, "CraftaxEnv"),
        ("mujoco_playground/manipulation/panda_pick_cube", ppo.PlaygroundConfig, "PlaygroundEnv"),
    ],
)
def test_environment_round_trip_and_factory(
    filename: str, config_class: type, constructor_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / f"configs/ppo/{filename}.yaml")
    assert isinstance(config.env, config_class)
    restored = Config(**yaml.safe_load(yaml.safe_dump(asdict(config))))
    assert restored == config and hash(restored) == hash(config)
    constructor = Mock()
    monkeypatch.setattr(ppo, constructor_name, constructor)
    assert ppo.make_env(config) is constructor.return_value
    constructor.assert_called_once_with(config.env.id if isinstance(config.env, ppo.GymnaxConfig) else config.env)


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
        ({"continuous_policy": "unknown"}, "continuous_policy"),
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


@pytest.mark.parametrize("kind", ["rgb", "xminigrid", "kinetix"])
def test_encoder_variant_round_trip(config_path: Path, kind: str) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings["env"] = {"id": {"rgb": "Breakout-MinAtar", "xminigrid": "XLand-MiniGrid", "kinetix": "Kinetix"}[kind]}
    settings["model"]["encoder"] = {"type": kind, "embedding_size": 8}
    config_path.write_text(yaml.safe_dump(settings))
    config = load_config(config_path)
    encoder_class = {"rgb": RGBEncoderConfig, "xminigrid": XMiniGridEncoderConfig, "kinetix": KinetixEncoderConfig}[
        kind
    ]
    assert isinstance(config.model.encoder, encoder_class)
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
        ({"type": "kinetix", "hidden_size": 0}, "hidden_size"),
        ({"type": "kinetix", "embedding_size": 0}, "embedding_size"),
        ({"type": "kinetix", "hidden_size": True}, "hidden_size"),
        ({"type": "kinetix", "stages": []}, "stages"),
    ],
)
def test_invalid_encoder_variants(config_path: Path, encoder: dict[str, object], match: str) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings["env"] = {"id": "XLand-MiniGrid"}
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


def test_xminigrid_requires_matching_encoder(config_path: Path) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings["env"] = {"id": "XLand-MiniGrid"}
    settings["model"]["encoder"] = {"type": "rgb"}
    config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match="XLand-MiniGrid requires model.encoder.type: xminigrid"):
        load_config(config_path)


@pytest.mark.parametrize(
    "env_id,encoder,match",
    [("Kinetix", "rgb", "Kinetix requires"), ("Breakout-MinAtar", "kinetix", "kinetix encoder requires")],
)
def test_kinetix_requires_matching_encoder(config_path: Path, env_id: str, encoder: str, match: str) -> None:
    settings = yaml.safe_load(config_path.read_text())
    settings["env"] = {"id": env_id}
    settings["model"]["encoder"] = {"type": encoder}
    config_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValidationError, match=match):
        load_config(config_path)
