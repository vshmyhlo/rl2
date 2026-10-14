"""Shipped MAPPO presets load into supported environments with separate checkpoints."""

from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from rl2.train_mappo import Config, MLPEncoderConfig, ModelConfig, load_config, make_env


@pytest.mark.parametrize("field", ["model", "encoder"])
def test_requires_encoder(field: str) -> None:
    settings = asdict(load_config(Path(__file__).resolve().parents[1] / "configs/mappo/smax.yaml"))
    (settings if field == "model" else settings["model"]).pop(field)
    with pytest.raises(ValidationError) as error:
        Config(**settings)
    assert error.value.errors()[0]["loc"] == (("model",) if field == "model" else ("model", "encoder"))
    assert error.value.errors()[0]["type"] == "missing"


@pytest.mark.parametrize(
    "encoder,field",
    [
        ({"type": "rgb"}, "type"),
        ({"type": "mlp", "embedding_size": 0}, "embedding_size"),
        ({"type": "mlp", "embedding_size": True}, "embedding_size"),
        ({"type": "mlp", "unknown": 1}, "unknown"),
    ],
)
def test_invalid_encoder(encoder: dict[str, object], field: str) -> None:
    settings = {"encoder": encoder}
    with pytest.raises(ValidationError, match=field):
        ModelConfig(**settings)


def test_encoder_round_trip() -> None:
    config = ModelConfig(encoder=MLPEncoderConfig(embedding_size=4))
    restored = ModelConfig(**asdict(config))
    assert restored == config and hash(restored) == hash(config)


@pytest.mark.parametrize(
    "env,error_type",
    [
        ({}, "union_tag_not_found"),
        ({"id": "unknown"}, "union_tag_invalid"),
        ({"id": "SMAX", "num_agents": 2}, "unexpected_keyword_argument"),
        ({"id": "JaxNav", "map_name": "3m"}, "unexpected_keyword_argument"),
        ({"id": "MPE_simple_spread_v3", "fill": 0.1}, "unexpected_keyword_argument"),
    ],
)
def test_invalid_environment(env: dict[str, object], error_type: str) -> None:
    settings = asdict(load_config(Path(__file__).resolve().parents[1] / "configs/mappo/smax.yaml"))
    settings["env"] = env
    with pytest.raises(ValidationError) as error:
        Config(**settings)
    assert error.value.errors()[0]["loc"][0] == "env"
    assert error.value.errors()[0]["type"] == error_type


def test_rejects_separate_environment_sections() -> None:
    settings = asdict(load_config(Path(__file__).resolve().parents[1] / "configs/mappo/smax.yaml"))
    env = settings.pop("env")
    settings["env_id"] = env.pop("id")
    settings["smax"] = env
    with pytest.raises(ValidationError) as error:
        Config(**settings)
    assert {item["loc"] for item in error.value.errors()} == {("env",), ("env_id",), ("smax",)}


def test_mappo_presets() -> None:
    paths = sorted((Path(__file__).resolve().parents[1] / "configs/mappo").rglob("*.yaml"))
    run_ids: set[str] = set()
    for path in paths:
        config = load_config(path)
        restored = Config(**asdict(config))
        assert restored == config and hash(restored) == hash(config)
        assert {"env_id", "smax", "jaxnav", "mpe"}.isdisjoint(asdict(config))
        assert config.model.encoder == MLPEncoderConfig(embedding_size=128)
        env = make_env(config)
        assert env.config == config.env
        assert env.num_agents > 1, path.name
        assert config.num_envs * env.num_agents % config.num_minibatches == 0, path.name
        assert config.total_steps >= config.num_envs * config.num_steps, path.name
        assert config.run_id and config.run_id not in run_ids, path.name
        assert config.ckpt_dir == f"/runs/mappo/{config.run_id}", path.name
        run_ids.add(config.run_id)
