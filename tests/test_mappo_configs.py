"""MAPPO model and environment configuration validation."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError
from test_train_mappo import config as config  # noqa: PLC0414

from rl2.configuration import load_config
from rl2.train_mappo import Config, GenericModelConfig, LuxModelConfig, MLPEncoderConfig, ModelConfig


@pytest.mark.parametrize(
    "model", [None, {}, {"encoder": {"type": "mlp", "embedding_size": 4}}], ids=["omitted", "empty", "untagged"]
)
def test_default_generic_model(config: Config, model: dict[str, object] | None) -> None:
    settings = asdict(config)
    settings.pop("model")
    if model is not None:
        settings["model"] = model
    restored = Config(**settings)
    assert isinstance(restored.model, GenericModelConfig)
    assert restored.model.type == "generic"
    if model and "encoder" in model:
        assert restored.model.encoder.embedding_size == 4


@pytest.mark.parametrize(
    "encoder,field",
    [
        ({"type": "conv2d"}, "type"),
        ({"type": "mlp", "embedding_size": 0}, "embedding_size"),
        ({"type": "mlp", "embedding_size": True}, "embedding_size"),
        ({"type": "mlp", "unknown": 1}, "unknown"),
    ],
)
def test_invalid_encoder(encoder: dict[str, object], field: str) -> None:
    settings = {"encoder": encoder}
    with pytest.raises(ValidationError, match=field):
        TypeAdapter(ModelConfig).validate_python(settings)


@pytest.mark.parametrize(
    "config",
    [
        GenericModelConfig(encoder=MLPEncoderConfig(embedding_size=4)),
        LuxModelConfig(
            conv_channels=4,
            num_blocks=3,
            heads=2,
            head_dim=4,
        ),
    ],
)
def test_model_round_trip(config: GenericModelConfig | LuxModelConfig) -> None:
    adapter = TypeAdapter(ModelConfig)
    restored = adapter.validate_json(adapter.dump_json(config))
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
def test_invalid_environment(config: Config, env: dict[str, object], error_type: str) -> None:
    settings = asdict(config)
    settings["env"] = env
    with pytest.raises(ValidationError) as error:
        Config(**settings)
    assert error.value.errors()[0]["loc"][0] == "env"
    assert error.value.errors()[0]["type"] == error_type


def test_rejects_separate_environment_sections(config: Config) -> None:
    settings = asdict(config)
    env = settings.pop("env")
    settings["env_id"] = env.pop("id")
    settings["smax"] = env
    with pytest.raises(ValidationError) as error:
        Config(**settings)
    assert {item["loc"] for item in error.value.errors()} == {("env",), ("env_id",), ("smax",)}


@pytest.mark.parametrize(
    "settings", [{"type": "unknown"}, {"type": "lux", "encoder": {"type": "lux"}}, {"type": "generic", "channels": 4}]
)
def test_rejects_invalid_model_variant(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ModelConfig).validate_python(settings)


def test_rejects_separate_ship_recurrence() -> None:
    with pytest.raises(ValidationError, match="ship_recurrent"):
        TypeAdapter(ModelConfig).validate_python({"type": "lux", "ship_recurrent": {"hidden_size": 4}})


def test_lux_rejects_separate_recurrent_settings() -> None:
    with pytest.raises(ValidationError, match="recurrent"):
        TypeAdapter(ModelConfig).validate_python({"type": "lux", "recurrent": {"hidden_size": 4}})


@pytest.mark.parametrize("field", ["embedding_size", "channels"])
def test_lux_rejects_removed_width_settings(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        TypeAdapter(ModelConfig).validate_python({"type": "lux", field: 4})


@pytest.mark.parametrize("num_blocks", [0, True], ids=["empty_stack", "non_integer"])
def test_rejects_invalid_lux_block_count(num_blocks: int) -> None:
    with pytest.raises(ValidationError, match="num_blocks"):
        TypeAdapter(ModelConfig).validate_python({"type": "lux", "num_blocks": num_blocks})


@pytest.mark.parametrize("field,value", [("heads", 0), ("heads", True), ("head_dim", 0), ("head_dim", 1.5)])
def test_rejects_invalid_attention_dimensions(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match=field):
        TypeAdapter(ModelConfig).validate_python({"type": "lux", field: value})


def test_eval_reference_config_roundtrip(config: Config, tmp_path: Path) -> None:
    config = replace(config, eval_ref_policies={"baseline_s7": "/runs/mappo/old-run/checkpoints/5400"})
    adapter = TypeAdapter(Config)
    path = tmp_path / "config.yaml"
    path.write_bytes(adapter.dump_json(config))
    assert load_config(path, Config) == config
    assert hash(config)


@pytest.mark.parametrize(
    "references",
    [
        {"Bad Name": "/runs/run/checkpoints/1"},
        {"baseline": ""},
        {"baseline": "   "},
    ],
    ids=["metric-label", "empty-path", "blank-path"],
)
def test_invalid_eval_references(config: Config, references: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="eval_ref_policies"):
        replace(config, eval_ref_policies=references)
