from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from rl2.alphazero.config import Config, EvaluationConfig, ModelConfig
from rl2.configuration import load_config


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"num_simulations": 0}, "num_simulations"),
        ({"checkpoint_interval": 0}, "checkpoint_interval"),
        ({"log_interval": 0}, "log_interval"),
        ({"eval_interval": 0}, "eval_interval"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"max_grad_norm": float("inf")}, "max_grad_norm"),
        ({"weight_decay": -1.0}, "weight_decay"),
        ({"dirichlet_fraction": 1.1}, "dirichlet_fraction"),
        ({"exploration_moves": -1}, "exploration_moves"),
        ({"seed": 2**32}, "seed"),
        ({"env_id": "go_9x9"}, "env_id"),
        ({"ckpt_dir": " "}, "ckpt_dir"),
        ({"run_id": " "}, "run_id"),
        ({"run_id": ".."}, "run_id"),
        ({"run_id": "a/b"}, "run_id"),
    ],
)
def test_config_validation(changes: dict[str, object], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        Config(**changes)


def test_config_zero_boundaries() -> None:
    # Accepted zero boundaries do not require initializing a model.
    Config(weight_decay=0.0, dirichlet_fraction=0.0, exploration_moves=0)


def test_config_remains_frozen_hashable_and_serializable() -> None:
    config = Config(model=ModelConfig(channels=4, num_blocks=1))
    restored = Config(**asdict(config))
    assert restored == config
    assert hash(restored) == hash(config)
    assert replace(config, iterations=1).iterations == 1
    with pytest.raises(FrozenInstanceError):
        config.iterations = 1
    with pytest.raises(FrozenInstanceError):
        config.model.channels = 8


def test_load_partial_model_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  channels: 4\n")
    config = load_config(path, Config)
    assert config.model == ModelConfig(channels=4)


def test_load_intervals(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("log_interval: 2\ncheckpoint_interval: 5\neval_interval: 15\n")
    config = load_config(path, Config)
    assert config.log_interval == 2
    assert config.checkpoint_interval == 5
    assert config.eval_interval == 15


def test_load_evaluation_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("evaluation:\n  num_openings: 3\n")
    config = load_config(path, Config)
    assert config.evaluation == EvaluationConfig(num_openings=3)


@pytest.mark.parametrize(
    "settings",
    [
        {"num_openings": 0},
        {"num_openings": True},
        {"opening_moves": -1},
        {"openings_per_batch": 0},
        {"seed": 2**32},
        {"unknown": 1},
    ],
)
def test_evaluation_config_validation(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        EvaluationConfig(**settings)


@pytest.mark.parametrize(
    "contents,match",
    [
        ("model:\n  channels: 0\n", "channels"),
        ("model:\n  num_blocks: 0\n", "num_blocks"),
        ("model:\n  channels: true\n", "channels"),
        ("model:\n  num_blocks: 1.5\n", "num_blocks"),
        ("model:\n  unknown: 1\n", "unknown"),
        ("channels: 4\n", "channels"),
    ],
)
def test_load_rejects_invalid_model_config(tmp_path: Path, contents: str, match: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(contents)
    with pytest.raises(ValidationError, match=match):
        load_config(path, Config)


def test_load_interpolated_run_paths(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "env_id: chess\n"
        "seed: 0\n"
        "run_id: alphazero_${path_name:${env_id}}_seed${seed}\n"
        "ckpt_dir: /runs/alphazero/${run_id}\n"
    )
    config = load_config(path, Config)
    assert config.env_id == "chess"
    assert config.run_id == "alphazero_chess_seed0"
    assert config.ckpt_dir == "/runs/alphazero/alphazero_chess_seed0"


@pytest.mark.parametrize(
    "contents,error,match",
    [
        ("[]", TypeError, "mapping"),
        ("num_envs: 1.5", ValueError, "num_envs"),
        ("num_envs: true", ValueError, "num_envs"),
        ("unknown_setting: 1", ValueError, "unknown_setting"),
        ("log_interval: 1.5", ValueError, "log_interval"),
        ("checkpoint_interval: true", ValueError, "checkpoint_interval"),
        ("eval_interval: -1", ValueError, "eval_interval"),
        ("log_interval_seconds: 60.0", ValueError, "log_interval_seconds"),
        ("checkpoint_interval_seconds: 600.0", ValueError, "checkpoint_interval_seconds"),
        ("evaluation:\n  interval_seconds: 1800.0", ValueError, "interval_seconds"),
    ],
)
def test_load_config_rejects_invalid_input(tmp_path: Path, contents: str, error: type[Exception], match: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(contents)
    with pytest.raises(error, match=match):
        load_config(path, Config)
