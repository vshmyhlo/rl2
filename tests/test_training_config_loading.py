from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import yaml
from omegaconf import OmegaConf
from pydantic import ValidationError

from rl2 import (
    ppo,
    ppo_rnd,
    train_alphazero,
    train_grpo_reasoning,
    train_karel_ast_ar_edit,
    train_karel_ast_grpo,
    train_karel_grpo,
    train_mappo,
    train_wm,
)
from rl2.configuration import BaseConfig
from rl2.karel import KarelConfig
from rl2.model.encoders.rgb import ConvStage


@pytest.mark.parametrize(
    "trainer,preset,field,reference",
    [
        (ppo, "ppo/craftax/symbolic.yaml", "seed", "num_steps"),
        (train_mappo, "mappo/smax.yaml", "seed", "num_steps"),
        (train_alphazero, "alphazero_chess.yaml", "iterations", "num_envs"),
        (ppo_rnd, "ppo_rnd_montezuma.yaml", "seed", "num_steps"),
        (train_wm, "wm_atari.yaml", "total_steps", "num_steps"),
        (train_grpo_reasoning, "grpo_reasoning/gemma3.yaml", "total_updates", "seed"),
        (train_karel_grpo, "karel_grpo.yaml", "total_updates", "seed"),
        (train_karel_ast_grpo, "karel_ast_grpo.yaml", "total_updates", "seed"),
        (train_karel_ast_ar_edit, "karel_ast_ar_edit.yaml", "total_updates", "seed"),
    ],
    ids=["ppo", "mappo", "alphazero", "ppo-rnd", "world-model", "reasoning", "karel", "karel-ast", "karel-edit"],
)
def test_training_cli_resolves_yaml(
    trainer: ModuleType,
    preset: str,
    field: str,
    reference: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / preset).read_text())
    settings[field] = f"${{{reference}}}"
    settings["run_id"] = "test_seed${seed}"
    settings["ckpt_dir"] = "gs://checkpoints/${run_id}"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(settings))
    train = Mock()
    monkeypatch.setattr(trainer, "train", train)
    monkeypatch.setattr("sys.argv", ["train", "--config", str(path)])
    trainer.main()
    train.assert_called_once()
    config = train.call_args.args[0]
    assert getattr(config, field) == settings[reference]
    assert type(getattr(config, field)) is int
    assert isinstance(config, BaseConfig)
    assert config.run_id == f"test_seed{config.seed}"
    assert config.ckpt_dir == f"gs://checkpoints/test_seed{config.seed}"
    if trainer is ppo_rnd:
        assert isinstance(config.encoder_channels, tuple)
    elif trainer is train_wm:
        assert isinstance(config.encoder_stages, tuple)
        assert all(isinstance(stage, ConvStage) for stage in config.encoder_stages)
    path.write_text(yaml.safe_dump(asdict(config)))
    assert trainer.load_config(path, trainer.Config) == config
    with pytest.raises(FrozenInstanceError):
        config.seed = 2
    if trainer is not train_grpo_reasoning:
        assert hash(config) == hash(trainer.load_config(path, trainer.Config))
    with pytest.raises(ValidationError, match="unknown_option"):
        replace(config, unknown_option=1)
    path.write_text(yaml.safe_dump(asdict(config) | {"unknown_option": 1}))
    with pytest.raises(ValidationError, match="unknown_option"):
        trainer.load_config(path, trainer.Config)


@pytest.mark.parametrize(
    "trainer,retired",
    [
        (train_karel_grpo, {"kl_coef": 0.1}),
        (train_karel_ast_grpo, {"kl_coef": 0.1, "decision_batch_size": 1}),
        (train_karel_ast_ar_edit, {"replay_batch_size": 1}),
    ],
    ids=["karel", "karel-ast", "karel-edit"],
)
def test_karel_config_defaults_and_legacy_roundtrip(
    trainer: ModuleType, retired: dict[str, float | int], tmp_path: Path
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("env:\n  height: 4\n  width: ${env.height}\n")
    config = trainer.load_config(path, trainer.Config)
    assert isinstance(config.env, KarelConfig)
    assert config.env.height == config.env.width == 4
    assert config.run_id is not None
    assert config.ckpt_dir == "runs"
    path.write_text(yaml.safe_dump(asdict(config) | retired))
    assert trainer.load_config(path, trainer.Config) == config


@pytest.mark.parametrize(
    "trainer,preset,field,value",
    [
        (ppo_rnd, "ppo_rnd_montezuma.yaml", "seed", True),
        (train_wm, "wm_atari.yaml", "frame_stack", "false"),
        (train_grpo_reasoning, "grpo_reasoning/gemma3.yaml", "dtype", "float16"),
        (train_karel_grpo, "karel_grpo.yaml", "d_model", 320.5),
        (train_karel_ast_grpo, "karel_ast_grpo.yaml", "learning_rate", "0.001"),
        (train_karel_ast_ar_edit, "karel_ast_ar_edit.yaml", "total_updates", "1000"),
    ],
    ids=[
        "boolean-integer",
        "string-boolean",
        "invalid-literal",
        "fractional-integer",
        "string-float",
        "string-integer",
    ],
)
def test_training_configs_reject_invalid_types(
    trainer: ModuleType, preset: str, field: str, value: object, tmp_path: Path
) -> None:
    config = trainer.load_config(Path(__file__).resolve().parents[1] / "configs" / preset, trainer.Config)
    with pytest.raises(ValidationError, match=field):
        replace(config, **{field: value})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(asdict(config) | {field: value}))
    with pytest.raises(ValidationError, match=field):
        trainer.load_config(path, trainer.Config)


def test_reasoning_prompt_examples_are_validated_on_construction() -> None:
    config = train_grpo_reasoning.Config(
        prompt_examples=[{"question": "How many?", "completion": "<answer>4</answer>"}]
    )
    assert config.prompt_examples == [
        train_grpo_reasoning.PromptExample(question="How many?", completion="<answer>4</answer>")
    ]
    for example in (
        {"question": " ", "completion": "<answer>4</answer>"},
        {"question": "How many?", "completion": "4"},
        {"question": "How many?", "completion": "<answer>4</answer>", "unknown": 1},
    ):
        with pytest.raises(ValidationError):
            train_grpo_reasoning.Config(prompt_examples=[example])


@pytest.mark.parametrize(
    "trainer,preset,root",
    [
        (ppo_rnd, "ppo_rnd_montezuma.yaml", "/runs/ppo_rnd"),
        (train_wm, "wm_atari.yaml", "/runs/wm"),
        (train_alphazero, "alphazero_chess.yaml", "/runs/alphazero"),
        (train_karel_grpo, "karel_grpo.yaml", "/runs/karel_grpo"),
        (train_karel_ast_grpo, "karel_ast_grpo.yaml", "/runs/karel_ast_grpo"),
        (train_karel_ast_ar_edit, "karel_ast_ar_edit.yaml", "/runs/karel_ast_ar_edit"),
    ],
)
def test_preset_checkpoint_directories(trainer: ModuleType, preset: str, root: str) -> None:
    config = trainer.load_config(Path(__file__).resolve().parents[1] / "configs" / preset, trainer.Config)
    assert config.ckpt_dir == f"{root}/{config.run_id}"
    assert "log_dir" not in asdict(config)


def test_all_presets_share_run_naming_convention() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    families = (
        "grpo_reasoning",
        "karel_ast_ar_edit",
        "karel_ast_grpo",
        "karel_grpo",
        "alphazero",
        "ppo_rnd",
        "mappo",
        "ppo",
        "wm",
    )
    run_ids: set[str] = set()
    for path in sorted(root.rglob("*.yaml")):
        family = next(name for name in families if str(path.relative_to(root)).startswith(name))
        settings = OmegaConf.load(path)
        run_id = settings.run_id
        assert run_id.startswith(f"{family}_"), path
        assert run_id.endswith(f"_seed{settings.seed}"), path
        assert settings.ckpt_dir == f"/runs/{family}/{run_id}", path
        assert run_id not in run_ids, path
        BaseConfig(run_id=run_id, ckpt_dir=settings.ckpt_dir)
        run_ids.add(run_id)
        settings.seed += 1
        assert settings.run_id.endswith(f"_seed{settings.seed}"), path
        assert settings.ckpt_dir == f"/runs/{family}/{settings.run_id}", path


def test_progress_logging_presets() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    for trainer, paths in (
        (ppo, [configs / "ppo.yaml", *sorted((configs / "ppo").rglob("*.yaml"))]),
        (train_mappo, sorted((configs / "mappo").rglob("*.yaml"))),
    ):
        for path in paths:
            config = trainer.load_config(path, trainer.Config)
            rollouts = config.total_steps // (config.num_envs * config.num_steps)
            assert config.video_interval == (rollouts + 9) // 10, path
            assert config.eval_interval == (rollouts + 4) // 5, path
    config = train_wm.load_config(configs / "wm_atari.yaml", train_wm.Config)
    assert config.video_every_steps == config.total_steps // 10
