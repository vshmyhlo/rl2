from dataclasses import replace
from types import ModuleType
from unittest.mock import Mock

import pytest

from rl2 import ppo, train_mappo


@pytest.mark.parametrize(
    ("trainer", "config_path"),
    [(ppo, "configs/ppo.yaml"), (train_mappo, "configs/mappo/jaxnav.yaml")],
    ids=["ppo", "mappo"],
)
def test_cli_shared_run_directory(trainer: ModuleType, config_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    config = trainer.load_config(config_path)
    train = Mock()
    monkeypatch.setattr(trainer, "train", train)
    monkeypatch.setattr("sys.argv", ["train", "--config", config_path, "--ckpt-dir", "/runs/test/run-1"])
    trainer.main()
    train.assert_called_once_with(replace(config, ckpt_dir="/runs/test/run-1"))
