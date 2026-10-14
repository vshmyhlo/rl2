from types import ModuleType
from unittest.mock import Mock

import pytest

from rl2 import ppo, train_mappo


@pytest.mark.parametrize(
    ("trainer", "config_path"),
    [
        (ppo, "configs/ppo/craftax/craftax.yaml"),
        (ppo, "configs/ppo/kinetix.yaml"),
        (ppo, "configs/ppo/xminigrid.yaml"),
        (ppo, "configs/ppo/minatar/breakout.yaml"),
        (train_mappo, "configs/mappo/jaxnav.yaml"),
        (train_mappo, "configs/mappo/smax.yaml"),
    ],
    ids=["craftax", "kinetix", "xminigrid", "breakout", "jaxnav", "smax"],
)
def test_cli_shared_run_directory(trainer: ModuleType, config_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    config = trainer.load_config(config_path)
    family = "ppo" if trainer is ppo else "mappo"
    assert config.ckpt_dir == f"/runs/{family}/{config.run_id}"
    assert "${" not in config.ckpt_dir
    train = Mock()
    monkeypatch.setattr(trainer, "train", train)
    monkeypatch.setattr("sys.argv", ["train", "--config", config_path])
    trainer.main()
    train.assert_called_once_with(config)


@pytest.mark.parametrize("trainer", [ppo, train_mappo], ids=["ppo", "mappo"])
def test_checkpoint_cli_override_removed(trainer: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["train", "--ckpt-dir", "/runs/override"])
    with pytest.raises(SystemExit) as error:
        trainer.main()
    assert error.value.code == 2
