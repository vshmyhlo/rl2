import importlib
import sys
from unittest.mock import MagicMock

import jax
import pytest


@pytest.mark.parametrize("name", ["ppo", "ppo_rnd", "train_karel_ast_grpo", "train_karel_grpo", "train_wm"])
def test_training_cli_uses_environment_backend(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module(f"rl2.{name}")
    config = object()
    load_config = MagicMock(return_value=config)
    train = MagicMock()
    update_config = MagicMock()
    monkeypatch.setattr(module, "load_config", load_config)
    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(jax.config, "update", update_config)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(sys, "argv", [name, "--config", "custom.yaml"])

    module.main()

    load_config.assert_called_once_with("custom.yaml")
    train.assert_called_once_with(config)
    update_config.assert_not_called()


@pytest.mark.parametrize("name", ["ppo", "ppo_rnd", "train_karel_ast_grpo", "train_karel_grpo", "train_wm"])
def test_training_cli_rejects_platform_option(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module(f"rl2.{name}")
    train = MagicMock()
    monkeypatch.setattr(module, "train", train)
    monkeypatch.setattr(sys, "argv", [name, "--platform", "cpu"])

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2
    train.assert_not_called()
