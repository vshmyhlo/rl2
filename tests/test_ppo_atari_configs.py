"""Keep the Atari configs' schedules and backbone comparison budgets consistent."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from rl2.ppo import GDN2Config, ModelType, load_config, make_model
from rl2.ppo_rnd import load_config as load_rnd_config


def test_ppo_configs_specify_distinct_run_ids() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    run_ids: set[str] = set()
    for path in configs.glob("ppo*.yaml"):
        loader = load_rnd_config if path.stem.startswith("ppo_rnd") else load_config
        config = loader(path)
        assert config.run_id == path.stem
        assert config.run_id not in run_ids
        run_ids.add(config.run_id)


def test_atari_configs_enable_cosine_decay() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    for path in configs.glob("ppo_atari*.yaml"):
        config = load_config(path)
        assert config.anneal_lr, path.name
        assert config.lr_decay == "cosine", path.name
        assert config.entropy_decay == "cosine", path.name


@pytest.mark.parametrize("model_type", ("lstm", "gdn2", "mamba3"))
def test_atari_config_parameter_budget(model_type: ModelType) -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    base = load_config(configs / "ppo_atari.yaml")
    config = load_config(configs / f"ppo_atari_{model_type}.yaml")
    assert config.model.type == model_type
    assert config.model.num_layers == 2
    assert config.log_dir == f"{base.log_dir}/ppo"
    assert replace(config, model=base.model, log_dir=base.log_dir, run_id=base.run_id) == base

    if isinstance(config.model, GDN2Config):
        assert config.model.backend == "triton"
        # Parameter shapes are backend-independent; keep this budget check CPU-only.
        config = replace(config, model=replace(config.model, backend="jax"))

    # Trace parameter shapes without allocating weights or compiling a training run.
    # Space Invaders uses six actions and four stacked 84x84 grayscale frames.
    model = make_model(config, num_actions=6)
    variables = jax.eval_shape(
        model.init,
        jax.random.key(0),
        jax.ShapeDtypeStruct((1, 1, 4, 84, 84), jnp.uint8),
        model.initial_carry(1),
        jax.ShapeDtypeStruct((1, 1), jnp.bool_),
    )
    count = sum(parameter.size for parameter in jax.tree.leaves(variables["params"]))
    assert 9_500_000 <= count <= 10_500_000
