"""Keep the Atari backbone comparison configs near the same total model budget."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from rl2.ppo import ModelType, load_config, make_model


@pytest.mark.parametrize("model_type", ("lstm", "gdn2", "mamba3"))
def test_atari_config_parameter_budget(model_type: ModelType) -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    base = load_config(configs / "ppo_atari.yaml")
    config = load_config(configs / f"ppo_atari_{model_type}.yaml")
    assert config.model.type == model_type
    assert config.log_dir == f"{base.log_dir}/ppo"
    assert replace(config, model=base.model, log_dir=base.log_dir) == base

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
