"""Keep the Atari configs' schedules and backbone comparison budgets consistent."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from rl2.ppo import GDN2Config, ModelType, load_config, make_model
from rl2.ppo_rnd import load_config as load_rnd_config
from rl2.shape_checker import ShapeChecker


def test_ppo_configs_specify_distinct_run_ids() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    run_ids: set[str] = set()
    for path in configs.glob("ppo*.yaml"):
        loader = load_rnd_config if path.stem.startswith("ppo_rnd") else load_config
        config = loader(path)
        assert config.run_id == path.stem
        assert config.run_id not in run_ids
        run_ids.add(config.run_id)


def test_atari_model_configs_enable_cosine_decay() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    for path in configs.glob("ppo_atari_*.yaml"):
        config = load_config(path)
        assert config.anneal_lr, path.name
        assert config.lr_decay == "cosine", path.name
        assert config.entropy_decay == "cosine", path.name


def test_atari_game_configs_match_base_settings() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    base = load_config(configs / "ppo_atari.yaml")
    games = {
        "pong": "Pong",
        "breakout": "Breakout",
        "space-invaders": "SpaceInvaders",
        "enduro": "Enduro",
        "qbert": "Qbert",
    }
    run_ids: set[str] = set()
    for filename, game in games.items():
        config = load_config(configs / "ppo" / "atari" / f"{filename}.yaml")
        assert config.env_id == f"ALE/{game}-v5"
        assert config.run_id == f"ppo_atari_ALE_{game}-v5"
        assert config.run_id not in run_ids
        run_ids.add(config.run_id)
        assert config.log_dir == f"gs://cohere-dev/vlad/rl2/{config.run_id}"
        assert replace(config, env_id=base.env_id, run_id=base.run_id, log_dir=base.log_dir) == base


def test_base_atari_config_paper_hyperparameters() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(configs / "ppo_atari.yaml")
    assert config.total_steps == 10_000_000
    assert config.num_envs == 8
    assert config.num_steps == 128
    assert config.num_envs % config.num_minibatches == 0
    assert config.num_envs * config.num_steps // config.num_minibatches == 256
    assert config.update_epochs == 3
    assert config.learning_rate == 0.00025
    assert config.anneal_lr and config.lr_decay == "linear"
    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert config.clip_coef == 0.1  # Linear clip annealing is not supported by the trainer.
    assert config.target_kl is None
    assert config.entropy_coef == 0.01 and config.entropy_decay == "constant"
    assert 0.5 * config.value_coef == 1.0  # Convert the trainer's half-MSE to paper Eq. 9.
    assert config.frame_stack and config.atari_preprocessing
    assert config.observation_size is None
    assert not config.bf16


def test_atari_model_configs_specify_experiment_settings() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    for model_type in ("lstm", "gdn2", "mamba3"):
        config = load_config(configs / f"ppo_atari_{model_type}.yaml")
        assert config.env_id == "ALE/Breakout-v5"
        assert config.total_steps == 10_000_000
        assert config.log_dir == f"gs://cohere-dev/vlad/rl2/ppo/{config.run_id}/ALE_Breakout-v5"


@pytest.mark.parametrize("model_type", ("lstm", "gdn2", "mamba3"))
def test_atari_config_parameter_budget(model_type: ModelType) -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    base = load_config(configs / "ppo_atari_lstm.yaml")
    config = load_config(configs / f"ppo_atari_{model_type}.yaml")
    assert config.model.type == model_type
    assert config.model.num_layers == 2
    assert (
        replace(
            config,
            env_id=base.env_id,
            model=base.model,
            encoder_stages=base.encoder_stages,
            log_dir=base.log_dir,
            run_id=base.run_id,
        )
        == base
    )

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


def test_base_atari_config_parameter_budget() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs"
    config = load_config(configs / "ppo_atari.yaml")
    assert config.model.type == "lstm"
    assert config.model.hidden_size == 512
    assert config.model.num_layers == 1
    assert config.model.intermediate_size == 0
    assert [(s.channels, s.kernel_size) for s in config.encoder_stages] == [(32, 8), (64, 4), (64, 3)]
    assert all(s.blocks == 0 for s in config.encoder_stages)
    height, width = 84, 84
    for stage, expected in zip(config.encoder_stages, (21, 11, 11), strict=True):
        height, width = stage.output_shape(height, width)
        assert (height, width) == (expected, expected)
    model = make_model(config, num_actions=6)
    (_, logits, values), variables = jax.eval_shape(
        model.init_with_output,
        jax.random.key(0),
        jax.ShapeDtypeStruct((1, 1, 4, 84, 84), jnp.uint8),
        model.initial_carry(1),
        jax.ShapeDtypeStruct((1, 1), jnp.bool_),
    )
    sc = ShapeChecker(T=1, B=1, A=6)
    sc.check(logits, "TBA", jnp.float32)
    sc.check(values, "TB", jnp.float32)
    assert variables["params"]["encoder"]["Dense_0"]["kernel"].shape == (11 * 11 * 64, 768)
    count = sum(parameter.size for parameter in jax.tree.leaves(variables["params"]))
    assert 9_000_000 <= count <= 9_200_000
