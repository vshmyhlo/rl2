"""Shipped MAPPO presets load into supported environments with separate checkpoints."""

from pathlib import Path

from rl2.train_mappo import load_config, make_env


def test_mappo_presets() -> None:
    paths = sorted((Path(__file__).resolve().parents[1] / "configs/mappo").glob("*.yaml"))
    run_ids: set[str] = set()
    for path in paths:
        config = load_config(path)
        env = make_env(config)
        assert env.num_agents > 1, path.name
        assert config.num_envs * env.num_agents % config.num_minibatches == 0, path.name
        assert config.total_steps >= config.num_envs * config.num_steps, path.name
        assert config.run_id and config.run_id not in run_ids, path.name
        assert config.ckpt_dir == f"/runs/mappo/{config.run_id}", path.name
        run_ids.add(config.run_id)
