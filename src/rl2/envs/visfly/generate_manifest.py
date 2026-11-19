"""Generate fixed VisFly evaluation scenarios offline, overwriting existing outputs."""

import argparse
import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

import torch
from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass

from rl2.configuration import load_config
from rl2.envs.visfly.routes import CATEGORIES
from rl2.envs.visfly.scenarios import Manifest, Scenario, SceneCoverage, select_scenarios, unique_scenes
from rl2.envs.visfly_env import VisFlyConfig, VisFlyEnv
from rl2.train_ppo_gym import Config
from rl2.utils import write_bytes


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class SceneResult:
    scenarios: tuple[Scenario, ...]
    reason: str | None


def generate_scene(config: VisFlyConfig, scene: str, seed: int, control_dt: float) -> SceneResult:
    """Generate one scene's scenarios, recording unsupported geometry as coverage."""
    settings = config.evaluation
    assert settings is not None
    try:
        with (
            closing(VisFlyEnv(config=config, num_envs=1, seed=seed, evaluation=True)) as env,
            env._use_rng(),
        ):
            env.backend.envs.sceneManager.load_scenes()
            routes = env.backend.route_catalogue(0, settings.routes)
            selected = select_scenarios(scene, routes, settings, seed, control_dt)
    except ValueError as error:
        return SceneResult(scenarios=(), reason=str(error))
    return SceneResult(scenarios=selected, reason=None)


@dataclass(frozen=True)
class SceneTask:
    config: VisFlyConfig
    scene: str
    seed: int
    control_dt: float


def initialize_worker() -> None:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    torch.set_num_threads(1)


def generate_and_save_scene(task: SceneTask) -> SceneResult:
    settings = task.config.evaluation
    assert settings is not None
    identity = {
        "config": asdict(task.config),
        "seed": task.seed,
        "control_dt": task.control_dt,
        "scene": json.loads(Path(task.config.kwargs["scene_path"]).read_bytes()),
    }
    digest = sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    cache_path = f"{settings.manifest_path}.scenes/{digest}.json"
    result = generate_scene(task.config, task.scene, task.seed, task.control_dt)
    write_bytes(cache_path, TypeAdapter(SceneResult).dump_json(result))
    print(f"Generated: {task.scene}", flush=True)
    return result


def generate_manifest(config: VisFlyConfig, seed: int, *, workers: int = 4) -> Manifest:
    """Regenerate all scenes; overwrite the manifest only after the inventory finishes."""
    if workers < 1:
        raise ValueError("workers must be greater than 0")
    settings = config.evaluation
    if settings is None:
        raise ValueError("Offline manifest generation requires env.evaluation")
    generation_config = replace(
        config,
        route_sampling=None,
        scene_rotation_env_steps=None,
        kwargs={**config.kwargs, "scene_split": "test", "scene_randomization": True},
    )
    with closing(VisFlyEnv(config=generation_config, num_envs=1, seed=seed, evaluation=True)) as env:
        manager = env.backend.envs.sceneManager
        root = Path(manager.root_path)
        inventory = unique_scenes(manager._scene_loader.paths, root)
        control_dt = float(env.backend.envs.dynamics.ctrl_dt)
    tasks = [
        SceneTask(
            config=replace(generation_config, kwargs={**generation_config.kwargs, "scene_path": str(root / scene)}),
            scene=scene,
            seed=(seed + index * len(CATEGORIES)) % 2**32,
            control_dt=control_dt,
        )
        for index, (scene, _) in enumerate(inventory)
    ]
    if workers == 1 or len(tasks) < 2:
        results = {index: generate_and_save_scene(task) for index, task in enumerate(tasks)}
    else:
        results = {}
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
        ) as pool:
            pending = {pool.submit(generate_and_save_scene, task): index for index, task in enumerate(tasks)}
            try:
                for future in as_completed(pending):
                    results[pending[future]] = future.result()
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
    scenarios = []
    coverage = []
    for index, (scene, aliases) in enumerate(inventory):
        result = results[index]
        missing = tuple(category for category in CATEGORIES if category not in {s.category for s in result.scenarios})
        if missing:
            logging.getLogger(__name__).warning(
                "VisFly evaluation %s: unavailable %s; %s",
                scene,
                missing,
                result.reason or "no qualifying route in the sampled roadmap",
            )
        coverage.append(SceneCoverage(scene=scene, aliases=aliases, unavailable=missing, reason=result.reason))
        scenarios.extend(result.scenarios)
    if not scenarios:
        raise ValueError(f"No evaluation routes found: {coverage}")
    manifest = Manifest(
        scenarios=tuple(scenarios), coverage=tuple(coverage), seed=seed, generation=settings, control_dt=control_dt
    )
    write_bytes(settings.manifest_path, TypeAdapter(Manifest).dump_json(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo_gym/visfly.yaml", help="Path to the training YAML config")
    parser.add_argument("--workers", type=int, default=4, help="Parallel scene processes (default: 4; 1 runs serially)")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be greater than 0")
    config = load_config(args.config, Config)
    if not isinstance(config.env, VisFlyConfig):
        parser.error("Offline manifest generation requires a VisFly environment")
    manifest = generate_manifest(config.env, config.eval_seed, workers=args.workers)
    print(f"Manifest ready: {manifest.generation.manifest_path} ({len(manifest.scenarios)} scenarios)", flush=True)


if __name__ == "__main__":
    main()
