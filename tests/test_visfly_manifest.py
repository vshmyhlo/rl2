"""Offline manifest persistence, interrupted generation, and read-only evaluation."""

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import gymnasium as gym
import pytest
from pydantic import ValidationError

from rl2.envs.visfly import generate_manifest as generation
from rl2.envs.visfly.routes import Route
from rl2.envs.visfly.scenarios import EvaluationConfig
from rl2.envs.visfly_env import VisFlyConfig, evaluation_manifest


@pytest.fixture
def config(tmp_path: Path) -> VisFlyConfig:
    root = tmp_path / "test"
    root.mkdir()
    for name, layout in (("a", 1), ("alias_a", 1), ("b", 2), ("bad", 3)):
        (root / f"{name}.json").write_text(f'{{"layout": {layout}}}')
    return VisFlyConfig(
        kwargs={"scene_kwargs": {"path": str(root)}},
        evaluation=EvaluationConfig(manifest_path=str(tmp_path / "manifest.json")),
    )


@pytest.fixture
def backends(config: VisFlyConfig, monkeypatch: pytest.MonkeyPatch) -> list[Mock]:
    root = Path(config.kwargs["scene_kwargs"]["path"])
    created = []
    route = Route(
        points=((1.0, 1.0, 1.0), (8.0, 8.0, 1.0)),
        direct_blocked=False,
        constrained=False,
        long=True,
        detour_threshold=1.25,
    )

    def create(**kwargs: Any) -> Mock:
        assert kwargs["scene_split"] == "test"
        backend = Mock(num_envs=1, action_space=gym.spaces.Box(low=-1, high=1, shape=(4,)))
        backend.envs = SimpleNamespace(
            dynamics=SimpleNamespace(ctrl_dt=0.03),
            sceneManager=SimpleNamespace(
                root_path=str(root),
                _scene_loader=SimpleNamespace(paths=[str(path) for path in root.glob("*.json")]),
                load_scenes=Mock(),
            ),
        )
        backend.route_catalogue.return_value = [route]
        if kwargs.get("scene_path", "").endswith("bad.json"):
            backend.route_catalogue.side_effect = ValueError("no sampling volume")
        created.append(backend)
        return backend

    monkeypatch.setattr("rl2.envs.visfly_env.load_env_creator", Mock(return_value=create))
    return created


def test_generate_unique_scenes_and_overwrite_manifest(config: VisFlyConfig, backends: list[Mock]) -> None:
    manifest = generation.generate_manifest(config, seed=3, workers=1)
    assert len(backends) == 4  # Inventory plus three unique layouts.
    assert all(backend.close.call_count == 1 for backend in backends)
    assert [item.scene for item in manifest.scenarios] == ["a.json", "b.json"]
    assert manifest.control_dt == 0.03
    assert manifest.coverage[0].aliases == ("alias_a.json",)
    assert manifest.coverage[-1].reason == "no sampling volume"
    assert len(manifest.coverage[-1].unavailable) == 4
    changed = replace(config, evaluation=replace(manifest.generation, reference_speed=5))
    with patch("rl2.envs.visfly_env.load_env_creator", side_effect=AssertionError("must reuse manifest")):
        assert evaluation_manifest(changed) == manifest
    regenerated = generation.generate_manifest(changed, seed=99, workers=1)
    assert regenerated.seed == 99
    assert regenerated.generation.reference_speed == 5
    assert evaluation_manifest(changed) == regenerated
    assert len(backends) == 8


def test_interrupted_generation_restarts_completed_scenes(
    config: VisFlyConfig, backends: list[Mock], tmp_path: Path
) -> None:
    original = generation.generate_scene

    def interrupted(config: VisFlyConfig, scene: str, seed: int, control_dt: float) -> generation.SceneResult:
        if scene == "b.json":
            raise RuntimeError("interrupted")
        return original(config, scene, seed, control_dt)

    with (
        patch.object(generation, "generate_scene", side_effect=interrupted),
        pytest.raises(RuntimeError, match="interrupted"),
    ):
        generation.generate_manifest(config, seed=3, workers=1)
    assert not (tmp_path / "manifest.json").exists()
    assert len(list((tmp_path / "manifest.json.scenes").glob("*.json"))) == 1
    with patch.object(generation, "generate_scene", wraps=original) as resumed:
        manifest = generation.generate_manifest(config, seed=3, workers=1)
    assert [call.args[1] for call in resumed.call_args_list] == ["a.json", "b.json", "bad.json"]
    assert [scenario.seed for scenario in manifest.scenarios] == [3, 7]
    assert all(backend.close.call_count == 1 for backend in backends)
    assert evaluation_manifest(config) == manifest


@pytest.mark.parametrize("invalid", [False, True], ids=["valid_outputs", "invalid_outputs"])
def test_generation_overwrites_existing_outputs_with_unchanged_inputs(
    config: VisFlyConfig, backends: list[Mock], tmp_path: Path, invalid: bool
) -> None:
    manifest = generation.generate_manifest(config, seed=3, workers=1)
    paths = [tmp_path / "manifest.json", *sorted((tmp_path / "manifest.json.scenes").glob("*.json"))]
    contents = {path: path.read_bytes() for path in paths}
    if invalid:
        for path in paths:
            path.write_bytes(b"invalid JSON")
    with patch.object(generation, "generate_scene", wraps=generation.generate_scene) as regenerated:
        assert generation.generate_manifest(config, seed=3, workers=1) == manifest
    assert [call.args[1] for call in regenerated.call_args_list] == ["a.json", "b.json", "bad.json"]
    assert {path: path.read_bytes() for path in paths} == contents


def test_failed_regeneration_preserves_existing_manifest(
    config: VisFlyConfig, backends: list[Mock], tmp_path: Path
) -> None:
    generation.generate_manifest(config, seed=3, workers=1)
    path = tmp_path / "manifest.json"
    previous = path.read_bytes()
    with (
        patch.object(generation, "generate_scene", side_effect=RuntimeError("interrupted")),
        pytest.raises(RuntimeError, match="interrupted"),
    ):
        generation.generate_manifest(config, seed=4, workers=1)
    assert path.read_bytes() == previous


@pytest.mark.parametrize(
    ("content", "error"),
    [(None, FileNotFoundError), (b"{}", ValidationError)],
    ids=["missing", "invalid"],
)
def test_evaluation_requires_existing_valid_manifest(
    config: VisFlyConfig, tmp_path: Path, content: bytes | None, error: type[Exception]
) -> None:
    if content is not None:
        (tmp_path / "manifest.json").write_bytes(content)
    with (
        patch("rl2.envs.visfly_env.load_env_creator", side_effect=AssertionError("evaluation must not generate")),
        pytest.raises(error) as raised,
    ):
        evaluation_manifest(config)
    if content is None:
        assert "rl2.envs.visfly.generate_manifest" in str(raised.value)
        assert not (tmp_path / "manifest.json.scenes").exists()


def test_generation_without_routes_keeps_coverage_but_does_not_publish(
    config: VisFlyConfig, backends: list[Mock], tmp_path: Path
) -> None:
    result = generation.SceneResult(scenarios=(), reason="no sampling volume")
    with (
        patch.object(generation, "generate_scene", return_value=result),
        pytest.raises(ValueError, match="No evaluation routes"),
    ):
        generation.generate_manifest(config, seed=3, workers=1)
    assert not (tmp_path / "manifest.json").exists()
    assert len(list((tmp_path / "manifest.json.scenes").glob("*.json"))) == 3


def test_generation_requires_evaluation_settings() -> None:
    with pytest.raises(ValueError, match="requires env.evaluation"):
        generation.generate_manifest(VisFlyConfig(), seed=3, workers=1)


@pytest.mark.parametrize("fail", [False, True], ids=["ordered_results", "worker_failure"])
def test_parallel_generation_retains_scene_order_and_completed_work(
    config: VisFlyConfig, backends: list[Mock], tmp_path: Path, fail: bool
) -> None:
    def submit(
        function: Callable[[generation.SceneTask], generation.SceneResult], task: generation.SceneTask
    ) -> Future[generation.SceneResult]:
        future: Future[generation.SceneResult] = Future()
        if fail and task.scene == "b.json":
            future.set_exception(RuntimeError("worker failed"))
        else:
            future.set_result(function(task))
        return future

    def reverse_completion(
        futures: Iterable[Future[generation.SceneResult]],
    ) -> Iterator[Future[generation.SceneResult]]:
        return reversed(list(futures))

    with (
        patch.object(generation, "ProcessPoolExecutor") as executor,
        patch.object(generation, "as_completed", side_effect=reverse_completion),
    ):
        executor.return_value.__enter__.return_value.submit.side_effect = submit
        if fail:
            with pytest.raises(RuntimeError, match="worker failed"):
                generation.generate_manifest(config, seed=3, workers=2)
            assert not (tmp_path / "manifest.json").exists()
            assert len(list((tmp_path / "manifest.json.scenes").glob("*.json"))) == 2
        else:
            manifest = generation.generate_manifest(config, seed=3, workers=2)
            assert [item.scene for item in manifest.coverage] == ["a.json", "b.json", "bad.json"]
            assert [item.seed for item in manifest.scenarios] == [3, 7]
        assert executor.call_args.kwargs["max_workers"] == 2
        assert executor.call_args.kwargs["mp_context"].get_start_method() == "spawn"


def test_generation_rejects_zero_workers(config: VisFlyConfig) -> None:
    with pytest.raises(ValueError, match="workers must be greater than 0"):
        generation.generate_manifest(config, seed=3, workers=0)
