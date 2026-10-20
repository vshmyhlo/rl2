import json
import os
import subprocess
import sys
from pathlib import Path
from threading import Event
from unittest.mock import Mock

import pytest

from rl2 import ray_timeline


def test_periodic_and_final_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger=ray_timeline.__name__)
    monkeypatch.setenv("RL2_RAY_TIMELINE_URL", "http://localhost:8085/?job=train-test")
    path = tmp_path / "profiles" / "ray.json"
    exported = Event()
    timeline = Mock(return_value=[{"name": "first"}])
    monkeypatch.setattr(ray_timeline.ray, "timeline", timeline)
    write_bytes = ray_timeline.write_bytes

    def save(destination: str, data: bytes) -> None:
        write_bytes(destination, data)
        exported.set()

    monkeypatch.setattr(ray_timeline, "write_bytes", save)
    with ray_timeline.record_timeline(str(path), interval=0.01):
        assert exported.wait(timeout=5)
        assert json.loads(path.read_text()) == [{"name": "first"}]
        timeline.return_value = [{"name": "final"}]
    assert json.loads(path.read_text()) == [{"name": "final"}]
    assert timeline.call_count >= 2
    assert list(path.parent.iterdir()) == [path]
    assert "Open Ray timeline: http://localhost:8085/?job=train-test" in caplog.text


@pytest.mark.parametrize("failure", ["query", "write"])
def test_export_failure_preserves_snapshot_and_training_error(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "ray.json"
    path.write_text('[{"name": "previous"}]')
    monkeypatch.setattr(
        ray_timeline.ray,
        "timeline",
        Mock(side_effect=RuntimeError("query failed")) if failure == "query" else Mock(return_value=[]),
    )
    if failure == "write":
        monkeypatch.setattr(ray_timeline, "write_bytes", Mock(side_effect=OSError("disk full")))
    with pytest.raises(ValueError, match="training failed"), ray_timeline.record_timeline(str(path)):
        raise ValueError("training failed")
    assert json.loads(path.read_text()) == [{"name": "previous"}]
    assert "Could not save Ray timeline" in caplog.text


def test_real_ray_actor_timeline(tmp_path: Path) -> None:
    path = tmp_path / "ray.json"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import sys
import ray
import ray._private.profiling as profiling
from rl2.ray_timeline import record_timeline

@ray.remote(runtime_env={"env_vars": {"JAX_PLATFORMS": "cpu"}})
class Probe:
    def run(self) -> str:
        with profiling.profile("probe.phase"):
            return os.environ["RAY_PROFILING"]

with (
    ray.init(
        address="local", num_cpus=1, num_gpus=0, include_dashboard=False,
        object_store_memory=80 * 1024**2,
        runtime_env={"env_vars": {"RAY_PROFILING": "1"}},
        _system_config={"enable_timeline": True, "task_events_report_interval_ms": 1000},
    ),
    record_timeline(sys.argv[1]),
):
    actor = Probe.remote()
    assert ray.get(actor.run.remote()) == "1"
""",
            str(path),
        ],
        env={**os.environ, "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0", "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = json.loads(path.read_text())
    assert any(event["name"] == "probe.phase" and event["dur"] >= 0 for event in events)
    assert any(event["name"] == "task:execute" for event in events)
