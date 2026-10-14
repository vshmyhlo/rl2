import json
import os
import runpy
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

submit = SimpleNamespace(**runpy.run_path(str(Path(__file__).resolve().parents[1] / "submit.py")))


def test_default_gpu_job() -> None:
    args = submit.parse_args(["--", "uv", "run", "python", "train.py"])
    job = submit.job_manifest(args=args, name="train-test", image="rl2.local/train:test")
    assert job["metadata"]["labels"] == {
        "kueue.x-k8s.io/queue-name": "gpu",
        "kueue.x-k8s.io/priority-class": "train-low",
    }
    assert job["spec"]["suspend"] is True
    assert job["spec"]["backoffLimit"] == 0
    pod = job["spec"]["template"]["spec"]
    assert pod["runtimeClassName"] == "nvidia"
    container = pod["containers"][0]
    assert container["imagePullPolicy"] == "Never"
    assert container["workingDir"] == "/app"
    assert container["resources"]["requests"] == {"cpu": "4", "memory": "16Gi", "nvidia.com/gpu": "1"}
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    run_mount = next(mount for mount in container["volumeMounts"] if mount["mountPath"] == "/runs")
    run_volume = next(volume for volume in pod["volumes"] if volume["name"] == run_mount["name"])
    assert run_volume["persistentVolumeClaim"]["claimName"] == "training-runs"


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([], id="missing-separator"),
        pytest.param(["--"], id="empty-command"),
        pytest.param(["--prio", "urgent", "--", "true"], id="unknown-priority"),
        pytest.param(["--cpus", "0", "--", "true"], id="zero-cpus"),
        pytest.param(["--cpus", "11", "--", "true"], id="cpu-over-quota"),
        pytest.param(["--memory-gib", "65", "--", "true"], id="memory-over-quota"),
        pytest.param(["--extra", "jaxmarl", "--", "true"], id="obsolete-extra-option"),
    ],
)
def test_invalid_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        submit.parse_args(argv)
    assert error.value.code == 2


def test_build_and_submit_preserves_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Invocation directory must not change which repository is built.
    monkeypatch.chdir(tmp_path)
    command = ["uv", "run", "python", "train.py", "--label", "a b; $(touch /tmp/unused)"]
    args = submit.parse_args(
        [
            "--prio",
            "high",
            "--cpus",
            "10",
            "--memory-gib",
            "64",
            "--",
            *command,
        ]
    )
    with patch("subprocess.run") as run:
        name = submit.submit(args)
    calls = run.call_args_list
    build = next(call for call in calls if "buildctl" in call.args[0])
    assert f"context={submit.ROOT}" in build.args[0]
    created = json.loads(calls[-1].kwargs["input"])
    assert "--dry-run=server" not in calls[-1].args[0]
    assert created["metadata"]["name"] == name
    assert created["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "train-high"
    container = created["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == command
    assert container["resources"]["requests"]["cpu"] == "10"
    assert container["resources"]["limits"]["memory"] == "64Gi"
    assert f"type=image,name={container['image']},unpack=true" in build.args[0]
    preview = next(call for call in calls if "--dry-run=server" in call.args[0])
    assert preview.kwargs["input"] == calls[-1].kwargs["input"]
    assert calls.index(preview) < calls.index(build) < len(calls) - 1
    assert all(not call.kwargs.get("shell", False) for call in calls)


@pytest.mark.parametrize("failed_step", [pytest.param(3, id="missing-storage"), pytest.param(5, id="build")])
def test_failure_does_not_submit(failed_step: int) -> None:
    args = submit.parse_args(["--", "true"])
    failure = subprocess.CalledProcessError(returncode=1, cmd="failed-step")
    with (
        patch("subprocess.run", side_effect=[None] * (failed_step - 1) + [failure]) as run,
        pytest.raises(subprocess.CalledProcessError),
    ):
        submit.submit(args)
    assert run.call_count == failed_step
    assert not any("create" in c.args[0] and "--dry-run=server" not in c.args[0] for c in run.call_args_list)


def test_dockerfile_dependency_sync(tmp_path: Path) -> None:
    dockerfile = (submit.ROOT / "infra/k3s/training/Dockerfile").read_text()
    command = next(line.strip() for line in dockerfile.splitlines() if line.strip().startswith("uv sync "))
    result = subprocess.run(
        [*shlex.split(command), "--dry-run", "--offline"],
        cwd=submit.ROOT,
        env={**os.environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
