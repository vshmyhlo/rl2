import json
import os
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

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
    assert container["securityContext"] == {"capabilities": {"add": ["SYS_PTRACE"]}}
    env = {entry["name"]: entry["value"] for entry in container["env"]}
    assert env["JAX_PLATFORMS"] == "cuda,cpu"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://tempo.monitoring.svc.cluster.local:4317"
    assert env["OTEL_RESOURCE_ATTRIBUTES"] == "k8s.job.name=train-test"
    assert "NVIDIA_VISIBLE_DEVICES" not in env  # The device plugin supplies the allocated GPU IDs.
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert set(env["NVIDIA_DRIVER_CAPABILITIES"].split(",")) >= {"compute", "utility", "graphics"}
    assert container["resources"]["requests"] == {"cpu": "10", "memory": "32Gi", "nvidia.com/gpu": "1"}
    assert container["resources"]["limits"] == {"memory": "32Gi", "nvidia.com/gpu": "1"}
    run_mount = next(mount for mount in container["volumeMounts"] if mount["mountPath"] == "/runs")
    run_volume = next(volume for volume in pod["volumes"] if volume["name"] == run_mount["name"])
    assert run_volume["persistentVolumeClaim"]["claimName"] == "training-runs"


def test_grpo_timeline_directory_is_persistent() -> None:
    preset = "configs/grpo_v2/codegym.yaml"
    args = submit.parse_args(
        ["--gpus", "2", "--", "uv", "run", "python", "-m", "rl2.train_grpo_v2", "--config", preset]
    )
    job = submit.job_manifest(args=args, name="train-grpo", image="rl2.local/train:test")
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    env = {entry["name"]: entry["value"] for entry in container["env"]}
    mount = next(
        mount
        for mount in container["volumeMounts"]
        if env["RL2_RAY_TIMELINE_PATH"].startswith(mount["mountPath"] + "/")
    )
    volume = next(volume for volume in pod["volumes"] if volume["name"] == mount["name"])
    assert volume["persistentVolumeClaim"]["claimName"] == "training-runs"
    assert container["command"] == args.command
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert env["RL2_RAY_TIMELINE_URL"] == "http://localhost:8085/?job=train-grpo"
    assert "RAY_PROFILING" not in env
    assert "RAY_enable_timeline" not in env
    assert env["RL2_RAY_DASHBOARD_HOST"] == "0.0.0.0"
    assert env["RL2_VLLM_HOST"] == "0.0.0.0"
    assert container["ports"] == [
        {"name": "ray-dashboard", "containerPort": 8265},
        {"name": "vllm", "containerPort": 8000},
    ]


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([], id="missing-separator"),
        pytest.param(["--"], id="empty-command"),
        pytest.param(["--prio", "urgent", "--", "true"], id="unknown-priority"),
        pytest.param(["--cpus", "0", "--", "true"], id="zero-cpus"),
        pytest.param(["--cpus", "11", "--", "true"], id="cpu-over-quota"),
        pytest.param(["--gpus", "0", "--", "true"], id="zero-gpus"),
        pytest.param(["--gpus", "3", "--", "true"], id="gpu-over-quota"),
        pytest.param(["--gpus", "1.5", "--", "true"], id="fractional-gpus"),
        pytest.param(["--memory-gib", "65", "--", "true"], id="memory-over-quota"),
        pytest.param(["--extra", "jaxmarl", "--", "true"], id="obsolete-extra-option"),
    ],
)
def test_invalid_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        submit.parse_args(argv)
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("argument", "escaped"),
    [
        pytest.param("$(JAX_PLATFORMS)", "$$(JAX_PLATFORMS)", id="environment-reference"),
        pytest.param("echo $$", "echo $$$$", id="shell-pid"),
        pytest.param("$$(JAX_PLATFORMS)", "$$$$(JAX_PLATFORMS)", id="literal-escaped-reference"),
        pytest.param("echo $HOME/${HOME}", "echo $$HOME/$${HOME}", id="shell-variables"),
    ],
)
def test_command_escapes_kubernetes_expansion(argument: str, escaped: str) -> None:
    args = submit.parse_args(["--", "sh", "-c", argument])
    job = submit.job_manifest(args=args, name="train-test", image="rl2.local/train:test")
    assert job["spec"]["template"]["spec"]["containers"][0]["command"] == ["sh", "-c", escaped]
    assert args.command == ["sh", "-c", argument]


def test_build_and_submit_preserves_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Invocation directory must not change which repository is built.
    monkeypatch.chdir(tmp_path)
    command = ["uv", "run", "python", "train.py", "--label", "a b; $(touch /tmp/unused)"]
    args = submit.parse_args(
        [
            "--prio",
            "high",
            "--gpus",
            "2",
            "--cpus",
            "4",
            "--memory-gib",
            "64",
            "--",
            *command,
        ]
    )
    with patch("subprocess.run") as run:
        names = submit.submit(args)
    calls = run.call_args_list
    build = next(call for call in calls if "buildctl" in call.args[0])
    assert f"context={submit.ROOT}" in build.args[0]
    created = json.loads(calls[-1].kwargs["input"])
    assert calls[-1].args[0][-2:] == [submit.IMAGE_MANAGER, "submit"]
    assert [created["metadata"]["name"]] == names
    assert created["metadata"]["labels"]["kueue.x-k8s.io/priority-class"] == "train-high"
    container = created["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["uv", "run", "python", "train.py", "--label", "a b; $$(touch /tmp/unused)"]
    assert container["resources"]["requests"]["cpu"] == "4"
    assert container["resources"]["limits"]["memory"] == "64Gi"
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "2"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"
    output = capsys.readouterr().out
    assert "priority=high, GPU=2" in output
    assert "Training jobs: http://localhost:8085/" in output
    assert f"Ray dashboard (when enabled): http://localhost:8085/dashboard/{names[0]}/" in output
    assert "port-forward" not in output
    assert "Login token" not in output
    assert f"Ray timeline (when recorded): http://localhost:8085/?job={names[0]}" in output
    assert f"type=image,name={container['image']},unpack=true" in build.args[0]
    preview = next(call for call in calls if "--dry-run=server" in call.args[0])
    assert preview.kwargs["input"] == calls[-1].kwargs["input"]
    assert calls.index(preview) < calls.index(build) < len(calls) - 1
    timer_check = next(call for call in calls if "systemctl" in call.args[0])
    assert calls.index(timer_check) < calls.index(build)
    assert all(not call.kwargs.get("shell", False) for call in calls)


def test_sweep_builds_once_and_resumes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(submit.submit.__globals__, "ROOT", tmp_path)
    (tmp_path / "base.yaml").write_text("seed: 1\nckpt_dir: /runs/ppo/${run_id}\n")
    path = tmp_path / "sweep.yaml"
    path.write_text("name: seeds\nconfig: base.yaml\nckpt_dir: /runs/sweep/${run_id}\nparameters: {seed: [1, 2]}\n")
    args = submit.parse_args(["--sweep", str(path), "--", "uv", "run", "ppo"])
    snapshots = []

    def capture_build(command: list[str], **kwargs: object) -> None:
        if "buildctl" in command:
            snapshots.extend(OmegaConf.load(path) for path in tmp_path.glob(".rl2-sweep-*/*.yaml"))

    identities = []
    for _ in range(2):
        with patch("subprocess.run", side_effect=capture_build) as run:
            names = submit.submit(args)
        calls = run.call_args_list
        builds = [c for c in calls if "buildctl" in c.args[0]]
        assert len(builds) == 1
        submissions = [c for c in calls if submit.IMAGE_MANAGER in c.args[0]]
        jobs = [json.loads(c.kwargs["input"]) for c in submissions]
        assert names == [job["metadata"]["name"] for job in jobs]
        assert len(jobs) == 2
        identities.append([job["metadata"]["annotations"]["rl2.local/run-id"] for job in jobs])
        assert len({job["spec"]["template"]["spec"]["containers"][0]["image"] for job in jobs}) == 1
        previews = [c for c in calls if "--dry-run=server" in c.args[0]]
        assert [c.kwargs["input"] for c in previews] == [c.kwargs["input"] for c in submissions]
        assert all(calls.index(c) < calls.index(builds[0]) for c in previews)
        for job in jobs:
            assert job["metadata"]["labels"]["rl2.local/sweep"] == "seeds"
            command = job["spec"]["template"]["spec"]["containers"][0]["command"]
            assert command[:4] == ["uv", "run", "ppo", "--config"]
            assert not (tmp_path / command[-1]).exists()
    assert identities[0] == identities[1]
    assert {trial.seed for trial in snapshots} == {1, 2}
    assert all(trial.ckpt_dir == f"/runs/sweep/{trial.run_id}" for trial in snapshots)
    assert all(
        OmegaConf.to_container(trial, resolve=False)["ckpt_dir"] == "/runs/sweep/${run_id}" for trial in snapshots
    )
    records = list(tmp_path.glob("runs/sweeps/seeds/*.json"))
    assert len(records) == 2
    assert set(json.loads(records[0].read_text())["configs"]) == set(identities[0])


def test_sweep_dry_run_has_no_side_effects(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("seed: 1\n")
    path = tmp_path / "sweep.yaml"
    path.write_text(f"config: {base}\nparameters: {{seed: [1, 2]}}\n")
    args = submit.parse_args(["--dry-run", "--gpus", "2", "--sweep", str(path), "--", "uv", "run", "ppo"])
    with patch("subprocess.run") as run:
        assert submit.submit(args) == []
    run.assert_not_called()
    preview = json.loads(capsys.readouterr().out)
    assert len(preview["trials"]) == 2
    assert preview["gpus"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["base.yaml", "sweep.yaml"]


@pytest.mark.parametrize("config_args", [["--config", "base.yaml"], ["--config=base.yaml"]])
def test_sweep_rejects_command_config(config_args: list[str]) -> None:
    with pytest.raises(SystemExit):
        submit.parse_args(["--sweep", "sweep.yaml", "--", "ppo", *config_args])


@pytest.mark.parametrize("stage", ["validation", "submission"])
def test_sweep_failure_preserves_record_and_cleans_configs(
    stage: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(submit.submit.__globals__, "ROOT", tmp_path)
    (tmp_path / "base.yaml").write_text("seed: 1\n")
    path = tmp_path / "sweep.yaml"
    path.write_text("config: base.yaml\nparameters: {seed: [1, 2, 3]}\n")
    args = submit.parse_args(["--sweep", str(path), "--", "ppo"])
    attempts = 0

    def fail_second_job(command: list[str], **kwargs: object) -> None:
        nonlocal attempts
        if "--dry-run=server" in command if stage == "validation" else submit.IMAGE_MANAGER in command:
            attempts += 1
            if attempts == 2:
                raise subprocess.CalledProcessError(returncode=1, cmd=command)

    with patch("subprocess.run", side_effect=fail_second_job) as run, pytest.raises(subprocess.CalledProcessError):
        submit.submit(args)
    assert attempts == 2
    assert not list(tmp_path.glob(".rl2-sweep-*"))
    records = list(tmp_path.glob("runs/sweeps/sweep/*.json"))
    assert len(records) == 1
    assert len(json.loads(records[0].read_text())["configs"]) == 3
    if stage == "validation":
        assert not any("buildctl" in c.args[0] or submit.IMAGE_MANAGER in c.args[0] for c in run.call_args_list)


def test_main_reports_malformed_sweep(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("parameters: [\n")
    with patch("sys.argv", ["submit.py", "--sweep", str(path), "--", "ppo"]), patch("subprocess.run") as run:
        assert submit.main() == 1
    run.assert_not_called()
    assert "Submission failed:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "failed_step",
    [pytest.param(3, id="missing-storage"), pytest.param(5, id="inactive-cleanup-timer"), pytest.param(6, id="build")],
)
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
    assert not any(submit.IMAGE_MANAGER in c.args[0] for c in run.call_args_list)


@pytest.mark.parametrize("extra", ["", "bfcl"], ids=["training", "bfcl"])
def test_dockerfile_dependency_sync(extra: str, tmp_path: Path) -> None:
    dockerfile = (submit.ROOT / "infra/k3s/training/Dockerfile").read_text()
    command = next(line.strip() for line in dockerfile.splitlines() if line.strip().startswith("uv sync "))
    result = subprocess.run(
        ["sh", "-c", command + " --dry-run --offline"],
        cwd=submit.ROOT,
        env={**os.environ, "UV_CACHE_DIR": str(tmp_path / "uv-cache"), "UV_EXTRA": extra},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
