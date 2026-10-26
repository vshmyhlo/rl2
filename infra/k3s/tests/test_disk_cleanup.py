import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "training" / "cleanup-disk.sh"


@pytest.fixture
def cleanup_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "pvc-test_training_training-runs"
    for name in ("10", "20.orbax-checkpoint-tmp"):
        checkpoint = runs / "run with spaces" / "checkpoints" / name
        checkpoint.mkdir(parents=True)
        (checkpoint / "weights").write_text("checkpoint")
    unrelated = runs / "unrelated.orbax-checkpoint-tmp"
    unrelated.mkdir()
    hf_home = tmp_path / "huggingface"
    for name in ("hub", "xet", "assets"):
        (hf_home / name).mkdir(parents=True)
        (hf_home / name / "data").write_text("cached")
    (hf_home / "token").write_text("test credential")
    (hf_home / "stored_tokens").write_text("test credentials")
    # A symlink under the volume must not expose another tree to cleanup.
    (runs / "external").symlink_to(hf_home, target_is_directory=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    commands = {
        "sudo": 'exec "$@"',
        "k3s": """
if [[ ${KUBE_FAIL:-0} == 1 ]]; then exit 9; fi
case " $* " in
  *" get pvc "*) echo pvc-test ;;
  *" get pv "*) echo "$RUNS_PATH" ;;
  *" get jobs "*) printf '%s' "$JOBS_JSON" ;;
  *" get pods "*)
    [[ " $* " == *"status.phase!=Succeeded,status.phase!=Failed"* ]] || exit 8
    printf '%s' "${ACTIVE_PODS:-}"
    ;;
  *) exit 8 ;;
esac
""",
        "buildctl": 'echo "$*" >> "$BUILD_LOG"; exit "${BUILD_EXIT:-0}"',
    }
    for name, body in commands.items():
        command = bin_dir / name
        command.write_text(f"#!/usr/bin/env bash\nset -eu\n{body}\n")
        command.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("RUNS_PATH", str(runs))
    monkeypatch.setenv("BUILD_LOG", str(tmp_path / "build.log"))
    monkeypatch.setenv("JOBS_JSON", json.dumps({"items": []}))
    for name in ("SUDO_USER", "ACTIVE_PODS", "KUBE_FAIL", "BUILD_EXIT"):
        monkeypatch.delenv(name, raising=False)


def run_cleanup(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=5, check=False)


def test_cleanup_apply(tmp_path: Path, cleanup_env: None) -> None:
    result = run_cleanup("--apply")
    assert result.returncode == 0, result.stderr
    runs = tmp_path / "pvc-test_training_training-runs"
    checkpoints = runs / "run with spaces" / "checkpoints"
    assert not (checkpoints / "20.orbax-checkpoint-tmp").exists()
    assert (checkpoints / "10" / "weights").read_text() == "checkpoint"
    assert (runs / "unrelated.orbax-checkpoint-tmp").is_dir()
    assert (runs / "external").is_symlink()
    assert sorted(path.name for path in (tmp_path / "huggingface").iterdir()) == ["stored_tokens", "token"]
    assert (tmp_path / "build.log").read_text().strip().endswith("prune --all")
    # Repeating cleanup with no remaining cache or partial checkpoints is harmless.
    assert run_cleanup("--apply").returncode == 0


def test_cleanup_defaults_to_preview(tmp_path: Path, cleanup_env: None) -> None:
    before = sorted(tmp_path.rglob("*"))
    result = run_cleanup()
    assert result.returncode == 0, result.stderr
    assert "20.orbax-checkpoint-tmp" in result.stdout
    assert before == sorted(path for path in tmp_path.rglob("*") if path.name != "build.log")
    assert (tmp_path / "build.log").read_text().strip().endswith(" du")


@pytest.mark.parametrize("mode", ["disabled", "preview", "apply"])
def test_run_retention(tmp_path: Path, cleanup_env: None, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    cutoff = 1_700_000_000
    runs = tmp_path / "pvc-test_training_training-runs"
    for name in ("old run", "recent-file", "recent-directory"):
        checkpoint = runs / "grpo_v2" / name / "checkpoints" / "10"
        checkpoint.mkdir(parents=True)
        (checkpoint / "weights").write_text("checkpoint")
    other = runs / "grpo_reasoning" / "old run"
    other.mkdir(parents=True)
    (other / "weights").write_text("keep")
    (runs / "grpo_v2" / "linked-run").symlink_to(other, target_is_directory=True)
    for path in runs.rglob("*"):
        os.utime(path, (cutoff - 1, cutoff - 1), follow_symlinks=False)
    for path in (
        runs / "grpo_v2" / "recent-file" / "checkpoints" / "10" / "weights",
        runs / "grpo_v2" / "recent-directory" / "checkpoints" / "10",
    ):
        os.utime(path, (cutoff + 1, cutoff + 1))
    date = tmp_path / "bin" / "date"
    date.write_text('#!/usr/bin/env bash\n[[ "$*" == *"7 days ago"* ]] || exit 8\necho 2023-11-14T22:13:20Z\n')
    date.chmod(0o755)
    monkeypatch.setenv(
        "JOBS_JSON",
        json.dumps(
            {
                "items": [
                    {"status": {"conditions": [{"type": status, "status": "True"}]}}
                    for status in ("Complete", "Failed")
                ]
            }
        ),
    )
    args = [] if mode == "disabled" else ["--run-retention-days", "7"]
    if mode != "preview":
        args.append("--apply")
    result = run_cleanup(*args)
    assert result.returncode == 0, result.stderr
    assert (runs / "grpo_v2" / "old run").exists() == (mode != "apply")
    assert (str(runs / "grpo_v2" / "old run") in result.stdout) == (mode != "disabled")
    assert (runs / "grpo_v2" / "recent-file" / "checkpoints" / "10" / "weights").exists()
    assert (runs / "grpo_v2" / "recent-directory" / "checkpoints" / "10" / "weights").exists()
    assert (runs / "grpo_v2" / "linked-run").is_symlink()
    assert (other / "weights").read_text() == "keep"


def test_retention_refuses_queued_jobs(tmp_path: Path, cleanup_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "JOBS_JSON", json.dumps({"items": [{"metadata": {"name": "queued-job"}, "spec": {"suspend": True}}]})
    )
    before = sorted(tmp_path.rglob("*"))
    result = run_cleanup("--apply", "--run-retention-days", "7")
    assert result.returncode != 0
    assert "queued-job" in result.stderr
    assert sorted(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("value", [None, "0", "-1", "1.5", "oops"])
def test_retention_rejects_invalid_age(tmp_path: Path, cleanup_env: None, value: str | None) -> None:
    before = sorted(tmp_path.rglob("*"))
    args = ["--apply", "--run-retention-days"]
    if value is not None:
        args.append(value)
    result = run_cleanup(*args)
    assert result.returncode != 0
    assert "positive integer" in result.stderr
    assert sorted(tmp_path.rglob("*")) == before


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("ACTIVE_PODS", "pod/train-active"),
        ("KUBE_FAIL", "1"),
        ("RUNS_PATH", "/"),
        ("BUILD_EXIT", "7"),
    ],
    ids=["active-training", "cluster-unavailable", "invalid-volume", "buildkit-failure"],
)
def test_cleanup_failure_preserves_files(
    tmp_path: Path, cleanup_env: None, monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    before = sorted(tmp_path.rglob("*"))
    monkeypatch.setenv(variable, value)
    result = run_cleanup("--apply")
    assert result.returncode != 0
    assert before == sorted(path for path in tmp_path.rglob("*") if path.name != "build.log")
