import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

submit = SimpleNamespace(**runpy.run_path(str(Path(__file__).resolve().parents[2] / "submit.py")))


def test_bfcl_job_build_and_submit() -> None:
    command = ["python", "-m", "rl2.bfcl.job", "--run-dir", "/runs/bfcl/qwen3", "--smoke"]
    args = submit.parse_args(["--extra", "bfcl", "--memory-gib", "32", "--", *command])
    with patch("subprocess.run") as run:
        submit.submit(args)
    build = next(call for call in run.call_args_list if "buildctl" in call.args[0])
    assert "build-arg:UV_EXTRA=bfcl" in build.args[0]
    manifest = json.loads(run.call_args.kwargs["input"])
    pod = manifest["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["command"] == command
    assert len(pod["containers"]) == 1
    assert "initContainers" not in pod
    assert container["resources"]["requests"] == {"cpu": "4", "memory": "32Gi", "nvidia.com/gpu": "1"}
    assert container["resources"]["limits"] == {"memory": "32Gi", "nvidia.com/gpu": "1"}


def test_bfcl_preview_does_not_submit(capsys: pytest.CaptureFixture[str]) -> None:
    args = submit.parse_args(["--dry-run", "--extra", "bfcl", "--", "python", "-m", "rl2.bfcl.job"])
    with patch("subprocess.run") as run:
        submit.submit(args)
    run.assert_not_called()
    preview = json.loads(capsys.readouterr().out)
    assert preview["extra"] == "bfcl"
    assert preview["command"] == args.command
