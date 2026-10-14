import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

pins = SimpleNamespace(**runpy.run_path(str(Path(__file__).resolve().parents[1] / "infra/k3s/training/image_pins.py")))


def job(name: str, condition: str | None = None) -> dict[str, Any]:
    return {
        "metadata": {"name": name, "namespace": "training"},
        "spec": {
            "suspend": True,
            "template": {"spec": {"containers": [{"image": f"{pins.IMAGE_PREFIX}{name}"}]}},
        },
        "status": {"conditions": [{"type": condition, "status": "True"}] if condition else []},
    }


def result(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)


def test_reconcile_preserves_pending_and_preempted_jobs() -> None:
    jobs = [job("queued"), job("preempted", "Suspended"), job("complete", "Complete"), job("failed", "Failed")]
    # One terminal Job sharing an image must not release the active Job's pin.
    shared = job("queued", "Complete")
    names = ["queued", "preempted", "complete", "failed", "deleted"]
    images = "\n".join([*(pins.IMAGE_PREFIX + name for name in names), "other/image:tag"])
    with patch(
        "subprocess.run", side_effect=[result(json.dumps({"items": [*jobs, shared]})), result(images)] + [result()] * 5
    ) as run:
        pins.reconcile()
    labels = {call.args[0][-2]: call.args[0][-1] for call in run.call_args_list[2:]}
    assert labels == {
        pins.IMAGE_PREFIX + name: f"{pins.PIN_LABEL}={'pinned' if name in ('queued', 'preempted') else ''}"
        for name in names
    }


def test_unavailable_api_preserves_all_pins() -> None:
    with (
        patch("subprocess.run", side_effect=subprocess.CalledProcessError(returncode=1, cmd="kubectl")) as run,
        pytest.raises(subprocess.CalledProcessError),
    ):
        pins.reconcile()
    assert run.call_count == 1


@pytest.mark.parametrize("creation_fails", [False, True], ids=["created", "response-lost"])
def test_pin_before_create_and_keep_pin_if_creation_is_uncertain(creation_fails: bool) -> None:
    manifest = job("test")
    creation_result = subprocess.CalledProcessError(returncode=1, cmd="kubectl") if creation_fails else result()
    with patch("subprocess.run", side_effect=[result(), creation_result]) as run:
        if creation_fails:
            with pytest.raises(subprocess.CalledProcessError):
                pins.submit_job(manifest)
        else:
            pins.submit_job(manifest)
    pin, create = run.call_args_list
    assert pin.args[0][-2:] == [pins.IMAGE_PREFIX + "test", pins.PIN_LABEL + "=pinned"]
    assert "create" in create.args[0]
    assert json.loads(create.kwargs["input"]) == manifest


def test_pin_failure_prevents_submission() -> None:
    with (
        patch("subprocess.run", side_effect=subprocess.CalledProcessError(returncode=1, cmd="ctr")) as run,
        pytest.raises(subprocess.CalledProcessError),
    ):
        pins.submit_job(job("test"))
    assert run.call_count == 1
