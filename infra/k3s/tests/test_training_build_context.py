"""Opt-in BuildKit check of training data layers and context exclusions."""

import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RL2_K3S_TESTS") != "1", reason="Requires local BuildKit")
ROOT = Path(__file__).resolve().parents[3]


def build_layers(context: Path) -> list[bytes]:
    result = subprocess.run(
        [
            "sudo",
            "-n",
            "buildctl",
            "--addr",
            "unix:///run/rl2-buildkit/buildkitd.sock",
            "build",
            "--frontend",
            "dockerfile.v0",
            "--local",
            f"context={context}",
            "--local",
            f"dockerfile={context}",
            "--output",
            "type=oci,compression=uncompressed",
            "--progress",
            "plain",
        ],
        capture_output=True,
        check=False,
        timeout=150,
    )
    assert result.returncode == 0, result.stderr.decode()
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:

        def read_member(name: str) -> bytes:
            member = archive.extractfile(name)
            assert member is not None
            with member:
                return member.read()

        index = json.loads(read_member("index.json"))
        manifest_path = "blobs/" + index["manifests"][0]["digest"].replace(":", "/")
        manifest = json.loads(read_member(manifest_path))
        return [read_member("blobs/" + layer["digest"].replace(":", "/")) for layer in manifest["layers"]]


@pytest.mark.parametrize("target", ["training", "visfly"])
def test_source_edits_reuse_data_layer(target: str, tmp_path: Path) -> None:
    dockerfile = (ROOT / "infra/k3s/training/Dockerfile").read_text().splitlines()
    # Exercise the real context-copy instructions without installing GPU dependencies.
    copies = []
    stage = ""
    for line in dockerfile:
        if line.startswith("FROM "):
            stage = line.split()[-1]
        if stage == target and line.startswith("COPY "):
            copies.append(line)
    (tmp_path / "Dockerfile").write_text(
        "\n".join([dockerfile[0], "FROM scratch", "COPY runtime.txt /runtime.txt", *copies]) + "\n"
    )
    (tmp_path / "runtime.txt").write_text("original runtime\n")
    (tmp_path / ".dockerignore").write_text((ROOT / ".dockerignore").read_text())
    included = ["data/alfworld/game.json", "src/rl2/model.py", ".rl2-sweep-test/trial.yaml", "nvida.env"]
    excluded = [
        "cluster-runs/output",
        "src/.venv/bin/python",
        "src/.cache/download",
        "src/.pytest_cache/result",
        "src/rl2/model.pyc",
        ".aws/credentials",
        ".env",
    ]
    for relative in [*included, *excluded]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("initial\n")

    original = build_layers(tmp_path)
    assert len(original) == 3
    with (
        tarfile.open(fileobj=io.BytesIO(original[1])) as data,
        tarfile.open(fileobj=io.BytesIO(original[2])) as source,
    ):
        assert "data/alfworld/game.json" in data.getnames()
        assert not any(name.startswith("data/") for name in source.getnames())
        assert set(included[1:]).issubset(source.getnames())
        assert set(excluded).isdisjoint(source.getnames())

    (tmp_path / "src/rl2/model.py").write_text("changed source\n")
    changed = build_layers(tmp_path)
    assert changed[:2] == original[:2]
    assert changed[2] != original[2]

    for relative in excluded:
        (tmp_path / relative).write_text("changed ignored file\n")
    assert build_layers(tmp_path) == changed

    (tmp_path / "runtime.txt").write_text("updated runtime\n")
    rebased = build_layers(tmp_path)
    assert rebased[0] != changed[0]
    assert rebased[1] == changed[1]
