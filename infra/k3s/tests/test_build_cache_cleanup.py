import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "training" / "cleanup-build-cache.sh"


@pytest.mark.parametrize("exit_code", [0, 7], ids=["success", "buildkit-failure"])
def test_cleanup_build_cache(tmp_path: Path, exit_code: int) -> None:
    buildctl = tmp_path / "buildctl"
    buildctl.write_text(
        f"#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\nsys.exit({exit_code})\n"
    )
    buildctl.chmod(0o755)

    result = subprocess.run(
        ["bash", str(SCRIPT), "--keep-duration", "24h", "--filter", "description~=some cache"],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == exit_code
    assert json.loads(result.stdout) == [
        "--addr",
        "unix:///run/rl2-buildkit/buildkitd.sock",
        "prune",
        "--all",
        "--keep-duration",
        "24h",
        "--filter",
        "description~=some cache",
    ]
