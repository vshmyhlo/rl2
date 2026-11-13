import os
import subprocess
import sys
from pathlib import Path


def test_mappo_help_in_fresh_process(tmp_path: Path) -> None:
    # In-process imports can hide native-library initialization order crashes.
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "src/rl2/train_mappo.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "MPLCONFIGDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--config" in result.stdout
