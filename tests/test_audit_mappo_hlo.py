"""Compile the actual MAPPO entry points with abstract inputs and a small environment."""

import hashlib
import json
import runpy
from pathlib import Path
from unittest.mock import patch

import jax
from test_train_mappo import ScriptedSMAX
from test_train_mappo import config as config  # noqa: PLC0414

from rl2 import train_mappo as ppo


def test_compile_steps_exports_reset_rollout_and_training(config: ppo.Config, tmp_path: Path) -> None:
    audit = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_mappo_hlo.py"))
    with patch.object(ppo, "make_env", return_value=ScriptedSMAX()):
        audit["compile_steps"](config, tmp_path)
    for name in ("reset", "rollout", "train"):
        report = json.loads((tmp_path / f"{name}.summary.json").read_text())
        assert report["step"] == name
        assert len(report["devices"]) == jax.device_count()
        assert report["config"]["num_envs"] == config.num_envs
        assert not report["executed"] and not report["host_callbacks"]
        assert report["memory_bytes"]["output_size_in_bytes"] > 0
        assert report["operations"]
        for filename, digest in report["source_sha256"].items():
            assert hashlib.sha256((tmp_path / "sources" / filename).read_bytes()).hexdigest() == digest
        if name != "reset":  # The scripted reset does not consume its key.
            assert "P('batch',)" in report["input_shardings"]
        assert "P('batch',)" in report["output_shardings"]
        assert "HloModule" in (tmp_path / f"{name}.optimized.hlo.txt").read_text()
        assert "module" in (tmp_path / f"{name}.stablehlo.mlir").read_text()
    assert report["memory_bytes"]["alias_size_in_bytes"] > 0
