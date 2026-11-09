"""The HLO audit compiles real training entry points with abstract inputs."""

import json
import runpy
from pathlib import Path

import jax

from rl2.alphazero.config import Config, ModelConfig


def test_compile_steps_exports_sharded_collect_and_update(tmp_path: Path) -> None:
    audit = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_alphazero_hlo.py"))
    config = Config(
        env_id="tic_tac_toe",
        num_envs=jax.device_count(),
        max_moves=2,
        num_simulations=1,
        batch_size=jax.device_count(),
        updates_per_iteration=2,
        model=ModelConfig(channels=2, num_blocks=1),
    )
    audit["compile_steps"](config, tmp_path)
    for name in ("collect", "update"):
        report = json.loads((tmp_path / f"{name}.summary.json").read_text())
        assert report["step"] == name
        assert len(report["devices"]) == jax.device_count()
        assert report["config"]["num_envs"] == config.num_envs
        assert not report["executed"] and not report["host_callbacks"]
        assert report["memory_bytes"]["output_size_in_bytes"] > 0
        assert report["operations"]
        assert "P('batch',)" in report["output_shardings" if name == "collect" else "input_shardings"]
        assert "HloModule" in (tmp_path / f"{name}.optimized.hlo.txt").read_text()
        assert "module" in (tmp_path / f"{name}.stablehlo.mlir").read_text()
