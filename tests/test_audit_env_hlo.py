import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any, TextIO
from unittest.mock import Mock

import jax
import pytest

from rl2 import ppo
from rl2.envs.gymnax_env import GymnaxConfig


@pytest.fixture(scope="module")
def audit() -> dict[str, Any]:
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_env_hlo.py"))


def test_compiles_abstract_batch_with_ppo_mesh(audit: dict[str, Any], tmp_path: Path) -> None:
    config = ppo.Config(
        env=GymnaxConfig(),
        seed=0,
        total_steps=jax.device_count(),
        num_envs=jax.device_count(),
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        model=ppo.ModelConfig(
            encoder=ppo.Conv2DEncoderConfig(embedding_size=2),
            recurrent=ppo.LSTMConfig(hidden_size=2),
        ),
        learning_rate=1e-3,
        anneal_lr=False,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        target_kl=None,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        video_fps=10.0,
    )
    report = audit["compile_case"]("config", config, jax.default_backend(), "batch", tmp_path)
    assert report["status"] == "compiled"
    assert report["device_count"] == config.num_envs
    assert report["batch_size_per_device"] == 1
    assert report["adapter"] == "GymnaxEnv" and report["environment"]["id"] == config.env.id
    assert not report["executed"] and not report["host_callbacks"]
    assert not report["collectives"]
    assert report["operations"]["fusion"] > 0
    assert report["memory_bytes"]["output_size_in_bytes"] > 0
    assert "P('batch',)" in report["output_shardings"]
    assert "HloModule" in (tmp_path / "step.optimized.hlo.txt").read_text()
    assert "module" in (tmp_path / "step.stablehlo.mlir").read_text()
    json.dumps(report)


def test_isolated_cases_report_timeout_and_continue(
    audit: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main = audit["main"]
    monkeypatch.setitem(main.__globals__, "CASES", {"gymnax": None, "pgx": None})
    stale = tmp_path / "gymnax" / "summary.json"
    stale.parent.mkdir()
    stale.write_text(json.dumps({"case": "gymnax", "status": "compiled"}))

    def run(
        command: list[str], *, stdout: TextIO, stderr: int, timeout: int, check: bool
    ) -> subprocess.CompletedProcess[str]:
        assert command[command.index("--batch-size") + 1] == "16"
        assert command[command.index("--config") + 1] == "fixture.yaml"
        name = command[command.index("--env") + 1]
        if name == "gymnax":
            raise subprocess.TimeoutExpired(command, timeout)
        (tmp_path / name / "summary.json").write_text(json.dumps({"case": name, "status": "compiled"}))
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit", "--backend", "cpu", "--output", str(tmp_path), "--batch-size", "16", "--config", "fixture.yaml"],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    reports = json.loads((tmp_path / "summary.json").read_text())
    assert reports[0]["status"] == "failed" and "exceeded" in reports[0]["error"]
    assert reports[1] == {"case": "pgx", "status": "compiled"}


def test_failed_single_case_does_not_leave_stale_hlo(
    audit: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main = audit["main"]
    for filename in ("summary.json", "step.optimized.hlo.txt", "step.stablehlo.mlir"):
        (tmp_path / filename).write_text("stale")
    monkeypatch.setitem(main.__globals__, "load_config", Mock(side_effect=ValueError("invalid fixture")))
    monkeypatch.setattr(sys, "argv", ["audit", "--backend", "cpu", "--env", "gymnax", "--output", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["status"] == "failed" and report["error"] == "ValueError: invalid fixture"
    assert not list(tmp_path.glob("step.*"))
