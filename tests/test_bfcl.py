import io
import json
import runpy
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from rl2 import bfcl as runner


def test_manifest_refuses_mixed_runs(tmp_path: Path) -> None:
    manifest = {"revision": "original", "temperature": 0.001}
    runner.write_manifest(tmp_path, manifest)
    runner.write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="fresh --run-dir"):
        runner.write_manifest(tmp_path, {**manifest, "revision": "changed"})
    assert json.loads((tmp_path / "protocol.json").read_text()) == manifest


@pytest.mark.parametrize("smoke", [False, True], ids=["full-category-coverage", "smoke-only"])
def test_evaluation_scope(smoke: bool) -> None:
    command = runner.evaluation_command("evaluate", Path("/checkpoint"), smoke)
    assert ("--partial-eval" in command) == smoke
    categories = command[command.index("--test-category") + 1]
    assert categories == (",".join(runner.SMOKE_IDS) if smoke else "single_turn,multi_turn")
    assert command[command.index("--model") + 1] == "Qwen/Qwen3-0.6B-FC"


def test_generate_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for filename in ("config.json", "tokenizer_config.json", "generation_config.json"):
        (checkpoint / filename).write_text("{}")
    download = Mock(return_value=str(checkpoint))
    invoke = Mock()
    monkeypatch.setattr(runner, "snapshot_download", download)
    monkeypatch.setattr(
        runner,
        "urlopen",
        Mock(return_value=io.StringIO(json.dumps({"data": [{"id": str(checkpoint), "max_model_len": 40960}]}))),
    )
    monkeypatch.setattr(subprocess, "run", invoke)
    monkeypatch.setattr("sys.argv", ["rl2-bfcl", "generate", "--smoke", "--run-dir", str(tmp_path / "run")])
    monkeypatch.setattr(runner, "version", Mock(return_value="test"))
    runner.main()
    assert download.call_args.kwargs["revision"] == runner.REVISION
    assert download.call_args.kwargs["local_files_only"]
    command = invoke.call_args.args[0]
    assert "--run-ids" in command and "--skip-server-setup" in command
    assert command[command.index("--local-model-path") + 1] == str(checkpoint)
    assert command[command.index("--temperature") + 1] == "0.001"
    assert invoke.call_args.kwargs["check"]
    assert invoke.call_args.kwargs["env"]["BFCL_PROJECT_ROOT"] == str(tmp_path / "run")
    assert json.loads((tmp_path / "run" / "test_case_ids_to_generate.json").read_text()) == runner.SMOKE_IDS
    assert json.loads((tmp_path / "run" / "protocol.json").read_text())["thinking"] is True


@pytest.mark.parametrize(
    "served_model",
    [{"id": "other-model", "max_model_len": 40960}, {"id": "/checkpoint", "max_model_len": 32768}],
    ids=["wrong-checkpoint", "context-shorter-than-bfcl-budget"],
)
def test_wrong_server_rejected(monkeypatch: pytest.MonkeyPatch, served_model: dict) -> None:
    monkeypatch.setattr(
        runner,
        "urlopen",
        Mock(return_value=io.StringIO(json.dumps({"data": [served_model]}))),
    )
    with pytest.raises(ValueError, match="pinned checkpoint"):
        runner.check_server(Path("/checkpoint"), 1053)


def test_server_uses_pinned_local_checkpoint() -> None:
    command = runner.server_command(Path("/pinned/snapshot"), 1053)
    assert command[command.index("--model") + 1] == "/pinned/snapshot"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--max-model-len") + 1] == "40960"
    assert "--enable-prefix-caching" in command
    assert "--tool-call-parser" not in command  # BFCL parses the raw completions itself.


def test_server_sampler_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    execute = Mock()
    monkeypatch.setattr(runner, "snapshot_download", Mock(return_value="/checkpoint"))
    monkeypatch.setattr(runner.os, "execv", Mock(side_effect=AssertionError("Missing sampler environment")))
    monkeypatch.setattr(runner.os, "execve", execute)
    monkeypatch.setenv("BFCL_TEST_ENV", "preserved")
    monkeypatch.setattr("sys.argv", ["rl2-bfcl", "serve"])
    runner.main()
    environment = execute.call_args.args[2]
    assert environment["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert environment["BFCL_TEST_ENV"] == "preserved"


def test_official_handler_registration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path))
    pytest.importorskip("bfcl_eval")
    module = runpy.run_module("rl2.bfcl_qwen")
    configs = module["MODEL_CONFIG_MAPPING"]
    small = configs["Qwen/Qwen3-0.6B-FC"]
    original = configs["Qwen/Qwen3-8B-FC"]
    assert small.model_handler is original.model_handler
    assert small.is_fc_model
    assert small.model_name == "Qwen/Qwen3-0.6B"
    assert original.model_name == "Qwen/Qwen3-8B"
