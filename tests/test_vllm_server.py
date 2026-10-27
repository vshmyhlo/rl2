import asyncio
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest
from vllm.lora.request import LoRARequest

from rl2.vllm_support import server


def test_vllm_import_from_training_script_directory() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from vllm.config.lora import MaxLoRARanks"],
        cwd=Path(__file__).resolve().parents[1] / "src/rl2",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_reload_flag_only_reaches_explicit_engine_updates() -> None:
    updates: list[tuple[str, bool]] = []

    async def add_lora(request: LoRARequest) -> bool:
        updates.append((request.lora_path, request.load_inplace))
        return True

    engine = Mock(add_lora=AsyncMock(side_effect=add_lora))
    models = server.ReloadOnceServingModels(engine_client=engine, base_model_paths=[])

    async def run() -> None:
        for path in ("/tmp/first", "/tmp/second"):
            result = await models.load_lora_adapter(
                LoadLoRAAdapterRequest(lora_name="training-policy", lora_path=path, load_inplace=True)
            )
            assert isinstance(result, str)
            sampling_request = models.lora_requests["training-policy"]
            assert sampling_request.lora_int_id == 1
            assert sampling_request.lora_path == path
            assert sampling_request.load_inplace is False

        engine.add_lora.side_effect = RuntimeError("reload failed")
        result = await models.load_lora_adapter(
            LoadLoRAAdapterRequest(lora_name="training-policy", lora_path="/tmp/failed", load_inplace=True)
        )
        assert isinstance(result, ErrorResponse)
        assert models.lora_requests["training-policy"] is sampling_request
        assert sampling_request.load_inplace is False

    asyncio.run(run())
    assert updates == [("/tmp/first", True), ("/tmp/second", True)]


def test_server_installs_reload_workaround_before_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.serving, "OpenAIServingModels", server.serving.OpenAIServingModels)

    def launch(module: str, *, run_name: str) -> None:
        assert module == "vllm.entrypoints.launchers.api_server.entry"
        assert run_name == "__main__"
        assert server.serving.OpenAIServingModels is server.ReloadOnceServingModels

    run_module = Mock(side_effect=launch)
    monkeypatch.setattr(server.runpy, "run_module", run_module)
    server.main()
    run_module.assert_called_once()
