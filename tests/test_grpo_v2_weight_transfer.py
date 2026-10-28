import json
from collections.abc import Generator
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any, Literal
from unittest.mock import Mock

import httpx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from safetensors.numpy import load_file
from vllm.lora.peft_helper import PEFTHelper

from rl2.grpo_v2 import weight_transfer as transfer
from rl2.qwen3 import LoRAParameters, LoRAWeights


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Generator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_trace, "get_tracer", Mock(return_value=provider.get_tracer("test")))
    with ExitStack() as resources:
        resources.callback(provider.shutdown)
        yield exporter


@pytest.mark.parametrize(
    "pause_mode,failure_path",
    [
        ("wait", None),
        ("abort", None),
        ("keep", None),
        ("wait", "/pause"),
        ("wait", "/v1/load_lora_adapter"),
        ("wait", "/resume"),
        ("wait", "export"),
    ],
)
def test_lora_publication(
    pause_mode: Literal["wait", "abort", "keep"],
    failure_path: str | None,
    monkeypatch: pytest.MonkeyPatch,
    spans: InMemorySpanExporter,
) -> None:
    params: LoRAParameters = {
        "model.layers.0.self_attn.q_proj.weight": LoRAWeights(
            a=jnp.ones((1, 3), dtype=jnp.float32),  # (R=1, I=3)
            b=jnp.zeros((2, 1), dtype=jnp.float32),  # (O=2, R)
        ),
        "model.layers.0.mlp.up_proj.weight": LoRAWeights(
            a=jnp.ones((1, 2), dtype=jnp.float32),  # (R=1, I=2)
            b=jnp.zeros((3, 1), dtype=jnp.float32),  # (O=3, R)
        ),
    }
    requests: list[httpx.Request] = []
    directories: list[Path] = []
    save_file = transfer.save_file

    def save(weights: dict[str, np.ndarray], filename: Path) -> None:  # PEFT (R, I), (O, R) leaves.
        assert all(not previous.exists() for previous in directories)
        directories.append(filename.parent)
        if failure_path == "export":
            raise OSError("export failed")
        save_file(weights, filename)

    monkeypatch.setattr(transfer, "save_file", save)

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        directory = directories[-1]
        assert directory.is_dir()  # Files survive until resume or failure.
        assert sorted(file.name for file in directory.iterdir()) == ["adapter_config.json", "adapter_model.safetensors"]
        if request.url.path == "/pause":
            assert dict(request.url.params) == {"mode": pause_mode, "clear_cache": "true"}
        elif request.url.path == "/v1/load_lora_adapter":
            assert json.loads(request.content) == {
                "lora_name": "policy",
                "lora_path": str(directory),
                "load_inplace": True,
            }
            weights = load_file(directory / "adapter_model.safetensors")  # PEFT (R, I), (O, R) leaves.
            expected = {
                f"base_model.model.{name.removesuffix('.weight')}.lora_{factor}.weight": value
                for name, adapter in params.items()
                for factor, value in (("A", adapter.a), ("B", adapter.b))
            }
            assert weights.keys() == expected.keys()
            for name, value in weights.items():  # (R, I) or (O, R)
                assert value.shape == expected[name].shape and value.dtype == np.float32
                np.testing.assert_array_equal(value, expected[name])
            config = json.loads((directory / "adapter_config.json").read_text())
            assert config["base_model_name_or_path"] == "Qwen/custom"
            assert config["target_modules"] == ["q_proj", "up_proj"]
            helper = PEFTHelper.from_dict(config)
            assert helper.r == 1 and helper.lora_alpha == 3
        return httpx.Response(status_code=500 if request.url.path == failure_path else 200)

    client = httpx.Client(base_url="http://vllm", transport=httpx.MockTransport(respond))
    factory = Mock(return_value=client)
    monkeypatch.setattr(transfer.httpx, "Client", factory)
    expected_error = OSError if failure_path == "export" else httpx.HTTPStatusError
    with (
        pytest.raises(expected_error) if failure_path else nullcontext(),
        transfer.lora_weight_publisher(
            lambda: params,
            server_url="http://vllm",
            model="Qwen/custom",
            adapter_name="policy",
            lora_rank=1,
            lora_alpha=3,
            pause_mode=pause_mode,
        ) as publish,
    ):
        publish()
        assert not directories[-1].exists()
        params = jax.tree.map(lambda value: value + 1, params)  # Current (R, I), (O, R) leaves.
        publish()
    paths = ["/pause", "/v1/load_lora_adapter", "/resume"]
    expected_paths = (
        [] if failure_path == "export" else paths[: paths.index(failure_path) + 1] if failure_path else paths * 2
    )
    assert [request.url.path for request in requests] == expected_paths
    assert all(request.method == "POST" for request in requests)
    assert all(not directory.exists() for directory in directories)
    assert client.is_closed
    factory.assert_called_once_with(base_url="http://vllm", timeout=600)
    finished = spans.get_finished_spans()
    parents = [item for item in finished if item.name == "learner.publish_weights"]
    phases = [item for item in finished if item.name.startswith("learner.publish_weights.")]
    phase_names = {"/pause": "pause", "/v1/load_lora_adapter": "load_adapter", "/resume": "resume"}
    assert [item.name for item in phases] == [f"learner.publish_weights.{phase_names[path]}" for path in expected_paths]
    assert all(item.parent.span_id in {parent.context.span_id for parent in parents} for item in phases)
    assert parents[-1].status.status_code == (
        otel_trace.StatusCode.ERROR if failure_path else otel_trace.StatusCode.UNSET
    )


def test_weight_source_uses_one_replica_and_current_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = Mesh(devices=jax.devices()[:2], axis_names=("replica",))
    params = jax.device_put(
        {
            "model.norm.weight": jnp.array([1.001, 0.999, 1.007], dtype=jnp.float32),  # (D=3,)
            "lm_head.weight": jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 7,  # (V=2, D)
        },
        NamedSharding(mesh, PartitionSpec()),
    )  # Replicated HF parameter shapes.
    source = transfer.JAXWeightSource(get_params=lambda: params)
    monkeypatch.setattr(jax, "device_get", Mock(side_effect=AssertionError("Host transfer is forbidden")))
    for version in range(2):
        metadata = source.metadata()
        weights = list(source)  # Torch HF parameter shapes.
        assert [meta.name for meta in metadata] == [name for name, _ in weights] == sorted(params)
        for meta, (name, weight) in zip(metadata, weights, strict=True):
            assert meta.shape == tuple(weight.shape) == params[name].shape
            assert meta.dtype == weight.dtype == torch.bfloat16
            assert weight.is_cuda == (params[name].addressable_data(0).device.platform == "gpu")
            np.testing.assert_array_equal(
                weight.float().cpu().numpy(), params[name].astype(jnp.bfloat16).astype(jnp.float32)
            )
        if version == 0:
            params = jax.tree.map(lambda value: value + 1, params)  # Same HF shapes; replaces the donated state.


def test_weight_source_rejects_sharded_parameters_before_transfer() -> None:
    if len(jax.devices()) < 2:
        pytest.skip("Requires two devices; use JAX_NUM_CPU_DEVICES=2 on CPU")
    mesh = Mesh(devices=jax.devices()[:2], axis_names=("replica",))
    params = {"weight": jax.device_put(jnp.ones((2, 3)), NamedSharding(mesh, PartitionSpec("replica")))}  # (O=2, I=3)
    source = transfer.JAXWeightSource(get_params=lambda: params)
    with pytest.raises(ValueError, match="replicated learner parameters"):
        source.metadata()


@pytest.mark.parametrize(
    "pause_mode,failure",
    [("wait", None), ("abort", None), ("keep", None), ("wait", "pause"), ("wait", "transfer"), ("wait", "resume")],
)
def test_publisher_lifecycle(
    pause_mode: Literal["wait", "abort", "keep"], failure: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        phase = request.url.path.removeprefix("/")
        events.append(phase)
        if phase == "pause":
            assert dict(request.url.params) == {"mode": pause_mode, "clear_cache": "true"}
        assert not request.content
        return httpx.Response(status_code=500 if failure == phase else 200)

    client = httpx.Client(base_url="http://vllm", transport=httpx.MockTransport(respond))
    monkeypatch.setattr(transfer.httpx, "Client", Mock(return_value=client))
    device = Mock(return_value=nullcontext())
    monkeypatch.setattr(transfer.torch.cuda, "device", device)
    monkeypatch.setattr(transfer, "get_open_port", Mock(return_value=12345))
    sync_client = Mock()
    sync_client.shutdown_weight_transfer_engine.side_effect = lambda: events.append("worker_shutdown")
    sync_factory = Mock(return_value=sync_client)
    monkeypatch.setattr(transfer, "HTTPVLLMWeightSyncClient", sync_factory)
    engine = Mock()
    engine.shutdown.side_effect = lambda: events.append("trainer_shutdown")

    def send() -> None:
        events.append("transfer")
        if failure == "transfer":
            raise RuntimeError("transfer failed")

    engine.send_weights.side_effect = send
    params = {"weight": jnp.ones((2, 3), dtype=jnp.float32)}  # (O=2, I=3)

    def initialize(info: transfer.NCCLTrainerInitInfo, *, client: Any, source: transfer.JAXWeightSource) -> Mock:
        events.append("init")
        assert client is sync_client
        assert source.get_params() is params
        assert info.master_address == "127.0.0.1" and info.master_port == 12345
        assert info.rank == 0 and info.world_size == 2
        assert info.packed and info.packed_buffer_size_bytes == 64 * 1024 * 1024 and info.packed_num_buffers == 2
        return engine

    initialize_mock = Mock(side_effect=initialize)
    monkeypatch.setattr(transfer.NCCLTrainerWeightTransferEngine, "trainer_init", initialize_mock)
    expected_error = RuntimeError if failure == "transfer" else httpx.HTTPStatusError
    with (
        pytest.raises(expected_error) if failure else nullcontext(),
        transfer.full_weight_publisher(
            lambda: params, server_url="http://vllm", device_index=1, pause_mode=pause_mode
        ) as publish,
    ):
        publish()
        publish()  # Reuses the communicator.
    expected = ["init", "pause", "transfer", "resume"]
    if failure:
        expected = expected[: expected.index(failure) + 1]
    else:
        expected += ["pause", "transfer", "resume"]
    assert events == expected + ["worker_shutdown", "trainer_shutdown"]
    initialize_mock.assert_called_once()
    device.assert_called_once_with(1)
    sync_factory.assert_called_once_with(base_url="http://vllm", timeout=600)
    assert client.is_closed
