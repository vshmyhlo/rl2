import asyncio
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import grpc
import pytest
import ray
from opentelemetry import trace
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
    TraceServiceServicer,
    add_TraceServiceServicer_to_server,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from ray.actor import ActorHandle

from rl2 import tracing


@pytest.mark.parametrize(
    ("endpoint", "disabled", "enabled"),
    [
        (None, "", False),
        ("OTEL_EXPORTER_OTLP_ENDPOINT", "", True),
        ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "", True),
        ("OTEL_EXPORTER_OTLP_ENDPOINT", "true", False),
    ],
)
def test_tracing_enabled(endpoint: str | None, disabled: str, enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_SDK_DISABLED", disabled)
    if endpoint:
        monkeypatch.setenv(endpoint, "http://tempo:4317")
    assert tracing.tracing_enabled() is enabled


def test_setup_once_and_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemorySpanExporter()
    factory = Mock(return_value=exporter)
    install = Mock()
    monkeypatch.setattr(tracing, "_provider", None)
    monkeypatch.setattr(tracing, "OTLPSpanExporter", factory)
    monkeypatch.setattr(trace, "set_tracer_provider", install)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://tempo:4317")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "test")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "k8s.job.name=test-job")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    tracing.setup_tracing()
    provider = install.call_args.args[0]
    with ExitStack() as resources:
        resources.callback(provider.shutdown)
        tracing.setup_tracing()
        with provider.get_tracer("test").start_as_current_span("exported"):
            pass
        tracing.flush_tracing()
        exported = exporter.get_finished_spans()
        assert len(exported) == 1
        assert exported[0].resource.attributes["service.name"] == "test"
        assert exported[0].resource.attributes["k8s.job.name"] == "test-job"
    factory.assert_called_once_with(timeout=5)
    install.assert_called_once()


@ray.remote(num_cpus=0)
class TraceWorker:
    def __init__(self) -> None:
        self.started = 0
        self.ready = asyncio.Event()

    async def work(self, label: str) -> str:
        with tracing.span("child", {"label": label}):
            before = trace.get_current_span().get_span_context()
            self.started += 1
            if self.started == 2:
                self.ready.set()
            await self.ready.wait()
            assert trace.get_current_span().get_span_context() == before
            return format(before.trace_id, "032x")


@ray.remote(num_cpus=0)
class TraceForwarder:
    async def work(self, worker: ActorHandle, label: str) -> str:
        return await worker.work.remote(label)


def run_ray_tracing() -> None:
    with (
        ExitStack() as resources,
        ray.init(
            address="local",
            num_cpus=1,
            num_gpus=0,
            include_dashboard=False,
            object_store_memory=80 * 1024**2,
            _tracing_startup_hook="rl2.tracing:setup_tracing",
        ),
    ):
        resources.callback(tracing.flush_tracing)
        worker = TraceWorker.remote()
        forwarder = TraceForwarder.remote()

        async def call(label: str) -> str:
            with tracing.span("root", {"label": label}, root=True):
                parent = trace.get_current_span().get_span_context()
                child_trace_id = await forwarder.work.remote(worker, label)
                assert child_trace_id == format(parent.trace_id, "032x")
                assert trace.get_current_span().get_span_context() == parent
                return child_trace_id

        async def run() -> None:
            previous = trace.get_current_span().get_span_context()
            async with asyncio.timeout(30):
                ids = await asyncio.gather(call("a"), call("b"))
            assert ids[0] != ids[1]
            assert trace.get_current_span().get_span_context() == previous

        with tracing.span("driver"):
            asyncio.run(run())
        stopped = [forwarder.__ray_terminate__.remote(), worker.__ray_terminate__.remote()]
        ray.wait(stopped, num_returns=2, timeout=10)


def test_ray_propagates_async_context_and_exports_on_exit() -> None:
    requests: list[ExportTraceServiceRequest] = []

    class Receiver(TraceServiceServicer):
        def Export(
            self, request: ExportTraceServiceRequest, context: grpc.ServicerContext
        ) -> ExportTraceServiceResponse:
            requests.append(request)
            return ExportTraceServiceResponse()

    # Ray's Unix socket paths exceed the OS limit under pytest's longer tmp_path.
    with (
        TemporaryDirectory(prefix="rl2-otel-") as directory,
        ThreadPoolExecutor(max_workers=2) as pool,
        ExitStack() as resources,
    ):
        server = grpc.server(pool)
        add_TraceServiceServicer_to_server(Receiver(), server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        resources.callback(lambda: server.stop(0).wait())
        result = subprocess.run(
            [sys.executable, __file__],
            env={
                **{key: value for key, value in os.environ.items() if not key.startswith("OTEL_")},
                "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{port}",
                "OTEL_BSP_SCHEDULE_DELAY": "60000",
                "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
                "RAY_TMPDIR": directory,
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    exported = [
        item
        for request in requests
        for resource in request.resource_spans
        for scope in resource.scope_spans
        for item in scope.spans
    ]
    roots = [item for item in exported if item.name == "root"]
    children = [item for item in exported if item.name == "child"]
    assert len(roots) == len(children) == 2
    assert {item.trace_id for item in roots} == {item.trace_id for item in children}
    by_id = {item.span_id: item for item in exported}
    for child in children:
        ancestor = child
        while ancestor.parent_span_id:
            ancestor = by_id[ancestor.parent_span_id]
        assert ancestor.name == "root"
        assert child.attributes == ancestor.attributes


if __name__ == "__main__":
    run_ray_tracing()
