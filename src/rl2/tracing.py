"""OTLP export and task-local spans for Ray training."""

import asyncio
import os
from collections.abc import Generator, Mapping
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.util.types import AttributeValue

_processor: BatchSpanProcessor | None = None


def tracing_enabled() -> bool:
    return os.getenv("OTEL_SDK_DISABLED", "").lower().strip() != "true" and bool(
        os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    )


def setup_tracing() -> None:
    """Ray startup hook, called once per process; atexit drains worker exporters."""
    global _processor
    if _processor is not None or not tracing_enabled():
        return
    provider = TracerProvider(
        resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "rl2"), "process.pid": os.getpid()})
    )
    _processor = BatchSpanProcessor(OTLPSpanExporter(timeout=5))
    provider.add_span_processor(_processor)
    trace.set_tracer_provider(provider)


def flush_tracing() -> None:
    """Drain this process's queued spans; blocks on exporter calls."""
    if _processor is not None:
        _processor.force_flush()


@contextmanager
def span(
    name: str, attributes: Mapping[str, AttributeValue] | None = None, *, root: bool = False
) -> Generator[trace.Span]:
    """Use inside the executing task; contextvars isolate concurrent awaits."""
    with trace.get_tracer(__name__).start_as_current_span(
        name, attributes=attributes, context=Context() if root else None
    ) as current:
        try:
            yield current
        except asyncio.CancelledError:
            current.set_attribute("cancelled", True)
            current.set_status(trace.StatusCode.ERROR, "Cancelled")
            raise
