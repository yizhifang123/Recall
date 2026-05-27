"""OTel span capture: in-process JSON exporter scoped per trial.

Architecture: one global TracerProvider is set up once at process start with
openinference instrumentors active. Each trial wraps execution in a
CaptureContext that attaches a fresh InMemorySpanExporter via its own
SimpleSpanProcessor, and detaches it (via shutdown) on exit. Spans collected
during the context are returned as JSON-serializable dicts.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

_provider_lock = threading.Lock()
_provider: TracerProvider | None = None
_instrumentors_installed = False


class InMemorySpanExporter(SpanExporter):
    """Collects spans into a list. Thread-safe."""

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []
        self._lock = threading.Lock()
        self._shutdown = False

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        if self._shutdown:
            return SpanExportResult.FAILURE
        with self._lock:
            self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        self._shutdown = True


def setup_tracer_provider() -> TracerProvider:
    """Idempotently install the global TracerProvider + openinference hooks."""
    global _provider, _instrumentors_installed
    with _provider_lock:
        if _provider is None:
            _provider = TracerProvider(
                resource=Resource.create({"service.name": "recall-harvester"})
            )
            trace.set_tracer_provider(_provider)
        if not _instrumentors_installed:
            try:
                from openinference.instrumentation.litellm import LiteLLMInstrumentor

                LiteLLMInstrumentor().instrument(tracer_provider=_provider)
            except Exception as exc:
                print(f"[capture] litellm instrumentor failed to install: {exc}")
            try:
                from openinference.instrumentation.langchain import (
                    LangChainInstrumentor,
                )

                LangChainInstrumentor().instrument(tracer_provider=_provider)
            except Exception as exc:
                print(f"[capture] langchain instrumentor failed to install: {exc}")
            _instrumentors_installed = True
        return _provider


def span_to_dict(span: ReadableSpan) -> dict[str, Any]:
    """Serialize a ReadableSpan to a JSON-compatible dict."""
    start = span.start_time
    end = span.end_time
    duration_ms = ((end - start) / 1_000_000) if (start and end) else None
    parent = None
    if span.parent is not None:
        parent = format(span.parent.span_id, "016x")
    return {
        "name": span.name,
        "span_id": format(span.context.span_id, "016x"),
        "trace_id": format(span.context.trace_id, "032x"),
        "parent_span_id": parent,
        "kind": span.kind.name,
        "start_time_ns": start,
        "end_time_ns": end,
        "duration_ms": duration_ms,
        "attributes": _json_safe(dict(span.attributes or {})),
        "status": {
            "code": span.status.status_code.name,
            "description": span.status.description,
        },
        "events": [
            {
                "name": e.name,
                "timestamp_ns": e.timestamp,
                "attributes": _json_safe(dict(e.attributes or {})),
            }
            for e in span.events
        ],
    }


def _json_safe(obj: Any) -> Any:
    """Best-effort JSON-safe coercion."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        # OTel attribute values can be bytes
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
    except Exception:
        pass
    return str(obj)


@contextmanager
def capture_spans() -> Iterator[list[dict[str, Any]]]:
    """Context manager: yields a list that is populated on __exit__.

    Usage:
        with capture_spans() as spans:
            run_agent_code()
        # spans is now a list[dict] of OTel spans
    """
    provider = setup_tracer_provider()
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    collected: list[dict[str, Any]] = []
    try:
        yield collected
    finally:
        try:
            processor.force_flush(timeout_millis=2000)
        except Exception:
            pass
        try:
            processor.shutdown()
        except Exception:
            pass
        collected.extend(span_to_dict(s) for s in exporter.spans)
