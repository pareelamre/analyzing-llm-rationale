"""Provider-neutral OpenTelemetry bootstrap with local decision diagnostics."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Attributes that record a decision rather than a step. A span carrying any of
# them is worth a log line on its own.
_DECISION_KEYS = (
    "outcome",
    "trade.fill_status",
    "trade.fill_outcome",
    "trade.sizing_reason",
    "trade.executable_price",
    "risk_guard.reason",
    "risk_guard.allowed",
    "failure.kind",
    "provider.state",
)


class _DecisionSpanLogger(SpanExporter):
    """Mirror decision-carrying spans into the ordinary log.

    The paid OTLP connector was removed. This processor retains bounded trade
    decision diagnostics in ordinary application logs, while tracing remains
    available to a deployment's own configured provider. Mirroring every span
    would bury the handful of decisions under fetch and parse noise.
    """

    def export(self, spans) -> SpanExportResult:
        for span in spans:
            attrs = dict(getattr(span, "attributes", None) or {})
            if not any(key in attrs for key in _DECISION_KEYS):
                continue
            detail = " ".join(f"{k}={attrs[k]}" for k in sorted(attrs))
            logger.info("trace %s %s", getattr(span, "name", "span"), detail)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

_INITIALIZED = False


def init_observability(app: "FastAPI | None" = None) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    if os.environ.get("ENABLE_OTEL", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    _INITIALIZED = True

    resource = Resource.create({
        "service.name": "foresea",
        "service.version": "1.0.0",
        "deployment.environment.name": os.environ.get("ENVIRONMENT", "production"),
        "vcs.repository.url.full": "https://github.com/pareelamre/analyzing-llm-rationale",
        "vcs.ref.head.revision": os.environ.get("GITHUB_SHA", ""),
    })

    # The provider has no paid remote exporter. Deployment infrastructure may
    # attach its own standard OpenTelemetry exporter without changing trading
    # behavior or source-level credentials.
    tracer_provider = TracerProvider(resource=resource)
    # Simple, not Batch: a tick is a short-lived process and a batched
    # processor can exit before flushing, which would drop exactly the last
    # decision of a cycle -- usually the interesting one.
    if os.environ.get("FORESEA_LOG_DECISION_SPANS", "1").strip().lower() not in (
        "0", "false", "no", "off",
    ):
        tracer_provider.add_span_processor(SimpleSpanProcessor(_DecisionSpanLogger()))
    trace.set_tracer_provider(tracer_provider)

    if app is not None:
        # FastAPI auto-instrumentation (HTTP spans, status codes, route templates)
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)

    # Outbound HTTP auto-instrumentation (provider calls via requests.Session)
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    RequestsInstrumentor().instrument()
