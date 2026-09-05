"""OpenTelemetry bootstrap for Foresea — traces, logs, and metrics to Superlog."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs._internal.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
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

    The OTLP exporter has been returning 402 Payment Required, so every
    ``outcome`` / ``fill_status`` / ``risk_guard.reason`` attribute the trading
    path sets has been discarded. Those are the fields that answer "why did it
    not trade?", and losing them meant reconstructing decisions from the
    audits endpoint and the venue's own API instead of reading a log.

    Spans still *record* -- only the export fails -- so a second processor
    recovers them without depending on the backend being paid up. Scoped to
    decision attributes deliberately: mirroring every span would bury the
    handful that matter under fetch/parse noise.
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

SUPERLOG_ENDPOINT = "https://intake.superlog.sh"
SUPERLOG_PUBLIC_TOKEN = "sl_public_4V3439ks2mBsBuUqSIoSBiG-b3F2pPcIWj89ONGBXgo"


def superlog_headers(token: str) -> dict[str, str]:
    return {"x-api-key": token}


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

    headers = superlog_headers(SUPERLOG_PUBLIC_TOKEN)

    # Traces
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{SUPERLOG_ENDPOINT}/v1/traces",
                headers=headers,
            )
        )
    )
    # Simple, not Batch: a tick is a short-lived process and a batched
    # processor can exit before flushing, which would drop exactly the last
    # decision of a cycle -- usually the interesting one.
    if os.environ.get("FORESEA_LOG_DECISION_SPANS", "1").strip().lower() not in (
        "0", "false", "no", "off",
    ):
        tracer_provider.add_span_processor(SimpleSpanProcessor(_DecisionSpanLogger()))
    trace.set_tracer_provider(tracer_provider)

    # Metrics
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(
            endpoint=f"{SUPERLOG_ENDPOINT}/v1/metrics",
            headers=headers,
        ),
        export_interval_millis=60_000,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Logs — bridge stdlib logging to OTLP so existing `logger.*` calls carry trace_id
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(
                endpoint=f"{SUPERLOG_ENDPOINT}/v1/logs",
                headers=headers,
            )
        )
    )
    set_logger_provider(logger_provider)

    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    LoggingInstrumentor().instrument(set_logging_format=False)

    # Add OTLP handler to root logger so all app loggers flow through it.
    # Filter out opentelemetry's own internal loggers to avoid a recursive loop
    # where export failures emit stdlib log records which re-enter the OTLP handler.
    from opentelemetry.sdk._logs._internal import LoggingHandler

    class _SuppressOtelInternal(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return not record.name.startswith("opentelemetry")

    otlp_handler = LoggingHandler(level=logging.DEBUG, logger_provider=logger_provider)
    otlp_handler.addFilter(_SuppressOtelInternal())
    logging.getLogger().addHandler(otlp_handler)

    if app is not None:
        # FastAPI auto-instrumentation (HTTP spans, status codes, route templates)
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)

    # Outbound HTTP auto-instrumentation (provider calls via requests.Session)
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    RequestsInstrumentor().instrument()
