"""Operational telemetry remains bounded, private, and non-authoritative."""
from __future__ import annotations

import ast
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

from analyzing_llm_rationale import observability
from analyzing_llm_rationale.twin.recovery import recovery_action
from analyzing_llm_rationale.twin.risk import RiskLimits, size_binary_entry
from analyzing_llm_rationale.twin.worker import (
    InMemoryWorkerJobs,
    TwinWorker,
    WorkerJob,
    WorkerJobError,
    WorkerJobKind,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


class _BrokenExporter(SpanExporter):
    def export(self, spans):
        raise RuntimeError("sink unavailable")


class TwinObservabilityTests(unittest.TestCase):
    def test_decision_log_has_allowlisted_fields_only(self):
        span = SimpleNamespace(name="twin.execution.submit", attributes={
            "outcome": "ambiguous",
            "twin.venue": "kalshi",
            "authorization": "Bearer secret-value",
            "task.payload": '{"api_key":"secret-value"}',
            "wallet.address": "0xprivate",
            "order.id": "venue-order-123",
        })
        with self.assertLogs(observability.logger, level="INFO") as captured:
            result = observability._DecisionSpanLogger().export([span])
        output = "\n".join(captured.output)
        self.assertEqual(result.name, "SUCCESS")
        self.assertIn("outcome=ambiguous", output)
        self.assertIn("twin.venue=kalshi", output)
        for secret in ("secret-value", "api_key", "0xprivate", "venue-order-123"):
            self.assertNotIn(secret, output)

    def test_local_log_export_failure_is_non_authoritative(self):
        exporter = observability._DecisionSpanLogger()
        with patch.object(observability.logger, "info", side_effect=RuntimeError("sink failed")):
            result = exporter.export([
                SimpleNamespace(name="twin.risk.evaluate", attributes={"outcome": "allowed"}),
            ])
        self.assertEqual(result.name, "FAILURE")

    def test_broken_span_exporter_does_not_change_risk_or_recovery(self):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(_BrokenExporter()))
        tracer = provider.get_tracer(__name__)
        with self.assertLogs("opentelemetry.sdk.trace.export", level="ERROR") as captured:
            with tracer.start_as_current_span("exporter-failure-proof"):
                result = size_binary_entry(
                    probability=Decimal("0.70"), ask=Decimal("0.40"),
                    fee_per_share=Decimal("0.01"), slippage_per_share=Decimal("0.01"),
                    available_cash=Decimal("10"), current_market_loss=Decimal("0"),
                    current_cluster_loss=Decimal("0"), drawdown=Decimal("0"),
                    tick_size=Decimal("0.01"), min_quantity=Decimal("1"),
                    limits=RiskLimits(
                        kelly_fraction=Decimal("0.25"), max_order_cash=Decimal("5"),
                        max_market_loss=Decimal("5"), max_cluster_loss=Decimal("5"),
                        max_drawdown=Decimal("0.25"),
                    ),
                    available_depth=Decimal("10"),
                )
                reconciliation = recovery_action("submission_unknown", None)
        self.assertIsNone(result.reason)
        self.assertGreater(result.quantity, 0)
        self.assertEqual(reconciliation, "pause_and_reconcile")
        self.assertNotIn("authorization", "\n".join(captured.output).lower())

    def test_worker_payload_and_startup_log_reject_secret_material(self):
        with self.assertRaises(WorkerJobError):
            WorkerJob(
                "job-1", "scope-1", WorkerJobKind.RECONCILE,
                {"authorization": "Bearer-secret"}, NOW + timedelta(minutes=1),
            )
        worker = TwinWorker(
            InMemoryWorkerJobs(), worker_id="worker-1",
            reconcile_startup=lambda: (_ for _ in ()).throw(
                PermissionError("Bearer secret-value"),
            ),
        )
        with self.assertLogs("analyzing_llm_rationale.twin.worker", level="ERROR") as captured:
            self.assertFalse(worker.start())
        output = "\n".join(captured.output)
        self.assertIn("PermissionError", output)
        self.assertNotIn("secret-value", output)

    def test_metric_labels_exclude_identifiers_and_free_text(self):
        files = (
            "account.py", "budget.py", "execution.py", "recovery.py", "risk.py",
            "scheduler.py", "store.py", "strategy.py", "worker.py",
        )
        forbidden = {
            "account_scope_id", "account_id", "command_id", "job_id", "order_id",
            "reservation_id", "wallet", "wallet_address", "payload", "message",
            "exception", "request", "response",
        }
        found: set[str] = set()
        for name in files:
            tree = ast.parse((ROOT / "src" / "analyzing_llm_rationale" / "twin" / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr not in {"add", "record"} or len(node.args) < 2:
                    continue
                labels = node.args[1]
                if isinstance(labels, ast.Dict):
                    found.update(
                        key.value for key in labels.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    )
        self.assertFalse(found & forbidden, found & forbidden)
        self.assertTrue({"operation", "outcome", "venue", "role"} <= found)

    def test_required_signals_and_runbooks_are_present(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "src" / "analyzing_llm_rationale" / "twin").glob("*.py")
        )
        for metric in (
            "twin.decisions", "twin.duplicate_suppressions", "twin.account.drift",
            "twin.submissions.ambiguous", "twin.data.stale", "twin.retries.exhausted",
            "twin.queue.lag", "twin.research.budget.usd", "twin.research.budget.tokens",
        ):
            self.assertIn(metric, source)
        operations = (ROOT / "docs" / "autonomous-twin" / "OPERATIONS.md").read_text(encoding="utf-8").lower()
        for incident in (
            "kill or pause", "lost research provider", "lost durable store", "stalled queue",
            "uncertain submission", "broken credentials", "missing or incomplete portfolio page",
            "venue halt", "deployment rollback", "settlement correction",
        ):
            self.assertIn(incident, operations)


if __name__ == "__main__":
    unittest.main()
