"""The decision-span log mirror that keeps trade diagnostics readable."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import observability  # noqa: E402


def _span(name, **attrs):
    return SimpleNamespace(name=name, attributes=attrs)


class DecisionSpanLoggerTests(unittest.TestCase):
    def setUp(self):
        self.exporter = observability._DecisionSpanLogger()

    def _export(self, *spans):
        with self.assertLogs(observability.logger, level="INFO") as captured:
            # Always export one known-decision span so assertLogs has output
            # even when the spans under test are correctly ignored.
            self.exporter.export(list(spans) + [_span("sentinel", outcome="ok")])
        return captured.output

    def test_a_trade_decision_reaches_the_log(self):
        # The ordinary log retains the fields that answer "why did it not
        # trade?" without requiring a remote telemetry subscription.
        lines = self._export(_span(
            "benchmark_tools.place_trade",
            outcome="rejected",
            **{"trade.executable_price": 1.0,
               "trade.submitted": False,
               "risk_guard.reason": "no_executable_price"},
        ))
        joined = "\n".join(lines)
        self.assertIn("benchmark_tools.place_trade", joined)
        self.assertIn("outcome=rejected", joined)
        self.assertIn("risk_guard.reason=no_executable_price", joined)
        self.assertIn("trade.executable_price=1.0", joined)

    def test_a_fill_status_span_reaches_the_log(self):
        lines = "\n".join(self._export(_span(
            "benchmark_tools.place_trade",
            **{"trade.fill_status": "shadow_filled_partial"},
        )))
        self.assertIn("trade.fill_status=shadow_filled_partial", lines)

    def test_ordinary_spans_are_not_mirrored(self):
        # Mirroring everything would bury the handful of decisions under
        # fetch/parse noise, which is the same as losing them.
        lines = "\n".join(self._export(
            _span("market_data.fetch", **{"server.address": "api.kalshi.com"}),
            _span("rag.embed", **{"chunk.count": 12}),
        ))
        self.assertNotIn("market_data.fetch", lines)
        self.assertNotIn("rag.embed", lines)

    def test_a_span_without_attributes_is_survivable(self):
        # Nothing here may raise: an exporter that throws takes the span
        # processor down with it.
        self.assertIsNotNone(
            self.exporter.export([SimpleNamespace(name="bare", attributes=None)]))

    def test_shutdown_and_flush_are_no_ops(self):
        self.assertIsNone(self.exporter.shutdown())
        self.assertTrue(self.exporter.force_flush())


class DecisionSpanLoggerWiringTests(unittest.TestCase):
    def test_the_mirror_can_be_switched_off(self):
        # On by default, but an operator can silence local decision mirroring.
        self.assertIn("FORESEA_LOG_DECISION_SPANS", Path(
            observability.__file__).read_text(encoding="utf-8"))

    def test_it_uses_a_simple_processor_not_a_batched_one(self):
        # A tick is short-lived: a batched processor can exit before flushing
        # and drop the last decision of a cycle, usually the interesting one.
        source = Path(observability.__file__).read_text(encoding="utf-8")
        self.assertIn("SimpleSpanProcessor(_DecisionSpanLogger())", source)

    def test_no_paid_superlog_connector_or_public_token_remains(self):
        source = Path(observability.__file__).read_text(encoding="utf-8").lower()
        self.assertNotIn("superlog", source)
        self.assertNotIn("intake.", source)
        self.assertNotIn("sl_public_", source)


if __name__ == "__main__":
    unittest.main()
