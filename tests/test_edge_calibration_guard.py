"""Hold the agents to what the published track record says about big edges.

The track record measures, per size of disagreement with the market,
whether the model beat it. It does not:

    edge bucket     n    model Brier  market Brier   skill    95% CI
    20pp+         185        0.504        0.206     -0.298   [-0.369, -0.227]
    10-20pp        51        0.427        0.314     -0.113   [-0.156, -0.069]
    5-10pp        102        0.169        0.155     -0.014   [-0.027, -0.002]
    0-5pp        3251        0.095        0.095     -0.000   [-0.000,  0.000]

The larger the disagreement, the more wrong the model -- and a large
disagreement is exactly what an agent treats as a large edge. The fleet's
own trades show the same gradient: entries claiming 20pp+ won 30% and lost
about $50 each. Kelly sizing shrank every claim a fixed fraction toward the
market regardless of size, so a 53pp claim was still sized as 26pp.

The guard refuses new exposure only where the record is both large enough
(n >= 30) and *materially* negative (whole CI below -0.01). On today's
record that is 10pp and above. 5-10pp is statistically negative but by
about a point and a half of Brier, and refusing it would stop the fleet
trading on an effect too small to justify that. The fixture here is that
record, frozen, so these tests do not move when a new one is published.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzing_llm_rationale import benchmark_tools  # noqa: E402
from analyzing_llm_rationale.track_record_live import _edge_label  # noqa: E402
from tests.test_benchmark_tools import _fetch_kalshi_quotes  # noqa: E402

PUBLISHED = [
    {"edge_bucket": "20pp+", "n": 185, "skill_vs_market": -0.2976,
     "skill_ci_low": -0.3688, "skill_ci_high": -0.2265},
    {"edge_bucket": "10-20pp", "n": 51, "skill_vs_market": -0.1128,
     "skill_ci_low": -0.1562, "skill_ci_high": -0.0693},
    {"edge_bucket": "5-10pp", "n": 102, "skill_vs_market": -0.0144,
     "skill_ci_low": -0.0265, "skill_ci_high": -0.0023},
    {"edge_bucket": "0-5pp", "n": 3251, "skill_vs_market": -0.0002,
     "skill_ci_low": -0.0003, "skill_ci_high": 0.0},
]


def verdict(edge, calibration=PUBLISHED):
    return benchmark_tools._edge_calibration_verdict(edge, calibration)


class VerdictTests(unittest.TestCase):
    def test_a_twenty_point_claim_is_refused(self):
        v = verdict(0.40)
        self.assertTrue(v["checked"])
        self.assertTrue(v["refuses"])
        self.assertEqual(v["edge_bucket"], "20pp+")

    def test_a_ten_to_twenty_point_claim_is_refused(self):
        self.assertTrue(verdict(0.15)["refuses"])

    def test_a_five_to_ten_point_claim_is_allowed(self):
        """Statistically negative, but not materially: CI high -0.0023."""
        v = verdict(0.07)
        self.assertTrue(v["checked"])
        self.assertFalse(v["refuses"])

    def test_a_near_market_claim_is_allowed(self):
        self.assertFalse(verdict(0.03)["refuses"])

    def test_the_boundaries_bucket_exactly_as_the_record_does(self):
        """Reusing _edge_label means the guard cannot drift from the record."""
        for edge in (0.0999999, 0.10, 0.1999999, 0.20, 0.05, 0.0499999):
            with self.subTest(edge=edge):
                self.assertEqual(verdict(edge)["edge_bucket"], _edge_label(edge))

    def test_a_negative_edge_is_judged_by_its_size(self):
        self.assertEqual(verdict(-0.30)["edge_bucket"], "20pp+")

    def test_a_thin_bucket_is_not_judged(self):
        thin = [dict(PUBLISHED[0], n=12)]
        v = verdict(0.40, thin)
        self.assertFalse(v["checked"])
        self.assertEqual(v["reason"], "insufficient_sample")

    def test_no_record_means_no_verdict_rather_than_a_refusal(self):
        self.assertFalse(verdict(0.40, None)["checked"])
        self.assertFalse(verdict(0.40, [])["checked"])
        self.assertEqual(verdict(0.40, [PUBLISHED[3]])["reason"], "bucket_not_published")

    def test_a_bucket_that_earns_skill_stops_being_refused(self):
        """No threshold is hard-coded per bucket: it follows the record."""
        improved = [dict(PUBLISHED[0], skill_vs_market=0.02,
                         skill_ci_low=-0.004, skill_ci_high=0.044)]
        self.assertFalse(verdict(0.40, improved)["refuses"])

    def test_the_materiality_line_is_strict(self):
        line = benchmark_tools._MATERIAL_SKILL_DEFICIT
        self.assertFalse(verdict(0.40, [dict(PUBLISHED[0], skill_ci_high=-line)])["refuses"])
        self.assertTrue(verdict(0.40, [dict(PUBLISHED[0], skill_ci_high=-line - 1e-6)])["refuses"])


class LoaderTests(unittest.TestCase):
    def setUp(self):
        benchmark_tools._EDGE_CALIBRATION_CACHE.clear()
        self.addCleanup(benchmark_tools._EDGE_CALIBRATION_CACHE.clear)

    def _record(self, td, rows):
        path = Path(td) / "track_record_live.json"
        path.write_text(json.dumps({"by_edge": rows, "overall": {}}), encoding="utf-8")
        return path

    def test_it_reads_by_edge_from_the_published_record(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._record(td, PUBLISHED)
            with mock.patch.dict(os.environ, {"FORESEA_TRACK_RECORD_PATH": str(path)}):
                self.assertEqual(benchmark_tools._published_edge_calibration(), PUBLISHED)

    def test_it_is_read_once_per_process(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._record(td, PUBLISHED)
            with mock.patch.dict(os.environ, {"FORESEA_TRACK_RECORD_PATH": str(path)}):
                benchmark_tools._published_edge_calibration()
                self._record(td, [])
                self.assertEqual(benchmark_tools._published_edge_calibration(), PUBLISHED)

    def test_an_unreadable_record_disables_the_guard_and_says_so(self):
        missing = str(Path(tempfile.gettempdir()) / "no-such-dir-foresea" / "missing.json")
        with mock.patch.dict(os.environ, {"FORESEA_TRACK_RECORD_PATH": missing}):
            with self.assertLogs(benchmark_tools.logger.name, "WARNING"):
                self.assertIsNone(benchmark_tools._published_edge_calibration())


class PlaceTradeTests(unittest.TestCase):
    """End to end: the refusal reaches the agent with its evidence."""

    def setUp(self):
        benchmark_tools._EDGE_CALIBRATION_CACHE["rows"] = PUBLISHED
        self.addCleanup(benchmark_tools._EDGE_CALIBRATION_CACHE.clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = {
            "FORESEA_AGENT_TOOL_LEDGER_PATH": str(Path(self.tmp.name) / "ledger.jsonl"),
            "FORESEA_AGENT_ACCOUNT_DB_PATH": str(Path(self.tmp.name) / "accounts.sqlite"),
            "FORESEA_AGENT_ACCOUNT_VALUE": "10000",
            "FORESEA_AGENT_PLACE_TRADE_MODE": "shadow",
            "FORESEA_MAX_ORDER_NOTIONAL": "1000",
        }
        patcher = mock.patch.dict(os.environ, env, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _trade(self, ticker, quotes, **order):
        ctx = benchmark_tools.ToolContext(agent_id="model-calibration", require_kelly_sizing=True)
        with mock.patch(
            "analyzing_llm_rationale.market_data.fetch_kalshi",
            side_effect=_fetch_kalshi_quotes(quotes),
        ):
            return benchmark_tools.place_trade({"ticker": ticker, **order}, ctx)

    def test_a_forty_point_claim_is_refused_with_the_evidence(self):
        result = self._trade(
            "KXBIG", {"KXBIG": 0.50}, side="yes", price=0.50, quantity=1,
            sizing_mode="quarter_kelly", model_probability=0.90,
        )
        self.assertFalse(result["ok"])
        guard = result["risk_guard"]
        self.assertIn("edge_bucket_underperforms_market", guard["reasons"])
        self.assertEqual(guard["edge_calibration"]["edge_bucket"], "20pp+")
        self.assertEqual(guard["edge_calibration"]["n"], 185)

    def test_a_modest_claim_still_trades(self):
        result = self._trade(
            "KXMODEST", {"KXMODEST": 0.50}, side="yes", price=0.50, quantity=1,
            sizing_mode="quarter_kelly", model_probability=0.58,
        )
        self.assertTrue(result["ok"], result.get("risk_guard", {}).get("reasons"))
        self.assertNotIn("edge_bucket_underperforms_market", result["risk_guard"]["reasons"])

    def test_a_close_is_never_refused_by_the_record(self):
        """Reducing risk must stay available whatever the record says."""
        quotes = {"KXEXIT": {"yes_ask": 0.50, "no_ask": 0.52}}
        opened = self._trade(
            "KXEXIT", quotes, side="yes", price=0.50, quantity=1,
            sizing_mode="quarter_kelly", model_probability=0.58,
        )
        self.assertTrue(opened["ok"], opened.get("risk_guard", {}).get("reasons"))
        closed = self._trade(
            "KXEXIT", quotes, side="no", price=0.52,
            quantity=opened["normalized_order"]["quantity"],
            sizing_mode="close", model_probability=0.05,
        )
        self.assertNotIn(
            "edge_bucket_underperforms_market",
            (closed.get("risk_guard") or {}).get("reasons", []),
        )


if __name__ == "__main__":
    unittest.main()
