"""Size a claimed edge by how reliable edges that size have actually been.

The published track record measures, per size of disagreement with the
market, how the model's forecasts scored against the market's:

    edge bucket     n    model Brier  market Brier
    20pp+         185        0.504        0.206
    10-20pp        51        0.427        0.314
    5-10pp        102        0.169        0.155
    0-5pp        3251        0.095        0.095

The further the model strays from the market, the less reliable it is. A
large disagreement is also exactly what an agent reads as a large edge, and
Kelly used to shrink every claim by one fixed fraction regardless of size,
so the largest stakes went on the least reliable calls. The fleet's opens
claiming 20pp+ won 30% and lost about $50 each.

Sizing now keeps (1 - policy shrinkage) x reliability of the model's
disagreement, where reliability = market_brier / model_brier capped at 1.
Near the market that is the old sizing exactly. At 20pp+ it keeps about 41%
of what the policy kept. It never reaches zero, so the agents keep trading --
the decision was to shrink stakes rather than refuse trades.

The record below is frozen, so these tests do not move when a new one is
published.
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
    {"edge_bucket": "20pp+", "n": 185, "model_brier": 0.5036, "market_brier": 0.2059},
    {"edge_bucket": "10-20pp", "n": 51, "model_brier": 0.4267, "market_brier": 0.3140},
    {"edge_bucket": "5-10pp", "n": 102, "model_brier": 0.1691, "market_brier": 0.1547},
    {"edge_bucket": "0-5pp", "n": 3251, "model_brier": 0.0950, "market_brier": 0.0949},
]


def reliability(edge, calibration=PUBLISHED):
    return benchmark_tools._edge_reliability(edge, calibration)


def sizing(model_probability, price=0.50, mode="quarter_kelly", calibration=PUBLISHED):
    benchmark_tools._EDGE_CALIBRATION_CACHE["rows"] = calibration
    try:
        return benchmark_tools._sizing_plan(
            {"sizing_mode": mode, "model_probability": model_probability},
            price=price, side="yes", account_value=10_000.0,
        )
    finally:
        benchmark_tools._EDGE_CALIBRATION_CACHE.clear()


class ReliabilityTests(unittest.TestCase):
    def test_the_weight_is_the_market_brier_over_the_model_brier(self):
        self.assertAlmostEqual(reliability(0.40)["weight"], 0.2059 / 0.5036, places=5)
        self.assertAlmostEqual(reliability(0.15)["weight"], 0.3140 / 0.4267, places=5)
        self.assertAlmostEqual(reliability(0.07)["weight"], 0.1547 / 0.1691, places=5)

    def test_bigger_claims_are_trusted_less(self):
        weights = [reliability(e)["weight"] for e in (0.03, 0.07, 0.15, 0.40)]
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_it_never_reaches_zero(self):
        """Shrink, not refuse: a positive claim keeps some weight."""
        self.assertGreater(reliability(0.40)["weight"], 0.0)

    def test_it_is_never_used_to_amplify_a_claim(self):
        better = [{"edge_bucket": "20pp+", "n": 185, "model_brier": 0.10, "market_brier": 0.30}]
        self.assertEqual(reliability(0.40, better)["weight"], 1.0)

    def test_the_boundaries_bucket_exactly_as_the_record_does(self):
        for edge in (0.0999999, 0.10, 0.1999999, 0.20, 0.05, 0.0499999):
            with self.subTest(edge=edge):
                self.assertEqual(reliability(edge)["edge_bucket"], _edge_label(edge))

    def test_a_negative_edge_is_judged_by_its_size(self):
        self.assertEqual(reliability(-0.30)["edge_bucket"], "20pp+")

    def test_without_enough_evidence_the_weight_is_neutral(self):
        thin = [dict(PUBLISHED[0], n=12)]
        self.assertEqual(reliability(0.40, thin)["weight"], 1.0)
        self.assertEqual(reliability(0.40, None)["weight"], 1.0)
        self.assertEqual(reliability(0.40, [])["weight"], 1.0)
        self.assertEqual(reliability(0.40, [PUBLISHED[3]])["reason"], "bucket_not_published")

    def test_a_zero_model_brier_does_not_divide(self):
        degenerate = [dict(PUBLISHED[0], model_brier=0.0)]
        self.assertEqual(reliability(0.40, degenerate)["weight"], 1.0)


class SizingTests(unittest.TestCase):
    def test_a_near_market_claim_is_sized_as_before(self):
        """Where the model matches the market, nothing changes."""
        calibrated = sizing(0.53)
        uncalibrated = sizing(0.53, calibration=None)
        self.assertAlmostEqual(calibrated["target_notional"], uncalibrated["target_notional"], delta=0.5)

    def test_a_large_claim_is_sized_down(self):
        calibrated = sizing(0.90)
        uncalibrated = sizing(0.90, calibration=None)
        self.assertLess(calibrated["target_notional"], uncalibrated["target_notional"])

    def test_a_large_claim_still_trades(self):
        plan = sizing(0.90)
        self.assertTrue(plan["eligible"])
        self.assertGreater(plan["target_notional"], 0.0)

    def test_the_kept_share_is_the_policy_share_times_reliability(self):
        plan = sizing(0.90)
        # The stated 40pp edge is capped at the 10pp credible ceiling for
        # sizing, so reliability comes from the 10-20pp bucket.
        weight = 0.3140 / 0.4267
        self.assertAlmostEqual(plan["effective_market_shrinkage"], 1.0 - 0.5 * weight, places=5)
        self.assertEqual(plan["market_shrinkage"], 0.5)

    def test_the_evidence_is_returned_with_the_plan(self):
        detail = sizing(0.90)["calibration_reliability"]
        # Sizing uses the capped edge (10pp -> 10-20pp bucket); the stated
        # edge remains visible in plan["edge"].
        self.assertEqual(detail["edge_bucket"], "10-20pp")
        self.assertEqual(detail["n"], 51)

    def test_with_no_record_the_old_formula_is_reproduced(self):
        plan = sizing(0.90, calibration=None)
        self.assertAlmostEqual(plan["effective_market_shrinkage"], 0.5)

    def test_edge_kelly_is_tempered_too(self):
        """At 0.62 neither side reaches the 8% cap, so the comparison bites.

        At 0.90 both are held at the cap, calibrated or not: edge Kelly
        starts from only 25% shrinkage, so its largest claims still size to
        the maximum position. The cap, not this weighting, bounds those.
        """
        calibrated = sizing(0.62, mode="edge_kelly")
        uncalibrated = sizing(0.62, mode="edge_kelly", calibration=None)
        self.assertLess(uncalibrated["target_notional"], 800.0, "must be below the cap")
        self.assertLess(calibrated["target_notional"], uncalibrated["target_notional"])


class CredibleEdgeSizingCapTests(unittest.TestCase):
    """Sizing stakes the credible-edge ceiling, not a fantasy edge.

    Llama claimed 66-83pp edges on the Khamenei market; quarter-Kelly sized
    those fantasies into stakes the guard was always going to reject, and it
    re-attempted the same trade 26 times in one month. The guard holds
    entries to a 10pp credible ceiling, so sizing must not pretend the edge
    beyond it is real.
    """

    def test_a_fantasy_edge_is_sized_like_the_ceiling(self):
        fantasy = sizing(0.94, price=0.11, mode="quarter_kelly")   # 83pp claimed
        ceiling = sizing(0.21, price=0.11, mode="quarter_kelly")   # exactly 10pp
        self.assertGreater(fantasy["target_notional"], 0.0)
        self.assertAlmostEqual(fantasy["target_notional"], ceiling["target_notional"], delta=1.0)

    def test_the_stated_edge_is_still_reported_untouched(self):
        plan = sizing(0.94, price=0.11, mode="quarter_kelly")
        self.assertEqual(plan["edge"], 0.83)

    def test_a_legitimate_edge_is_not_capped(self):
        honest = sizing(0.18, price=0.11, mode="quarter_kelly")    # 7pp edge
        self.assertGreater(honest["target_notional"], 0.0)
        # 7pp < 10pp ceiling: sizing must reflect the full stated edge
        ceiling = sizing(0.21, price=0.11, mode="quarter_kelly")
        self.assertLess(honest["target_notional"], ceiling["target_notional"])

    def test_auto_mode_does_not_escalate_a_capped_edge(self):
        """A fantasy edge must not buy edge_kelly treatment via auto sizing."""
        plan = sizing(0.94, price=0.11, mode="auto")
        self.assertNotEqual(plan["mode"], "edge_kelly")


class ProbabilityBiasCorrectionTests(unittest.TestCase):
    """The tick feeds each agent's measured calibration bias back into sizing.

    A positive measured bias means the agent systematically overestimates YES;
    the correction shifts its stated P(YES) down before Kelly sizes the stake.
    The raw stated probability stays in the audit trail.
    """

    def _sizing_with_bias(self, model_probability, bias, price=0.50, mode="quarter_kelly"):
        benchmark_tools._EDGE_CALIBRATION_CACHE["rows"] = PUBLISHED
        try:
            with mock.patch.dict(os.environ, {"FORESEA_AGENT_PROBABILITY_BIAS": str(bias)}):
                return benchmark_tools._sizing_plan(
                    {"sizing_mode": mode, "model_probability": model_probability},
                    price=price, side="yes", account_value=10_000.0,
                )
        finally:
            benchmark_tools._EDGE_CALIBRATION_CACHE.clear()

    def test_a_positive_bias_shifts_stated_probability_down(self):
        # deepseek-v4-flash: measured bias +0.219. Stated P(YES)=0.60 on a
        # 0.50 market is a 10pp edge raw; corrected it is -0.119 -> no trade.
        raw = sizing(0.60)
        corrected = self._sizing_with_bias(0.60, 0.219)
        self.assertTrue(raw["eligible"])
        self.assertFalse(corrected["eligible"])
        self.assertEqual(corrected["reason"], "edge_below_threshold")

    def test_the_correction_is_bounded_so_it_cannot_invert_the_probability(self):
        # A bias beyond +-0.5 would invert the probability's sign (0.90 - 5.0
        # = -4.1), which is a measurement failure, not a calibration signal.
        # The clamp bounds the correction; the corrected probability can
        # still fall below the price, which correctly makes the trade
        # ineligible rather than crashing or trading an inverted probability.
        corrected = self._sizing_with_bias(0.90, 5.0)
        self.assertFalse(corrected["eligible"])
        self.assertEqual(corrected["reason"], "edge_below_threshold")

    def test_a_clamped_bias_still_sizes_a_genuinely_high_claim(self):
        # Even with the clamp at 0.5, a very high stated probability on a
        # cheap contract keeps a positive corrected edge and trades.
        corrected = self._sizing_with_bias(0.99, 5.0, price=0.11)
        self.assertTrue(corrected["eligible"])
        self.assertGreater(corrected["target_notional"], 0.0)

    def test_no_bias_env_var_means_no_correction(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORESEA_AGENT_PROBABILITY_BIAS", None)
            benchmark_tools._EDGE_CALIBRATION_CACHE["rows"] = PUBLISHED
            try:
                plan = benchmark_tools._sizing_plan(
                    {"sizing_mode": "quarter_kelly", "model_probability": 0.60},
                    price=0.50, side="yes", account_value=10_000.0,
                )
            finally:
                benchmark_tools._EDGE_CALIBRATION_CACHE.clear()
        self.assertTrue(plan["eligible"])

    def test_a_negative_bias_shifts_stated_probability_up(self):
        # An agent that systematically underestimates YES gets shifted up.
        # P(YES)=0.40 on a 0.50 market is a -10pp edge raw; corrected by
        # -0.15 it becomes 0.55 -> +5pp edge -> a larger stake.
        raw = sizing(0.40)
        corrected = self._sizing_with_bias(0.40, -0.15)
        self.assertFalse(raw["eligible"], "raw -10pp edge must not trade")
        self.assertTrue(corrected["eligible"], "corrected +5pp edge should trade")


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

    def test_an_unreadable_record_leaves_sizing_alone_and_says_so(self):
        missing = str(Path(tempfile.gettempdir()) / "no-such-dir-foresea" / "missing.json")
        with mock.patch.dict(os.environ, {"FORESEA_TRACK_RECORD_PATH": missing}):
            with self.assertLogs(benchmark_tools.logger.name, "WARNING"):
                self.assertIsNone(benchmark_tools._published_edge_calibration())


class PlaceTradeTests(unittest.TestCase):
    """End to end: a big claim still fills, at a smaller stake."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(benchmark_tools._EDGE_CALIBRATION_CACHE.clear)

    def _stake(self, calibration, ticker):
        benchmark_tools._EDGE_CALIBRATION_CACHE["rows"] = calibration
        folder = Path(self.tmp.name) / ticker
        folder.mkdir()
        env = {
            "FORESEA_AGENT_TOOL_LEDGER_PATH": str(folder / "ledger.jsonl"),
            "FORESEA_AGENT_ACCOUNT_DB_PATH": str(folder / "accounts.sqlite"),
            "FORESEA_AGENT_ACCOUNT_VALUE": "10000",
            "FORESEA_AGENT_PLACE_TRADE_MODE": "shadow",
            "FORESEA_MAX_ORDER_NOTIONAL": "1000",
            # The claim sits in the 10-20pp bucket on purpose, to exercise the
            # reliability weight; 25pp keeps the ceiling running without refusing it.
            "FORESEA_AGENT_MAX_CREDIBLE_EDGE": "0.25",
        }
        ctx = benchmark_tools.ToolContext(agent_id="model-reliability", require_kelly_sizing=True)
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch(
                "analyzing_llm_rationale.market_data.fetch_kalshi",
                side_effect=_fetch_kalshi_quotes({ticker: 0.50}),
            ),
        ):
            return benchmark_tools.place_trade(
                {"ticker": ticker, "side": "yes", "price": 0.50, "quantity": 1,
                 "sizing_mode": "quarter_kelly", "model_probability": 0.62},
                ctx,
            )

    def test_a_claim_in_an_unreliable_bucket_fills_smaller(self):
        calibrated = self._stake(PUBLISHED, "KXCAL")
        uncalibrated = self._stake(None, "KXRAW")
        self.assertTrue(calibrated["ok"], calibrated.get("risk_guard", {}).get("reasons"))
        self.assertTrue(uncalibrated["ok"], uncalibrated.get("risk_guard", {}).get("reasons"))
        self.assertLess(
            calibrated["normalized_order"]["quantity"],
            uncalibrated["normalized_order"]["quantity"],
        )
        self.assertEqual(calibrated["sizing"]["calibration_reliability"]["edge_bucket"], "10-20pp")


if __name__ == "__main__":
    unittest.main()
