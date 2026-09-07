"""Polymarket settlements were charged a Kalshi settlement fee.

`_order_fee` has always taken a platform and returns 0 off Kalshi, because
Polymarket's CLOB charges no per-trade fee. `_settlement_fee_rate` took no
platform at all and returned a flat 1.4% of payout for every venue.

The published audits show it happening. Across 228 records, Polymarket
taker fees are correctly 0.0000 -- and its settlements carry 15.40:

    gpt-oss-120b   qty  100   settlement_fee  1.40
    llama-3.3-70b  qty 1000   settlement_fee 14.00

That is 0.014 per contract, the Kalshi rate, on a venue whose resolution
is free. It overstates cost and understates the agent's realised PnL, and
it grows with Polymarket settlement volume -- which is over half the
traded notional in that window.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import benchmark_tools  # noqa: E402


class SettlementFeeIsVenueAwareTests(unittest.TestCase):
    def test_kalshi_still_pays_a_settlement_fee(self):
        self.assertEqual(
            benchmark_tools._settlement_fee_rate("kalshi"),
            benchmark_tools.DEFAULT_SETTLEMENT_FEE_RATE,
        )

    def test_polymarket_pays_none(self):
        self.assertEqual(benchmark_tools._settlement_fee_rate("polymarket"), 0.0)

    def test_the_venue_name_is_matched_loosely(self):
        """Platform strings arrive from several places and vary in case."""
        for name in ("Kalshi", "KALSHI", " kalshi "):
            with self.subTest(name=name):
                self.assertGreater(benchmark_tools._settlement_fee_rate(name), 0.0)
        for name in ("Polymarket", "POLYMARKET", " polymarket ", "", None):
            with self.subTest(name=name):
                self.assertEqual(benchmark_tools._settlement_fee_rate(name), 0.0)

    def test_the_default_is_still_kalshi(self):
        """No caller loses its fee by omitting the argument."""
        self.assertEqual(
            benchmark_tools._settlement_fee_rate(),
            benchmark_tools.DEFAULT_SETTLEMENT_FEE_RATE,
        )

    def test_the_env_override_still_applies_to_kalshi(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_SETTLEMENT_FEE_RATE": "0.05"}):
            self.assertAlmostEqual(benchmark_tools._settlement_fee_rate("kalshi"), 0.05)

    def test_the_env_override_does_not_resurrect_a_polymarket_fee(self):
        with mock.patch.dict(os.environ, {"FORESEA_AGENT_SETTLEMENT_FEE_RATE": "0.05"}):
            self.assertEqual(benchmark_tools._settlement_fee_rate("polymarket"), 0.0)

    def test_the_published_charges_would_now_be_zero(self):
        """The two settlements that were actually charged."""
        for quantity, charged in ((100.0, 1.40), (1000.0, 14.00)):
            with self.subTest(quantity=quantity):
                self.assertAlmostEqual(
                    quantity * benchmark_tools._settlement_fee_rate("kalshi"),
                    charged, places=6,
                )
                self.assertEqual(
                    quantity * benchmark_tools._settlement_fee_rate("polymarket"), 0.0
                )


class BothSettlementPathsPassTheVenueTests(unittest.TestCase):
    """Two settlement paths exist -- sqlite and Datastore -- and both charged."""

    def _rate_calls(self):
        import ast

        source = Path(benchmark_tools.__file__).read_text(encoding="utf-8", errors="replace")
        calls = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_settlement_fee_rate":
                calls.append((node.lineno, len(node.args) + len(node.keywords)))
        return calls

    def test_the_scan_finds_both_paths(self):
        self.assertGreaterEqual(len(self._rate_calls()), 2)

    def test_no_call_site_omits_the_venue(self):
        bare = [line for line, argc in self._rate_calls() if argc == 0]
        self.assertEqual(
            bare, [],
            f"these settle at the Kalshi rate whatever the venue: lines {bare}",
        )


if __name__ == "__main__":
    unittest.main()
