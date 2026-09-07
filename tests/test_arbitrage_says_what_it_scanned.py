"""No arbitrage and no data must not look the same.

/v1/arbitrage/cross-venue answered

    {"timestamp": ..., "n_opportunities": 0, "opportunities": []}

whether both venues were read and simply disagreed about nothing, or the
fetch raised and nothing was compared at all. The fetch failure was
swallowed by a bare `except Exception` that turned each venue into an empty
list, so a scanner that could not reach either venue reported a clean scan.

Checked live on 2026-09-07: both fetchers do work -- 28 Polymarket and 30
Kalshi rows, all with probabilities -- so the zero on the endpoint is real.
This is about being able to tell, not about a broken fetch.

Same shape as the min_credibility filter that silently stopped filtering,
which #531 fixed by saying so in the response.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import arbitrage_scanner, market_data  # noqa: E402


def _venues(poly_fn, kalshi_fn):
    """Patch the real fetchers, not the module object.

    The scanner does `from analyzing_llm_rationale import market_data`, which
    reads the attribute already bound on the package. Replacing the entry in
    the module cache is therefore ignored once anything else in the suite has
    imported it -- the first version of these tests did that, passed when run
    alone, and failed under discover.
    """
    return (
        mock.patch.object(market_data, "list_polymarket", poly_fn),
        mock.patch.object(market_data, "list_kalshi", kalshi_fn),
    )


class FetchReportsReachabilityTests(unittest.TestCase):
    def test_a_successful_fetch_is_reported_reachable(self):
        poly_p, kalshi_p = _venues(
            lambda limit: [{"question": "a", "probability": 0.5}],
            lambda limit: [{"question": "a", "probability": 0.6}],
        )
        with poly_p, kalshi_p:
            poly, kalshi, reachable = arbitrage_scanner.fetch_markets_to_scan()
        self.assertTrue(reachable)
        self.assertEqual(len(poly), 1)
        self.assertEqual(len(kalshi), 1)

    def test_a_failed_fetch_is_reported_unreachable_and_logged(self):
        def boom(limit):
            raise RuntimeError("venue down")

        poly_p, kalshi_p = _venues(boom, boom)
        with poly_p, kalshi_p:
            with self.assertLogs(arbitrage_scanner.logger, level="WARNING") as caught:
                poly, kalshi, reachable = arbitrage_scanner.fetch_markets_to_scan()

        self.assertFalse(reachable)
        self.assertEqual((poly, kalshi), ([], []))
        self.assertIn("fetch failed", "\n".join(caught.output))

    def test_it_degrades_rather_than_raising(self):
        """A scanner returning nothing beats one that 500s."""
        def boom(limit):
            raise RuntimeError("venue down")

        poly_p, kalshi_p = _venues(boom, boom)
        with poly_p, kalshi_p:
            with self.assertLogs(arbitrage_scanner.logger, level="WARNING"):
                arbitrage_scanner.fetch_markets_to_scan()  # must not raise


class TheEndpointSaysWhatItComparedTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from analyzing_llm_rationale import server as server_module

        self.server = server_module
        self.client = TestClient(server_module.app)

    def _get(self, poly, kalshi, reachable):
        with mock.patch.object(
            self.server, "_check_rate_limit", lambda request: None,
        ), mock.patch(
            "analyzing_llm_rationale.arbitrage_scanner.fetch_markets_to_scan",
            return_value=(poly, kalshi, reachable),
        ):
            return self.client.get("/v1/arbitrage/cross-venue").json()

    def test_a_real_scan_with_no_hits_says_how_much_it_compared(self):
        body = self._get(
            [{"question": "will alpha happen", "probability": 0.50}],
            [{"question": "will beta happen", "probability": 0.90}],
            True,
        )
        self.assertEqual(body["n_opportunities"], 0)
        self.assertEqual(body["markets_scanned"], {"polymarket": 1, "kalshi": 1})
        self.assertTrue(body["venues_reachable"])

    def test_an_unreachable_venue_is_distinguishable_from_no_arbitrage(self):
        body = self._get([], [], False)
        self.assertEqual(body["n_opportunities"], 0)
        self.assertEqual(body["markets_scanned"], {"polymarket": 0, "kalshi": 0})
        self.assertFalse(body["venues_reachable"])

    def test_the_two_cases_differ_in_the_response(self):
        """The point of the change, stated as one assertion."""
        scanned = self._get(
            [{"question": "will alpha happen", "probability": 0.5}],
            [{"question": "will beta happen", "probability": 0.9}],
            True,
        )
        blind = self._get([], [], False)
        for key in ("markets_scanned", "venues_reachable"):
            self.assertNotEqual(scanned[key], blind[key])


if __name__ == "__main__":
    unittest.main()
