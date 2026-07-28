from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.market_data import (  # noqa: E402
    MarketDataError,
    fetch_kalshi,
    fetch_polymarket,
    list_kalshi,
    list_polymarket,
    resolve_kalshi,
    resolve_polymarket,
)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def _fake_requests(payload, status=200, capture=None):
    def fake_get(url, params=None, headers=None, timeout=None):
        if capture is not None:
            capture.append((url, params))
        return FakeResponse(payload, status)

    return SimpleNamespace(get=fake_get)


class MarketDataTests(unittest.TestCase):
    def setUp(self):
        self._orig = sys.modules.get("requests")

    def tearDown(self):
        if self._orig is None:
            sys.modules.pop("requests", None)
        else:
            sys.modules["requests"] = self._orig

    def test_polymarket_parses_json_encoded_outcomes(self):
        payload = [{
            "question": "Will X happen?",
            "slug": "will-x",
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.62", "0.38"]',
            "description": "Resolves Yes if X happens before the deadline.",
            "events": [{
                "description": "Background on X.",
                "eventMetadata": {"context_description": "Latest context about X."},
            }],
        }]
        capture = []
        sys.modules["requests"] = _fake_requests(payload, capture=capture)

        quote = fetch_polymarket(slug="will-x")

        self.assertEqual(quote["platform"], "Polymarket")
        self.assertEqual(quote["outcome"], "Yes")
        self.assertAlmostEqual(quote["probability"], 0.62)
        self.assertEqual(quote["market_url"], "https://polymarket.com/market/will-x")
        self.assertEqual(len(quote["outcomes"]), 2)
        self.assertEqual(
            quote["resolution_criteria"],
            "Resolves Yes if X happens before the deadline.",
        )
        self.assertEqual(quote["description"], "Latest context about X.")
        self.assertEqual(capture[0][1]["slug"], "will-x")

    def test_polymarket_requires_identifier(self):
        with self.assertRaises(MarketDataError):
            fetch_polymarket()

    def test_polymarket_empty_result_is_not_found(self):
        sys.modules["requests"] = _fake_requests([])
        with self.assertRaises(MarketDataError):
            fetch_polymarket(slug="missing")

    def test_kalshi_reads_dollar_price(self):
        payload = {"market": {"ticker": "KXTEST", "title": "Will Y?", "last_price_dollars": "0.42"}}
        sys.modules["requests"] = _fake_requests(payload)

        quote = fetch_kalshi("kxtest")

        self.assertEqual(quote["platform"], "Kalshi")
        self.assertAlmostEqual(quote["probability"], 0.42)
        self.assertEqual(quote["outcome"], "Yes")
        # Web URL is rooted on the (lowercase) series ticker, not the event ticker.
        self.assertEqual(quote["market_url"], "https://kalshi.com/markets/kxtest")
        self.assertAlmostEqual(quote["outcomes"][1]["probability"], 0.58)

    def test_kalshi_url_uses_series_ticker_not_event_ticker(self):
        payload = {"market": {
            "ticker": "KXMEDIARELEASEPRISONBREAK-30JAN01-26JUL01",
            "event_ticker": "KXMEDIARELEASEPRISONBREAK-30JAN01",
            "title": "When will Prison Break return?",
            "last_price_dollars": "0.07",
        }}
        sys.modules["requests"] = _fake_requests(payload)
        quote = fetch_kalshi("KXMEDIARELEASEPRISONBREAK-30JAN01-26JUL01")
        # Must drop the event's -30JAN01 date suffix and lowercase.
        self.assertEqual(quote["market_url"], "https://kalshi.com/markets/kxmediareleaseprisonbreak")

    def test_kalshi_uses_bid_ask_midpoint_without_last_price(self):
        payload = {"market": {"ticker": "KXT", "title": "t",
                              "last_price_dollars": None, "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.50"}}
        sys.modules["requests"] = _fake_requests(payload)

        quote = fetch_kalshi("KXT")

        self.assertAlmostEqual(quote["probability"], 0.45)

    def test_kalshi_requires_ticker(self):
        with self.assertRaises(MarketDataError):
            fetch_kalshi("")

    def test_list_polymarket_keeps_priced_binary_only(self):
        payload = [
            {"question": "A?", "slug": "a", "outcomes": '["Yes","No"]', "outcomePrices": '["0.3","0.7"]'},
            {"question": "B (no price)", "slug": "b", "outcomes": '["Yes","No"]', "outcomePrices": '[]'},
            {"question": "C (multi)", "slug": "c", "outcomes": '["X","Y","Z"]', "outcomePrices": '["0.3","0.3","0.4"]'},
        ]
        sys.modules["requests"] = _fake_requests(payload)
        quotes = list_polymarket(limit=10)
        self.assertEqual([q["question"] for q in quotes], ["A?"])
        self.assertAlmostEqual(quotes[0]["probability"], 0.3)

    def test_list_kalshi_via_events_skips_unpriced_and_mve(self):
        payload = {"events": [{"title": "Event One",
            "sub_title": "Event background.",
            "settlement_sources": [{"name": "Official source", "url": "https://official.example"}],
            "markets": [
            {"ticker": "T1", "title": "One", "last_price_dollars": "0.60",
             "close_time": "2026-12-01T00:00:00Z",
             "rules_primary": "Resolves Yes when the official source reports one."},
            {"ticker": "T2", "title": "Two", "last_price_dollars": None,
             "yes_bid_dollars": None, "yes_ask_dollars": None},
            {"ticker": "T3", "title": "Parlay", "last_price_dollars": "0.50",
             "mve_collection_ticker": "KXMVE-X"},
        ]}]}
        sys.modules["requests"] = _fake_requests(payload)
        quotes = list_kalshi(limit=10)
        # Unpriced (T2) and MVE parlay (T3) dropped; only the real priced binary kept.
        self.assertEqual([q["question"] for q in quotes], ["Event One"])
        self.assertAlmostEqual(quotes[0]["probability"], 0.6)
        self.assertEqual(
            quotes[0]["resolution_criteria"],
            "Resolves Yes when the official source reports one.",
        )
        self.assertIn("https://official.example", quotes[0]["resolution_source"])

    def test_list_polymarket_keyword_filter(self):
        payload = [
            {"question": "Will the Lakers win the NBA title?", "slug": "nba1",
             "outcomes": '["Yes","No"]', "outcomePrices": '["0.4","0.6"]'},
            {"question": "Will the Fed cut rates?", "slug": "fed",
             "outcomes": '["Yes","No"]', "outcomePrices": '["0.5","0.5"]'},
        ]
        sys.modules["requests"] = _fake_requests(payload)
        quotes = list_polymarket(limit=10, query="nba")
        self.assertEqual([q["question"] for q in quotes], ["Will the Lakers win the NBA title?"])

    def test_list_kalshi_keyword_filter(self):
        payload = {"events": [
            {"title": "Will an NBA team relocate?", "markets": [
                {"ticker": "NBA1", "last_price_dollars": "0.45", "close_time": "2026-12-01T00:00:00Z"}]},
            {"title": "Will CPI exceed 3%?", "markets": [
                {"ticker": "CPI1", "last_price_dollars": "0.50", "close_time": "2026-12-01T00:00:00Z"}]},
        ]}
        sys.modules["requests"] = _fake_requests(payload)
        quotes = list_kalshi(limit=10, query="nba")
        self.assertEqual([q["question"] for q in quotes], ["Will an NBA team relocate?"])

    def test_list_kalshi_horizon_window(self):
        payload = {"events": [{"title": "soon vs far", "markets": [
            {"ticker": "SOON", "yes_sub_title": "soon", "last_price_dollars": "0.50",
             "close_time": "2099-01-01T00:00:00Z"},   # ~decades out -> excluded by max
        ]}]}
        sys.modules["requests"] = _fake_requests(payload)
        self.assertEqual(list_kalshi(limit=10, min_close_days=2, max_close_days=365), [])

    def test_resolve_polymarket_yes_no(self):
        # Resolved YES (Yes price settled to 1).
        sys.modules["requests"] = _fake_requests([
            {"closed": True, "umaResolutionStatus": "resolved",
             "outcomes": '["Yes","No"]', "outcomePrices": '["1","0"]'}
        ])
        self.assertEqual(resolve_polymarket("slug"), 1)
        # Resolved NO.
        sys.modules["requests"] = _fake_requests([
            {"closed": True, "umaResolutionStatus": "resolved",
             "outcomes": '["Yes","No"]', "outcomePrices": '["0","1"]'}
        ])
        self.assertEqual(resolve_polymarket("slug"), 0)
        # Still open -> None.
        sys.modules["requests"] = _fake_requests([
            {"closed": False, "umaResolutionStatus": "",
             "outcomes": '["Yes","No"]', "outcomePrices": '["0.5","0.5"]'}
        ])
        self.assertIsNone(resolve_polymarket("slug"))

    def test_resolve_kalshi_settled(self):
        sys.modules["requests"] = _fake_requests({"market": {"status": "finalized", "result": "yes"}})
        self.assertEqual(resolve_kalshi("T1"), 1)
        sys.modules["requests"] = _fake_requests({"market": {"status": "settled", "result": "no"}})
        self.assertEqual(resolve_kalshi("T1"), 0)
        sys.modules["requests"] = _fake_requests({"market": {"status": "active", "result": ""}})
        self.assertIsNone(resolve_kalshi("T1"))


if __name__ == "__main__":
    unittest.main()
