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
        }]
        capture = []
        sys.modules["requests"] = _fake_requests(payload, capture=capture)

        quote = fetch_polymarket(slug="will-x")

        self.assertEqual(quote["platform"], "Polymarket")
        self.assertEqual(quote["outcome"], "Yes")
        self.assertAlmostEqual(quote["probability"], 0.62)
        self.assertEqual(quote["market_url"], "https://polymarket.com/market/will-x")
        self.assertEqual(len(quote["outcomes"]), 2)
        self.assertEqual(capture[0][1]["slug"], "will-x")

    def test_polymarket_requires_identifier(self):
        with self.assertRaises(MarketDataError):
            fetch_polymarket()

    def test_polymarket_empty_result_is_not_found(self):
        sys.modules["requests"] = _fake_requests([])
        with self.assertRaises(MarketDataError):
            fetch_polymarket(slug="missing")

    def test_kalshi_converts_cents_to_probability(self):
        payload = {"market": {"ticker": "KXTEST", "title": "Will Y?", "last_price": 42}}
        sys.modules["requests"] = _fake_requests(payload)

        quote = fetch_kalshi("kxtest")

        self.assertEqual(quote["platform"], "Kalshi")
        self.assertAlmostEqual(quote["probability"], 0.42)
        self.assertEqual(quote["outcome"], "Yes")
        self.assertEqual(quote["market_url"], "https://kalshi.com/markets/KXTEST")
        self.assertAlmostEqual(quote["outcomes"][1]["probability"], 0.58)

    def test_kalshi_uses_bid_ask_midpoint_without_last_price(self):
        payload = {"market": {"ticker": "KXT", "title": "t", "last_price": None, "yes_bid": 40, "yes_ask": 50}}
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

    def test_list_kalshi_skips_unpriced(self):
        payload = {"markets": [
            {"ticker": "T1", "title": "One", "last_price": 60},
            {"ticker": "T2", "title": "Two", "last_price": None, "yes_bid": None, "yes_ask": None},
        ]}
        sys.modules["requests"] = _fake_requests(payload)
        quotes = list_kalshi(limit=10)
        self.assertEqual([q["question"] for q in quotes], ["One"])
        self.assertAlmostEqual(quotes[0]["probability"], 0.6)


if __name__ == "__main__":
    unittest.main()
