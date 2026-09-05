"""Venue contract regressions using the public APIs' actual payload shapes."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import market_data as md  # noqa: E402
from analyzing_llm_rationale.mcp_server import ForeseaClient  # noqa: E402


class VenueApiContractTests(unittest.TestCase):
    def test_long_kalshi_ticker_uses_kalshi_trade_query(self):
        ticker = "KXBTC15M-26SEP050930-30"
        with patch.object(md, "_get_json", return_value={"trades": [{"ticker": ticker}], "cursor": ""}) as get:
            self.assertEqual(md.fetch_recent_trades("kalshi", ticker)[0]["ticker"], ticker)
        self.assertTrue(get.call_args.args[0].endswith("/markets/trades"))
        self.assertEqual(get.call_args.kwargs["params"]["ticker"], ticker)

    def test_public_poly_trades_resolve_token_and_filter_its_outcome(self):
        condition = "0x" + "a" * 64
        payloads = [
            [{"conditionId": condition, "clobTokenIds": '["123", "456"]'}],
            [{"asset": "456", "size": 1}, {"asset": "123", "size": 2}],
        ]
        with patch.object(md, "_get_json", side_effect=payloads) as get:
            trades = md.fetch_recent_trades("polymarket", "123")
        self.assertEqual(trades, [{"asset": "123", "size": 2}])
        self.assertEqual(get.call_args.args[0], "https://data-api.polymarket.com/trades")
        self.assertEqual(get.call_args.kwargs["params"], {"limit": 20, "market": condition})

    def test_public_poly_condition_id_needs_no_token_lookup(self):
        condition = "0x" + "b" * 64
        with patch.object(md, "_get_json", return_value=[]) as get:
            self.assertEqual(md.fetch_recent_trades("polymarket", condition), [])
        self.assertEqual(get.call_count, 1)
        self.assertEqual(get.call_args.kwargs["params"]["market"], condition)

    def test_unknown_poly_token_does_not_fetch_unfiltered_trade_tape(self):
        with patch.object(md, "_get_json", return_value=[]) as get, self.assertRaises(md.MarketDataInputError):
            md.fetch_recent_trades("polymarket", "123")
        self.assertEqual(get.call_count, 1)

    def test_venue_failure_is_not_an_empty_trade_or_leaderboard(self):
        for helper, args in ((md.fetch_recent_trades, ("kalshi",)), (md.fetch_trader_leaderboard, ())):
            with self.subTest(helper=helper.__name__), patch.object(md, "_get_json", side_effect=md.MarketDataError("offline")), self.assertRaises(md.MarketDataError):
                helper(*args)

    def test_leaderboard_uses_versioned_endpoint_and_validates_shape(self):
        with patch.object(md, "_get_json", return_value=[{"rank": "1"}]) as get:
            self.assertEqual(md.fetch_trader_leaderboard()[0]["rank"], "1")
        self.assertEqual(get.call_args.args[0], "https://data-api.polymarket.com/v1/leaderboard")
        with patch.object(md, "_get_json", return_value={"error": "offline"}), self.assertRaises(md.MarketDataError):
            md.fetch_trader_leaderboard()

    def test_candlestick_defaults_include_required_time_parameters(self):
        with patch.object(md, "_get_json", return_value={"candlesticks": []}) as get:
            md.fetch_kalshi_candlesticks("KXBTC15M-TEST")
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["end_ts"] - params["start_ts"], 86400)
        self.assertEqual(params["period_interval"], 60)

    def test_invalid_candle_window_never_calls_upstream(self):
        with patch.object(md, "_get_json") as get, self.assertRaises(md.MarketDataInputError):
            md.fetch_kalshi_candlesticks("KXTEST", start_ts=20, end_ts=10)
        get.assert_not_called()

    def test_live_event_and_game_stats_use_distinct_paths(self):
        cases = [
            ({"event_ticker": "KXBTC-TEST"}, "/live_data/events/KXBTC-TEST"),
            ({"milestone_id": "milestone-1", "data_type": "game_stats"}, "/live_data/milestone/milestone-1/game_stats"),
        ]
        for args, path in cases:
            with self.subTest(path=path), patch.object(md, "_get_json", return_value={"live_data": {}}) as get:
                md.fetch_kalshi_live_data(**args)
            self.assertTrue(get.call_args.args[0].endswith(path))
        with self.assertRaises(md.MarketDataInputError):
            md.fetch_kalshi_live_data()

    def test_comments_use_parent_entity_filter(self):
        with patch.object(md, "_get_json", side_effect=[[{"id": "42", "events": [{"id": "99"}]}], []]) as get:
            md.fetch_polymarket_comments("42")
        self.assertEqual(get.call_args.kwargs["params"], {"parent_entity_type": "Event", "parent_entity_id": "99"})

    def test_explicit_event_comments_skip_market_lookup(self):
        with patch.object(md, "_get_json", return_value=[]) as get:
            md.fetch_polymarket_comments("99", "Event")
        self.assertEqual(get.call_count, 1)
        self.assertEqual(get.call_args.kwargs["params"]["parent_entity_type"], "Event")

    def test_unknown_market_comments_do_not_return_global_discussion(self):
        with patch.object(md, "_get_json", return_value=[]) as get, self.assertRaises(md.MarketDataInputError):
            md.fetch_polymarket_comments("42")
        self.assertEqual(get.call_count, 1)

    def test_mcp_long_ticker_orderbook_and_teams(self):
        client = ForeseaClient(base_url="https://foresea.test")
        with patch.object(md, "_get_json", return_value={"orderbook": {"yes": []}}) as get:
            client.orderbook("KXBTC15M-26SEP050930-30")
        self.assertIn("kalshi.com", get.call_args.args[0])
        with patch.object(md, "_get_json", return_value=[{"name": "Team"}]) as get:
            self.assertEqual(client.polymarket_meta("teams"), [{"name": "Team"}])
        self.assertTrue(get.call_args.args[0].endswith("/teams"))

    def test_mcp_does_not_hide_upstream_errors(self):
        client = ForeseaClient(base_url="https://foresea.test")
        with patch.object(md, "_get_json", side_effect=md.MarketDataError("offline")), self.assertRaises(md.MarketDataError):
            client.recent_trades("kalshi", "KXTEST")

    def test_invalid_exchange_status_is_not_reported_as_active(self):
        with patch.object(md, "_get_json", return_value=[]), self.assertRaises(md.MarketDataError):
            md.fetch_kalshi_exchange_status()


if __name__ == "__main__":
    unittest.main()
