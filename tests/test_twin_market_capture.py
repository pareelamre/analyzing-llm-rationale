import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from analyzing_llm_rationale.market_data import MarketDataError
from analyzing_llm_rationale.twin.market_capture import (
    InMemoryMarketCaptureStore,
    MarketCaptureError,
    MarketCapturePolicy,
    capture_markets,
)

NOW = datetime(2026, 9, 13, tzinfo=timezone.utc)


def kalshi_market():
    return {
        "ticker": "KXTEST", "status": "active",
        "close_time": (NOW + timedelta(days=2)).isoformat(),
        "rules_primary": "Official Kalshi rule source.", "category": "politics",
        "tick_size": "0.01", "min_contracts": "1",
        "yes_bid_dollars": "0.40", "yes_ask_dollars": "0.42",
        "no_bid_dollars": "0.58", "no_ask_dollars": "0.60",
        "yes_ask_size_fp": "12.5", "no_ask_size_fp": "9.5",
        "series_ticker": "KXTEST", "_foresea_fee_schedule": {
            "ticker": "KXTEST", "fee_type": "quadratic", "fee_multiplier": 1,
            "last_updated_ts": "2026-09-12T00:00:00Z",
        },
    }


def polymarket_market():
    return {
        "id": "market-1", "conditionId": "condition-1",
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["yes-token", "no-token"]',
        "active": True, "closed": False, "acceptingOrders": True,
        "endDateIso": (NOW + timedelta(days=3)).isoformat(),
        "rules": "Official Polymarket rule source.", "category": "politics",
        "minimum_tick_size": "0.01", "minimum_order_size": "1",
        "feesEnabled": True,
        "feeSchedule": {"rate": 0.04, "exponent": 1, "takerOnly": True},
    }


class Gateway:
    def __init__(self, *, discovery_error=False):
        self.discovery_error = discovery_error

    def discover(self, venue, **_kwargs):
        if self.discovery_error and venue == "kalshi":
            raise MarketDataError("offline")
        if venue == "kalshi":
            return [{"ident": "KXTEST"}]
        return [{"ident": "slug-1", "market_id": "market-1"}]

    def fetch(self, venue, _identifier):
        if venue == "kalshi":
            return kalshi_market(), {}
        def level(price, size):
            return {"price": price, "size": size}
        return polymarket_market(), {
            "yes-token": {
                "bids": [level("0.40", "8")],
                "asks": [level("0.60", "4"), level("0.42", "7")],
            },
            "no-token": {
                "bids": [level("0.40", "6")],
                "asks": [level("0.60", "3"), level("0.45", "5")],
            },
        }


class MarketCaptureTests(unittest.TestCase):
    def test_capture_interleaves_venues_and_preserves_exact_ask_depth(self):
        batch = capture_markets(Gateway(), now=NOW)
        self.assertEqual([item.instrument.venue for item in batch.markets], [
            "kalshi", "polymarket",
        ])
        self.assertEqual(batch.markets[0].yes_ask_depth, Decimal("12.5"))
        self.assertEqual(batch.markets[0].no_ask_depth, Decimal("9.5"))
        self.assertEqual(batch.markets[1].snapshot.yes_ask, Decimal("0.42"))
        self.assertEqual(batch.markets[1].yes_ask_depth, Decimal("7"))
        self.assertEqual(batch.markets[1].no_ask_depth, Decimal("5"))
        self.assertEqual(batch.markets[1].settlement_rules, "Official Polymarket rule source.")
        self.assertEqual(batch.markets[0].trading_cost.yes_fee_per_share, Decimal("0.0171"))
        self.assertEqual(batch.markets[1].trading_cost.yes_fee_per_share, Decimal("0.009744"))
        self.assertEqual(
            batch.markets[1].instrument.fee_version,
            batch.markets[1].trading_cost.schedule_version,
        )

    def test_missing_or_unsupported_fee_schedule_is_rejected(self):
        gateway = Gateway()
        market = polymarket_market()
        market.pop("feeSchedule")
        gateway.discover = lambda venue, **_kwargs: (
            [] if venue == "kalshi" else [{"market_id": "market-1"}]
        )
        books = Gateway().fetch("polymarket", "market-1")[1]
        gateway.fetch = lambda _venue, _identifier: (market, books)
        batch = capture_markets(gateway, now=NOW)
        self.assertEqual(batch.markets, ())
        self.assertEqual(batch.rejections[0].reason, "missing_verified_fee_schedule")

    def test_fee_free_polymarket_has_verified_zero_cost(self):
        gateway = Gateway()
        market = polymarket_market()
        market["feesEnabled"] = False
        market.pop("feeSchedule")
        original_fetch = gateway.fetch
        gateway.fetch = lambda venue, identifier: (
            (market, original_fetch(venue, identifier)[1])
            if venue == "polymarket" else original_fetch(venue, identifier)
        )
        batch = capture_markets(gateway, now=NOW)
        poly = next(item for item in batch.markets if item.instrument.venue == "polymarket")
        self.assertEqual(poly.trading_cost.rate, Decimal("0"))

    def test_missing_depth_is_rejected_without_partial_candidate(self):
        gateway = Gateway()
        market = kalshi_market()
        market.pop("yes_ask_size_fp")
        gateway.fetch = lambda _venue, _identifier: (market, {})
        batch = capture_markets(
            gateway, now=NOW, policy=MarketCapturePolicy(max_candidates=1),
        )
        self.assertEqual(batch.markets, ())
        self.assertEqual(batch.rejections[0].reason, "missing_executable_depth")

    def test_one_venue_outage_does_not_hide_other_public_capture(self):
        batch = capture_markets(Gateway(discovery_error=True), now=NOW)
        self.assertEqual([item.instrument.venue for item in batch.markets], ["polymarket"])
        self.assertEqual(batch.rejections[0].reason, "data_unavailable")

    def test_policy_bounds_market_and_discovery_work(self):
        with self.assertRaisesRegex(Exception, "candidate limit"):
            MarketCapturePolicy(max_candidates=7)
        with self.assertRaisesRegex(Exception, "horizon"):
            MarketCapturePolicy(min_close_days=30, max_close_days=1)

    def test_capture_store_is_create_once_and_round_trips(self):
        store = InMemoryMarketCaptureStore()
        batch = capture_markets(Gateway(), now=NOW)
        self.assertTrue(store.record("cycle-1", batch))
        self.assertFalse(store.record("cycle-1", batch))
        self.assertEqual(store.get("cycle-1"), batch)
        changed = capture_markets(Gateway(), now=NOW + timedelta(seconds=1))
        with self.assertRaisesRegex(MarketCaptureError, "conflicting"):
            store.record("cycle-1", changed)


if __name__ == "__main__":
    unittest.main()
