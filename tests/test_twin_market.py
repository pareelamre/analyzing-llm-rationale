from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from analyzing_llm_rationale.twin import RejectionReason
from analyzing_llm_rationale.twin.market import normalize_market

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)
CLOSE = NOW + timedelta(days=10)


def poly_market(**updates):
    result = {
        "id": "market-001",
        "conditionId": "condition-001",
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["yes-token-001", "no-token-001"]',
        "active": True,
        "closed": False,
        "acceptingOrders": True,
        "endDateIso": CLOSE.isoformat(),
        "rules": "Resolves against the official source.",
        "category": "politics",
        "minimum_tick_size": "0.01",
        "minimum_order_size": "1",
    }
    result.update(updates)
    return result


def books(*, include_no=True, zero=False):
    level = {"price": "0.50", "size": "0" if zero else "10"}
    result = {"yes-token-001": {"bids": [level], "asks": [level]}}
    if include_no:
        result["no-token-001"] = {"bids": [level], "asks": [level]}
    return result


class TwinMarketTests(unittest.TestCase):
    def test_complete_binary_market_has_precise_instrument_and_actual_depth(self):
        assessment = normalize_market("polymarket", poly_market(holders=None), received_at=NOW, sequence=1, orderbooks=books())
        self.assertTrue(assessment.eligible)
        self.assertEqual(assessment.instrument.id, "polymarket:live:condition-001:market-001")
        self.assertEqual(assessment.instrument.yes_token_id, "yes-token-001")
        self.assertEqual(str(assessment.snapshot.yes_ask), "0.50")

    def test_polymarket_uses_full_end_time_and_consistent_book_precision(self):
        market = poly_market(
            endDate=(NOW + timedelta(days=5)).isoformat(),
            endDateIso="2025-01-06",
            minimum_tick_size=None,
            minimum_order_size=None,
        )
        precision_books = books()
        for book in precision_books.values():
            book["tick_size"] = "0.01"
            book["min_order_size"] = "5"
        assessment = normalize_market(
            "polymarket", market, received_at=NOW, sequence=1,
            orderbooks=precision_books,
        )
        self.assertTrue(assessment.eligible)
        self.assertEqual(assessment.instrument.close_at, NOW + timedelta(days=5))
        self.assertEqual(assessment.instrument.tick_size, Decimal("0.01"))
        self.assertEqual(assessment.instrument.min_quantity, Decimal("5"))

        precision_books["no-token-001"]["tick_size"] = "0.001"
        rejected = normalize_market(
            "polymarket", market, received_at=NOW, sequence=2,
            orderbooks=precision_books,
        )
        self.assertIn(RejectionReason.PASS_INCOMPLETE_DATA, rejected.reasons)

    def test_polymarket_selects_highest_bid_and_lowest_ask(self):
        varied_books = books()
        for book in varied_books.values():
            book["bids"] = [
                {"price": "0.40", "size": "5"},
                {"price": "0.45", "size": "7"},
            ]
            book["asks"] = [
                {"price": "0.60", "size": "5"},
                {"price": "0.55", "size": "7"},
            ]
        assessment = normalize_market(
            "polymarket", poly_market(), received_at=NOW, sequence=1,
            orderbooks=varied_books,
        )
        self.assertEqual(assessment.snapshot.yes_bid, Decimal("0.45"))
        self.assertEqual(assessment.snapshot.yes_ask, Decimal("0.55"))

    def test_missing_no_book_and_zero_depth_are_incomplete_not_tradeable(self):
        missing = normalize_market("polymarket", poly_market(), received_at=NOW, sequence=1, orderbooks=books(include_no=False))
        empty = normalize_market("polymarket", poly_market(), received_at=NOW, sequence=1, orderbooks=books(zero=True))
        self.assertEqual(missing.reasons, (RejectionReason.PASS_INCOMPLETE_DATA,))
        self.assertEqual(empty.reasons, (RejectionReason.PASS_INCOMPLETE_DATA,))

    def test_malformed_or_incomplete_contract_data_is_passed(self):
        malformed = normalize_market("polymarket", poly_market(outcomes="not-json"), received_at=NOW, sequence=1, orderbooks=books())
        no_tick = normalize_market("polymarket", poly_market(minimum_tick_size=None), received_at=NOW, sequence=1, orderbooks=books())
        self.assertIn(RejectionReason.PASS_UNSUPPORTED_INSTRUMENT, malformed.reasons)
        self.assertIn(RejectionReason.PASS_INCOMPLETE_DATA, no_tick.reasons)

    def test_future_venue_timestamp_and_suspended_market_are_rejected(self):
        future = normalize_market("polymarket", poly_market(updatedAt=(NOW + timedelta(minutes=1)).isoformat()), received_at=NOW, sequence=1, orderbooks=books())
        suspended = normalize_market("polymarket", poly_market(acceptingOrders=False), received_at=NOW, sequence=1, orderbooks=books())
        self.assertIn(RejectionReason.PASS_STALE_DATA, future.reasons)
        self.assertIn(RejectionReason.PASS_UNSUPPORTED_INSTRUMENT, suspended.reasons)

    def test_same_title_with_different_settlement_condition_has_distinct_identity(self):
        first = normalize_market("polymarket", poly_market(title="Will X?"), received_at=NOW, sequence=1, orderbooks=books())
        second = normalize_market("polymarket", poly_market(id="market-002", conditionId="condition-002", title="Will X?"), received_at=NOW, sequence=1, orderbooks=books())
        self.assertNotEqual(first.instrument.id, second.instrument.id)
        self.assertNotEqual(first.instrument.settlement_spec_hash, second.instrument.settlement_spec_hash)

    def test_kalshi_requires_both_outcomes_and_known_metadata(self):
        market = {
            "ticker": "KXTEST", "status": "active", "close_time": CLOSE.isoformat(),
            "rules_primary": "Official source", "category": "politics", "tick_size": "0.01",
            "min_contracts": "1", "yes_bid_dollars": "0.49", "yes_ask_dollars": "0.51",
            "no_bid_dollars": "0.49", "no_ask_dollars": "0.51",
        }
        accepted = normalize_market("kalshi", market, received_at=NOW, sequence=1)
        rejected = normalize_market("kalshi", {**market, "no_ask_dollars": None}, received_at=NOW, sequence=1)
        self.assertTrue(accepted.eligible)
        self.assertIn(RejectionReason.PASS_INCOMPLETE_DATA, rejected.reasons)

    def test_kalshi_uses_fixed_point_price_grid_and_quantity_granularity(self):
        market = {
            "ticker": "KXTEST", "event_ticker": "KXEVENT", "status": "active",
            "close_time": CLOSE.isoformat(), "rules_primary": "Official source",
            "price_level_structure": "linear_cent",
            "price_ranges": [{"start": "0", "end": "1", "step": "0.01"}],
            "yes_bid_dollars": "0.49", "yes_ask_dollars": "0.51",
            "no_bid_dollars": "0.49", "no_ask_dollars": "0.51",
        }
        assessment = normalize_market("kalshi", market, received_at=NOW, sequence=1)
        self.assertTrue(assessment.eligible)
        self.assertEqual(assessment.instrument.tick_size, Decimal("0.01"))
        self.assertEqual(assessment.instrument.min_quantity, Decimal("0.01"))
        self.assertEqual(assessment.instrument.cluster_id, "KXEVENT")

        market["price_ranges"].append({"start": "0.9", "end": "1", "step": "0.001"})
        rejected = normalize_market("kalshi", market, received_at=NOW, sequence=2)
        self.assertIn(RejectionReason.PASS_INCOMPLETE_DATA, rejected.reasons)


if __name__ == "__main__":
    unittest.main()
