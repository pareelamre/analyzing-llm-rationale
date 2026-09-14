import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from analyzing_llm_rationale.twin.models import Instrument
from analyzing_llm_rationale.twin.public_evidence import (
    PublicEvidenceError,
    PublicEvidencePolicy,
    acquire_public_evidence,
    captured_market_listing_evidence,
)

NOW = datetime(2026, 9, 13, 12, tzinfo=timezone.utc)


def instrument():
    return Instrument(
        id="kalshi:live:KXTEST", venue="kalshi", environment="live",
        venue_instrument_id="KXTEST", condition_id=None, yes_token_id=None,
        no_token_id=None, settlement_spec_hash="rules", category="politics",
        event_id="KXTEST", cluster_id="KXTEST", tick_size="0.01",
        min_quantity="1", fee_version="fee-v1", capability_version="kalshi-v2-limit",
        status="open", close_at=NOW + timedelta(days=2),
        resolution_at=NOW + timedelta(days=2), created_at=NOW - timedelta(days=1),
        display_title="Will the bill pass by Friday?", display_slug="kxtest",
    )


class Gateway:
    def __init__(self, rows=None, error=None):
        self.rows = rows or []
        self.error = error
        self.calls = []

    def search(self, query, *, limit):
        self.calls.append((query, limit))
        if self.error:
            raise self.error
        return self.rows


class PublicEvidenceTests(unittest.TestCase):
    def test_acquires_bounded_typed_public_evidence(self):
        rows = [
            {
                "title": "Committee advances bill",
                "summary": "The committee approved the measure for a floor vote.",
                "url": "https://example.com/report",
                "publish_date": "2026-09-13T10:00:00Z",
                "relevance": 0.9,
            },
            {
                "title": "Opposition announces delay",
                "text": "Leaders said the vote may move to next week.",
                "url": "https://news.example.org/counter",
                "publish_date": "Sun, 13 Sep 2026 09:00:00 +0000",
                "relevance": 0.8,
            },
        ]
        gateway = Gateway(rows)
        evidence = acquire_public_evidence(gateway, instrument=instrument(), now=NOW)
        self.assertEqual(gateway.calls, [(instrument().display_title, 5)])
        self.assertEqual(len(evidence), 2)
        self.assertTrue(all(item.id.startswith("evidence-") for item in evidence))
        self.assertEqual(evidence[0].retrieved_at, NOW)

    def test_filters_private_stale_future_and_malformed_sources(self):
        rows = [
            {"title": "private", "url": "http://127.0.0.1/a", "relevance": 1},
            {"title": "stale", "url": "https://old.example/a", "publish_date": "2020-01-01T00:00:00Z", "relevance": 1},
            {"title": "future", "url": "https://future.example/a", "publish_date": "2027-01-01T00:00:00Z", "relevance": 1},
            {"title": "missing URL"},
        ]
        with self.assertRaisesRegex(PublicEvidenceError, "no recent"):
            acquire_public_evidence(Gateway(rows), instrument=instrument(), now=NOW)

    def test_bounds_long_public_source_identifiers(self):
        evidence = acquire_public_evidence(Gateway([{
            "title": "Long public URL",
            "summary": "A valid timestamped public report.",
            "url": "https://example.com/" + "x" * 400,
            "publish_date": "2026-09-13T10:00:00Z",
            "relevance": 0.9,
        }]), instrument=instrument(), now=NOW)
        self.assertEqual(len(evidence[0].source_id), 256)

    def test_missing_title_and_provider_failure_fail_closed(self):
        with self.assertRaisesRegex(PublicEvidenceError, "title"):
            acquire_public_evidence(
                Gateway([]), instrument=replace(instrument(), display_title=None), now=NOW,
            )
        with self.assertRaisesRegex(PublicEvidenceError, "unavailable"):
            acquire_public_evidence(
                Gateway(error=RuntimeError("offline")), instrument=instrument(), now=NOW,
            )

    def test_captured_listing_is_a_deterministic_public_fallback(self):
        evidence = captured_market_listing_evidence(
            instrument=instrument(), rules="The contract resolves YES after passage.",
            retrieved_at=NOW,
        )
        repeat = captured_market_listing_evidence(
            instrument=instrument(), rules="The contract resolves YES after passage.",
            retrieved_at=NOW + timedelta(minutes=1),
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(
            evidence[0].source_id,
            "https://api.elections.kalshi.com/trade-api/v2/markets/KXTEST",
        )
        self.assertIn("Contract rules", evidence[0].text)
        self.assertEqual(evidence[0].id, repeat[0].id)
        self.assertEqual(repeat[0].retrieved_at, NOW + timedelta(minutes=1))

    def test_captured_listing_rejects_unsupported_or_incomplete_markets(self):
        with self.assertRaisesRegex(PublicEvidenceError, "unsupported"):
            captured_market_listing_evidence(
                instrument=replace(
                    instrument(), id="other:live:KXTEST", venue="other",
                ), rules="rules",
                retrieved_at=NOW,
            )
        with self.assertRaisesRegex(PublicEvidenceError, "incomplete"):
            captured_market_listing_evidence(
                instrument=replace(instrument(), display_title=None), rules="",
                retrieved_at=NOW,
            )

    def test_policy_is_bounded(self):
        with self.assertRaises(PublicEvidenceError):
            PublicEvidencePolicy(max_evidence=9)


if __name__ == "__main__":
    unittest.main()
