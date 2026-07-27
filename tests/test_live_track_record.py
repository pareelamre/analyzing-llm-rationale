from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.live_track_record import (  # noqa: E402
    LiveTrackRecordConfig,
    LiveTrackRecordReader,
    edge_board_order_context,
    pick_best_strategy,
)


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def warning(self, message, exc_info=False):
        self.messages.append((message, exc_info))


class LiveTrackRecordTests(unittest.TestCase):
    def test_reader_falls_back_to_bundled_copy_when_remote_fetch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            bundled = Path(td) / "track_record_live.json"
            bundled.write_text('{"generated_at":"2026-07-27T10:00:00+00:00","edge_board":[{"question":"Q"}]}')

            cache = {}

            def _cache_key(namespace: str, version: str) -> str:
                return f"{namespace}:{version}"

            def _cache_get(key: str):
                return cache.get(key)

            def _cache_set(key: str, value, ttl: int):
                cache[key] = (value, ttl)

            def _requests_get(*args, **kwargs):
                raise RuntimeError("network down")

            logger = _FakeLogger()
            reader = LiveTrackRecordReader(
                cache_key=_cache_key,
                cache_get=_cache_get,
                cache_set=_cache_set,
                config=LiveTrackRecordConfig(
                    live_url="https://example.test/track_record_live.json",
                    ttl_seconds=30,
                    stale_after_seconds=1800,
                    bundled_path=bundled,
                ),
                logger=logger,
                requests_get=_requests_get,
            )

            payload = reader.read()

        self.assertEqual(payload["edge_board"][0]["question"], "Q")
        self.assertIn("track_record_live:v3", cache)
        self.assertEqual(cache["track_record_live:v3"][1], 30)
        self.assertEqual(logger.messages[0][0], "live track record fetch failed; trying bundled copy")

    def test_freshness_uses_configured_staleness_window(self):
        reader = LiveTrackRecordReader(
            cache_key=lambda namespace, version: f"{namespace}:{version}",
            cache_get=lambda key: None,
            cache_set=lambda key, value, ttl: None,
            config=LiveTrackRecordConfig(
                live_url="https://example.test/track_record_live.json",
                ttl_seconds=30,
                stale_after_seconds=1800,
                bundled_path=Path("unused.json"),
            ),
            logger=_FakeLogger(),
        )
        now = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)

        fresh = reader.freshness(
            {"generated_at": (now - timedelta(minutes=10)).isoformat()},
            now=now,
        )
        stale = reader.freshness(
            {"generated_at": (now - timedelta(hours=1)).isoformat()},
            now=now,
        )

        self.assertFalse(fresh["stale"])
        self.assertEqual(fresh["age_seconds"], 600)
        self.assertTrue(stale["stale"])
        self.assertEqual(stale["stale_after_seconds"], 1800)

    def test_edge_board_order_context_uses_best_strategy_filters(self):
        context = edge_board_order_context(
            {
                "paper_pnl": {
                    "flat": {"roi": 0.05, "n_bets": 50},
                    "smart": {"roi": 0.12, "n_bets": 50},
                },
                "edge_board": [
                    {
                        "question": "Filtered extreme price",
                        "platform": "Kalshi",
                        "side": "YES",
                        "market_probability": 0.91,
                        "model_probability": 0.95,
                        "abs_edge": 0.04,
                        "entry_price": 0.91,
                        "payout_odds": 0.1,
                        "market_url": "https://example.test/extreme",
                        "track_record": {"skill_significant": True},
                    },
                    {
                        "question": "Keep this setup",
                        "platform": "Polymarket",
                        "side": "NO",
                        "market_probability": 0.4,
                        "model_probability": 0.25,
                        "abs_edge": 0.15,
                        "entry_price": 0.4,
                        "payout_odds": 1.5,
                        "market_url": "https://example.test/keep",
                        "track_record": {"skill_significant": False},
                    },
                ],
            }
        )

        self.assertIn("Best back-tested strategy: **smart**", context)
        self.assertIn("Keep this setup", context)
        self.assertNotIn("Filtered extreme price", context)

    def test_pick_best_strategy_falls_back_to_flat_when_sample_is_small(self):
        name, data = pick_best_strategy(
            {
                "flat": {"roi": 0.01, "n_bets": 8},
                "smart": {"roi": 0.15, "n_bets": 12},
            }
        )

        self.assertEqual(name, "flat")
        self.assertEqual(data["roi"], 0.01)
