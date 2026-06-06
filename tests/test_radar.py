from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import radar  # noqa: E402


class RadarTests(unittest.TestCase):
    def test_tags_and_score_surface_product_risks(self):
        item = {
            "question": "Will Oracle mention OpenAI on its earnings call before July 1?",
            "market_url": "https://example.com/market",
            "market_probability": 0.71,
            "probability": 0.71,
            "close_time": "2026-06-10T00:00:00Z",
            "volume": 500,
            "liquidity": 250,
            "reddit_url": "https://old.reddit.com/r/PredictionMarkets/example",
        }
        now = datetime(2026, 6, 6, tzinfo=timezone.utc)

        tags = radar.market_tags(item, now)

        self.assertIn("thin liquidity", tags)
        self.assertIn("resolution-rule risk", tags)
        self.assertIn("news catalyst", tags)
        self.assertIn("near-term", tags)
        self.assertIn("reddit-discussed", tags)
        self.assertLess(radar.credibility_score(item, now), 80)

    def test_build_radar_merges_live_snapshot_and_public_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data" / "track_record_store.json").write_text(json.dumps({
                "ForecastSnapshot": {
                    "snap1": {
                        "ident": "KXTEST",
                        "market_url": "https://kalshi.com/markets/KXTEST",
                        "model_probability": 0.68,
                        "market_probability": 0.52,
                        "snapshot_date": "2026-06-06",
                        "snapshot_ts": {"__dt__": "2026-06-06T08:00:00+00:00"},
                    }
                }
            }))

            kalshi_quote = {
                "platform": "Kalshi",
                "ident": "KXTEST",
                "question": "Will test event happen before July 1?",
                "market_url": "https://kalshi.com/markets/KXTEST",
                "outcome": "Yes",
                "probability": 0.52,
                "close_time": "2026-07-01T00:00:00Z",
                "volume": 5000,
                "liquidity": 8000,
                "category": "Politics",
            }
            with (
                mock.patch.object(radar, "_SEED_MARKETS", []),
                mock.patch.object(radar.market_data, "list_polymarket", return_value=[]),
                mock.patch.object(radar.market_data, "list_kalshi", return_value=[kalshi_quote]),
                mock.patch.object(radar, "fetch_reddit_discussions", return_value=[]),
            ):
                result = radar.build_radar(root, limit=5)

        self.assertEqual(len(result["items"]), 1)
        item = result["items"][0]
        self.assertTrue(item["tracked_live"])
        self.assertAlmostEqual(item["foresea_probability"], 0.68)
        self.assertAlmostEqual(item["edge"], 0.16)
        self.assertIn("tracked live", item["tags"])
        self.assertEqual(item["evidence_links"][0]["url"], "https://kalshi.com/markets/KXTEST")


if __name__ == "__main__":
    unittest.main()
