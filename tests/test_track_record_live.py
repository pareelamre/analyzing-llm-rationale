"""Live track-record trajectory harness: daily snapshots per market until
resolution, scored by horizon. Exercised with an in-memory fake Datastore, a
fake market_data module, and a stub forecast function — no network/SCADS."""
from __future__ import annotations

import asyncio
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import track_record_live as trl  # noqa: E402


class FakeEntity(dict):
    def __init__(self, key=None, exclude_from_indexes=()):
        super().__init__()
        self.key = key


class FakeKey:
    def __init__(self, kind, id_=None):
        self.kind = kind
        self.id = id_


class FakeClient:
    def __init__(self):
        self.store = {}

    def key(self, kind, id_=None):
        return FakeKey(kind, id_)

    def get(self, key):
        return self.store.get((key.kind, key.id))

    def put(self, entity):
        self.store[(entity.key.kind, entity.key.id)] = entity

    def query(self, kind):
        return _FakeQuery(self, kind)


class _FakeQuery:
    def __init__(self, client, kind):
        self.client, self.kind, self.filters = client, kind, []

    def add_filter(self, name, op, value):
        self.filters.append((name, op, value))

    def fetch(self):
        return [e for (k, _i), e in self.client.store.items()
                if k == self.kind and all(e.get(n) == v for n, _o, v in self.filters)]


def _fake_market_data(close_iso):
    md = types.ModuleType("market_data_fake")

    class MarketDataError(RuntimeError):
        pass

    md.MarketDataError = MarketDataError
    md._poly = {"platform": "Polymarket", "question": "Will A happen?",
                "market_url": "https://polymarket.com/market/slug-a", "outcome": "Yes",
                "probability": 0.40, "close_time": close_iso}
    md._kalshi = {"platform": "Kalshi", "question": "Will B happen?",
                  "market_url": "https://kalshi.com/markets/TICKERB", "outcome": "Yes",
                  "probability": 0.45, "close_time": close_iso}
    md.list_polymarket = lambda limit=3, **kw: [dict(md._poly)]
    md.list_kalshi = lambda limit=3, **kw: [dict(md._kalshi)]
    md.fetch_polymarket = lambda slug=None, market_id=None: dict(md._poly)
    md.fetch_kalshi = lambda ticker: dict(md._kalshi)
    md.resolve_polymarket = lambda ident: None
    md.resolve_kalshi = lambda ident: None
    return md


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        fake_ds = types.ModuleType("google.cloud.datastore")
        fake_ds.Entity = FakeEntity
        self._p = mock.patch.dict(sys.modules, {"google.cloud.datastore": fake_ds})
        self._p.start()
        self.model_probs = {"Will A happen?": 0.70, "Will B happen?": 0.30}

        async def forecast_fn(quote, top_k):
            return (self.model_probs[quote["question"]], quote["probability"])

        self.forecast_fn = forecast_fn

    def tearDown(self):
        self._p.stop()

    def _record(self, md, day):
        with mock.patch.object(trl, "_today", return_value=day):
            return asyncio.run(trl.record_snapshots(self.client, md, self.forecast_fn, per_venue=3))

    def test_lead_time_and_horizon_labels(self):
        self.assertEqual(trl._horizon_label(20.0), "14-30d")
        self.assertEqual(trl._horizon_label(0.5), "<1d")
        self.assertEqual(trl._horizon_label(8.0), "7-14d")

    def test_short_dated_market_not_discovered(self):
        # Market resolves in 6 hours -> below min_discovery_lead_days, skipped.
        soon = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
        md = _fake_market_data(soon)
        recorded = self._record(md, "2026-06-03")
        self.assertEqual(recorded, 0)

    def test_one_snapshot_per_market_per_day(self):
        far = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        md = _fake_market_data(far)
        self.assertEqual(self._record(md, "2026-06-03"), 2)         # A + B
        self.assertEqual(self._record(md, "2026-06-03"), 0)         # same day, no dupes
        self.assertEqual(self._record(md, "2026-06-04"), 2)         # next day, new snapshots
        snaps = [k for k in self.client.store if k[0] == trl.SNAPSHOT_KIND]
        self.assertEqual(len(snaps), 4)

    def test_resolution_scores_all_snapshots_and_buckets_by_horizon(self):
        far = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        md = _fake_market_data(far)
        self._record(md, "2026-06-03")  # both at ~10d horizon
        # Now both markets resolve: A -> YES(1), B -> NO(0).
        md.resolve_polymarket = lambda ident: 1
        md.resolve_kalshi = lambda ident: 0
        scored = trl.resolve_open_snapshots(self.client, md)
        self.assertEqual(scored, 2)

        agg = trl.aggregate(self.client, model="m", variant="v", temperature=0.0)
        self.assertEqual(agg["n_snapshots_resolved"], 2)
        self.assertEqual(agg["n_markets_resolved"], 2)
        self.assertEqual(agg["overall"]["accuracy"], 1.0)
        # model brier .09 each; market brier (.36 + .2025)/2 = .28125
        self.assertAlmostEqual(agg["overall"]["model_brier"], 0.09, places=4)
        self.assertAlmostEqual(agg["overall"]["skill_vs_market"], 0.19125, places=4)
        # Both snapshots were ~10 days out -> the 7-14d bucket.
        labels = {b["horizon"] for b in agg["by_horizon"]}
        self.assertIn("7-14d", labels)

    def test_trajectory_series_captured(self):
        far = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        md = _fake_market_data(far)
        self._record(md, "2026-06-03")
        self._record(md, "2026-06-04")  # second snapshot -> trajectory has 2 points
        md.resolve_polymarket = lambda ident: 1
        md.resolve_kalshi = lambda ident: 0
        trl.resolve_open_snapshots(self.client, md)
        agg = trl.aggregate(self.client, model="m", variant="v", temperature=0.0)
        traj = {t["question"]: t for t in agg["trajectories"]}
        self.assertEqual(len(traj["Will A happen?"]["points"]), 2)

    def test_empty_aggregate_is_safe(self):
        agg = trl.aggregate(self.client, model="m", variant="v", temperature=0.0)
        self.assertEqual(agg["n_snapshots_resolved"], 0)
        self.assertIsNone(agg["overall"])
        self.assertEqual(agg["by_horizon"], [])


if __name__ == "__main__":
    unittest.main()
