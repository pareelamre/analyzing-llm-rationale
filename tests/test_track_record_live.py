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
                "probability": 0.40, "close_time": close_iso, "category": "World", "volume": 1234.0}
    md._kalshi = {"platform": "Kalshi", "question": "Will B happen?",
                  "market_url": "https://kalshi.com/markets/TICKERB", "outcome": "Yes",
                  "probability": 0.45, "close_time": close_iso, "category": "Politics", "volume": 88.0}
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
            return {"model_probability": self.model_probs[quote["question"]],
                    "market_probability": quote["probability"], "evidence_count": 2}

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

    def test_hourly_price_points_dedup_per_hour(self):
        far = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        md = _fake_market_data(far)
        self._record(md, "2026-06-03")  # 2 open markets (A, B)
        h9 = datetime(2026, 6, 4, 9, 0, tzinfo=timezone.utc)
        h10 = datetime(2026, 6, 4, 10, 0, tzinfo=timezone.utc)
        with mock.patch.object(trl, "_now", return_value=h9):
            self.assertEqual(trl.record_price_points(self.client, md), 2)
            self.assertEqual(trl.record_price_points(self.client, md), 0)  # same hour -> dedup
        with mock.patch.object(trl, "_now", return_value=h10):
            self.assertEqual(trl.record_price_points(self.client, md), 2)  # new hour
        pts = [e for (k, _i), e in self.client.store.items() if k == trl.PRICE_KIND]
        self.assertEqual(len(pts), 4)
        # Price points carry the market price only — no model forecast / no LLM.
        self.assertIn("market_probability", pts[0])
        self.assertNotIn("model_probability", pts[0])

    def test_trajectory_includes_price_points(self):
        far = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        md = _fake_market_data(far)
        self._record(md, "2026-06-03")
        with mock.patch.object(trl, "_now", return_value=datetime(2026, 6, 3, 12, tzinfo=timezone.utc)):
            trl.record_price_points(self.client, md)
        md.resolve_polymarket = lambda ident: 1
        md.resolve_kalshi = lambda ident: 0
        trl.resolve_open_snapshots(self.client, md)
        agg = trl.aggregate(self.client, model="m", variant="v", temperature=0.0)
        traj = {t["question"]: t for t in agg["trajectories"]}
        self.assertGreaterEqual(len(traj["Will A happen?"]["price_points"]), 1)

    def test_training_features_captured(self):
        far = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        md = _fake_market_data(far)
        self._record(md, "2026-06-03")
        snaps = [e for (k, _i), e in self.client.store.items() if k == trl.SNAPSHOT_KIND]
        poly = next(e for e in snaps if e.get("platform") == "Polymarket")
        self.assertEqual(poly.get("category"), "World")
        self.assertEqual(poly.get("market_volume"), 1234.0)
        self.assertEqual(poly.get("evidence_count"), 2)

    def test_empty_aggregate_is_safe(self):
        agg = trl.aggregate(self.client, model="m", variant="v", temperature=0.0)
        self.assertEqual(agg["n_snapshots_resolved"], 0)
        self.assertIsNone(agg["overall"])
        self.assertEqual(agg["by_horizon"], [])
        self.assertFalse(agg["calibration_model"]["applied"])


class CalibrationTests(unittest.TestCase):
    def test_isotonic_is_monotonic_and_corrects_bias(self):
        # Raw probs all 0.8 but true base rate 0.5 -> isotonic should map ~0.8 -> ~0.5.
        bp = trl._fit_isotonic([(0.8, 1)] * 10 + [(0.8, 0)] * 10)
        self.assertAlmostEqual(trl._apply_isotonic(bp, 0.8), 0.5, places=2)

    def test_isotonic_enforces_non_decreasing(self):
        bp = trl._fit_isotonic([(0.2, 1), (0.4, 0), (0.6, 1)])
        xs, ys = bp
        self.assertEqual(ys, sorted(ys))  # non-decreasing

    def _rows(self, model_p, outcomes, market_brier=0.3):
        return [{"model_probability": model_p, "outcome": o,
                 "model_brier": (model_p - o) ** 2, "market_brier": market_brier}
                for o in outcomes]

    def test_gate_insufficient_data(self):
        rep = trl._calibration_report(self._rows(0.8, [1, 0] * 5))  # 10 rows
        self.assertFalse(rep["applied"])
        self.assertEqual(rep["reason"], "insufficient_data")

    def test_gate_already_well_calibrated(self):
        # 40 rows at p=0.5 with 50% base rate -> ECE ~ 0 -> no need to calibrate.
        rep = trl._calibration_report(self._rows(0.5, [1, 0] * 20))
        self.assertFalse(rep["applied"])
        self.assertEqual(rep["reason"], "already_well_calibrated")

    def test_gate_applies_when_miscalibrated(self):
        # 50 rows: model says 0.8, true base rate 0.5 -> ECE 0.3 -> calibrate.
        rep = trl._calibration_report(self._rows(0.8, [1, 0] * 25))
        self.assertTrue(rep["applied"])
        self.assertEqual(rep["method"], "isotonic")
        # Calibrated (out-of-fold) Brier should beat raw on this biased set.
        self.assertLess(rep["calibrated_brier_cv"], rep["raw_brier"])


if __name__ == "__main__":
    unittest.main()
