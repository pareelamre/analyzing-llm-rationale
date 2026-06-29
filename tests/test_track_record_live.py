"""Live track-record trajectory harness: daily snapshots per market until
resolution, scored by horizon. Exercised with an in-memory fake Datastore, a
fake market_data module, and a stub forecast function — no network/SCADS."""
from __future__ import annotations

import asyncio
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import track_record_live as trl  # noqa: E402
from analyzing_llm_rationale.trackrec_store import DuckDBStore, Entity, FileStore  # noqa: E402


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
                "probability": 0.40, "close_time": close_iso, "category": "World",
                "volume": 1234.0, "liquidity": 500.0, "yes_bid": 0.39,
                "yes_ask": 0.41, "last_trade_price": 0.40}
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

        async def forecast_fn(quote, top_k, model=None):
            return {"model_probability": self.model_probs[quote["question"]],
                    "market_probability": quote["probability"], "evidence_count": 2}

        self.forecast_fn = forecast_fn

    def tearDown(self):
        self._p.stop()

    def _record(self, md, day):
        now = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        with mock.patch.object(trl, "_now", return_value=now):
            return asyncio.run(trl.record_snapshots(
                self.client, md, self.forecast_fn, default_model="m", per_venue=3))

    def test_lead_time_and_horizon_labels(self):
        self.assertEqual(trl._horizon_label(20.0), "14-30d")
        self.assertEqual(trl._horizon_label(0.5), "<1d")
        self.assertEqual(trl._horizon_label(8.0), "7-14d")

    def test_seed_idents_enroll_market_without_discovery(self):
        far = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        md = _fake_market_data(far)
        with mock.patch.object(trl, "_now", return_value=datetime(2026, 6, 3, tzinfo=timezone.utc)):
            recorded = asyncio.run(trl.record_snapshots(
                self.client, md, self.forecast_fn, default_model="m",
                per_venue=0, seed_idents=[("Polymarket", "slug-a")]))
        self.assertGreaterEqual(recorded, 1)
        snaps = [e for (k, _i), e in self.client.store.items() if k == trl.SNAPSHOT_KIND]
        self.assertTrue(any(s.get("ident") == "slug-a" for s in snaps))

    def test_short_dated_market_not_discovered(self):
        # Market resolves in 6 hours -> below min_discovery_lead_days, skipped.
        soon = (datetime(2026, 6, 3, tzinfo=timezone.utc) + timedelta(hours=6)).isoformat()
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

    def test_reforecast_each_tick_overwrites_same_day_snapshot(self):
        far = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        md = _fake_market_data(far)
        self.assertEqual(self._record(md, "2026-06-03"), 2)  # initial A + B
        # Same day, model opinion changed; with reforecast_each_tick it re-runs
        # and overwrites today's snapshot rather than skipping.
        self.model_probs = {"Will A happen?": 0.55, "Will B happen?": 0.45}
        with mock.patch.object(trl, "_now", return_value=datetime(2026, 6, 3, tzinfo=timezone.utc)):
            again = asyncio.run(trl.record_snapshots(
                self.client, md, self.forecast_fn, default_model="m",
                per_venue=3, reforecast_each_tick=True))
        self.assertEqual(again, 2)  # both re-forecast, not skipped
        snaps = [e for (k, _i), e in self.client.store.items() if k == trl.SNAPSHOT_KIND]
        self.assertEqual(len(snaps), 2)  # still one per (market, model, day)
        a = next(e for e in snaps if e.get("question") == "Will A happen?")
        self.assertAlmostEqual(a["model_probability"], 0.55)  # refreshed

    def test_hourly_price_points_store_liquidity_and_forecast_context_is_stateful(self):
        far = (datetime(2026, 6, 3, tzinfo=timezone.utc) + timedelta(days=10)).isoformat()
        md = _fake_market_data(far)
        seen_quotes = []

        async def forecast_fn(quote, top_k, model=None):
            seen_quotes.append(dict(quote))
            return {"model_probability": 0.70,
                    "market_probability": quote["probability"], "evidence_count": 2}

        with tempfile.TemporaryDirectory() as td:
            store = DuckDBStore(Path(td) / "track.duckdb")
            try:
                with mock.patch.object(trl, "_now", return_value=datetime(2026, 6, 3, 0, tzinfo=timezone.utc)):
                    self.assertEqual(asyncio.run(trl.record_snapshots(
                        store, md, forecast_fn, default_model="m", per_venue=1)), 2)

                md._poly.update(probability=0.46, volume=2200.0, liquidity=900.0,
                                yes_bid=0.45, yes_ask=0.47, last_trade_price=0.46)
                with mock.patch.object(trl, "_now", return_value=datetime(2026, 6, 3, 1, tzinfo=timezone.utc)):
                    self.assertEqual(trl.record_price_points(store, md), 2)

                hist = trl._get_price_history(store, "slug-a")
                self.assertEqual(hist[0]["probability"], 0.46)
                self.assertEqual(hist[0]["volume"], 2200.0)
                self.assertEqual(hist[0]["liquidity"], 900.0)
                self.assertEqual(hist[0]["bid"], 0.45)
                self.assertEqual(hist[0]["ask"], 0.47)

                with mock.patch.object(trl, "_now", return_value=datetime(2026, 6, 4, 0, tzinfo=timezone.utc)):
                    self.assertEqual(asyncio.run(trl.record_snapshots(
                        store, md, forecast_fn, default_model="m", per_venue=1)), 2)

                second_poly = next(
                    q for q in seen_quotes
                    if q.get("question") == "Will A happen?" and q.get("forecast_history")
                )
                self.assertEqual(second_poly["market_price_history"][0]["liquidity"], 900.0)
                self.assertEqual(second_poly["forecast_history"][0]["model_probability"], 0.70)
                self.assertEqual(second_poly["forecast_history"][0]["market_liquidity"], 500.0)
            finally:
                store.close()

    def test_short_horizon_markets_get_intraday_slots(self):
        ref = datetime(2026, 6, 3, 1, tzinfo=timezone.utc)
        # 5-day close: inside the 6h tier but outside the 1h expiry tier (≤3d).
        close = (ref + timedelta(days=5)).isoformat()
        md = _fake_market_data(close)
        with mock.patch.object(trl, "_now", return_value=ref):
            self.assertEqual(asyncio.run(trl.record_snapshots(
                self.client, md, self.forecast_fn, default_model="m", per_venue=3,
                short_horizon_reforecast_lead_days=7, short_horizon_slot_hours=6,
                expiry_reforecast_lead_days=3, expiry_slot_hours=1)), 2)
        # hour=5 is still in the 00-06 slot → no new snapshots
        with mock.patch.object(trl, "_now", return_value=ref.replace(hour=5)):
            self.assertEqual(asyncio.run(trl.record_snapshots(
                self.client, md, self.forecast_fn, default_model="m", per_venue=3,
                short_horizon_reforecast_lead_days=7, short_horizon_slot_hours=6,
                expiry_reforecast_lead_days=3, expiry_slot_hours=1)), 0)
        # hour=7 crosses into the 06-12 slot → new snapshots
        with mock.patch.object(trl, "_now", return_value=ref.replace(hour=7)):
            self.assertEqual(asyncio.run(trl.record_snapshots(
                self.client, md, self.forecast_fn, default_model="m", per_venue=3,
                short_horizon_reforecast_lead_days=7, short_horizon_slot_hours=6,
                expiry_reforecast_lead_days=3, expiry_slot_hours=1)), 2)
        snaps = [k[1] for k in self.client.store if k[0] == trl.SNAPSHOT_KIND]
        self.assertTrue(any(":2026-06-03T00" in k for k in snaps))
        self.assertTrue(any(":2026-06-03T06" in k for k in snaps))

    def test_expiry_markets_get_hourly_slots(self):
        ref = datetime(2026, 6, 3, 1, tzinfo=timezone.utc)
        # 2-day close: inside the 1h expiry tier (≤3d)
        close = (ref + timedelta(days=2)).isoformat()
        md = _fake_market_data(close)
        with mock.patch.object(trl, "_now", return_value=ref):
            self.assertEqual(asyncio.run(trl.record_snapshots(
                self.client, md, self.forecast_fn, default_model="m", per_venue=3,
                short_horizon_reforecast_lead_days=7, short_horizon_slot_hours=6,
                expiry_reforecast_lead_days=3, expiry_slot_hours=1)), 2)
        # hour=1 is a new 1h slot → records again
        with mock.patch.object(trl, "_now", return_value=ref.replace(hour=2)):
            self.assertEqual(asyncio.run(trl.record_snapshots(
                self.client, md, self.forecast_fn, default_model="m", per_venue=3,
                short_horizon_reforecast_lead_days=7, short_horizon_slot_hours=6,
                expiry_reforecast_lead_days=3, expiry_slot_hours=1)), 2)
        snaps = [k[1] for k in self.client.store if k[0] == trl.SNAPSHOT_KIND]
        self.assertTrue(any(":2026-06-03T01" in k for k in snaps))
        self.assertTrue(any(":2026-06-03T02" in k for k in snaps))

    def test_drop_stale_open_removes_orphaned_readings(self):
        rows = [
            {"ident": "fresh", "snapshot_date": "2026-06-11", "model_probability": 0.10},
            {"ident": "recent", "snapshot_date": "2026-06-10", "model_probability": 0.30},
            {"ident": "stale", "snapshot_date": "2026-06-07", "model_probability": 0.68},
        ]
        kept = {r["ident"] for r in trl._drop_stale_open(rows)}
        self.assertEqual(kept, {"fresh", "recent"})  # 06-07 is >2 days older than newest

    def test_drop_stale_open_window_is_per_model(self):
        # A cheap heartbeat model (crowd-follow) snapshots every day; the primary
        # LLM falls a few days behind. The LLM's snapshots must survive on their
        # own per-model window, not be evicted by the heartbeat's newer date.
        rows = [
            {"ident": "m1", "model": "crowd-follow", "snapshot_date": "2026-06-24"},
            {"ident": "m1", "model": "gpt-oss-120b", "snapshot_date": "2026-06-21"},
            {"ident": "m1", "model": "gpt-oss-120b", "snapshot_date": "2026-06-15"},
        ]
        kept = {(r["model"], r["snapshot_date"]) for r in trl._drop_stale_open(rows)}
        # gpt-oss 06-21 kept (newest for its model); 06-15 dropped (>2d behind 06-21).
        self.assertEqual(kept, {
            ("crowd-follow", "2026-06-24"),
            ("gpt-oss-120b", "2026-06-21"),
        })

    def test_resolution_scores_all_snapshots_and_buckets_by_horizon(self):
        far = (datetime(2026, 6, 3, tzinfo=timezone.utc) + timedelta(days=10)).isoformat()
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

    def test_kalshi_uses_market_ticker_not_series_url(self):
        far = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        md = _fake_market_data(far)
        md._kalshi.update({
            "market_url": "https://kalshi.com/markets/kxnflretire",
            "ident": "KXNFLRETIRE-MSTAFFORD9-2627",
        })

        def fetch_kalshi(ticker):
            if ticker != "KXNFLRETIRE-MSTAFFORD9-2627":
                raise md.MarketDataError(f"wrong ticker: {ticker}")
            return dict(md._kalshi)

        md.fetch_kalshi = fetch_kalshi
        self._record(md, "2026-06-03")
        snaps = [e for (k, _i), e in self.client.store.items() if k == trl.SNAPSHOT_KIND]
        kalshi_snap = next(e for e in snaps if e.get("platform") == "Kalshi")
        self.assertEqual(kalshi_snap.get("ident"), "KXNFLRETIRE-MSTAFFORD9-2627")

        with mock.patch.object(trl, "_now", return_value=datetime(2026, 6, 4, 9, tzinfo=timezone.utc)):
            self.assertEqual(trl.record_price_points(self.client, md), 2)
        pts = [e for (k, _i), e in self.client.store.items() if k == trl.PRICE_KIND]
        self.assertTrue(any(
            p.get("platform") == "Kalshi"
            and p.get("ident") == "KXNFLRETIRE-MSTAFFORD9-2627"
            for p in pts
        ))

        agg = trl.aggregate(self.client, model="m", variant="v", temperature=0.0)
        self.assertTrue(any(e.get("platform") == "Kalshi" for e in agg["edge_board"]))

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

    def test_backfill_uses_crowd_follow_resolved_reference_when_primary_missing(self):
        with tempfile.TemporaryDirectory() as td:
            store = DuckDBStore(Path(td) / "track.duckdb")
            try:
                key = store.key(
                    trl.SNAPSHOT_KIND,
                    "Polymarket:slug-a:crowd-follow:2026-06-03",
                )
                ref = Entity(key)
                ref.update(
                    platform="Polymarket",
                    ident="slug-a",
                    model="crowd-follow",
                    question="Will A happen?",
                    market_url="https://polymarket.com/market/slug-a",
                    description="",
                    resolution_criteria="",
                    publish_time="2026-06-01T00:00:00+00:00",
                    snapshot_ts="2026-06-03T00:00:00+00:00",
                    snapshot_date="2026-06-03",
                    model_probability=0.4,
                    market_probability=0.4,
                    close_time="2026-06-04T00:00:00+00:00",
                    lead_time_days=1.0,
                    horizon="1-3d",
                    category="World",
                    market_volume=100.0,
                    market_liquidity=50.0,
                    evidence_count=0,
                    resolved=True,
                    outcome=1,
                    resolved_ts="2026-06-04T01:00:00+00:00",
                    model_brier=0.36,
                    market_brier=0.36,
                    model_correct=False,
                    domain="world",
                    entities=[],
                    rationale="",
                )
                store.put(ref)

                async def forecast_fn(quote, top_k, model=None):
                    return {
                        "model_probability": 0.72 if model == "council" else 0.65,
                        "market_probability": quote["probability"],
                        "evidence_count": 2,
                        "rationale": f"{model} backfill",
                    }

                wrote = asyncio.run(trl.backfill_missing_model_snapshots(
                    store,
                    forecast_fn,
                    models=["council", "gpt-oss-120b", "crowd-follow"],
                    default_model="council",
                ))

                self.assertEqual(wrote, 2)
                agg = trl.aggregate(store, model="council", variant="v", temperature=0.0)
                self.assertEqual(agg["n_snapshots_resolved"], 0)
                self.assertEqual(agg["n_markets_resolved"], 0)
                self.assertEqual(agg["primary_n_snapshots_resolved"], 0)
                self.assertEqual(agg["primary_n_markets_resolved"], 0)
                self.assertIsNone(agg["overall"])
            finally:
                store.close()


class DigestTests(unittest.TestCase):
    def test_empty_digest(self):
        out = trl.format_digest(None)
        self.assertIn("Foresea forecast track record", out)
        self.assertIn("foresea.ink/track-record", out)

    def test_populated_digest(self):
        agg = {"n_snapshots_resolved": 12, "n_markets_resolved": 8, "n_markets_open": 5,
               "overall": {"accuracy": 0.75, "model_brier": 0.18, "market_brier": 0.22,
                           "skill_vs_market": 0.04},
               "by_horizon": [{"horizon": "7-14d", "n": 5, "skill_vs_market": 0.06}]}
        out = trl.format_digest(agg)
        self.assertIn("8 resolved", out)
        self.assertIn("75%", out)
        self.assertIn("beating the market", out)
        self.assertIn("7-14d", out)


class FileStoreTests(unittest.TestCase):
    def test_round_trips_entities_datetimes_and_queries(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "store.json"
            store = FileStore(path)
            ts = datetime(2026, 6, 5, 12, 30, tzinfo=timezone.utc)
            entity = Entity(store.key("Thing", "one"))
            entity.update(status="open", ts=ts)
            store.put(entity)

            reloaded = FileStore(path)
            got = reloaded.get(reloaded.key("Thing", "one"))
            self.assertIsNotNone(got)
            self.assertEqual(got.key.name, "one")
            self.assertEqual(got["ts"], ts)

            query = reloaded.query(kind="Thing")
            query.add_filter("status", "=", "open")
            self.assertEqual([e.key.name for e in query.fetch()], ["one"])

    def test_aggregate_persists_for_readback(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "store.json"
            store = FileStore(path)
            snap = Entity(store.key(trl.SNAPSHOT_KIND, "Kalshi:TICK:2026-06-05"))
            snap.update(
                platform="Kalshi",
                ident="TICK",
                question="Will it happen?",
                market_url="https://kalshi.com/markets/TICK",
                snapshot_ts=datetime(2026, 6, 5, 12, tzinfo=timezone.utc),
                snapshot_date="2026-06-05",
                model_probability=0.7,
                market_probability=0.4,
                lead_time_days=10.0,
                horizon="7-14d",
                resolved=True,
                outcome=1,
                resolved_ts=datetime(2026, 6, 6, 12, tzinfo=timezone.utc),
                model_brier=trl.brier(0.7, 1),
                market_brier=trl.brier(0.4, 1),
                model_correct=True,
            )
            store.put(snap)

            agg = trl.aggregate(store, model="m", variant="v", temperature=0.0)
            self.assertEqual(agg["n_snapshots_resolved"], 1)

            reloaded = FileStore(path)
            readback = trl.read_aggregate(reloaded)
            self.assertIsNotNone(readback)
            self.assertEqual(readback["overall"]["accuracy"], 1.0)
            self.assertEqual(readback["by_horizon"][0]["horizon"], "7-14d")

    def test_aggregate_headline_counts_comparable_llm_cohort_not_all_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "store.json"
            store = FileStore(path)
            rows = [
                ("Kalshi", "A", "council", 0.7, 0.4, 1),
                ("Kalshi", "A", "gpt-oss-120b", 0.65, 0.4, 1),
                ("Kalshi", "B", "gpt-oss-120b", 0.3, 0.6, 0),
                ("Kalshi", "B", "crowd-follow", 0.6, 0.6, 0),
            ]
            for platform, ident, model, model_p, market_p, outcome in rows:
                snap = Entity(store.key(
                    trl.SNAPSHOT_KIND,
                    f"{platform}:{ident}:{model}:2026-06-05",
                ))
                snap.update(
                    platform=platform,
                    ident=ident,
                    model=model,
                    question=f"Will {ident} happen?",
                    market_url=f"https://kalshi.com/markets/{ident}",
                    snapshot_ts=datetime(2026, 6, 5, 12, tzinfo=timezone.utc),
                    snapshot_date="2026-06-05",
                    model_probability=model_p,
                    market_probability=market_p,
                    lead_time_days=10.0,
                    horizon="7-14d",
                    resolved=True,
                    outcome=outcome,
                    resolved_ts=datetime(2026, 6, 6, 12, tzinfo=timezone.utc),
                    model_brier=trl.brier(model_p, outcome),
                    market_brier=trl.brier(market_p, outcome),
                    model_correct=(model_p >= 0.5) == (outcome == 1),
                )
                store.put(snap)

            agg = trl.aggregate(store, model="council", variant="v", temperature=0.0)
            self.assertEqual(agg["n_snapshots_resolved"], 2)
            self.assertEqual(agg["n_markets_resolved"], 2)
            self.assertEqual(agg["primary_model"], "council")
            self.assertEqual(agg["primary_n_snapshots_resolved"], 1)
            self.assertEqual(agg["primary_n_markets_resolved"], 1)
            self.assertEqual(agg["paper_pnl"]["flat"]["n_bets"], 2)
            self.assertEqual(agg["paper_pnl"]["crowd_baseline"]["n_bets"], 2)
            self.assertEqual(agg["primary_paper_pnl"]["flat"]["n_bets"], 1)
            comparison = {m["model"]: m for m in agg["models_comparison"]}
            self.assertEqual(comparison["crowd-follow"]["n_snapshots_resolved"], 2)

    def test_aggregate_excludes_diagnostic_backfill_snapshots(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "store.json"
            store = FileStore(path)
            rows = [
                ("Kalshi:TICK:gpt-oss-120b:2026-06-05", "gpt-oss-120b", 0.7),
                ("Kalshi:TICK:kimi-k2.6:2026-06-05_backfill", "kimi-k2.6", 0.65),
            ]
            for key_id, model, model_p in rows:
                snap = Entity(store.key(trl.SNAPSHOT_KIND, key_id))
                snap.update(
                    platform="Kalshi",
                    ident="TICK",
                    model=model,
                    question="Will TICK happen?",
                    market_url="https://kalshi.com/markets/TICK",
                    snapshot_ts=datetime(2026, 6, 5, 12, tzinfo=timezone.utc),
                    snapshot_date="2026-06-05",
                    model_probability=model_p,
                    market_probability=0.4,
                    lead_time_days=10.0,
                    horizon="7-14d",
                    resolved=True,
                    outcome=1,
                    resolved_ts=datetime(2026, 6, 6, 12, tzinfo=timezone.utc),
                    model_brier=trl.brier(model_p, 1),
                    market_brier=trl.brier(0.4, 1),
                    model_correct=True,
                )
                store.put(snap)

            agg = trl.aggregate(store, model="gpt-oss-120b", variant="v", temperature=0.0)
            self.assertEqual(agg["n_snapshots_resolved"], 1)
            self.assertEqual(agg["paper_pnl"]["flat"]["n_bets"], 1)
            comparison = {m["model"]: m for m in agg["models_comparison"]}
            self.assertNotIn("kimi-k2.6", comparison)


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


class EdgeAnalyticsTests(unittest.TestCase):
    """Edge calibration, lead/lag, and the live edge board."""

    def _res(self, model_p, market_p, outcome):
        return {"model_probability": model_p, "market_probability": market_p,
                "outcome": outcome,
                "model_brier": trl.brier(model_p, outcome),
                "market_brier": trl.brier(market_p, outcome),
                "model_correct": (model_p >= 0.5) == (outcome == 1)}

    def test_edge_calibration_significant_when_disagreement_pays(self):
        # 20 forecasts disagreeing by 30pp and resolving in the model's favour.
        resolved = [self._res(0.8, 0.5, 1) for _ in range(20)]
        buckets = {b["edge_bucket"]: b for b in trl.edge_calibration(resolved)}
        self.assertIn("20pp+", buckets)
        b = buckets["20pp+"]
        self.assertEqual(b["n"], 20)
        self.assertGreater(b["skill_vs_market"], 0)
        self.assertTrue(b["skill_significant"])          # CI lower bound clears 0

    def test_edge_calibration_not_significant_when_disagreement_is_coinflip(self):
        resolved = ([self._res(0.8, 0.5, 1)] * 10) + ([self._res(0.8, 0.5, 0)] * 10)
        buckets = {b["edge_bucket"]: b for b in trl.edge_calibration(resolved)}
        self.assertFalse(buckets["20pp+"]["skill_significant"])

    def test_lead_lag_detects_market_moving_toward_model(self):
        now = datetime.now(timezone.utc)

        def snap(ts, m, k, outcome):
            return {"snapshot_ts": ts, "model_probability": m,
                    "market_probability": k, "outcome": outcome}

        by_market = {("Polymarket", "A"): [
            snap(now, 0.8, 0.5, 1),                         # disagree by +30pp
            snap(now + timedelta(hours=2), 0.8, 0.7, 1),    # price drifts toward model
        ]}
        ll = trl.lead_lag(by_market)
        self.assertEqual(ll["n_markets"], 1)
        self.assertEqual(ll["market_converged_to_model_pct"], 1.0)
        self.assertEqual(ll["model_right_pct"], 1.0)
        self.assertAlmostEqual(ll["avg_convergence_fraction"], (0.7 - 0.5) / (0.8 - 0.5), places=3)

    def test_edge_board_ranks_by_disagreement_and_annotates(self):
        now = datetime.now(timezone.utc)
        open_rows = [
            {"platform": "Polymarket", "ident": "A", "model_probability": 0.8,
             "market_probability": 0.5, "snapshot_ts": now, "question": "Q1",
             "market_url": "u1", "horizon": "30d+", "lead_time_days": 40.0},
            {"platform": "Kalshi", "ident": "B", "model_probability": 0.52,
             "market_probability": 0.5, "snapshot_ts": now, "question": "Q2",
             "market_url": "u2", "horizon": "7-14d", "lead_time_days": 10.0},
        ]
        edge_calib = [{"edge_bucket": "20pp+", "n": 20, "skill_vs_market": 0.05,
                       "skill_ci_low": 0.02, "skill_significant": True}]
        board = trl.build_edge_board(open_rows, {"A": 0.5, "B": 0.5}, edge_calib)
        self.assertEqual([e["platform"] for e in board], ["Polymarket", "Kalshi"])  # by abs edge
        top = board[0]
        self.assertEqual(top["edge"], 0.3)
        self.assertEqual(top["edge_bucket"], "20pp+")
        self.assertEqual(top["stance"], "model_above_market")
        self.assertTrue(top["track_record"]["skill_significant"])
        self.assertIsNone(board[1]["track_record"])  # 0-5pp gap has no calibration row

    def test_paper_pnl_positive_on_winning_edge(self):
        resolved = [self._res(0.8, 0.5, 1) for _ in range(10)]
        pnl = trl.paper_pnl(resolved, trl.edge_calibration(resolved))
        self.assertEqual(pnl["flat"]["n_bets"], 10)
        self.assertEqual(pnl["flat"]["win_rate"], 1.0)
        self.assertAlmostEqual(pnl["flat"]["roi"], 1.0, places=3)   # (1-0.5)/0.5 per win
        self.assertGreater(pnl["flat"]["pnl"], 0)
        self.assertEqual(len(pnl["flat"]["equity_curve"]), 10)
        self.assertIsNotNone(pnl["validated_only"])                 # 20pp+ bucket is significant
        self.assertEqual(pnl["validated_only"]["n_bets"], 10)

    def test_growth_curve_endpoint_matches_roi(self):
        # The edge-board chart plots growth_curve; its last point must equal
        # 100*(1+roi) so the line ends where the displayed ROI says it should.
        # Regression guard: a compounded curve silently diverged from ROI.
        resolved = ([self._res(0.8, 0.5, 1) for _ in range(7)]
                    + [self._res(0.3, 0.5, 0) for _ in range(3)])
        pnl = trl.paper_pnl(resolved, trl.edge_calibration(resolved))
        for name, s in pnl.items():
            if not isinstance(s, dict) or not s.get("growth_curve"):
                continue
            roi = s["roi"]
            self.assertAlmostEqual(
                s["growth_curve"][-1], 100.0 * (1.0 + roi), places=2,
                msg=f"{name}: growth_curve endpoint != 100*(1+roi)")

    def test_paper_pnl_is_net_of_kalshi_fees(self):
        # Same winning bet on Kalshi (price-based fee) vs Polymarket (fee-free):
        # Kalshi ROI must be lower, and the fee must be reported and positive.
        kalshi = [dict(self._res(0.8, 0.5, 1), platform="Kalshi") for _ in range(10)]
        poly = [dict(self._res(0.8, 0.5, 1), platform="Polymarket") for _ in range(10)]
        pk = trl.paper_pnl(kalshi, [])
        pp = trl.paper_pnl(poly, [])
        self.assertGreater(pk["flat"]["fees"], 0)            # Kalshi charged a fee
        self.assertEqual(pp["flat"]["fees"], 0)              # Polymarket fee-free
        self.assertLess(pk["flat"]["roi"], pp["flat"]["roi"])  # fees drag real ROI down
        # Fee ≈ 0.07 * stake * (1 - p_side) = 0.07 * 1 * 0.5 = 0.035 per bet.
        self.assertAlmostEqual(pk["flat"]["fees"], 10 * 0.035, places=4)

    def test_paper_pnl_bets_the_models_own_side_not_against_it(self):
        # Model says 60% YES (its call is YES) while the market is higher at 75%.
        # The bet must follow the model's call (YES) — so a YES outcome WINS,
        # never staking against the model. win_rate must equal accuracy (1.0).
        resolved = [self._res(0.6, 0.75, 1) for _ in range(8)]
        pnl = trl.paper_pnl(resolved, [])
        self.assertEqual(pnl["flat"]["win_rate"], 1.0)               # model's YES call was right
        self.assertGreater(pnl["flat"]["pnl"], 0)                    # bought YES @ 0.75, paid off
        # Sanity: win_rate tracks accuracy (model_correct) on the same set.
        acc = sum(r["model_correct"] for r in resolved) / len(resolved)
        self.assertEqual(pnl["flat"]["win_rate"], acc)

    def test_paper_pnl_bets_every_forecast_without_min_edge(self):
        # No minimum edge: even a 2pp disagreement is bet (the organizing axis is
        # lead time, not edge size). None only when there are no resolved forecasts.
        resolved = [self._res(0.52, 0.5, 1) for _ in range(5)]      # 2pp edge
        pnl = trl.paper_pnl(resolved, [])
        self.assertIsNotNone(pnl)
        self.assertEqual(pnl["flat"]["n_bets"], 5)
        self.assertEqual(pnl["min_edge"], 0.0)
        self.assertIsNone(trl.paper_pnl([], []))                    # nothing resolved

    def test_paper_pnl_public_bet_log_dedupes_market_model(self):
        resolved = [
            dict(
                self._res(0.8, 0.5, 1),
                platform="Polymarket",
                ident="same-market",
                model="council",
                snapshot_ts=f"2026-06-0{i}T00:00:00+00:00",
            )
            for i in range(1, 4)
        ]
        pnl = trl.paper_pnl(resolved, [])
        self.assertEqual(pnl["flat"]["n_bets"], 3)
        self.assertEqual(len(pnl["bets"]), 1)
        self.assertEqual(pnl["bets"][0]["snapshot_ts"], "2026-06-01T00:00:00+00:00")

    def test_edge_board_links_to_lead_time_track_record(self):
        now = datetime.now(timezone.utc)
        # Open market closing ~40 days out → "30d+" horizon bucket.
        open_rows = [{"platform": "Polymarket", "ident": "A", "model_probability": 0.8,
                      "market_probability": 0.5, "snapshot_ts": now, "question": "Q1",
                      "market_url": "u1", "close_time": (now + timedelta(days=40)).isoformat()}]
        horizon_calib = [{"horizon": "30d+", "n": 12, "accuracy": 0.58, "skill_vs_market": 0.04,
                          "skill_ci_low": 0.01, "skill_significant": True}]
        board = trl.build_edge_board(open_rows, {"A": 0.5}, [], horizon_calib)
        ltr = board[0]["lead_track_record"]
        self.assertEqual(board[0]["lead_bucket"], "30d+")
        self.assertEqual(ltr["horizon"], "30d+")
        self.assertTrue(ltr["skill_significant"])
        self.assertEqual(ltr["n"], 12)
        self.assertEqual(ltr["accuracy"], 0.58)

    def test_edge_board_accepts_datetime_close_time(self):
        now = datetime.now(timezone.utc)
        open_rows = [{"platform": "Polymarket", "ident": "A", "model_probability": 0.8,
                      "market_probability": 0.5, "snapshot_ts": now, "question": "Q1",
                      "market_url": "u1", "close_time": now + timedelta(days=40)}]
        board = trl.build_edge_board(open_rows, {"A": 0.5}, [])
        self.assertEqual(len(board), 1)
        self.assertEqual(board[0]["question"], "Q1")

    def test_paper_pnl_validated_only_skips_unproven_buckets(self):
        # Coin-flip disagreements -> bucket not significant -> validated_only empty.
        resolved = ([self._res(0.8, 0.5, 1)] * 5) + ([self._res(0.8, 0.5, 0)] * 5)
        pnl = trl.paper_pnl(resolved, trl.edge_calibration(resolved))
        self.assertIsNotNone(pnl["flat"])
        self.assertIsNone(pnl["validated_only"])

    def test_models_comparison_ranks_models_by_paper_edge(self):
        # gpt-oss disagrees and wins; gemma disagrees and loses -> gpt-oss ranks first.
        good = [dict(self._res(0.8, 0.5, 1), model="gpt-oss-120b", platform="P", ident=f"g{i}")
                for i in range(8)]
        bad = [dict(self._res(0.8, 0.5, 0), model="gemma-4-31b-it", platform="P", ident=f"b{i}")
               for i in range(8)]
        comp = trl.build_models_comparison(good + bad, default_model="gpt-oss-120b")
        self.assertEqual([m["model"] for m in comp], ["gpt-oss-120b", "gemma-4-31b-it"])
        self.assertEqual(comp[0]["n_snapshots_resolved"], 8)
        self.assertGreater(comp[0]["paper_roi"], comp[1]["paper_roi"])

    def test_models_comparison_defaults_missing_model_label(self):
        rows = [dict(self._res(0.8, 0.5, 1), platform="P", ident=f"x{i}") for i in range(3)]
        comp = trl.build_models_comparison(rows, default_model="gpt-oss-120b")
        self.assertEqual(comp[0]["model"], "gpt-oss-120b")  # missing model -> primary

    def test_edge_board_trade_direction_and_payout_odds(self):
        now = datetime.now(timezone.utc)
        rows = [
            {"platform": "P", "ident": "A", "model_probability": 0.9, "market_probability": 0.02,
             "snapshot_ts": now, "question": "AGI", "market_url": "u", "horizon": "30d+", "lead_time_days": 40.0},
            {"platform": "P", "ident": "B", "model_probability": 0.1, "market_probability": 0.40,
             "snapshot_ts": now, "question": "NO", "market_url": "u", "horizon": "30d+", "lead_time_days": 40.0},
        ]
        bd = {b["question"]: b for b in trl.build_edge_board(rows, {"A": 0.02, "B": 0.40}, [])}
        a = bd["AGI"]  # model 0.9 > market 0.02 -> buy YES cheap, ~49:1
        self.assertEqual((a["side"], a["entry_price"]), ("YES", 0.02))
        self.assertEqual(a["payout_odds"], round((1 - 0.02) / 0.02, 1))
        b = bd["NO"]   # model 0.1 < market 0.40 -> buy NO at 0.60
        self.assertEqual((b["side"], b["entry_price"]), ("NO", 0.6))
        self.assertEqual(b["payout_odds"], round((1 - 0.6) / 0.6, 1))

    def test_edge_board_uses_latest_live_price_over_snapshot(self):
        now = datetime.now(timezone.utc)
        rows = [{"platform": "Polymarket", "ident": "A", "model_probability": 0.8,
                 "market_probability": 0.79, "snapshot_ts": now, "question": "Q",
                 "market_url": "u", "horizon": "30d+", "lead_time_days": 40.0}]
        # Snapshot price was 0.79 (tiny gap); live price has since dropped to 0.50.
        board = trl.build_edge_board(rows, {"A": 0.50}, [])
        self.assertEqual(board[0]["market_probability"], 0.5)
        self.assertEqual(board[0]["edge"], 0.3)

    def test_is_similar_question(self):
        q1 = "Will the Federal Reserve cut interest rates in September 2026?"
        q2 = "Will the Fed cut interest rates in September 2026?"
        self.assertTrue(trl._is_similar_question(q1, q2))

        q3 = "Will the Fed cut interest rates in September 2025?"
        self.assertFalse(trl._is_similar_question(q1, q3))

        q4 = "Will SpaceX launch Starship in September 2026?"
        self.assertFalse(trl._is_similar_question(q1, q4))

    def test_build_arbitrage_board(self):
        now = datetime.now(timezone.utc)
        open_rows = [
            {
                "platform": "Polymarket",
                "ident": "fed-sept-26",
                "question": "Will the Fed cut interest rates in September 2026?",
                "market_url": "https://polymarket.com/fed-sept-26",
                "snapshot_ts": now,
            },
            {
                "platform": "Kalshi",
                "ident": "FED-26SEPT",
                "question": "Will the Federal Reserve cut interest rates in September 2026?",
                "market_url": "https://kalshi.com/FED-26SEPT",
                "snapshot_ts": now,
            }
        ]
        latest_prices = {
            "fed-sept-26": 0.40,
            "FED-26SEPT": 0.55
        }
        signals = trl.build_arbitrage_board(open_rows, latest_prices)
        self.assertEqual(len(signals), 1)
        sig = signals[0]
        self.assertEqual(sig["price1"], 0.40)
        self.assertEqual(sig["price2"], 0.55)
        self.assertEqual(sig["arbitrage_gap"], 0.15)
        self.assertEqual(sig["total_cost"], 0.85)
        self.assertEqual(sig["net_profit"], 0.15)
        self.assertAlmostEqual(sig["roi"], 0.1765, places=4)


if __name__ == "__main__":
    unittest.main()
