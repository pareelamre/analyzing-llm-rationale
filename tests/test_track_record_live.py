"""Live track-record harness: record-once, resolve+score, aggregate.

Exercised with an in-memory fake Datastore, a fake market_data module, and a
stub forecast function — no network, no SCADS, no real Datastore."""
from __future__ import annotations

import asyncio
import contextlib
import sys
import types
import unittest
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

    def transaction(self):
        return contextlib.nullcontext()

    def query(self, kind):
        return _FakeQuery(self, kind)


class _FakeQuery:
    def __init__(self, client, kind):
        self.client = client
        self.kind = kind
        self.filters = []

    def add_filter(self, name, op, value):
        self.filters.append((name, op, value))

    def fetch(self):
        out = []
        for (k, _id), e in self.client.store.items():
            if k != self.kind:
                continue
            if all(op == "=" and e.get(n) == v for n, op, v in self.filters):
                out.append(e)
        return out


def _fake_market_data():
    md = types.ModuleType("market_data_fake")

    class MarketDataError(RuntimeError):
        pass

    md.MarketDataError = MarketDataError
    md.list_polymarket = lambda limit=3: [
        {"platform": "Polymarket", "question": "Will A happen?",
         "market_url": "https://polymarket.com/market/slug-a", "outcome": "Yes", "probability": 0.40}
    ]
    md.list_kalshi = lambda limit=3: [
        {"platform": "Kalshi", "question": "Will B happen?",
         "market_url": "https://kalshi.com/markets/TICKERB", "outcome": "Yes", "probability": 0.45}
    ]
    md.resolve_polymarket = lambda ident: 1 if ident == "slug-a" else None
    md.resolve_kalshi = lambda ident: 0 if ident == "TICKERB" else None
    return md


class TrackRecordLiveTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.md = _fake_market_data()
        fake_ds = types.ModuleType("google.cloud.datastore")
        fake_ds.Entity = FakeEntity
        self._p = mock.patch.dict(sys.modules, {"google.cloud.datastore": fake_ds})
        self._p.start()
        # Stub forecast: A -> model 0.70, B -> model 0.30 (market prob passed through).
        self.model_probs = {"Will A happen?": 0.70, "Will B happen?": 0.30}

        async def forecast_fn(quote, top_k):
            return (self.model_probs[quote["question"]], quote["probability"])

        self.forecast_fn = forecast_fn

    def tearDown(self):
        self._p.stop()

    def test_ident_parsing(self):
        self.assertEqual(trl.ident_from_url("Polymarket", "https://polymarket.com/market/slug-a"), "slug-a")
        self.assertEqual(trl.ident_from_url("Kalshi", "https://kalshi.com/markets/TICKERB"), "TICKERB")

    def test_record_is_once_per_market(self):
        n1 = asyncio.run(trl.record_new_forecasts(self.client, self.md, self.forecast_fn, per_venue=3))
        self.assertEqual(n1, 2)
        # Re-running records nothing new — the market is already tracked.
        n2 = asyncio.run(trl.record_new_forecasts(self.client, self.md, self.forecast_fn, per_venue=3))
        self.assertEqual(n2, 0)
        self.assertEqual(len([k for k in self.client.store if k[0] == trl.LIVE_KIND]), 2)

    def test_resolve_and_aggregate_skill_vs_market(self):
        asyncio.run(trl.record_new_forecasts(self.client, self.md, self.forecast_fn, per_venue=3))
        resolved = trl.resolve_open_forecasts(self.client, self.md)
        self.assertEqual(resolved, 2)

        agg = trl.aggregate(self.client, model="gpt-oss-120b", variant="v0", temperature=0.0)
        self.assertEqual(agg["n_resolved"], 2)
        self.assertEqual(agg["n_open"], 0)
        # A: model .7 vs outcome 1 -> correct; B: model .3 vs outcome 0 -> correct.
        self.assertEqual(agg["accuracy"], 1.0)
        # model brier = mean(.09, .09) = .09; market brier = mean(.36, .2025) = .28125
        self.assertAlmostEqual(agg["brier_score"], 0.09, places=4)
        self.assertAlmostEqual(agg["market_brier_score"], 0.28125, places=4)
        # skill vs market positive => model beat the market price.
        self.assertAlmostEqual(agg["skill_vs_market"], 0.19125, places=4)
        self.assertEqual(agg["source"], "live")

    def test_aggregate_empty_is_safe(self):
        agg = trl.aggregate(self.client, model="m", variant="v", temperature=0.0)
        self.assertEqual(agg["n_resolved"], 0)
        self.assertIsNone(agg["brier_score"])
        self.assertEqual(agg["log"], [])

    def test_read_aggregate_roundtrip(self):
        asyncio.run(trl.record_new_forecasts(self.client, self.md, self.forecast_fn, per_venue=3))
        trl.resolve_open_forecasts(self.client, self.md)
        trl.aggregate(self.client, model="gpt-oss-120b", variant="v0", temperature=0.0)
        live = trl.read_aggregate(self.client)
        self.assertIsNotNone(live)
        self.assertEqual(live["n_resolved"], 2)

    def test_unresolved_market_stays_open(self):
        asyncio.run(trl.record_new_forecasts(self.client, self.md, self.forecast_fn, per_venue=3))
        # Market that never resolves.
        self.md.resolve_polymarket = lambda ident: None
        self.md.resolve_kalshi = lambda ident: None
        resolved = trl.resolve_open_forecasts(self.client, self.md)
        self.assertEqual(resolved, 0)
        agg = trl.aggregate(self.client, model="m", variant="v", temperature=0.0)
        self.assertEqual(agg["n_open"], 2)
        self.assertEqual(agg["n_resolved"], 0)


if __name__ == "__main__":
    unittest.main()
