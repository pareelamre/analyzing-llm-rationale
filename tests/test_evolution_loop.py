"""Evolution loop: agent forecasts enrol markets (Datastore pointer), the bridge
endpoints serve/flip them, all against an in-memory fake Datastore."""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

os.environ["ANALYTICS_DB"] = str(Path(tempfile.gettempdir()) / "foresea_test_evo.duckdb")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from analyzing_llm_rationale import server  # noqa: E402
from analyzing_llm_rationale.server import app  # noqa: E402


class FakeKey:
    def __init__(self, kind, id_=None):
        self.kind, self.id = kind, id_


class FakeEntity(dict):
    def __init__(self, key=None, exclude_from_indexes=()):
        super().__init__()
        self.key = key


class _FakeQuery:
    def __init__(self, store, kind):
        self.store, self.kind, self.filters, self.order = store, kind, [], []

    def add_filter(self, name, op, value):
        self.filters.append((name, op, value))

    def fetch(self, limit=None):
        rows = [e for (k, _i), e in self.store.items()
                if k == self.kind and all(e.get(n) == v for n, _o, v in self.filters)]
        if self.order:
            field = self.order[0].lstrip("-")
            rows.sort(key=lambda e: e.get(field) or datetime.min.replace(tzinfo=timezone.utc))
        return rows[:limit] if limit else rows


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
        return _FakeQuery(self.store, kind)

    def delete_multi(self, keys):
        for k in keys:
            self.store.pop((k.kind, k.id), None)


def _ts(day):
    return datetime(2026, 6, day, tzinfo=timezone.utc)


class EvolutionLoopTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeClient()
        fake_module = types.ModuleType("google.cloud.datastore")
        fake_module.Entity = FakeEntity
        self.patches = [
            mock.patch.object(server, "_get_datastore", return_value=self.fake),
            mock.patch.dict(sys.modules, {"google.cloud.datastore": fake_module}),
        ]
        for p in self.patches:
            p.start()
        self.client = TestClient(app)

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def _seed(self, ident, enrolled, day, enrolled_ts=None):
        k = self.fake.key(server._ENROLLED_MARKET_KIND, f"polymarket:{ident}")
        e = FakeEntity(k)
        e.update(platform="polymarket", ident=ident, market_url="https://u", question="q",
                 seen_count=1, enrolled=enrolled, first_seen_ts=_ts(day))
        if enrolled_ts:
            e["enrolled_ts"] = enrolled_ts
        self.fake.put(e)

    # ── enrollment helper ──
    def test_enroll_creates_then_dedups(self):
        server._enroll_market_sync("polymarket", "slug-x", "https://u", "Q?", "predict")
        server._enroll_market_sync("polymarket", "slug-x", "https://u", "Q?", "predict")
        e = self.fake.store[(server._ENROLLED_MARKET_KIND, "polymarket:slug-x")]
        self.assertEqual(e["seen_count"], 2)
        self.assertFalse(e["enrolled"])
        self.assertEqual(e["request_source"], "predict")

    def test_enroll_noop_without_datastore(self):
        with mock.patch.object(server, "_get_datastore", return_value=None):
            server._enroll_market_sync("polymarket", "slug-y", "https://u", "q", "predict")  # no raise

    # ── pending-markets endpoint ──
    def test_pending_lists_only_unenrolled(self):
        self._seed("m0", False, 1)
        self._seed("m1", False, 2)
        self._seed("m2", True, 3)
        with mock.patch.object(server, "_TRACK_RECORD_TOKEN", "tok"):
            r = self.client.get("/track-record/pending-markets", headers={"X-Track-Token": "tok"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual({m["ident"] for m in r.json()["markets"]}, {"m0", "m1"})

    def test_pending_auth(self):
        with mock.patch.object(server, "_TRACK_RECORD_TOKEN", "tok"):
            self.assertEqual(self.client.get("/track-record/pending-markets").status_code, 401)
        with mock.patch.object(server, "_TRACK_RECORD_TOKEN", None):
            r = self.client.get("/track-record/pending-markets", headers={"X-Track-Token": "x"})
            self.assertEqual(r.status_code, 503)

    # ── mark-enrolled endpoint ──
    def test_mark_enrolled_flips_and_prunes(self):
        self._seed("m0", False, 1)
        old = datetime.now(timezone.utc).replace(year=2026, month=1)
        self._seed("m9", True, 1, enrolled_ts=old)  # stale tracked → pruned
        with mock.patch.object(server, "_TRACK_RECORD_TOKEN", "tok"):
            r = self.client.post("/track-record/mark-enrolled",
                                 json={"idents": ["polymarket:m0"]},
                                 headers={"X-Track-Token": "tok"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["marked"], 1)
        self.assertTrue(self.fake.store[(server._ENROLLED_MARKET_KIND, "polymarket:m0")]["enrolled"])
        self.assertNotIn((server._ENROLLED_MARKET_KIND, "polymarket:m9"), self.fake.store)


class FeedbackTests(unittest.TestCase):
    # ── 4a: live calibration ──
    def test_calibrate_applies_when_warranted(self):
        live = {"calibration_model": {"applied": True, "breakpoints": [[0.0, 0.0], [1.0, 0.5]]}}
        with mock.patch.object(server, "_read_live_track_record", return_value=live):
            self.assertAlmostEqual(server._calibrate_probability(0.5), 0.25, places=3)

    def test_calibrate_passthrough_when_not_applied(self):
        with mock.patch.object(server, "_read_live_track_record",
                               return_value={"calibration_model": {"applied": False}}):
            self.assertEqual(server._calibrate_probability(0.8), 0.8)

    # ── 4b: model auto-selection ──
    def test_auto_select_picks_best_smart_strategy_model(self):
        live = {"models_comparison": [
            {"model": "gpt-oss-120b", "paper_roi_smart": 0.01},
            {"model": "gemma-4-26b-a4b-it", "paper_roi_smart": 0.10},
        ]}
        with mock.patch.dict(server._state, {"model_key": "gpt-oss-120b"}), \
                mock.patch.object(server, "_AUTO_SELECT_MODEL", True), \
                mock.patch.object(server, "_read_live_track_record", return_value=live):
            self.assertEqual(server._auto_selected_model(), "gemma-4-26b-a4b-it")

    def test_auto_select_respects_margin(self):
        live = {"models_comparison": [
            {"model": "gpt-oss-120b", "paper_roi_smart": 0.10},
            {"model": "gemma-4-26b-a4b-it", "paper_roi_smart": 0.105},  # +0.005 < 0.02 margin
        ]}
        with mock.patch.dict(server._state, {"model_key": "gpt-oss-120b"}), \
                mock.patch.object(server, "_AUTO_SELECT_MODEL", True), \
                mock.patch.object(server, "_read_live_track_record", return_value=live):
            self.assertIsNone(server._auto_selected_model())

    def test_auto_select_requires_incumbent_model_history(self):
        live = {"models_comparison": [
            {"model": "council", "paper_roi_smart": 0.20},
        ]}
        with mock.patch.dict(server._state, {"model_key": "test-model"}), \
                mock.patch.object(server, "_AUTO_SELECT_MODEL", True), \
                mock.patch.object(server, "_read_live_track_record", return_value=live):
            self.assertIsNone(server._auto_selected_model())


if __name__ == "__main__":
    unittest.main()
