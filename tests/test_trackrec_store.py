"""Tests for the normalized `markets` table and its opt-in hydration helpers.

Market-level fields (question, description, resolution_criteria, market_url,
publish_time, close_time, category, domain) used to be duplicated onto every
forecast_snapshot row -- these tests cover the normalized replacement: one row
per (platform, ident) in `markets`, joined in on read only where explicitly
requested (never automatically, since a few callers fetch-then-put() the same
entity back and must not silently re-persist a joined-in value)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.trackrec_store import MARKET_FIELDS, DuckDBStore, Entity, FileStore


class MarketKindTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = DuckDBStore(Path(self.temp_dir.name) / "track.duckdb")

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_market_kind_roundtrips_via_generic_get_put(self):
        key = self.store.key("Market", "Polymarket:slug-a")
        entity = Entity(key)
        entity.update(platform="Polymarket", ident="slug-a", question="Will A happen?",
                       market_url="https://polymarket.com/slug-a")
        self.store.put(entity)

        fetched = self.store.get(key)
        self.assertEqual(fetched["question"], "Will A happen?")
        self.assertEqual(fetched["market_url"], "https://polymarket.com/slug-a")

    def test_hydrate_markets_fills_missing_fields_from_markets_table(self):
        market = Entity(self.store.key("Market", "Polymarket:slug-a"))
        market.update(platform="Polymarket", ident="slug-a", question="Will A happen?",
                       market_url="https://polymarket.com/slug-a", description="A market about A.",
                       domain="world")
        self.store.put(market)

        snap = Entity(self.store.key("ForecastSnapshot", "Polymarket:slug-a:m:slot"))
        snap.update(platform="Polymarket", ident="slug-a", model="m", model_probability=0.6)
        # Row itself carries no market-level fields -- the normalized shape.

        hydrated = self.store.hydrate_market(snap)
        self.assertEqual(hydrated["question"], "Will A happen?")
        self.assertEqual(hydrated["market_url"], "https://polymarket.com/slug-a")
        self.assertEqual(hydrated["description"], "A market about A.")
        self.assertEqual(hydrated["domain"], "world")
        # Fields with no markets-table value stay absent/empty, not crash.
        self.assertFalse(hydrated.get("category"))

    def test_hydrate_markets_prefers_the_rows_own_value_when_present(self):
        # Historical rows (written before normalization) already carry their
        # own value -- hydration must not overwrite it even if `markets`
        # somehow disagrees.
        market = Entity(self.store.key("Market", "Polymarket:slug-a"))
        market.update(platform="Polymarket", ident="slug-a", question="A newer question text")
        self.store.put(market)

        snap = Entity(self.store.key("ForecastSnapshot", "Polymarket:slug-a:m:slot"))
        snap.update(platform="Polymarket", ident="slug-a", model="m",
                     question="The original stored question")

        hydrated = self.store.hydrate_market(snap)
        self.assertEqual(hydrated["question"], "The original stored question")

    def test_hydrate_markets_does_not_mutate_the_input_entity(self):
        # Critical: some callers fetch an entity, hydrate it for read-only use,
        # but later put() the *original* (unhydrated) object back. If hydrate
        # mutated in place, that put() would re-persist the joined-in value
        # onto the row and silently undo the normalization.
        market = Entity(self.store.key("Market", "Polymarket:slug-a"))
        market.update(platform="Polymarket", ident="slug-a", question="Will A happen?")
        self.store.put(market)

        snap = Entity(self.store.key("ForecastSnapshot", "Polymarket:slug-a:m:slot"))
        snap.update(platform="Polymarket", ident="slug-a", model="m")

        hydrated = self.store.hydrate_market(snap)
        self.assertEqual(hydrated["question"], "Will A happen?")
        self.assertIsNone(snap.get("question"))  # original untouched
        self.assertIsNot(hydrated, snap)

    def test_hydrate_markets_with_no_matching_market_row_is_a_noop(self):
        snap = Entity(self.store.key("ForecastSnapshot", "Polymarket:unknown:m:slot"))
        snap.update(platform="Polymarket", ident="unknown", model="m", model_probability=0.5)
        hydrated = self.store.hydrate_market(snap)
        self.assertIsNone(hydrated.get("question"))
        self.assertEqual(hydrated["model_probability"], 0.5)
        # Must still be a genuine copy, not the same object -- a caller
        # relying on hydrate_market's "safe to mutate independently" contract
        # must get that guarantee even when there's nothing to fill in.
        self.assertIsNot(hydrated, snap)

    def test_hydrate_markets_always_returns_copies_even_when_fully_populated(self):
        snap = Entity(self.store.key("ForecastSnapshot", "Polymarket:slug-a:m:slot"))
        snap.update(platform="Polymarket", ident="slug-a", model="m",
                     **{f: f"stored-{f}" for f in MARKET_FIELDS})
        hydrated = self.store.hydrate_market(snap)
        self.assertIsNot(hydrated, snap)
        self.assertEqual(dict(hydrated), dict(snap))

    def test_hydrate_market_on_none_returns_none(self):
        self.assertIsNone(self.store.hydrate_market(None))

    def test_hydrate_markets_empty_list(self):
        self.assertEqual(self.store.hydrate_markets([]), [])

    def test_hydrate_markets_batch_handles_mixed_rows(self):
        m1 = Entity(self.store.key("Market", "Polymarket:a"))
        m1.update(platform="Polymarket", ident="a", question="Q-A")
        self.store.put(m1)
        m2 = Entity(self.store.key("Market", "Polymarket:b"))
        m2.update(platform="Polymarket", ident="b", question="Q-B")
        self.store.put(m2)

        snap_a = Entity(self.store.key("ForecastSnapshot", "Polymarket:a:m:slot"))
        snap_a.update(platform="Polymarket", ident="a", model="m")
        snap_b_old = Entity(self.store.key("ForecastSnapshot", "Polymarket:b:m:slot"))
        snap_b_old.update(platform="Polymarket", ident="b", model="m", question="Stored Q-B")

        out = self.store.hydrate_markets([snap_a, snap_b_old])
        by_ident = {e["ident"]: e for e in out}
        self.assertEqual(by_ident["a"]["question"], "Q-A")           # hydrated
        self.assertEqual(by_ident["b"]["question"], "Stored Q-B")    # kept own value


class FileStoreHydrationIsNoopTests(unittest.TestCase):
    """FileStore has no `markets` table -- hydration must be a harmless passthrough."""

    def test_hydrate_market_returns_entity_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            store = FileStore(Path(td) / "store.json")
            entity = Entity(store.key("ForecastSnapshot", "Polymarket:a:m:slot"))
            entity.update(platform="Polymarket", ident="a", question="Q")
            self.assertIs(store.hydrate_market(entity), entity)

    def test_hydrate_markets_returns_list_copy(self):
        with tempfile.TemporaryDirectory() as td:
            store = FileStore(Path(td) / "store.json")
            entities = [Entity(store.key("ForecastSnapshot", "k1")), Entity(store.key("ForecastSnapshot", "k2"))]
            self.assertEqual(store.hydrate_markets(entities), entities)


if __name__ == "__main__":
    unittest.main()
