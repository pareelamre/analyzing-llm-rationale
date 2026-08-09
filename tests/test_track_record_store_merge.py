from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.trackrec_store import DuckDBStore, Entity

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "merge_track_record_store.py"
_SPEC = importlib.util.spec_from_file_location("merge_track_record_store", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
merge_track_record_store = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(merge_track_record_store)


class TrackRecordStoreMergeTests(unittest.TestCase):
    def test_merge_upserts_model_snapshot_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target_path = Path(td) / "target.duckdb"
            source_path = Path(td) / "source.duckdb"

            target = DuckDBStore(target_path)
            original = Entity(target.key("ForecastSnapshot", "kalshi:RATE:gpt:slot"))
            original.update({
                "platform": "Kalshi",
                "ident": "RATE",
                "model": "gpt",
                "snapshot_date": "2026-07-29T23:30",
                "model_probability": 0.44,
                "market_probability": 0.50,
                "resolved": False,
            })
            target.put(original)
            target._con.close()

            source = DuckDBStore(source_path)
            revised = Entity(source.key("ForecastSnapshot", "kalshi:RATE:gpt:slot"))
            revised.update({
                "platform": "Kalshi",
                "ident": "RATE",
                "model": "gpt",
                "snapshot_date": "2026-07-29T23:30",
                "model_probability": 0.61,
                "market_probability": 0.50,
                "resolved": False,
            })
            new_row = Entity(source.key("ForecastSnapshot", "kalshi:JOBS:qwen:slot"))
            new_row.update({
                "platform": "Kalshi",
                "ident": "JOBS",
                "model": "qwen",
                "snapshot_date": "2026-07-29T23:30",
                "model_probability": 0.33,
                "market_probability": 0.30,
                "resolved": False,
            })
            source.put(revised)
            source.put(new_row)
            source._con.close()

            merge_track_record_store.merge_stores(target_path, [source_path])

            merged = DuckDBStore(target_path)
            self.assertAlmostEqual(
                merged.get(merged.key("ForecastSnapshot", "kalshi:RATE:gpt:slot"))[
                    "model_probability"
                ],
                0.61,
            )
            self.assertAlmostEqual(
                merged.get(merged.key("ForecastSnapshot", "kalshi:JOBS:qwen:slot"))[
                    "model_probability"
                ],
                0.33,
            )
            merged._con.close()

    def test_merge_upserts_markets_table(self) -> None:
        # Each model job's local store independently upserts its own copy of
        # `markets` for every market it forecasts. If the merge script forgot
        # this table, the published store's `markets` would stay empty and
        # every hydration read (server.py, aggregate(), the ledger) would
        # silently fall back to NULL for rows written via this workflow.
        with tempfile.TemporaryDirectory() as td:
            target_path = Path(td) / "target.duckdb"
            source_path = Path(td) / "source.duckdb"

            target = DuckDBStore(target_path)
            target._con.close()

            source = DuckDBStore(source_path)
            market = Entity(source.key("Market", "Kalshi:RATE"))
            market.update(platform="Kalshi", ident="RATE", question="Will rates be cut?",
                          market_url="https://kalshi.com/RATE")
            source.put(market)
            source._con.close()

            merge_track_record_store.merge_stores(target_path, [source_path])

            merged = DuckDBStore(target_path)
            row = merged.get(merged.key("Market", "Kalshi:RATE"))
            self.assertEqual(row["question"], "Will rates be cut?")
            self.assertEqual(row["market_url"], "https://kalshi.com/RATE")
            merged._con.close()

    def test_merge_does_not_let_an_incomplete_shard_clobber_markets_fields(self) -> None:
        # `markets` is keyed by (platform, ident) only (no `model`), so two
        # per-model shard jobs can legitimately write independently-derived
        # rows for the SAME key in the same tick. Model A's local copy
        # captured `description`; model B's local copy never did (e.g. its
        # own reference snapshot lacked it). Merging B in after A must not
        # wipe out the description A already contributed.
        with tempfile.TemporaryDirectory() as td:
            target_path = Path(td) / "target.duckdb"
            source_a_path = Path(td) / "source_a.duckdb"
            source_b_path = Path(td) / "source_b.duckdb"

            target = DuckDBStore(target_path)
            target._con.close()

            source_a = DuckDBStore(source_a_path)
            market_a = Entity(source_a.key("Market", "Kalshi:RATE"))
            market_a.update(platform="Kalshi", ident="RATE", question="Will rates be cut?",
                            market_url="https://kalshi.com/RATE",
                            description="Fed decision on the September meeting.")
            source_a.put(market_a)
            source_a._con.close()

            source_b = DuckDBStore(source_b_path)
            market_b = Entity(source_b.key("Market", "Kalshi:RATE"))
            market_b.update(platform="Kalshi", ident="RATE", question="Will rates be cut?",
                            market_url="https://kalshi.com/RATE", description=None)
            source_b.put(market_b)
            source_b._con.close()

            merge_track_record_store.merge_stores(target_path, [source_a_path, source_b_path])

            merged = DuckDBStore(target_path)
            row = merged.get(merged.key("Market", "Kalshi:RATE"))
            self.assertEqual(row["description"], "Fed decision on the September meeting.")
            merged._con.close()


if __name__ == "__main__":
    unittest.main()
