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


if __name__ == "__main__":
    unittest.main()
