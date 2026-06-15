#!/usr/bin/env python3
"""One-time migration: track_record_store.json → track_record_store.duckdb.

Usage:
  python scripts/migrate_store_to_duckdb.py

Reads data/track_record_store.json, writes data/track_record_store.duckdb.
The JSON file is left in place; delete it manually once the DuckDB file is
committed and the tick has switched over.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale.trackrec_store import DuckDBStore, FileStore  # noqa: E402

SRC = ROOT / "data" / "track_record_store.json"
DST = ROOT / "data" / "track_record_store.duckdb"

if not SRC.exists():
    print(f"Source not found: {SRC}", file=sys.stderr)
    sys.exit(1)

if DST.exists():
    print(f"Destination already exists: {DST} — delete it first to re-migrate.")
    sys.exit(1)

print(f"Loading {SRC} …")
src = FileStore(SRC, autosave=False)

counts: dict[str, int] = {}
for (kind, _id), _e in src.items():
    counts[kind] = counts.get(kind, 0) + 1
for kind, n in counts.items():
    print(f"  {kind}: {n} rows")

print(f"Writing {DST} …")
dst = DuckDBStore(DST)

for (_kind, _id), entity in src.items():
    dst.put(entity)

dst.save()
dst.close()

# Verify
_DUCKDB_KINDS = {"ForecastSnapshot", "MarketPricePoint"}
verify = DuckDBStore(DST)
for kind, n in counts.items():
    if kind not in _DUCKDB_KINDS:
        print(f"  {kind}: skipped (served via static JSON, not stored in DuckDB)")
        continue
    n_db = verify.count(kind)
    status = "✓" if n_db == n else f"✗ expected {n}"
    print(f"  {kind}: {n_db} rows {status}")
verify.close()

print("Done.")
