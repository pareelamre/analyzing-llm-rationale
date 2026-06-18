#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale import track_record_live as trl
from analyzing_llm_rationale.trackrec_store import DuckDBStore

STORE_PATH = ROOT / "data" / "track_record_store.duckdb"
PUBLIC_PATH = ROOT / "static" / "track_record_live.json"

def main():
    if not STORE_PATH.exists():
        print(f"Error: {STORE_PATH} does not exist.")
        return 1
    
    print("Loading DuckDB store...")
    store = DuckDBStore(STORE_PATH)
    
    print("Recomputing aggregate track record...")
    agg = trl.aggregate(
        store, 
        model="gpt-oss-120b", 
        variant="variant0_neutral_baseline", 
        temperature=0.0
    )
    
    print(f"Writing updated aggregate to {PUBLIC_PATH}...")
    PUBLIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_PATH.write_text(json.dumps(agg, indent=2, sort_keys=True) + "\n")
    print("Success!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
