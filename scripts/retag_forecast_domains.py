#!/usr/bin/env python3
"""Retag stored forecast_snapshot domain/entity fields with the current tagger."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale.entity_tagger import tag_question  # noqa: E402

import duckdb  # noqa: E402


def main() -> int:
    db_path = ROOT / "data" / "track_record_store.duckdb"
    if not db_path.exists():
        print(f"Error: {db_path} does not exist.")
        return 1

    con = duckdb.connect(str(db_path))
    rows = con.execute(
        """
        SELECT key, question, category
        FROM forecast_snapshot
        """
    ).fetchall()

    updated = 0
    for key, question, category in rows:
        tags = tag_question(question or "", category)
        con.execute(
            """
            UPDATE forecast_snapshot
            SET domain = ?, entities = ?
            WHERE key = ?
            """,
            [tags["domain"], json.dumps(tags["entities"]), key],
        )
        updated += 1

    con.close()
    print(f"Retagged {updated} forecast_snapshot rows in {db_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
