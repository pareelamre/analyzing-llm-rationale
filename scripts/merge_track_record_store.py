#!/usr/bin/env python3
"""Merge one or more forecast DuckDB stores into the checked-out store.

Forecast model jobs run independently from the same base repository snapshot.
Before each job publishes, it fetches current ``main`` and uses this utility to
upsert only the rows from its model-local DuckDB copy into the current shared
store. This avoids direct binary-file rebases between parallel model jobs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyzing_llm_rationale.trackrec_store import (  # noqa: E402
    _CONVERGENCE_COLS,
    _CONVERGENCE_TABLE,
    _LEDGER_EVENT_COLS,
    _LEDGER_EVENT_TABLE,
    _MARKET_COLS,
    _MARKET_TABLE,
    _PRICE_COLS,
    _PRICE_TABLE,
    _SNAPSHOT_COLS,
    _SNAPSHOT_TABLE,
    DuckDBStore,
)

TABLES = (
    (_SNAPSHOT_TABLE, tuple(_SNAPSHOT_COLS)),
    (_PRICE_TABLE, tuple(_PRICE_COLS)),
    (_CONVERGENCE_TABLE, tuple(_CONVERGENCE_COLS)),
    (_LEDGER_EVENT_TABLE, tuple(_LEDGER_EVENT_COLS)),
    # Each model job's local store independently upserts its own `markets`
    # copy for every market it forecasts (see track_record_live._upsert_market)
    # -- merge those in too, or the published store's `markets` table would
    # stay empty and every hydration read (server.py, aggregate(), the ledger)
    # would silently fall back to NULL for rows written via this workflow.
    (_MARKET_TABLE, tuple(_MARKET_COLS)),
)


def _sql_string(value: Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def merge_stores(target: Path, sources: list[Path]) -> dict[str, int]:
    target = target.resolve()
    source_paths = [source.resolve() for source in sources]
    missing = [str(source) for source in source_paths if not source.exists()]
    if missing:
        raise FileNotFoundError("missing source store(s): " + ", ".join(missing))

    target.parent.mkdir(parents=True, exist_ok=True)
    target_store = DuckDBStore(target)
    con = target_store._con
    merged: dict[str, int] = {table: 0 for table, _cols in TABLES}

    for idx, source in enumerate(source_paths):
        source_store = DuckDBStore(source)
        source_store._con.close()
        alias = f"src_{idx}"
        con.execute(f"ATTACH {_sql_string(source)} AS {_quote_ident(alias)}")
        try:
            for table, cols in TABLES:
                col_sql = ", ".join(_quote_ident(col) for col in cols)
                before = con.execute(
                    f"SELECT COUNT(*) FROM {_quote_ident(table)}"
                ).fetchone()[0]
                if table == _MARKET_TABLE:
                    # `markets` is keyed by (platform, ident) only -- unlike
                    # every other table here, its key has no `model` component,
                    # so multiple per-model shard jobs legitimately write
                    # independently-derived rows for the *same* key in the same
                    # tick (see track_record_live._upsert_market). A blind
                    # whole-row replace would let whichever shard merges last
                    # silently clobber fields an earlier shard had already
                    # populated (e.g. wipe out `description` because this
                    # shard's own local copy never captured it). Merge
                    # field-by-field instead, preferring whichever side is
                    # non-null -- the same "value or existing" preference
                    # _upsert_market itself uses.
                    set_sql = ", ".join(
                        f"{_quote_ident(c)} = COALESCE(excluded.{_quote_ident(c)}, "
                        f"{_quote_ident(table)}.{_quote_ident(c)})"
                        for c in cols if c != "key"
                    )
                    con.execute(
                        f"INSERT INTO {_quote_ident(table)} ({col_sql}) "
                        f"SELECT {col_sql} FROM {_quote_ident(alias)}.{_quote_ident(table)} "
                        f"ON CONFLICT (key) DO UPDATE SET {set_sql}"
                    )
                else:
                    con.execute(
                        f"INSERT OR REPLACE INTO {_quote_ident(table)} ({col_sql}) "
                        f"SELECT {col_sql} FROM {_quote_ident(alias)}.{_quote_ident(table)}"
                    )
                after = con.execute(
                    f"SELECT COUNT(*) FROM {_quote_ident(table)}"
                ).fetchone()[0]
                merged[table] += max(0, int(after) - int(before))
        finally:
            con.execute(f"DETACH {_quote_ident(alias)}")

    con.close()
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    merged = merge_stores(args.target, args.sources)
    print("merged forecast stores:", merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
