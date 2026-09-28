from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

import duckdb


def export_dataset_to_parquet(
    table_name: str,
    records: List[Dict[str, Any]],
    compression: str = "ZSTD",
) -> bytes:
    """Convert a list of dictionary records to compressed Parquet bytes using in-memory DuckDB.

    Parameters
    ----------
    table_name: Identifier for the DuckDB table (e.g. 'edge_board', 'forecast_snapshots')
    records: List of dictionaries representing table rows.
    compression: Compression algorithm ('ZSTD', 'SNAPPY', 'GZIP', 'UNCOMPRESSED')

    Returns
    -------
    Parquet file content as bytes.
    """
    if not records:
        # Create empty table with placeholder column
        conn = duckdb.connect(":memory:")
        temp_file = tempfile.mktemp(suffix=".parquet")
        try:
            conn.execute(f"CREATE TABLE {table_name} (id VARCHAR, placeholder VARCHAR)")
            conn.execute(
                f"COPY {table_name} TO '{temp_file}' (FORMAT PARQUET, COMPRESSION {compression})"
            )
            with open(temp_file, "rb") as f:
                return f.read()
        finally:
            conn.close()
            if os.path.exists(temp_file):
                os.remove(temp_file)

    conn = duckdb.connect(":memory:")
    temp_json = tempfile.mktemp(suffix=".json")
    temp_parquet = tempfile.mktemp(suffix=".parquet")
    try:
        with open(temp_json, "w", encoding="utf-8") as f:
            json.dump(records, f)

        conn.execute(
            f"CREATE TABLE {table_name} AS SELECT * FROM read_json_auto('{temp_json}')"
        )
        conn.execute(
            f"COPY {table_name} TO '{temp_parquet}' (FORMAT PARQUET, COMPRESSION {compression})"
        )
        with open(temp_parquet, "rb") as f:
            return f.read()
    finally:
        conn.close()
        if os.path.exists(temp_json):
            os.remove(temp_json)
        if os.path.exists(temp_parquet):
            os.remove(temp_parquet)


def build_edge_board_parquet(mtm_data: Dict[str, Any]) -> bytes:
    """Export edge board opportunities from mark_to_market_live into Parquet format."""
    rows = mtm_data.get("edge_board") or []
    if isinstance(rows, dict):
        rows = rows.get("markets") or rows.get("rows") or []
    return export_dataset_to_parquet("edge_board", rows)


def build_models_comparison_parquet(track_record_data: Dict[str, Any]) -> bytes:
    """Export models comparison metrics from track_record_live into Parquet format."""
    rows = track_record_data.get("models_comparison") or track_record_data.get("by_domain") or []
    return export_dataset_to_parquet("models_comparison", rows)
