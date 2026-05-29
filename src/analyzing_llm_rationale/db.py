from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import List, Optional

DB_PATH = Path(__file__).resolve().parents[2] / "forecasting.duckdb"

DDL = """
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY,
    question TEXT,
    answer VARCHAR,
    categories JSON,
    resolve_time VARCHAR
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id VARCHAR,
    model VARCHAR,
    temperature FLOAT,
    variant VARCHAR,
    question_id INTEGER,
    predicted_answer VARCHAR,
    confidence FLOAT,
    rationale TEXT,
    ingested_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS news_articles (
    question_id INTEGER,
    run_id VARCHAR,
    title TEXT,
    url TEXT,
    publish_date VARCHAR,
    relevance_score FLOAT,
    summary TEXT,
    fetched_at TIMESTAMP DEFAULT current_timestamp
);
"""


def get_connection(path: Optional[Path] = None):
    import duckdb
    conn = duckdb.connect(str(path or DB_PATH))
    conn.execute(DDL)
    return conn


def _tag_to_temperature(tag: str) -> float:
    """Convert 'temperature_025' → 0.25, 'temperature_00' → 0.0."""
    digits = re.sub(r"^temperature_", "", tag)
    if not digits:
        return 0.0
    if len(digits) == 2:
        return float(digits) / 10.0
    if len(digits) == 3:
        return float(digits) / 100.0
    return float(digits) / (10 ** (len(digits) - 1))


def ingest_results_json(
    path: Path,
    model: str,
    temperature: float,
    variant: str,
    conn=None,
) -> int:
    """Load a results_variant*.json file into the predictions table."""
    close = conn is None
    if conn is None:
        conn = get_connection()

    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        return 0

    run_id = str(uuid.uuid4())
    rows = [
        (
            run_id,
            model,
            temperature,
            variant,
            int(r.get("id", 0)),
            r.get("predicted_answer"),
            r.get("confidence"),
            r.get("rationale"),
        )
        for r in records
        if isinstance(r, dict)
    ]
    conn.executemany(
        """
        INSERT OR IGNORE INTO predictions
          (run_id, model, temperature, variant, question_id,
           predicted_answer, confidence, rationale)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    if close:
        conn.close()
    return len(rows)


def ingest_all_results(results_root: Optional[Path] = None, conn=None) -> int:
    """Walk results/<model>/<temp>/ and ingest every results_variant*.json."""
    if results_root is None:
        results_root = Path(__file__).resolve().parents[2] / "results"

    close = conn is None
    if conn is None:
        conn = get_connection()

    total = 0
    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for temp_dir in sorted(model_dir.iterdir()):
            if not temp_dir.is_dir():
                continue
            temperature = _tag_to_temperature(temp_dir.name)
            for result_file in sorted(temp_dir.glob("results_variant*.json")):
                variant = result_file.stem.replace("results_", "")
                try:
                    total += ingest_results_json(
                        result_file, model, temperature, variant, conn=conn
                    )
                except Exception:
                    pass

    if close:
        conn.close()
    return total


def ingest_dataset(dataset_path: Optional[Path] = None, conn=None) -> int:
    """Load the Metaculus dataset into the questions table."""
    if dataset_path is None:
        dataset_path = next(
            Path(__file__).resolve().parents[2].glob(
                "forecasting_qa_news_metaculus_*.json"
            )
        )

    close = conn is None
    if conn is None:
        conn = get_connection()

    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = [
        (
            int(r.get("id", 0)),
            r.get("question"),
            r.get("answer"),
            json.dumps(r.get("categories") or []),
            r.get("resolve_time"),
        )
        for r in records
        if isinstance(r, dict)
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO questions (id, question, answer, categories, resolve_time)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    if close:
        conn.close()
    return len(rows)


def store_news_articles(
    question_id: int,
    articles: List[dict],
    run_id: Optional[str] = None,
    conn=None,
) -> int:
    close = conn is None
    if conn is None:
        conn = get_connection()

    run_id = run_id or str(uuid.uuid4())
    rows = [
        (
            question_id,
            run_id,
            a.get("title"),
            a.get("url"),
            a.get("publish_date"),
            a.get("relevance_score"),
            a.get("summary"),
        )
        for a in articles
    ]
    conn.executemany(
        """
        INSERT INTO news_articles
          (question_id, run_id, title, url, publish_date, relevance_score, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    if close:
        conn.close()
    return len(rows)
