"""
10 medium-level SQL analytics queries on the forecasting dataset.

Usage:
    python scripts/sql_analytics.py [--db forecasting.duckdb] [--ingest]
    python scripts/sql_analytics.py --db analysis/forecasting_analytics.duckdb \
        --ingest --replace --output-dir analysis/sql_analytics
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.db import _tag_to_temperature, get_connection, ingest_dataset

QUERIES = [
    (
        "1. Accuracy per model (all variants combined)",
        """
        SELECT
            p.model,
            COUNT(*) AS n,
            ROUND(AVG(CASE WHEN LOWER(p.predicted_answer) = LOWER(q.answer) THEN 1.0 ELSE 0.0 END), 4) AS accuracy
        FROM predictions p
        JOIN questions q ON p.question_id = q.id
        WHERE LOWER(p.predicted_answer) IN ('yes', 'no')
        GROUP BY p.model
        ORDER BY accuracy DESC
        """,
    ),
    (
        "2. Best-performing variant per model",
        """
        WITH variant_acc AS (
            SELECT
                p.model,
                p.variant,
                ROUND(AVG(CASE WHEN LOWER(p.predicted_answer) = LOWER(q.answer) THEN 1.0 ELSE 0.0 END), 4) AS accuracy,
                COUNT(*) AS n
            FROM predictions p
            JOIN questions q ON p.question_id = q.id
            WHERE LOWER(p.predicted_answer) IN ('yes', 'no')
            GROUP BY p.model, p.variant
        )
        SELECT model, variant, accuracy, n
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY model ORDER BY accuracy DESC) AS rn
            FROM variant_acc
        )
        WHERE rn = 1
        ORDER BY accuracy DESC
        """,
    ),
    (
        "3. Confidence calibration (10 bins): stated confidence vs actual accuracy",
        """
        SELECT
            FLOOR(p.confidence * 10) / 10.0 AS conf_bin,
            COUNT(*) AS n,
            ROUND(AVG(p.confidence), 3) AS avg_confidence,
            ROUND(AVG(CASE WHEN LOWER(p.predicted_answer) = LOWER(q.answer) THEN 1.0 ELSE 0.0 END), 4) AS accuracy,
            ROUND(
                AVG(p.confidence) -
                AVG(CASE WHEN LOWER(p.predicted_answer) = LOWER(q.answer) THEN 1.0 ELSE 0.0 END),
                4
            ) AS calibration_gap
        FROM predictions p
        JOIN questions q ON p.question_id = q.id
        WHERE LOWER(p.predicted_answer) IN ('yes', 'no')
          AND p.confidence IS NOT NULL
        GROUP BY conf_bin
        ORDER BY conf_bin
        """,
    ),
    (
        "4. Brier score per model (lower is better)",
        """
        SELECT
            p.model,
            COUNT(*) AS n,
            ROUND(AVG(
                POWER(p.confidence - CASE WHEN LOWER(q.answer) = 'yes' THEN 1.0 ELSE 0.0 END, 2)
            ), 5) AS brier_score
        FROM predictions p
        JOIN questions q ON p.question_id = q.id
        WHERE LOWER(p.predicted_answer) IN ('yes', 'no')
          AND p.confidence IS NOT NULL
        GROUP BY p.model
        ORDER BY brier_score ASC
        """,
    ),
    (
        "5. Consensus questions: all models predict the same answer",
        """
        SELECT
            p.question_id,
            q.question,
            q.answer AS ground_truth,
            COUNT(DISTINCT p.model) AS model_count,
            MAX(p.predicted_answer) AS consensus_answer,
            ROUND(AVG(p.confidence), 3) AS avg_confidence
        FROM predictions p
        JOIN questions q ON p.question_id = q.id
        WHERE LOWER(p.predicted_answer) IN ('yes', 'no')
          AND p.variant = 'variant0_neutral_baseline'
        GROUP BY p.question_id, q.question, q.answer
        HAVING COUNT(DISTINCT p.predicted_answer) = 1
           AND COUNT(DISTINCT p.model) >= 2
        ORDER BY avg_confidence DESC
        LIMIT 10
        """,
    ),
    (
        "6. Disagreement questions: highest confidence variance across models",
        """
        SELECT
            p.question_id,
            q.question,
            q.answer AS ground_truth,
            COUNT(DISTINCT p.model) AS model_count,
            ROUND(STDDEV(p.confidence), 4) AS confidence_stddev,
            COUNT(DISTINCT p.predicted_answer) AS answer_variety
        FROM predictions p
        JOIN questions q ON p.question_id = q.id
        WHERE LOWER(p.predicted_answer) IN ('yes', 'no')
          AND p.variant = 'variant0_neutral_baseline'
          AND p.confidence IS NOT NULL
        GROUP BY p.question_id, q.question, q.answer
        HAVING COUNT(DISTINCT p.model) >= 2
        ORDER BY confidence_stddev DESC
        LIMIT 10
        """,
    ),
    (
        "7. Variant lift over baseline (variant0): accuracy delta per model",
        """
        WITH base AS (
            SELECT p.model,
                ROUND(AVG(CASE WHEN LOWER(p.predicted_answer) = LOWER(q.answer) THEN 1.0 ELSE 0.0 END), 4) AS base_acc
            FROM predictions p
            JOIN questions q ON p.question_id = q.id
            WHERE p.variant = 'variant0_neutral_baseline'
              AND LOWER(p.predicted_answer) IN ('yes', 'no')
            GROUP BY p.model
        ),
        variants AS (
            SELECT p.model, p.variant,
                ROUND(AVG(CASE WHEN LOWER(p.predicted_answer) = LOWER(q.answer) THEN 1.0 ELSE 0.0 END), 4) AS var_acc
            FROM predictions p
            JOIN questions q ON p.question_id = q.id
            WHERE p.variant != 'variant0_neutral_baseline'
              AND LOWER(p.predicted_answer) IN ('yes', 'no')
            GROUP BY p.model, p.variant
        )
        SELECT v.model, v.variant,
            b.base_acc,
            v.var_acc,
            ROUND(v.var_acc - b.base_acc, 4) AS delta
        FROM variants v
        JOIN base b ON v.model = b.model
        ORDER BY delta DESC
        LIMIT 15
        """,
    ),
    (
        "8. Temperature sensitivity: accuracy by temperature per model",
        """
        SELECT
            p.model,
            p.temperature,
            COUNT(*) AS n,
            ROUND(AVG(CASE WHEN LOWER(p.predicted_answer) = LOWER(q.answer) THEN 1.0 ELSE 0.0 END), 4) AS accuracy
        FROM predictions p
        JOIN questions q ON p.question_id = q.id
        WHERE LOWER(p.predicted_answer) IN ('yes', 'no')
          AND p.variant = 'variant0_neutral_baseline'
        GROUP BY p.model, p.temperature
        ORDER BY p.model, p.temperature
        """,
    ),
    (
        "9. Overconfident errors: wrong predictions with confidence > 0.8",
        """
        SELECT
            p.model,
            p.variant,
            p.question_id,
            p.predicted_answer,
            q.answer AS ground_truth,
            ROUND(p.confidence, 3) AS confidence
        FROM predictions p
        JOIN questions q ON p.question_id = q.id
        WHERE LOWER(p.predicted_answer) IN ('yes', 'no')
          AND LOWER(p.predicted_answer) != LOWER(q.answer)
          AND p.confidence > 0.8
        ORDER BY p.confidence DESC
        LIMIT 15
        """,
    ),
    (
        "10. Category difficulty: accuracy per question category (hardest first)",
        """
        SELECT
            cat.category,
            COUNT(*) AS n,
            ROUND(AVG(CASE WHEN LOWER(p.predicted_answer) = LOWER(q.answer) THEN 1.0 ELSE 0.0 END), 4) AS accuracy
        FROM predictions p
        JOIN questions q ON p.question_id = q.id,
        LATERAL (
            SELECT UNNEST(json_extract_string(q.categories, '$[*]')) AS category
        ) cat
        WHERE LOWER(p.predicted_answer) IN ('yes', 'no')
          AND p.variant = 'variant0_neutral_baseline'
        GROUP BY cat.category
        HAVING COUNT(*) >= 10
        ORDER BY accuracy ASC
        LIMIT 15
        """,
    ),
]


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug[:80]


def _write_report(output_dir: Path, results: list[tuple[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# Forecasting SQL Analytics",
        "",
        "DuckDB analytics over the Metaculus-style forecasting questions, model predictions, and news logs.",
        "",
    ]

    for title, df in results:
        slug = _slugify(title)
        csv_path = output_dir / f"{slug}.csv"
        df.to_csv(csv_path, index=False)
        report_lines.extend([
            f"## {title}",
            "",
            f"CSV: `{csv_path.name}`",
            "",
        ])
        if df.empty:
            report_lines.extend(["No rows returned.", ""])
        else:
            report_lines.extend(["```text", df.to_string(index=False), "```", ""])

    (output_dir / "README.md").write_text("\n".join(report_lines), encoding="utf-8")


def _ingest_results_for_analytics(conn, include_rationales: bool = False) -> int:
    import pandas as pd

    results_root = Path(__file__).resolve().parents[1] / "results"
    total = 0

    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for temp_dir in sorted(model_dir.iterdir()):
            if not temp_dir.is_dir():
                continue
            temperature = _tag_to_temperature(temp_dir.name)
            if temperature is None:
                continue
            for result_file in sorted(temp_dir.glob("results_variant*.json")):
                variant = result_file.stem.replace("results_", "")
                try:
                    records = json.loads(result_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(records, list):
                    continue

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
                        r.get("rationale") if include_rationales else None,
                    )
                    for r in records
                    if isinstance(r, dict)
                ]
                df = pd.DataFrame(
                    rows,
                    columns=[
                        "run_id",
                        "model",
                        "temperature",
                        "variant",
                        "question_id",
                        "predicted_answer",
                        "confidence",
                        "rationale",
                    ],
                )
                conn.register("prediction_batch", df)
                conn.execute(
                    """
                    INSERT INTO predictions
                      (run_id, model, temperature, variant, question_id,
                       predicted_answer, confidence, rationale)
                    SELECT
                        run_id,
                        model,
                        temperature,
                        variant,
                        question_id,
                        predicted_answer,
                        confidence,
                        rationale
                    FROM prediction_batch
                    """
                )
                conn.unregister("prediction_batch")
                total += len(rows)
    return total


def main():
    parser = argparse.ArgumentParser(description="Run 10 SQL analytics on forecasting data.")
    parser.add_argument("--db", type=Path, default=None, help="Path to DuckDB file.")
    parser.add_argument(
        "--ingest", action="store_true",
        help="Ingest results + dataset before querying."
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Clear analytics tables before ingesting, producing a reproducible snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for a markdown report plus one CSV per query.",
    )
    parser.add_argument(
        "--include-rationales",
        action="store_true",
        help="Also ingest full rationale text. The 10 SQL queries do not require it.",
    )
    args = parser.parse_args()

    conn = get_connection(args.db)

    if args.ingest:
        if args.replace:
            conn.execute("DELETE FROM pipeline_runs")
            conn.execute("DELETE FROM news_articles")
            conn.execute("DELETE FROM predictions")
            conn.execute("DELETE FROM questions")
        print("Ingesting dataset...", flush=True)
        n_q = ingest_dataset(conn=conn)
        print(f"  {n_q} questions loaded.", flush=True)
        print("Ingesting results...", flush=True)
        n_p = _ingest_results_for_analytics(
            conn,
            include_rationales=args.include_rationales,
        )
        print(f"  {n_p} prediction rows loaded.\n", flush=True)

    results = []
    for title, sql in QUERIES:
        print(f"\n{'='*60}")
        print(title)
        print('='*60)
        try:
            result = conn.execute(sql).fetchdf()
            results.append((title, result))
            if result.empty:
                print("  (no rows — run with --ingest first)")
            else:
                print(result.to_string(index=False))
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if args.output_dir:
        _write_report(args.output_dir, results)
        print(f"\nWrote SQL report to {args.output_dir / 'README.md'}")

    conn.close()


if __name__ == "__main__":
    main()
