"""
10 medium-level SQL analytics queries on the forecasting dataset.

Usage:
    python scripts/sql_analytics.py [--db forecasting.duckdb] [--ingest]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.db import get_connection, ingest_all_results, ingest_dataset

QUERIES = [
    (
        "1. Accuracy per model (all variants combined)",
        """
        SELECT
            p.model,
            COUNT(*) AS n,
            ROUND(AVG(CASE WHEN p.predicted_answer = q.answer THEN 1.0 ELSE 0.0 END), 4) AS accuracy
        FROM predictions p
        JOIN questions q ON p.question_id = q.id
        WHERE p.predicted_answer IN ('Yes', 'No')
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
                ROUND(AVG(CASE WHEN p.predicted_answer = q.answer THEN 1.0 ELSE 0.0 END), 4) AS accuracy,
                COUNT(*) AS n
            FROM predictions p
            JOIN questions q ON p.question_id = q.id
            WHERE p.predicted_answer IN ('Yes', 'No')
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
            ROUND(AVG(CASE WHEN p.predicted_answer = q.answer THEN 1.0 ELSE 0.0 END), 4) AS accuracy,
            ROUND(
                AVG(p.confidence) -
                AVG(CASE WHEN p.predicted_answer = q.answer THEN 1.0 ELSE 0.0 END),
                4
            ) AS calibration_gap
        FROM predictions p
        JOIN questions q ON p.question_id = q.id
        WHERE p.predicted_answer IN ('Yes', 'No')
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
                POWER(p.confidence - CASE WHEN q.answer = 'Yes' THEN 1.0 ELSE 0.0 END, 2)
            ), 5) AS brier_score
        FROM predictions p
        JOIN questions q ON p.question_id = q.id
        WHERE p.predicted_answer IN ('Yes', 'No')
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
        WHERE p.predicted_answer IN ('Yes', 'No')
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
        WHERE p.predicted_answer IN ('Yes', 'No')
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
                ROUND(AVG(CASE WHEN p.predicted_answer = q.answer THEN 1.0 ELSE 0.0 END), 4) AS base_acc
            FROM predictions p
            JOIN questions q ON p.question_id = q.id
            WHERE p.variant = 'variant0_neutral_baseline'
              AND p.predicted_answer IN ('Yes', 'No')
            GROUP BY p.model
        ),
        variants AS (
            SELECT p.model, p.variant,
                ROUND(AVG(CASE WHEN p.predicted_answer = q.answer THEN 1.0 ELSE 0.0 END), 4) AS var_acc
            FROM predictions p
            JOIN questions q ON p.question_id = q.id
            WHERE p.variant != 'variant0_neutral_baseline'
              AND p.predicted_answer IN ('Yes', 'No')
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
            ROUND(AVG(CASE WHEN p.predicted_answer = q.answer THEN 1.0 ELSE 0.0 END), 4) AS accuracy
        FROM predictions p
        JOIN questions q ON p.question_id = q.id
        WHERE p.predicted_answer IN ('Yes', 'No')
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
        WHERE p.predicted_answer IN ('Yes', 'No')
          AND p.predicted_answer != q.answer
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
            ROUND(AVG(CASE WHEN p.predicted_answer = q.answer THEN 1.0 ELSE 0.0 END), 4) AS accuracy
        FROM predictions p
        JOIN questions q ON p.question_id = q.id,
        LATERAL (
            SELECT UNNEST(json_extract_string(q.categories, '$[*]')) AS category
        ) cat
        WHERE p.predicted_answer IN ('Yes', 'No')
          AND p.variant = 'variant0_neutral_baseline'
        GROUP BY cat.category
        HAVING COUNT(*) >= 10
        ORDER BY accuracy ASC
        LIMIT 15
        """,
    ),
]


def main():
    parser = argparse.ArgumentParser(description="Run 10 SQL analytics on forecasting data.")
    parser.add_argument("--db", type=Path, default=None, help="Path to DuckDB file.")
    parser.add_argument(
        "--ingest", action="store_true",
        help="Ingest results + dataset before querying."
    )
    args = parser.parse_args()

    conn = get_connection(args.db)

    if args.ingest:
        print("Ingesting dataset...")
        n_q = ingest_dataset(conn=conn)
        print(f"  {n_q} questions loaded.")
        print("Ingesting results...")
        n_p = ingest_all_results(conn=conn)
        print(f"  {n_p} prediction rows loaded.\n")

    for title, sql in QUERIES:
        print(f"\n{'='*60}")
        print(title)
        print('='*60)
        try:
            result = conn.execute(sql).fetchdf()
            if result.empty:
                print("  (no rows — run with --ingest first)")
            else:
                print(result.to_string(index=False))
        except Exception as exc:
            print(f"  ERROR: {exc}")

    conn.close()


if __name__ == "__main__":
    main()
