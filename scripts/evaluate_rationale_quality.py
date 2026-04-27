#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as stats
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
DEFAULT_RESULTS = (
    ROOT
    / "results"
    / "Qwen2.5-7b-instruct"
    / "temperature_000"
    / "results_variant0_neutral_baseline.json"
)
DEFAULT_OUTPUT_CSV = ROOT / "analysis" / "qwen25_v0_t0_rationale_eval.csv"
DEFAULT_OUTPUT_JSON = ROOT / "analysis" / "qwen25_v0_t0_rationale_eval_summary.json"
DEFAULT_OUTPUT_MD = ROOT / "analysis" / "qwen25_v0_t0_rationale_eval_summary.md"

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
SENTENCE_RE = re.compile(r"[.!?]+")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
TIME_RE = re.compile(
    r"\b(before|after|by|during|through|within|until|from|between|end|start|deadline|"
    r"year|month|quarter|week|day|january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)
HEDGE_RE = re.compile(
    r"\b(likely|unlikely|possible|possibly|probably|may|might|could|appears|seems|"
    r"suggests|mixed|uncertain|uncertainty|unclear)\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "being",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "such",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "to",
    "was",
    "were",
    "will",
    "with",
    "would",
    "using",
    "used",
    "use",
    "then",
    "also",
    "should",
    "can",
    "could",
    "may",
    "might",
    "must",
    "our",
    "your",
    "his",
    "her",
    "they",
    "them",
    "he",
    "she",
    "we",
    "you",
    "i",
    "do",
    "does",
    "did",
    "not",
    "no",
    "yes",
    "about",
    "against",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "own",
    "same",
    "so",
    "too",
    "very",
}


@dataclass(frozen=True)
class ArticleContext:
    token_set: set[str]
    credibility: float
    frs: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Heuristic rationale evaluation for a forecasting run. "
            "Correctness is computed against the dataset ground truth."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def normalize_answer(value: object) -> str | None:
    if value is None:
        return None
    answer = str(value).strip().lower()
    return answer if answer in {"yes", "no"} else None


def normalize_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(confidence) or math.isinf(confidence) or not (0.0 <= confidence <= 1.0):
        return None
    return confidence


def normalize_token(token: str) -> str:
    token = token.lower()
    if token.isdigit():
        return token
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text):
        token = normalize_token(raw)
        if token.isdigit():
            tokens.append(token)
            continue
        if len(token) < 4:
            continue
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def sentence_count(text: str) -> int:
    pieces = [piece.strip() for piece in SENTENCE_RE.split(text) if piece.strip()]
    return len(pieces)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def coerce_scalar_score(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return clamp(float(value))
    if isinstance(value, dict):
        score = value.get("score")
        if isinstance(score, (int, float)):
            return clamp(float(score))
        return default
    try:
        return clamp(float(value))
    except (TypeError, ValueError):
        return default


def build_article_contexts(row: dict) -> list[ArticleContext]:
    contexts: list[ArticleContext] = []
    for article in row.get("news_articles", []):
        parts: list[str] = []
        for key in ("title", "summary_llm", "summary", "keywords"):
            value = article.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif value:
                parts.append(str(value))
        token_set = set(content_tokens(" ".join(parts)))
        credibility = coerce_scalar_score(article.get("credibility"), default=0.0)
        frs = coerce_scalar_score(article.get("frs"), default=1.0 if article.get("frs") else 0.0)
        contexts.append(ArticleContext(token_set=token_set, credibility=credibility, frs=frs))
    return contexts


def build_context_text(row: dict) -> str:
    parts = [
        str(row.get("question") or ""),
        str(row.get("description") or ""),
        str(row.get("resolution_criteria") or ""),
    ]
    for article in row.get("news_articles", []):
        for key in ("title", "summary_llm", "summary", "keywords"):
            value = article.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif value:
                parts.append(str(value))
    return "\n".join(parts)


def overlap_fraction(needle_tokens: set[str], haystack_tokens: set[str]) -> float:
    if not needle_tokens:
        return 0.0
    return len(needle_tokens & haystack_tokens) / len(needle_tokens)


def score_temporal_specificity(rationale: str, row: dict) -> float:
    temporal_need = bool(
        YEAR_RE.search(str(row.get("question") or ""))
        or YEAR_RE.search(str(row.get("resolution_criteria") or ""))
        or TIME_RE.search(str(row.get("question") or ""))
        or TIME_RE.search(str(row.get("resolution_criteria") or ""))
    )
    if not temporal_need:
        return 1.0
    return 1.0 if YEAR_RE.search(rationale) or TIME_RE.search(rationale) else 0.0


def score_conciseness(word_count: int, sent_count: int) -> float:
    if 20 <= word_count <= 70:
        word_score = 1.0
    elif word_count < 20:
        word_score = clamp(word_count / 20.0)
    else:
        word_score = clamp(1.0 - ((word_count - 70) / 50.0))

    if 2 <= sent_count <= 4:
        sentence_score = 1.0
    elif sent_count == 1 or sent_count == 5:
        sentence_score = 0.5
    else:
        sentence_score = 0.0
    return (word_score + sentence_score) / 2.0


def score_hedging_alignment(confidence: float, rationale: str) -> float:
    has_hedge = bool(HEDGE_RE.search(rationale))
    if confidence >= 0.85:
        return 1.0 if not has_hedge else 0.3
    if confidence < 0.70:
        return 1.0 if has_hedge else 0.2
    return 1.0 if not has_hedge else 0.8


def score_article_support(rationale_token_set: set[str], article_contexts: list[ArticleContext]) -> float:
    if not rationale_token_set or not article_contexts:
        return 0.0
    best = 0.0
    for article in article_contexts:
        overlap = overlap_fraction(rationale_token_set, article.token_set)
        weight = (0.5 + 0.5 * article.credibility) * (0.5 + 0.5 * article.frs)
        best = max(best, overlap * weight)
    return best


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    index = int(round((len(values) - 1) * q))
    return values[index]


def main() -> None:
    args = parse_args()
    dataset_rows = json.loads(args.dataset.read_text())
    result_rows = json.loads(args.results.read_text())

    dataset_by_id = {int(row["id"]): row for row in dataset_rows}
    result_by_id = {int(row["id"]): row for row in result_rows}

    per_example_rows: list[dict[str, object]] = []
    metric_values: defaultdict[str, list[float]] = defaultdict(list)
    metric_values_by_correctness: dict[int, defaultdict[str, list[float]]] = {
        0: defaultdict(list),
        1: defaultdict(list),
    }

    for rid, result in sorted(result_by_id.items()):
        row = dataset_by_id.get(rid)
        if row is None:
            continue

        predicted_answer = normalize_answer(result.get("predicted_answer"))
        confidence = normalize_confidence(result.get("confidence"))
        rationale = str(result.get("rationale") or "").strip()
        target_answer = normalize_answer(row.get("answer"))

        if predicted_answer is None or confidence is None or target_answer is None or not rationale:
            continue

        correct = int(predicted_answer == target_answer)

        rationale_tokens = set(content_tokens(rationale))
        question_tokens = set(content_tokens(str(row.get("question") or "")))
        context_tokens = set(content_tokens(build_context_text(row)))
        article_contexts = build_article_contexts(row)

        words = len(rationale.split())
        sentences = sentence_count(rationale)

        context_grounding = overlap_fraction(rationale_tokens, context_tokens)
        question_focus = overlap_fraction(question_tokens, rationale_tokens)
        article_support = score_article_support(rationale_tokens, article_contexts)
        temporal_specificity = score_temporal_specificity(rationale, row)
        conciseness = score_conciseness(words, sentences)
        hedging_alignment = score_hedging_alignment(confidence, rationale)
        quality_proxy = stats.fmean(
            [
                context_grounding,
                question_focus,
                article_support,
                temporal_specificity,
                conciseness,
                hedging_alignment,
            ]
        )

        example = {
            "id": rid,
            "predicted_answer": predicted_answer,
            "target_answer": target_answer,
            "confidence": confidence,
            "forecast_correct": correct,
            "word_count": words,
            "sentence_count": sentences,
            "context_grounding": context_grounding,
            "question_focus": question_focus,
            "article_support": article_support,
            "temporal_specificity": temporal_specificity,
            "conciseness": conciseness,
            "hedging_alignment": hedging_alignment,
            "quality_proxy": quality_proxy,
            "question": str(row.get("question") or ""),
            "rationale": rationale,
        }
        per_example_rows.append(example)

        for key in (
            "forecast_correct",
            "confidence",
            "word_count",
            "sentence_count",
            "context_grounding",
            "question_focus",
            "article_support",
            "temporal_specificity",
            "conciseness",
            "hedging_alignment",
            "quality_proxy",
        ):
            value = float(example[key])
            metric_values[key].append(value)
            metric_values_by_correctness[correct][key].append(value)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "predicted_answer",
        "target_answer",
        "confidence",
        "forecast_correct",
        "word_count",
        "sentence_count",
        "context_grounding",
        "question_focus",
        "article_support",
        "temporal_specificity",
        "conciseness",
        "hedging_alignment",
        "quality_proxy",
        "question",
        "rationale",
    ]
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_example_rows)

    summary_metrics = {}
    for key, values in metric_values.items():
        summary_metrics[key] = {
            "mean": stats.fmean(values),
            "median": stats.median(values),
            "p10": quantile(values, 0.10),
            "p90": quantile(values, 0.90),
        }

    breakdown = {}
    for correct_flag in (1, 0):
        label = "correct" if correct_flag else "incorrect"
        breakdown[label] = {}
        for key, values in metric_values_by_correctness[correct_flag].items():
            breakdown[label][key] = stats.fmean(values)

    strongest_examples = [
        {
            "id": row["id"],
            "confidence": row["confidence"],
            "quality_proxy": row["quality_proxy"],
            "context_grounding": row["context_grounding"],
            "question_focus": row["question_focus"],
            "article_support": row["article_support"],
            "question": row["question"],
            "rationale": row["rationale"],
        }
        for row in sorted(
            [row for row in per_example_rows if row["forecast_correct"] == 1],
            key=lambda row: (
                float(row["quality_proxy"]),
                float(row["context_grounding"]),
                float(row["article_support"]),
            ),
            reverse=True,
        )[:5]
    ]

    high_confidence_failures = [
        {
            "id": row["id"],
            "confidence": row["confidence"],
            "quality_proxy": row["quality_proxy"],
            "context_grounding": row["context_grounding"],
            "question_focus": row["question_focus"],
            "article_support": row["article_support"],
            "question": row["question"],
            "rationale": row["rationale"],
        }
        for row in sorted(
            [row for row in per_example_rows if row["forecast_correct"] == 0],
            key=lambda row: (
                float(row["confidence"]),
                -float(row["context_grounding"]),
                -float(row["question_focus"]),
            ),
            reverse=True,
        )[:5]
    ]

    grounded_but_wrong = [
        {
            "id": row["id"],
            "confidence": row["confidence"],
            "quality_proxy": row["quality_proxy"],
            "context_grounding": row["context_grounding"],
            "question_focus": row["question_focus"],
            "article_support": row["article_support"],
            "question": row["question"],
            "rationale": row["rationale"],
        }
        for row in sorted(
            [
                row
                for row in per_example_rows
                if row["forecast_correct"] == 0 and float(row["context_grounding"]) >= 0.70
            ],
            key=lambda row: (
                float(row["context_grounding"]),
                float(row["question_focus"]),
                float(row["article_support"]),
            ),
            reverse=True,
        )[:5]
    ]

    summary = {
        "dataset": str(args.dataset),
        "results": str(args.results),
        "n_scored": len(per_example_rows),
        "correctness_definition": "predicted_answer == dataset answer",
        "metrics": summary_metrics,
        "breakdown_by_forecast_correctness": breakdown,
        "strongest_correct_examples": strongest_examples,
        "high_confidence_failures": high_confidence_failures,
        "grounded_but_wrong_examples": grounded_but_wrong,
    }
    args.output_json.write_text(json.dumps(summary, indent=2))

    lines = [
        "# Rationale Evaluation Summary",
        "",
        f"- Results: `{args.results}`",
        f"- Dataset: `{args.dataset}`",
        f"- Scored examples: `{len(per_example_rows)}`",
        "- Correctness uses the dataset ground-truth answer.",
        "",
        "## Overall",
        "",
        "| Metric | Mean | Median | P10 | P90 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in (
        "forecast_correct",
        "confidence",
        "context_grounding",
        "question_focus",
        "article_support",
        "temporal_specificity",
        "conciseness",
        "hedging_alignment",
        "quality_proxy",
    ):
        metric = summary_metrics[key]
        lines.append(
            f"| {key} | {metric['mean']:.3f} | {metric['median']:.3f} | "
            f"{metric['p10']:.3f} | {metric['p90']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Correct vs Incorrect Forecasts",
            "",
            "| Metric | Correct | Incorrect | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key in (
        "confidence",
        "context_grounding",
        "question_focus",
        "article_support",
        "temporal_specificity",
        "conciseness",
        "hedging_alignment",
        "quality_proxy",
    ):
        correct_mean = breakdown["correct"][key]
        incorrect_mean = breakdown["incorrect"][key]
        lines.append(
            f"| {key} | {correct_mean:.3f} | {incorrect_mean:.3f} | {correct_mean - incorrect_mean:+.3f} |"
        )

    def append_examples(title: str, rows: list[dict[str, object]]) -> None:
        lines.extend(["", f"## {title}", ""])
        for row in rows:
            lines.append(
                f"- ID `{row['id']}` | conf `{float(row['confidence']):.2f}` | "
                f"quality `{float(row['quality_proxy']):.3f}` | "
                f"grounding `{float(row['context_grounding']):.3f}` | "
                f"article `{float(row['article_support']):.3f}`"
            )
            lines.append(f"  Question: {row['question']}")
            lines.append(f"  Rationale: {row['rationale']}")

    append_examples("Strongest Correct Examples", strongest_examples)
    append_examples("High-Confidence Failures", high_confidence_failures)
    append_examples("Grounded But Wrong", grounded_but_wrong)

    args.output_md.write_text("\n".join(lines) + "\n")

    print(f"Wrote {len(per_example_rows)} rows to {args.output_csv}")
    print(f"Wrote summary JSON to {args.output_json}")
    print(f"Wrote summary Markdown to {args.output_md}")


if __name__ == "__main__":
    main()
