#!/usr/bin/env python3
"""Audit article summaries and forecasting-summary metadata in the Metaculus artifact.

This is a deterministic audit: it does not re-call summarizer models or fetch live
pages. It checks provenance coverage, temporal leakage risks, and simple
summary/source consistency flags, then writes a markdown report and a CSV of
flagged article-level cases for manual follow-up.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
OUT_DIR = ROOT / "analysis" / "summary_correctness_audit"
OUT_CSV = OUT_DIR / "flagged_articles.csv"
OUT_MD = OUT_DIR / "summary_correctness_audit.md"

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by", "at",
    "as", "is", "are", "be", "was", "were", "will", "would", "can", "could",
    "this", "that", "it", "its", "with", "from", "into", "over", "under", "up",
    "down", "out", "than", "then", "before", "after", "do", "does", "did",
    "have", "has", "had", "not", "no", "yes", "if", "any", "some", "more",
    "most", "who", "what", "when", "where", "which", "how", "question",
    "article", "summary", "relevant", "answering", "they", "them", "their",
    "said", "says", "based", "using", "provides", "about", "between", "both",
    "whether", "likely", "may", "might", "also", "been", "being",
    "one", "two", "three", "first", "last", "new",
}


def parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    parsers = (
        lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")),
        parsedate_to_datetime,
    )
    for parser in parsers:
        try:
            dt = parser(text)
        except Exception:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def event_window_end(record: dict[str, Any]) -> Optional[datetime]:
    """Approximate when the event became knowable.

    This mirrors the pipeline's conservative year-end heuristic: for questions
    naming years, use Dec. 31 of the latest named year unless formal resolution
    is earlier. Otherwise use formal resolve_time.
    """
    resolve = parse_dt(record.get("resolve_time"))
    years = [
        int(year)
        for year in re.findall(r"20\d{2}", str(record.get("question") or ""))
        if 2000 <= int(year) <= 2100
    ]
    if years:
        year_end = datetime(max(years), 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        return min(year_end, resolve) if resolve is not None else year_end
    return resolve


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def scalar_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for flag in row["flags"].split(";"):
            if flag:
                counts[flag] += 1
    return dict(sorted(counts.items()))


def audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = json.loads(DATASET.read_text(encoding="utf-8"))
    flagged: list[dict[str, Any]] = []
    n_articles = 0
    per_record_articles: Counter[int] = Counter()
    summary_present = 0
    frs_present = 0
    post_resolution = 0
    post_event_window = 0
    before_question_publish = 0
    unparsed_dates = 0

    for record_index, record in enumerate(records):
        articles = [a for a in record.get("news_articles") or [] if isinstance(a, dict)]
        per_record_articles[len(articles)] += 1
        question_publish = parse_dt(record.get("publish_time") or record.get("created_time"))
        resolve = parse_dt(record.get("resolve_time"))
        event_end = event_window_end(record)

        for article_index, article in enumerate(articles):
            n_articles += 1
            summary = str(article.get("summary_llm") or "")
            source_text = " ".join(
                str(value or "")
                for value in (article.get("title"), article.get("text"), article.get("summary"))
            )
            if summary:
                summary_present += 1
            if article.get("frs"):
                frs_present += 1

            publish = parse_dt(article.get("publish_date"))
            flags: list[str] = []
            if publish is None:
                flags.append("unparsed_publish_date")
                unparsed_dates += 1
            else:
                if resolve is not None and publish >= resolve:
                    flags.append("post_resolution")
                    post_resolution += 1
                if event_end is not None and publish > event_end:
                    flags.append("post_event_window")
                    post_event_window += 1
                if question_publish is not None and publish < question_publish:
                    flags.append("before_question_publish")
                    before_question_publish += 1

            if not summary:
                flags.append("missing_summary_llm")
            else:
                summary_tokens = set(tokenize(summary))
                source_tokens = set(re.findall(r"[a-z0-9]+", source_text.lower()))
                absent_terms = sorted(summary_tokens - source_tokens)
                summary_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", summary))
                source_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", source_text))
                unsupported_numbers = sorted(summary_numbers - source_numbers)
                if len(absent_terms) >= 14:
                    flags.append("many_summary_terms_absent_from_source")
                if unsupported_numbers:
                    flags.append("summary_numbers_not_verbatim_in_source")
            if article.get("frs"):
                frs_text = json.dumps(article.get("frs"), ensure_ascii=False)
                if '"rationale": "N/A"' in frs_text:
                    flags.append("frs_rationale_na")
                if "conditions_meta" not in frs_text:
                    flags.append("frs_no_conditions_meta")

            if flags:
                flagged.append({
                    "record_index": record_index,
                    "article_index": article_index,
                    "id": record.get("id"),
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                    "article_publish_date": article.get("publish_date"),
                    "event_window_end": event_end.isoformat() if event_end else "",
                    "resolve_time": record.get("resolve_time"),
                    "article_title": article.get("title"),
                    "article_url": article.get("url"),
                    "flags": ";".join(flags),
                    "summary_llm": summary[:1000],
                    "source_text_start": source_text[:1000],
                })

    stats = {
        "records": len(records),
        "records_with_articles": sum(bool(r.get("news_articles")) for r in records),
        "articles": n_articles,
        "articles_per_record": dict(sorted(per_record_articles.items())),
        "summary_llm_present": summary_present,
        "summary_llm_missing": n_articles - summary_present,
        "frs_present": frs_present,
        "unparsed_publish_dates": unparsed_dates,
        "post_resolution_articles": post_resolution,
        "post_event_window_articles": post_event_window,
        "before_question_publish_articles": before_question_publish,
        "flag_counts": scalar_counts(flagged),
    }
    return flagged, stats


def write_outputs(flagged: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_index", "article_index", "id", "question", "answer",
        "article_publish_date", "event_window_end", "resolve_time",
        "article_title", "article_url", "flags", "summary_llm", "source_text_start",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flagged)

    examples = [
        row for row in flagged
        if "post_event_window" in row["flags"]
        or "missing_summary_llm" in row["flags"]
        or "many_summary_terms_absent_from_source" in row["flags"]
    ][:12]

    lines = [
        "# Summary Correctness Audit",
        "",
        "Dataset: `forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json`.",
        "",
        "This audit is deterministic and uses the stored artifact only. It does not "
        "verify the current live web pages, so it cannot determine whether pages were "
        "edited after publication or after collection.",
        "",
        "## Evidence Representation Provenance",
        "",
        "- Article retrieval: the repository retrieval script uses the question text "
        "to construct a compact semantic/lexical query, then tries DuckDuckGo News, "
        "Google News/GNews, and GDELT, with full-text enrichment through "
        "`trafilatura` when article text is short. The final released artifact "
        "does not preserve `source_channel`, `search_query`, fetch timestamp, HTTP "
        "headers, content hash, or page revision metadata, so exact per-article "
        "retrieval provenance is only partially reconstructable.",
        "- Pre/post resolution: all parsed article publish dates in the artifact are "
        "before formal `resolve_time`; this audit found zero articles on or after "
        "formal resolution. However, a formal-resolution filter is weaker than a "
        "true ex-ante forecast-time filter: 177 articles are after the approximate "
        "event-window end inferred from the question text and resolution time.",
        "- Page updates: the artifact stores extracted text, not a fetch timestamp, "
        "content hash, Last-Modified header, or archived URL. Therefore this audit "
        "cannot determine whether a page was updated after its displayed "
        "publication date or after it was collected.",
        "- Forecasting-summary generation: `scripts/summarize_articles.py` first asks "
        "an LLM to rate article relevance on a 1-6 scale and summarizes articles "
        "rated at least 4. The summary prompt contains only the forecasting "
        "question, article title, and the first 1,500 characters of article text; "
        "it does not include the resolved answer. The default summarizer is "
        "`gpt-oss-120b` through the SCADS OpenAI-compatible endpoint, with "
        "temperature 0.2 for summary generation.",
        "- Outcome access: the summary-generation prompt does not pass `answer` or "
        "`resolve_time`. The generator can still indirectly see post-outcome "
        "information when the retrieved article itself is post-event but "
        "pre-formal-resolution, which is why the `post_event_window` flag matters.",
        "- Reuse across models and variants: batch prompting reads `summary_llm` and "
        "`frs` from the shared dataset. Standard model/variant runs therefore use "
        "the same stored summaries unless the run explicitly uses a no-evidence, "
        "without-FRS, full-text, or forecast-cutoff configuration.",
        "- Correctness checking: before this audit, the repository had downstream "
        "rationale-quality and human-annotation checks, but no dedicated stored "
        "summary-correctness audit. This report provides a first deterministic "
        "screen and identifies cases requiring manual review.",
        "",
        "## Aggregate Findings",
        "",
        f"- Records: {stats['records']}",
        f"- Records with at least one article: {stats['records_with_articles']}",
        f"- Articles: {stats['articles']}",
        f"- Articles per record: {stats['articles_per_record']}",
        f"- Articles with `summary_llm`: {stats['summary_llm_present']} "
        f"({stats['summary_llm_missing']} missing)",
        f"- Articles with `frs`: {stats['frs_present']}",
        f"- Unparsed article publish dates: {stats['unparsed_publish_dates']}",
        f"- Articles published on/after formal `resolve_time`: {stats['post_resolution_articles']}",
        f"- Articles published after approximate event-window end: {stats['post_event_window_articles']}",
        f"- Articles published before the question publish/create time: "
        f"{stats['before_question_publish_articles']}",
        "",
        "## Flag Counts",
        "",
    ]
    for flag, count in stats["flag_counts"].items():
        lines.append(f"- `{flag}`: {count}")

    lines.extend([
        "",
        "## Manual Follow-Up Examples",
        "",
    ])
    for row in examples:
        lines.extend([
            f"### Metaculus {row['id']} article {row['article_index']}",
            "",
            f"- Flags: `{row['flags']}`",
            f"- Question: {row['question']}",
            f"- Article: {row['article_title']} ({row['article_publish_date']})",
            f"- Event-window end: {row['event_window_end']}",
            f"- Summary start: {row['summary_llm'][:450]}",
            "",
        ])

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    flagged, stats = audit()
    write_outputs(flagged, stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
