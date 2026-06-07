"""Build the Foresea knowledge-graph dataset from resolved Metaculus questions.

Reads the raw Metaculus FRS-format JSON, applies entity tagging, filters
articles to only those published STRICTLY before the question's resolve_time
(no look-ahead), and writes a clean dataset to data/kg_dataset.json.

Usage:
    python scripts/build_kg_dataset.py [--input PATH] [--output PATH] [--min-articles N]

The output is a list of records:
    {
        "id":           str,            # "metaculus:{question_id}"
        "question":     str,
        "outcome":      int,            # 1=yes 0=no
        "resolve_time": str,            # ISO-8601
        "created_time": str,
        "domain":       str,            # entity_tagger domain label
        "entities":     list[str],      # named entities
        "categories":   list[str],      # Metaculus categories
        "articles":     list[{          # pre-resolution articles only
            "title", "url", "publish_date",
            "text", "summary", "source"
        }],
        "n_articles":   int,
        "horizon_days": float | null,   # days from created to resolved
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Path setup ───────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from analyzing_llm_rationale.entity_tagger import tag_question  # noqa: E402

_DEFAULT_INPUT = _REPO / "forecasting_qa_news_metaculus_2025-02-01_to_today.metaculus_frs_format.json"
_DEFAULT_OUTPUT = _REPO / "data" / "kg_dataset.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.rstrip("Z").replace(" ", "T")
    # Handle varying precision
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _outcome(answer: Optional[str]) -> Optional[int]:
    a = (answer or "").strip().lower()
    if a == "yes":
        return 1
    if a == "no":
        return 0
    return None


def _clean_article(art: Dict[str, Any], before: datetime) -> Optional[Dict[str, Any]]:
    """Return a stripped article dict if it was published before `before`, else None."""
    pub = _parse_dt(art.get("publish_date"))
    if pub is None or pub >= before:
        return None
    return {
        "title":        (art.get("title") or "").strip(),
        "url":          (art.get("url") or "").strip(),
        "publish_date": art["publish_date"],
        "text":         (art.get("text") or "").strip(),
        "summary":      (art.get("summary") or art.get("summary_llm") or "").strip(),
        "source":       _source_from_url(art.get("url") or ""),
    }


def _source_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc
        return host.removeprefix("www.")
    except Exception:
        return ""


# ── Main ─────────────────────────────────────────────────────────────────────

def build(
    input_path: Path,
    output_path: Path,
    min_articles: int = 0,
    created_after: Optional[str] = None,
) -> List[Dict[str, Any]]:
    print(f"Reading {input_path} …")
    with open(input_path) as f:
        raw = json.load(f)
    print(f"  {len(raw)} records")

    created_after_dt: Optional[datetime] = None
    if created_after:
        created_after_dt = _parse_dt(created_after)

    dataset: List[Dict[str, Any]] = []
    skipped_no_outcome = 0
    skipped_no_resolve = 0
    skipped_min_articles = 0
    total_articles_kept = 0
    total_articles_dropped = 0

    for i, rec in enumerate(raw):
        outcome = _outcome(rec.get("answer"))
        if outcome is None:
            skipped_no_outcome += 1
            continue

        resolve_time = _parse_dt(rec.get("resolve_time"))
        if resolve_time is None:
            skipped_no_resolve += 1
            continue

        # Filter articles to pre-resolution only
        articles: List[Dict[str, Any]] = []
        for art in rec.get("news_articles") or []:
            cleaned = _clean_article(art, resolve_time)
            if cleaned is not None:
                articles.append(cleaned)
                total_articles_kept += 1
            else:
                total_articles_dropped += 1

        if len(articles) < min_articles:
            skipped_min_articles += 1
            continue

        created_time = _parse_dt(rec.get("created_time") or rec.get("publish_time"))
        horizon_days: Optional[float] = None
        if created_time and resolve_time:
            horizon_days = round((resolve_time - created_time).total_seconds() / 86400, 1)

        question = (rec.get("question") or "").strip()
        categories = [c for c in (rec.get("categories") or []) if isinstance(c, str)]
        tags = tag_question(question, categories[0] if categories else None)

        record: Dict[str, Any] = {
            "id":           f"metaculus:{rec.get('id', i)}",
            "question":     question,
            "outcome":      outcome,
            "resolve_time": resolve_time.isoformat(),
            "created_time": created_time.isoformat() if created_time else None,
            "horizon_days": horizon_days,
            "domain":       tags["domain"],
            "entities":     tags["entities"],
            "categories":   categories,
            "n_articles":   len(articles),
            "articles":     articles,
        }
        dataset.append(record)

        if (i + 1) % 200 == 0:
            print(f"  … {i+1}/{len(raw)} processed, {len(dataset)} kept")

    print(f"\nDone:")
    print(f"  Records kept:           {len(dataset)}")
    print(f"  Skipped (no outcome):   {skipped_no_outcome}")
    print(f"  Skipped (no resolve):   {skipped_no_resolve}")
    print(f"  Skipped (min articles): {skipped_min_articles}")
    print(f"  Articles kept:          {total_articles_kept}")
    print(f"  Articles dropped (post-resolution): {total_articles_dropped}")

    # Domain breakdown
    from collections import Counter
    domains = Counter(r["domain"] for r in dataset)
    print("\nDomain distribution:")
    for d, n in domains.most_common():
        yes = sum(1 for r in dataset if r["domain"] == d and r["outcome"] == 1)
        pct = 100 * yes / n
        print(f"  {d:20s} {n:5d}  yes-rate={pct:.0f}%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nSaved → {output_path}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(_DEFAULT_INPUT))
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--min-articles", type=int, default=0,
                        help="Drop records with fewer than N pre-resolution articles")
    args = parser.parse_args()
    build(Path(args.input), Path(args.output), min_articles=args.min_articles)


if __name__ == "__main__":
    main()
