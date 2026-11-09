#!/usr/bin/env python3
"""Headlines for the fund's news-sentiment evaluation from a historical corpus.

Implements Amendment 1 of the pre-registered rule (docs/fund.md): headlines come from the
``idleengine/financial-news-multisource`` corpus (non-commercial research terms), subsets in
SUBSETS. Only each item's first line, its headline, is kept. A headline joins every group
whose terms it matches (every sp500_daily_headlines item is US equities). Minute-level
timestamps are used as given; an item stamped midnight UTC is day-level and is treated as
seen at the end of its UTC day, so it can never inform that day's close. A headline repeated
within a group on the same day counts once, at its earliest time.

Output, the same as fund_news_collect.py: OUT_DIR/<group>.jsonl, one {"seen", "title"} line
per headline, and OUT_DIR/_coverage.json with counts and date ranges per subset and group.

Usage: fund_news_corpus.py [--corpus DIR] [--out data/fund_news]
       (without --corpus the subsets are downloaded from Hugging Face first)
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

REPO = "idleengine/financial-news-multisource"
SUBSETS = ("sp500_daily_headlines", "cnbc_headlines", "finsen_us_2007_2023", "yahoo_finance_felixdrinkall",
           "bloomberg_reuters", "us_financial_news_2018_jeet")
ALL_US_EQUITY = "sp500_daily_headlines"
GROUPS = {
    "us_equity": re.compile(r"\bS&P\b|\bwall street\b|\bdow\b|\bnasdaq\b|\bU\.?S\.? stocks\b", re.I),
    "crude": re.compile(r"\bcrude\b|\boil prices?\b|\bbrent\b|\bopec\b|\bwti\b", re.I),
    "india_equity": re.compile(r"\bnifty\b|\bsensex\b|\bindian (?:shares|stocks)\b", re.I),
}


def headline(text: str) -> str:
    """The item's first line; Bloomberg's headlines start with '-- '."""
    first = (text or "").strip().split("\n", 1)[0].strip()
    return first[3:].strip() if first.startswith("-- ") else first


def seen_at(stamp: str) -> datetime:
    """When the headline could first be known: midnight UTC means a day-level stamp."""
    ts = datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)
    if (ts.hour, ts.minute, ts.second, ts.microsecond) == (0, 0, 0, 0):
        ts += timedelta(days=1)
    return ts


def groups_for(subset: str, title: str) -> list[str]:
    hits = [group for group, pattern in GROUPS.items() if pattern.search(title)]
    if subset == ALL_US_EQUITY and "us_equity" not in hits:
        hits.append("us_equity")
    return hits


def read_subset(path: Path) -> Iterator[tuple[str, str]]:
    """(date, text) for every row of a subset's parquet files, a batch at a time."""
    import pyarrow.parquet as pq

    for file in sorted(path.glob("*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=20_000, columns=["date", "text"]):
            cols = batch.to_pydict()
            yield from zip(cols["date"], cols["text"])


def build(rows_by_subset: dict[str, Iterable[tuple[str, str]]], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    kept: dict[str, dict[tuple[str, str], datetime]] = {group: {} for group in GROUPS}
    subsets = {}
    for subset, rows in rows_by_subset.items():
        n = matched = 0
        first = last = None
        for stamp, text in rows:
            n += 1
            title = headline(text)
            if not stamp or not title:
                continue
            try:
                seen = seen_at(stamp)
            except ValueError:
                continue
            hits = groups_for(subset, title)
            if not hits:
                continue
            matched += 1
            first, last = min(first or seen, seen), max(last or seen, seen)
            for group in hits:
                key = (title, seen.date().isoformat())
                if key not in kept[group] or seen < kept[group][key]:
                    kept[group][key] = seen
        subsets[subset] = {"rows": n, "matched": matched, "first": first and first.isoformat(),
                           "last": last and last.isoformat()}
        print(f"{subset}: {n} rows, {matched} matched a group", flush=True)
    groups = {}
    for group, items in kept.items():
        ordered = sorted((seen, title) for (title, _), seen in items.items())
        with (out_dir / f"{group}.jsonl").open("w", encoding="utf-8") as fh:
            for seen, title in ordered:
                fh.write(json.dumps({"seen": seen.isoformat(), "title": title}, ensure_ascii=False) + "\n")
        groups[group] = {"headlines": len(ordered), "first": ordered[0][0].isoformat() if ordered else None,
                         "last": ordered[-1][0].isoformat() if ordered else None}
    coverage = {"source": f"huggingface.co/datasets/{REPO}", "subsets": subsets, "groups": groups}
    (out_dir / "_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    return coverage


def download(dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO, repo_type="dataset", local_dir=dest,
                                  allow_patterns=[f"data/{subset}/*" for subset in SUBSETS]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, help="a local copy of the corpus (its data/ folder's parent)")
    parser.add_argument("--out", type=Path, default=Path("data/fund_news"))
    args = parser.parse_args()
    root = args.corpus or download(Path("data/hf_corpus"))
    coverage = build({subset: read_subset(root / "data" / subset) for subset in SUBSETS}, args.out)
    print(json.dumps(coverage["groups"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
