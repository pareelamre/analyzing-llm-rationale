#!/usr/bin/env python3
"""Fetch 5 pre-resolution articles per KG dataset question.

Sources (tried in order per question):
  1. DuckDuckGo News (ddgs) — fast, parallel; post-filtered to date < resolve_time.
  2. GNews — Google News RSS; exact start/end date support; no API key needed.yeah fi
  3. GDELT — exact ENDDATETIME; serialized at 1 req/5 s.

After collecting URLs, full article text is fetched via trafilatura (parallel,
8 s timeout each) to match reference-dataset text quality (~5–40k chars).
DDG + GNews run in parallel (--workers threads).  GDELT is always serialised
(rate-limit shared across workers via a lock).

Checkpoints to the output file every CHECKPOINT_EVERY records.

Usage:
    python scripts/fetch_kg_articles.py
    python scripts/fetch_kg_articles.py --workers 8 --source kalshi --limit 200
    python scripts/fetch_kg_articles.py --dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
TARGET_ARTICLES = 5          # paper: 5+ articles → approaches crowd-level Brier score
GDELT_RATE_LIMIT_S = 5.5     # GDELT free tier hard limit: 1 req / 5 s
DDG_SLEEP_S = 0.3            # polite delay per DDG call within a worker
DEFAULT_WORKERS = 8
CHECKPOINT_EVERY = 100
DEFAULT_INPUT = Path("data/kg_dataset.json")
MIN_TEXT_LEN = 400           # enrich via full-text fetch if text shorter than this
MAX_TEXT_CHARS = 40_000      # cap stored text to match reference dataset
URL_FETCH_TIMEOUT = 8        # seconds per article URL fetch
URL_FETCH_WORKERS = 5        # parallel URL fetches per question

_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# ---------------------------------------------------------------------------
# Trafilatura full-text extraction
# ---------------------------------------------------------------------------

try:
    import trafilatura as _traf
    _TRAF_OK = True
except ImportError:
    _TRAF_OK = False


_GOOGLE_URL_PAT = re.compile(
    r"^https?://(?:(?:www\.)?google\.com|news\.google\.com)(?:/|$)"
)
# Fingerprint of Google consent/redirect page extracted by trafilatura
_GARBAGE_PAT = re.compile(r"EnglishUnited States|Sign in - Google Accounts|Before you continue")


def _fetch_full_text(url: str) -> str:
    """Fetch and extract full article text from a URL using trafilatura."""
    if not _TRAF_OK or not url:
        return ""
    if _GOOGLE_URL_PAT.match(url):
        return ""  # Google redirect URLs → consent page, not article content
    try:
        r = requests.get(url, headers=_FETCH_HEADERS,
                         timeout=URL_FETCH_TIMEOUT, allow_redirects=True)
        if r.status_code != 200:
            return ""
        # Also skip if we landed on a Google page after redirect
        if _GOOGLE_URL_PAT.match(r.url):
            return ""
        text = _traf.extract(r.text, include_comments=False, include_tables=False,
                             no_fallback=False)
        if not text or _GARBAGE_PAT.search(text):
            return ""
        return text[:MAX_TEXT_CHARS]
    except Exception:
        return ""


def _resolve_title_to_url(title: str) -> str:
    """Use DDG web search to find a real article URL from a headline title."""
    if not _DDG_OK or not title:
        return ""
    try:
        with _DDGS() as ddgs:
            results = list(ddgs.text(title, max_results=1))
        if results:
            return results[0].get("href") or results[0].get("url", "")
    except Exception:
        pass
    return ""


def _enrich_texts(arts: List[Dict]) -> List[Dict]:
    """Fetch full text for articles whose stored text is too short.

    For google.com redirect URLs (GNews), resolves the real article URL via a
    DDG title search before fetching, since Google's consent redirect blocks
    direct access.
    """
    need = [i for i, a in enumerate(arts)
            if len(a.get("text") or "") < MIN_TEXT_LEN and a.get("url")]
    if not need:
        return arts

    def fetch_one(i: int) -> Tuple[int, str]:
        a = arts[i]
        url = a.get("url", "")
        # For Google News encoded URLs, find the real article URL via title search
        if _GOOGLE_URL_PAT.match(url):
            title = a.get("title", "")
            real_url = _resolve_title_to_url(title)
            if real_url:
                url = real_url
            else:
                return i, ""
        return i, _fetch_full_text(url)

    with concurrent.futures.ThreadPoolExecutor(max_workers=URL_FETCH_WORKERS) as pool:
        for i, text in pool.map(fetch_one, need):
            if text:
                arts[i]["text"] = text
                arts[i]["summary"] = text[:500]
    return arts


# ---------------------------------------------------------------------------
# Semantic query construction (NLTK NER)
# ---------------------------------------------------------------------------

try:
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        import nltk as _nltk
        from nltk import ne_chunk as _nc
        from nltk import pos_tag as _pt
        from nltk import word_tokenize as _wt
        for _r in ("punkt_tab", "averaged_perceptron_tagger_eng",
                   "maxent_ne_chunker_tab", "words"):
            _nltk.download(_r, quiet=True)
    _NLTK_OK = True
except Exception:
    _NLTK_OK = False

try:
    from ddgs import DDGS as _DDGS
    _DDG_OK = True
except ImportError:
    try:
        from duckduckgo_search import DDGS as _DDGS
        _DDG_OK = True
    except ImportError:
        _DDG_OK = False

try:
    from gnews import GNews as _GNews
    _GNEWS_OK = True
except ImportError:
    _GNEWS_OK = False

_STOPWORDS = {
    "will", "the", "a", "an", "is", "it", "be", "been", "before", "after",
    "by", "on", "or", "and", "to", "in", "at", "of", "for", "with", "this",
    "that", "than", "more", "most", "over", "under", "have", "has", "do",
    "does", "did", "would", "could", "should", "may", "might", "what",
    "when", "where", "who", "which", "how", "why", "if", "not", "no",
}


def _search_query(question: str, max_terms: int = 8) -> str:
    if _NLTK_OK:
        return _semantic_query(question, max_terms)
    return _lexical_query(question, max_terms)


def _semantic_query(question: str, max_terms: int) -> str:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tokens = _wt(question)
            tagged = _pt(tokens)
            tree = _nc(tagged)
        except Exception:
            return _lexical_query(question, max_terms)
    parts: List[str] = []
    seen: set = set()
    for subtree in tree:
        if hasattr(subtree, "label"):
            phrase = " ".join(w for w, _ in subtree.leaves())
            if phrase.lower() not in {"will"}:
                parts.append(f'"{phrase}"' if " " in phrase else phrase)
                seen.update(w.lower() for w, _ in subtree.leaves())
    for word, tag in tagged:
        if tag in ("NNP", "NNPS") and word.lower() not in seen:
            parts.append(word)
            seen.add(word.lower())
    for word, tag in tagged:
        if (tag in ("NN", "NNS") and word.lower() not in seen
                and word.lower() not in _STOPWORDS and len(word) > 3):
            parts.append(word)
            seen.add(word.lower())
    for word, tag in tagged:
        if tag == "CD" and re.fullmatch(r"20\d{2}|19\d{2}", word) and word not in parts:
            parts.append(word)
    return " ".join(parts[:max_terms]) if parts else _lexical_query(question, max_terms)


def _lexical_query(question: str, max_terms: int) -> str:
    terms = re.findall(r"[A-Za-z0-9$€£]+", question)
    kept = [t for t in terms if t.lower() not in _STOPWORDS and len(t) > 2]
    return " ".join(kept[:max_terms]) or question[:100]


# ---------------------------------------------------------------------------
# Article helpers
# ---------------------------------------------------------------------------

def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _art(title: str, url: str, pub: str, src: str, ch: str,
         body: str = "") -> Dict[str, Any]:
    text = body or title
    return {"title": title, "url": url, "publish_date": pub,
            "text": text, "summary": text, "source": src, "source_channel": ch}


def _date_filter(arts: List[Dict], before: datetime) -> List[Dict]:
    out = []
    for a in arts:
        dt = _parse_iso(a.get("publish_date", ""))
        if dt is not None and dt < before:
            out.append(a)
    return out


def _merge(*lists: List[Dict]) -> List[Dict]:
    seen_urls: set = set()
    out = []
    for lst in lists:
        for a in lst:
            u = a.get("url", "")
            if u not in seen_urls:
                seen_urls.add(u)
                out.append(a)
    return out


# ---------------------------------------------------------------------------
# Source 1: DuckDuckGo News
# ---------------------------------------------------------------------------

def _fetch_ddg(query: str, resolve_dt: datetime, limit: int = 10) -> List[Dict]:
    if not _DDG_OK:
        return []
    try:
        with _DDGS() as ddgs:
            raw = list(ddgs.news(query, max_results=limit))
        time.sleep(DDG_SLEEP_S)
        arts = [_art(r.get("title", ""), r.get("url", ""), r.get("date", ""),
                     r.get("source", "DDG"), "ddg", body=r.get("body", ""))
                for r in raw]
        return _date_filter(arts, resolve_dt)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Source 2: GNews (Google News RSS — exact date range support)
# ---------------------------------------------------------------------------

def _fetch_gnews(query: str, resolve_dt: datetime, limit: int = 10) -> List[Dict]:
    if not _GNEWS_OK:
        return []
    try:
        start = resolve_dt - timedelta(days=365)
        gn = _GNews(
            language="en",
            country="US",
            max_results=limit,
            start_date=(start.year, start.month, start.day),
            end_date=(resolve_dt.year, resolve_dt.month, resolve_dt.day),
        )
        raw = gn.get_news(query)
        return [
            _art(
                title=r.get("title", ""),
                url=r.get("url", ""),
                pub=r.get("published date", ""),
                src=r.get("publisher", {}).get("title", "GNews"),
                ch="gnews",
                body=r.get("description", ""),
            )
            for r in raw
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Source 3: GDELT (serialised via a shared lock)
# ---------------------------------------------------------------------------

_gdelt_lock = threading.Lock()
_gdelt_last_call = 0.0


def _gdelt_dt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def _fetch_gdelt_locked(query: str, before: datetime,
                        after: Optional[datetime] = None,
                        limit: int = 10) -> List[Dict]:
    global _gdelt_last_call
    with _gdelt_lock:
        elapsed = time.time() - _gdelt_last_call
        if elapsed < GDELT_RATE_LIMIT_S:
            time.sleep(GDELT_RATE_LIMIT_S - elapsed)
        result = _fetch_gdelt_raw(query, before, after, limit)
        _gdelt_last_call = time.time()
    return result


def _fetch_gdelt_raw(query: str, before: datetime,
                     after: Optional[datetime] = None,
                     limit: int = 10) -> List[Dict]:
    params: Dict[str, Any] = {
        "query": query, "mode": "ArtList", "format": "json",
        "maxrecords": min(limit, 250), "sort": "HybridRel",
        "ENDDATETIME": _gdelt_dt(before),
    }
    if after:
        params["STARTDATETIME"] = _gdelt_dt(after)
    try:
        r = requests.get(GDELT_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            return []
        return [_art(a.get("title", ""), a.get("url", ""),
                     a.get("seendate", ""),
                     a.get("source") or a.get("domain") or "GDELT", "gdelt")
                for a in data.get("articles", [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Per-question fetch: DDG → GNews → GDELT → full-text enrichment
# ---------------------------------------------------------------------------

def fetch_for_question(
    question: str,
    resolve_time: str,
    existing_articles: Optional[List[Dict]] = None,
    article_sources: Optional[set[str]] = None,
) -> List[Dict]:
    resolve_dt = _parse_iso(resolve_time)
    if not resolve_dt:
        return []
    query = _search_query(question)
    existing = _date_filter(existing_articles or [], resolve_dt)
    enabled = article_sources or {"ddg", "gnews", "gdelt"}

    # 1. DDG — fast, no hard rate limit
    arts = (
        _fetch_ddg(query, resolve_dt, limit=TARGET_ARTICLES + 5)
        if "ddg" in enabled else []
    )
    if len(arts) >= TARGET_ARTICLES:
        arts = arts[:TARGET_ARTICLES]
    elif "gnews" in enabled:
        # 2. GNews — exact date range via Google News RSS
        arts = _merge(arts, _fetch_gnews(query, resolve_dt, limit=TARGET_ARTICLES + 5))
    if len(arts) >= TARGET_ARTICLES:
        arts = arts[:TARGET_ARTICLES]
    elif "gdelt" in enabled:
        # 3. GDELT — widening windows (rate-limited, exact ENDDATETIME)
        for days in (30, 90, 365):
            arts = _merge(arts, _fetch_gdelt_locked(
                query, before=resolve_dt,
                after=resolve_dt - timedelta(days=days),
                limit=TARGET_ARTICLES + 5))
            if len(arts) >= TARGET_ARTICLES:
                break
        else:
            # 4. GDELT — no lower bound
            arts = _merge(arts, _fetch_gdelt_locked(
                query, before=resolve_dt, limit=TARGET_ARTICLES + 5))
        arts = arts[:TARGET_ARTICLES]

    arts = _date_filter(_merge(existing, arts), resolve_dt)[:TARGET_ARTICLES]

    # 5. Enrich short texts by fetching full article content
    return _enrich_texts(arts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=None)
    ap.add_argument("--source", default=None,
                    help="kalshi|polymarket|manifold|metaculus")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-missing", action="store_true",
                    help="Only enrich records with zero existing articles (leave under-served records untouched).")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"Parallel workers (default {DEFAULT_WORKERS})")
    ap.add_argument("--article-sources", default="ddg,gnews,gdelt",
                    help="Comma-separated article sources: ddg,gnews,gdelt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    valid_sources = {"ddg", "gnews", "gdelt"}
    article_sources = {s.strip().lower()
                       for s in args.article_sources.split(",") if s.strip()}
    unknown_sources = article_sources - valid_sources
    if unknown_sources:
        raise SystemExit(f"Unknown article source(s): {', '.join(sorted(unknown_sources))}")
    if not article_sources:
        raise SystemExit("At least one --article-sources value is required")

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    print(f"Sources: DDG={'yes' if _DDG_OK and 'ddg' in article_sources else 'no'}  "
          f"GNews={'yes' if _GNEWS_OK and 'gnews' in article_sources else 'no'}  "
          f"GDELT={'yes' if 'gdelt' in article_sources else 'no'}  "
          f"trafilatura={'yes' if _TRAF_OK else 'no'}  workers={args.workers}")
    records: List[Dict[str, Any]] = json.loads(input_path.read_text())
    print(f"Loaded {len(records)} records")

    threshold = 1 if args.only_missing else TARGET_ARTICLES
    needs: List[Tuple[int, Dict]] = [
        (i, r) for i, r in enumerate(records)
        if len(r.get("news_articles") or []) < threshold
        and (args.source is None or r.get("source") == args.source)
    ]
    if args.limit:
        needs = needs[:args.limit]

    by_src: Dict[str, int] = {}
    for _, r in needs:
        s = r.get("source", "?")
        by_src[s] = by_src.get(s, 0) + 1
    print(f"{len(needs)} records need articles:")
    for s, n in sorted(by_src.items(), key=lambda x: -x[1]):
        print(f"  {s}: {n}")

    if args.dry_run:
        est_s = len(needs) * (TARGET_ARTICLES * URL_FETCH_TIMEOUT) / args.workers
        print(f"\nEstimated runtime (URL-fetch-bound): ~{est_s/3600:.1f} h with {args.workers} workers")
        return

    _lock = threading.Lock()
    updated = [0]
    skipped = [0]
    seq_counter = [0]

    def process(seq: int, idx: int, record: Dict) -> Tuple[int, int, List[Dict], str]:
        q = record.get("question", "")
        rt = record.get("resolve_time", "")
        arts = fetch_for_question(
            q,
            rt,
            record.get("news_articles") or [],
            article_sources=article_sources,
        )
        channels = ",".join(sorted(
            {a.get("source_channel", "existing") for a in arts}
        )) if arts else ""
        return seq, idx, arts, channels

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, seq, idx, r): seq
                   for seq, (idx, r) in enumerate(needs)}

        for future in concurrent.futures.as_completed(futures):
            _, r_idx, r_arts, r_ch = future.result()
            q_text = records[r_idx].get("question", "")[:60]
            src = records[r_idx].get("source", "?")
            avg_len = (sum(len(a.get("text", "")) for a in r_arts) // len(r_arts)
                       if r_arts else 0)

            with _lock:
                seq_counter[0] += 1
                seq = seq_counter[0]
                if r_arts:
                    records[r_idx]["news_articles"] = r_arts
                    updated[0] += 1
                else:
                    skipped[0] += 1

            tag = f"→ {len(r_arts)} [{r_ch}] avg_txt={avg_len}" if r_arts else "→ none"
            print(f"  [{seq}/{len(needs)}] {src}: {q_text}… {tag}", flush=True)

            if seq % CHECKPOINT_EVERY == 0:
                with _lock:
                    output_path.write_text(
                        json.dumps(records, ensure_ascii=False, indent=2))
                print(f"  ── checkpoint {seq} (updated={updated[0]} skipped={skipped[0]}) ──",
                      flush=True)

    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"\nDone: {updated[0]} updated · {skipped[0]} no articles → {output_path}")


if __name__ == "__main__":
    main()
