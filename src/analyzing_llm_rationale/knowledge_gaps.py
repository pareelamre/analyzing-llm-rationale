"""Which of Foresea's live questions the knowledge base cannot help with yet.

The base should grow toward what Foresea is actually asked, not toward whatever report is handy.
This scan takes the open questions Foresea tracks (the track-record store and the Edge board) and
the fund's commodity pairs, asks the base about each one the way the forecast and agent paths do
(a claim counts only when two retrievers agree), and lists what came back empty.

Agreement alone is generous, so a question counts as covered only when an agreed claim is on its
topic: it shares a topical entity with the question (a commodity, technology, policy, event; not
a country or region) or a topic word in its text. Questions where the base answers with off-topic
claims are listed apart, since that is noise reaching the forecast prompt.

Only long-horizon questions count: a market that closes within ``MIN_DAYS`` is decided by news,
not by the kind of report the base holds, and sports and entertainment markets are out of scope.
Every input is a public market question, so the report is safe to keep in the private base repo;
it never contains the base's own text.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from . import fund_knowledge, knowledge

MIN_DAYS = 30
SKIP_DOMAINS = frozenset({"sports", "entertainment"})
TOPIC_WORDS = 15


def _when(value: Any) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return knowledge.parse_when(text[:10])


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def open_questions(root: Path, today: date, *, min_days: int = MIN_DAYS) -> list[dict[str, Any]]:
    """Open long-horizon questions from the track-record store and the Edge board, one row each."""
    horizon = today + timedelta(days=min_days)
    rows: dict[str, dict[str, Any]] = {}

    def keep(question: Any, closes: Any, domain: Any, origin: str, url: Any) -> None:
        text = " ".join(str(question or "").split())
        when = _when(closes)
        dom = str(domain or "other").strip().lower() or "other"
        if not text or when is None or when < horizon or dom in SKIP_DOMAINS:
            return
        key = text.lower()
        if key not in rows:
            rows[key] = {"question": text, "domain": dom, "closes": when.isoformat(), "origin": origin,
                         "url": str(url or "") or None}

    store = _load(root / "data" / "track_record_store.json") or {}
    for snap in store.get("ForecastSnapshot") or []:
        if isinstance(snap, Mapping) and not snap.get("resolved"):
            keep(snap.get("question"), snap.get("close_time"), snap.get("domain"), "track record",
                 snap.get("market_url"))
    board = _load(root / "static" / "mark_to_market_live.json") or {}
    for row in board.get("edge_board") or []:
        if isinstance(row, Mapping):
            keep(row.get("question"), row.get("resolve_time"), row.get("domain"), "edge board",
                 row.get("market_url"))
    return sorted(rows.values(), key=lambda r: (r["domain"], r["closes"], r["question"]))


def fund_pairs(root: Path) -> list[dict[str, str]]:
    """The fund's commodity pairs as (key, leg a, leg b) underlyings."""
    universe = _load(root / "static" / "fund_universe.json") or {}
    out = []
    for pair in universe.get("pairs") or []:
        if not isinstance(pair, Mapping):
            continue
        a, b = (pair.get("a") or {}).get("underlying"), (pair.get("b") or {}).get("underlying")
        if pair.get("key") and a and b:
            out.append({"key": str(pair["key"]), "a": str(a), "b": str(b)})
    return out


def _leg_gap(index: knowledge.Index, underlying: str, today: date) -> Optional[str]:
    """Why the base cannot forecast a leg for this year and next, or None when it can."""
    metric = fund_knowledge.LEG_METRICS.get(underlying)
    if metric is None:
        return f"no knowledge-base metric is mapped to {underlying}"
    years = {knowledge.parse_when(f["claim"].get("horizon")).year
             for f in index.figures(metric[0], as_of=today, geo=metric[1])
             if knowledge.parse_when(f["claim"].get("horizon")) is not None}
    missing = [y for y in (today.year, today.year + 1) if y not in years]
    if missing:
        return f"no {metric[0]} forecast for {', '.join(str(y) for y in missing)}"
    return None


def topic_words(questions: Iterable[str], *, k: int = TOPIC_WORDS) -> list[tuple[str, int]]:
    """The words that recur across questions, ignoring filler, numbers and years."""
    counts: Counter[str] = Counter()
    for question in questions:
        counts.update(w for w in knowledge.terms(question) if knowledge.topical_word(w))
    return [(w, n) for w, n in counts.most_common(k) if n > 1]


def scan(index: knowledge.Index, questions: Sequence[Mapping[str, Any]],
         pairs: Sequence[Mapping[str, str]], today: date) -> dict[str, Any]:
    """Ask the base about every question and pair; report what it covers and what it misses."""
    covered, gaps = [], []
    for row in questions:
        question = str(row["question"])
        on = index.agreed(question, as_of=today)
        # What plain agreement would have let through: the noise the topic check now keeps out.
        loose = [h for h in index.search(question, as_of=today, k=24) if h.support >= 2 and not h.superseded]
        entry = {**row, "claims": len(on), "off_topic": len(loose) - len(on)}
        if on:
            entry["sources"] = sorted({h.cite() for h in on[:4]})
            covered.append(entry)
        else:
            gaps.append(entry)
    pair_rows = []
    for pair in pairs:
        if pair["a"] == pair["b"]:
            # One commodity on two venues: a basis trade, which a price-ratio forecast cannot inform.
            continue
        view = fund_knowledge.pair_view(index, pair["a"], pair["b"], today)
        reasons = [r for r in (_leg_gap(index, pair["a"], today), _leg_gap(index, pair["b"], today)) if r]
        if view is None and not reasons:
            reasons = ["no single source forecasts both legs for this year and next"]
        pair_rows.append({"key": pair["key"], "covered": view is not None,
                          "source": (view or {}).get("cite"), "missing": reasons if view is None else []})
    by_domain: dict[str, dict[str, int]] = {}
    for entries, hit in ((covered, 1), (gaps, 0)):
        for entry in entries:
            cell = by_domain.setdefault(entry["domain"], {"questions": 0, "covered": 0})
            cell["questions"] += 1
            cell["covered"] += hit
    return {
        "date": today.isoformat(),
        "questions": len(questions), "covered": len(covered),
        "pairs": len(pair_rows), "pairs_covered": sum(1 for p in pair_rows if p["covered"]),
        "by_domain": dict(sorted(by_domain.items())),
        "topic_words": topic_words(g["question"] for g in gaps),
        "off_topic_only": sum(1 for g in gaps if g["off_topic"]),
        "gaps": gaps, "covered_questions": covered, "fund_pairs": pair_rows,
    }


def markdown(report: Mapping[str, Any]) -> str:
    """The report as a page a person reads to pick the next sources."""
    q, c = report["questions"], report["covered"]
    lines = [f"# Knowledge base gaps, {report['date']}", "",
             f"- Open long-horizon questions: {q}; the base has something on {c}"
             + (f" ({c / q:.0%})" if q else "") + ".",
             f"- Fund pairs with a knowledge-base view: {report['pairs_covered']} of {report['pairs']}"
             " (pairs of one commodity on two venues are left out).",
             f"- Unanswered questions where agreement alone would have passed off-topic claims (now held back): {report['off_topic_only']}.", ""]
    if report["topic_words"]:
        lines += ["## Most common words in unanswered questions", "",
                  ", ".join(f"{w} ({n})" for w, n in report["topic_words"]), ""]
    lines += ["## By domain", "", "| Domain | Questions | Covered |", "|---|---|---|"]
    lines += [f"| {d} | {v['questions']} | {v['covered']} |" for d, v in report["by_domain"].items()]
    missing_pairs = [p for p in report["fund_pairs"] if not p["covered"]]
    if missing_pairs:
        lines += ["", "## Fund pairs without a view", ""]
        lines += [f"- {p['key']}: {'; '.join(p['missing'])}" for p in missing_pairs]
    lines += ["", "## Unanswered questions", ""]
    for domain in sorted({g["domain"] for g in report["gaps"]}):
        lines += [f"### {domain}", ""]
        lines += [f"- {g['question']} (closes {g['closes']}, {g['origin']})"
                  + (f" — {g['off_topic']} off-topic claims held back" if g["off_topic"] else "")
                  for g in report["gaps"] if g["domain"] == domain]
        lines.append("")
    if report["covered_questions"]:
        lines += ["## Questions the base already helps with", ""]
        lines += [f"- {e['question']} ({', '.join(e['sources'])})" for e in report["covered_questions"]]
    return "\n".join(lines).rstrip() + "\n"


def history_row(report: Mapping[str, Any]) -> dict[str, Any]:
    return {k: report[k] for k in ("date", "questions", "covered", "pairs", "pairs_covered")}


def write(report: Mapping[str, Any], out: Path) -> None:
    """latest.json, latest.md and one history line per day (a rerun replaces that day's line)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "latest.md").write_text(markdown(report), encoding="utf-8")
    path = out / "history.jsonl"
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("date") != report["date"]] + [history_row(report)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def today_utc() -> date:
    return datetime.now(timezone.utc).date()
