"""A paper collector for the knowledge base: forecasting in every form, quantitative finance and
prediction markets (owner, 2026-11-28: "start a collector of papers for quant finance and
prediction markets to enrich foresea knowledge and for foresea to evolve using this knowledge
corpus, ideally let it run indefinitely" ... "cover all forms of forecasting").

Daily, in the data repo's "Watch and gaps" job (no model involved):

- **arXiv** daily RSS feeds (rss.arxiv.org, which has no robots rules; one request per archive):
  every quantitative-finance paper, and papers from the other archives (economics, statistics,
  machine learning, AI, language, atmospheric physics, epidemiology, systems) whose title or
  abstract is about forecasting (``FORECASTING``). The feeds are empty at weekends.
- **OpenAlex** (api.openalex.org, robots: allow all; its metadata is CC0): new works in the
  forecasting, finance, weather, epidemics, elections and sports journals in ``JOURNALS``, and
  works anywhere about prediction markets and forecast aggregation (``SEARCHES``).

Each new paper becomes a card in ``papers/cards/<YYYY-MM>.jsonl`` (the month it was found): title,
authors, abstract, venue, date, links, licence, the forecasting topics it touches and a
relevance score for Foresea (``score``: its main topic's weight from ``TOPICS``, half the next
one's, and a point each for an open PDF, a reviewed venue and a forecasting journal). An arXiv paper OpenAlex also indexes keeps its arXiv card.

Weekly (Mondays, ``weekly``): the best few unread cards with an open PDF (``PER_WEEK``, score at
least ``MIN_READ_SCORE``, a main topic core to Foresea) go to ``inbox.jsonl`` as reports, so the Mac's night reader reads them
and the morning review decides every claim, the same gates as any report; and
``papers/digest/latest.md`` lists the week's best papers by the part of Foresea they could
improve, the input for Foresea's next tests.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from analyzing_llm_rationale import knowledge_watch as watch

ARXIV_RSS = "https://rss.arxiv.org/rss/{feed}"
#: arXiv archive -> "all" (keep every paper) or "forecast" (keep it only if it is about forecasting).
ARXIV_FEEDS: dict[str, str] = {
    "q-fin": "all", "econ": "forecast", "stat": "forecast", "cs.LG": "forecast", "cs.AI": "forecast",
    "cs.CL": "forecast", "physics.ao-ph": "forecast", "q-bio.PE": "forecast", "eess.SY": "forecast",
}
OPENALEX = "https://api.openalex.org/works"
#: OpenAlex source ids (checked 2026-11-28 by ISSN), with the ones that are forecasting journals.
JOURNALS: dict[str, str] = {
    "International Journal of Forecasting": "S39841227", "Journal of Forecasting": "S122161766",
    "The Journal of Prediction Markets": "S4210199549", "Quantitative Finance": "S182689569",
    "Journal of Financial Econometrics": "S207154782", "Weather and Forecasting": "S203100778",
    "Monthly Weather Review": "S90791864", "Journal of Business and Economic Statistics": "S18095783",
    "Journal of Applied Econometrics": "S85739584", "Journal of Futures Markets": "S136612413",
    "Energy Economics": "S94499970", "Decision Analysis": "S33235364",
    "Judgment and Decision Making": "S179575189", "Mathematical Finance": "S133335107",
    "Journal of commodity markets": "S2898531903", "Technological Forecasting and Social Change": "S39307421",
    "Epidemics": "S24717195", "Quarterly Journal of the Royal Meteorological Society": "S70522100",
    "Journal of Quantitative Analysis in Sports": "S78466176", "Electoral Studies": "S124584455",
    "Journal of Financial Economics": "S149240962", "Review of Financial Studies": "S170137484",
    "The Journal of Finance": "S5353659", "Journal of Banking & Finance": "S2876017",
}
FORECASTING_JOURNALS = {"International Journal of Forecasting", "Journal of Forecasting",
                        "The Journal of Prediction Markets", "Weather and Forecasting"}
SEARCHES = ('"prediction market" OR "prediction markets" OR "betting markets" OR "forecast aggregation" '
            'OR superforecasting OR "forecasting tournament" OR "wisdom of crowds"',)
LOOKBACK_DAYS = 30
OPENALEX_PAGES = 5
PER_WEEK = 3
MIN_READ_SCORE = 7
#: Only papers whose main topic is core to Foresea are read in full: prediction markets, LLM
#: forecasting, calibration and scoring, the fund's pairs, commodities and options.
MIN_READ_TOPIC = 4
#: Self-publishing sites and repositories: anything can be posted there, so search results from
#: them are left out (journals and arXiv, SSRN and the like are kept).
UNREVIEWED = re.compile(r"zenodo|figshare|researchgate|qeios|preprints\.org|authorea|scilit|hal open", re.I)
KEEP_TYPES = {"article", "preprint", "review", "book-chapter"}
READ_WINDOW_DAYS = 21
ABSTRACT_CHARS = 2000

FORECASTING = re.compile(
    r"\b(forecast\w*|nowcast\w*|prediction markets?|betting markets?|predictive distributions?|"
    r"probabilistic predictions?|scoring rules?|superforecast\w*|wisdom of (?:the )?crowds?|"
    r"early[- ]warning)\b", re.I)

#: (topic, weight for Foresea, pattern on title and abstract). Weights follow what Foresea does:
#: prediction-market forecasts and their blend, the fund's pairs and index put spreads first.
TOPICS: list[tuple[str, int, re.Pattern]] = [(t, w, re.compile(p, re.I)) for t, w, p in [
    ("prediction-markets", 5, r"\b(prediction markets?|betting markets?|event contracts?|polymarket|kalshi|"
                              r"predictit|metaculus|information markets?|forecast aggregation|forecasting "
                              r"tournaments?|superforecast\w*|wisdom of (?:the )?crowds?|good judgment)\b"),
    ("llm-forecasting", 4, r"\b(LLMs?|large language models?|language agents?|GPT-?\d*)\b"),
    ("calibration-scoring", 4, r"\b(scoring rules?|brier|log(?:arithmic)? scores?|(?:well-|mis)calibrat\w*|"
                               r"calibration (?:error|curve|of (?:probabilit|forecast|predict))\w*|reliability "
                               r"diagrams?|forecast (?:evaluation|combination)s?|ensemble forecasts?|"
                               r"probabilistic forecast\w*)\b"),
    ("pairs-relative-value", 5, r"\b(pairs? trading|relative value|cointegrat\w*|mean[- ]revers\w*|spread "
                                r"trading|statistical arbitrage)\b"),
    ("commodities", 4, r"\b(commodit\w*|crude oil|natural gas|copper|alumin(?:i)?um|zinc|nickel|gold|silver|"
                       r"metals? prices?|futures markets?)\b"),
    ("options-volatility", 4, r"\b(option pricing|implied volatility|volatility risk premium|variance risk "
                              r"premium|put options?|options? writing|VIX|volatility forecast\w*)\b"),
    ("india", 3, r"\b(India\w*|Nifty|NSE|MCX|BSE|Sensex|rupee)\b"),
    ("macro-rates", 3, r"\b(inflation|central banks?|interest rates?|monetary policy|GDP|nowcast\w*|"
                       r"recessions?|yield curves?|exchange rates?)\b"),
    ("energy-power", 3, r"\b(electricity|power prices?|load forecast\w*|energy prices?|renewables?|"
                        r"wind power|solar)\b"),
    ("weather-climate", 3, r"\b(weather|temperatures?|precipitation|rainfall|numerical weather|"
                           r"climate|hurricanes?|tropical cyclones?|heat ?waves?)\b"),
    ("elections-politics", 3, r"\b(elections?|electoral|polls?|polling|geopolitic\w*|armed conflicts?|wars?)\b"),
    ("judgment", 3, r"\b(judgment(?:al)?|expert elicitation|human forecasters?|overconfiden\w*|"
                    r"cognitive bias\w*)\b"),
    ("sports", 2, r"\b(sports?|football|soccer|basketball|cricket|tennis|match outcomes?)\b"),
    ("time-series-ml", 2, r"\b(time[- ]series|deep learning|neural networks?|transformers?|foundation "
                          r"models?|machine learning)\b"),
    ("quant-trading", 2, r"\b(portfolios?|asset pricing|momentum|return predictab\w*|market microstructure|"
                         r"order books?|high[- ]frequency|risk premi\w*|hedg\w*)\b"),
    ("epidemics", 1, r"\b(epidemi\w*|pandemics?|influenza|COVID\S*|outbreaks?)\b"),
]]

#: The part of Foresea each topic could improve, for the weekly digest.
USES: dict[str, str] = {
    "prediction-markets": "prediction-market forecasts and the Edge board",
    "llm-forecasting": "the forecasting models and their checker",
    "calibration-scoring": "the forecast blend and its calibration",
    "judgment": "the forecast blend and its checker",
    "pairs-relative-value": "the fund's commodity pairs",
    "commodities": "the fund's commodity pairs",
    "options-volatility": "the fund's index put spreads",
    "india": "the fund's Indian markets",
    "quant-trading": "fund sizing and risk",
    "macro-rates": "long-horizon questions (rates, inflation, growth)",
    "energy-power": "long-horizon questions (energy)",
    "weather-climate": "weather markets",
    "elections-politics": "election and geopolitics questions",
    "sports": "sports markets",
    "time-series-ml": "the forecasting models",
    "epidemics": "health questions",
}


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------

def topics(text: str) -> list[str]:
    return [name for name, _w, pattern in TOPICS if pattern.search(text)]


def score(card: Mapping[str, Any]) -> int:
    """Its main topic's weight, half the next one's, and points for an open PDF, a reviewed venue
    (a listed journal or arXiv) and a forecasting journal. Counting every keyword instead would
    put essays that name every topic first."""
    weights = {name: w for name, w, _p in TOPICS}
    w = sorted((weights[t] for t in card.get("topics") or []), reverse=True)
    s = (w[0] if w else 0) + (w[1] // 2 if len(w) > 1 else 0)
    s += 1 if card.get("pdf_url") else 0
    s += 1 if card.get("venue") in JOURNALS or card.get("source") == "arxiv" else 0
    s += 1 if card.get("venue") in FORECASTING_JOURNALS else 0
    return s


def main_weight(card: Mapping[str, Any]) -> int:
    weights = {name: w for name, w, _p in TOPICS}
    return max((weights[t] for t in card.get("topics") or []), default=0)


def make_card(**fields: Any) -> dict[str, Any]:
    text = f"{fields.get('title') or ''}. {fields.get('abstract') or ''}"
    card = {k: v for k, v in fields.items() if v not in (None, "", [])}
    card["abstract"] = (card.get("abstract") or "")[:ABSTRACT_CHARS]
    card["topics"] = topics(text)
    card["score"] = score(card)
    return card


# --------------------------------------------------------------------------------------------
# arXiv RSS
# --------------------------------------------------------------------------------------------

_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def parse_arxiv_rss(xml: str, feed: str, mode: str, found: str) -> list[dict[str, Any]]:
    """Cards from one arXiv RSS feed: new papers and cross-lists, not replacements."""
    root = ET.fromstring(xml)
    ns = {"dc": "http://purl.org/dc/elements/1.1/", "arxiv": "http://arxiv.org/schemas/atom"}
    out = []
    for item in root.iter("item"):
        kind = (item.findtext("arxiv:announce_type", "", ns) or "new").strip()
        if kind.startswith("replace"):
            continue
        link = (item.findtext("link") or "").strip()
        m = _ARXIV_ID.search(link) or _ARXIV_ID.search(item.findtext("guid") or "")
        if not m:
            continue
        aid = m.group(1)
        title = " ".join((item.findtext("title") or "").split())
        desc = item.findtext("description") or ""
        abstract = " ".join(desc.split("Abstract:", 1)[-1].split())
        if mode != "all" and not FORECASTING.search(f"{title} {abstract}"):
            continue
        authors = [a.strip() for a in (item.findtext("dc:creator", "", ns) or "").split(",") if a.strip()]
        out.append(make_card(
            id=f"arxiv:{aid}", source="arxiv", feed=feed, title=title, authors=authors[:8],
            abstract=abstract, venue="arXiv", categories=[c.text for c in item.findall("category") if c.text],
            url=f"https://arxiv.org/abs/{aid}", pdf_url=f"https://arxiv.org/pdf/{aid}",
            licence=(item.findtext("dc:rights", "", ns) or "").strip(), found=found))
    return out


# --------------------------------------------------------------------------------------------
# OpenAlex
# --------------------------------------------------------------------------------------------

OPENALEX_FIELDS = ("id,doi,display_name,publication_date,authorships,primary_location,best_oa_location,"
                   "abstract_inverted_index,type")


def abstract_text(index: Optional[Mapping[str, Sequence[int]]]) -> str:
    if not index:
        return ""
    words = sorted((pos, word) for word, positions in index.items() for pos in positions)
    return " ".join(word for _pos, word in words)


def _english(title: str) -> bool:
    letters = [ch for ch in title if ch.isalpha()]
    return bool(letters) and sum(ch.isascii() for ch in letters) / len(letters) >= 0.8


def openalex_card(work: Mapping[str, Any], found: str) -> Optional[dict[str, Any]]:
    """A card for an OpenAlex work, or None for one that cannot be ranked or read: no title, not in
    English, a dataset or other non-paper, a self-publishing site, or no abstract outside the
    listed journals."""
    doi = str(work.get("doi") or "").replace("https://doi.org/", "").lower()
    title = " ".join(str(work.get("display_name") or "").split())
    if not title or not _english(title) or (work.get("type") or "article") not in KEEP_TYPES:
        return None
    venue_name = str(((work.get("primary_location") or {}).get("source") or {}).get("display_name") or "")
    if UNREVIEWED.search(venue_name) or (not work.get("abstract_inverted_index") and venue_name not in JOURNALS):
        return None
    m = re.match(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", doi)
    cid = f"arxiv:{m.group(1)}" if m else (f"doi:{doi}" if doi else f"openalex:{str(work.get('id', '')).rsplit('/', 1)[-1]}")
    venue = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
    oa = work.get("best_oa_location") or {}
    authors = [((a.get("author") or {}).get("display_name") or "") for a in work.get("authorships") or []]
    return make_card(
        id=cid, source="openalex", title=title, authors=[a for a in authors if a][:8],
        abstract=abstract_text(work.get("abstract_inverted_index")), venue=venue,
        published=work.get("publication_date"), url=f"https://doi.org/{doi}" if doi else work.get("id"),
        pdf_url=oa.get("pdf_url"), licence=oa.get("license"), kind=work.get("type"), found=found)


def _openalex_pages(crawler: watch.Crawler, url: str, today: date) -> tuple[list[dict], Optional[str]]:
    """Every page of one OpenAlex query (cursor paging, at most OPENALEX_PAGES pages)."""
    cards, cursor = [], "*"
    for _ in range(OPENALEX_PAGES):
        resp = crawler.fetch(f"{url}&cursor={urllib.parse.quote(cursor)}")
        if resp.error or resp.status != 200:
            return cards, (resp.error or f"HTTP {resp.status}") if not cards else None
        data = json.loads(resp.text())
        cards += [c for w in data.get("results") or [] if (c := openalex_card(w, today.isoformat()))]
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor or not data.get("results"):
            break
    return cards, None


def openalex_urls(today: date) -> list[tuple[str, str]]:
    since = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    window = f"from_publication_date:{since},to_publication_date:{today.isoformat()}"
    ids = "|".join(JOURNALS.values())
    base = {"per-page": "200", "select": OPENALEX_FIELDS}
    urls = [("openalex:journals", OPENALEX + "?" + urllib.parse.urlencode(
        {**base, "filter": f"primary_location.source.id:{ids},{window}"}))]
    for i, q in enumerate(SEARCHES):
        urls.append((f"openalex:search{i + 1}", OPENALEX + "?" + urllib.parse.urlencode(
            {**base, "search": q, "filter": window})))
    return urls


# --------------------------------------------------------------------------------------------
# The base: cards, state, inbox, digest
# --------------------------------------------------------------------------------------------

def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def all_cards(root: Path) -> list[dict]:
    return [c for p in sorted((root / "papers" / "cards").glob("*.jsonl")) for c in _jsonl(p)]


def collect(root: Path, crawler: watch.Crawler, today: date) -> dict[str, Any]:
    """Fetch every feed, keep the papers the base has not seen, and append their cards."""
    seen = {c["id"] for c in all_cards(root)}
    found, report = [], {}
    sources: list[tuple[str, str, str]] = [(f"arxiv:{feed}", ARXIV_RSS.format(feed=feed), mode)
                                           for feed, mode in ARXIV_FEEDS.items()]
    sources += [(name, url, "openalex") for name, url in openalex_urls(today)]
    for name, url, mode in sources:
        try:
            if mode == "openalex":
                cards, why = _openalex_pages(crawler, url, today)
            else:
                resp = crawler.fetch(url)
                why = resp.error or (f"HTTP {resp.status}" if resp.status != 200 else None)
                cards = [] if why else parse_arxiv_rss(resp.text(), name.split(":", 1)[1], mode, today.isoformat())
        except (ValueError, ET.ParseError) as exc:
            cards, why = [], f"unreadable ({exc.__class__.__name__})"
        if why:
            report[name] = {"status": "failed", "why": why}
            continue
        new = [c for c in cards if c["id"] not in seen]
        seen.update(c["id"] for c in new)
        found += new
        report[name] = {"status": "ok", "papers": len(cards), "new": len(new)}
    if found:
        path = root / "papers" / "cards" / f"{today:%Y-%m}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for card in found:
                fh.write(json.dumps(card, ensure_ascii=False, sort_keys=True) + "\n")
    return {"found": found, "feeds": report}


def queued_ids(root: Path) -> set[str]:
    return {str(r.get("edition")) for r in _jsonl(root / "inbox.jsonl") if r.get("source") == "papers"}


def pick_to_read(cards: Sequence[Mapping[str, Any]], queued: Iterable[str], today: date, *,
                 per_week: int = PER_WEEK, min_score: int = MIN_READ_SCORE) -> list[Mapping[str, Any]]:
    """The best recent unread cards with an open PDF, at most one per first author."""
    since = (today - timedelta(days=READ_WINDOW_DAYS)).isoformat()
    queued = set(queued)
    pool = sorted((c for c in cards if c.get("pdf_url") and c.get("found", "") >= since
                   and c["id"] not in queued and int(c.get("score") or 0) >= min_score
                   and main_weight(c) >= MIN_READ_TOPIC),
                  key=lambda c: (-int(c["score"]), c["id"]))
    out, authors = [], set()
    for c in pool:
        lead = (c.get("authors") or [""])[0]
        if lead and lead in authors:
            continue
        authors.add(lead)
        out.append(c)
        if len(out) == per_week:
            break
    return out


def inbox_row(card: Mapping[str, Any], today: date) -> dict[str, Any]:
    return {"found": today.isoformat(), "source": "papers", "publisher": card.get("venue") or "arXiv",
            "title": card["title"], "kind": "report", "edition": card["id"], "url": card.get("url"),
            "published": card.get("published") or card.get("found"),
            "files": [{"url": card["pdf_url"], "content_type": "application/pdf"}], "status": "new",
            "paper": {"score": card.get("score"), "topics": card.get("topics"), "licence": card.get("licence"),
                      "authors": (card.get("authors") or [])[:3]}}


def digest(cards: Sequence[Mapping[str, Any]], read: Sequence[Mapping[str, Any]], today: date, *,
           top: int = 15) -> str:
    """The week's best papers by the part of Foresea they could improve (Markdown)."""
    since = (today - timedelta(days=7)).isoformat()
    week = [c for c in cards if c.get("found", "") >= since]
    lines = [f"# Papers: week to {today.isoformat()}", "",
             f"{len(week)} new papers ({sum(c.get('source') == 'arxiv' for c in week)} arXiv, "
             f"{sum(c.get('source') == 'openalex' for c in week)} journals and other indexes via OpenAlex).", ""]
    counts: dict[str, int] = {}
    for c in week:
        for t in c.get("topics") or []:
            counts[t] = counts.get(t, 0) + 1
    if counts:
        lines += ["**By topic:** " + ", ".join(f"{t} {n}" for t, n in sorted(counts.items(), key=lambda x: -x[1])), ""]
    lines += ["## Best for Foresea", ""]
    for c in sorted(week, key=lambda c: (-int(c.get("score") or 0), c["id"]))[:top]:
        uses = sorted({USES[t] for t in c.get("topics") or [] if t in USES})
        who = (c.get("authors") or ["?"])[0] + (" et al." if len(c.get("authors") or []) > 1 else "")
        lines.append(f"- **{c['title']}** ({who}, {c.get('venue') or 'arXiv'}, score {c.get('score')}): "
                     f"could inform {', '.join(uses) or 'general forecasting'}. {c.get('url', '')}")
    lines += ["", "## Queued for the night reader", ""]
    lines += [f"- {c['title']} ({c['id']})" for c in read] or ["- nothing this week"]
    return "\n".join(lines) + "\n"


def weekly(root: Path, today: date, *, per_week: int = PER_WEEK) -> dict[str, Any]:
    """Queue the week's best papers for reading and write the digest."""
    cards = all_cards(root)
    read = pick_to_read(cards, queued_ids(root), today, per_week=per_week)
    if read:
        with (root / "inbox.jsonl").open("a", encoding="utf-8") as fh:
            for c in read:
                fh.write(json.dumps(inbox_row(c, today), ensure_ascii=False) + "\n")
    text = digest(cards, read, today)
    folder = root / "papers" / "digest"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "latest.md").write_text(text, encoding="utf-8")
    year, week, _ = today.isocalendar()
    (folder / f"{year}-W{week:02d}.md").write_text(text, encoding="utf-8")
    return {"queued": [c["id"] for c in read]}
