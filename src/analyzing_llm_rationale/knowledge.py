"""Foresea's knowledge base: dated claims, projections and a knowledge graph built from reports.

The data lives outside this public repo, in a private GitHub repo (pareelamre/foresea-knowledge)
cloned at ``KNOWLEDGE_DIR``, because reports are often licensed and must not be republished.
This module only reads it. Layout (owner, 2026-11-17):

    entities.jsonl                  {"id", "name", "type", "aliases": [...]}
    sources/<id>/source.json        {"id", "title", "publisher", "published", "summary", ...}
    sources/<id>/claims.jsonl       one atomic claim per line, in our own words, with its page
    sources/<id>/relations.jsonl    {"subject", "predicate", "object", "claims": [...]}
    sources/<id>/chunks.jsonl       page-tagged passages of the source (stays private)

A claim is the unit Foresea reasons with: a dated fact, estimate, policy or projection
(``kind``), the entities it is about, and for numbers its ``metric``, ``value``, ``unit``,
``geo`` and ``horizon`` (the year it refers to). Projections are what make the base useful for
long-horizon questions, and ``timeline`` lines up every source's projection for the same
metric so a newer report's revision is visible next to the one it replaced.

Retrieval fuses up to four ranked lists with reciprocal rank fusion: BM25 over claims (SQLite
FTS5), dense similarity (MiniLM, when an embedder is available), the knowledge graph (claims
about the entities a question names, then their neighbours), and the horizon (projections for
the years a question asks about). Every query can be pinned to an ``as_of`` date, so a backtest
never sees a report published after its forecast.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import tarfile
import tempfile
import threading
import time
from array import array
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

KINDS = ("projection", "estimate", "fact", "policy", "finding", "risk")
#: Kinds that carry a number for a future (or measured) year.
NUMERIC_KINDS = ("projection", "estimate")
RRF_K = 60
#: Each ranked list contributes at most this many candidates to the fusion.
LIST_DEPTH = 50
#: A timeline keeps rows scoring at least this share of its best row.
TIMELINE_FLOOR = 0.75
YEAR = re.compile(r"\b(19[5-9]\d|20\d\d|21\d\d)\b")
DATE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")
WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOP = frozenset("""a an and are as at be been by can could do does for from had has have how if in into
is it its may might more most no not of on or our over per should so than that the their them then there
these they this those through to under up was we were what when where which while who will with would
you your yes""".split())
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

Embedder = Callable[[list[str]], list[list[float]]]
Reranker = Callable[[str, list[str]], Optional[list[float]]]


# --------------------------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------------------------

@dataclass
class Corpus:
    root: Path
    sources: dict[str, dict] = field(default_factory=dict)
    claims: list[dict] = field(default_factory=list)
    entities: dict[str, dict] = field(default_factory=dict)
    relations: list[dict] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def default_dir(*, allow_sibling: bool = False) -> Optional[Path]:
    """``KNOWLEDGE_DIR``, or (for local tools only) a ``foresea-knowledge`` clone beside the repo."""
    raw = os.environ.get("KNOWLEDGE_DIR", "").strip()
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_dir() else None
    if allow_sibling:
        sibling = Path(__file__).resolve().parents[3] / "foresea-knowledge"
        if sibling.is_dir():
            return sibling
    return None


def _jsonl(path: Path, errors: list[str]) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{n}: not JSON ({exc.msg})")
            continue
        if not isinstance(row, dict):
            errors.append(f"{path.name}:{n}: not an object")
            continue
        rows.append(row)
    return rows


def parse_when(value: Any) -> Optional[date]:
    """A YYYY, YYYY-MM or YYYY-MM-DD string as the first day it covers."""
    text = str(value or "").strip()
    if not DATE.match(text):
        return None
    parts = [int(p) for p in text.split("-")]
    try:
        return date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1)
    except ValueError:
        return None


def load(root: Path) -> Corpus:
    """Read a knowledge directory and record every problem in ``errors`` instead of raising."""
    root = Path(root)
    corpus = Corpus(root=root)
    errors = corpus.errors
    for row in _jsonl(root / "entities.jsonl", errors):
        eid = str(row.get("id") or "")
        if not eid or not row.get("name"):
            errors.append(f"entities.jsonl: entity without id or name: {row}")
        elif eid in corpus.entities:
            errors.append(f"entities.jsonl: duplicate entity {eid}")
        else:
            row["aliases"] = [str(a) for a in row.get("aliases") or []]
            corpus.entities[eid] = row
    for folder in sorted(p for p in (root / "sources").glob("*") if p.is_dir()):
        sid = folder.name
        try:
            source = json.loads((folder / "source.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"sources/{sid}: unreadable source.json ({exc})")
            continue
        if source.get("id") != sid:
            errors.append(f"sources/{sid}: source.json id {source.get('id')!r} does not match its folder")
        for key in ("title", "publisher"):
            if not source.get(key):
                errors.append(f"sources/{sid}: source.json has no {key}")
        if parse_when(source.get("published")) is None:
            errors.append(f"sources/{sid}: published must be YYYY-MM-DD, got {source.get('published')!r}")
        source["id"] = sid
        corpus.sources[sid] = source
        for claim in _jsonl(folder / "claims.jsonl", errors):
            claim.setdefault("source", sid)
            corpus.claims.append(claim)
        for rel in _jsonl(folder / "relations.jsonl", errors):
            rel.setdefault("source", sid)
            corpus.relations.append(rel)
        for chunk in _jsonl(folder / "chunks.jsonl", errors):
            chunk.setdefault("source", sid)
            corpus.chunks.append(chunk)
    _validate(corpus)
    return corpus


def _validate(corpus: Corpus) -> None:
    errors, seen = corpus.errors, set()
    for claim in corpus.claims:
        cid = str(claim.get("id") or "")
        where = f"claim {cid or '?'} ({claim.get('source')})"
        if not cid:
            errors.append(f"{where}: no id")
        elif cid in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(cid)
        if claim.get("kind") not in KINDS:
            errors.append(f"{where}: kind must be one of {', '.join(KINDS)}")
        if not str(claim.get("text") or "").strip():
            errors.append(f"{where}: no text")
        for eid in claim.get("entities") or []:
            if eid not in corpus.entities:
                errors.append(f"{where}: unknown entity {eid}")
        if claim.get("horizon") not in (None, "") and parse_when(claim["horizon"]) is None:
            errors.append(f"{where}: horizon must be YYYY, YYYY-MM or YYYY-MM-DD")
        if claim.get("page") is not None and not isinstance(claim["page"], int):
            errors.append(f"{where}: page must be an integer")
    for rel in corpus.relations:
        for end in ("subject", "object"):
            if rel.get(end) not in corpus.entities:
                errors.append(f"relation {rel.get('subject')} {rel.get('predicate')} {rel.get('object')}: "
                              f"unknown {end} {rel.get(end)!r}")
        if not rel.get("predicate"):
            errors.append(f"relation {rel.get('subject')} -> {rel.get('object')}: no predicate")
        for cid in rel.get("claims") or []:
            if cid not in seen:
                errors.append(f"relation {rel.get('subject')} {rel.get('predicate')} {rel.get('object')}: "
                              f"unknown claim {cid}")
    chunk_ids = set()
    for chunk in corpus.chunks:
        if chunk.get("id") in chunk_ids:
            errors.append(f"chunk {chunk.get('id')}: duplicate id")
        chunk_ids.add(chunk.get("id"))


# --------------------------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------------------------

def normalize(text: str) -> str:
    return " " + " ".join(WORD.findall(str(text).lower())) + " "


def terms(text: str) -> list[str]:
    out, seen = [], set()
    for word in WORD.findall(str(text).lower()):
        if len(word) > 1 and word not in STOP and word not in seen:
            seen.add(word)
            out.append(word)
    return out


def years(text: str) -> list[int]:
    return sorted({int(y) for y in YEAR.findall(str(text))})


def _fts_query(text: str) -> str:
    return " OR ".join(f'"{t}"' for t in terms(text)[:32])


def _unit(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else []


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def month_label(published: Any) -> str:
    when = parse_when(published)
    return f"{MONTHS[when.month - 1]} {when.year}" if when else str(published or "")


# --------------------------------------------------------------------------------------------
# Index and retrieval
# --------------------------------------------------------------------------------------------

@dataclass
class Hit:
    claim: dict
    source: dict
    score: float
    lists: list[str]
    #: A newer source projects the same metric, place and year; this one is history.
    superseded: bool = False

    @property
    def support(self) -> int:
        return len(self.lists)

    def cite(self) -> str:
        page = f", p. {self.claim['page']}" if self.claim.get("page") else ""
        return f"{self.source.get('publisher')}, {month_label(self.source.get('published'))}{page}"


class Index:
    """An in-memory SQLite index over a Corpus. Building one takes milliseconds per thousand claims."""

    def __init__(self, corpus: Corpus, *, embed: Optional[Embedder] = None,
                 vector_cache: Optional[Path] = None) -> None:
        self.corpus = corpus
        self.embed = embed
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.claims = list(corpus.claims)
        self.by_id = {c["id"]: i for i, c in enumerate(self.claims) if c.get("id")}
        self._published = [str(corpus.sources.get(c.get("source"), {}).get("published") or "") for c in self.claims]
        self._keys = [_projection_key(c) for c in self.claims]
        self._by_key: dict[tuple, list[int]] = {}
        for i, key in enumerate(self._keys):
            if key is not None:
                self._by_key.setdefault(key, []).append(i)
        self._aliases = self._alias_table()
        self._entity_claims: dict[str, list[int]] = {}
        for i, claim in enumerate(self.claims):
            for eid in claim.get("entities") or []:
                self._entity_claims.setdefault(eid, []).append(i)
        self._neighbours: dict[str, set[str]] = {}
        for rel in corpus.relations:
            a, b = rel.get("subject"), rel.get("object")
            if a and b:
                self._neighbours.setdefault(a, set()).add(b)
                self._neighbours.setdefault(b, set()).add(a)
            for cid in rel.get("claims") or []:
                i = self.by_id.get(cid)
                if i is None:
                    continue
                for eid in (a, b):
                    if eid and i not in self._entity_claims.setdefault(eid, []):
                        self._entity_claims[eid].append(i)
        self._build_fts()
        self._claim_vecs: list[list[float]] = []
        self._chunk_vecs: list[list[float]] = []
        if embed is not None:
            cache = _VectorCache(vector_cache)
            self._claim_vecs = cache.embed(embed, [self._claim_text(c) for c in self.claims])
            self._chunk_vecs = cache.embed(embed, [str(c.get("text") or "") for c in corpus.chunks])
            cache.close()

    # -- building ------------------------------------------------------------------------------

    def _alias_table(self) -> list[tuple[str, str]]:
        table = []
        for eid, ent in self.corpus.entities.items():
            for name in [ent.get("name"), *ent.get("aliases", [])]:
                norm = normalize(name or "").strip()
                if norm:
                    table.append((f" {norm} ", eid))
        return sorted(set(table), key=lambda t: -len(t[0]))

    def _claim_text(self, claim: dict) -> str:
        names = [self.corpus.entities.get(e, {}).get("name", e) for e in claim.get("entities") or []]
        extra = [claim.get("metric"), claim.get("geo"), claim.get("horizon")]
        return " | ".join(str(x) for x in [claim.get("text"), *extra, *names] if x)

    def _build_fts(self) -> None:
        db = self.db
        db.execute("CREATE VIRTUAL TABLE claims_fts USING fts5(body, context, tokenize='porter unicode61')")
        db.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(body, section, tokenize='porter unicode61')")
        rows = []
        for i, claim in enumerate(self.claims):
            source = self.corpus.sources.get(claim.get("source"), {})
            names = []
            for eid in claim.get("entities") or []:
                ent = self.corpus.entities.get(eid, {})
                names += [ent.get("name") or eid, *ent.get("aliases", [])]
            # The source title stays out: it would match every claim in a report at once.
            context = " ".join(str(x) for x in [claim.get("metric"), claim.get("unit"), claim.get("geo"),
                                                claim.get("horizon"), claim.get("section"), claim.get("scenario"),
                                                source.get("publisher"), *names] if x)
            rows.append((i + 1, claim.get("text") or "", context))
        db.executemany("INSERT INTO claims_fts(rowid, body, context) VALUES (?, ?, ?)", rows)
        db.executemany("INSERT INTO chunks_fts(rowid, body, section) VALUES (?, ?, ?)",
                       [(i + 1, c.get("text") or "", c.get("section") or "") for i, c in enumerate(self.corpus.chunks)])

    # -- pieces of a query --------------------------------------------------------------------

    def link(self, text: str) -> list[str]:
        """Entities a text names, longest alias first, without double-counting nested names."""
        padded, found = normalize(text), []
        for alias, eid in self._aliases:
            if alias in padded:
                if eid not in found:
                    found.append(eid)
                padded = padded.replace(alias, " ")
        return found

    def _visible(self, i: int, as_of: Optional[date], kinds: Optional[Sequence[str]]) -> bool:
        claim = self.claims[i]
        if kinds and claim.get("kind") not in kinds:
            return False
        if as_of is not None:
            published = parse_when(self.corpus.sources.get(claim.get("source"), {}).get("published"))
            if published is None or published > as_of:
                return False
        return True

    def _lexical(self, question: str, table: str = "claims_fts") -> list[int]:
        query = _fts_query(question)
        if not query:
            return []
        weights = "4.0, 1.0"
        sql = f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY bm25({table}, {weights}) LIMIT 200"
        try:
            return [r[0] - 1 for r in self.db.execute(sql, (query,))]
        except sqlite3.OperationalError:
            return []

    def _dense(self, question: str, vecs: list[list[float]], floor: float = 0.2) -> list[int]:
        if not vecs or self.embed is None:
            return []
        try:
            q = _unit(self.embed([question])[0])
        except Exception:
            return []
        if not q:
            return []
        scored = [(_dot(q, v), i) for i, v in enumerate(vecs) if v]
        return [i for s, i in sorted(scored, reverse=True) if s >= floor][:200]

    def _graph(self, entities: list[str]) -> list[int]:
        """Claims about the named entities (more of them first), then about their neighbours."""
        direct: dict[int, int] = {}
        for eid in entities:
            for i in self._entity_claims.get(eid, []):
                direct[i] = direct.get(i, 0) + 1
        ranked = [i for i, _ in sorted(direct.items(), key=lambda kv: (-kv[1], _desc(self._published[kv[0]]), kv[0]))]
        near = []
        for eid in entities:
            for nb in sorted(self._neighbours.get(eid, ())):
                if nb in entities:
                    continue
                fresh = [i for i in self._entity_claims.get(nb, []) if i not in direct and i not in near]
                near += sorted(fresh, key=lambda i: (_desc(self._published[i]), i))
        return ranked + near

    def _history(self, i: int, as_of: Optional[date]) -> list[int]:
        """Older visible claims projecting the same metric, place and year as claim ``i``, newest first."""
        key = self._keys[i]
        if key is None:
            return []
        older = [j for j in self._by_key.get(key, [])
                 if j != i and self._published[j] < self._published[i] and self._visible(j, as_of, None)]
        return sorted(older, key=lambda j: _desc(self._published[j]))

    def _horizon(self, asked: list[int]) -> list[int]:
        """Claims about the years a question asks about: projections first, then risks and policies."""
        if not asked:
            return []
        scored = []
        for i, claim in enumerate(self.claims):
            when = parse_when(claim.get("horizon"))
            if when is None:
                continue
            gap = min(abs(when.year - y) for y in asked)
            if gap <= 2:
                kind_rank = 0 if claim.get("kind") in NUMERIC_KINDS else 1
                scored.append((gap, kind_rank, self._published[i], i))
        scored.sort(key=lambda t: (t[0], t[1], _desc(t[2])))
        return [i for *_, i in scored]

    # -- public queries -----------------------------------------------------------------------

    def search(self, question: str, *, as_of: Optional[date] = None, k: int = 8,
               kinds: Optional[Sequence[str]] = None, rerank: Optional[Reranker] = None) -> list[Hit]:
        """Claims for a question, best first, fused across lexical, dense, graph and horizon lists."""
        entities = self.link(question)
        lists = {
            "lexical": self._lexical(question),
            "dense": self._dense(question, self._claim_vecs),
            "graph": self._graph(entities),
            "horizon": self._horizon(years(question)),
        }
        scores: dict[int, float] = {}
        found: dict[int, list[str]] = {}
        for name, ranked in lists.items():
            kept = [i for i in ranked if self._visible(i, as_of, kinds)][:LIST_DEPTH]
            for rank, i in enumerate(kept, 1):
                scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank)
                found.setdefault(i, []).append(name)
        # Knowledge moves on: when a newer visible source projects the same thing, the older
        # projection drops back (it still shows as history in timeline and evidence).
        superseded = {i for i in scores if self._keys[i] is not None and any(
            self._published[j] > self._published[i] and self._visible(j, as_of, None)
            for j in self._by_key[self._keys[i]])}
        for i in superseded:
            scores[i] *= 0.5
        order = sorted(scores, key=lambda i: (-scores[i], _desc(self._published[i]), i))
        if rerank is not None and order:
            head = order[: max(k * 2, 10)]
            try:
                rescored = rerank(question, [self.claims[i]["text"] for i in head])
            except Exception:
                rescored = None
            if rescored and len(rescored) == len(head):
                head = [i for _, i in sorted(zip(rescored, head), key=lambda t: -t[0])]
                order = head + order[len(head):]
        return [Hit(self.claims[i], self.corpus.sources.get(self.claims[i].get("source"), {}), scores[i], found[i],
                    superseded=i in superseded) for i in order[:k]]

    def passages(self, question: str, *, as_of: Optional[date] = None, k: int = 5) -> list[dict]:
        """Source passages for a question (lexical + dense), for private research only."""
        chunks = self.corpus.chunks
        lists = [self._lexical(question, "chunks_fts"), self._dense(question, self._chunk_vecs)]
        scores: dict[int, float] = {}
        for ranked in lists:
            kept = []
            for i in ranked:
                published = parse_when(self.corpus.sources.get(chunks[i].get("source"), {}).get("published"))
                if as_of is None or (published is not None and published <= as_of):
                    kept.append(i)
            for rank, i in enumerate(kept[:LIST_DEPTH], 1):
                scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank)
        return [{**chunks[i], "score": scores[i]} for i in sorted(scores, key=lambda i: -scores[i])[:k]]

    def timeline(self, topic: str, *, as_of: Optional[date] = None) -> list[dict]:
        """Every projection or estimate on a topic, grouped by what it measures and for which year.

        Within a group the newest source comes first and the older ones are its ``history``, so a
        revision (a 2035 estimate raised or cut by a later report) reads as one line.
        """
        # Scenario numbers (a "risk" with a value) count too; a claim reached only because it
        # shares a year with the topic does not.
        hits = self.search(topic, as_of=as_of, k=200)
        groups: dict[tuple, list[Hit]] = {}
        for hit in hits:
            c = hit.claim
            if c.get("value") is None or not c.get("metric") or not c.get("horizon") or hit.lists == ["horizon"]:
                continue
            key = (str(c.get("metric")).lower(), str(c.get("geo") or "").lower(), str(c.get("horizon") or ""),
                   str(c.get("unit") or "").lower(), str(c.get("scenario") or "").lower())
            groups.setdefault(key, []).append(hit)
        out = []
        for group in groups.values():
            group.sort(key=lambda h: _desc(str(h.source.get("published") or "")))
            latest = group[0]
            out.append({"metric": latest.claim.get("metric"), "geo": latest.claim.get("geo"),
                        "horizon": latest.claim.get("horizon"), "unit": latest.claim.get("unit"),
                        "scenario": latest.claim.get("scenario"), "value": latest.claim.get("value"),
                        "claim": latest.claim, "cite": latest.cite(), "score": max(h.score for h in group),
                        "history": [{"value": h.claim.get("value"), "cite": h.cite(), "claim": h.claim}
                                    for h in group[1:]]})
        out.sort(key=lambda g: (-g["score"], str(g["horizon"])))
        # Keep what is about the topic: rows well below the best match only share a word or a year.
        best = out[0]["score"] if out else 0.0
        return [g for g in out if g["score"] >= TIMELINE_FLOOR * best]

    def evidence_articles(self, question: str, *, as_of: Optional[date] = None, k: int = 2,
                          per_source: int = 4, min_support: int = 2) -> list[dict]:
        """The base's best claims for a forecast, one article-shaped dict per source.

        Only claims that at least ``min_support`` retrievers agree on are used, so a question that
        merely shares a word with a report gets nothing. Claims are our own words with the page
        they come from; source passages never leave the base.
        """
        hits = [h for h in self.search(question, as_of=as_of, k=24) if h.support >= min_support and not h.superseded]
        by_source: dict[str, list[Hit]] = {}
        for hit in hits:
            by_source.setdefault(hit.claim.get("source"), []).append(hit)
        articles = []
        for group in list(by_source.values())[:k]:
            source = group[0].source
            lines = []
            for hit in group[:per_source]:
                page = f" (p. {hit.claim['page']})" if hit.claim.get("page") else ""
                earlier = ""
                history = self._history(self.by_id[hit.claim["id"]], as_of)
                if history:
                    old = Hit(self.claims[history[0]], self.corpus.sources.get(self.claims[history[0]].get("source"), {}),
                              0.0, [])
                    unit = f" {old.claim['unit']}" if old.claim.get("unit") else ""
                    earlier = f" Earlier: {old.claim.get('value')}{unit} ({old.cite()})."
                lines.append(f"- {hit.claim['text']}{page}{earlier}")
            body = "\n".join(lines)
            articles.append({
                "title": f"{source.get('title')} ({source.get('publisher')}, {month_label(source.get('published'))})",
                "summary": body, "text": body, "source": source.get("publisher") or "Knowledge base",
                "url": source.get("url") or None, "publish_date": source.get("published"),
                "source_channel": "knowledge_base",
                # Share of the four retrievers that agree on the lead claim, not a cosine.
                "relevance_score": round(min(1.0, group[0].support / 4), 2),
            })
        return articles

    def brief(self, question: str, *, as_of: Optional[date] = None, k: int = 4,
              min_support: int = 2) -> list[dict]:
        """A compact list of the base's best current claims for an agent's research packet."""
        out = []
        for hit in self.search(question, as_of=as_of, k=24):
            if hit.support < min_support or hit.superseded:
                continue
            row = {"claim": hit.claim["text"], "kind": hit.claim.get("kind"), "source": hit.cite(),
                   "published": hit.source.get("published")}
            if hit.claim.get("horizon"):
                row["horizon"] = hit.claim["horizon"]
            if hit.source.get("url"):
                row["url"] = hit.source["url"]
            out.append(row)
            if len(out) == k:
                break
        return out

    def figures(self, metric: str, *, as_of: Optional[date] = None, geo: Optional[str] = None) -> list[dict]:
        """Every visible claim stating a value for ``metric`` (case-insensitive), with its source."""
        want, place = metric.strip().lower(), (geo or "").strip().lower()
        out = []
        for i, claim in enumerate(self.claims):
            if str(claim.get("metric") or "").strip().lower() != want or claim.get("value") is None:
                continue
            if place and str(claim.get("geo") or "").strip().lower() != place:
                continue
            if not self._visible(i, as_of, None):
                continue
            out.append({"claim": claim, "source": self.corpus.sources.get(claim.get("source"), {}),
                        "published": self._published[i]})
        return out


def brief(question: str, *, as_of: Optional[date] = None, k: int = 4) -> list[dict]:
    """``Index.brief`` over the shared base; empty when no base is configured or anything fails."""
    try:
        index = shared_index()
        return index.brief(question, as_of=as_of or date.today(), k=k) if index is not None else []
    except Exception:  # aqg: top-level boundary -- the base is optional context
        return []


def _projection_key(claim: dict) -> Optional[tuple]:
    """What a numeric claim measures, where and for when: two claims with one key are rival estimates."""
    if claim.get("kind") not in NUMERIC_KINDS or claim.get("value") is None or not claim.get("metric") \
            or not claim.get("horizon"):
        return None
    return tuple(str(claim.get(f) or "").strip().lower() for f in ("metric", "geo", "horizon", "unit", "scenario"))


def _desc(text: str) -> str:
    """Sort key that orders ISO date strings newest first."""
    return "".join(chr(0x10FFFF - ord(ch)) for ch in text)


class _VectorCache:
    """Embeddings keyed by a hash of the text, so a rebuild only embeds what is new."""

    def __init__(self, path: Optional[Path]) -> None:
        self.db = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(path))
            self.db.execute("CREATE TABLE IF NOT EXISTS vec (key TEXT PRIMARY KEY, data BLOB)")

    def embed(self, embed: Embedder, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        out: list[Optional[list[float]]] = [None] * len(texts)
        if self.db is not None:
            for n, key in enumerate(keys):
                row = self.db.execute("SELECT data FROM vec WHERE key = ?", (key,)).fetchone()
                if row:
                    out[n] = list(array("f", row[0]))
        todo = [n for n, v in enumerate(out) if v is None and texts[n].strip()]
        for start in range(0, len(todo), 64):
            batch = todo[start:start + 64]
            vecs = embed([texts[n] for n in batch])
            for n, vec in zip(batch, vecs):
                unit = _unit([float(x) for x in vec]) if vec else []
                out[n] = unit
                if self.db is not None and unit:
                    self.db.execute("INSERT OR REPLACE INTO vec VALUES (?, ?)", (keys[n], array("f", unit).tobytes()))
        if self.db is not None:
            self.db.commit()
        return [v or [] for v in out]

    def close(self) -> None:
        if self.db is not None:
            self.db.close()


def default_embedder() -> Optional[Embedder]:
    """MiniLM: the app's sentence-transformer when installed, else the same model via transformers."""
    try:
        from . import rag
        if rag.embeddings_available():
            return lambda texts: rag.embed(list(texts))
    except Exception:
        pass
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception:
        return None
    name = os.environ.get("KNOWLEDGE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        model = AutoModel.from_pretrained(name, local_files_only=True)
    except Exception:
        try:
            tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name)
        except Exception:
            return None
    model.eval()

    def run(texts: list[str]) -> list[list[float]]:
        with torch.no_grad():
            batch = tok(list(texts), padding=True, truncation=True, max_length=256, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.tolist()
    return run


_shared: dict[str, Any] = {}
_refresh = threading.Lock()
logger = logging.getLogger(__name__)

# The live server's copy of the base: a private Cloudflare R2 object written by knowledge-sync.yml.
REMOTE_KEY = "knowledge/knowledge.tar.gz"
_REMOTE_MAX_BYTES = 64 * 1024 * 1024


def _restat_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("KNOWLEDGE_RESTAT_S", "300")))
    except ValueError:
        return 300.0


def remote_config() -> Optional[dict[str, str]]:
    """``KNOWLEDGE_R2_*`` settings for the private R2 copy, or None unless all four are set."""
    cfg = {name: os.environ.get(f"KNOWLEDGE_R2_{name.upper()}", "").strip()
           for name in ("endpoint_url", "bucket", "access_key_id", "secret_access_key")}
    if not all(cfg.values()):
        return None
    cfg["key"] = os.environ.get("KNOWLEDGE_R2_KEY", "").strip() or REMOTE_KEY
    return cfg


def _remote_client(cfg: dict[str, str]):
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=cfg["endpoint_url"], region_name="auto",
                        aws_access_key_id=cfg["access_key_id"], aws_secret_access_key=cfg["secret_access_key"],
                        config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 2}))


def unpack(data: bytes, dest: Path) -> None:
    """Extract a knowledge archive into ``dest``, refusing links, absolute paths and ``..``."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"unsafe member in knowledge archive: {member.name!r}")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, members=members, filter="data")
        else:  # pragma: no cover - Python without extraction filters
            tar.extractall(dest, members=members)


def fetch_remote(cfg: dict[str, str], *, client=None) -> Path:
    """The local copy of the R2 archive, downloaded again only when its ETag changes."""
    client = client or _remote_client(cfg)
    head = client.head_object(Bucket=cfg["bucket"], Key=cfg["key"])
    etag = str(head.get("ETag") or "")
    current = _shared.get("remote_dir")
    if current and etag and _shared.get("remote_etag") == etag and Path(current).is_dir():
        return Path(current)
    if int(head.get("ContentLength") or 0) > _REMOTE_MAX_BYTES:
        raise ValueError("knowledge archive is larger than expected")
    body = client.get_object(Bucket=cfg["bucket"], Key=cfg["key"], IfMatch=etag)["Body"].read(_REMOTE_MAX_BYTES + 1)
    if len(body) > _REMOTE_MAX_BYTES:
        raise ValueError("knowledge archive is larger than expected")
    cache = Path(os.environ.get("KNOWLEDGE_CACHE_DIR") or Path(tempfile.gettempdir()) / "foresea-knowledge")
    cache.mkdir(parents=True, exist_ok=True)
    dest = Path(tempfile.mkdtemp(prefix="base-", dir=cache))
    try:
        unpack(body, dest)
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    _shared.update(remote_dir=str(dest), remote_etag=etag)
    if current and current != str(dest):
        shutil.rmtree(current, ignore_errors=True)
    return dest


def shared_index() -> Optional[Index]:
    """The server's index (lexical, graph and horizon only, no cold start).

    The base is ``KNOWLEDGE_DIR`` when that folder exists, else the private R2 copy named by
    ``KNOWLEDGE_R2_*`` (the live server). Rebuilt when a file is added, removed or changed;
    None when no base is configured or it has errors. The source is looked at again at most
    every ``KNOWLEDGE_RESTAT_S`` seconds (default 300) rather than on every request. If the R2
    copy cannot be fetched, or a new one has errors, the last good index keeps serving.
    """
    root = default_dir()
    remote = remote_config() if root is None else None
    if root is None and remote is None:
        return None
    source = str(root) if root is not None else f"r2://{remote['bucket']}/{remote['key']}"
    now = time.monotonic()
    same = _shared.get("source") == source
    if same and now - _shared.get("checked", float("-inf")) < _restat_seconds():
        return _shared.get("index")
    if not _refresh.acquire(blocking=False):  # another request is refreshing: answer from what we have
        return _shared.get("index") if same else None
    try:
        previous = _shared.get("index") if same else None
        _shared.update(source=source, checked=now, index=previous)
        if remote is not None:
            try:
                root = fetch_remote(remote)
            except Exception:  # the base is optional context; keep serving the last good copy
                logger.warning("knowledge base: could not refresh the R2 copy", exc_info=True)
                return previous
        files = [p.stat().st_mtime for p in root.rglob("*.json*")]
        key = (str(root), max(files, default=0.0), len(files))
        if _shared.get("key") == key:
            return previous
        corpus = load(root)
        if corpus.errors and remote is not None and previous is not None:
            logger.warning("knowledge base: the new R2 copy has %d problems; keeping the last good one",
                           len(corpus.errors))
            return previous
        index = None if corpus.errors else Index(corpus)
        _shared.update(index=index, key=key)
        return index
    finally:
        _refresh.release()


# --------------------------------------------------------------------------------------------
# Ingest: pages -> chunks -> claims
# --------------------------------------------------------------------------------------------

HEADING = re.compile(r"^(Section \d+\.\s+\S.*|\d+(?:\.\d+)+\.?\s+[A-Z].{2,120})$")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def slug(text: str) -> str:
    return "_".join(WORD.findall(str(text).lower().replace(".", " ")))[:60] or "entity"


def pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader
    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


def split_pages(text: str) -> list[str]:
    """Pages of a plain-text export: form feeds separate them."""
    return text.split("\f")


def chunk_pages(source_id: str, pages: Sequence[str], *, size: int = 1400, overlap: int = 200) -> list[dict]:
    """Page-tagged passages that never cross a page, carrying the section heading they sit under."""
    chunks: list[dict] = []
    section = ""
    for page_no, page in enumerate(pages, 1):
        paragraphs: list[str] = []
        for raw in page.splitlines():
            line = re.sub(r"\s+", " ", raw).strip()
            if not line:
                continue
            if HEADING.match(line):
                section = line[:140]
            paragraphs.append(line)
        text = " ".join(paragraphs)
        if not text:
            continue
        start, n = 0, 0
        while start < len(text):
            end = min(len(text), start + size)
            if end < len(text):
                cut = max(text.rfind(". ", start, end), text.rfind("; ", start, end))
                if cut > start + size // 2:
                    end = cut + 1
            piece = text[start:end].strip()
            if piece:
                n += 1
                chunks.append({"id": f"{source_id}:p{page_no}:{n}", "source": source_id, "page": page_no,
                               "section": section, "text": piece})
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def grounded(claim: dict, passage: str) -> bool:
    """Every number a claim states must appear in the passage it came from."""
    seen = {n.replace(",", "") for n in NUMBER.findall(passage)}
    stated = NUMBER.findall(str(claim.get("text") or ""))
    if claim.get("value") is not None:
        stated.append(str(claim["value"]))
    for number in stated:
        plain = number.replace(",", "")
        alt = plain[:-2] if plain.endswith(".0") else plain
        if plain not in seen and alt not in seen:
            return False
    return True


EXTRACT_PROMPT = """You turn a passage of a report into atomic claims for a forecasting knowledge base.
Report: {title} ({publisher}, published {published}). Section: {section}. Page {page}.

Rules:
- Write each claim as one plain sentence in your own words; do not copy sentences.
- Only claims the passage states. Every number you write must appear in the passage.
- kind is one of: projection (a number for a future year), estimate, fact, policy, finding, risk.
- For numbers give metric, value (a number), unit, geo and horizon (the year, YYYY).
- entities: the organisations, places, technologies, commodities and policies the claim is about.
- relations: links between entities the passage states, as subject, predicate (snake_case verb), object.

Reply with JSON only:
{{"claims": [{{"kind": "...", "text": "...", "entities": ["..."], "metric": null, "value": null,
  "unit": null, "geo": null, "horizon": null, "scenario": null}}],
 "relations": [{{"subject": "...", "predicate": "...", "object": "...", "claim": 0}}]}}

Passage:
{text}"""


def _first_json(text: str) -> Optional[dict]:
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S)
    start = text.find("{")
    while start != -1:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:end + 1])
                        return parsed if isinstance(parsed, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def extract(chunk: dict, source: dict, call: Callable[[str], str]) -> tuple[list[dict], list[dict]]:
    """Claims and relations an LLM finds in one chunk, keeping only claims grounded in it.

    Entities come back as names; ``merge_entities`` turns them into ids.
    """
    prompt = EXTRACT_PROMPT.format(title=source.get("title"), publisher=source.get("publisher"),
                                   published=source.get("published"), section=chunk.get("section") or "-",
                                   page=chunk.get("page"), text=chunk.get("text"))
    parsed = _first_json(call(prompt)) or {}
    claims, kept = [], {}
    for n, raw in enumerate(parsed.get("claims") or []):
        if not isinstance(raw, dict) or raw.get("kind") not in KINDS or not str(raw.get("text") or "").strip():
            continue
        claim = {k: raw.get(k) for k in ("kind", "text", "metric", "value", "unit", "geo", "horizon", "scenario")
                 if raw.get(k) not in (None, "")}
        if claim.get("value") is not None:
            try:
                claim["value"] = float(str(claim["value"]).replace(",", ""))
                if claim["value"].is_integer():
                    claim["value"] = int(claim["value"])
            except ValueError:
                claim.pop("value")
        if claim.get("horizon") is not None:
            claim["horizon"] = str(claim["horizon"])[:10]
            if parse_when(claim["horizon"]) is None:
                claim.pop("horizon")
        if not grounded(claim, str(chunk.get("text") or "")):
            continue
        claim["entities"] = [str(e) for e in raw.get("entities") or [] if str(e).strip()]
        claim.update(page=chunk.get("page"), section=chunk.get("section") or None, chunk=chunk.get("id"))
        kept[n] = len(claims)
        claims.append(claim)
    relations = []
    for raw in parsed.get("relations") or []:
        if not isinstance(raw, dict) or not all(raw.get(k) for k in ("subject", "predicate", "object")):
            continue
        rel = {"subject": str(raw["subject"]), "predicate": slug(raw["predicate"]), "object": str(raw["object"])}
        if isinstance(raw.get("claim"), int) and raw["claim"] in kept:
            rel["claim_index"] = kept[raw["claim"]]
        relations.append(rel)
    return claims, relations


def merge_entities(entities: dict[str, dict], names: Iterable[str]) -> dict[str, str]:
    """Map entity names to ids, adding unseen names as new entities. Returns name -> id."""
    lookup = {}
    for eid, ent in entities.items():
        for name in [ent.get("name"), *ent.get("aliases", [])]:
            lookup[normalize(name or "").strip()] = eid
    out = {}
    for name in names:
        key = normalize(name).strip()
        if not key:
            continue
        eid = lookup.get(key)
        if eid is None:
            eid = slug(name)
            while eid in entities and normalize(entities[eid].get("name", "")).strip() != key:
                eid += "_x"
            entities.setdefault(eid, {"id": eid, "name": name.strip(), "type": "concept", "aliases": []})
            lookup[key] = eid
        out[name] = eid
    return out


def llm_from_env() -> Optional[Callable[[str], str]]:
    """The self-hosted model (OpenAI-compatible chat endpoint) used to extract claims."""
    import requests
    url = (os.environ.get("KNOWLEDGE_LLM_URL") or os.environ.get("FUND_LLM_URL") or "").strip()
    key = os.environ.get("SELF_HOSTED_LLM_API_KEY", "").strip()
    if not url:
        return None
    model = os.environ.get("KNOWLEDGE_LLM_MODEL", "mlx-community/Qwen3.5-9B-4bit")

    def call(prompt: str) -> str:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        resp = requests.post(url, timeout=300, headers=headers,
                             json={"model": model, "max_tokens": 1800, "temperature": 0.0,
                                   "messages": [{"role": "user", "content": prompt}]})
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    return call


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def stats(corpus: Corpus) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    for claim in corpus.claims:
        kinds[claim.get("kind")] = kinds.get(claim.get("kind"), 0) + 1
    horizons = sorted({parse_when(c.get("horizon")).year for c in corpus.claims if parse_when(c.get("horizon"))})
    return {"sources": len(corpus.sources), "claims": len(corpus.claims), "by_kind": kinds,
            "entities": len(corpus.entities), "relations": len(corpus.relations), "chunks": len(corpus.chunks),
            "horizons": [horizons[0], horizons[-1]] if horizons else [], "errors": len(corpus.errors)}
