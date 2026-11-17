"""Read new reports into the knowledge base with the self-hosted model, on the Mac, while the live
site leaves the model alone (owner, 2026-11-17: phase 4 of growing the base).

A job is one report edition, queued from the watch list's inbox (``kind: report``) or by hand. Its
work lives in the knowledge clone's gitignored ``.kb/reports/<id>/``:

    job.json          source details and status: new, reading, review, published or dropped
    report.pdf        the file (checked against the inbox row's sha256 when it has one)
    chunks.jsonl      every page-tagged passage (published with the source, stays private)
    done.jsonl        one row per passage read: kept claims, relations, what the gates dropped
    candidates.jsonl  the claims waiting for review once every passage is read

Three gates stand between the model and the base:

1. grounded: every number a claim states appears in its passage (``knowledge.grounded``);
2. own words: no run of eight consecutive words copied from the passage (``knowledge.copied``); a
   claim that copies gets one chance to be restated by the model (``knowledge.rephrase``) and must
   then pass both gates again;
3. review: each candidate is marked keep or drop (and may be corrected) before ``publish``.

The model is the live site's model. A passage is sent only while no live request is in flight and
none finished in the last ``IDLE_S`` seconds, and a passage being read is abandoned the moment a
live request arrives (the server stops work for a client that hangs up), to be read again later.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional
from urllib.parse import urljoin

from . import knowledge
from . import knowledge_watch as watch

WORK = Path(".kb") / "reports"
#: Live traffic reaches the model through Caddy on this port; the extractor talks to 8080 directly.
LIVE_PORT = 8090
LIVE_LOG = Path.home() / "local-llm" / "access.log"
#: Wait this long after the last live request before sending a passage.
IDLE_S = 120.0
#: While a passage streams, look for live traffic this often.
CHECK_S = 2.0
LOCAL_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "mlx-community/Qwen3.5-9B-4bit"
MAX_TOKENS = 1800
#: A passage that fails this many times (not counting live-traffic pauses) is recorded as failed.
MAX_TRIES = 3

FORWARD = re.compile(r"\b(expect|project|forecast|outlook|scenario|risk|likely|unlikely|will|could|would|"
                     r"should|target|policy|plan|estimate|grow|decline|rise|fall|increase|decrease|by 20\d\d)",
                     re.I)
SKIP_SECTION = re.compile(r"\b(contents|references|bibliography|acknowledg|glossary|abbreviations|acronyms|"
                          r"annex table|statistical (annex|appendix)|index of|list of (figures|tables))", re.I)
LEADERS = re.compile(r"\.{4,}|(?:\. ){4,}")


class ReportError(Exception):
    """A job cannot go on: bad file, wrong status, or a base that would break."""


class Busy(Exception):
    """The live site needs the model; the passage is read again later."""


# --------------------------------------------------------------------------------------------
# Live traffic
# --------------------------------------------------------------------------------------------

def live_connections(port: int = LIVE_PORT, *, netstat: Optional[Callable[[], str]] = None) -> int:
    """Open connections into the live entry port: a forecast in flight."""
    if netstat is None:
        def netstat() -> str:
            return subprocess.run(["netstat", "-an", "-p", "tcp"], capture_output=True, text=True,
                                  timeout=10).stdout
    count = 0
    for line in netstat().splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[-1] == "ESTABLISHED" and parts[3].endswith(f".{port}"):
            count += 1
    return count


def last_live_request(log: Path = LIVE_LOG, *, tail_bytes: int = 65536) -> Optional[float]:
    """When the newest answered model request in Caddy's access log finished (epoch seconds)."""
    try:
        with open(log, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - tail_bytes))
            lines = fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        uri = str((row.get("request") or {}).get("uri") or "")
        if uri.startswith("/v1/") and row.get("status") == 200 and row.get("ts"):
            return float(row["ts"])
    return None


def live_gate(*, port: int = LIVE_PORT, log: Path = LIVE_LOG, idle_s: float = IDLE_S,
              clock: Callable[[], float] = time.time) -> Callable[[bool], Optional[str]]:
    """``busy(starting)`` -> why the model is not ours right now, or None.

    Before a passage (``starting``) the site must have been quiet for ``idle_s``; while one streams,
    only a request in flight interrupts it.
    """
    def busy(starting: bool = True) -> Optional[str]:
        n = live_connections(port)
        if n:
            return f"{n} live request(s) in flight"
        if starting:
            last = last_live_request(log)
            if last is not None and clock() - last < idle_s:
                return f"live request {clock() - last:.0f}s ago"
        return None
    return busy


def stream_call(prompt: str, busy: Callable[[bool], Optional[str]], *, url: str = LOCAL_URL,
                model: str = MODEL, key: str = "", max_tokens: int = MAX_TOKENS, check_s: float = CHECK_S,
                post: Optional[Callable[..., Any]] = None) -> str:
    """Stream one completion, hanging up (``Busy``) as soon as live traffic shows up."""
    if post is None:
        import requests
        post = requests.post
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "messages": [{"role": "user", "content": prompt}]}
    resp = post(url, json=body, headers=headers, stream=True, timeout=(10, 300))
    try:
        resp.raise_for_status()
        parts: list[str] = []
        last = time.monotonic()
        for line in resp.iter_lines(decode_unicode=True):
            if time.monotonic() - last >= check_s:
                last = time.monotonic()
                why = busy(False)
                if why:
                    raise Busy(why)
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = (json.loads(data).get("choices") or [{}])[0].get("delta") or {}
            except ValueError:
                continue
            if delta.get("content"):
                parts.append(delta["content"])
        return "".join(parts)
    finally:
        resp.close()


# --------------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def folder(root: Path, jid: str) -> Path:
    return root / WORK / jid


def load_job(root: Path, jid: str) -> dict:
    path = folder(root, jid) / "job.json"
    if not path.exists():
        raise ReportError(f"no job {jid}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_job(root: Path, job: Mapping[str, Any]) -> None:
    path = folder(root, job["id"]) / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def jobs(root: Path) -> list[dict]:
    base = root / WORK
    if not base.is_dir():
        return []
    out = [json.loads(p.read_text(encoding="utf-8")) for p in base.glob("*/job.json")]
    return sorted(out, key=lambda j: (j.get("queued") or "", j["id"]))


def queue(root: Path, *, jid: str, title: str, publisher: str, published: str, url: Optional[str] = None,
          file_url: Optional[str] = None, path: Optional[Path] = None, sha256: Optional[str] = None,
          inbox: Optional[Mapping[str, str]] = None, note: Optional[str] = None,
          extra: Optional[Mapping[str, Any]] = None) -> dict:
    """Add a report to read. Queuing the same id again returns the existing job."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{4,90}", jid):
        raise ReportError(f"bad id {jid!r}: use <publisher>-<yyyy>-<mm>-<topic>, lower case")
    if (root / "sources" / jid).exists():
        raise ReportError(f"sources/{jid} is already in the base")
    if knowledge.parse_when(published) is None:
        raise ReportError(f"published must be YYYY-MM-DD, got {published!r}")
    where = folder(root, jid)
    if (where / "job.json").exists():
        job = load_job(root, jid)
        if job.get("status") in ("new", "stuck") and (file_url or path is not None):
            # Queued again with its file: the night run could not find or fetch it.
            for key in ("error", "error_at", "fetch_failures", "pdf_options"):
                job.pop(key, None)
            job["status"] = "new"
            if file_url:
                job["file_url"] = file_url
            if path is not None:
                data = Path(path).expanduser().read_bytes()
                _check_pdf(data, None)
                (where / "report.pdf").write_bytes(data)
                job["sha256"] = hashlib.sha256(data).hexdigest()
            save_job(root, job)
        return job
    where.mkdir(parents=True, exist_ok=True)
    job: dict[str, Any] = {"id": jid, "title": title, "publisher": publisher, "published": published,
                           "status": "new", "queued": _now()}
    if url:
        job["url"] = url
    if file_url:
        job["file_url"] = file_url
    if sha256:
        job["sha256"] = sha256
    if inbox:
        job["inbox"] = dict(inbox)
    if note:
        job["note"] = note
    if extra:
        job.update({k: v for k, v in extra.items() if v})
    if path is not None:
        data = Path(path).expanduser().read_bytes()
        _check_pdf(data, sha256)
        (where / "report.pdf").write_bytes(data)
        job["sha256"] = hashlib.sha256(data).hexdigest()
    save_job(root, job)
    return job


def _check_pdf(data: bytes, sha256: Optional[str]) -> None:
    if not data.startswith(b"%PDF"):
        raise ReportError("the file is not a PDF")
    if sha256 and hashlib.sha256(data).hexdigest() != sha256:
        raise ReportError("the file differs from the one the watch list found (sha256)")


def _published(row: Mapping[str, Any]) -> tuple[str, bool]:
    """A publication date for an inbox row: the edition key when it is a date, else when it was found."""
    key = str(row.get("edition") or "")
    for fmt in ("%B %d, %Y", "%Y/%m/%d", "%Y-%m-%d", "%Y%m%d", "%d %B %Y"):
        try:
            return datetime.strptime(key, fmt).date().isoformat(), False
        except ValueError:
            continue
    return str(row.get("found") or date.today().isoformat()), True


def job_id(row: Mapping[str, Any], published: str) -> str:
    publisher = str(row.get("source") or "report").split("-")[0]
    topic = re.sub(r"[^a-z0-9]+", "-", str(row.get("edition") or "").lower()).strip("-")
    topic = re.sub(r"^(19|20)\d\d-?|-?(19|20)\d\d$", "", topic).strip("-") or "report"
    return f"{publisher}-{published[:7]}-{topic}"[:80].rstrip("-")


def from_inbox(root: Path) -> list[dict]:
    """Queue every new report edition the watch list found; returns the jobs queued now."""
    queued = []
    have = {((j.get("inbox") or {}).get("source"), (j.get("inbox") or {}).get("edition")) for j in jobs(root)}
    for row in _read_jsonl(root / "inbox.jsonl"):
        if row.get("status") != "new" or row.get("kind") != "report":
            continue
        if (row.get("source"), row.get("edition")) in have:
            continue
        published, guessed = _published(row)
        pdfs = [f for f in row.get("files") or [] if str(f.get("url", "")).lower().split("?")[0].endswith(".pdf")
                or str(f.get("content_type", "")).endswith("pdf")]
        first = pdfs[0] if pdfs else {}
        jid = job_id(row, published)
        if (root / "sources" / jid).exists():
            continue
        note = "publication date is when the watch list found it: check it" if guessed else None
        job = queue(root, jid=jid, title=f"{row.get('title') or row.get('source')}: {row.get('edition')}",
                    publisher=str(row.get("publisher") or ""), published=published, url=row.get("url"),
                    file_url=first.get("url"), sha256=first.get("sha256"),
                    inbox={"source": row.get("source"), "edition": row.get("edition")}, note=note)
        queued.append(job)
    return queued


def find_pdf(page: str, base: str, title: str = "") -> tuple[Optional[str], list[str]]:
    """The report's PDF on an edition page: the only PDF link, or the one that says it is the report."""
    links = []
    for href, label in re.findall(r'<a\b[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, flags=re.S | re.I):
        url = urljoin(base, href.strip())
        if url.lower().split("?")[0].endswith(".pdf") and url not in [u for u, _ in links]:
            links.append((url, re.sub(r"<[^>]+>|\s+", " ", label).strip().lower()))
    if len(links) == 1:
        return links[0][0], [links[0][0]]
    words = [w for w in knowledge.WORD.findall(title.lower()) if len(w) > 3]
    best = [u for u, label in links
            if re.search(r"full report|download (the )?report|read the report|full text|report \(pdf\)", label)]
    if not best and words:
        scored = sorted(((sum(w in (u.lower() + " " + label) for w in words), u) for u, label in links), reverse=True)
        if scored and scored[0][0] >= 2 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            best = [scored[0][1]]
    return (best[0] if len(best) == 1 else None), [u for u, _ in links]


def fetch_file(root: Path, job: dict, crawler: watch.Crawler) -> Path:
    """Download the job's PDF (finding it on the edition page if needed)."""
    where = folder(root, job["id"])
    pdf = where / "report.pdf"
    if pdf.exists():
        return pdf
    url = job.get("file_url")
    if not url and job.get("url"):
        resp = crawler.fetch(job["url"], max_bytes=watch.FILE_BYTES)
        if not resp.ok:
            raise ReportError(f"edition page: {resp.error or resp.status}")
        if resp.body.startswith(b"%PDF"):
            url = job["url"]
        else:
            url, options = find_pdf(resp.text(), resp.url or job["url"], job.get("title", ""))
            if not url:
                job["pdf_options"] = options[:20]
                raise ReportError(f"could not tell which of {len(options)} PDF links is the report; "
                                  "queue it again with the file URL")
        job["file_url"] = url
    if not url:
        raise ReportError("no file URL")
    resp = crawler.fetch(url, max_bytes=watch.FILE_BYTES)
    if not resp.ok:
        raise ReportError(f"file: {resp.error or resp.status}")
    _check_pdf(resp.body, job.get("sha256"))
    pdf.write_bytes(resp.body)
    job["sha256"] = hashlib.sha256(resp.body).hexdigest()
    return pdf


def worth_reading(chunk: Mapping[str, Any]) -> bool:
    """Passages that can hold a claim: a number or forward-looking wording, not contents or references."""
    text = str(chunk.get("text") or "")
    if SKIP_SECTION.search(str(chunk.get("section") or "")) or SKIP_SECTION.search(text[:80]):
        return False
    if len(LEADERS.findall(text)) >= 3 or text.count("http") + text.lower().count("doi") >= 3:
        return False
    words = knowledge.WORD.findall(text.lower())
    if len(words) < 25:
        return False
    numbers = sum(w[0].isdigit() for w in words)
    if numbers > 0.5 * len(words):  # a table of figures, not prose
        return False
    return bool(numbers) or bool(FORWARD.search(text))


def prepare(root: Path, job: dict, crawler: Optional[watch.Crawler] = None,
            pages: Optional[Callable[[Path], list[str]]] = None) -> list[dict]:
    """Have the file and its passages; returns the passages."""
    where = folder(root, job["id"])
    chunks_path = where / "chunks.jsonl"
    if chunks_path.exists():
        return _read_jsonl(chunks_path)
    pdf = where / "report.pdf"
    if not pdf.exists():
        if crawler is None:
            raise ReportError("no file yet and no crawler to fetch it")
        pdf = fetch_file(root, job, crawler)
    text_pages = (pages or knowledge.pdf_pages)(pdf)
    chunks = knowledge.chunk_pages(job["id"], text_pages)
    if not chunks:
        raise ReportError("the PDF has no text (a scan?)")
    knowledge.write_jsonl(chunks_path, chunks)
    job.update(pages=len(text_pages), passages=len(chunks), to_read=sum(map(worth_reading, chunks)),
               status="reading")
    save_job(root, job)
    return chunks


def read(root: Path, job: dict, call: Callable[[str], str], busy: Callable[[bool], Optional[str]], *,
         deadline: float, crawler: Optional[watch.Crawler] = None, wait_s: float = 30.0,
         sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time,
         log: Callable[[str], None] = lambda line: None) -> dict:
    """Read the job's unread passages until done or ``deadline``; returns counts for this session."""
    chunks = prepare(root, job, crawler)
    where = folder(root, job["id"])
    done = {row["chunk"] for row in _read_jsonl(where / "done.jsonl")}
    tries: dict = job.setdefault("tries", {})
    source = {k: job.get(k) for k in ("title", "publisher", "published")}
    stats = {"read": 0, "claims": 0, "dropped": 0, "paused_s": 0.0, "interrupted": 0}
    job["status"] = "reading"
    for chunk in chunks:
        if chunk["id"] in done or not worth_reading(chunk):
            continue
        while True:  # until this passage is read, has failed for good, or time is up
            if clock() >= deadline:
                save_job(root, job)
                return {**stats, "finished": False}
            if busy(True):
                stats["paused_s"] += wait_s
                sleep(wait_s)
                continue
            started = clock()
            rejected: list[dict] = []
            try:
                claims, relations = knowledge.extract(chunk, source, call, rejected=rejected)
                copies = [r["claim"] for r in rejected if r["why"] == "copies the passage" and r.get("claim")]
                rescued = knowledge.rephrase(copies, str(chunk.get("text") or ""), call) if copies else []
            except Busy as exc:
                log(f"{chunk['id']}: handed the model back to the site ({exc})")
                stats["interrupted"] += 1
                continue
            except Exception as exc:  # one bad passage must not stop the report
                n = tries[chunk["id"]] = tries.get(chunk["id"], 0) + 1
                log(f"{chunk['id']}: failed ({type(exc).__name__}: {str(exc)[:120]})")
                save_job(root, job)
                if n >= MAX_TRIES:
                    _append_jsonl(where / "done.jsonl", {"chunk": chunk["id"], "page": chunk.get("page"),
                                                        "error": type(exc).__name__, "claims": [], "relations": []})
                    break
                sleep(wait_s)  # e.g. the model server restarting
                continue
            dropped = [{"text": r["text"], "why": r["why"]} for r in rejected]
            _append_jsonl(where / "done.jsonl", {"chunk": chunk["id"], "page": chunk.get("page"),
                                                "claims": claims + rescued, "relations": relations,
                                                "dropped": dropped, "rephrased": len(rescued),
                                                "seconds": round(clock() - started, 1)})
            stats["read"] += 1
            stats["claims"] += len(claims) + len(rescued)
            stats["dropped"] += len(rejected) - len(rescued)
            log(f"{chunk['id']}: {len(claims)} claims + {len(rescued)} restated, "
                f"{len(rejected) - len(rescued)} dropped, {clock() - started:.0f}s")
            break
    finish(root, job)
    return {**stats, "finished": True}


def unread(root: Path, job: Mapping[str, Any]) -> int:
    where = folder(root, job["id"])
    done = {row["chunk"] for row in _read_jsonl(where / "done.jsonl")}
    return sum(1 for c in _read_jsonl(where / "chunks.jsonl") if worth_reading(c) and c["id"] not in done)


def _gist(text: str) -> tuple[frozenset, frozenset]:
    words = [w for w in knowledge.WORD.findall(str(text).lower()) if w not in knowledge.STOP]
    return (frozenset(w for w in words if not w[0].isdigit()),
            frozenset(n.replace(",", "") for n in knowledge.NUMBER.findall(str(text))))


def same_claim(a: str, b: str, *, overlap: float = 0.6) -> bool:
    """Two wordings of one claim: the same numbers and mostly the same words.

    Passages overlap, and the model states a range twice (once per bound), so a report yields
    many near-copies of a claim.
    """
    (wa, na), (wb, nb) = _gist(a), _gist(b)
    if na != nb or not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= overlap


def finish(root: Path, job: dict) -> list[dict]:
    """Collect the read passages' claims into numbered candidates for review, without near-copies."""
    where = folder(root, job["id"])
    seen: set[str] = set()
    candidates: list[dict] = []
    relations: list[dict] = []
    dropped = failed = 0
    for row in _read_jsonl(where / "done.jsonl"):
        dropped += len(row.get("dropped") or []) - int(row.get("rephrased") or 0)
        failed += bool(row.get("error"))
        local: dict[int, str] = {}
        for i, claim in enumerate(row.get("claims") or []):
            key = knowledge.normalize(claim["text"])
            if key in seen or any(same_claim(claim["text"], c["text"]) for c in candidates[-40:]):
                continue
            seen.add(key)
            cid = f"c{len(candidates) + 1:03d}"
            local[i] = cid
            candidates.append({"n": cid, **claim, "review": None})
        for rel in row.get("relations") or []:
            cid = local.get(rel.get("claim_index")) if rel.get("claim_index") is not None else None
            relations.append({"subject": rel["subject"], "predicate": rel["predicate"], "object": rel["object"],
                              "claim": cid})
    knowledge.write_jsonl(where / "candidates.jsonl", candidates)
    knowledge.write_jsonl(where / "candidate_relations.jsonl", relations)
    job.update(status="review", candidates=len(candidates), gate_dropped=dropped, failed_passages=failed,
               read_done=_now())
    save_job(root, job)
    return candidates


# --------------------------------------------------------------------------------------------
# Review and publish
# --------------------------------------------------------------------------------------------

def review_sheet(root: Path, jid: str, *, passage_chars: int = 700) -> str:
    """Every candidate beside the passage it came from, for the reviewer (stays local: it quotes the report)."""
    job = load_job(root, jid)
    where = folder(root, jid)
    chunks = {c["id"]: c for c in _read_jsonl(where / "chunks.jsonl")}
    lines = [f"# Review {jid}", "", f"{job.get('title')} ({job.get('publisher')}, {job.get('published')})",
             f"{job.get('candidates', 0)} candidates; the gates dropped {job.get('gate_dropped', 0)}.", ""]
    if job.get("note"):
        lines += [f"Note: {job['note']}", ""]
    last_chunk = None
    for c in _read_jsonl(where / "candidates.jsonl"):
        if c.get("chunk") != last_chunk:
            text = str((chunks.get(c.get("chunk")) or {}).get("text") or "")
            lines += ["", f"## {c.get('chunk')} (page {c.get('page')})", "", "> " + text[:passage_chars], ""]
            last_chunk = c.get("chunk")
        nums = ", ".join(f"{k}={c[k]}" for k in ("metric", "value", "unit", "geo", "horizon", "scenario") if c.get(k))
        mark = {"keep": "[keep] ", "drop": "[drop] "}.get(c.get("review") or "", "")
        if c.get("rephrased"):
            mark += "(restated) "
        lines.append(f"- {c['n']} {mark}{c['kind']}: {c['text']}" + (f"  ({nums})" if nums else "")
                     + f"  entities: {', '.join(c.get('entities') or [])}")
    return "\n".join(lines) + "\n"


def decide(root: Path, jid: str, *, keep: Iterable[str] = (), drop: Iterable[str] = (),
           rest: Optional[str] = None, edits: Optional[Mapping[str, Mapping[str, Any]]] = None) -> dict:
    """Mark candidates keep or drop, optionally correcting fields; ``rest`` decides all the others."""
    path = folder(root, jid) / "candidates.jsonl"
    rows = _read_jsonl(path)
    names = {r["n"] for r in rows}
    keep, drop = set(keep), set(drop)
    unknown = (keep | drop | set(edits or {})) - names
    if unknown:
        raise ReportError(f"no such candidates: {', '.join(sorted(unknown))}")
    if keep & drop:
        raise ReportError(f"both kept and dropped: {', '.join(sorted(keep & drop))}")
    for row in rows:
        fix = (edits or {}).get(row["n"])
        if fix:
            bad = set(fix) - {"kind", "text", "metric", "value", "unit", "geo", "horizon", "scenario", "entities"}
            if bad:
                raise ReportError(f"{row['n']}: cannot edit {', '.join(sorted(bad))}")
            row.update(fix)
            row["edited"] = True
        if row["n"] in keep:
            row["review"] = "keep"
        elif row["n"] in drop:
            row["review"] = "drop"
        elif rest and not row.get("review"):
            row["review"] = rest
    knowledge.write_jsonl(path, rows)
    return {"keep": sum(r.get("review") == "keep" for r in rows), "drop": sum(r.get("review") == "drop" for r in rows),
            "undecided": sum(not r.get("review") for r in rows)}


def _lookup(entities: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    out = {}
    for eid, ent in entities.items():
        for name in [ent.get("name"), *ent.get("aliases", [])]:
            out[knowledge.normalize(name or "").strip()] = eid
    return out


def entity_names(root: Path, jid: str) -> dict[str, Any]:
    """The kept candidates' entity names: those the base already knows (name -> id) and the new ones
    with how often they occur, for the reviewer's entity map."""
    lookup = _lookup(knowledge.load(root).entities)
    known: dict[str, str] = {}
    new: dict[str, int] = {}
    for row in _read_jsonl(folder(root, jid) / "candidates.jsonl"):
        if row.get("review") == "drop":
            continue
        for name in row.get("entities") or []:
            eid = lookup.get(knowledge.normalize(name).strip())
            if eid:
                known[name] = eid
            else:
                new[name] = new.get(name, 0) + 1
    return {"known": known, "new": dict(sorted(new.items(), key=lambda kv: -kv[1]))}


def resolve_entities(names: Iterable[str], entities: dict[str, dict], entity_map: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Entity ids for a claim's names: the base's own entities by name or alias, else the reviewer's
    map (``{"map": {name: id or null}, "new": {id: {name, type, aliases}}}``). A name the map does not
    cover is left off the claim (returned second), so the model cannot fill the graph with vague
    nodes such as "governments" or "costs"; the claim's text stays searchable."""
    lookup = _lookup(entities)
    mapped = {knowledge.normalize(k).strip(): v for k, v in (entity_map.get("map") or {}).items()}
    new = entity_map.get("new") or {}
    ids, unlinked = [], []
    for name in names:
        key = knowledge.normalize(name).strip()
        eid = lookup.get(key)
        if eid is None and key in mapped:
            eid = mapped[key]
            if eid and eid not in entities:
                if eid not in new:
                    raise ReportError(f"entity map sends {name!r} to {eid!r}, which is neither in the base nor in 'new'")
                spec = new[eid]
                if not spec.get("name") or not spec.get("type"):
                    raise ReportError(f"new entity {eid!r} needs a name and a type")
                entities[eid] = {"id": eid, "name": spec["name"], "type": spec["type"],
                                 "aliases": list(spec.get("aliases") or [])}
        if eid:
            ids.append(eid)
        elif key not in mapped:
            unlinked.append(name)
    return sorted(set(ids)), unlinked


def publish(root: Path, jid: str, *, reviewer: str, today: Optional[date] = None,
            source_extra: Optional[Mapping[str, Any]] = None,
            entity_map: Optional[Mapping[str, Any]] = None) -> Path:
    """Write the kept candidates into ``sources/<id>/``; undo everything if the base would break."""
    job = load_job(root, jid)
    if job.get("status") != "review":
        raise ReportError(f"{jid} is {job.get('status')}, not waiting for review")
    where = folder(root, jid)
    rows = _read_jsonl(where / "candidates.jsonl")
    undecided = [r["n"] for r in rows if r.get("review") not in ("keep", "drop")]
    if undecided:
        raise ReportError(f"{len(undecided)} candidates have no decision: {', '.join(undecided[:8])}")
    kept = [r for r in rows if r["review"] == "keep"]
    if not kept:
        raise ReportError("nothing kept: drop the job instead")
    chunks = {c["id"]: c for c in _read_jsonl(where / "chunks.jsonl")}
    for row in kept:  # an edit must still pass the first two gates
        passage = str((chunks.get(row.get("chunk")) or {}).get("text") or "")
        if not knowledge.grounded(row, passage):
            raise ReportError(f"{row['n']}: a number is not in its passage")
        if knowledge.copied(row["text"], passage):
            raise ReportError(f"{row['n']}: copies its passage")
    target = root / "sources" / jid
    if target.exists():
        raise ReportError(f"sources/{jid} already exists")
    entities_path = root / "entities.jsonl"
    before = entities_path.read_text(encoding="utf-8") if entities_path.exists() else ""
    corpus = knowledge.load(root)
    if corpus.errors:
        raise ReportError("the base has errors: " + "; ".join(corpus.errors[:3]))
    entities = {k: dict(v) for k, v in corpus.entities.items()}
    entity_map = entity_map or {}
    unlinked: set[str] = set()
    claims, ids = [], {}
    for n, row in enumerate(kept, 1):
        ents, missing = resolve_entities([str(e) for e in row.get("entities") or []], entities, entity_map)
        unlinked.update(missing)
        claim = {"id": f"{jid}:c{n:03d}", "source": jid}
        claim.update({k: row[k] for k in ("kind", "text", "metric", "value", "unit", "geo", "horizon", "scenario")
                      if row.get(k) not in (None, "")})
        claim.update(entities=ents, page=row.get("page"))
        if row.get("section"):
            claim["section"] = row["section"]
        claim["chunk"] = row.get("chunk")
        claim["extracted_by"] = "self-hosted Qwen3.5-9B, reviewed" + (" and corrected" if row.get("edited") else "")
        claims.append(claim)
        ids[row["n"]] = claim["id"]
    relations = []
    for rel in _read_jsonl(where / "candidate_relations.jsonl"):
        if rel.get("claim") and rel["claim"] not in ids:
            continue  # its claim was dropped
        subject, _ = resolve_entities([rel["subject"]], entities, entity_map)
        obj, _ = resolve_entities([rel["object"]], entities, entity_map)
        if not subject or not obj or subject == obj:
            continue  # a relation needs two linked entities
        relations.append({"subject": subject[0], "predicate": rel["predicate"], "object": obj[0],
                          "claims": [ids[rel["claim"]]] if rel.get("claim") else []})
    source = {"id": jid, "title": job["title"], "publisher": job["publisher"], "published": job["published"],
              "kind": "report", "added": (today or date.today()).isoformat(),
              "extracted_by": f"self-hosted Qwen3.5-9B; reviewed by {reviewer}"}
    for key in ("url", "file_url", "sha256"):
        if job.get(key):
            source[key] = job[key]
    source.update({k: v for k, v in (source_extra or {}).items() if v})
    used = {c["chunk"] for c in claims}
    target.mkdir(parents=True)
    try:
        (target / "source.json").write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        knowledge.write_jsonl(target / "claims.jsonl", claims)
        knowledge.write_jsonl(target / "relations.jsonl", relations)
        knowledge.write_jsonl(target / "chunks.jsonl", [c for c in chunks.values() if c["id"] in used])
        knowledge.write_jsonl(entities_path, [entities[k] for k in sorted(entities)])
        errors = knowledge.load(root).errors
        if errors:
            raise ReportError("the base would not load: " + "; ".join(errors[:3]))
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        entities_path.write_text(before, encoding="utf-8")
        raise
    if job.get("inbox"):
        _mark_inbox(root, job["inbox"], jid)
    job.update(status="published", published_claims=len(claims), reviewer=reviewer, published_at=_now(),
               unlinked_entities=sorted(unlinked))
    save_job(root, job)
    return target


def _mark_inbox(root: Path, inbox: Mapping[str, Any], jid: str) -> None:
    path = root / "inbox.jsonl"
    rows = _read_jsonl(path)
    changed = False
    for row in rows:
        if row.get("source") == inbox.get("source") and row.get("edition") == inbox.get("edition"):
            row.update(status="read", kb_source=jid)
            changed = True
    if changed:
        knowledge.write_jsonl(path, rows)


def drop_job(root: Path, jid: str, why: str) -> dict:
    job = load_job(root, jid)
    job.update(status="dropped", why=why, dropped_at=_now())
    save_job(root, job)
    if job.get("inbox"):
        path = root / "inbox.jsonl"
        rows = _read_jsonl(path)
        for row in rows:
            if row.get("source") == job["inbox"].get("source") and row.get("edition") == job["inbox"].get("edition"):
                row.update(status="skipped", why=why)
        if rows:
            knowledge.write_jsonl(path, rows)
    return job


# --------------------------------------------------------------------------------------------
# The night run
# --------------------------------------------------------------------------------------------

def in_window(hours: str, now: Optional[datetime] = None) -> bool:
    """Whether local time is inside ``"22-8"`` (wrapping midnight) or ``"0-24"``."""
    start, end = (int(x) for x in hours.split("-"))
    hour = (now or datetime.now()).hour
    return start <= hour < end if start <= end else hour >= start or hour < end


def run(root: Path, *, call: Callable[[str], str], busy: Callable[[bool], Optional[str]], deadline: float,
        crawler: Optional[watch.Crawler] = None, sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time, log: Callable[[str], None] = lambda line: None) -> list[dict]:
    """Queue new inbox reports, then read jobs oldest first until the deadline."""
    results = []
    for job in from_inbox(root):
        log(f"queued {job['id']}")
    for job in jobs(root):
        if job.get("status") not in ("new", "reading"):
            continue
        if clock() >= deadline:
            break
        try:
            stats = read(root, job, call, busy, deadline=deadline, crawler=crawler, sleep=sleep, clock=clock, log=log)
            results.append({"id": job["id"], "status": job["status"], **stats})
        except ReportError as exc:
            job.update(status="new", error=str(exc), error_at=_now())
            failures = job["fetch_failures"] = job.get("fetch_failures", 0) + 1
            if failures >= 3:
                job["status"] = "stuck"
            save_job(root, job)
            results.append({"id": job["id"], "status": job["status"], "error": str(exc)})
            log(f"{job['id']}: {exc}")
    return results


def status_table(root: Path) -> str:
    lines = ["| Report | Status | Passages to read | Left | Candidates |", "|---|---|---|---|---|"]
    for job in jobs(root):
        left = unread(root, job) if job.get("status") == "reading" else ""
        lines.append(f"| {job['id']} | {job.get('status')} | {job.get('to_read', '')} | {left} | "
                     f"{job.get('candidates', job.get('published_claims', ''))} |")
    return "\n".join(lines) + "\n"
