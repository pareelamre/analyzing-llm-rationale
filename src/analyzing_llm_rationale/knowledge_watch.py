"""Watch trusted publishers for new editions of the reports Foresea's knowledge base reads.

The watch list lives in the private knowledge repo (``watchlist.json``): for each publisher, the
page that lists its editions and a pattern that picks the editions out of that page (``link``: a
regex over the page's links, its first group naming the edition; ``text``: a regex over the page
text, for publishers that reuse one file name and only change a release date). ``watch_state.json``
remembers which editions were already seen, and every new one is appended to ``inbox.jsonl`` with
its files, which are downloaded for the reading step.

The crawler is deliberately polite: it names itself honestly, obeys robots.txt, pauses between
requests to a host, and reports a site that blocks it instead of working around the block. The
first check of a publisher only records what is already there (a baseline), so adding a source
never floods the inbox with its back catalogue.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

USER_AGENT = "ForeseaKnowledgeWatch/1.0 (+https://foresea.ink)"
#: Seconds between two requests to one host.
PAUSE_S = 1.5
PAGE_BYTES = 4 * 1024 * 1024
FILE_BYTES = 80 * 1024 * 1024
#: More new editions than this in one check means the page changed shape, not that a publisher
#: released a dozen reports overnight: the editions are recorded but nothing is downloaded.
MAX_NEW = 5
KINDS = ("table", "report", "data")
ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
HREF = re.compile(r"""href\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>'"]+))""", re.I)
TAGS = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.I | re.S)


@dataclass
class Response:
    status: int
    body: bytes
    content_type: str = ""
    url: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status == 200

    def text(self) -> str:
        return self.body.decode("utf-8", "replace")


Getter = Callable[..., Response]


def http_get(url: str, *, max_bytes: int = PAGE_BYTES, timeout: float = 30.0) -> Response:
    """GET with the watcher's user agent; a body over ``max_bytes`` is an error, not a truncation."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = resp.read(max_bytes + 1)
            if len(body) > max_bytes:
                return Response(resp.status, b"", resp.headers.get("Content-Type", ""), resp.geturl(),
                                f"larger than {max_bytes // (1024 * 1024)} MB")
            return Response(resp.status, body, resp.headers.get("Content-Type", ""), resp.geturl())
    except urllib.error.HTTPError as exc:
        return Response(exc.code, b"", "", url)
    except Exception as exc:  # aqg: network boundary -- a dead site is reported, never raised
        return Response(0, b"", "", url, f"{type(exc).__name__}: {exc}")


class Crawler:
    """Fetches pages for the watcher: robots.txt first, one host at a time, with a pause."""

    def __init__(self, get: Getter = http_get, *, pause: float = PAUSE_S,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.get, self.pause, self.sleep, self.clock = get, pause, sleep, clock
        self._robots: dict[str, tuple[Optional[urllib.robotparser.RobotFileParser], Optional[str]]] = {}
        self._last: dict[str, float] = {}

    def _call(self, url: str, max_bytes: int) -> Response:
        host = urllib.parse.urlsplit(url).netloc.lower()
        if host in self._last:
            wait = self.pause - (self.clock() - self._last[host])
            if wait > 0:
                self.sleep(wait)
        try:
            return self.get(url, max_bytes=max_bytes)
        finally:
            self._last[host] = self.clock()

    def allowed(self, url: str) -> tuple[bool, Optional[str]]:
        """Whether robots.txt lets this crawler fetch ``url``, and why not."""
        parts = urllib.parse.urlsplit(url)
        host = parts.netloc.lower()
        if host not in self._robots:
            resp = self._call(f"{parts.scheme}://{parts.netloc}/robots.txt", 512 * 1024)
            # RFC 9309: a 4xx robots.txt is "unavailable" and means no rules, except that we read
            # 401/403 as the site refusing this crawler; 429, 5xx and network errors mean "try later".
            if resp.ok:
                parser = urllib.robotparser.RobotFileParser()
                parser.parse(resp.text().splitlines())
                self._robots[host] = (parser, None)
            elif resp.status in (401, 403):
                self._robots[host] = (None, f"{host} blocks this crawler (HTTP {resp.status})")
            elif resp.error is None and 400 <= resp.status < 500 and resp.status != 429:
                self._robots[host] = (None, None)
            else:
                self._robots[host] = (None, f"could not read {host}/robots.txt ({resp.error or resp.status})")
        parser, refusal = self._robots[host]
        if refusal:
            return False, refusal
        if parser is not None and not parser.can_fetch(USER_AGENT, url):
            return False, f"robots.txt disallows {parts.path or '/'}"
        return True, None

    def fetch(self, url: str, *, max_bytes: int = PAGE_BYTES) -> Response:
        ok, why = self.allowed(url)
        if not ok:
            return Response(0, b"", "", url, why)
        return self._call(url, max_bytes)


# --------------------------------------------------------------------------------------------
# Pages -> editions
# --------------------------------------------------------------------------------------------

def links(page: str, base: str) -> list[str]:
    """Every link on a page, absolute, unescaped, without fragments, in page order."""
    out, seen = [], set()
    for match in HREF.finditer(page):
        raw = html.unescape(next(g for g in match.groups() if g is not None)).strip()
        if not raw or raw.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        url = urllib.parse.urldefrag(urllib.parse.urljoin(base, raw))[0]
        if url.startswith(("http://", "https://")) and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def page_text(page: str) -> str:
    return " ".join(html.unescape(TAGS.sub(" ", page)).split())


def editions(source: Mapping[str, Any], page: str, base: str) -> list[dict[str, str]]:
    """The editions a publisher's page lists right now, in page order: [{key, url}]."""
    rule = source["edition"]
    if "link" in rule:
        pattern, out, seen = re.compile(rule["link"]), [], set()
        for url in links(page, base):
            match = pattern.search(url)
            if not match:
                continue
            key = match.group(1) if pattern.groups else url
            if key not in seen:
                seen.add(key)
                out.append({"key": key, "url": url})
        return out
    match = re.search(rule["text"], page_text(page))
    if not match:
        return []
    return [{"key": match.group(1) if match.groups() else match.group(0), "url": base}]


def newest(source: Mapping[str, Any], found: Sequence[Mapping[str, str]]) -> Optional[Mapping[str, str]]:
    if not found:
        return None
    order = source.get("newest", "first")
    if order == "max":
        return max(found, key=lambda e: e["key"])
    return found[-1] if order == "last" else found[0]


def edition_files(source: Mapping[str, Any], edition: Mapping[str, str], page: str, base: str) -> list[str]:
    """What to download for a new edition: the page's links matching ``files``, else the edition itself."""
    patterns = [re.compile(p) for p in source.get("files") or []]
    if not patterns:
        return [edition["url"]]
    return [u for u in links(page, base) if any(p.search(u) for p in patterns)]


def _file_name(url: str, content_type: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(urllib.parse.urlsplit(url).path).name).strip("._")
    if not name or "." not in name:
        name = (name or "page") + (".html" if "html" in content_type else ".bin")
    return name[:120]


def download(crawler: Crawler, url: str, folder: Path) -> dict[str, Any]:
    resp = crawler.fetch(url, max_bytes=FILE_BYTES)
    row: dict[str, Any] = {"url": url}
    if not resp.ok:
        row["error"] = resp.error or f"HTTP {resp.status}"
        return row
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / _file_name(url, resp.content_type)
    path.write_bytes(resp.body)
    row.update(bytes=len(resp.body), sha256=hashlib.sha256(resp.body).hexdigest(),
               content_type=resp.content_type.split(";")[0].strip(), saved=str(path))
    return row


def check(source: Mapping[str, Any], seen: Mapping[str, Any], crawler: Crawler, *,
          baseline: bool, download_dir: Optional[Path]) -> dict[str, Any]:
    """Look at one publisher's page: what it lists, what is new, and the new editions' files."""
    out: dict[str, Any] = {"id": source["id"], "baseline": baseline, "new": []}
    resp = crawler.fetch(source["page"])
    if not resp.ok:
        why = resp.error or f"HTTP {resp.status}"
        out.update(status="blocked" if ("blocks" in why or "disallows" in why) else "error", detail=why)
        return out
    page = resp.text()
    found = editions(source, page, resp.url or source["page"])
    if not found:
        out.update(status="no-editions", detail="the edition pattern matched nothing; the page may have changed")
        return out
    latest = newest(source, found)
    out.update(status="ok", editions=len(found), latest=latest["key"], latest_url=latest["url"],
               found=found)
    if baseline:
        return out
    fresh = [e for e in found if e["key"] not in seen]
    if len(fresh) > MAX_NEW:
        out.update(status="suspect", detail=f"{len(fresh)} new editions at once: check the pattern",
                   new=[{**e, "files": []} for e in fresh])
        return out
    for edition in fresh:
        files = edition_files(source, edition, page, resp.url or source["page"])
        if download_dir is None:
            rows = [{"url": u} for u in files]
        else:
            folder = download_dir / source["id"] / re.sub(r"[^A-Za-z0-9._-]+", "_", edition["key"])[:80]
            rows = [download(crawler, u, folder) for u in files]
        out["new"].append({**edition, "files": rows})
    return out


def run(watchlist: Mapping[str, Any], state: Mapping[str, Any], crawler: Crawler, *, today: date,
        download_dir: Optional[Path] = None, only: Sequence[str] = ()) -> tuple[list[dict], dict, list[dict]]:
    """Check every source; return (results, the new state, inbox rows for new editions)."""
    results, inbox = [], []
    new_state = {k: dict(v) for k, v in state.items()}
    for source in watchlist.get("sources") or []:
        if only and source["id"] not in only:
            continue
        before = new_state.get(source["id"]) or {}
        seen = dict(before.get("seen") or {})
        # Baseline until a check has actually seen editions: a source that was blocked or broken on
        # its first check must not later count its whole back catalogue as new.
        result = check(source, seen, crawler, baseline=not seen, download_dir=download_dir)
        for edition in result.pop("found", []):
            seen.setdefault(edition["key"], {"url": edition["url"], "first_seen": today.isoformat()})
        for edition in result["new"]:
            seen.setdefault(edition["key"], {"url": edition["url"], "first_seen": today.isoformat()})
            inbox.append({"found": today.isoformat(), "source": source["id"], "publisher": source["publisher"],
                          "title": source["title"], "kind": source["kind"], "edition": edition["key"],
                          "url": edition["url"], "files": edition["files"],
                          "status": "check" if result["status"] == "suspect" else "new"})
        entry = {"seen": seen, "checked": today.isoformat(), "status": result["status"]}
        if result.get("latest"):
            entry["latest"] = result["latest"]
        if result.get("detail"):
            entry["detail"] = result["detail"]
        new_state[source["id"]] = entry
        results.append(result)
    return results, new_state, inbox


def validate(watchlist: Any) -> list[str]:
    """Problems with a watch list, as readable lines (empty when it is fine)."""
    if not isinstance(watchlist, Mapping) or not isinstance(watchlist.get("sources"), list):
        return ['watchlist.json must be {"sources": [...]}']
    errors, ids = [], set()
    for n, source in enumerate(watchlist["sources"]):
        where = f"sources[{n}]"
        if not isinstance(source, Mapping):
            errors.append(f"{where}: not an object")
            continue
        sid = source.get("id")
        where = f"{where} ({sid})"
        if not isinstance(sid, str) or not ID.match(sid):
            errors.append(f"{where}: id must be lower-case words joined by hyphens")
        elif sid in ids:
            errors.append(f"{where}: duplicate id")
        ids.add(sid)
        for field in ("publisher", "title", "cadence", "licence"):
            if not str(source.get(field) or "").strip():
                errors.append(f"{where}: missing {field}")
        if not str(source.get("page") or "").startswith("https://"):
            errors.append(f"{where}: page must be an https URL")
        if source.get("kind") not in KINDS:
            errors.append(f"{where}: kind must be one of {', '.join(KINDS)}")
        if source.get("newest", "first") not in ("first", "last", "max"):
            errors.append(f"{where}: newest must be first, last or max")
        rule = source.get("edition")
        if not isinstance(rule, Mapping) or len([k for k in ("link", "text") if k in rule]) != 1:
            errors.append(f"{where}: edition needs exactly one of link or text")
            rule = {}
        if "text" in rule and not source.get("files"):
            errors.append(f"{where}: a text edition needs files to download")
        for pattern in [*(v for k, v in rule.items() if k in ("link", "text")), *(source.get("files") or [])]:
            try:
                re.compile(pattern)
            except (re.error, TypeError) as exc:
                errors.append(f"{where}: bad pattern {pattern!r}: {exc}")
    return errors


def summary(results: Sequence[Mapping[str, Any]]) -> str:
    """One Markdown table for a run log or a job summary."""
    lines = ["| Source | Status | Latest edition | New |", "|---|---|---|---|"]
    for r in results:
        status = r["status"] + (" (baseline)" if r.get("baseline") and r["status"] == "ok" else "")
        latest = r.get("latest") or r.get("detail") or ""
        lines.append(f"| {r['id']} | {status} | {latest} | {len(r.get('new') or [])} |")
    return "\n".join(lines) + "\n"


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
