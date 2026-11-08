"""Read a public web page a user linked, without letting the link reach private hosts.

Every hop (including redirects) must resolve only to globally routable
addresses, and the connection is pinned to the address that was checked, so a
DNS answer that changes between the check and the request (rebinding) cannot
redirect it to localhost, the VPC, or the cloud metadata server.
"""
from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
import zlib
from time import perf_counter
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlsplit

MAX_BYTES = 2_000_000
MAX_REDIRECTS = 3
TEXT_CHARS = 20000
_TEXT_TYPES = {"text/html", "application/xhtml+xml", "text/plain"}
_USER_AGENT = "Mozilla/5.0 (compatible; Foresea/1.0; +https://foresea.ink)"
_LINK_RE = re.compile(r"https?://[^\s<>\"']+")


class PageFetchError(ValueError):
    """The page is not public, not text, too large, or did not answer in time."""


def find_links(text: str) -> List[str]:
    """http(s) links in `text`, in order, without trailing punctuation."""
    return list(dict.fromkeys(m.rstrip(").,;:!?]") for m in _LINK_RE.findall(text or "")))


def _public_address(host: str, port: int) -> str:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise PageFetchError(f"Could not resolve {host}.") from exc
    addresses = list(dict.fromkeys(info[4][0] for info in infos))
    if not addresses or any(not ipaddress.ip_address(a.split("%")[0]).is_global for a in addresses):
        raise PageFetchError("Only public web pages can be read.")
    return addresses[0]


def _get_once(url: str, deadline: float) -> tuple[int, Optional[str], str, str]:
    """One GET -> (status, redirect location, content type, text)."""
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise PageFetchError("URL must start with http:// or https://")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise PageFetchError("Only public web pages can be read.") from exc
    if parsed.username or parsed.password or port not in (80, 443):
        raise PageFetchError("Only public web pages can be read.")
    host = parsed.hostname
    address = _public_address(host, port)

    def remaining() -> float:
        left = deadline - perf_counter()
        if left <= 0:
            raise PageFetchError("The page took too long to answer.")
        return min(5.0, left)

    https = parsed.scheme == "https"
    connection = (http.client.HTTPSConnection if https else http.client.HTTPConnection)(host, port)
    try:
        sock = socket.create_connection((address, port), timeout=remaining())
        connection.sock = sock
        if https:
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
            connection.sock = sock
        path = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")
        connection.request("GET", path, headers={
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8",
            "Accept-Encoding": "gzip, identity",
            "Connection": "close",
        })
        response = connection.getresponse()
        if response.status in (301, 302, 303, 307, 308):
            return response.status, response.getheader("Location"), "", b""
        content_type = response.headers.get_content_type()
        if response.status != 200 or content_type not in _TEXT_TYPES:
            return response.status, None, content_type, b""
        encoding = (response.getheader("Content-Encoding") or "identity").lower()
        if encoding not in ("identity", "gzip"):
            raise PageFetchError("The page used an unsupported encoding.")
        body = bytearray()
        while True:
            # Each read is bounded by the socket's 5 s timeout; the overall
            # deadline is checked between reads. (A "Connection: close" reply
            # hands the socket to the response, so it can't be re-timed here.)
            remaining()
            chunk = response.read1(65536)
            if not chunk:
                break
            body.extend(chunk)
            if len(body) > MAX_BYTES:
                raise PageFetchError("The page is too large to read.")
        if encoding == "gzip":
            inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
            body = bytearray(inflater.decompress(bytes(body), MAX_BYTES + 1))
            if len(body) > MAX_BYTES:
                raise PageFetchError("The page is too large to read.")
        try:
            text = bytes(body).decode(response.headers.get_content_charset() or "utf-8", errors="replace")
        except LookupError:
            text = bytes(body).decode("utf-8", errors="replace")
        return 200, None, content_type, text
    except (OSError, http.client.HTTPException) as exc:
        raise PageFetchError(f"Could not read the page: {exc}") from exc
    finally:
        connection.close()


def _html_text(html: str, url: str) -> Dict[str, Any]:
    try:
        import trafilatura

        doc = trafilatura.bare_extraction(html, url=url, with_metadata=True)
        if doc is not None and not isinstance(doc, dict):
            doc = doc.as_dict()
        if doc and (doc.get("text") or "").strip():
            return {"title": doc.get("title"), "text": doc["text"], "publish_date": doc.get("date"),
                    "source": doc.get("sitename")}
    except ImportError:
        pass
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else None
    return {"title": title, "text": re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))}


def read_page(url: str, timeout_s: float = 12.0) -> Dict[str, Any]:
    """Fetch a public page and return `{title, text, source, url, publish_date}`."""
    deadline = perf_counter() + timeout_s
    current = url.strip()
    for _ in range(MAX_REDIRECTS + 1):
        status, location, content_type, text = _get_once(current, deadline)
        if location:
            current = urljoin(current, location)
            continue
        if status != 200:
            raise PageFetchError(f"The page returned HTTP {status}.")
        if content_type not in _TEXT_TYPES:
            raise PageFetchError(f"The page is {content_type or 'not text'}, not a readable web page.")
        page = _html_text(text, current) if content_type != "text/plain" else {"title": None, "text": text}
        parts = urlsplit(current)
        host = parts.hostname or ""
        # Home and section pages (bbc.com/news) carry a stale site-wide date in
        # their metadata, which the model would read as the stories' date.
        is_index_page = len([p for p in parts.path.split("/") if p]) <= 1
        return {
            "title": (page.get("title") or host)[:500],
            "text": (page.get("text") or "").strip()[:TEXT_CHARS],
            "source": (page.get("source") or host)[:200],
            "url": current,
            "publish_date": None if is_index_page else page.get("publish_date"),
        }
    raise PageFetchError("The page redirected too many times.")
