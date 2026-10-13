"""Approved-source enrichment; no cookies, redirects, proxies or paywall bypass."""
from __future__ import annotations

import http.client
import ipaddress
import logging
import re
import socket
import ssl
import threading
from datetime import datetime, timezone
from html.parser import HTMLParser
from time import perf_counter
from urllib.parse import urlsplit

from opentelemetry import metrics, trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)
retrievals = meter.create_counter("metaculus.sources.retrieved", unit="1")
duration = meter.create_histogram("metaculus.sources.duration", unit="s")
PUBLIC_HOSTS = frozenset({
    "news.bloomberglaw.com", "www.judiciary.senate.gov", "www.senate.gov",
    "www.uscourts.gov", "www.congress.gov", "www.whitehouse.gov",
    "www.bls.gov", "www.bea.gov", "www.federalreserve.gov",
    "www.census.gov", "www.noaa.gov", "www.nasa.gov",
    "apnews.com", "www.reuters.com", "www.bbc.com",
})
MAX_BYTES = 512000
HIDDEN_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "iframe", "object"}


class _PageText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = []

    def handle_starttag(self, tag, attrs):
        if tag in HIDDEN_TAGS:
            self.hidden.append(tag)
        if tag in {"p", "br", "li", "tr", "h1", "h2", "h3"} and not self.hidden:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.hidden and tag == self.hidden[-1]:
            self.hidden.pop()

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _approved_url(url):
    try:
        parsed = urlsplit(url)
        return (parsed.scheme == "https" and parsed.hostname in PUBLIC_HOSTS
                and parsed.port in (None, 443) and not parsed.username and not parsed.password)
    except ValueError:
        return False


def _expire_socket(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass  # Concurrent normal close already ended the request.


def _fetch_page(url, deadline):
    """Pin public DNS results and verify TLS for the original hostname.

    OS DNS resolution is not interruptible; HTTP headers/body are interrupted
    at the remaining deadline. read1 does not wait for a full-sized chunk.
    """
    parsed = urlsplit(url)
    addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses):
        raise ValueError("Source DNS is not exclusively public")
    remaining = deadline - perf_counter()
    if remaining <= 0:
        raise TimeoutError("Source deadline expired")
    connection = http.client.HTTPSConnection(parsed.hostname, timeout=min(5, remaining))
    raw_socket = socket.create_connection((addresses[0][4][0], 443), timeout=min(5, remaining))
    timer = None
    try:
        tls_socket = ssl.create_default_context().wrap_socket(raw_socket, server_hostname=parsed.hostname)
        connection.sock = tls_socket
        timer = threading.Timer(max(0, deadline - perf_counter()), _expire_socket, args=(tls_socket,))
        timer.daemon = True
        timer.start()
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        connection.request("GET", path, headers={
            "User-Agent": "Foresea-Metaculus-Research/1.0", "Accept-Encoding": "identity",
        })
        response = connection.getresponse()
        if response.status != 200 or response.headers.get_content_type() != "text/html":
            return None
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            return None
        content = bytearray()
        while True:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                raise TimeoutError("Source deadline expired")
            tls_socket.settimeout(min(5, remaining))
            chunk = response.read1(min(16384, MAX_BYTES + 1 - len(content)))
            if not chunk:
                return content.decode(response.headers.get_content_charset() or "utf-8", errors="replace")
            content.extend(chunk)
            if len(content) > MAX_BYTES:
                return None
    finally:
        if timer is not None:
            timer.cancel()
        connection.close()
        raw_socket.close()


@tracer.start_as_current_span("metaculus.sources.enrich")
def enrich_evidence(post, articles):
    """Supplement snippets with up to four approved linked pages.

    Page text is not asserted relevant/current merely because it was fetched.
    Google RSS redirects and blocked/paywalled publishers remain snippets.
    """
    started = perf_counter()
    output = [dict(article, content_kind="snippet") for article in articles[:12]]
    question = post.get("question") or {}
    context = "\n".join(str(value or "") for value in (
        question.get("resolution_criteria"), question.get("fine_print"),
        post.get("description"), question.get("description"), post.get("title"), question.get("title"),
    ))
    urls = re.findall(r'https://[^\s<>\]"\)]+', context)
    urls += [str(article.get("url") or "") for article in output]
    urls = list(dict.fromkeys(url.rstrip(".,;") for url in urls if _approved_url(url)))[:4]
    for url in urls:
        outcome = "unavailable"
        try:
            page = _fetch_page(url, started + 22)
            if page is not None:
                parser = _PageText()
                parser.feed(page)
                text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
                if len(text) < 500:
                    continue
                existing = next((item for item in output if item.get("url") == url), None)
                item = existing if existing is not None else {"url": url, "title": urlsplit(url).hostname}
                item["original_snippet"] = item.get("summary") or item.get("text")
                item["original_publication_metadata"] = {
                    key: item.pop(key) for key in ("publish_date", "published_at", "publishedAt", "date") if key in item
                }
                item.update(text=text[:12000], summary=text[:6000], content_kind="retrieved_page_text",
                            retrieved_at=datetime.now(timezone.utc).isoformat())
                if existing is None:
                    output.insert(0, item)
                outcome = "retrieved"
        except (OSError, http.client.HTTPException, ValueError, LookupError):
            logger.warning("Linked Metaculus source unavailable", extra={"outcome": "unavailable"})
        finally:
            retrievals.add(1, {"outcome": outcome})
        if perf_counter() - started > 22:
            break
    trace.get_current_span().set_attribute("sources.enriched", sum(
        item["content_kind"] == "retrieved_page_text" for item in output))
    duration.record(perf_counter() - started)
    return output[:12]
