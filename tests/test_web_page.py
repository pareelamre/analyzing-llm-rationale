from __future__ import annotations

import socket
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import web_page  # noqa: E402
from analyzing_llm_rationale.web_page import PageFetchError, find_links, read_page  # noqa: E402

_DNS = {
    "news.example": "93.184.216.34",
    "loopback.example": "127.0.0.1",
    "vpc.example": "10.0.0.5",
    "metadata.example": "169.254.169.254",
    "tailnet.example": "100.100.1.1",
}


def _fake_getaddrinfo(host, port, *args, **kwargs):
    if host not in _DNS:
        raise socket.gaierror("unknown host")
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (_DNS[host], port))]


def _canned_server(responses):
    """create_connection stand-in: each call gets the next canned HTTP response."""
    calls = []

    def create_connection(address, timeout=None):
        calls.append(address)
        client, server = socket.socketpair()
        raw = responses.pop(0)

        def serve():
            try:
                server.recv(65536)
                server.sendall(raw)
            except OSError:
                pass  # The reader hung up early (e.g. the page was too large).
            finally:
                server.close()

        threading.Thread(target=serve, daemon=True).start()
        return client

    return create_connection, calls


class FindLinksTests(unittest.TestCase):
    def test_strips_trailing_punctuation_and_dedupes(self):
        self.assertEqual(
            find_links("see https://news.example/a), and (https://news.example/b). again https://news.example/a"),
            ["https://news.example/a", "https://news.example/b"],
        )


@patch.object(web_page.socket, "getaddrinfo", _fake_getaddrinfo)
class ReadPageTests(unittest.TestCase):
    def test_refuses_hosts_that_resolve_to_non_public_addresses(self):
        for host in ("loopback.example", "vpc.example", "metadata.example", "tailnet.example"):
            with self.subTest(host=host), patch.object(web_page.socket, "create_connection") as connect:
                with self.assertRaisesRegex(PageFetchError, "Only public"):
                    read_page(f"http://{host}/")
                connect.assert_not_called()

    def test_refuses_literal_private_ips_odd_ports_and_credentials(self):
        for url in ("http://127.0.0.1/", "http://[::1]/", "https://news.example:8443/", "https://user:pw@news.example/"):
            with self.subTest(url=url), self.assertRaises(PageFetchError):
                read_page(url)

    def test_redirect_to_a_private_host_is_refused_before_connecting(self):
        create_connection, calls = _canned_server([
            b"HTTP/1.1 302 Found\r\nLocation: http://metadata.example/computeMetadata/v1/\r\nContent-Length: 0\r\n\r\n",
        ])
        with patch.object(web_page.socket, "create_connection", create_connection):
            with self.assertRaisesRegex(PageFetchError, "Only public"):
                read_page("http://news.example/story")
        # One connection, pinned to the address that passed the check.
        self.assertEqual(calls, [("93.184.216.34", 80)])

    def test_reads_title_and_text_of_a_public_page(self):
        html = (
            b"<html><head><title>Fed holds rates</title></head><body><nav>Menu</nav>"
            b"<article><p>The Federal Reserve held rates steady on Wednesday, citing sticky inflation.</p>"
            b"<p>Officials signalled one more cut is possible before the end of the year.</p></article></body></html>"
        )
        create_connection, _ = _canned_server([
            b"HTTP/1.1 301 Moved\r\nLocation: /story-2\r\nContent-Length: 0\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\n"
            + f"Content-Length: {len(html)}\r\n\r\n".encode() + html,
        ])
        with patch.object(web_page.socket, "create_connection", create_connection):
            page = read_page("http://news.example/story")

        self.assertEqual(page["url"], "http://news.example/story-2")
        self.assertEqual(page["title"], "Fed holds rates")
        self.assertIn("held rates steady", page["text"])

    def test_drops_the_metadata_date_of_home_and_section_pages(self):
        ok = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 13\r\n\r\n<p>Story</p>\n"
        dated = {"title": "t", "text": "Story", "publish_date": "2026-01-28"}
        for path, expected in (("/news", None), ("/news/articles/fed-holds", "2026-01-28")):
            create_connection, _ = _canned_server([ok])
            with self.subTest(path=path), patch.object(web_page.socket, "create_connection", create_connection), \
                    patch.object(web_page, "_html_text", return_value=dated):
                self.assertEqual(read_page(f"http://news.example{path}")["publish_date"], expected)

    def test_rejects_non_text_pages(self):
        create_connection, _ = _canned_server([
            b"HTTP/1.1 200 OK\r\nContent-Type: application/pdf\r\nContent-Length: 4\r\n\r\n%PDF",
        ])
        with patch.object(web_page.socket, "create_connection", create_connection):
            with self.assertRaisesRegex(PageFetchError, "application/pdf"):
                read_page("http://news.example/report.pdf")

    def test_rejects_oversized_pages(self):
        create_connection, _ = _canned_server([
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n" + b"x" * (web_page.MAX_BYTES + 10),
        ])
        with patch.object(web_page.socket, "create_connection", create_connection):
            with self.assertRaisesRegex(PageFetchError, "too large"):
                read_page("http://news.example/huge.txt")


if __name__ == "__main__":
    unittest.main()
