"""The knowledge watch: a polite crawler that spots new report editions and downloads them.

Every site here is a fake served from a dict, so the tests never touch the network.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyzing_llm_rationale import knowledge_watch as watch  # noqa: E402

PAGE = "https://stats.example.org/outlook/"
PDF = b"%PDF-1.7 fake"


class FakeWeb:
    def __init__(self, pages: dict):
        self.pages, self.calls = dict(pages), []

    def __call__(self, url, *, max_bytes=watch.PAGE_BYTES):
        self.calls.append(url)
        hit = self.pages.get(url)
        if hit is None:
            return watch.Response(404, b"", "", url)
        if isinstance(hit, int):
            return watch.Response(hit, b"", "", url)
        body, ctype = hit if isinstance(hit, tuple) else (hit, "text/html")
        return watch.Response(200, body.encode() if isinstance(body, str) else body, ctype, url)


def crawler(pages: dict, **kw) -> tuple[watch.Crawler, FakeWeb]:
    web = FakeWeb(pages)
    return watch.Crawler(web, pause=0, sleep=lambda s: None, **kw), web


def listing(*editions: str) -> str:
    rows = "".join(f'<li><a href="files/report-{e}.pdf">Report {e}</a></li>' for e in editions)
    return f"<html><body><a href='/about'>About</a><ul>{rows}</ul></body></html>"


SOURCE = {"id": "stats-outlook", "publisher": "Stats Office", "title": "Outlook", "page": PAGE,
          "cadence": "monthly", "licence": "CC BY 4.0", "kind": "table",
          "edition": {"link": r"/files/report-(\d{4}-\d{2})\.pdf$"}, "newest": "max"}


class RobotsTests(unittest.TestCase):
    def test_disallowed_paths_are_refused(self):
        c, web = crawler({"https://stats.example.org/robots.txt": "User-agent: *\nDisallow: /outlook/\n",
                          PAGE: listing("2026-10")})
        resp = c.fetch(PAGE)
        self.assertFalse(resp.ok)
        self.assertIn("robots.txt disallows", resp.error)
        self.assertNotIn(PAGE, web.calls)

    def test_a_forbidden_robots_file_means_the_site_blocks_us(self):
        c, _ = crawler({"https://stats.example.org/robots.txt": 403, PAGE: listing("2026-10")})
        self.assertIn("blocks this crawler", c.fetch(PAGE).error)

    def test_an_unavailable_robots_file_means_no_rules(self):
        for status in (404, 418):
            c, _ = crawler({"https://stats.example.org/robots.txt": status, PAGE: listing("2026-10")})
            self.assertTrue(c.fetch(PAGE).ok, status)

    def test_a_server_error_on_robots_means_try_later(self):
        c, web = crawler({"https://stats.example.org/robots.txt": 503, PAGE: listing("2026-10")})
        self.assertIn("could not read", c.fetch(PAGE).error)
        self.assertNotIn(PAGE, web.calls)

    def test_robots_is_read_once_per_host_and_requests_are_spaced(self):
        waits, now = [], [0.0]
        web = FakeWeb({PAGE: listing("2026-10"), PAGE + "a": "x"})
        c = watch.Crawler(web, pause=1.5, sleep=lambda s: (waits.append(s), now.__setitem__(0, now[0] + s)),
                          clock=lambda: now[0])
        c.fetch(PAGE)
        c.fetch(PAGE + "a")
        self.assertEqual(web.calls.count("https://stats.example.org/robots.txt"), 1)
        self.assertEqual(len(waits), 2)
        self.assertTrue(all(abs(w - 1.5) < 1e-9 for w in waits))


class PageTests(unittest.TestCase):
    def test_links_are_absolute_unescaped_and_unique(self):
        page = """<a href="a.pdf">x</a><a href='/b.pdf?x=1&amp;y=2'>y</a><a href=c.PDF>z</a>
                  <a href="a.pdf#p2">again</a><a href="mailto:x@y.z">m</a><a href="#top">t</a>"""
        self.assertEqual(watch.links(page, "https://s.example.org/dir/page.html"), [
            "https://s.example.org/dir/a.pdf", "https://s.example.org/b.pdf?x=1&y=2",
            "https://s.example.org/dir/c.PDF"])

    def test_link_editions_are_named_by_the_pattern_group(self):
        found = watch.editions(SOURCE, listing("2026-09", "2026-10"), PAGE)
        self.assertEqual([e["key"] for e in found], ["2026-09", "2026-10"])
        self.assertEqual(watch.newest(SOURCE, found)["key"], "2026-10")
        self.assertEqual(watch.newest({**SOURCE, "newest": "first"}, found)["key"], "2026-09")

    def test_a_text_edition_reads_the_release_date(self):
        source = {**SOURCE, "edition": {"text": r"(?<!Next )Release Date:\s*([A-Z][a-z]+ \d{1,2}, \d{4})"},
                  "files": [r"/full\.pdf$"]}
        page = "<p>Release Date: <b>November 10, 2026</b> | Next Release Date: December 8, 2026</p>" \
               "<a href='/full.pdf'>Full</a><a href='/other.pdf'>Other</a>"
        found = watch.editions(source, page, PAGE)
        self.assertEqual(found, [{"key": "November 10, 2026", "url": PAGE}])
        self.assertEqual(watch.edition_files(source, found[0], page, PAGE), ["https://stats.example.org/full.pdf"])


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inbox = Path(self.tmp.name) / "inbox"

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, pages, state, day=date(2026, 11, 17)):
        c, web = crawler(pages)
        results, new_state, rows = watch.run({"sources": [SOURCE]}, state, c, today=day, download_dir=self.inbox)
        return results[0], new_state, rows, web

    def test_the_first_check_is_a_baseline_and_downloads_nothing(self):
        result, state, rows, web = self._run({PAGE: listing("2026-09", "2026-10")}, {})
        self.assertTrue(result["baseline"])
        self.assertEqual(rows, [])
        self.assertEqual(set(state["stats-outlook"]["seen"]), {"2026-09", "2026-10"})
        self.assertEqual(state["stats-outlook"]["latest"], "2026-10")
        self.assertFalse(any(u.endswith(".pdf") for u in web.calls))

    def test_a_new_edition_goes_to_the_inbox_with_its_file(self):
        _, state, _, _ = self._run({PAGE: listing("2026-10")}, {})
        new_pdf = "https://stats.example.org/outlook/files/report-2026-11.pdf"
        result, state, rows, _ = self._run({PAGE: listing("2026-10", "2026-11"), new_pdf: (PDF, "application/pdf")},
                                           state, date(2026, 12, 2))
        self.assertEqual([r["edition"] for r in rows], ["2026-11"])
        row = rows[0]
        self.assertEqual((row["source"], row["status"], row["url"]), ("stats-outlook", "new", new_pdf))
        f = row["files"][0]
        self.assertEqual(f["sha256"], hashlib.sha256(PDF).hexdigest())
        self.assertEqual(Path(f["saved"]).read_bytes(), PDF)
        self.assertEqual(state["stats-outlook"]["seen"]["2026-11"]["first_seen"], "2026-12-02")
        # Nothing is new the next day.
        _, _, again, _ = self._run({PAGE: listing("2026-10", "2026-11")}, state)
        self.assertEqual(again, [])

    def test_a_flood_of_new_editions_is_flagged_not_downloaded(self):
        _, state, _, _ = self._run({PAGE: listing("2026-10")}, {})
        many = [f"2027-{m:02d}" for m in range(1, watch.MAX_NEW + 2)]
        result, _, rows, web = self._run({PAGE: listing("2026-10", *many)}, state)
        self.assertEqual(result["status"], "suspect")
        self.assertEqual({r["status"] for r in rows}, {"check"})
        self.assertFalse(any(u.endswith(".pdf") for u in web.calls))

    def test_a_blocked_first_check_still_baselines_later(self):
        blocked = {"https://stats.example.org/robots.txt": 403}
        result, state, _, _ = self._run(blocked, {})
        self.assertEqual(result["status"], "blocked")
        result, state, rows, _ = self._run({PAGE: listing("2026-09", "2026-10")}, state)
        self.assertTrue(result["baseline"])
        self.assertEqual(rows, [])

    def test_a_page_that_lost_its_editions_is_reported(self):
        _, state, _, _ = self._run({PAGE: listing("2026-10")}, {})
        result, state, _, _ = self._run({PAGE: "<html>redesigned</html>"}, state)
        self.assertEqual(result["status"], "no-editions")
        self.assertIn("2026-10", state["stats-outlook"]["seen"])


class ValidateTests(unittest.TestCase):
    def test_a_good_list_passes(self):
        self.assertEqual(watch.validate({"sources": [SOURCE]}), [])

    def test_mistakes_are_named(self):
        bad = [SOURCE, {**SOURCE, "page": "http://x.org", "kind": "blog", "edition": {"link": "("}},
               {**SOURCE, "id": "text-only", "edition": {"text": "Release"}}]
        errors = "\n".join(watch.validate({"sources": bad}))
        for expected in ("duplicate id", "https", "kind must be", "bad pattern", "needs files"):
            self.assertIn(expected, errors)


class CliTests(unittest.TestCase):
    def test_a_run_writes_state_and_inbox(self):
        import knowledge_watch as cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "watchlist.json").write_text(json.dumps({"sources": [SOURCE]}), encoding="utf-8")
            (root / "watch_state.json").write_text(json.dumps(
                {"stats-outlook": {"seen": {"2026-10": {"url": "x", "first_seen": "2026-11-01"}}}}), encoding="utf-8")
            new_pdf = "https://stats.example.org/outlook/files/report-2026-11.pdf"
            c, _ = crawler({PAGE: listing("2026-10", "2026-11"), new_pdf: (PDF, "application/pdf")})
            out = io.StringIO()
            with mock.patch.object(cli.watch, "Crawler", return_value=c), contextlib.redirect_stdout(out), \
                    mock.patch.dict("os.environ", {"GITHUB_ACTIONS": "", "GITHUB_STEP_SUMMARY": ""}):
                self.assertEqual(cli.main(["--kb", tmp]), 0)
            self.assertIn("new: stats-outlook 2026-11", out.getvalue())
            rows = [json.loads(line) for line in (root / "inbox.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["edition"], "2026-11")
            self.assertTrue((root / ".kb" / "inbox" / "stats-outlook" / "2026-11" / "report-2026-11.pdf").exists())
            self.assertIn("2026-11", json.loads((root / "watch_state.json").read_text())["stats-outlook"]["seen"])

    def test_validate_reports_problems(self):
        import knowledge_watch as cli
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "watchlist.json").write_text(json.dumps({"sources": [{"id": "Bad Id"}]}), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(cli.main(["--kb", tmp, "--validate"]), 1)
            self.assertIn("id must be", err.getvalue())


if __name__ == "__main__":
    unittest.main()
