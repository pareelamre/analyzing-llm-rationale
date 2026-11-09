"""Evidence sources added for Edge-board misses: beat feeds, market-echo filtering, and
the recent Google News pass (news_pipeline.BEAT_FEEDS and friends)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale import news_pipeline as np_mod  # noqa: E402
from analyzing_llm_rationale.news_pipeline import NewsPipeline  # noqa: E402

ARTWORK = "Will an artwork sell for $150 million by December 31?"
STAFFORD = "Will Matthew Stafford announce his retirement before the 2027 season?"


def _bare(sources):
    pipeline = NewsPipeline.__new__(NewsPipeline)
    pipeline._newsapi_key = None
    pipeline._fetch_sources = tuple(sources)
    return pipeline


class BeatRegistryTests(unittest.TestCase):
    def test_every_beat_names_its_miss_and_uses_https_feeds(self):
        for beat in np_mod.BEAT_FEEDS:
            self.assertTrue(beat["hints"], beat["beat"])
            self.assertTrue(beat["found"].startswith("20"), beat["beat"])
            for url in beat["feeds"]:
                self.assertTrue(url.startswith("https://"), url)

    def test_questions_route_to_their_beats(self):
        self.assertEqual([b["beat"] for b in np_mod._beats_for(ARTWORK)], ["art market"])
        self.assertIn("nfl", [b["beat"] for b in np_mod._beats_for(STAFFORD)])
        self.assertIn("celebrity", [b["beat"] for b in np_mod._beats_for(
            "Will Harry Styles and Zoe Kravitz be married before Jul 1, 2027?")])
        self.assertIn("art market", [b["beat"] for b in np_mod._beats_for("Will Sotheby's top lot beat its estimate?")])
        self.assertEqual(np_mod._beats_for("Will the Fed cut rates in December?"), [])

    def test_hints_match_whole_words_only(self):
        self.assertEqual(np_mod._beats_for("Will the startup raise a Series B?"), [
            b for b in np_mod.BEAT_FEEDS if b["beat"] == "film and tv"])  # "series" is a film/tv hint
        self.assertEqual(np_mod._beats_for("Will Smartphone sales rise?"), [])  # "art" inside a word


class MarketEchoTests(unittest.TestCase):
    def test_market_pages_and_mirrors_are_echo(self):
        for article in (
            {"url": "https://polymarket.com/event/x", "source": "polymarket.com"},
            {"url": "https://web3.bitget.com/en/x", "source": "web3.bitget.com"},
            {"url": "https://news.google.com/rss/articles/abc", "source": "Polymarket"},
            {"url": "https://www.trackerpoly.com/m/x", "source": "trackerpoly.com"},
        ):
            self.assertTrue(np_mod._is_market_echo(article), article)

    def test_news_about_a_market_is_not_echo(self):
        self.assertFalse(np_mod._is_market_echo({"url": "https://www.si.com/nfl/stafford-kalshi", "source": "si.com"}))
        self.assertFalse(np_mod._is_market_echo({"url": "https://news.artnet.com/market/x", "source": "Artnet News"}))


class FetchTests(unittest.TestCase):
    def test_echo_pages_are_dropped_and_the_recent_pass_runs_first(self):
        pipeline = _bare(("google-news",))
        queries = []

        def fake_google(query, limit):
            queries.append(query)
            return [{"title": f"Market page {len(queries)}", "url": "https://polymarket.com/event/x",
                     "source": "polymarket.com"},
                    {"title": f"Sotheby's sale {len(queries)}", "url": f"https://news.artnet.com/{len(queries)}",
                     "source": "Artnet News"}]

        pipeline._fetch_google_news = fake_google
        found = pipeline.fetch(ARTWORK, top_k=5)
        self.assertEqual(queries, ["artwork sell 150 million December 31 when:30d", ARTWORK])
        self.assertEqual([a["title"] for a in found], ["Sotheby's sale 1", "Sotheby's sale 2"])

    def test_beats_are_fetched_only_when_enabled_and_on_topic(self):
        beat_item = {"title": "Van Gogh leads Sotheby's sale", "url": "https://news.artnet.com/vg",
                     "source": "Artnet News", "source_channel": "beat"}
        on = _bare(("beat",))
        with mock.patch.object(on, "_fetch_beats", return_value=[beat_item]) as beats:
            self.assertEqual(on.fetch(ARTWORK), [beat_item])
            beats.assert_called_once_with(ARTWORK)
        with mock.patch.object(on, "_fetch_beats") as beats:
            self.assertEqual(on.fetch("Will the Fed cut rates in December?"), [])
            beats.assert_not_called()
        off = _bare(("web",))
        off._fetch_web = lambda query, limit: []
        with mock.patch.object(off, "_fetch_beats") as beats:
            off.fetch(ARTWORK)
            beats.assert_not_called()

    def test_fetch_beats_reads_every_entry_and_survives_a_dead_feed(self):
        calls = []

        def fake_get(url, timeout, headers):
            calls.append((url, timeout, headers["User-Agent"]))
            if url == "https://www.artnews.com/feed/":
                raise OSError("feed down")
            return SimpleNamespace(content=url.encode(), raise_for_status=lambda: None)

        def parse(content):
            return SimpleNamespace(entries=[{"title": f"{content.decode()} #{i}", "link": f"{content.decode()}/{i}"}
                                            for i in range(30)], feed={"title": "Feed"})

        with mock.patch.dict(sys.modules, {"requests": SimpleNamespace(get=fake_get),
                                           "feedparser": SimpleNamespace(parse=parse)}):
            found = _bare(("beat",))._fetch_beats(ARTWORK)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(timeout == np_mod.BEAT_TIMEOUT_S for _, timeout, _ in calls))
        self.assertTrue(all("foresea.ink" in agent for _, _, agent in calls))
        self.assertEqual(len(found), 60)  # two live feeds, all 30 entries each
        self.assertEqual({a["source_channel"] for a in found}, {"beat"})


if __name__ == "__main__":
    unittest.main()
