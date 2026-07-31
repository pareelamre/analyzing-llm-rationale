from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyzing_llm_rationale.entity_tagger import tag_question  # noqa: E402


class EntityTaggerTests(unittest.TestCase):
    def test_short_crypto_symbol_does_not_match_inside_words(self):
        tagged = tag_question("Will Netherlands win on 2026-06-20?")
        self.assertEqual(tagged["domain"], "other")

    def test_crypto_symbol_still_matches_as_token(self):
        tagged = tag_question("Will ETH be above $4,000 on August 1?")
        self.assertEqual(tagged["domain"], "crypto")

    def test_category_fallback_still_applies_when_keywords_do_not_match(self):
        tagged = tag_question("Will Netherlands win on 2026-06-20?", category="Sports")
        self.assertEqual(tagged["domain"], "sports")

    def test_prefix_keyword_still_matches_domain_stems(self):
        tagged = tag_question("Will geopolitical risks rise in 2027?")
        self.assertEqual(tagged["domain"], "geopolitics")


if __name__ == "__main__":
    unittest.main()
