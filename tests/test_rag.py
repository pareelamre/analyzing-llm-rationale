from __future__ import annotations

import math
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("ANALYTICS_DB", str(Path(tempfile.gettempdir()) / "foresea_rag_test.duckdb"))
os.environ["ENABLE_OTEL"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from analyzing_llm_rationale import rag  # noqa: E402
from analyzing_llm_rationale.server import _issue_session, _state, app  # noqa: E402


class FakeEmbedder:
    """Deterministic bag-of-words embedder so cosine is meaningful in tests."""
    VOCAB = ["fed", "rate", "cut", "cpi", "inflation", "spacex", "launch", "market"]

    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            toks = set(re.findall(r"[a-z]+", t.lower()))
            v = [1.0 if w in toks else 0.0 for w in self.VOCAB]
            if normalize_embeddings:
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                v = [x / n for x in v]
            out.append(v)
        return out


class RagHelperTests(unittest.TestCase):
    def test_chunk_text(self):
        chunks = rag.chunk_text("word " * 500, size=200, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 200 for c in chunks))
        self.assertEqual(rag.chunk_text(""), [])

    def test_cosine(self):
        self.assertAlmostEqual(rag.cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(rag.cosine([1, 0], [0, 1]), 0.0)

    def test_rerank_uses_injected_model_and_falls_back(self):
        try:
            class FakeCE:
                def predict(self, pairs):
                    return [len(d) for _q, d in pairs]
            rag.set_reranker(FakeCE())
            self.assertEqual(rag.rerank("q", ["a", "abc"]), [1.0, 3.0])
            rag.set_reranker(None)
            self.assertIsNone(rag.rerank("q", ["a"]))  # unavailable -> None (hybrid fallback)
            self.assertEqual(rag.rerank("q", []), [])
        finally:
            rag.set_reranker(None)
        self.assertEqual(rag.cosine([], [1]), 0.0)

    def test_top_k_ranks_and_filters(self):
        items = [
            {"text": "a", "embedding": [1.0, 0.0]},
            {"text": "b", "embedding": [0.0, 1.0]},
        ]
        hits = rag.top_k([1.0, 0.0], items, k=5, min_score=0.5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["text"], "a")
        self.assertNotIn("embedding", hits[0])


class RagEndpointTests(unittest.TestCase):
    def setUp(self):
        rag.set_embedder(FakeEmbedder())
        _state.clear()
        _state["rag"] = {}
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer " + _issue_session("u1", "u1@example.com", "U", "")}

    def tearDown(self):
        rag.set_embedder(None)
        _state.clear()

    def test_ingest_then_search(self):
        ingest = self.client.post(
            "/rag/ingest",
            json={"text": "The Fed will cut the rate amid falling inflation.", "title": "Fed note"},
            headers=self.headers,
        )
        self.assertEqual(ingest.status_code, 200)
        self.assertGreaterEqual(ingest.json()["chunks_added"], 1)

        search = self.client.get("/rag/search", params={"q": "fed rate cut"}, headers=self.headers)
        self.assertEqual(search.status_code, 200)
        results = search.json()
        self.assertTrue(results)
        self.assertIn("Fed", results[0]["text"])

        docs = self.client.get("/rag/documents", headers=self.headers).json()
        self.assertEqual(len(docs["documents"]), 1)

    def test_ingest_requires_session(self):
        self.assertEqual(self.client.post("/rag/ingest", json={"text": "x y z"}).status_code, 401)

    def test_delete_clears_namespace(self):
        self.client.post("/rag/ingest", json={"text": "spacex launch market"}, headers=self.headers)
        removed = self.client.delete("/rag/documents", headers=self.headers).json()
        self.assertGreaterEqual(removed["removed"], 1)
        self.assertEqual(self.client.get("/rag/search", params={"q": "spacex"}, headers=self.headers).json(), [])


if __name__ == "__main__":
    unittest.main()
