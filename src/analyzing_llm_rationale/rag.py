"""Lightweight RAG helpers: embeddings, chunking, cosine similarity.

Embeddings use the local sentence-transformers model (already in the serve
image); it is lazy-loaded so it only costs memory/startup when RAG is used.
Persistence and per-user storage live in the server (Cloud Datastore), keeping
this module a pure, testable helper layer.
"""
from __future__ import annotations

import os
from typing import List

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# In production the model is mounted from a GCS volume; EMBED_MODEL_PATH points
# at that mount (a local dir of real files) so loading is offline and reliable.
# When unset (local/dev/tests), fall back to the model name (HF resolves it).
EMBED_MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", EMBED_MODEL_NAME)

_embedder = None
_embedder_loaded = False


def _get_embedder():
    global _embedder, _embedder_loaded
    if _embedder_loaded:
        return _embedder
    _embedder_loaded = True
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL_PATH, device="cpu")
    except Exception:
        _embedder = None
    return _embedder


def set_embedder(fn) -> None:
    """Inject an embedder (used by tests) — any object with .encode(list)->vectors,
    or None to disable."""
    global _embedder, _embedder_loaded
    _embedder = fn
    _embedder_loaded = True


def embeddings_available() -> bool:
    return _get_embedder() is not None


def is_loaded() -> bool:
    """True only if the embedder is already in memory — does NOT trigger a load.
    Lets callers do cheap opportunistic indexing without paying a cold start."""
    return _embedder_loaded and _embedder is not None


def embed(texts: List[str]) -> List[List[float]]:
    """Return a normalized embedding per text, or [] per text if unavailable."""
    if not texts:
        return []
    model = _get_embedder()
    if model is None:
        return [[] for _ in texts]
    vectors = model.encode(list(texts), normalize_embeddings=True)
    return [[float(x) for x in vec] for vec in vectors]


def embed_one(text: str) -> List[float]:
    out = embed([text])
    return out[0] if out else []


def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity. Inputs are normalized by embed(), so this is a dot
    product, but normalize defensively in case callers pass raw vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def chunk_text(text: str, size: int = 700, overlap: int = 100, max_chunks: int = 60) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    text = " ".join((text or "").split())
    if not text:
        return []
    chunks: List[str] = []
    step = max(1, size - overlap)
    i = 0
    while i < len(text) and len(chunks) < max_chunks:
        chunks.append(text[i:i + size])
        i += step
    return chunks


def top_k(query_vec: List[float], items: List[dict], k: int = 5,
          min_score: float = 0.2) -> List[dict]:
    """Rank items (each with an 'embedding') by cosine to query_vec; return top k
    above min_score, each annotated with its 'score'."""
    if not query_vec:
        return []
    scored = []
    for it in items:
        emb = it.get("embedding") or []
        s = cosine(query_vec, emb)
        if s >= min_score:
            out = {key: val for key, val in it.items() if key != "embedding"}
            out["score"] = round(s, 4)
            scored.append(out)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]
