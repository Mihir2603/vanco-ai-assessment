"""
BM25 keyword search index using rank-bm25.
Provides exact/term-based retrieval to complement semantic search.
"""
from __future__ import annotations
import pickle
import re
from typing import List

from rank_bm25 import BM25Okapi

from app.config import BM25_FILE, TOP_K_BM25
from app.ingestion import Chunk


_bm25:   BM25Okapi | None = None
_chunks: List[Chunk] | None = None


def _tokenize(text: str) -> List[str]:
    """Lowercase + split on non-alphanumeric, keep physics symbols."""
    text = text.lower()
    # Preserve common physics notation like v², F=ma etc.
    text = re.sub(r"[^\w\s²³⁻⁺αβγδθλμω∫∑√π∞=<>]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def build_bm25_index(chunks: List[Chunk]) -> None:
    global _bm25, _chunks
    _chunks = chunks
    corpus  = [_tokenize(c.text) for c in chunks]
    _bm25   = BM25Okapi(corpus)
    with open(BM25_FILE, "wb") as f:
        pickle.dump({"bm25": _bm25, "chunks": _chunks}, f)
    print(f"  BM25 index built over {len(chunks)} chunks.")


def _load_if_needed() -> None:
    global _bm25, _chunks
    if _bm25 is None:
        with open(BM25_FILE, "rb") as f:
            data    = pickle.load(f)
            _bm25   = data["bm25"]
            _chunks = data["chunks"]


def keyword_search(query: str, top_k: int = TOP_K_BM25) -> List[dict]:
    """Return top-k chunks by BM25 score."""
    _load_if_needed()
    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    hits = []
    for idx in top_idx:
        if scores[idx] <= 0:
            continue
        c = _chunks[idx]
        hits.append({
            "text":      c.text,
            "source":    c.to_dict()["source"],
            "page":      c.page,
            "chapter":   c.chapter,
            "section":   c.section,
            "score":     round(float(scores[idx]), 4),
            "retriever": "keyword",
        })
    return hits
