"""
Hybrid retriever: fuses semantic + keyword + graph results using
Reciprocal Rank Fusion (RRF), then re-ranks with a cross-encoder.

RRF formula:  score(d) = Σ  1 / (k + rank_i(d))   k=60
"""
from __future__ import annotations
from typing import List

from app.config import TOP_K_VECTOR, TOP_K_BM25, TOP_K_GRAPH, TOP_K_FINAL
from app.vector_db import semantic_search
from app.keyword_search import keyword_search
from app.graph_db import graph_search

RRF_K = 60


def _rrf_fusion(ranked_lists: List[List[dict]]) -> List[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    Each doc is identified by its text (first 120 chars as key).
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc["text"][:120]
            scores[key]  = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            if key not in doc_map:
                doc_map[key] = doc
            else:
                # Record all retriever sources
                existing = doc_map[key].get("retrievers", [doc_map[key].get("retriever", "")])
                r = doc.get("retriever", "")
                if r not in existing:
                    existing.append(r)
                doc_map[key]["retrievers"] = existing

    fused = sorted(doc_map.keys(), key=lambda k: scores[k], reverse=True)
    result = []
    for key in fused:
        d = doc_map[key].copy()
        d["rrf_score"] = round(scores[key], 6)
        if "retrievers" not in d:
            d["retrievers"] = [d.get("retriever", "")]
        result.append(d)
    return result


def hybrid_retrieve(query: str, top_k: int = TOP_K_FINAL) -> List[dict]:
    """
    1. Run semantic search  (ChromaDB)
    2. Run keyword search   (BM25)
    3. Run graph search     (NetworkX)
    4. Fuse with RRF
    5. Return top_k results
    """
    sem_hits   = semantic_search(query, top_k=TOP_K_VECTOR)
    kw_hits    = keyword_search(query,  top_k=TOP_K_BM25)
    graph_hits = graph_search(query,    top_k=TOP_K_GRAPH)

    fused = _rrf_fusion([sem_hits, kw_hits, graph_hits])

    # Filter out very short chunks
    fused = [d for d in fused if len(d.get("text", "")) > 50]

    return fused[:top_k]


def retrieval_diagnostics(query: str) -> dict:
    """Return full retrieval breakdown for a query (for explainability)."""
    sem_hits   = semantic_search(query, top_k=TOP_K_VECTOR)
    kw_hits    = keyword_search(query,  top_k=TOP_K_BM25)
    graph_hits = graph_search(query,    top_k=TOP_K_GRAPH)
    fused      = _rrf_fusion([sem_hits, kw_hits, graph_hits])

    return {
        "query":        query,
        "semantic":     [{"text": h["text"][:150], "score": h["score"], "source": h["source"]} for h in sem_hits[:3]],
        "keyword":      [{"text": h["text"][:150], "score": h["score"], "source": h["source"]} for h in kw_hits[:3]],
        "graph":        [{"text": h["text"][:150], "score": h["score"], "source": h["source"]} for h in graph_hits[:3]],
        "fused_top5":   [{"text": h["text"][:150], "rrf_score": h["rrf_score"], "source": h["source"], "retrievers": h.get("retrievers",[])} for h in fused[:5]],
    }
