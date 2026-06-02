"""
ChromaDB vector store for semantic retrieval.
Uses sentence-transformers embeddings (local, no API key needed).
"""
from __future__ import annotations
from typing import List

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import INDEX_DIR, EMBED_MODEL, CHROMA_COLLECTION, TOP_K_VECTOR
from app.ingestion import Chunk


_client: chromadb.ClientAPI | None = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        _client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def build_vector_index(chunks: List[Chunk], batch_size: int = 256) -> None:
    """Embed all chunks and store in ChromaDB."""
    col = _get_collection()
    existing = set(col.get()["ids"])

    ids, docs, metas = [], [], []
    for chunk in chunks:
        if chunk.chunk_id in existing:
            continue
        ids.append(chunk.chunk_id)
        docs.append(chunk.text)
        metas.append({
            "page":       chunk.page,
            "chapter":    chunk.chapter,
            "section":    chunk.section,
            "subsection": chunk.subsection,
            "source":     chunk.to_dict()["source"],
        })
        if len(ids) == batch_size:
            col.add(ids=ids, documents=docs, metadatas=metas)
            ids, docs, metas = [], [], []

    if ids:
        col.add(ids=ids, documents=docs, metadatas=metas)

    print(f"  Vector index: {col.count()} chunks in ChromaDB.")


def semantic_search(query: str, top_k: int = TOP_K_VECTOR) -> List[dict]:
    """Return top-k semantically similar chunks."""
    col = _get_collection()
    results = col.query(
        query_texts=[query],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text":     doc,
            "source":   meta.get("source", ""),
            "page":     meta.get("page", 0),
            "chapter":  meta.get("chapter", ""),
            "section":  meta.get("section", ""),
            "score":    round(1 - dist, 4),   # cosine similarity
            "retriever": "semantic",
        })
    return hits
