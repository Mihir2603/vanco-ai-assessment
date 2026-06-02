"""
Ingestion script: parse PDF → build all indexes.
Run once: python scripts/ingest.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config    import PDF_PATH, CHUNKS_FILE
from app.ingestion import ingest_pdf, save_chunks
from app.vector_db import build_vector_index
from app.keyword_search import build_bm25_index
from app.graph_db  import build_graph, save_graph


def main():
    print("=" * 55)
    print("NCERT Physics RAG — Ingestion Pipeline")
    print("=" * 55)

    print(f"\n[1/4] Parsing PDF: {PDF_PATH}")
    chunks = ingest_pdf(PDF_PATH)
    save_chunks(chunks, CHUNKS_FILE)
    print(f"      Saved {len(chunks)} chunks → {CHUNKS_FILE}")

    print("\n[2/4] Building Vector Index (ChromaDB)...")
    build_vector_index(chunks)

    print("\n[3/4] Building BM25 Keyword Index...")
    build_bm25_index(chunks)

    print("\n[4/4] Building Knowledge Graph...")
    G = build_graph(chunks)
    save_graph(G)

    print("\n✓ All indexes built successfully!")
    print(f"  Chunks : {len(chunks)}")
    print(f"  Indexes: {CHUNKS_FILE.parent}")


if __name__ == "__main__":
    main()
