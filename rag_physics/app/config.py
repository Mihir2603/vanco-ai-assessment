"""Central configuration loaded from .env"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = Path(__file__).parent.parent
PDF_PATH   = BASE_DIR / os.getenv("PDF_PATH", "data/pdf/ncert_physics_12_part1.pdf")
INDEX_DIR  = BASE_DIR / os.getenv("INDEX_DIR", "indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL   = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", 8))
TOP_K_BM25   = int(os.getenv("TOP_K_BM25",   8))
TOP_K_GRAPH  = int(os.getenv("TOP_K_GRAPH",  5))
TOP_K_FINAL  = int(os.getenv("TOP_K_FINAL",  6))

# Groq model preference
GROQ_MODEL   = "llama-3.1-8b-instant"
OPENAI_MODEL = "gpt-3.5-turbo"

CHROMA_COLLECTION = "ncert_physics"
GRAPH_FILE        = INDEX_DIR / "knowledge_graph.gpickle"
BM25_FILE         = INDEX_DIR / "bm25_index.pkl"
CHUNKS_FILE       = INDEX_DIR / "chunks.pkl"
