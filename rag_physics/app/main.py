"""
FastAPI backend for NCERT Physics Hybrid RAG.

Endpoints
---------
GET  /              → serve frontend HTML
POST /ask           → main Q&A endpoint
POST /diagnostics   → show retrieval breakdown
GET  /graph/stats   → knowledge graph statistics
GET  /health        → health check
"""
from __future__ import annotations
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config     import PDF_PATH, CHUNKS_FILE
from app.retriever  import hybrid_retrieve, retrieval_diagnostics
from app.generator  import generate_answer
from app.graph_db   import get_graph_stats


app = FastAPI(
    title="NCERT Physics RAG",
    description="Hybrid RAG (Vector + Graph + BM25) for NCERT Class 12 Physics",
    version="1.0.0",
)

FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    show_evidence: bool = False

class AskResponse(BaseModel):
    answer:     str
    sources:    list
    model_used: str
    evidence:   list = []

class DiagRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if FRONTEND.exists():
        return HTMLResponse(FRONTEND.read_text())
    return HTMLResponse("<h2>Frontend not found. Use POST /ask</h2>")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pdf_ready":    PDF_PATH.exists(),
        "index_ready":  CHUNKS_FILE.exists(),
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    chunks = hybrid_retrieve(req.question)
    if not chunks:
        return AskResponse(
            answer="No relevant content found in the document.",
            sources=[], model_used="none", evidence=[]
        )

    result  = generate_answer(req.question, chunks)
    evidence = []
    if req.show_evidence:
        evidence = [
            {
                "rank":       i + 1,
                "text":       c["text"][:300],
                "source":     c.get("source", ""),
                "retrievers": c.get("retrievers", [c.get("retriever", "")]),
                "rrf_score":  c.get("rrf_score", 0),
            }
            for i, c in enumerate(chunks)
        ]

    return AskResponse(
        answer     = result["answer"],
        sources    = result["sources"],
        model_used = result["model_used"],
        evidence   = evidence,
    )


@app.post("/diagnostics")
async def diagnostics(req: DiagRequest):
    return retrieval_diagnostics(req.question)


@app.get("/graph/stats")
async def graph_stats():
    try:
        return get_graph_stats()
    except Exception as e:
        raise HTTPException(500, f"Graph not ready: {e}")
