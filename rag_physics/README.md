# NCERT Physics Hybrid RAG — Use Case 3

A production-quality RAG application for answering questions from **NCERT Class 12 Physics Part 1** using hybrid retrieval (Vector DB + Graph DB + BM25 keyword search).

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │           User Question                  │
                        └────────────────┬────────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                           ▼
    ┌──────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
    │  Semantic Search │    │   Keyword Search     │    │  Graph Search    │
    │  (ChromaDB +     │    │   (BM25 / rank-bm25) │    │  (NetworkX KG)   │
    │  MiniLM-L6-v2)  │    │                      │    │                  │
    └────────┬─────────┘    └──────────┬───────────┘    └────────┬─────────┘
             │                         │                          │
             └─────────────────────────┼──────────────────────────┘
                                       ▼
                          ┌────────────────────────┐
                          │   RRF Fusion           │
                          │   score = Σ 1/(60+rank)│
                          └────────────┬───────────┘
                                       ▼
                          ┌────────────────────────┐
                          │   Top-K Chunks         │
                          │   + Source Citations   │
                          └────────────┬───────────┘
                                       ▼
                          ┌────────────────────────┐
                          │   LLM (Groq/OpenAI)    │
                          │   Grounded Answer      │
                          └────────────────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| PDF Ingestion | PyMuPDF + section-aware chunking | Parse PDF, detect headings via font-size, chunk ~1500 chars with 200-char overlap |
| Vector DB | ChromaDB (persistent) + `all-MiniLM-L6-v2` | Semantic similarity retrieval |
| Keyword Search | BM25 (rank-bm25) | Exact term matching, formula/law names |
| Knowledge Graph | NetworkX DiGraph | Chapter→Section→Chunk→Concept→Formula relationships |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF) | Merge ranked lists from all 3 retrievers |
| Answer Generation | Groq LLaMA-3 8B (or OpenAI GPT-3.5) | Grounded answers with citations |
| Backend | FastAPI | REST API |
| Frontend | Vanilla HTML/CSS/JS | Chat UI with evidence panel |

---

## Setup

### 1. Install dependencies

```bash
cd /home/intel/Vanco/rag_physics
pip install -r requirements.txt --break-system-packages
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env — set GROQ_API_KEY (free at https://console.groq.com)
```

### 3. Add the NCERT PDF

Place the NCERT Class 12 Physics Part 1 PDF at:
```
data/pdf/ncert_physics_12_part1.pdf
```
Download from: https://ncert.nic.in/textbook.php?leph1=0-8

### 4. Build all indexes

```bash
python scripts/ingest.py
```

This will:
- Parse the PDF into ~400-600 chunks
- Build ChromaDB vector index
- Build BM25 keyword index
- Build NetworkX knowledge graph
- Save all to `indexes/`

### 5. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Open the UI

Visit: http://localhost:8000

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat UI (frontend) |
| `GET` | `/health` | Health check — index status |
| `POST` | `/ask` | Ask a question, get grounded answer |
| `POST` | `/diagnostics` | Show retrieval breakdown per retriever |
| `GET` | `/graph/stats` | Knowledge graph statistics |

### POST /ask example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Gauss law?", "show_evidence": true}'
```

Response:
```json
{
  "answer": "Gauss's law states that the total electric flux through any closed surface...[1]",
  "sources": [{"ref": 1, "source": "Page 32 | Chapter 1 > Electric Flux and Gauss Law"}],
  "model_used": "groq/llama3-8b-8192",
  "evidence": [{"rank": 1, "text": "...", "rrf_score": 0.033}]
}
```

### POST /diagnostics example

```bash
curl -X POST http://localhost:8000/diagnostics \
  -H "Content-Type: application/json" \
  -d '{"question": "explain Faraday law of induction"}'
```

---

## Chunking Strategy

**Section-aware + page-aware chunking** with sentence-boundary splitting:

1. PyMuPDF extracts text blocks with font-size and bold metadata
2. Headings detected by font-size thresholds relative to document median:
   - `size ≥ 1.6× median + bold` → Chapter
   - `size ≥ 1.3× median + bold` → Section
   - `size ≥ 1.1× median + bold` → Sub-section
3. Body text accumulated until next heading, then split at sentence boundaries
4. Max chunk size: 1500 chars (~350 tokens), 200-char overlap
5. Rich metadata preserved: page, chapter, section, subsection

**Trade-offs:**
- ✅ Section-aware keeps semantically coherent content together
- ✅ Sentence boundary split avoids mid-sentence truncation
- ⚠️ Font-size heuristic may miss headings with atypical formatting
- ⚠️ Formulas spanning lines may get split

---

## Knowledge Graph Schema

```
(Chapter) ──HAS_SECTION──> (Section) ──HAS_CHUNK──> (Chunk)
                                                         │
                                              ┌──────────┴──────────┐
                                              ▼                     ▼
                                          (Concept)            (Formula)
                                              │
                                        RELATED_TO
                                              │
                                          (Concept)
```

---

## Example Questions to Test

| Type | Example |
|------|---------|
| Factual | "What is the SI unit of electric field?" |
| Conceptual | "Explain the principle of superposition of electric forces." |
| Formula | "Write the formula for capacitance of a parallel plate capacitor." |
| Comparison | "What is the difference between electric potential and potential energy?" |
| Law-based | "State Gauss's law and explain its significance." |
| Out-of-doc | "What is Newton's 3rd law?" → should say "not available in document" |

---

## Limitations & Improvement Plan

| Limitation | Improvement |
|------------|-------------|
| Heuristic heading detection | Use ML-based layout analysis (pdfplumber + vision model) |
| Formula handling | Use LaTeX extraction or MathPix for structured formula parsing |
| Graph quality | Add coreference resolution, NER-based concept extraction |
| No reranking | Add cross-encoder reranker (ms-marco-MiniLM) after RRF |
| Single PDF | Extend to multi-document corpus with per-source metadata |
| Latency ~2-3s | Cache embeddings, pre-warm BM25, use async retrieval |

---

## Project Structure

```
rag_physics/
├── app/
│   ├── config.py          # Central config from .env
│   ├── ingestion.py       # PDF parser + section-aware chunker
│   ├── vector_db.py       # ChromaDB semantic search
│   ├── keyword_search.py  # BM25 keyword search
│   ├── graph_db.py        # NetworkX knowledge graph
│   ├── retriever.py       # Hybrid RRF fusion
│   ├── generator.py       # Groq/OpenAI answer generation
│   └── main.py            # FastAPI backend
├── frontend/
│   └── index.html         # Chat UI
├── scripts/
│   └── ingest.py          # Build all indexes
├── data/
│   └── pdf/               # Place NCERT PDF here
├── indexes/               # Auto-created by ingest.py
├── requirements.txt
├── .env.example
└── README.md
```
