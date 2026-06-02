# Architecture Report — NCERT Physics Hybrid RAG

## 1. Problem Statement

Answer questions from NCERT Class 12 Physics Part 1 with grounded, cited responses. The system must:
- Retrieve relevant context from the textbook (not hallucinate)
- Handle factual, conceptual, formula, and law-based questions
- Cite source pages and sections
- Respond in ~2-3 seconds

---

## 2. System Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              User Question                   │
                    │         (via Web UI or REST API)             │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │           FastAPI Backend (/ask)             │
                    └──┬──────────────┬──────────────┬────────────┘
                       │              │               │
                       ▼              ▼               ▼
          ┌────────────────┐ ┌───────────────┐ ┌──────────────────┐
          │ Semantic Search│ │Keyword Search │ │  Graph Search    │
          │                │ │               │ │                  │
          │ ChromaDB       │ │ BM25          │ │ NetworkX DiGraph │
          │ (persistent)   │ │ (rank-bm25)   │ │ (KG)             │
          │                │ │               │ │                  │
          │ all-MiniLM     │ │ Token overlap │ │ Chapter→Section  │
          │ -L6-v2 (384d)  │ │ IDF weighting │ │ →Chunk→Concept   │
          │                │ │               │ │ →Formula         │
          └───────┬────────┘ └───────┬───────┘ └──────┬───────────┘
                  │                  │                 │
                  └──────────────────┼─────────────────┘
                                     │
                    ┌────────────────▼────────────────────────────┐
                    │         Reciprocal Rank Fusion (RRF)         │
                    │                                              │
                    │   score_i = Σ  1 / (60 + rank_j(i))         │
                    │          j∈{semantic, bm25, graph}           │
                    │                                              │
                    │   Top-K=5 chunks selected                   │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │          Context Assembly                     │
                    │                                              │
                    │   [1] Page 32 | Chapter 1 > Electric Flux    │
                    │   [2] Page 45 | Chapter 1 > Gauss Law        │
                    │   [3] Page 47 | Chapter 1 > Applications     │
                    │   ...                                        │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │   LLM (Groq LLaMA-3 8B / OpenAI GPT-3.5)    │
                    │                                              │
                    │   System: "Answer only from provided         │
                    │   context. Cite sources as [N]."             │
                    │                                              │
                    │   Grounded answer + [N] citations            │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────┐
                    │           Response to User                   │
                    │   { answer, sources, model_used, evidence }  │
                    └─────────────────────────────────────────────┘
```

---

## 3. Ingestion Pipeline

```
NCERT PDF
    │
    ├─► PyMuPDF (fitz) — extract text blocks with font metadata
    │
    ├─► Heading detection (font-size heuristics)
    │   • size ≥ 1.6× median + bold → Chapter
    │   • size ≥ 1.3× median + bold → Section
    │   • size ≥ 1.1× median + bold → Sub-section
    │
    ├─► Section-aware chunking
    │   • Accumulate body text within section
    │   • Split at sentence boundaries when > 1500 chars
    │   • 200-char overlap between adjacent chunks
    │
    ├─► Rich metadata per chunk:
    │   { chunk_id, text, page, chapter, section, subsection }
    │
    ├─► ChromaDB → embed with MiniLM-L6-v2 → persist to indexes/chroma/
    ├─► BM25 index → tokenise + IDF → persist to indexes/bm25_index.pkl
    ├─► NetworkX KG → Chapter→Section→Chunk→Concept→Formula
    │                → persist to indexes/knowledge_graph.gpickle
    └─► Chunk store → indexes/chunks.pkl (for BM25 text lookup)
```

---

## 4. Component Details

### 4.1 ChromaDB (Semantic Search)

- **Collection:** `physics_chunks`
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Distance metric:** cosine similarity
- **Query:** top-5 nearest neighbours by cosine distance
- **Why MiniLM over OpenAI ada-002?** Local, free, 10× faster, 95% quality for science Q&A

### 4.2 BM25 (Keyword Search)

- **Library:** `rank-bm25` (BM25Okapi)
- **Tokenisation:** lowercase → split on whitespace + punctuation
- **Strength:** exact formula names ("Gauss's law", "Coulomb"), law numbers, SI units
- **Weakness:** no semantic understanding (synonyms fail)

### 4.3 Knowledge Graph (Graph Search)

- **Library:** NetworkX DiGraph
- **Node types:** Chapter, Section, Chunk, Concept, Formula
- **Edge types:** HAS_SECTION, HAS_CHUNK, HAS_CONCEPT, HAS_FORMULA, RELATED_TO
- **Query:** BFS from question-matched concepts → retrieve associated chunks
- **Strength:** multi-hop reasoning (concept → related concept → chunk)

### 4.4 RRF Fusion

```python
def rrf_score(rank, k=60):
    return 1.0 / (k + rank)

# Merge ranked lists from all 3 retrievers
for retriever_results in [semantic, bm25, graph]:
    for rank, chunk_id in enumerate(retriever_results):
        scores[chunk_id] += rrf_score(rank)

# Select top-K by total RRF score
```

**Why RRF over weighted average?** RRF is rank-based, not score-based — no normalisation needed across retrievers with different score scales. k=60 smooths rank differences without over-rewarding top-ranked results.

### 4.5 Answer Generation

- **Primary:** Groq LLaMA-3 8B (free tier, ~1.5s response)
- **Fallback:** OpenAI GPT-3.5-turbo (configure in `.env`)
- **System prompt:** Forces grounding — LLM must cite `[N]` and not answer from prior knowledge
- **Temperature:** 0.1 (near-deterministic for factual Q&A)

---

## 5. API Endpoints

| Method | Path | Input | Output |
|--------|------|-------|--------|
| GET | `/` | — | Chat UI (HTML) |
| GET | `/health` | — | Index status, chunk count |
| POST | `/ask` | `{question, show_evidence}` | `{answer, sources, model_used, evidence}` |
| POST | `/diagnostics` | `{question}` | Per-retriever breakdown |
| GET | `/graph/stats` | — | KG node/edge counts |

---

## 6. Design Trade-offs

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Embedding model | MiniLM-L6-v2 | OpenAI ada-002 | Local, free, fast; ada-002 costs money + API latency |
| Vector DB | ChromaDB | Pinecone, Weaviate | Local persistent store; no cloud dependency for demo |
| LLM | Groq LLaMA-3 8B | GPT-4, Claude | Free tier, 2× faster; GPT-4 overkill for grounded Q&A |
| Fusion | RRF | Learned reranker | RRF is parameter-free; reranker needs training data |
| PDF parser | PyMuPDF | PDFPlumber, Camelot | Font metadata access for heading detection; fast |
| KG construction | Rule-based | NER + coreference | Simpler, reproducible; NER needs fine-tuned model |

---

## 7. Chunking Strategy

**Section-aware + sentence-boundary chunking:**

- ✅ Keeps semantically coherent content together (no mid-topic splits)
- ✅ Sentence boundaries prevent truncated context
- ✅ Rich metadata enables precise citations (Page X | Chapter Y > Section Z)
- ⚠️ Font-size heuristic may miss headings with non-standard formatting
- ⚠️ Formulas spanning multiple lines may get split at line breaks

---

## 8. Example Responses

**Question:** "What is Gauss's law?"  
**Retrieved chunks:** Page 32 (Electric Flux definition), Page 45 (Gauss Law statement)  
**Answer:** "Gauss's law states that the total electric flux through any closed surface is equal to the net charge enclosed divided by the permittivity of free space: Φ = Q_enc/ε₀ [1][2]."

**Out-of-document question:** "What is Newton's third law?"  
**Answer:** "The provided document does not contain information about Newton's third law. This document covers NCERT Class 12 Physics Part 1 topics including electrostatics, current electricity, and magnetic effects."

---

## 9. Limitations & Improvement Roadmap

| Priority | Improvement | Expected Impact |
|---|---|---|
| High | Cross-encoder reranker (ms-marco-MiniLM) after RRF | +10-15% retrieval precision |
| High | RAGAS evaluation (faithfulness, relevance, correctness) | Quantitative quality measurement |
| Medium | LaTeX extraction for structured formula parsing | Better formula Q&A |
| Medium | Multi-document corpus (Parts 1 & 2, Chemistry) | Broader coverage |
| Medium | Streaming responses (SSE) | Perceived latency improvement |
| Low | Caching (Redis) for frequent questions | 10× faster repeated queries |
| Low | ONNX export of MiniLM for faster embedding | 2× embedding speed |
