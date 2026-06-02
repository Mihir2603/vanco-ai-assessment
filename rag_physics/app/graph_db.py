"""
Knowledge graph built from NCERT Physics chunks using NetworkX.

Graph schema
------------
Nodes:
  chapter     (type=chapter,  label=chapter_name)
  section     (type=section,  label=section_name)
  concept     (type=concept,  label=term)
  formula     (type=formula,  label=formula_text)
  chunk       (type=chunk,    chunk_id, page)

Edges:
  HAS_SECTION  (chapter → section)
  HAS_CHUNK    (section → chunk)
  MENTIONS     (chunk → concept/formula)
  RELATED_TO   (concept → concept)  — co-occurrence in same chunk
"""
from __future__ import annotations
import re
import pickle
from typing import List

import networkx as nx

from app.config import GRAPH_FILE, TOP_K_GRAPH
from app.ingestion import Chunk


# ---------------------------------------------------------------------------
# Physics concept patterns (domain-specific extraction)
# ---------------------------------------------------------------------------

CONCEPT_PATTERNS = [
    r"\b(electric field|magnetic field|coulomb'?s? law|gauss'?s? law|"
    r"ohm'?s? law|faraday'?s? law|ampere'?s? law|lenz'?s? law|"
    r"capacitance|inductance|resistance|permittivity|permeability|"
    r"potential difference|emf|current|voltage|power|energy|force|"
    r"torque|flux|wave|frequency|wavelength|amplitude|refraction|"
    r"reflection|diffraction|interference|polarization|"
    r"photoelectric effect|de broglie|quantum|photon|electron|"
    r"nucleus|atom|semiconductor|diode|transistor)\b",
]

FORMULA_PATTERN = re.compile(
    r"[A-Za-zαβγδθλμω]\s*[=]\s*[^\n,;]{3,60}|"
    r"F\s*=|E\s*=|V\s*=|I\s*=|P\s*=|q\s*="
)


def _extract_concepts(text: str) -> List[str]:
    found = set()
    for pat in CONCEPT_PATTERNS:
        for m in re.finditer(pat, text, re.I):
            found.add(m.group(0).lower().strip())
    return list(found)


def _extract_formulas(text: str) -> List[str]:
    return [m.group(0).strip() for m in FORMULA_PATTERN.finditer(text)]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(chunks: List[Chunk]) -> nx.DiGraph:
    G = nx.DiGraph()

    for chunk in chunks:
        chap = chunk.chapter or "Unknown Chapter"
        sec  = chunk.section  or "General"
        cid  = chunk.chunk_id

        # Nodes
        if not G.has_node(chap):
            G.add_node(chap, type="chapter", label=chap)
        sec_key = f"{chap}::{sec}"
        if not G.has_node(sec_key):
            G.add_node(sec_key, type="section", label=sec, chapter=chap)
        if not G.has_node(cid):
            G.add_node(cid, type="chunk", page=chunk.page,
                       chapter=chap, section=sec, text=chunk.text[:200])

        # Structural edges
        G.add_edge(chap, sec_key, rel="HAS_SECTION")
        G.add_edge(sec_key, cid,  rel="HAS_CHUNK")

        # Concept nodes + edges
        concepts  = _extract_concepts(chunk.text)
        formulas  = _extract_formulas(chunk.text)

        for concept in concepts:
            if not G.has_node(concept):
                G.add_node(concept, type="concept", label=concept)
            G.add_edge(cid, concept, rel="MENTIONS")

        for formula in formulas[:3]:   # cap to avoid noise
            fkey = f"formula::{formula[:50]}"
            if not G.has_node(fkey):
                G.add_node(fkey, type="formula", label=formula[:80])
            G.add_edge(cid, fkey, rel="HAS_FORMULA")

        # Co-occurrence edges between concepts
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                if not G.has_edge(c1, c2):
                    G.add_edge(c1, c2, rel="RELATED_TO", weight=1)
                else:
                    G[c1][c2]["weight"] = G[c1][c2].get("weight", 1) + 1

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def save_graph(G: nx.DiGraph, path=GRAPH_FILE) -> None:
    with open(path, "wb") as f:
        pickle.dump(G, f)


def load_graph(path=GRAPH_FILE) -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Graph retrieval
# ---------------------------------------------------------------------------

_G: nx.DiGraph | None = None


def _get_graph() -> nx.DiGraph:
    global _G
    if _G is None:
        _G = load_graph()
    return _G


def graph_search(query: str, top_k: int = TOP_K_GRAPH) -> List[dict]:
    """
    1. Find concept nodes that match query terms.
    2. Walk neighbours to find chunk nodes.
    3. Return chunk texts as evidence.
    """
    G = _get_graph()
    query_terms = [t.lower() for t in query.split() if len(t) > 3]
    matched_concepts = []
    for node, data in G.nodes(data=True):
        if data.get("type") in ("concept", "formula"):
            for term in query_terms:
                if term in str(node).lower():
                    matched_concepts.append(node)
                    break

    # Expand: find chunk nodes that mention these concepts
    chunk_scores: dict[str, int] = {}
    for concept in matched_concepts:
        # predecessors of concept = chunk nodes that MENTION it
        for pred in G.predecessors(concept):
            if G.nodes[pred].get("type") == "chunk":
                chunk_scores[pred] = chunk_scores.get(pred, 0) + 1

    # Sort by score
    sorted_chunks = sorted(chunk_scores, key=lambda x: chunk_scores[x], reverse=True)[:top_k]

    hits = []
    for cid in sorted_chunks:
        data = G.nodes[cid]
        hits.append({
            "text":     data.get("text", ""),
            "source":   f"Page {data.get('page','')} | {data.get('chapter','')} > {data.get('section','')}",
            "page":     data.get("page", 0),
            "chapter":  data.get("chapter", ""),
            "section":  data.get("section", ""),
            "score":    float(chunk_scores[cid]),
            "retriever": "graph",
        })
    return hits


def get_graph_stats() -> dict:
    G = _get_graph()
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "chapters": [n for n, d in G.nodes(data=True) if d.get("type") == "chapter"],
        "concepts": len([n for n, d in G.nodes(data=True) if d.get("type") == "concept"]),
        "formulas": len([n for n, d in G.nodes(data=True) if d.get("type") == "formula"]),
    }
