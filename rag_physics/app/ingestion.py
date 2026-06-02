"""
PDF ingestion pipeline for NCERT Class 12 Physics Part 1.

Strategy: Section-aware + page-aware chunking
-------------------------------------------------
1. Parse every page with PyMuPDF preserving text blocks, font sizes, bold flags.
2. Detect headings by font-size threshold → identify chapter / section / sub-section.
3. Chunk text within each section (max ~400 tokens / ~1600 chars) with 20% overlap.
4. Store rich metadata per chunk: page, chapter, section, chunk_id.

Trade-offs discussed in report:
- Section-aware > fixed-size: keeps semantically coherent content together.
- Overlap prevents context loss at chunk boundaries.
- Downside: section detection is heuristic (font size) and may miss some headings.
"""
from __future__ import annotations
import re, pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id:   str
    text:       str
    page:       int
    chapter:    str
    section:    str
    subsection: str
    char_start: int = 0
    metadata:   dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id":   self.chunk_id,
            "text":       self.text,
            "page":       self.page,
            "chapter":    self.chapter,
            "section":    self.section,
            "subsection": self.subsection,
            "source":     f"Page {self.page} | {self.chapter} > {self.section}",
        }


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

CHAPTER_RE  = re.compile(r"^(chapter\s+(one|two|three|four|five|six|seven|eight|nine|ten|\d+))", re.I)
FORMULA_RE  = re.compile(r"[=+\-*/^∫∑√π∞αβγδθλμω]")


CHAPTER_TITLE_RE = re.compile(
    r"^(Chapter\s+(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|\d+)|"
    r"CHAPTER\s+\d+|ELECTRIC\s+CHARGES|ELECTROSTATIC|CURRENT\s+ELECTRICITY|"
    r"MOVING\s+CHARGES|MAGNETISM|ELECTROMAGNETIC|ALTERNATING\s+CURRENT|"
    r"DUAL\s+NATURE|ATOMS|NUCLEI|SEMICONDUCTOR)", re.I
)
SECTION_NUM_RE = re.compile(r"^\d+\.\d+\s+\S")


def _detect_heading_level(span_text: str, size: float, bold: bool,
                           median_size: float) -> int:
    """Return 1=chapter, 2=section, 3=subsection, 0=body.

    NCERT PDFs use large font sizes but not bold flags for headings.
    Thresholds tuned to NCERT Physics (median body ≈ 10pt):
      chapter  : size ≥ 2.5× median  (≈25pt+)  OR chapter-name regex
      section  : size ≥ 1.4× median  (≈14pt+)  OR numbered section pattern
      subsection: size ≥ 1.1× median (≈11pt+)
    """
    t = span_text.strip()
    if not t:
        return 0
    # Chapter: very large title OR explicit chapter header text
    if CHAPTER_RE.match(t) or CHAPTER_TITLE_RE.match(t):
        return 1
    if size >= median_size * 2.5:
        return 1
    # Section: moderately large OR numbered "1.2 Something"
    if size >= median_size * 1.4 or (SECTION_NUM_RE.match(t) and size >= median_size * 1.1):
        return 2
    # Sub-section: slightly larger than body
    if (size >= median_size * 1.1 and bold) or (size >= median_size * 1.15):
        return 3
    return 0


# ---------------------------------------------------------------------------
# Page parser
# ---------------------------------------------------------------------------

def _extract_page_blocks(page: fitz.Page, median_size: float) -> list[dict]:
    """Extract text blocks with heading level annotation."""
    blocks = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:   # skip image blocks
            continue
        lines_text = []
        is_heading  = 0
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt  = span.get("text", "").strip()
                size = span.get("size", 10)
                bold = bool(span.get("flags", 0) & 2**4)
                lvl  = _detect_heading_level(txt, size, bold, median_size)
                if lvl and not is_heading:
                    is_heading = lvl
                lines_text.append(txt)
        text = " ".join(lines_text).strip()
        if text:
            blocks.append({"text": text, "heading_level": is_heading})
    return blocks


def _median_font_size(doc: fitz.Document) -> float:
    sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes.append(span.get("size", 10))
    if not sizes:
        return 10.0
    sizes.sort()
    return sizes[len(sizes) // 2]


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

MAX_CHUNK_CHARS = 1500
OVERLAP_CHARS   = 200


def _split_text(text: str, max_chars: int = MAX_CHUNK_CHARS,
                overlap: int = OVERLAP_CHARS) -> List[str]:
    """Split long text into overlapping chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chars and current:
            chunks.append(current.strip())
            # Keep overlap from end of current chunk
            current = current[-overlap:] + " " + sent if overlap else sent
        else:
            current = (current + " " + sent).strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: str | Path) -> List[Chunk]:
    """
    Parse the PDF and return a list of Chunk objects with rich metadata.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    median_size = _median_font_size(doc)

    chunks:   List[Chunk] = []
    chapter   = "Unknown Chapter"
    section   = "Introduction"
    subsection = ""
    chunk_idx  = 0

    for page_num, page in enumerate(doc, start=1):
        blocks = _extract_page_blocks(page, median_size)

        # Accumulate body text until next heading
        body_buffer = ""

        def flush_buffer(buf: str, pg: int):
            nonlocal chunk_idx
            if not buf.strip():
                return
            for part in _split_text(buf):
                if len(part.strip()) < 40:   # skip tiny fragments
                    continue
                chunks.append(Chunk(
                    chunk_id   = f"chunk_{chunk_idx:05d}",
                    text       = part.strip(),
                    page       = pg,
                    chapter    = chapter,
                    section    = section,
                    subsection = subsection,
                ))
                chunk_idx += 1

        for block in blocks:
            lvl  = block["heading_level"]
            text = block["text"]

            if lvl == 1:
                flush_buffer(body_buffer, page_num); body_buffer = ""
                # Only accept clean chapter names (short, no leading digits/parens/FIGURE)
                if (len(text) <= 80
                        and not re.match(r'^[\d\(\*]', text.strip())
                        and not re.match(r'^(FIGURE|TABLE|NOTE|Example)\b', text.strip(), re.I)):
                    chapter    = text
                    section    = "Introduction"
                    subsection = ""
            elif lvl == 2:
                flush_buffer(body_buffer, page_num); body_buffer = ""
                if len(text) <= 120 and not re.match(r'^[\(\*]', text.strip()):
                    section    = text
                    subsection = ""
            elif lvl == 3:
                flush_buffer(body_buffer, page_num); body_buffer = ""
                subsection = text
            else:
                body_buffer += " " + text

        flush_buffer(body_buffer, page_num)

    doc.close()
    print(f"  Ingested {len(chunks)} chunks from {page_num} pages.")
    return chunks


def save_chunks(chunks: List[Chunk], path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(chunks, f)


def load_chunks(path: str | Path) -> List[Chunk]:
    with open(path, "rb") as f:
        return pickle.load(f)
