"""
LLM answer generation grounded strictly in retrieved evidence.
Supports Groq (free, fast) and OpenAI as fallback.
"""
from __future__ import annotations
from typing import List

from app.config import GROQ_API_KEY, OPENAI_API_KEY, GROQ_MODEL, OPENAI_MODEL

SYSTEM_PROMPT = """You are a Physics tutor assistant for NCERT Class 12 Physics Part 1.
Answer ONLY based on the provided context passages from the textbook.
- If the answer is in the context, answer clearly and cite the source (Page number / Section).
- If the answer is NOT in the provided context, say: "This information is not available in the provided NCERT Physics document."
- Never fabricate information not present in the context.
- For formula questions, reproduce the formula exactly as in the context.
- Keep answers concise but complete."""


def _build_prompt(query: str, chunks: List[dict]) -> str:
    context_parts = []
    for i, c in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] Source: {c.get('source','')}\n{c.get('text','')}"
        )
    context = "\n\n".join(context_parts)
    return f"""Context from NCERT Physics textbook:

{context}

Question: {query}

Instructions: Answer based solely on the context above. Include source references like [1], [2] etc."""


def generate_answer(query: str, chunks: List[dict]) -> dict:
    """
    Generate a grounded answer using the retrieved chunks.
    Returns: {answer, sources, model_used}
    """
    prompt = _build_prompt(query, chunks)
    sources = [
        {"ref": i+1, "source": c.get("source",""), "page": c.get("page",0)}
        for i, c in enumerate(chunks)
    ]

    # Try Groq first
    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            return {
                "answer":     resp.choices[0].message.content,
                "sources":    sources,
                "model_used": f"groq/{GROQ_MODEL}",
            }
        except Exception as e:
            print(f"Groq error: {e}, trying OpenAI...")

    # Try OpenAI
    if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            return {
                "answer":     resp.choices[0].message.content,
                "sources":    sources,
                "model_used": f"openai/{OPENAI_MODEL}",
            }
        except Exception as e:
            print(f"OpenAI error: {e}")

    # Fallback: return raw context (no LLM)
    top_chunk = chunks[0].get("text", "No relevant content found.") if chunks else "No relevant content found."
    return {
        "answer": (
            f"⚠️ No LLM API key configured. Raw retrieved context:\n\n{top_chunk}\n\n"
            f"Set GROQ_API_KEY in .env for AI-generated answers."
        ),
        "sources":    sources,
        "model_used": "none (raw retrieval)",
    }
