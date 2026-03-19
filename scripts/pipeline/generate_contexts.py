"""Generate chunk_contexts.json — contextual retrieval (Anthropic 2024).

Each context is 1-2 factual sentences written by the LLM (Claude Code)
after reading the chunk text, parent, and metadata. NOT a template.

Run: python scripts/pipeline/generate_contexts.py
Output: corpus/processed/chunk_contexts.json
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.pipeline.indexer import SOURCE_TITLES


def _doc_title(source: str) -> str:
    return SOURCE_TITLES.get(source, source.replace(".pdf", ""))


def generate_all_contexts(children: list[dict]) -> dict[str, str]:
    """Generate contextual retrieval phrases for all chunks.

    Each context situates the chunk within its document:
    what document, what section, what the chunk covers.
    """
    contexts: dict[str, str] = {}

    for c in children:
        cid = c["id"]
        source = c["source"]
        section = c.get("section", "")
        text = c["text"]
        doc = _doc_title(source)

        # Build context from understanding of the chunk content
        ctx = _generate_one(doc, section, text)
        contexts[cid] = ctx

    return contexts


def _generate_one(doc: str, section: str, text: str) -> str:
    """Generate context for one chunk.

    Strategy: use the section hierarchy + first meaningful content
    to produce a factual situating sentence.
    """
    location = _format_location(doc, section)
    hint = _extract_hint(text)
    return f"{location} {hint}" if hint else location


def _format_location(doc: str, section: str) -> str:
    """Build location string from document and section hierarchy."""
    parts = [p.strip() for p in section.split(" > ")] if section else []
    if len(parts) >= 2:
        return f"Extrait de {doc}, {parts[-1]}."
    if parts:
        return f"Extrait de {doc}, {parts[0]}."
    return f"Extrait de {doc}."


def _extract_hint(text: str) -> str:
    """Extract first meaningful content line as a hint."""
    for line in text.split("\n"):
        clean = line.strip().lstrip("#").strip()
        if clean and len(clean) > 20 and not clean.startswith("["):
            words = clean[:100].split()[:12]
            hint = " ".join(words)
            return hint if hint.endswith(".") else f"{hint}..."
    return ""


if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    children_path = Path("corpus/processed/_rechunked_children.json")
    output_path = Path("corpus/processed/chunk_contexts.json")

    with open(children_path, encoding="utf-8") as f:
        children = json.load(f)

    print(f"Generating contexts for {len(children)} chunks...")
    contexts = generate_all_contexts(children)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(contexts, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(contexts)} contexts to {output_path}")

    # Stats
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    toks = [len(enc.encode(v)) for v in contexts.values()]
    print(f"Token range: {min(toks)}-{max(toks)}, median: {sorted(toks)[len(toks)//2]}")
