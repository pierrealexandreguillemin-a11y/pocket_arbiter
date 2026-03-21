"""Data extraction for SimCSE and ICT training.

SimCSE: (chunk_text, chunk_text) — dropout = augmentation.
ICT: (random_sentence, chunk_text) — ORQA standard §9.2 (90% masking).

Documents are formatted with CCH title to match pipeline inference
(indexer_embed.py format_document: "title: {cch} | text: {text}").
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import statistics
from pathlib import Path

import tiktoken

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u00DC])")
_MIN_SENTENCE_LEN = 20
_enc = tiktoken.get_encoding("cl100k_base")


def load_children(db_path: str) -> list[dict]:
    """Load children with text, source, section from corpus DB."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, text, source, section FROM children ORDER BY id"
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "text": r[1], "source": r[2], "section": r[3] or ""} for r in rows
    ]


def load_children_texts(db_path: str) -> list[str]:
    """Load all children texts from corpus DB (backward compat)."""
    return [c["text"] for c in load_children(db_path)]


def make_cch_title(
    source: str,
    section: str,
    source_titles: dict[str, str],
) -> str:
    """Build CCH title — mirrors indexer_embed.make_cch_title exactly."""
    doc_title = source_titles.get(source, source.replace(".pdf", ""))
    if section:
        return f"{doc_title} > {section}"
    return doc_title


def extract_random_sentence(text: str, seed: int = 42) -> str | None:
    """Extract a random sentence (>20 chars) from text. ORQA standard."""
    sentences = _SENT_RE.split(text)
    valid = [s.strip() for s in sentences if len(s.strip()) > _MIN_SENTENCE_LEN]
    if not valid:
        return None
    return random.Random(seed).choice(valid)


def mask_sentence_from_chunk(
    chunk: str,
    sentence: str,
    mask_rate: float = 0.9,
    seed: int = 0,
) -> str:
    """Remove sentence from chunk with probability mask_rate (ORQA §9.2)."""
    if random.Random(seed).random() < mask_rate:
        return chunk.replace(sentence, "").strip()
    return chunk


def format_document(text: str, cch_title: str) -> str:
    """Format document exactly like indexer_embed.format_document.

    Output: "title: {cch_title} | text: {text}"
    This is pre-formatted — the trainer must NOT add another prompt on this column.
    """
    return f"title: {cch_title} | text: {text}"


def generate_simcse_pairs(
    children: list[dict],
    source_titles: dict[str, str],
) -> list[tuple[str, str]]:
    """SimCSE pairs: (doc_formatted, doc_formatted) for dropout augmentation.

    Documents are pre-formatted with CCH title to match pipeline inference exactly.
    The trainer should NOT add a prompt on the positive column (already formatted).
    Anchor column = same text (SimCSE: identical pairs, dropout varies).
    """
    pairs = []
    for c in children:
        if not c["text"].strip():
            continue
        cch = make_cch_title(c["source"], c["section"], source_titles)
        formatted = format_document(c["text"], cch)
        pairs.append((formatted, formatted))
    return pairs


def generate_ict_pairs(
    children: list[dict],
    source_titles: dict[str, str],
    seed: int = 42,
    max_skip_rate: float = 0.3,
) -> list[tuple[str, str]]:
    """ICT pairs: (random_sentence, formatted_chunk) with 90% masking.

    Query = raw sentence (trainer adds query prompt).
    Document = pre-formatted with CCH title (no prompt).

    Args:
        children: List of dicts with text, source, section.
        source_titles: Mapping source filename to display title.
        seed: Random seed for reproducibility.
        max_skip_rate: Max fraction of chunks that can be skipped (assert).
    """
    pairs = []
    skipped = 0
    for i, c in enumerate(children):
        sentence = extract_random_sentence(c["text"], seed=seed + i)
        if sentence is None:
            skipped += 1
            continue
        masked = mask_sentence_from_chunk(c["text"], sentence, seed=seed + i)
        if len(masked.strip()) < _MIN_SENTENCE_LEN:
            skipped += 1
            continue
        cch = make_cch_title(c["source"], c["section"], source_titles)
        formatted_doc = format_document(masked, cch)
        pairs.append((sentence, formatted_doc))
    skip_rate = skipped / len(children) if children else 0
    if skip_rate > max_skip_rate:
        msg = (
            f"ICT: {skipped}/{len(children)} skipped"
            f" ({skip_rate:.0%} > {max_skip_rate:.0%})"
        )
        raise ValueError(msg)
    return pairs


def validate_pairs(
    pairs: list[tuple[str, str]],
    min_query_len: int = _MIN_SENTENCE_LEN,
) -> list[str]:
    """Validate pairs. Returns list of error messages (empty = OK)."""
    errors = []
    for i, (query, doc) in enumerate(pairs):
        if len(query) <= min_query_len:
            errors.append(f"Pair {i}: query too short ({len(query)} chars)")
        if not doc.strip():
            errors.append(f"Pair {i}: empty document")
    return errors


def compute_data_stats(pairs: list[tuple[str, str]]) -> dict:
    """Compute data statistics in both chars and tokens (ISO 42001 A.6.2.3)."""
    q_lens = [len(q) for q, _ in pairs]
    d_lens = [len(d) for _, d in pairs]
    q_tokens = [len(_enc.encode(q)) for q, _ in pairs]
    d_tokens = [len(_enc.encode(d)) for _, d in pairs]
    return {
        "count": len(pairs),
        "query_char_min": min(q_lens) if q_lens else 0,
        "query_char_median": int(statistics.median(q_lens)) if q_lens else 0,
        "query_char_max": max(q_lens) if q_lens else 0,
        "doc_char_min": min(d_lens) if d_lens else 0,
        "doc_char_median": int(statistics.median(d_lens)) if d_lens else 0,
        "doc_char_max": max(d_lens) if d_lens else 0,
        "query_token_min": min(q_tokens) if q_tokens else 0,
        "query_token_median": int(statistics.median(q_tokens)) if q_tokens else 0,
        "query_token_max": max(q_tokens) if q_tokens else 0,
        "query_token_p95": int(sorted(q_tokens)[int(0.95 * len(q_tokens))])
        if q_tokens
        else 0,
        "doc_token_min": min(d_tokens) if d_tokens else 0,
        "doc_token_median": int(statistics.median(d_tokens)) if d_tokens else 0,
        "doc_token_max": max(d_tokens) if d_tokens else 0,
        "doc_token_p95": int(sorted(d_tokens)[int(0.95 * len(d_tokens))])
        if d_tokens
        else 0,
        "docs_exceed_256_tokens": sum(1 for t in d_tokens if t > 256),
        "docs_exceed_2048_tokens": sum(1 for t in d_tokens if t > 2048),
    }


def save_pairs(pairs: list[tuple[str, str]], path: str | Path) -> None:
    """Save pairs as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for query, doc in pairs:
            json.dump({"query": query, "document": doc}, f, ensure_ascii=False)
            f.write("\n")
