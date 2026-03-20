"""Data extraction for SimCSE and ICT training.

SimCSE: (chunk_text, chunk_text) — dropout = augmentation.
ICT: (random_sentence, chunk_text) — ORQA standard §9.2 (90% masking).
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import statistics
from pathlib import Path

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u00DC])")
_MIN_SENTENCE_LEN = 20


def load_children_texts(db_path: str) -> list[str]:
    """Load all children texts from corpus DB."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT text FROM children ORDER BY id").fetchall()
    conn.close()
    return [r[0] for r in rows]


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


def generate_simcse_pairs(texts: list[str]) -> list[tuple[str, str]]:
    """SimCSE pairs: (text, text) for dropout augmentation."""
    return [(t, t) for t in texts if t.strip()]


def generate_ict_pairs(
    texts: list[str],
    seed: int = 42,
) -> list[tuple[str, str]]:
    """ICT pairs: (random_sentence, chunk) with 90% masking."""
    pairs = []
    for i, text in enumerate(texts):
        sentence = extract_random_sentence(text, seed=seed + i)
        if sentence is None:
            continue
        masked = mask_sentence_from_chunk(text, sentence, seed=seed + i)
        if len(masked.strip()) < _MIN_SENTENCE_LEN:
            continue
        pairs.append((sentence, masked))
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
    """Compute data statistics for validation (ISO 42001 A.6.2.3)."""
    q_lens = [len(q) for q, _ in pairs]
    d_lens = [len(d) for _, d in pairs]
    return {
        "count": len(pairs),
        "query_len_min": min(q_lens) if q_lens else 0,
        "query_len_median": int(statistics.median(q_lens)) if q_lens else 0,
        "query_len_max": max(q_lens) if q_lens else 0,
        "query_len_p95": int(sorted(q_lens)[int(0.95 * len(q_lens))]) if q_lens else 0,
        "doc_len_min": min(d_lens) if d_lens else 0,
        "doc_len_median": int(statistics.median(d_lens)) if d_lens else 0,
        "doc_len_max": max(d_lens) if d_lens else 0,
        "doc_len_p95": int(sorted(d_lens)[int(0.95 * len(d_lens))]) if d_lens else 0,
    }


def save_pairs(pairs: list[tuple[str, str]], path: str | Path) -> None:
    """Save pairs as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for query, doc in pairs:
            json.dump({"query": query, "document": doc}, f, ensure_ascii=False)
            f.write("\n")
