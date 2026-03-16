"""Cosine brute-force search over children + table summaries.

Searches children and table summary embeddings, returns parent text
for LLM context (small-to-big retrieval). Supports score-based
adaptive k to reduce noise and hallucination.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SearchIndex:
    """In-memory search index loaded from SQLite."""

    child_ids: list[str]
    child_embeddings: np.ndarray  # (N, 768)
    child_parent_ids: list[str]
    child_metadata: list[dict]

    summary_ids: list[str]
    summary_embeddings: np.ndarray  # (M, 768)
    summary_metadata: list[dict]

    parents: dict[str, dict]  # parent_id -> {text, source, section, page}


def load_index(db_path: Path) -> SearchIndex:
    """Load all embeddings and parents from SQLite into memory."""
    conn = sqlite3.connect(db_path)

    # Load children
    rows = conn.execute(
        "SELECT id, embedding, parent_id, source, article_num, section, tokens, page "
        "FROM children"
    ).fetchall()

    child_ids = [r[0] for r in rows]
    child_embeddings = (
        np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        if rows else np.empty((0, 768), dtype=np.float32)
    )
    child_parent_ids = [r[2] for r in rows]
    child_metadata = [
        {"source": r[3], "article_num": r[4], "section": r[5],
         "tokens": r[6], "page": r[7]}
        for r in rows
    ]

    # Load table summaries
    srows = conn.execute(
        "SELECT id, embedding, summary_text, raw_table_text, source, page "
        "FROM table_summaries"
    ).fetchall()

    summary_ids = [r[0] for r in srows]
    summary_embeddings = (
        np.array([np.frombuffer(r[1], dtype=np.float32) for r in srows])
        if srows else np.empty((0, 768), dtype=np.float32)
    )
    summary_metadata = [
        {"summary_text": r[2], "raw_table_text": r[3], "source": r[4], "page": r[5]}
        for r in srows
    ]

    # Load parents
    prows = conn.execute(
        "SELECT id, text, source, section, page FROM parents"
    ).fetchall()
    parents_dict = {
        r[0]: {"text": r[1], "source": r[2], "section": r[3], "page": r[4]}
        for r in prows
    }

    conn.close()

    return SearchIndex(
        child_ids=child_ids,
        child_embeddings=child_embeddings,
        child_parent_ids=child_parent_ids,
        child_metadata=child_metadata,
        summary_ids=summary_ids,
        summary_embeddings=summary_embeddings,
        summary_metadata=summary_metadata,
        parents=parents_dict,
    )


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between query vector and matrix of vectors."""
    if matrix.shape[0] == 0:
        return np.array([])
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norm @ query_norm


def search(
    index: SearchIndex,
    query_embedding: np.ndarray,
    k: int = 10,
    score_threshold: float | None = None,
    gap_threshold: float | None = None,
) -> list[dict]:
    """Search for top-k children + summaries, return parents.

    Args:
        index: Pre-loaded search index.
        query_embedding: Query vector (768D).
        k: Maximum number of top results.
        score_threshold: Minimum cosine score to include (adaptive k).
        gap_threshold: Maximum score gap between consecutive results (adaptive k).

    Returns:
        List of result dicts with: child_id, score, parent_id, parent_text,
        source, section, result_type ("child" or "table_summary").
    """
    results = []

    # Score children
    child_scores = _cosine_similarity(query_embedding, index.child_embeddings)
    for i, score in enumerate(child_scores):
        results.append({
            "child_id": index.child_ids[i],
            "score": float(score),
            "parent_id": index.child_parent_ids[i],
            "result_type": "child",
            **index.child_metadata[i],
        })

    # Score table summaries
    summary_scores = _cosine_similarity(query_embedding, index.summary_embeddings)
    for i, score in enumerate(summary_scores):
        results.append({
            "child_id": index.summary_ids[i],
            "score": float(score),
            "parent_id": None,
            "result_type": "table_summary",
            **index.summary_metadata[i],
        })

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)

    # Apply adaptive k filtering
    filtered = []
    prev_score = None
    for r in results:
        if len(filtered) >= k:
            break
        if score_threshold is not None and r["score"] < score_threshold:
            break
        if (
            gap_threshold is not None
            and prev_score is not None
            and (prev_score - r["score"]) > gap_threshold
        ):
            break
        filtered.append(r)
        prev_score = r["score"]

    results = filtered

    # Attach parent text
    seen_parents: set[str] = set()
    for r in results:
        if r["result_type"] == "child" and r["parent_id"]:
            parent = index.parents.get(r["parent_id"], {})
            r["parent_text"] = parent.get("text", "")
            seen_parents.add(r["parent_id"])
        elif r["result_type"] == "table_summary":
            r["parent_text"] = r.get("raw_table_text", "")

    return results
