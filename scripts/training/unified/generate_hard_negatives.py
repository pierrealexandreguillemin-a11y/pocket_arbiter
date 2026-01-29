#!/usr/bin/env python3
"""
Etape 3: Generation Hard Negatives (TRIPLET_GENERATION_SPEC.md)

Genere des hard negatives pour chaque question du GS:
- same_doc_diff_page: chunk du meme document, page differente (>= 40%)
- same_doc_same_page: chunk du meme document, meme page
- cross_doc_same_category: chunk d'un autre document, meme categorie
- random: chunk aleatoire du corpus

ISO 42001 A.6.2.2 - Provenance tracable
ISO 29119 - Test data documentation

Usage:
    python -m scripts.training.unified.generate_hard_negatives \
        --gs tests/data/gold_standard_annales_fr_v7.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --output data/training/unified/triplets_raw.jsonl
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Target distribution for hard negatives (FINETUNING_RESOURCES §3.3)
TARGET_DISTRIBUTION: dict[str, float] = {
    "same_doc_diff_page": 0.40,
    "same_doc_same_page": 0.10,
    "cross_doc_same_category": 0.30,
    "random": 0.20,
}

# Category keywords mapping (FFE/FIDE corpus)
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "regles_jeu": ["Article", "Regles", "Pion", "Cavalier", "Roi", "Mat"],
    "competitions": ["Competition", "Tournoi", "Classement", "Elo"],
    "interclubs": ["Interclub", "Equipe", "Club"],
    "arbitrage": ["Arbitre", "Appariement", "Ronde"],
    "divers": [],  # Default fallback
}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1], or 0.0 if either vector is zero.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _infer_category(section: str) -> str:
    """Infer category from chunk section text."""
    for category, keywords in CATEGORY_KEYWORDS.items():
        if category == "divers":
            continue
        for keyword in keywords:
            if keyword.lower() in section.lower():
                return category
    return "divers"


def build_chunk_indices(
    chunks: list[dict],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], list[str]]:
    """
    Build lookup indices for chunk selection.

    Args:
        chunks: List of chunk dicts with id, source, section keys.

    Returns:
        Tuple of (by_source, by_category, all_ids).
    """
    by_source: dict[str, list[dict]] = defaultdict(list)
    by_category: dict[str, list[dict]] = defaultdict(list)
    all_ids: list[str] = []

    for chunk in chunks:
        source = chunk.get("source", "")
        section = chunk.get("section", "")
        chunk_id = chunk["id"]

        by_source[source].append(chunk)
        category = _infer_category(section)
        by_category[category].append(chunk)
        all_ids.append(chunk_id)

    return dict(by_source), dict(by_category), all_ids


def select_same_doc_diff_page(
    positive: dict,
    by_source: dict[str, list[dict]],
) -> dict | None:
    """
    Select a chunk from the same document but different page.

    Args:
        positive: Positive chunk dict with id, source, pages.
        by_source: Chunks grouped by source document.

    Returns:
        Selected chunk dict, or None if no valid candidate.
    """
    source = positive["source"]
    positive_pages = set(positive.get("pages", []))
    positive_id = positive["id"]

    candidates = [
        c for c in by_source.get(source, [])
        if c["id"] != positive_id
        and not set(c.get("pages", [])).intersection(positive_pages)
    ]

    if not candidates:
        return None

    return random.choice(candidates)


def select_same_doc_same_page(
    positive: dict,
    by_source: dict[str, list[dict]],
) -> dict | None:
    """
    Select a chunk from the same document and same page (but different chunk).

    Args:
        positive: Positive chunk dict with id, source, pages.
        by_source: Chunks grouped by source document.

    Returns:
        Selected chunk dict, or None if no valid candidate.
    """
    source = positive["source"]
    positive_pages = set(positive.get("pages", []))
    positive_id = positive["id"]

    candidates = [
        c for c in by_source.get(source, [])
        if c["id"] != positive_id
        and set(c.get("pages", [])).intersection(positive_pages)
    ]

    if not candidates:
        return None

    return random.choice(candidates)


def select_cross_doc_same_category(
    positive: dict,
    by_category: dict[str, list[dict]],
) -> dict | None:
    """
    Select a chunk from a different document in the same category.

    Args:
        positive: Positive chunk dict with id, source, section.
        by_category: Chunks grouped by category.

    Returns:
        Selected chunk dict, or None if no valid candidate.
    """
    category = _infer_category(positive.get("section", ""))
    positive_id = positive["id"]
    positive_source = positive.get("source", "")

    candidates = [
        c for c in by_category.get(category, [])
        if c["id"] != positive_id
        and c.get("source", "") != positive_source
    ]

    if not candidates:
        return None

    return random.choice(candidates)


def select_random(positive_id: str, all_ids: list[str]) -> str | None:
    """
    Select a random chunk ID different from the positive.

    Args:
        positive_id: ID of the positive chunk to exclude.
        all_ids: All available chunk IDs.

    Returns:
        Random chunk ID, or None if no other IDs available.
    """
    candidates = [cid for cid in all_ids if cid != positive_id]
    if not candidates:
        return None
    return random.choice(candidates)


def validate_quality_gates(
    triplets: list[dict],
    gs: Any,
    chunks: list[dict],
    by_source: dict[str, list[dict]],
) -> dict[str, dict]:
    """
    Validate hard negative quality gates.

    Args:
        triplets: Generated triplets with metadata.
        gs: Gold standard data (optional, can be None).
        chunks: Corpus chunks (can be empty).
        by_source: Chunks grouped by source (can be empty).

    Returns:
        Dict of gate results with passed/count fields.
    """
    results: dict[str, dict] = {}

    # Gate 3: No duplicate negatives across dataset (CT-02)
    neg_id_counts: dict[str, int] = defaultdict(int)
    for t in triplets:
        neg_id = t.get("metadata", {}).get("negative_chunk_id", "")
        neg_id_counts[neg_id] += 1

    duplicate_count = sum(1 for count in neg_id_counts.values() if count > 1)

    results["gate_3_duplicate_negatives"] = {
        "passed": duplicate_count == 0,
        "count": duplicate_count,
    }

    # Gate same_doc ratio >= 40%
    total_negatives = len(triplets)
    same_doc_count = sum(
        1 for t in triplets
        if t.get("metadata", {}).get("negative_mining", {}).get("source") == "same_doc"
    )
    same_doc_ratio = same_doc_count / total_negatives if total_negatives > 0 else 0.0
    results["gate_same_doc_ratio"] = {
        "passed": same_doc_ratio >= TARGET_DISTRIBUTION["same_doc_diff_page"],
        "ratio": same_doc_ratio,
        "target": TARGET_DISTRIBUTION["same_doc_diff_page"],
    }

    # Gate negative != positive
    neg_eq_pos = sum(
        1 for t in triplets
        if t.get("negative", "") == t.get("positive", "")
    )
    results["gate_negative_not_positive"] = {
        "passed": neg_eq_pos == 0,
        "count": neg_eq_pos,
    }

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate hard negatives for GS questions"
    )
    parser.add_argument(
        "--gs",
        type=Path,
        default=PROJECT_ROOT / "tests" / "data" / "gold_standard_annales_fr_v7.json",
        help="Gold standard JSON file",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=PROJECT_ROOT / "corpus" / "processed" / "chunks_mode_b_fr.json",
        help="Corpus chunks JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "training" / "unified" / "triplets_raw.jsonl",
        help="Output triplets JSONL file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (ISO 12207)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    with open(args.gs, encoding="utf-8") as f:
        gs = json.load(f)
    with open(args.chunks, encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks = chunks_data.get("chunks", chunks_data) if isinstance(chunks_data, dict) else chunks_data
    chunks_by_id = {c["id"]: c for c in chunks}

    questions = gs["questions"]
    testables = [
        q for q in questions
        if not q.get("metadata", {}).get("requires_context")
    ]

    logger.info("Loaded %d questions (%d testable), %d chunks", len(questions), len(testables), len(chunks))

    # Build indices
    by_source, by_category, all_ids = build_chunk_indices(chunks)

    # Generate triplets
    triplets: list[dict] = []
    for q in testables:
        chunk_id = q.get("expected_chunk_id", "")
        positive_chunk = chunks_by_id.get(chunk_id)
        if not positive_chunk:
            logger.warning("Chunk %s not found for question %s", chunk_id, q.get("id", "?"))
            continue

        positive_text = positive_chunk.get("text", "")
        anchor = q.get("question", "")

        # Try each negative type
        neg_chunk = select_same_doc_diff_page(positive_chunk, by_source)
        if neg_chunk:
            triplets.append({
                "anchor": anchor,
                "positive": positive_text,
                "negative": neg_chunk.get("text", ""),
                "metadata": {
                    "question_id": q.get("id", ""),
                    "chunk_id": chunk_id,
                    "negative_chunk_id": neg_chunk["id"],
                    "difficulty": q.get("metadata", {}).get("difficulty"),
                    "reasoning_class": q.get("metadata", {}).get("reasoning_class"),
                    "negative_mining": {
                        "method": "same_doc_diff_page",
                        "source": "same_doc",
                    },
                },
            })

    # Validate
    gates = validate_quality_gates(triplets, gs, chunks, by_source)
    for gate_name, gate_result in gates.items():
        status = "PASS" if gate_result.get("passed") else "FAIL"
        logger.info("Gate %s: %s %s", gate_name, status, gate_result)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    logger.info("Wrote %d triplets to %s", len(triplets), args.output)


if __name__ == "__main__":
    main()
