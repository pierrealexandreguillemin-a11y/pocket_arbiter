#!/usr/bin/env python3
"""
Etape 3: Generation Hard Negatives (UNIFIED_TRAINING_DATA_SPEC.md)

Genere des triplets (anchor, positive, negative) avec hard negatives intelligents.
Strategie hierarchique:
- same_doc_diff_page (40%): meme document, pages differentes
- same_category (30%): meme categorie de reglement
- semantic_similar (20%): semantiquement proche via embeddings
- random (10%): baseline aleatoire

ISO 42001 A.6.2.2 - Provenance tracable
ISO 25010 - Exactitude fonctionnelle

Usage:
    python -m scripts.training.unified.generate_hard_negatives \
        --input data/training/unified/gs_reformulated.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --output data/training/unified/triplets_raw.jsonl
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

DEFAULT_INPUT = PROJECT_ROOT / "data" / "training" / "unified" / "gs_reformulated.json"
DEFAULT_CHUNKS = PROJECT_ROOT / "corpus" / "processed" / "chunks_mode_b_fr.json"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "corpus" / "processed" / "embeddings_mode_b_fr.npy"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training" / "unified" / "triplets_raw.jsonl"

# Distribution cible des methodes de selection
TARGET_DISTRIBUTION = {
    "same_doc_diff_page": 0.40,
    "same_category": 0.30,
    "semantic_similar": 0.20,
    "random": 0.10,
}


def load_json_file(path: Path) -> Any:
    """Load JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_embeddings(path: Path) -> np.ndarray | None:
    """Load embeddings numpy array if it exists."""
    if path.exists():
        return np.load(path)
    logger.warning(f"Embeddings not found: {path}")
    return None


def build_chunk_index(chunks_data: Any) -> dict[str, dict]:
    """Build index of chunks by ID."""
    chunks = chunks_data.get("chunks", chunks_data) if isinstance(chunks_data, dict) else chunks_data
    return {chunk["id"]: chunk for chunk in chunks}


def build_chunk_indices(chunks: list[dict]) -> tuple[dict, dict, dict]:
    """
    Build multiple indices for efficient negative selection.

    Returns:
        - by_source: chunks grouped by source document
        - by_category: chunks grouped by category (from metadata or inferred)
        - all_ids: list of all chunk IDs
    """
    by_source: dict[str, list[dict]] = defaultdict(list)
    by_category: dict[str, list[dict]] = defaultdict(list)

    for chunk in chunks:
        # Index by source
        source = chunk.get("source", "unknown")
        by_source[source].append(chunk)

        # Index by category (from section or inferred)
        section = chunk.get("section", "").lower()
        category = "general"
        if "article" in section:
            category = "regles_jeu"
        elif "competition" in section or "tournoi" in section:
            category = "competitions"
        elif "interclub" in section:
            category = "interclubs"
        elif "disciplin" in section:
            category = "disciplinaire"
        by_category[category].append(chunk)

    return dict(by_source), dict(by_category), [c["id"] for c in chunks]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def select_same_doc_diff_page(
    positive_chunk: dict,
    by_source: dict[str, list[dict]],
) -> dict | None:
    """Select negative from same document but different pages."""
    source = positive_chunk.get("source", "")
    positive_pages = set(positive_chunk.get("pages", [positive_chunk.get("page", -1)]))

    candidates = by_source.get(source, [])
    valid = [
        c for c in candidates
        if c["id"] != positive_chunk["id"]
        and not set(c.get("pages", [c.get("page", -1)])).intersection(positive_pages)
    ]

    return random.choice(valid) if valid else None


def select_same_category(
    positive_chunk: dict,
    question: dict,
    by_category: dict[str, list[dict]],
) -> dict | None:
    """Select negative from same category."""
    # Get question category
    q_category = question.get("category", "general")

    # Map to chunk category
    category_mapping = {
        "regles_jeu": "regles_jeu",
        "competitions": "competitions",
        "interclubs": "interclubs",
        "tournoi": "competitions",
        "regles_ffe": "general",
        "open": "general",
    }
    chunk_category = category_mapping.get(q_category, "general")

    candidates = by_category.get(chunk_category, [])
    valid = [c for c in candidates if c["id"] != positive_chunk["id"]]

    return random.choice(valid) if valid else None


def select_semantic_similar(
    positive_idx: int,
    chunk_ids: list[str],
    embeddings: np.ndarray,
    positive_id: str,
    min_similarity: float = 0.3,
    max_similarity: float = 0.9,
) -> str | None:
    """Select semantically similar but not identical chunk."""
    if embeddings is None or positive_idx >= len(embeddings):
        return None

    positive_emb = embeddings[positive_idx]

    # Compute similarities to all chunks
    similarities = []
    for i, emb in enumerate(embeddings):
        if chunk_ids[i] != positive_id:
            sim = cosine_similarity(positive_emb, emb)
            if min_similarity <= sim <= max_similarity:
                similarities.append((chunk_ids[i], sim))

    if not similarities:
        return None

    # Sort by similarity descending, take from top half
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_candidates = similarities[:max(1, len(similarities) // 2)]
    return random.choice(top_candidates)[0]


def select_random(
    positive_id: str,
    all_ids: list[str],
) -> str | None:
    """Select random chunk."""
    candidates = [cid for cid in all_ids if cid != positive_id]
    return random.choice(candidates) if candidates else None


def select_hard_negative(
    question: dict,
    positive_chunk: dict,
    chunk_index: dict[str, dict],
    by_source: dict[str, list[dict]],
    by_category: dict[str, list[dict]],
    all_ids: list[str],
    embeddings: np.ndarray | None,
    method_counts: dict[str, int],
) -> tuple[dict | None, str]:
    """
    Select hard negative using hierarchical strategy.

    Tries to balance according to TARGET_DISTRIBUTION.
    """
    positive_id = positive_chunk["id"]
    positive_idx = all_ids.index(positive_id) if positive_id in all_ids else -1

    # Calculate current distribution
    total = sum(method_counts.values()) or 1
    current_ratios = {m: method_counts[m] / total for m in TARGET_DISTRIBUTION}

    # Try methods in order of need (most underrepresented first)
    methods_by_need = sorted(
        TARGET_DISTRIBUTION.keys(),
        key=lambda m: TARGET_DISTRIBUTION[m] - current_ratios.get(m, 0),
        reverse=True,
    )

    for method in methods_by_need:
        negative = None

        if method == "same_doc_diff_page":
            negative = select_same_doc_diff_page(positive_chunk, by_source)

        elif method == "same_category":
            negative = select_same_category(positive_chunk, question, by_category)

        elif method == "semantic_similar" and embeddings is not None:
            neg_id = select_semantic_similar(
                positive_idx, all_ids, embeddings, positive_id
            )
            if neg_id:
                negative = chunk_index.get(neg_id)

        elif method == "random":
            neg_id = select_random(positive_id, all_ids)
            if neg_id:
                negative = chunk_index.get(neg_id)

        if negative:
            return negative, method

    # Fallback to random
    neg_id = select_random(positive_id, all_ids)
    return chunk_index.get(neg_id) if neg_id else None, "random_fallback"


def generate_triplets(
    gs_data: dict[str, Any],
    chunk_index: dict[str, dict],
    by_source: dict[str, list[dict]],
    by_category: dict[str, list[dict]],
    all_ids: list[str],
    embeddings: np.ndarray | None,
) -> tuple[list[dict], dict]:
    """Generate triplets for all questions with chunks."""
    questions = gs_data.get("questions", [])

    triplets = []
    method_counts: dict[str, int] = defaultdict(int)
    skipped = 0

    for question in tqdm(questions, desc="Generating triplets"):
        chunk_id = question.get("expected_chunk_id")
        if not chunk_id:
            skipped += 1
            continue

        positive_chunk = chunk_index.get(chunk_id)
        if not positive_chunk:
            logger.warning(f"Chunk not found: {chunk_id}")
            skipped += 1
            continue

        # Get question text (prefer reformulated)
        anchor = question.get("question_reformulated", question.get("question", ""))
        if not anchor:
            skipped += 1
            continue

        # Select hard negative
        negative_chunk, method = select_hard_negative(
            question,
            positive_chunk,
            chunk_index,
            by_source,
            by_category,
            all_ids,
            embeddings,
            method_counts,
        )

        if not negative_chunk:
            skipped += 1
            continue

        method_counts[method] += 1

        triplet = {
            "anchor": anchor,
            "positive": positive_chunk["text"],
            "negative": negative_chunk["text"],
            "metadata": {
                "question_id": question.get("id"),
                "positive_chunk_id": chunk_id,
                "negative_chunk_id": negative_chunk["id"],
                "selection_method": method,
                "category": question.get("category"),
                "difficulty": question.get("difficulty"),
                "original_question": question.get("original_annales", question.get("question")),
            },
        }
        triplets.append(triplet)

    # Calculate statistics
    total_generated = len(triplets)
    stats = {
        "total_questions": len(questions),
        "triplets_generated": total_generated,
        "skipped": skipped,
        "method_distribution": dict(method_counts),
        "method_percentages": {
            m: round(c / total_generated * 100, 1) if total_generated > 0 else 0
            for m, c in method_counts.items()
        },
    }

    return triplets, stats


def save_triplets_jsonl(triplets: list[dict], output_path: Path) -> None:
    """Save triplets in JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for triplet in triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + "\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate hard negatives for triplet training"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input reformulated GS JSON",
    )
    parser.add_argument(
        "--chunks", "-c",
        type=Path,
        default=DEFAULT_CHUNKS,
        help="Chunks JSON file",
    )
    parser.add_argument(
        "--embeddings", "-e",
        type=Path,
        default=DEFAULT_EMBEDDINGS,
        help="Embeddings numpy file (optional, for semantic similarity)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output triplets JSONL",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("ETAPE 3: Generation Hard Negatives")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading GS: {args.input}")
    gs_data = load_json_file(args.input)

    logger.info(f"Loading chunks: {args.chunks}")
    chunks_data = load_json_file(args.chunks)
    chunks = chunks_data.get("chunks", chunks_data) if isinstance(chunks_data, dict) else chunks_data
    chunk_index = build_chunk_index(chunks_data)
    logger.info(f"  Loaded {len(chunks)} chunks")

    # Build indices
    logger.info("Building indices...")
    by_source, by_category, all_ids = build_chunk_indices(chunks)
    logger.info(f"  {len(by_source)} sources, {len(by_category)} categories")

    # Load embeddings (optional)
    embeddings = load_embeddings(args.embeddings)
    if embeddings is not None:
        logger.info(f"Loaded embeddings: {embeddings.shape}")
    else:
        logger.info("No embeddings - semantic_similar method disabled")

    # Generate triplets
    logger.info("Generating triplets...")
    triplets, stats = generate_triplets(
        gs_data,
        chunk_index,
        by_source,
        by_category,
        all_ids,
        embeddings,
    )

    # Save output
    save_triplets_jsonl(triplets, args.output)
    logger.info(f"Saved {len(triplets)} triplets to: {args.output}")

    # Save report
    report = {
        "statistics": stats,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
    }
    report_path = args.output.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Print summary
    logger.info("")
    logger.info("SUMMARY:")
    logger.info(f"  Total questions: {stats['total_questions']}")
    logger.info(f"  Triplets generated: {stats['triplets_generated']}")
    logger.info(f"  Skipped: {stats['skipped']}")
    logger.info("")
    logger.info("Method distribution:")
    for method, count in stats["method_distribution"].items():
        pct = stats["method_percentages"].get(method, 0)
        target = TARGET_DISTRIBUTION.get(method, 0) * 100
        logger.info(f"  {method}: {count} ({pct}%) [target: {target}%]")


if __name__ == "__main__":
    main()
