"""
Hard Negative Mining pour fine-tuning - Pocket Arbiter

Trouve des chunks similaires mais non-pertinents pour ameliorer
le fine-tuning des embeddings.

ISO Reference: ISO/IEC 42001 A.6.2.2, ISO/IEC 25010 S4.2

Usage:
    python -m scripts.training.hard_negative_mining \
        --pairs data/training/synthetic_pairs.jsonl \
        --db corpus/processed/corpus_fr_v3.db \
        --output data/training/triplets.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from scripts.pipeline.utils import get_timestamp, save_json

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 10
DEFAULT_MIN_SCORE = 0.3
DEFAULT_MAX_SCORE = 0.9


def load_pairs_jsonl(input_path: Path) -> list[dict]:
    """Charge les paires depuis un fichier JSONL."""
    if not input_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {input_path}")
    pairs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def save_triplets_jsonl(triplets: list[dict], output_path: Path) -> None:
    """Sauvegarde les triplets au format JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for triplet in triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(triplets)} triplets to {output_path}")


def retrieve_candidates(
    db_path: Path,
    query_embedding: np.ndarray,
    top_k: int,
    exclude_page: int | None = None,
) -> list[dict]:
    """Recupere les candidats negatifs potentiels."""
    from scripts.pipeline.export_search import retrieve_similar

    results = retrieve_similar(db_path, query_embedding, top_k=top_k * 2)
    filtered = [r for r in results if exclude_page is None or r["page"] != exclude_page]
    return filtered[:top_k]


def select_hard_negative(
    candidates: list[dict],
    min_score: float,
    max_score: float,
) -> dict | None:
    """Selectionne le meilleur hard negative parmi les candidats."""
    eligible = [c for c in candidates if min_score <= c.get("score", 0) <= max_score]
    if not eligible:
        below_max = [c for c in candidates if c.get("score", 0) <= max_score]
        return max(below_max, key=lambda x: x.get("score", 0)) if below_max else None
    return max(eligible, key=lambda x: x.get("score", 0))


def mine_hard_negatives(
    pairs: list[dict],
    model: "SentenceTransformer",
    db_path: Path,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    max_score: float = DEFAULT_MAX_SCORE,
) -> list[dict]:
    """Mine des hard negatives pour chaque paire (query, positive)."""
    from scripts.pipeline.embeddings import embed_query

    triplets = []
    skipped = 0

    for pair in tqdm(pairs, desc="Mining hard negatives"):
        query_emb = embed_query(pair["query"], model)
        candidates = retrieve_candidates(db_path, query_emb, top_k, pair.get("page"))

        if not candidates:
            skipped += 1
            continue

        hard_neg = select_hard_negative(candidates, min_score, max_score)
        if hard_neg is None:
            skipped += 1
            continue

        triplets.append(
            {
                "anchor": pair["query"],
                "positive": pair["positive"],
                "negative": hard_neg["text"],
                "positive_chunk_id": pair.get("chunk_id"),
                "negative_chunk_id": hard_neg["id"],
                "negative_score": hard_neg["score"],
            }
        )

    logger.info(f"Generated {len(triplets)} triplets ({skipped} skipped)")
    return triplets


def convert_to_training_format(triplets: list[dict]) -> list[dict]:
    """Convertit les triplets au format sentence-transformers."""
    return [
        {"anchor": t["anchor"], "positive": t["positive"], "negative": t["negative"]}
        for t in triplets
    ]


def main() -> None:
    """Point d'entree CLI."""
    parser = argparse.ArgumentParser(description="Hard negative mining")
    parser.add_argument("--pairs", "-p", type=Path, required=True)
    parser.add_argument("--db", "-d", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--max-score", type=float, default=DEFAULT_MAX_SCORE)
    parser.add_argument("--model", "-m", type=str, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Loading pairs from: {args.pairs}")
    pairs = load_pairs_jsonl(args.pairs)

    from scripts.pipeline.embeddings import MODEL_ID, load_embedding_model

    model_id = args.model or MODEL_ID
    model = load_embedding_model(model_id)

    triplets = mine_hard_negatives(
        pairs, model, args.db, args.top_k, args.min_score, args.max_score
    )

    save_triplets_jsonl(triplets, args.output)
    training_triplets = convert_to_training_format(triplets)
    training_output = args.output.with_stem(args.output.stem + "_training")
    save_triplets_jsonl(training_triplets, training_output)

    report = {
        "total_pairs": len(pairs),
        "total_triplets": len(triplets),
        "conversion_rate": round(len(triplets) / max(len(pairs), 1), 4),
        "timestamp": get_timestamp(),
    }
    save_json(report, args.output.with_suffix(".report.json"))

    logger.info(f"Generated {len(triplets)} triplets from {len(pairs)} pairs")


if __name__ == "__main__":
    main()
