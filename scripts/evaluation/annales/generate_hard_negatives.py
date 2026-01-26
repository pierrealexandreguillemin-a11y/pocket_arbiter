"""
Generation Hard Negatives - Pocket Arbiter

Genere des hard negatives pour triplets RAG selon NV-Retriever TopK-PercPos.

ISO Reference:
    - ISO/IEC 42001 A.6.2.2 - Tracabilite donnees
    - ISO/IEC 25010 FA-01 - Exactitude fonctionnelle

Standards:
    - NV-Retriever (arXiv:2405.17428): TopK-PercPos avec margin=0.05
    - MTEB #1 methodology

Usage:
    python -m scripts.evaluation.annales.generate_hard_negatives \
        --gs tests/data/gold_standard_annales_fr_v7.json \
        --embeddings corpus/processed/embeddings_mode_b_fr.npy \
        --ids corpus/processed/embeddings_mode_b_fr.ids.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --output tests/data/triplets_annales_fr.json
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from scripts.pipeline.embeddings import load_embedding_model
from scripts.pipeline.embeddings_config import MODEL_ID, PROMPT_QUERY
from scripts.pipeline.utils import load_json, save_json

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# NV-Retriever parameters (standard industrie)
RELATIVE_MARGIN = 0.05  # TopK-PercPos margin
TOP_K_CANDIDATES = 50  # Initial candidates
FINAL_K = 5  # Hard negatives per question
EXCLUDE_SAME_DOC = True  # Diversity: exclude same document


def cosine_similarity_batch(
    query_emb: np.ndarray, corpus_embs: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between query and all corpus embeddings.

    Args:
        query_emb: Query embedding (dim,)
        corpus_embs: Corpus embeddings (N, dim)

    Returns:
        Similarities array (N,)
    """
    # Normalize
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    corpus_norms = corpus_embs / (
        np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-8
    )

    # Dot product = cosine for normalized vectors
    return np.dot(corpus_norms, query_norm)


def extract_doc_id(chunk_id: str) -> str:
    """Extract document ID from chunk ID for diversity filtering."""
    # Format: "filename.pdf-pXXX-parentYYY-childZZ"
    parts = chunk_id.split("-p")
    return parts[0] if parts else chunk_id


def generate_hard_negatives_for_question(
    question: dict,
    corpus_embeddings: np.ndarray,
    chunk_ids: list[str],
    model: "SentenceTransformer",
    relative_margin: float = RELATIVE_MARGIN,
    top_k: int = TOP_K_CANDIDATES,
    final_k: int = FINAL_K,
    exclude_same_doc: bool = EXCLUDE_SAME_DOC,
) -> dict:
    """
    Generate hard negatives for a single question using TopK-PercPos.

    TopK-PercPos (NV-Retriever):
    1. Compute positive score
    2. Filter: keep chunks with score >= (1 - margin) * positive_score
    3. These are hard negatives (close to positive but incorrect)

    Args:
        question: Question dict with question text and expected_chunk_id.
        corpus_embeddings: Pre-computed corpus embeddings (N, dim).
        chunk_ids: List of chunk IDs corresponding to embeddings.
        model: Embedding model for query encoding.
        relative_margin: TopK-PercPos margin (default 0.05 = 5%).
        top_k: Number of initial candidates.
        final_k: Number of hard negatives to return.
        exclude_same_doc: Exclude chunks from same document.

    Returns:
        Triplet dict with query, positive, and hard_negatives.
    """
    q_text = question.get("question", "")
    pos_chunk_id = question.get("expected_chunk_id")
    q_id = question.get("id", "unknown")

    # Encode query
    query_emb = model.encode(
        f"{PROMPT_QUERY}{q_text}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Compute all similarities
    similarities = cosine_similarity_batch(query_emb, corpus_embeddings)

    # Find positive score
    if not pos_chunk_id:
        return {
            "question_id": q_id,
            "query": q_text,
            "positive": pos_chunk_id,
            "positive_score": None,
            "hard_negatives": [],
            "error": "missing_chunk_id",
        }

    try:
        pos_idx = chunk_ids.index(pos_chunk_id)
        pos_score = float(similarities[pos_idx])
    except (ValueError, IndexError):
        # Positive chunk not in corpus - skip
        return {
            "question_id": q_id,
            "query": q_text,
            "positive": pos_chunk_id,
            "positive_score": None,
            "hard_negatives": [],
            "error": "positive_chunk_not_found",
        }

    # Threshold for hard negatives
    threshold = (1 - relative_margin) * pos_score

    # Get document ID for diversity
    pos_doc_id = extract_doc_id(pos_chunk_id)

    # Filter candidates
    candidates = []
    for idx, (chunk_id, score) in enumerate(zip(chunk_ids, similarities)):
        # Skip positive
        if chunk_id == pos_chunk_id:
            continue

        # Skip same document if enabled
        if exclude_same_doc:
            if extract_doc_id(chunk_id) == pos_doc_id:
                continue

        # Hard negative: close to positive but not the same
        if score >= threshold:
            candidates.append((chunk_id, float(score)))

    # Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Take top-K as hard negatives
    hard_negatives = [
        {"chunk_id": c[0], "score": round(c[1], 4)} for c in candidates[:final_k]
    ]

    return {
        "question_id": q_id,
        "query": q_text,
        "positive": pos_chunk_id,
        "positive_score": round(pos_score, 4),
        "hard_negatives": hard_negatives,
        "threshold_used": round(threshold, 4),
        "candidates_found": len(candidates),
    }


def generate_triplets(
    gs_path: Path,
    embeddings_path: Path,
    ids_path: Path,
    chunks_path: Path,
    model_id: str = MODEL_ID,
    relative_margin: float = RELATIVE_MARGIN,
    top_k: int = TOP_K_CANDIDATES,
    final_k: int = FINAL_K,
    exclude_same_doc: bool = EXCLUDE_SAME_DOC,
) -> dict:
    """
    Generate triplets for all testable questions in gold standard.

    Args:
        gs_path: Path to gold standard JSON.
        embeddings_path: Path to corpus embeddings .npy file.
        ids_path: Path to chunk IDs JSON.
        chunks_path: Path to chunks JSON (for metadata).
        model_id: Embedding model ID.
        relative_margin: TopK-PercPos margin.
        top_k: Initial candidates.
        final_k: Hard negatives per question.
        exclude_same_doc: Exclude same document chunks.

    Returns:
        Triplets report with all generated triplets.
    """
    logger.info(f"Loading gold standard: {gs_path}")
    gs_data = load_json(gs_path)
    questions = gs_data.get("questions", [])

    logger.info(f"Loading embeddings: {embeddings_path}")
    corpus_embeddings = np.load(embeddings_path)
    logger.info(f"Loaded embeddings shape: {corpus_embeddings.shape}")

    logger.info(f"Loading chunk IDs: {ids_path}")
    ids_data = load_json(ids_path)
    chunk_ids = ids_data.get("chunk_ids", [])
    logger.info(f"Loaded {len(chunk_ids)} chunk IDs")

    # Load model
    logger.info(f"Loading model: {model_id}")
    model = load_embedding_model(model_id)

    # Generate triplets
    triplets = []
    stats: dict[str, int | float] = {
        "total_questions": 0,
        "successful": 0,
        "skipped_requires_context": 0,
        "skipped_no_chunk": 0,
        "errors": 0,
        "avg_hard_negatives": 0.0,
        "questions_with_no_hn": 0,
    }

    for i, q in enumerate(questions):
        # Skip requires_context
        if q.get("metadata", {}).get("requires_context", False):
            stats["skipped_requires_context"] += 1
            continue

        # Skip if no chunk_id
        if not q.get("expected_chunk_id"):
            stats["skipped_no_chunk"] += 1
            continue

        stats["total_questions"] += 1

        # Generate hard negatives
        triplet = generate_hard_negatives_for_question(
            q,
            corpus_embeddings,
            chunk_ids,
            model,
            relative_margin,
            top_k,
            final_k,
            exclude_same_doc,
        )

        triplets.append(triplet)

        if triplet.get("error"):
            stats["errors"] += 1
        else:
            stats["successful"] += 1
            if len(triplet["hard_negatives"]) == 0:
                stats["questions_with_no_hn"] += 1

        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(questions)} questions...")

    # Calculate average hard negatives
    successful_triplets = [t for t in triplets if not t.get("error")]
    if successful_triplets:
        total_hn = sum(len(t["hard_negatives"]) for t in successful_triplets)
        stats["avg_hard_negatives"] = round(total_hn / len(successful_triplets), 2)

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "gold_standard_file": str(gs_path),
        "gold_standard_version": gs_data.get("version", "unknown"),
        "embeddings_file": str(embeddings_path),
        "model_id": model_id,
        "parameters": {
            "relative_margin": relative_margin,
            "top_k_candidates": top_k,
            "final_k": final_k,
            "exclude_same_doc": exclude_same_doc,
        },
        "statistics": stats,
        "triplets": triplets,
    }

    return report


def export_training_format(triplets: list[dict], output_path: Path) -> None:
    """
    Export triplets in sentence-transformers training format.

    Format: {"query": str, "positive": str, "negative": str}
    One line per (query, positive, negative) tuple.
    """
    training_data = []
    for t in triplets:
        if t.get("error"):
            continue
        query = t["query"]
        positive = t["positive"]
        for hn in t["hard_negatives"]:
            training_data.append(
                {
                    "query": query,
                    "positive": positive,
                    "negative": hn["chunk_id"],
                }
            )

    save_json({"triplets": training_data, "total": len(training_data)}, output_path)
    logger.info(f"Exported {len(training_data)} training triplets: {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate hard negatives for RAG triplets"
    )

    parser.add_argument(
        "--gs",
        "-g",
        type=Path,
        required=True,
        help="Gold standard JSON file",
    )
    parser.add_argument(
        "--embeddings",
        "-e",
        type=Path,
        required=True,
        help="Corpus embeddings .npy file",
    )
    parser.add_argument(
        "--ids",
        "-i",
        type=Path,
        required=True,
        help="Chunk IDs JSON file",
    )
    parser.add_argument(
        "--chunks",
        "-c",
        type=Path,
        required=True,
        help="Chunks JSON file",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=MODEL_ID,
        help=f"Embedding model ID (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output triplets JSON file",
    )
    parser.add_argument(
        "--training-output",
        "-t",
        type=Path,
        default=None,
        help="Output training format JSON (optional)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=RELATIVE_MARGIN,
        help=f"TopK-PercPos margin (default: {RELATIVE_MARGIN})",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=FINAL_K,
        help=f"Number of hard negatives per question (default: {FINAL_K})",
    )
    parser.add_argument(
        "--include-same-doc",
        action="store_true",
        help="Include chunks from same document (default: exclude)",
    )

    args = parser.parse_args()

    report = generate_triplets(
        args.gs,
        args.embeddings,
        args.ids,
        args.chunks,
        args.model,
        args.margin,
        TOP_K_CANDIDATES,
        args.final_k,
        not args.include_same_doc,
    )

    # Print summary
    stats = report["statistics"]
    logger.info("=" * 60)
    logger.info("HARD NEGATIVES GENERATION REPORT")
    logger.info("=" * 60)
    logger.info(f"Total questions processed: {stats['total_questions']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Skipped (requires_context): {stats['skipped_requires_context']}")
    logger.info(f"Skipped (no chunk): {stats['skipped_no_chunk']}")
    logger.info("-" * 60)
    logger.info(f"Average hard negatives per question: {stats['avg_hard_negatives']}")
    logger.info(f"Questions with 0 hard negatives: {stats['questions_with_no_hn']}")

    # Show sample
    successful = [t for t in report["triplets"] if not t.get("error")]
    if successful:
        logger.info("-" * 60)
        logger.info("Sample triplet:")
        sample = successful[0]
        logger.info(f"  Query: {sample['query'][:60]}...")
        logger.info(
            f"  Positive: {sample['positive']} (score: {sample['positive_score']})"
        )
        logger.info(f"  Hard negatives: {len(sample['hard_negatives'])}")
        for hn in sample["hard_negatives"][:3]:
            logger.info(f"    - {hn['chunk_id']} (score: {hn['score']})")

    # Save report
    save_json(report, args.output)
    logger.info(f"Report saved: {args.output}")

    # Export training format if requested
    if args.training_output:
        export_training_format(report["triplets"], args.training_output)


if __name__ == "__main__":
    main()
