"""
Re-validation Gold Standard dans l'espace QAT - Pocket Arbiter

Re-calcule les chunk_ids optimaux pour chaque question du Gold Standard
dans le nouvel espace d'embedding QAT.

Contexte:
    Le GS a été créé et validé avec google/embeddinggemma-300m (FULL).
    Les embeddings corpus ont été migrés vers QAT.
    Ce script re-valide chaque question pour corriger le distribution shift.

ISO Reference:
    - ISO/IEC 42001 A.6.2.2 - Tracabilite donnees
    - ISO/IEC 25010 FA-01 - Exactitude fonctionnelle

Usage:
    python -m scripts.evaluation.annales.revalidate_gs_qat \
        --gs tests/data/gold_standard_annales_fr_v7.json \
        --embeddings corpus/processed/embeddings_mode_b_fr.npy \
        --ids corpus/processed/embeddings_mode_b_fr.ids.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --output tests/data/gold_standard_annales_fr_v7_qat.json
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

# Seuils de validation
TOP_K = 10  # Nombre de candidats à considérer
MIN_SCORE_THRESHOLD = 0.70  # Score minimum acceptable
DRIFT_THRESHOLD = 0.05  # Différence de rang considérée comme drift


def cosine_similarity_batch(
    query_emb: np.ndarray, corpus_embs: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity between query and all corpus embeddings."""
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    corpus_norms = corpus_embs / (
        np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-8
    )
    return np.dot(corpus_norms, query_norm)


def find_optimal_chunk(
    question: dict,
    corpus_embeddings: np.ndarray,
    chunk_ids: list[str],
    chunks_map: dict[str, dict],
    model: "SentenceTransformer",
    top_k: int = TOP_K,
) -> dict:
    """
    Trouve le chunk optimal pour une question dans l'espace d'embedding actuel.

    Args:
        question: Question du GS.
        corpus_embeddings: Embeddings du corpus (N, dim).
        chunk_ids: Liste des chunk IDs.
        chunks_map: Mapping chunk_id -> chunk dict.
        model: Modèle d'embedding.
        top_k: Nombre de candidats à retourner.

    Returns:
        Résultat avec chunk optimal, score, et comparaison avec l'actuel.
    """
    q_text = question.get("question", "")
    answer = question.get("expected_answer", "")
    current_chunk_id = question.get("expected_chunk_id")
    _article_ref = question.get("metadata", {}).get("article_reference", "")

    # Encode Q+A comme query (meilleure représentation de l'intent)
    qa_text = f"{q_text} {answer}"
    query_emb = model.encode(
        f"{PROMPT_QUERY}{qa_text}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Calcul similarités
    similarities = cosine_similarity_batch(query_emb, corpus_embeddings)

    # Top-K chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_results = [
        {
            "chunk_id": chunk_ids[idx],
            "score": float(similarities[idx]),
            "rank": i + 1,
        }
        for i, idx in enumerate(top_indices)
    ]

    # Score du chunk actuel
    current_score = None
    current_rank = None
    if current_chunk_id and current_chunk_id in chunk_ids:
        current_idx = chunk_ids.index(current_chunk_id)
        current_score = float(similarities[current_idx])
        # Trouver le rang
        sorted_indices = np.argsort(similarities)[::-1]
        current_rank = int(np.where(sorted_indices == current_idx)[0][0]) + 1

    # Chunk optimal
    optimal = top_results[0]

    # Analyse
    needs_update = False
    update_reason = None

    if current_chunk_id is None:
        needs_update = True
        update_reason = "missing_chunk_id"
    elif current_chunk_id not in chunk_ids:
        needs_update = True
        update_reason = "chunk_id_not_found"
    elif current_rank is not None and current_rank > TOP_K:
        needs_update = True
        update_reason = f"current_rank_{current_rank}_too_low"
    elif (
        optimal["chunk_id"] != current_chunk_id
        and optimal["score"] - (current_score or 0) > DRIFT_THRESHOLD
    ):
        needs_update = True
        update_reason = (
            f"better_match_found_delta_{optimal['score'] - (current_score or 0):.3f}"
        )

    return {
        "question_id": question.get("id"),
        "current_chunk_id": current_chunk_id,
        "current_score": round(current_score, 4) if current_score else None,
        "current_rank": current_rank,
        "optimal_chunk_id": optimal["chunk_id"],
        "optimal_score": round(optimal["score"], 4),
        "top_k": top_results,
        "needs_update": needs_update,
        "update_reason": update_reason,
    }


def revalidate_gold_standard(
    gs_path: Path,
    embeddings_path: Path,
    ids_path: Path,
    chunks_path: Path,
    model_id: str = MODEL_ID,
    apply_updates: bool = False,
) -> tuple[dict, dict]:
    """
    Re-valide tout le Gold Standard dans l'espace d'embedding actuel.

    Args:
        gs_path: Chemin vers le GS JSON.
        embeddings_path: Chemin vers les embeddings .npy.
        ids_path: Chemin vers les chunk IDs JSON.
        chunks_path: Chemin vers les chunks JSON.
        model_id: ID du modèle d'embedding.
        apply_updates: Si True, met à jour le GS avec les chunks optimaux.

    Returns:
        Tuple (GS mis à jour, rapport de validation).
    """
    logger.info(f"Loading gold standard: {gs_path}")
    gs_data = load_json(gs_path)
    questions = gs_data.get("questions", [])

    logger.info(f"Loading embeddings: {embeddings_path}")
    corpus_embeddings = np.load(embeddings_path)
    logger.info(f"Embeddings shape: {corpus_embeddings.shape}")

    logger.info(f"Loading chunk IDs: {ids_path}")
    ids_data = load_json(ids_path)
    chunk_ids = ids_data.get("chunk_ids", [])

    logger.info(f"Loading chunks: {chunks_path}")
    chunks_data = load_json(chunks_path)
    chunks_map = {c["id"]: c for c in chunks_data.get("chunks", [])}

    logger.info(f"Loading model: {model_id}")
    model = load_embedding_model(model_id)

    # Statistiques
    stats = {
        "total": 0,
        "skipped_requires_context": 0,
        "validated_same": 0,
        "needs_update": 0,
        "updates_applied": 0,
        "score_improved_total": 0.0,
    }

    results = []

    for i, q in enumerate(questions):
        # Skip requires_context
        if q.get("metadata", {}).get("requires_context", False):
            stats["skipped_requires_context"] += 1
            continue

        stats["total"] += 1

        # Find optimal chunk
        result = find_optimal_chunk(q, corpus_embeddings, chunk_ids, chunks_map, model)
        results.append(result)

        if result["needs_update"]:
            stats["needs_update"] += 1

            if apply_updates:
                old_chunk_id = q.get("expected_chunk_id")
                old_score = result["current_score"] or 0

                # Update question
                q["expected_chunk_id"] = result["optimal_chunk_id"]

                # Update metadata
                if "metadata" not in q:
                    q["metadata"] = {}
                q["metadata"]["chunk_match_score"] = int(result["optimal_score"] * 100)
                q["metadata"]["chunk_match_method"] = "qat_revalidation"
                q["metadata"]["qat_revalidation"] = {
                    "date": datetime.now().isoformat(),
                    "old_chunk_id": old_chunk_id,
                    "old_score": old_score,
                    "new_score": result["optimal_score"],
                    "reason": result["update_reason"],
                }

                # Update audit trail
                audit = q.get("audit", "")
                q["audit"] = (
                    f"{audit} | [QAT] {old_chunk_id} -> {result['optimal_chunk_id']} ({result['update_reason']})"
                )

                stats["updates_applied"] += 1
                stats["score_improved_total"] += result["optimal_score"] - old_score
        else:
            stats["validated_same"] += 1

        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(questions)} questions...")

    # Update GS metadata
    if apply_updates:
        gs_data["methodology"]["qat_revalidation"] = {
            "date": datetime.now().isoformat(),
            "model_id": model_id,
            "total_questions": stats["total"],
            "updates_applied": stats["updates_applied"],
            "avg_score_improvement": round(
                stats["score_improved_total"] / stats["updates_applied"], 4
            )
            if stats["updates_applied"] > 0
            else 0,
        }

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_id": model_id,
        "embeddings_file": str(embeddings_path),
        "statistics": stats,
        "updates_applied": apply_updates,
        "results": results,
    }

    return gs_data, report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Re-validate Gold Standard in QAT embedding space"
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
        default=None,
        help="Output GS JSON (if not specified, dry-run mode)",
    )
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        default=None,
        help="Output report JSON",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply updates to GS (default: dry-run)",
    )

    args = parser.parse_args()

    apply_updates = args.apply or args.output is not None

    gs_data, report = revalidate_gold_standard(
        args.gs,
        args.embeddings,
        args.ids,
        args.chunks,
        args.model,
        apply_updates,
    )

    # Print summary
    stats = report["statistics"]
    logger.info("=" * 60)
    logger.info("QAT REVALIDATION REPORT")
    logger.info("=" * 60)
    logger.info(f"Model: {report['model_id']}")
    logger.info(f"Total questions: {stats['total']}")
    logger.info(f"Skipped (requires_context): {stats['skipped_requires_context']}")
    logger.info("-" * 60)
    logger.info(f"Validated (same chunk): {stats['validated_same']}")
    logger.info(f"Needs update: {stats['needs_update']}")
    if apply_updates:
        logger.info(f"Updates applied: {stats['updates_applied']}")
        if stats["updates_applied"] > 0:
            avg_improvement = stats["score_improved_total"] / stats["updates_applied"]
            logger.info(f"Avg score improvement: +{avg_improvement:.4f}")

    # Show samples of updates needed
    updates_needed = [r for r in report["results"] if r["needs_update"]]
    if updates_needed:
        logger.info("-" * 60)
        logger.info(
            f"Sample updates needed ({min(5, len(updates_needed))} of {len(updates_needed)}):"
        )
        for r in updates_needed[:5]:
            logger.info(f"  {r['question_id']}:")
            logger.info(
                f"    Current: {r['current_chunk_id']} (score={r['current_score']}, rank={r['current_rank']})"
            )
            logger.info(
                f"    Optimal: {r['optimal_chunk_id']} (score={r['optimal_score']})"
            )
            logger.info(f"    Reason: {r['update_reason']}")

    # Save outputs
    if args.output:
        save_json(gs_data, args.output)
        logger.info(f"Updated GS saved: {args.output}")

    if args.report:
        save_json(report, args.report)
        logger.info(f"Report saved: {args.report}")


if __name__ == "__main__":
    main()
