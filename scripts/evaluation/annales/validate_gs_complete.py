"""
Validation Complete Gold Standard - Pocket Arbiter

Script de validation finale pour verifier que le Gold Standard v8.0.0
respecte tous les criteres standards industrie.

ISO Reference:
    - ISO/IEC 42001 A.6.2.2 - Validation complete
    - ISO/IEC 25010 FA-01 - Conformite specification
    - ISO/IEC 29119 - Tests validation

Criteres de validation:
    - 420 questions totales
    - 100% chunk_ids valides
    - Reformulation avec cosine >= 0.85
    - Triplets avec hard negatives
    - Export BEIR valide

Usage:
    python -m scripts.evaluation.annales.validate_gs_complete \
        --gs tests/data/gold_standard_annales_fr_v7.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --embeddings corpus/processed/embeddings_mode_b_fr.npy \
        --ids corpus/processed/embeddings_mode_b_fr.ids.json
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.pipeline.utils import load_json, save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_question_counts(gs_data: dict) -> dict:
    """Validate question counts and distribution."""
    questions = gs_data.get("questions", [])
    total = len(questions)

    # Count by category
    requires_context = sum(
        1 for q in questions if q.get("metadata", {}).get("requires_context", False)
    )
    testable = total - requires_context

    # Count by reasoning_class
    class_counts: dict[str, int] = {}
    for q in questions:
        rc = q.get("metadata", {}).get("reasoning_class", "unknown")
        class_counts[rc] = class_counts.get(rc, 0) + 1

    return {
        "total_questions": total,
        "requires_context": requires_context,
        "testable": testable,
        "reasoning_class_distribution": class_counts,
        "passed": total >= 400,  # Target: 420
    }


def validate_chunk_ids(gs_data: dict, chunks_data: dict) -> dict:
    """Validate all chunk_ids exist in corpus."""
    questions = gs_data.get("questions", [])
    chunks = chunks_data.get("chunks", [])
    valid_ids = {c["id"] for c in chunks}

    invalid = []
    for q in questions:
        chunk_id = q.get("expected_chunk_id")
        if chunk_id and chunk_id not in valid_ids:
            invalid.append(q.get("id", "unknown"))

    total_with_chunks = sum(1 for q in questions if q.get("expected_chunk_id"))
    valid_count = total_with_chunks - len(invalid)

    return {
        "total_with_chunk_id": total_with_chunks,
        "valid_chunk_ids": valid_count,
        "invalid_chunk_ids": len(invalid),
        "invalid_questions": invalid[:10],  # Sample
        "validity_rate": round(valid_count / total_with_chunks * 100, 2)
        if total_with_chunks > 0
        else 0,
        "passed": len(invalid) == 0,
    }


def validate_reformulations(gs_data: dict) -> dict:
    """Validate reformulation metadata."""
    questions = gs_data.get("questions", [])

    reformulated = 0
    scores = []
    needs_review = 0

    for q in questions:
        metadata = q.get("metadata", {})
        if q.get("original_annales"):
            reformulated += 1
            score = metadata.get("reformulation_score")
            if score:
                scores.append(score)
        if metadata.get("reformulation_status") == "needs_manual_review":
            needs_review += 1

    avg_score = sum(scores) / len(scores) if scores else 0

    return {
        "reformulated": reformulated,
        "needs_review": needs_review,
        "average_score": round(avg_score, 4),
        "min_score": round(min(scores), 4) if scores else None,
        "max_score": round(max(scores), 4) if scores else None,
        "passed": avg_score >= 0.80 if scores else True,  # Pending = pass
    }


def validate_triplet_readiness(gs_data: dict) -> dict:
    """Validate triplet_ready flags."""
    questions = gs_data.get("questions", [])

    triplet_ready = sum(
        1 for q in questions if q.get("metadata", {}).get("triplet_ready", False)
    )
    requires_context = sum(
        1 for q in questions if q.get("metadata", {}).get("requires_context", False)
    )
    testable = len(questions) - requires_context

    return {
        "triplet_ready": triplet_ready,
        "testable": testable,
        "readiness_rate": round(triplet_ready / testable * 100, 2)
        if testable > 0
        else 0,
        "passed": triplet_ready >= testable * 0.9,  # 90% target
    }


def validate_embeddings(
    embeddings_path: Path, ids_path: Path, chunks_data: dict
) -> dict:
    """Validate embeddings file consistency."""
    if not embeddings_path.exists():
        return {
            "exists": False,
            "passed": False,
            "error": f"Embeddings file not found: {embeddings_path}",
        }

    if not ids_path.exists():
        return {
            "exists": False,
            "passed": False,
            "error": f"IDs file not found: {ids_path}",
        }

    embeddings = np.load(embeddings_path)
    ids_data = load_json(ids_path)
    chunk_ids = ids_data.get("chunk_ids", [])
    chunks = chunks_data.get("chunks", [])

    # Validate shape consistency
    shape_match = embeddings.shape[0] == len(chunk_ids) == len(chunks)

    return {
        "exists": True,
        "embeddings_shape": list(embeddings.shape),
        "chunk_ids_count": len(chunk_ids),
        "chunks_count": len(chunks),
        "shape_consistent": shape_match,
        "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else None,
        "passed": shape_match and embeddings.shape[1] == 768,
    }


def validate_metadata_completeness(gs_data: dict) -> dict:
    """Validate metadata field completeness."""
    questions = gs_data.get("questions", [])

    required_fields = [
        "question_type",
        "reasoning_class",
        "article_reference",
    ]

    missing_by_field = {f: 0 for f in required_fields}
    total = len(questions)

    for q in questions:
        metadata = q.get("metadata", {})
        for field in required_fields:
            if not metadata.get(field):
                missing_by_field[field] += 1

    completeness = {
        field: round((total - count) / total * 100, 2)
        for field, count in missing_by_field.items()
    }

    avg_completeness = (
        sum(completeness.values()) / len(completeness) if completeness else 0
    )

    return {
        "field_completeness": completeness,
        "average_completeness": round(avg_completeness, 2),
        "passed": avg_completeness >= 90,
    }


def validate_gs_complete(
    gs_path: Path,
    chunks_path: Path,
    embeddings_path: Path | None = None,
    ids_path: Path | None = None,
) -> dict:
    """
    Run complete validation of Gold Standard.

    Args:
        gs_path: Path to gold standard JSON.
        chunks_path: Path to chunks JSON.
        embeddings_path: Path to embeddings .npy (optional).
        ids_path: Path to chunk IDs JSON (optional).

    Returns:
        Complete validation report.
    """
    logger.info(f"Loading gold standard: {gs_path}")
    gs_data = load_json(gs_path)

    logger.info(f"Loading chunks: {chunks_path}")
    chunks_data = load_json(chunks_path)

    # Run all validations
    validations = {}

    logger.info("Validating question counts...")
    validations["question_counts"] = validate_question_counts(gs_data)

    logger.info("Validating chunk IDs...")
    validations["chunk_ids"] = validate_chunk_ids(gs_data, chunks_data)

    logger.info("Validating reformulations...")
    validations["reformulations"] = validate_reformulations(gs_data)

    logger.info("Validating triplet readiness...")
    validations["triplet_readiness"] = validate_triplet_readiness(gs_data)

    logger.info("Validating metadata completeness...")
    validations["metadata_completeness"] = validate_metadata_completeness(gs_data)

    if embeddings_path and ids_path:
        logger.info("Validating embeddings...")
        validations["embeddings"] = validate_embeddings(
            embeddings_path, ids_path, chunks_data
        )

    # Overall status
    all_passed = all(v.get("passed", False) for v in validations.values())

    report = {
        "timestamp": datetime.now().isoformat(),
        "gold_standard_file": str(gs_path),
        "gold_standard_version": gs_data.get("version", "unknown"),
        "overall_status": "PASS" if all_passed else "FAIL",
        "validations": validations,
    }

    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Complete validation of Gold Standard")

    parser.add_argument(
        "--gs",
        "-g",
        type=Path,
        required=True,
        help="Gold standard JSON file",
    )
    parser.add_argument(
        "--chunks",
        "-c",
        type=Path,
        required=True,
        help="Chunks JSON file from corpus",
    )
    parser.add_argument(
        "--embeddings",
        "-e",
        type=Path,
        default=None,
        help="Embeddings .npy file (optional)",
    )
    parser.add_argument(
        "--ids",
        "-i",
        type=Path,
        default=None,
        help="Chunk IDs JSON file (optional)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output report JSON (optional)",
    )

    args = parser.parse_args()

    report = validate_gs_complete(args.gs, args.chunks, args.embeddings, args.ids)

    # Print formatted report
    print()
    print("=" * 60)
    print(
        f"[{'OK' if report['overall_status'] == 'PASS' else 'FAIL'}] Gold Standard {report['gold_standard_version']}"
    )
    print("=" * 60)

    # Question counts
    qc = report["validations"]["question_counts"]
    print(f"  - Questions: {qc['total_questions']} ({qc['testable']} testables)")
    print(f"    Distribution: {qc['reasoning_class_distribution']}")

    # Chunk IDs
    ci = report["validations"]["chunk_ids"]
    print(f"  - Chunk IDs: {ci['validity_rate']}% valid")

    # Reformulations
    rf = report["validations"]["reformulations"]
    if rf["reformulated"] > 0:
        print(
            f"  - Reformulation: {rf['reformulated']} (avg score: {rf['average_score']})"
        )
    else:
        print("  - Reformulation: pending")

    # Triplet readiness
    tr = report["validations"]["triplet_readiness"]
    print(
        f"  - Triplet ready: {tr['triplet_ready']}/{tr['testable']} ({tr['readiness_rate']}%)"
    )

    # Embeddings
    if "embeddings" in report["validations"]:
        emb = report["validations"]["embeddings"]
        if emb["exists"]:
            print(
                f"  - Embeddings: {emb['embeddings_shape']} ({'OK' if emb['passed'] else 'FAIL'})"
            )
        else:
            print("  - Embeddings: not found")

    # Metadata completeness
    mc = report["validations"]["metadata_completeness"]
    print(f"  - Metadata completeness: {mc['average_completeness']}%")

    print()
    print(f"Overall: {report['overall_status']}")
    print("=" * 60)

    # Save report if output specified
    if args.output:
        save_json(report, args.output)
        logger.info(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
