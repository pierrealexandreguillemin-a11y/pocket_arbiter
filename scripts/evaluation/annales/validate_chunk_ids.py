"""
Validation des chunk_ids du Gold Standard - Pocket Arbiter

Verifie que tous les expected_chunk_id du gold standard existent
dans le corpus de chunks.

ISO Reference:
    - ISO/IEC 42001 A.6.2.2 - Tracabilite donnees
    - ISO/IEC 25010 FA-01 - Exactitude fonctionnelle

Usage:
    python -m scripts.evaluation.annales.validate_chunk_ids \
        --gs tests/data/gold_standard_annales_fr_v7.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --output tests/data/chunk_id_validation_report.json
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from scripts.pipeline.utils import load_json, save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_chunk_ids(
    gs_path: Path,
    chunks_path: Path,
) -> dict:
    """
    Valide que tous les expected_chunk_id du GS existent dans le corpus.

    Args:
        gs_path: Chemin vers le gold standard JSON.
        chunks_path: Chemin vers le fichier chunks JSON.

    Returns:
        Rapport de validation avec statistiques et liste des IDs invalides.
    """
    logger.info(f"Loading gold standard: {gs_path}")
    gs_data = load_json(gs_path)
    questions = gs_data.get("questions", [])

    logger.info(f"Loading chunks: {chunks_path}")
    chunks_data = load_json(chunks_path)
    chunks = chunks_data.get("chunks", [])

    # Build set of valid chunk IDs
    valid_chunk_ids = {chunk["id"] for chunk in chunks}
    logger.info(f"Found {len(valid_chunk_ids)} unique chunk IDs in corpus")

    # Validate each question's chunk_id
    valid_questions = []
    invalid_questions = []
    requires_context_count = 0

    for q in questions:
        chunk_id = q.get("expected_chunk_id")
        q_id = q.get("id", "unknown")

        # Skip requires_context questions (they may have special handling)
        if q.get("metadata", {}).get("requires_context", False):
            requires_context_count += 1
            valid_questions.append(q_id)
            continue

        if not chunk_id:
            invalid_questions.append(
                {
                    "question_id": q_id,
                    "expected_chunk_id": None,
                    "reason": "missing_chunk_id",
                    "question_preview": q.get("question", "")[:80],
                }
            )
        elif chunk_id not in valid_chunk_ids:
            invalid_questions.append(
                {
                    "question_id": q_id,
                    "expected_chunk_id": chunk_id,
                    "reason": "chunk_id_not_found",
                    "question_preview": q.get("question", "")[:80],
                }
            )
        else:
            valid_questions.append(q_id)

    # Generate report
    total = len(questions)
    valid_count = len(valid_questions)
    invalid_count = len(invalid_questions)
    validity_rate = (valid_count / total * 100) if total > 0 else 0

    report = {
        "timestamp": datetime.now().isoformat(),
        "gold_standard_file": str(gs_path),
        "gold_standard_version": gs_data.get("version", "unknown"),
        "chunks_file": str(chunks_path),
        "summary": {
            "total_questions": total,
            "valid_chunk_ids": valid_count,
            "invalid_chunk_ids": invalid_count,
            "requires_context_skipped": requires_context_count,
            "validity_rate_percent": round(validity_rate, 2),
            "status": "PASS" if invalid_count == 0 else "FAIL",
        },
        "invalid_questions": invalid_questions,
    }

    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate gold standard chunk_ids against corpus"
    )

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
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output report JSON file (optional)",
    )

    args = parser.parse_args()

    report = validate_chunk_ids(args.gs, args.chunks)

    # Print summary
    summary = report["summary"]
    logger.info("=" * 50)
    logger.info(f"Total questions: {summary['total_questions']}")
    logger.info(f"Valid chunk_ids: {summary['valid_chunk_ids']}")
    logger.info(f"Invalid chunk_ids: {summary['invalid_chunk_ids']}")
    logger.info(f"Validity rate: {summary['validity_rate_percent']}%")
    logger.info(f"Status: {summary['status']}")

    if report["invalid_questions"]:
        logger.warning("Invalid chunk_ids found:")
        for inv in report["invalid_questions"][:10]:  # Show first 10
            logger.warning(f"  - {inv['question_id']}: {inv['reason']}")
        if len(report["invalid_questions"]) > 10:
            logger.warning(f"  ... and {len(report['invalid_questions']) - 10} more")

    # Save report if output specified
    if args.output:
        save_json(report, args.output)
        logger.info(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
