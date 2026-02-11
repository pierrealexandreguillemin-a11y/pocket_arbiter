"""
Validation Semantique par Classe - Pocket Arbiter

Valide les questions du Gold Standard avec des methodes adaptees
par reasoning_class (Know Your RAG - COLING 2025).

ISO Reference:
    - ISO/IEC 42001 A.6.2.2 - Tracabilite validation
    - ISO/IEC 25010 FA-01 - Exactitude fonctionnelle

Seuils industrie:
    - fact_single: Cosine >= 0.85 (NV-Embed-v2)
    - summary: Cosine >= 0.80 (GTE)
    - arithmetic: 100% composants numeriques
    - reasoning: Coverage >= 90% (HotpotQA)

Usage:
    python -m scripts.evaluation.annales.validate_by_class \
        --gs tests/data/gold_standard_annales_fr_v7.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --output tests/data/validation_by_class_report.json
"""

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.pipeline.embeddings import load_embedding_model
from scripts.pipeline.embeddings_config import (
    MODEL_ID,
    PROMPT_DOCUMENT,
    PROMPT_QUERY,
)
from scripts.pipeline.utils import cosine_similarity, load_json, save_json

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Seuils de validation (standards industrie)
THRESHOLD_FACT_SINGLE = 0.85  # NV-Embed-v2 strict threshold
THRESHOLD_SUMMARY = 0.80  # GTE threshold for synthesis
THRESHOLD_REASONING = 0.90  # HotpotQA multi-hop coverage


def validate_fact_single(
    question: dict,
    chunk_text: str,
    model: "SentenceTransformer",
) -> dict:
    """
    Validate fact_single: Cosine(answer, chunk) >= 0.85 (NV-Embed-v2 style).

    For factual extraction, the answer should be directly present in chunk.
    """
    answer = question.get("expected_answer", "")

    # Embed answer and chunk using document prompt (both are text, not queries)
    emb_answer = model.encode(
        f"{PROMPT_DOCUMENT}{answer}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    emb_chunk = model.encode(
        f"{PROMPT_DOCUMENT}{chunk_text}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    score = cosine_similarity(emb_answer, emb_chunk)

    return {
        "method": "fact_single",
        "score": round(score, 4),
        "threshold": THRESHOLD_FACT_SINGLE,
        "passed": score >= THRESHOLD_FACT_SINGLE,
    }


def validate_summary(
    question: dict,
    chunk_text: str,
    model: "SentenceTransformer",
) -> dict:
    """
    Validate summary: Cosine(Q+A, chunk) >= 0.80 (GTE style).

    For synthesis questions, the combined Q+A should relate to chunk.
    """
    q_text = question.get("question", "")
    answer = question.get("expected_answer", "")
    qa = f"{q_text} {answer}"

    # Embed Q+A as query, chunk as document
    emb_qa = model.encode(
        f"{PROMPT_QUERY}{qa}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    emb_chunk = model.encode(
        f"{PROMPT_DOCUMENT}{chunk_text}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    score = cosine_similarity(emb_qa, emb_chunk)

    return {
        "method": "summary",
        "score": round(score, 4),
        "threshold": THRESHOLD_SUMMARY,
        "passed": score >= THRESHOLD_SUMMARY,
    }


def validate_arithmetic(
    question: dict,
    chunk_text: str,
) -> dict:
    """
    Validate arithmetic: Source numbers present in chunk.

    For calculation questions, verify numeric components are extractable.
    """
    answer = question.get("expected_answer", "")

    # Extract numbers from chunk and answer
    # Pattern: integers and decimals with comma or period
    number_pattern = r"\b\d+(?:[.,]\d+)?\b"
    chunk_numbers = set(re.findall(number_pattern, chunk_text))
    answer_numbers = set(re.findall(number_pattern, answer))

    # Normalize numbers (replace comma with period)
    chunk_numbers_norm = {n.replace(",", ".") for n in chunk_numbers}
    answer_numbers_norm = {n.replace(",", ".") for n in answer_numbers}

    # For arithmetic: either answer numbers are in chunk, or answer is computed
    # (e.g., "30 minutes" computed from "15 + 15")
    components_present = (
        answer_numbers_norm.issubset(chunk_numbers_norm)
        or len(answer_numbers_norm) == 0
    )

    return {
        "method": "arithmetic",
        "answer_numbers": sorted(answer_numbers),
        "chunk_numbers": sorted(list(chunk_numbers)[:20]),  # Limit for report
        "components_present": components_present,
        "passed": components_present,
    }


def validate_reasoning(
    question: dict,
    chunk_text: str,
    model: "SentenceTransformer",
) -> dict:
    """
    Validate reasoning: Multi-chunk coverage via Q+A similarity.

    For reasoning questions, check if the chunk covers the reasoning path.
    Uses higher threshold (0.90) for multi-hop questions.
    """
    q_text = question.get("question", "")
    answer = question.get("expected_answer", "")
    qa = f"{q_text} {answer}"

    # Embed Q+A as query, chunk as document
    emb_qa = model.encode(
        f"{PROMPT_QUERY}{qa}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    emb_chunk = model.encode(
        f"{PROMPT_DOCUMENT}{chunk_text}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    score = cosine_similarity(emb_qa, emb_chunk)

    return {
        "method": "reasoning",
        "score": round(score, 4),
        "threshold": THRESHOLD_REASONING,
        "passed": score >= THRESHOLD_REASONING,
        "note": "Multi-hop reasoning requires high coverage",
    }


def validate_question(
    question: dict,
    chunks_map: dict[str, str],
    model: "SentenceTransformer",
) -> dict:
    """
    Validate a single question using class-specific method.

    Args:
        question: Question dict from gold standard.
        chunks_map: Mapping of chunk_id -> chunk_text.
        model: Loaded embedding model.

    Returns:
        Validation result dict.
    """
    q_id = question.get("id", "unknown")
    chunk_id = question.get("expected_chunk_id")
    reasoning_class = question.get("metadata", {}).get("reasoning_class", "fact_single")

    # Check if chunk exists
    if not chunk_id or chunk_id not in chunks_map:
        return {
            "question_id": q_id,
            "reasoning_class": reasoning_class,
            "validation": {
                "method": "error",
                "error": "chunk_not_found",
                "passed": False,
            },
        }

    chunk_text = chunks_map[chunk_id]

    # Apply class-specific validation
    if reasoning_class == "fact_single":
        validation = validate_fact_single(question, chunk_text, model)
    elif reasoning_class == "summary":
        validation = validate_summary(question, chunk_text, model)
    elif reasoning_class == "arithmetic":
        validation = validate_arithmetic(question, chunk_text)
    elif reasoning_class == "reasoning":
        validation = validate_reasoning(question, chunk_text, model)
    else:
        # Default to summary method for unknown classes
        validation = validate_summary(question, chunk_text, model)
        validation["note"] = f"Unknown class '{reasoning_class}', using summary method"

    return {
        "question_id": q_id,
        "reasoning_class": reasoning_class,
        "validation": validation,
    }


def validate_by_class(
    gs_path: Path,
    chunks_path: Path,
    model_id: str = MODEL_ID,
) -> dict:
    """
    Validate all questions in gold standard by reasoning_class.

    Args:
        gs_path: Path to gold standard JSON.
        chunks_path: Path to chunks JSON.
        model_id: HuggingFace model ID for embeddings.

    Returns:
        Validation report with per-question results and statistics.
    """
    logger.info(f"Loading gold standard: {gs_path}")
    gs_data = load_json(gs_path)
    questions = gs_data.get("questions", [])

    logger.info(f"Loading chunks: {chunks_path}")
    chunks_data = load_json(chunks_path)
    chunks = chunks_data.get("chunks", [])

    # Build chunk_id -> text map
    chunks_map = {chunk["id"]: chunk["text"] for chunk in chunks}
    logger.info(f"Loaded {len(chunks_map)} chunks")

    # Load embedding model
    logger.info(f"Loading model: {model_id}")
    model = load_embedding_model(model_id)

    # Validate each question
    results = []
    stats_by_class: dict[str, dict] = {}

    for i, q in enumerate(questions):
        # Skip requires_context questions
        if q.get("metadata", {}).get("requires_context", False):
            continue

        result = validate_question(q, chunks_map, model)
        results.append(result)

        # Update stats
        rc = result["reasoning_class"]
        if rc not in stats_by_class:
            stats_by_class[rc] = {"total": 0, "passed": 0, "failed": 0}

        stats_by_class[rc]["total"] += 1
        if result["validation"].get("passed", False):
            stats_by_class[rc]["passed"] += 1
        else:
            stats_by_class[rc]["failed"] += 1

        if (i + 1) % 50 == 0:
            logger.info(f"Validated {i + 1}/{len(questions)} questions...")

    # Calculate overall stats
    total = sum(s["total"] for s in stats_by_class.values())
    passed = sum(s["passed"] for s in stats_by_class.values())
    pass_rate = (passed / total * 100) if total > 0 else 0

    # Add pass rates per class
    for rc, stats in stats_by_class.items():
        stats["pass_rate_percent"] = (
            round(stats["passed"] / stats["total"] * 100, 2)
            if stats["total"] > 0
            else 0
        )

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "gold_standard_file": str(gs_path),
        "gold_standard_version": gs_data.get("version", "unknown"),
        "model_id": model_id,
        "thresholds": {
            "fact_single": THRESHOLD_FACT_SINGLE,
            "summary": THRESHOLD_SUMMARY,
            "reasoning": THRESHOLD_REASONING,
            "arithmetic": "100% components present",
        },
        "summary": {
            "total_validated": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate_percent": round(pass_rate, 2),
            "status": "PASS" if pass_rate >= 80 else "NEEDS_REVIEW",
        },
        "stats_by_class": stats_by_class,
        "results": results,
    }

    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate gold standard questions by reasoning_class"
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
        help="Output report JSON file (optional)",
    )

    args = parser.parse_args()

    report = validate_by_class(args.gs, args.chunks, args.model)

    # Print summary
    summary = report["summary"]
    logger.info("=" * 60)
    logger.info("VALIDATION BY CLASS REPORT")
    logger.info("=" * 60)
    logger.info(f"Total validated: {summary['total_validated']}")
    logger.info(f"Passed: {summary['passed']} ({summary['pass_rate_percent']}%)")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Status: {summary['status']}")
    logger.info("-" * 60)

    # Per-class breakdown
    logger.info("Per-class breakdown:")
    for rc, stats in report["stats_by_class"].items():
        logger.info(
            f"  {rc}: {stats['passed']}/{stats['total']} "
            f"({stats['pass_rate_percent']}%)"
        )

    # Show some failed examples
    failed = [r for r in report["results"] if not r["validation"].get("passed", False)]
    if failed:
        logger.info("-" * 60)
        logger.info(f"Sample failures ({min(5, len(failed))} of {len(failed)}):")
        for r in failed[:5]:
            logger.info(f"  - {r['question_id']} ({r['reasoning_class']})")
            if "score" in r["validation"]:
                logger.info(
                    f"    Score: {r['validation']['score']} "
                    f"< {r['validation']['threshold']}"
                )

    # Save report if output specified
    if args.output:
        save_json(report, args.output)
        logger.info(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
