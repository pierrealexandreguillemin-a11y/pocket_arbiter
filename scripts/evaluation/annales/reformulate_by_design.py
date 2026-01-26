"""
Reformulation BY DESIGN - Pocket Arbiter

Reformule les questions en langage naturel ancre dans le chunk source.
Validation anti-drift avec seuils standards industrie.

ISO Reference:
    - ISO/IEC 42001 A.6.2.2 - Tracabilite reformulation
    - ISO/IEC 25010 FA-01 - Preservation sens

Standards:
    - Cosine >= 0.85 (ParKQ, NV-Embed-v2) pour equivalence semantique
    - 100% answerability: reponse extractable du chunk
    - Preservation question_type et reasoning_class

Usage:
    python -m scripts.evaluation.annales.reformulate_by_design \
        --gs tests/data/gold_standard_annales_fr_v7.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --output tests/data/gold_standard_annales_fr_v8.json
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

# Anti-drift threshold (ParKQ, NV-Embed-v2)
THRESHOLD_REFORMULATION = 0.85


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def validate_reformulation(
    original: str,
    reformulated: str,
    model: "SentenceTransformer",
) -> dict:
    """
    Validate reformulation preserves semantic meaning.

    Uses query-to-query similarity to ensure paraphrase equivalence.

    Args:
        original: Original question text.
        reformulated: Reformulated question text.
        model: Embedding model.

    Returns:
        Validation result with score and pass status.
    """
    # Both are queries, use query prompt
    emb_orig = model.encode(
        f"{PROMPT_QUERY}{original}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    emb_reform = model.encode(
        f"{PROMPT_QUERY}{reformulated}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    score = cosine_similarity(emb_orig, emb_reform)

    return {
        "similarity_score": round(score, 4),
        "threshold": THRESHOLD_REFORMULATION,
        "passed": score >= THRESHOLD_REFORMULATION,
    }


def check_answerability(
    answer: str,
    chunk_text: str,
) -> dict:
    """
    Check if answer is extractable from chunk.

    Args:
        answer: Expected answer text.
        chunk_text: Chunk content.

    Returns:
        Answerability check result.
    """
    answer_lower = answer.lower().strip()
    chunk_lower = chunk_text.lower()

    # Direct presence
    answer_in_chunk = answer_lower in chunk_lower

    # Key words coverage
    answer_words = [w for w in answer_lower.split() if len(w) > 3]
    if answer_words:
        words_found = sum(1 for w in answer_words if w in chunk_lower)
        word_coverage = words_found / len(answer_words)
    else:
        word_coverage = 1.0 if answer_in_chunk else 0.0

    return {
        "answer_in_chunk": answer_in_chunk,
        "word_coverage": round(word_coverage, 2),
        "passed": answer_in_chunk or word_coverage >= 0.8,
    }


def generate_natural_question(
    original_question: str,
    answer: str,
    chunk_text: str,
    article_reference: str | None = None,
) -> str:
    """
    Generate natural language question grounded in chunk.

    This is a template-based approach. For production, use an LLM.

    Args:
        original_question: Original MCQ question.
        answer: Expected answer.
        chunk_text: Source chunk content.
        article_reference: Article reference if available.

    Returns:
        Reformulated natural question.
    """
    # Clean up original question
    q = original_question.strip()

    # Remove MCQ-specific language
    mcq_patterns = [
        r"Quelle proposition parmi les suivantes",
        r"Parmi les propositions suivantes",
        r"Laquelle des propositions suivantes",
        r"Quelle affirmation est (vraie|fausse|correcte|incorrecte)",
        r"ne correspond pas",
        r"est (vraie|fausse|correcte|incorrecte)",
    ]

    import re

    for pattern in mcq_patterns:
        q = re.sub(pattern, "", q, flags=re.IGNORECASE)

    # Clean up punctuation
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"^[,;:\s]+", "", q)

    # If question became too short or empty, fall back to topic extraction
    if len(q) < 20:
        # Extract topic from answer or article reference
        if article_reference:
            topic = re.sub(r"^[A-Z]+\s*-\s*", "", article_reference)
            topic = re.sub(r"Art\.?\s*\d+.*$", "", topic).strip()
            q = f"Selon le reglement, {topic.lower()} ?"
        else:
            # Use first part of answer as topic
            q = f"Quelle est la regle concernant {answer[:50].lower()} ?"

    # Ensure ends with question mark
    q = q.rstrip(".!,;:")
    if not q.endswith("?"):
        q += " ?"

    return q


def reformulate_question(
    question: dict,
    chunk_text: str,
    model: "SentenceTransformer",
) -> dict:
    """
    Reformulate a single question with validation.

    Args:
        question: Original question dict.
        chunk_text: Source chunk text.
        model: Embedding model for validation.

    Returns:
        Reformulation result with new question and validation.
    """
    q_id = question.get("id", "unknown")
    original = question.get("question", "")
    answer = question.get("expected_answer", "")
    article_ref = question.get("metadata", {}).get("article_reference")

    # Generate reformulation
    reformulated = generate_natural_question(original, answer, chunk_text, article_ref)

    # Validate semantic preservation
    semantic_validation = validate_reformulation(original, reformulated, model)

    # Check answerability
    answerability = check_answerability(answer, chunk_text)

    # Overall validation
    passed = semantic_validation["passed"] and answerability["passed"]

    return {
        "question_id": q_id,
        "original_question": original,
        "reformulated_question": reformulated,
        "validation": {
            "semantic": semantic_validation,
            "answerability": answerability,
            "overall_passed": passed,
        },
    }


def reformulate_gold_standard(
    gs_path: Path,
    chunks_path: Path,
    model_id: str = MODEL_ID,
    skip_requires_context: bool = True,
) -> tuple[dict, dict]:
    """
    Reformulate all testable questions in gold standard.

    Args:
        gs_path: Path to gold standard JSON.
        chunks_path: Path to chunks JSON.
        model_id: Embedding model ID.
        skip_requires_context: Skip questions marked requires_context.

    Returns:
        Tuple of (updated gold standard data, reformulation report).
    """
    logger.info(f"Loading gold standard: {gs_path}")
    gs_data = load_json(gs_path)
    questions = gs_data.get("questions", [])

    logger.info(f"Loading chunks: {chunks_path}")
    chunks_data = load_json(chunks_path)
    chunks_map = {c["id"]: c["text"] for c in chunks_data.get("chunks", [])}

    # Load embedding model
    logger.info(f"Loading model: {model_id}")
    model = load_embedding_model(model_id)

    # Process questions
    reformulation_results = []
    skipped = 0
    passed = 0
    failed = 0

    for i, q in enumerate(questions):
        # Skip requires_context
        if skip_requires_context and q.get("metadata", {}).get(
            "requires_context", False
        ):
            skipped += 1
            continue

        # Get chunk
        chunk_id = q.get("expected_chunk_id")
        if not chunk_id or chunk_id not in chunks_map:
            skipped += 1
            continue

        chunk_text = chunks_map[chunk_id]

        # Reformulate
        result = reformulate_question(q, chunk_text, model)
        reformulation_results.append(result)

        if result["validation"]["overall_passed"]:
            passed += 1
            # Update question in place
            q["original_annales"] = q["question"]
            q["question"] = result["reformulated_question"]
            if "metadata" not in q:
                q["metadata"] = {}
            q["metadata"]["reformulation_method"] = "template_by_design"
            q["metadata"]["reformulation_date"] = datetime.now().strftime("%Y-%m-%d")
            q["metadata"]["reformulation_score"] = result["validation"]["semantic"][
                "similarity_score"
            ]
        else:
            failed += 1
            # Keep original but mark as needing manual reformulation
            if "metadata" not in q:
                q["metadata"] = {}
            q["metadata"]["reformulation_status"] = "needs_manual_review"

        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(questions)} questions...")

    # Update version
    old_version = gs_data.get("version", "7.0.0")
    new_version = old_version.replace("7.", "7.5.")  # Minor version bump

    gs_data["version"] = new_version
    gs_data["methodology"]["reformulation"] = {
        "method": "template_by_design",
        "date": datetime.now().isoformat(),
        "model_id": model_id,
        "threshold": THRESHOLD_REFORMULATION,
        "total_reformulated": passed,
        "needs_review": failed,
        "skipped": skipped,
    }

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "gold_standard_file": str(gs_path),
        "model_id": model_id,
        "threshold": THRESHOLD_REFORMULATION,
        "summary": {
            "total_questions": len(questions),
            "reformulated": passed,
            "needs_review": failed,
            "skipped": skipped,
            "success_rate_percent": round(passed / (passed + failed) * 100, 2)
            if (passed + failed) > 0
            else 0,
        },
        "results": reformulation_results,
    }

    return gs_data, report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Reformulate gold standard questions in natural language"
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
        help="Output gold standard JSON (optional, updates in place if not provided)",
    )
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        default=None,
        help="Output reformulation report JSON (optional)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate report, don't modify gold standard",
    )

    args = parser.parse_args()

    gs_data, report = reformulate_gold_standard(args.gs, args.chunks, args.model)

    # Print summary
    summary = report["summary"]
    logger.info("=" * 60)
    logger.info("REFORMULATION REPORT")
    logger.info("=" * 60)
    logger.info(f"Total questions: {summary['total_questions']}")
    logger.info(f"Reformulated: {summary['reformulated']}")
    logger.info(f"Needs review: {summary['needs_review']}")
    logger.info(f"Skipped: {summary['skipped']}")
    logger.info(f"Success rate: {summary['success_rate_percent']}%")

    # Show samples
    passed_results = [r for r in report["results"] if r["validation"]["overall_passed"]]
    failed_results = [
        r for r in report["results"] if not r["validation"]["overall_passed"]
    ]

    if passed_results:
        logger.info("-" * 60)
        logger.info("Sample reformulations:")
        for r in passed_results[:3]:
            logger.info(f"  Original: {r['original_question'][:60]}...")
            logger.info(f"  Reform:   {r['reformulated_question'][:60]}...")
            logger.info(
                f"  Score:    {r['validation']['semantic']['similarity_score']}"
            )
            logger.info("")

    if failed_results:
        logger.info("-" * 60)
        logger.info(f"Failed reformulations ({len(failed_results)}):")
        for r in failed_results[:3]:
            logger.info(
                f"  - {r['question_id']}: score={r['validation']['semantic']['similarity_score']}"
            )

    # Save outputs
    if not args.dry_run:
        output_path = args.output or args.gs
        save_json(gs_data, output_path)
        logger.info(f"Gold standard saved: {output_path}")

    if args.report:
        save_json(report, args.report)
        logger.info(f"Report saved: {args.report}")


if __name__ == "__main__":
    main()
