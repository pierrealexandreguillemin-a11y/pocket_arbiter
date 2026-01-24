#!/usr/bin/env python3
"""
Etape 1: Mapping Pages → Chunks (UNIFIED_TRAINING_DATA_SPEC.md)

Mappe les questions du Gold Standard v6.7.0 aux chunks du corpus Mode B.
Utilise expected_docs + expected_pages pour trouver le chunk exact.

ISO 42001 A.6.2.2 - Provenance tracable
ISO 29119-3 - Test data documented

Usage:
    python -m scripts.training.unified.map_pages_to_chunks \
        --gold-standard tests/data/gold_standard_annales_fr.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --output data/training/unified/gs_with_chunks.json
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Default paths
DEFAULT_GS = PROJECT_ROOT / "tests" / "data" / "gold_standard_annales_fr.json"
DEFAULT_CHUNKS = PROJECT_ROOT / "corpus" / "processed" / "chunks_mode_b_fr.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training" / "unified" / "gs_with_chunks.json"


def normalize_source_name(source: str) -> str:
    """Normalize source name for matching by removing accents."""
    replacements = [
        ("é", "e"), ("è", "e"), ("ê", "e"), ("ë", "e"),
        ("à", "a"), ("â", "a"),
        ("ù", "u"), ("û", "u"),
        ("î", "i"), ("ï", "i"),
        ("ô", "o"),
        ("ç", "c"),
    ]
    result = source.lower()
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """Load chunks from JSON file."""
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("chunks", data) if isinstance(data, dict) else data


def load_gold_standard(gs_path: Path) -> dict[str, Any]:
    """Load gold standard JSON file."""
    with open(gs_path, encoding="utf-8") as f:
        return json.load(f)


def find_chunks_for_pages(
    chunks: list[dict[str, Any]],
    source_pattern: str,
    pages: list[int],
) -> list[dict[str, Any]]:
    """
    Find all chunks matching the source pattern and containing any of the pages.

    Args:
        chunks: List of chunk dictionaries
        source_pattern: Pattern to match source (e.g., "LA-octobre2025.pdf")
        pages: List of page numbers

    Returns:
        List of matching chunks sorted by page
    """
    source_norm = normalize_source_name(source_pattern)
    pages_set = set(pages)

    matching = []
    for chunk in chunks:
        chunk_source = chunk.get("source", "")
        chunk_pages = set(chunk.get("pages", [chunk.get("page", -1)]))

        # Match source (partial match allowed)
        if source_norm not in normalize_source_name(chunk_source):
            continue

        # Match any page
        if chunk_pages.intersection(pages_set):
            matching.append(chunk)

    # Sort by first page
    matching.sort(key=lambda c: min(c.get("pages", [c.get("page", 999)])))
    return matching


def score_chunk_for_question(
    chunk: dict[str, Any],
    question: dict[str, Any],
) -> float:
    """
    Score a chunk's relevance to a question using multiple signals.

    Higher score = better match.
    """
    score = 0.0
    chunk_text = chunk.get("text", "").lower()

    # Keywords matching
    keywords = question.get("keywords", [])
    for kw in keywords:
        if kw.lower() in chunk_text:
            score += 2.0

    # Article reference matching (strong signal)
    article_ref = question.get("article_reference", "")
    if article_ref:
        # Extract article numbers like "1.3", "4.7.3"
        article_nums = re.findall(r'\d+(?:\.\d+)*', article_ref)
        for num in article_nums:
            if num in chunk_text:
                score += 5.0

        # Check for "article" keyword near number
        if "article" in article_ref.lower():
            for num in article_nums:
                pattern = rf'article\s+{re.escape(num)}'
                if re.search(pattern, chunk_text, re.IGNORECASE):
                    score += 10.0

    # Answer text matching (if extractive)
    answer_text = question.get("answer_text", "")
    if answer_text and len(answer_text) > 20:
        # Check for partial answer presence
        answer_words = set(re.findall(r'\b\w{5,}\b', answer_text.lower()))
        chunk_words = set(re.findall(r'\b\w{5,}\b', chunk_text))
        overlap = len(answer_words.intersection(chunk_words))
        score += overlap * 1.5

    # Section matching
    section = chunk.get("section", "")
    if section and article_ref:
        section_lower = section.lower()
        for num in re.findall(r'\d+(?:\.\d+)*', article_ref):
            if num in section_lower:
                score += 3.0

    return score


def map_question_to_chunk(
    question: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Map a single question to its best matching chunk.

    Returns mapping result with chunk_id, method, and confidence.
    """
    qid = question.get("id", "unknown")
    expected_docs = question.get("expected_docs", [])
    expected_pages = question.get("expected_pages", [])

    result = {
        "question_id": qid,
        "expected_chunk_id": None,
        "mapping_method": None,
        "mapping_confidence": 0.0,
        "candidates_count": 0,
        "error": None,
    }

    # Case 1: No expected pages (adversarial question)
    if not expected_pages:
        result["mapping_method"] = "adversarial_no_pages"
        result["error"] = "No expected_pages - likely adversarial"
        return result

    # Get source pattern
    source_pattern = expected_docs[0] if expected_docs else ""

    # Find candidate chunks
    candidates = find_chunks_for_pages(chunks, source_pattern, expected_pages)
    result["candidates_count"] = len(candidates)

    if not candidates:
        result["mapping_method"] = "no_candidates"
        result["error"] = f"No chunks found for pages {expected_pages} in {source_pattern}"
        return result

    # Score all candidates
    scored = [(chunk, score_chunk_for_question(chunk, question)) for chunk in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_chunk, best_score = scored[0]

    # Determine mapping method and confidence
    if best_score >= 15.0:
        method = "exact_article_match"
        confidence = 1.0
    elif best_score >= 10.0:
        method = "strong_keyword_match"
        confidence = 0.9
    elif best_score >= 5.0:
        method = "keyword_match"
        confidence = 0.7
    elif len(candidates) == 1:
        method = "single_candidate"
        confidence = 0.8
    else:
        method = "best_available"
        confidence = 0.5

    result["expected_chunk_id"] = best_chunk["id"]
    result["mapping_method"] = method
    result["mapping_confidence"] = confidence

    return result


def process_gold_standard(
    gs_data: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Process entire gold standard, mapping questions to chunks.

    Returns:
        - Updated gold standard with expected_chunk_id
        - Report with statistics
    """
    questions = gs_data.get("questions", [])

    stats = {
        "total": len(questions),
        "mapped": 0,
        "adversarial": 0,
        "failed": 0,
        "by_method": {},
        "by_confidence": {"high": 0, "medium": 0, "low": 0},
    }

    failed_questions = []

    for question in questions:
        mapping = map_question_to_chunk(question, chunks)

        if mapping["expected_chunk_id"]:
            question["expected_chunk_id"] = mapping["expected_chunk_id"]
            question["chunk_mapping"] = {
                "method": mapping["mapping_method"],
                "confidence": mapping["mapping_confidence"],
                "candidates": mapping["candidates_count"],
            }
            stats["mapped"] += 1

            # Track by method
            method = mapping["mapping_method"]
            stats["by_method"][method] = stats["by_method"].get(method, 0) + 1

            # Track by confidence
            conf = mapping["mapping_confidence"]
            if conf >= 0.9:
                stats["by_confidence"]["high"] += 1
            elif conf >= 0.7:
                stats["by_confidence"]["medium"] += 1
            else:
                stats["by_confidence"]["low"] += 1
        else:
            if mapping["mapping_method"] == "adversarial_no_pages":
                stats["adversarial"] += 1
            else:
                stats["failed"] += 1
                failed_questions.append({
                    "id": question.get("id"),
                    "error": mapping["error"],
                })

    # Calculate coverage
    mappable = stats["total"] - stats["adversarial"]
    coverage = stats["mapped"] / mappable if mappable > 0 else 0

    report = {
        "statistics": stats,
        "coverage": round(coverage, 4),
        "quality_gate": coverage >= 0.80,
        "failed_questions": failed_questions[:20],  # First 20 only
    }

    return gs_data, report


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Map Gold Standard questions to corpus chunks"
    )
    parser.add_argument(
        "--gold-standard", "-g",
        type=Path,
        default=DEFAULT_GS,
        help="Path to gold standard JSON",
    )
    parser.add_argument(
        "--chunks", "-c",
        type=Path,
        default=DEFAULT_CHUNKS,
        help="Path to chunks JSON",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for enriched gold standard",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("ETAPE 1: Mapping Pages → Chunks")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading gold standard: {args.gold_standard}")
    gs_data = load_gold_standard(args.gold_standard)

    logger.info(f"Loading chunks: {args.chunks}")
    chunks = load_chunks(args.chunks)
    logger.info(f"  Loaded {len(chunks)} chunks")

    # Process
    logger.info("Processing questions...")
    enriched_gs, report = process_gold_standard(gs_data, chunks)

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(enriched_gs, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved enriched GS to: {args.output}")

    # Save report
    report_path = args.output.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved report to: {report_path}")

    # Print summary
    stats = report["statistics"]
    logger.info("")
    logger.info("SUMMARY:")
    logger.info(f"  Total questions: {stats['total']}")
    logger.info(f"  Mapped: {stats['mapped']}")
    logger.info(f"  Adversarial (no pages): {stats['adversarial']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Coverage: {report['coverage']:.1%}")
    logger.info(f"  Quality Gate (>=80%): {'PASS' if report['quality_gate'] else 'FAIL'}")
    logger.info("")
    logger.info("By method:")
    for method, count in stats["by_method"].items():
        logger.info(f"  {method}: {count}")
    logger.info("")
    logger.info("By confidence:")
    logger.info(f"  High (>=0.9): {stats['by_confidence']['high']}")
    logger.info(f"  Medium (>=0.7): {stats['by_confidence']['medium']}")
    logger.info(f"  Low (<0.7): {stats['by_confidence']['low']}")

    if report["failed_questions"]:
        logger.warning("")
        logger.warning(f"Failed questions ({len(report['failed_questions'])} shown):")
        for fq in report["failed_questions"][:5]:
            logger.warning(f"  - {fq['id']}: {fq['error']}")


if __name__ == "__main__":
    main()
