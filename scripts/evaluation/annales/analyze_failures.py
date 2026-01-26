"""
Analyse des Questions Echouees - Pocket Arbiter

Analyse les questions qui echouent la validation semantique
et categorise les causes pour actions correctives.

ISO Reference:
    - ISO/IEC 42001 A.6.2.2 - Tracabilite qualite
    - ISO/IEC 29119 - Analyse defauts

Categories de causes:
    - chunk_id_incorrect: Le chunk assigne ne contient pas la reponse
    - needs_reclassification: La reasoning_class devrait etre changee
    - requires_context: Question necessite contexte visuel/externe
    - answer_mismatch: La reponse n'est pas extractable du chunk

Usage:
    python -m scripts.evaluation.annales.analyze_failures \
        --validation tests/data/validation_by_class_report.json \
        --gs tests/data/gold_standard_annales_fr_v7.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --output tests/data/failure_analysis_report.json
"""

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

from scripts.pipeline.utils import load_json, save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Keywords indicating visual/external context needed
CONTEXT_KEYWORDS = [
    "ci-dessous",
    "ci-dessus",
    "ci-contre",
    "figure",
    "image",
    "tableau suivant",
    "schema",
    "diagramme",
    "dessin",
    "position suivante",
    "position ci-dessous",
]

# Common names in exam scenarios (require context)
EXAM_NAMES = [
    "daniela",
    "albert",
    "paul",
    "marie",
    "jean",
    "pierre",
    "sophie",
    "marc",
    "anne",
    "luc",
]


def detect_requires_context(question: dict) -> tuple[bool, list[str]]:
    """
    Detect if question requires visual/external context.

    Returns:
        (requires_context, list of indicators found)
    """
    q_text = question.get("question", "").lower()
    answer = question.get("expected_answer", "").lower()
    combined = f"{q_text} {answer}"

    indicators = []

    # Check for context keywords
    for kw in CONTEXT_KEYWORDS:
        if kw.lower() in combined:
            indicators.append(f"keyword:{kw}")

    # Check for exam-specific names
    for name in EXAM_NAMES:
        if name.lower() in combined:
            indicators.append(f"name:{name}")

    # Check for image markers
    if "<!-- image -->" in question.get("question", ""):
        indicators.append("marker:image")

    return len(indicators) > 0, indicators


def detect_needs_reclassification(
    question: dict,
    validation_result: dict,
) -> tuple[bool, str | None]:
    """
    Detect if question should be reclassified to different reasoning_class.

    Returns:
        (needs_reclassification, suggested_class)
    """
    current_class = question.get("metadata", {}).get("reasoning_class", "unknown")
    answer = question.get("expected_answer", "")
    q_text = question.get("question", "")

    # fact_single failing might be summary
    if current_class == "fact_single":
        # If answer is long or contains synthesis language
        if len(answer.split()) > 15:
            return True, "summary"
        # If question asks for explanation/description
        synthesis_words = ["expliquer", "decrire", "resume", "synthese", "ensemble"]
        if any(w in q_text.lower() for w in synthesis_words):
            return True, "summary"

    # Check if contains calculations
    if current_class in ["fact_single", "summary"]:
        # Look for arithmetic indicators
        calc_patterns = [
            r"\d+\s*[+\-*/]\s*\d+",  # Explicit calculation
            r"total|somme|difference|produit",  # Calc keywords
            r"\d+\s*minutes?\s*(?:et|plus|\+)",  # Time calculations
        ]
        for pattern in calc_patterns:
            if re.search(pattern, f"{q_text} {answer}", re.IGNORECASE):
                return True, "arithmetic"

    return False, None


def detect_chunk_id_issues(
    question: dict,
    chunk_text: str,
) -> tuple[bool, dict]:
    """
    Detect if chunk_id assignment is incorrect.

    Returns:
        (has_issue, details)
    """
    answer = question.get("expected_answer", "")
    article_ref = question.get("metadata", {}).get("article_reference", "")

    # Check if answer text is in chunk
    answer_in_chunk = answer.lower() in chunk_text.lower()

    # Check key answer words
    answer_words = [w for w in answer.lower().split() if len(w) > 3]
    words_in_chunk = sum(1 for w in answer_words if w in chunk_text.lower())
    word_coverage = words_in_chunk / len(answer_words) if answer_words else 1.0

    # Check if article reference matches chunk section
    article_match = False
    if article_ref:
        # Extract article numbers
        article_nums = re.findall(r"Art(?:icle)?\.?\s*(\d+(?:\.\d+)?)", article_ref)
        for num in article_nums:
            if f"Article {num}" in chunk_text or f"Art. {num}" in chunk_text:
                article_match = True
                break

    has_issue = not answer_in_chunk and word_coverage < 0.5 and not article_match

    return has_issue, {
        "answer_in_chunk": answer_in_chunk,
        "word_coverage": round(word_coverage, 2),
        "article_match": article_match,
    }


def analyze_failure(
    question: dict,
    validation_result: dict,
    chunk_text: str | None,
) -> dict:
    """
    Analyze a single failed question.

    Returns:
        Analysis with categorized cause and suggested action.
    """
    q_id = question.get("id", "unknown")
    reasoning_class = question.get("metadata", {}).get("reasoning_class", "unknown")

    analysis = {
        "question_id": q_id,
        "reasoning_class": reasoning_class,
        "validation_score": validation_result.get("validation", {}).get("score"),
        "threshold": validation_result.get("validation", {}).get("threshold"),
        "cause": "unknown",
        "suggested_action": None,
        "details": {},
    }

    # 1. Check requires_context
    needs_context, context_indicators = detect_requires_context(question)
    if needs_context:
        analysis["cause"] = "requires_context"
        analysis["suggested_action"] = "Mark metadata.requires_context=true"
        analysis["details"]["context_indicators"] = context_indicators
        return analysis

    # 2. Check reclassification
    needs_reclass, suggested_class = detect_needs_reclassification(
        question, validation_result
    )
    if needs_reclass:
        analysis["cause"] = "needs_reclassification"
        analysis["suggested_action"] = f"Change reasoning_class to '{suggested_class}'"
        analysis["details"]["current_class"] = reasoning_class
        analysis["details"]["suggested_class"] = suggested_class
        return analysis

    # 3. Check chunk_id issues
    if chunk_text:
        chunk_issue, chunk_details = detect_chunk_id_issues(question, chunk_text)
        if chunk_issue:
            analysis["cause"] = "chunk_id_incorrect"
            analysis["suggested_action"] = "Find correct chunk via article_reference"
            analysis["details"] = chunk_details
            analysis["details"]["article_reference"] = question.get("metadata", {}).get(
                "article_reference", ""
            )
            return analysis

    # 4. Default: answer not extractable
    analysis["cause"] = "answer_mismatch"
    analysis["suggested_action"] = "Review answer against chunk content"
    if chunk_text:
        analysis["details"]["chunk_preview"] = chunk_text[:200]
    analysis["details"]["answer"] = question.get("expected_answer", "")[:100]

    return analysis


def analyze_failures(
    validation_path: Path,
    gs_path: Path,
    chunks_path: Path,
) -> dict:
    """
    Analyze all failed questions from validation report.

    Args:
        validation_path: Path to validation_by_class_report.json.
        gs_path: Path to gold standard JSON.
        chunks_path: Path to chunks JSON.

    Returns:
        Analysis report with categorized failures.
    """
    logger.info(f"Loading validation report: {validation_path}")
    validation_data = load_json(validation_path)

    logger.info(f"Loading gold standard: {gs_path}")
    gs_data = load_json(gs_path)
    questions = {q["id"]: q for q in gs_data.get("questions", [])}

    logger.info(f"Loading chunks: {chunks_path}")
    chunks_data = load_json(chunks_path)
    chunks_map = {c["id"]: c["text"] for c in chunks_data.get("chunks", [])}

    # Find failed questions
    failed_results = [
        r
        for r in validation_data.get("results", [])
        if not r.get("validation", {}).get("passed", True)
    ]

    logger.info(f"Analyzing {len(failed_results)} failed questions...")

    # Analyze each failure
    analyses = []
    cause_counts: dict[str, int] = {}

    for result in failed_results:
        q_id = result.get("question_id")
        if q_id not in questions:
            continue

        question = questions[q_id]
        chunk_id = question.get("expected_chunk_id")
        chunk_text = chunks_map.get(chunk_id) if chunk_id else None

        analysis = analyze_failure(question, result, chunk_text)
        analyses.append(analysis)

        cause = analysis["cause"]
        cause_counts[cause] = cause_counts.get(cause, 0) + 1

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "validation_file": str(validation_path),
        "gold_standard_file": str(gs_path),
        "summary": {
            "total_failures_analyzed": len(analyses),
            "cause_distribution": cause_counts,
        },
        "by_cause": {
            cause: [a for a in analyses if a["cause"] == cause]
            for cause in cause_counts
        },
        "all_analyses": analyses,
    }

    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze failed questions from semantic validation"
    )

    parser.add_argument(
        "--validation",
        "-v",
        type=Path,
        required=True,
        help="Validation report JSON from validate_by_class.py",
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
        help="Output analysis report JSON (optional)",
    )

    args = parser.parse_args()

    report = analyze_failures(args.validation, args.gs, args.chunks)

    # Print summary
    summary = report["summary"]
    logger.info("=" * 60)
    logger.info("FAILURE ANALYSIS REPORT")
    logger.info("=" * 60)
    logger.info(f"Total failures analyzed: {summary['total_failures_analyzed']}")
    logger.info("-" * 60)
    logger.info("Cause distribution:")
    for cause, count in sorted(summary["cause_distribution"].items()):
        pct = count / summary["total_failures_analyzed"] * 100
        logger.info(f"  {cause}: {count} ({pct:.1f}%)")

    # Show samples per cause
    for cause, analyses in report["by_cause"].items():
        logger.info("-" * 60)
        logger.info(f"Samples for '{cause}' ({len(analyses)} total):")
        for a in analyses[:3]:
            logger.info(f"  - {a['question_id']}")
            logger.info(f"    Action: {a['suggested_action']}")

    # Save report if output specified
    if args.output:
        save_json(report, args.output)
        logger.info(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
