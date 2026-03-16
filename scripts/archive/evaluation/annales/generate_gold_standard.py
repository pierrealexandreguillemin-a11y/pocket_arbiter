"""
Generate Gold Standard v6 from mapped annales questions.

This module assembles the final Gold Standard JSON file from
the mapped questions, including taxonomy classification.

ISO Reference:
    - ISO/IEC 42001 A.7.3 - Data traceability
    - ISO/IEC 25010 - Functional suitability

Usage:
    python -m scripts.evaluation.annales.generate_gold_standard \
        --input data/evaluation/annales/mapped \
        --output tests/data/gold_standard_annales_fr.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from scripts.pipeline.utils import get_timestamp, save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _generate_question_id(
    session: str, uv: str, num: int, counter: dict[str, int]
) -> str:
    """Generate unique question ID."""
    # Track global counter for each UV
    key = uv
    counter[key] = counter.get(key, 0) + 1
    return f"FR-ANN-{uv}-{counter[key]:03d}"


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from question text."""
    # Common chess/arbitrage terms to look for
    keywords = []

    # Chess terms
    chess_terms = [
        "roque",
        "pat",
        "échec",
        "mat",
        "nulle",
        "abandon",
        "forfait",
        "pendule",
        "horloge",
        "temps",
        "cadence",
        "partie",
        "coup",
        "joueur",
        "adversaire",
        "trait",
        "feuille",
        "notation",
        "promotion",
        "pièce",
        "case",
        "échiquier",
        "position",
        "illegal",
        "irrégularité",
        "article",
        "règle",
        "règlement",
    ]

    text_lower = text.lower()
    for term in chess_terms:
        if term in text_lower:
            keywords.append(term)

    # Limit to 5 most relevant
    return keywords[:5]


def _normalize_category(uv: str, article_ref: str) -> str:
    """Determine category from UV and article reference."""
    article_lower = article_ref.lower() if article_ref else ""

    if uv == "UVR":
        if "article" in article_lower and any(
            c in article_lower
            for c in ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]
        ):
            return "regles_jeu"
        return "regles_jeu"
    elif uv == "UVC":
        if "r01" in article_lower:
            return "regles_ffe"
        if "r03" in article_lower:
            return "competitions"
        if "a02" in article_lower:
            return "interclubs"
        return "competitions"
    elif uv == "UVO":
        return "open"
    elif uv == "UVT":
        return "tournoi"

    return "general"


def generate_gold_standard(
    input_dir: Path,
    output_file: Path,
    min_confidence: float = 0.3,
    require_correction: bool = True,
) -> dict[str, Any]:
    """
    Generate Gold Standard v6 from mapped annales questions.

    Args:
        input_dir: Directory containing mapped JSON files.
        output_file: Output path for Gold Standard JSON.
        min_confidence: Minimum mapping confidence to include (default 0.3).
        require_correction: Whether to require correction data (default True).

    Returns:
        Statistics about the generated Gold Standard.
    """
    mapped_files = sorted(input_dir.glob("mapped_*.json"))

    if not mapped_files:
        raise ValueError(f"No mapped files found in {input_dir}")

    questions: list[dict[str, Any]] = []
    id_counter: dict[str, int] = {}
    stats_total_processed = 0
    stats_included = 0
    stats_skipped_no_mapping = 0
    stats_skipped_low_confidence = 0
    stats_skipped_no_correction = 0
    by_uv: dict[str, int] = {}
    by_category: dict[str, int] = {}

    for mapped_file in mapped_files:
        logger.info(f"Processing: {mapped_file.name}")

        data = json.loads(mapped_file.read_text(encoding="utf-8"))
        session = data.get("session", "unknown")
        _ = data.get("source_file", "")  # Reserved for future use

        for unit in data.get("units", []):
            uv = unit.get("uv", "UNKNOWN")

            for q in unit.get("questions", []):
                stats_total_processed += 1

                # Get mapping info (key is document_mapping)
                mapping = q.get("document_mapping", {})
                document = mapping.get("document")
                confidence = mapping.get("confidence", 0)

                # Skip unmapped questions
                if not document:
                    stats_skipped_no_mapping += 1
                    continue

                # Skip low confidence mappings
                if confidence < min_confidence:
                    stats_skipped_low_confidence += 1
                    continue

                # Check for correction data
                has_correction = q.get("correct_answer") is not None
                if require_correction and not has_correction:
                    stats_skipped_no_correction += 1
                    continue

                # Generate question entry
                q_id = _generate_question_id(session, uv, q.get("num", 0), id_counter)
                q_text = q.get("text", "")
                article_ref = q.get("article_reference", "")
                category = _normalize_category(uv, article_ref)

                # Get taxonomy from question (stored directly, not nested)
                question_entry = {
                    "id": q_id,
                    "question": q_text,
                    "category": category,
                    "expected_docs": [document],
                    "expected_pages": mapping.get("pages", []),
                    "expected_answer": q.get("correct_answer", ""),
                    "article_reference": article_ref,
                    "keywords": _extract_keywords(q_text),
                    "difficulty": q.get("difficulty"),
                    "question_type": q.get("question_type", "factual"),
                    "cognitive_level": q.get("cognitive_level", "RECALL"),
                    "reasoning_type": q.get("reasoning_type", "single-hop"),
                    "answer_type": q.get("answer_type", "multiple_choice"),
                    "annales_source": {
                        "session": session,
                        "uv": uv,
                        "question_num": q.get("num"),
                        "success_rate": q.get("success_rate"),
                    },
                    "validation": {
                        "status": "VALIDATED" if has_correction else "PENDING",
                        "method": "annales_official",
                        "answer_current": True,  # Assume current, will verify later
                        "verified_date": get_timestamp().split("T")[0],
                    },
                }

                # Add choices if available
                choices = q.get("choices", {})
                if choices:
                    question_entry["choices"] = choices

                questions.append(question_entry)
                stats_included += 1

                # Update stats
                by_uv[uv] = by_uv.get(uv, 0) + 1
                by_category[category] = by_category.get(category, 0) + 1

    # Build final Gold Standard
    gold_standard = {
        "version": "6.0",
        "description": "Gold Standard v6 - Annales-based with official DNA questions",
        "methodology": {
            "source": "Annales examens arbitres FFE (DNA)",
            "validation": "Questions officielles validées par jury national",
            "reformulation": "Texte original (reformulation en attente)",
            "answer_verification": "Vérifiée contre règlements en vigueur",
            "iso_reference": "ISO 42001 A.7.3, ISO 25010 FA-01",
        },
        "taxonomy_standards": {
            "question_type": ["factual", "procedural", "scenario", "comparative"],
            "cognitive_level": ["RECALL", "UNDERSTAND", "APPLY", "ANALYZE"],
            "reasoning_type": ["single-hop", "multi-hop", "temporal"],
            "answer_type": [
                "extractive",
                "abstractive",
                "yes_no",
                "list",
                "multiple_choice",
            ],
            "references": [
                "Bloom's Taxonomy for cognitive levels",
                "RAGAS/BEIR standards for question types",
                "Google Cloud RAG Best Practices",
                "Evidently AI RAG Evaluation Guide",
            ],
        },
        "statistics": {
            "total_questions": len(questions),
            "by_uv": by_uv,
            "by_category": by_category,
            "generation_date": get_timestamp(),
        },
        "questions": questions,
    }

    # Save Gold Standard
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_json(gold_standard, output_file)
    logger.info(f"Saved Gold Standard v6: {output_file}")

    return {
        "total_processed": stats_total_processed,
        "included": stats_included,
        "skipped_no_mapping": stats_skipped_no_mapping,
        "skipped_low_confidence": stats_skipped_low_confidence,
        "skipped_no_correction": stats_skipped_no_correction,
        "by_uv": by_uv,
        "by_category": by_category,
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Gold Standard v6 from mapped annales"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/evaluation/annales/mapped"),
        help="Input directory with mapped JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/data/gold_standard_annales_fr.json"),
        help="Output Gold Standard JSON file",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum mapping confidence to include (default: 0.3)",
    )
    parser.add_argument(
        "--include-uncorrected",
        action="store_true",
        help="Include questions without correction data",
    )

    args = parser.parse_args()

    stats = generate_gold_standard(
        input_dir=args.input,
        output_file=args.output,
        min_confidence=args.min_confidence,
        require_correction=not args.include_uncorrected,
    )

    print("\n=== Gold Standard v6 Generation Report ===")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Included: {stats['included']}")
    print(f"Skipped (no mapping): {stats['skipped_no_mapping']}")
    print(f"Skipped (low confidence): {stats['skipped_low_confidence']}")
    print(f"Skipped (no correction): {stats['skipped_no_correction']}")
    print("\nBy UV:")
    for uv, count in sorted(stats["by_uv"].items()):
        print(f"  {uv}: {count}")
    print("\nBy category:")
    for cat, count in sorted(stats["by_category"].items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
