"""
Reformulate exam-style questions to natural user queries.

This module transforms formal exam questions into natural language queries
that better represent how users would actually ask questions to a RAG system.

Industry Standards:
    - BEIR benchmark: Query diversity improves retrieval robustness
    - RAGAS: Natural queries test semantic understanding
    - Google RAG Best Practices: Real user queries differ from formal questions

ISO Reference:
    - ISO/IEC 42001 A.7.3 - Data traceability
    - ISO/IEC 25010 - Functional suitability (realistic test data)

Usage:
    python -m scripts.evaluation.annales.reformulate_questions \
        --input tests/data/gold_standard_annales_fr.json \
        --output tests/data/gold_standard_annales_fr_reformulated.json
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from scripts.pipeline.utils import get_timestamp, save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Patterns to identify question types and transform them
EXAM_PATTERNS = [
    # Scenario patterns (situation-based)
    (r"^(Dans|Lors|Pendant|Au cours)", "scenario"),
    (r"^(Un joueur|Une joueuse|Le joueur|La joueuse)", "scenario"),
    (r"^(Un arbitre|Une arbitre|Un organisateur)", "scenario"),  # "Un arbitre constate..."
    (r"^(Vous êtes|Vous arbitrez|Vous observez)", "scenario"),  # Various "Vous" patterns
    (r"^Cadence\s*[:\d]", "scenario"),  # "Cadence : 1h30..." or "Cadence 50..."
    (r"^##?\s*Cadence", "scenario"),  # Markdown "## Cadence"
    (r"^En (Nationale|N\d|Open|Interclub|tournoi|match|coupe|cadence)", "scenario"),
    (r"^(Deux|Trois|Quatre|Cinq) joueurs?", "scenario"),
    (r"^(Le|La|Les) (blanc|noir|partie|match|tournoi)", "scenario"),
    (r"^(Après|Avant|Suite à)", "scenario"),
    (r"^(À|A) la (table|ronde|fin)", "scenario"),
    (r"^Partie\s", "scenario"),
    (r"^Une (fois|rencontre|partie|équipe)", "scenario"),  # "Une fois le..." "Une rencontre..."
    # Direct questions
    (r"^(Que|Qu'|Quoi|Comment|Pourquoi|Quand|Où|Combien|Quel|Quelle)", "direct"),
    # Selection/choice patterns
    (r"^Parmi (les|ces)", "selection"),  # "Parmi les propositions..."
    (r"^Pour (la|le|les|obtenir|cette)", "instruction"),  # "Pour la ronde..."
    # Conditional patterns
    (r"^(Si |Lorsque |Quand )", "conditional"),
    # Statement patterns (affirmations to verify)
    (r"^(Il est|C'est|La règle|L'article|Le règlement)", "statement"),
    # Article/rule reference patterns
    (r"^(Article|Art\.|Règle|Selon)", "reference"),
    # Instruction patterns
    (r"^(Saisissez|Indiquez|Cochez|Précisez)", "instruction"),
]

# Transformation templates for different question types
SCENARIO_TRANSFORMS = [
    # Remove verbose scenario intro, extract core question
    (r"^(Dans|Lors|Pendant|Au cours d')[^,\.]+[,\.]\s*", ""),
    # Simplify subject
    (r"^Un (jeune )?joueur", "Un joueur"),
    (r"^Une (jeune )?joueuse", "Un joueur"),
    # Remove "vous" perspective
    (r"Vous (devez|pouvez|êtes)", "L'arbitre doit"),
    (r"vous comptez les coups", "on compte les coups"),
]

# Common exam phrases to simplify
SIMPLIFICATION_PATTERNS = [
    (r"\s+se sont écoulés depuis", " après"),
    (r"avec (un )?incrément de \d+ secondes?", "avec incrément"),
    (r"en cadence (lente|rapide|Fischer)", r"en cadence \1"),
    (r"sur (son |l')échiquier", ""),
    (r"au bord de l'échiquier", ""),
    (r"après avoir (joué|appuyé|effectué)", "après"),
]


def detect_question_type(text: str) -> str:
    """Detect the type of exam question based on patterns."""
    for pattern, q_type in EXAM_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return q_type
    return "unknown"


def extract_core_question(text: str) -> str:
    """
    Extract the core question from verbose exam text.

    Args:
        text: Original exam question text.

    Returns:
        Simplified question focusing on the core query.
    """
    # Remove multiple choice options if present in text
    text = re.sub(r"\s*[A-D]\s*[-–:]\s*[^\n]+", "", text)

    # Find interrogative sentence if present
    sentences = re.split(r"[.!?]", text)
    for sentence in sentences:
        sentence = sentence.strip()
        if re.search(r"^(Que|Qu'|Quoi|Comment|Pourquoi|Quel|Quelle)", sentence, re.I):
            return sentence + "?"

    # If no direct question found, return cleaned original
    return text.strip()


def simplify_scenario(text: str) -> str:
    """
    Simplify scenario-based questions.

    Args:
        text: Original scenario question.

    Returns:
        Simplified query.
    """
    result = text

    # Apply simplification patterns
    for pattern, replacement in SIMPLIFICATION_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Apply scenario transforms
    for pattern, replacement in SCENARIO_TRANSFORMS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result.strip()


def reformulate_question(question: dict[str, Any]) -> dict[str, Any]:
    """
    Reformulate a single question to natural query format.

    Args:
        question: Gold Standard question dict.

    Returns:
        Updated question with reformulated query and original preserved.
    """
    original_text = question.get("question", "")
    article_ref = question.get("article_reference", "")

    # Preserve original
    question["question_original"] = original_text

    # Detect question type
    q_type = detect_question_type(original_text)

    # Apply reformulation based on type
    if q_type == "scenario":
        # For scenarios, try to extract the core question
        core = extract_core_question(original_text)
        simplified = simplify_scenario(core)

        # If we have article reference, can create focused query
        if article_ref:
            # Create article-based query variant
            question["query_variants"] = [
                simplified,
                f"Que dit l'article {article_ref} ?",
                f"Règle {article_ref} : quelle est la procédure ?",
            ]
        else:
            question["query_variants"] = [simplified]

        question["question"] = simplified

    elif q_type == "direct":
        # Direct questions are already natural, just clean up
        cleaned = simplify_scenario(original_text)
        question["question"] = cleaned
        question["query_variants"] = [cleaned]

    elif q_type == "conditional":
        # Convert conditional to direct question
        simplified = simplify_scenario(original_text)
        # Try to extract the "what happens" part
        match = re.search(r"(que|quoi|comment)\s+.*\?", simplified, re.I)
        if match:
            question["question"] = match.group(0).capitalize()
        else:
            question["question"] = simplified
        question["query_variants"] = [question["question"]]

    elif q_type == "statement":
        # Convert statement to question form
        # "La règle X dit..." -> "Que dit la règle X ?"
        if re.match(r"^(La règle|L'article|Le règlement)", original_text, re.I):
            question["question"] = f"Que dit {original_text[:50]}... ?"
        else:
            question["question"] = simplify_scenario(original_text)
        question["query_variants"] = [question["question"]]

    elif q_type == "reference":
        # Reference-based questions - keep the article reference clear
        simplified = simplify_scenario(original_text)
        question["question"] = simplified
        if article_ref:
            question["query_variants"] = [
                simplified,
                f"Que dit l'article {article_ref} ?",
            ]
        else:
            question["query_variants"] = [simplified]

    elif q_type == "selection":
        # "Parmi les propositions..." - keep as is, convert to question
        simplified = simplify_scenario(original_text)
        question["question"] = simplified
        question["query_variants"] = [simplified]

    elif q_type == "instruction":
        # "Saisissez...", "Pour obtenir..." - convert to question form
        simplified = simplify_scenario(original_text)
        if article_ref:
            question["question"] = f"Comment appliquer l'article {article_ref} ?"
            question["query_variants"] = [
                question["question"],
                simplified,
            ]
        else:
            question["question"] = simplified
            question["query_variants"] = [simplified]

    else:
        # Unknown type - just clean up
        question["question"] = simplify_scenario(original_text)
        question["query_variants"] = [question["question"]]

    # Add metadata about reformulation
    question["reformulation"] = {
        "original_type": q_type,
        "method": "rule_based_v1",
        "timestamp": get_timestamp().split("T")[0],
    }

    return question


def reformulate_gold_standard(
    input_path: Path,
    output_path: Path,
    preserve_original: bool = True,
) -> dict[str, Any]:
    """
    Reformulate all questions in Gold Standard.

    Args:
        input_path: Path to Gold Standard JSON.
        output_path: Output path for reformulated version.
        preserve_original: Whether to keep original question text.

    Returns:
        Statistics about reformulation.
    """
    gs_data = json.loads(input_path.read_text(encoding="utf-8"))
    questions = gs_data.get("questions", [])

    logger.info(f"Reformulating {len(questions)} questions")

    stats_total = 0
    stats_reformulated = 0
    by_type: dict[str, int] = {}

    total_original_len = 0
    total_reformulated_len = 0

    for q in questions:
        stats_total += 1
        original_len = len(q.get("question", ""))
        total_original_len += original_len

        reformulated = reformulate_question(q)

        reformulated_len = len(reformulated.get("question", ""))
        total_reformulated_len += reformulated_len

        q_type = reformulated.get("reformulation", {}).get("original_type", "unknown")
        by_type[q_type] = by_type.get(q_type, 0) + 1

        if reformulated_len < original_len:
            stats_reformulated += 1

    # Calculate average length reduction
    avg_length_reduction = 0.0
    if total_original_len > 0:
        avg_length_reduction = (
            (total_original_len - total_reformulated_len) / total_original_len
        ) * 100

    # Update methodology
    gs_data["methodology"]["reformulation"] = (
        f"Reformulated {stats_reformulated}/{stats_total} questions "
        f"({avg_length_reduction:.1f}% avg length reduction)"
    )

    gs_data["reformulation_stats"] = {
        "total_questions": stats_total,
        "questions_shortened": stats_reformulated,
        "avg_length_reduction_pct": round(avg_length_reduction, 1),
        "by_type": by_type,
        "reformulation_date": get_timestamp(),
    }

    # Save reformulated Gold Standard
    save_json(gs_data, output_path)
    logger.info(f"Saved reformulated Gold Standard: {output_path}")

    return {
        "total": stats_total,
        "reformulated": stats_reformulated,
        "by_type": by_type,
        "avg_length_reduction": avg_length_reduction,
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reformulate exam questions to natural queries"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tests/data/gold_standard_annales_fr.json"),
        help="Input Gold Standard JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/data/gold_standard_annales_fr.json"),
        help="Output reformulated JSON (can be same as input)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without saving",
    )

    args = parser.parse_args()

    if args.dry_run:
        # Just analyze without saving
        gs_data = json.loads(args.input.read_text(encoding="utf-8"))
        questions = gs_data.get("questions", [])

        type_counts: dict[str, int] = {}
        for q in questions:
            q_type = detect_question_type(q.get("question", ""))
            type_counts[q_type] = type_counts.get(q_type, 0) + 1

        print("\n=== Question Type Analysis ===")
        for q_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {q_type}: {count}")
        return

    stats = reformulate_gold_standard(
        input_path=args.input,
        output_path=args.output,
    )

    print("\n=== Reformulation Report ===")
    print(f"Total questions: {stats['total']}")
    print(f"Questions shortened: {stats['reformulated']}")
    print(f"Avg length reduction: {stats['avg_length_reduction']:.1f}%")
    print("\nBy original type:")
    for q_type, count in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
        print(f"  {q_type}: {count}")


if __name__ == "__main__":
    main()
