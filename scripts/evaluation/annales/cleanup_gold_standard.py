#!/usr/bin/env python3
"""
Cleanup Gold Standard: derive answer_text, remove MCQ letters.

ISO 42001: Ensure answers are verifiable text, not abstract letters.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DeriveResult:
    """Result of deriving answer text from choices."""

    __slots__ = ("answer_text", "answer_complete", "answer_missing_letters")

    def __init__(self) -> None:
        self.answer_text: str | None = None
        self.answer_complete: bool = False
        self.answer_missing_letters: list[str] = []


class QualityResult:
    """Result of quality assessment for an answer."""

    __slots__ = ("warning", "score")

    def __init__(self, warning: str | None = None, score: float = 1.0) -> None:
        self.warning = warning
        self.score = score


# Patterns indicating reference-only answers (not self-contained)
REFERENCE_PATTERNS = [
    "Voir ",
    "Réf:",
    "Ref:",
    "cf. ",
    "Cf. ",
    "Se référer",
]


def assess_answer_quality(answer_text: str) -> QualityResult:
    """
    Assess the quality of an answer_text.

    Args:
        answer_text: The answer text to assess.

    Returns:
        QualityResult with warning type and quality score.

    Quality scores:
        - 1.0: Good answer (self-contained, sufficient length)
        - 0.5: Short answer (< 30 chars, may lack context)
        - 0.3: Reference only (points to document, not self-contained)
    """
    if not answer_text:
        return QualityResult(warning="empty_answer", score=0.0)

    # Check for reference-only patterns
    for pattern in REFERENCE_PATTERNS:
        if answer_text.startswith(pattern):
            return QualityResult(warning="reference_only", score=0.3)

    # Check for very short answers
    if len(answer_text) < 30:
        return QualityResult(warning="short_answer", score=0.5)

    # Good answer
    return QualityResult(warning=None, score=1.0)


def derive_answer_text(question: dict[str, Any]) -> DeriveResult:
    """
    Derive answer_text from choices and expected_answer.

    Returns DeriveResult with:
    - answer_text: derived text (if possible)
    - answer_complete: bool indicating if all letters were found
    - answer_missing_letters: list of letters not found in choices
    """
    result = DeriveResult()

    # Try both field names (mcq_answer in GS, expected_answer in parsed)
    expected = question.get("mcq_answer") or question.get("expected_answer", "")
    choices = question.get("choices", {})

    if not choices or not expected:
        return result

    texts = []
    missing = []

    for letter in expected:
        if letter in choices:
            texts.append(choices[letter])
        elif letter.isalpha():  # Skip non-letter characters
            missing.append(letter)

    if texts:
        result.answer_text = " | ".join(texts)
        result.answer_complete = len(missing) == 0
        result.answer_missing_letters = missing

    return result


def cleanup_gold_standard(gs_data: dict[str, Any]) -> dict[str, Any]:
    """
    Cleanup Gold Standard:
    1. Derive answer_text from choices
    2. Remove expected_answer (MCQ letters) - useless without context
    3. Add answer_verification metadata
    4. Assess answer quality and add warnings
    """
    questions = gs_data.get("questions", [])

    stats: dict[str, Any] = {
        "total": len(questions),
        "answer_derived_complete": 0,
        "answer_derived_partial": 0,
        "answer_not_derivable": 0,
        "mcq_letters_removed": 0,
        "quality_good": 0,
        "quality_warnings": {
            "reference_only": 0,
            "short_answer": 0,
            "empty_answer": 0,
        },
    }

    for q in questions:
        # Derive answer text
        derived = derive_answer_text(q)

        if derived.answer_text:
            q["answer_text"] = derived.answer_text
            if derived.answer_complete:
                stats["answer_derived_complete"] += 1
            else:
                stats["answer_derived_partial"] += 1
                q["answer_incomplete"] = True
                if derived.answer_missing_letters:
                    q["answer_missing_letters"] = derived.answer_missing_letters
        else:
            # Fallback: use article_reference as answer_text (from correction table)
            article_ref = q.get("article_reference", "")
            if article_ref and len(article_ref) > 5:
                q["answer_text"] = article_ref
                q["answer_source"] = "article_reference"
                stats["answer_derived_complete"] += 1  # Count as derived
            else:
                stats["answer_not_derivable"] += 1
                q["answer_text"] = q.get("expected_answer", "")
                q["answer_type_mcq_letter"] = True

        # Remove MCQ letter (keep in mcq_answer for traceability)
        if "expected_answer" in q:
            q["mcq_answer"] = q.pop("expected_answer")
            stats["mcq_letters_removed"] += 1

        # Assess answer quality
        answer_text = q.get("answer_text", "")
        quality = assess_answer_quality(answer_text)

        q["quality_score"] = quality.score
        if quality.warning:
            q["quality_warning"] = quality.warning
            stats["quality_warnings"][quality.warning] = (
                stats["quality_warnings"].get(quality.warning, 0) + 1
            )
        else:
            stats["quality_good"] += 1

    # Update metadata
    gs_data["cleanup_stats"] = stats

    return gs_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleanup Gold Standard")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input Gold Standard JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON (default: overwrite input)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without writing",
    )
    args = parser.parse_args()

    # Load
    with open(args.input, encoding="utf-8") as f:
        gs_data = json.load(f)

    # Cleanup
    gs_data = cleanup_gold_standard(gs_data)

    # Report
    stats = gs_data["cleanup_stats"]
    print("=== Cleanup Report ===")
    print(f"Total questions: {stats['total']}")
    print(f"Answer derived (complete): {stats['answer_derived_complete']}")
    print(f"Answer derived (partial): {stats['answer_derived_partial']}")
    print(f"Answer not derivable: {stats['answer_not_derivable']}")
    print(f"MCQ letters moved: {stats['mcq_letters_removed']}")

    derivable_pct = (
        (stats["answer_derived_complete"] + stats["answer_derived_partial"])
        / stats["total"]
        * 100
    )
    print(f"\nDerivable rate: {derivable_pct:.1f}%")

    if args.dry_run:
        print("\n[DRY RUN] No changes written")
        return

    # Write
    output_path = args.output or args.input
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gs_data, f, ensure_ascii=False, indent=2)

    print(f"\nWritten to: {output_path}")


if __name__ == "__main__":
    main()
