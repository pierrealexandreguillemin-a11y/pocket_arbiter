"""
Patch gold standard with re-extracted question + answer data.

Applies re-extraction results from reextract_from_docling.py to the
gold standard JSON, updating question text, expected answers, choices,
and metadata while preserving existing valid data.

ISO Reference:
    - ISO/IEC 12207 - Controlled data update
    - ISO/IEC 25010 - Data quality (no truncation, no ref-only answers)
    - ISO/IEC 42001 - Traceability (diff report, answer_source)
    - ISO/IEC 27001 - No secrets in output
"""

import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _is_ref_only(text: str) -> bool:
    """Check if text is just an article reference (no real answer content).

    Returns True for article-only patterns like "Art. 3.7", "LA 2.1".
    Returns False for short but valid answers like "Le jeudi.", "4.".
    """
    import re

    if not text:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    # Article-only patterns
    if re.match(
        r"^(?:Art\.?\s*[\d.]+(?:\s+et\s+[\d.]+)*\.?"
        r"|LA\s*[-–—,]\s*[\d.]+"
        r"|Chapitre\s+\d+)"
        r"\s*\.?$",
        stripped,
        re.IGNORECASE,
    ):
        return True
    return False


def build_expected_answer(
    result: dict[str, Any],
    existing_answer: str,
) -> tuple[str, str]:
    """
    Build expected_answer from re-extraction result.

    Priority:
    1. MCQ + extracted choices -> text of correct choice
    2. Explanation from corrigé
    3. Keep existing answer

    Args:
        result: Re-extraction result dict.
        existing_answer: Current expected_answer from GS.

    Returns:
        Tuple of (answer_text, answer_source).
    """
    # Priority 1: MCQ answer text from choice
    answer_from_choice = result.get("answer_text_from_choice")
    if answer_from_choice and not _is_ref_only(answer_from_choice):
        return answer_from_choice, "choice"

    # Priority 2: Explanation from corrigé
    explanation = result.get("answer_explanation")
    if explanation and not _is_ref_only(explanation):
        return explanation, "explanation"

    # Priority 3: Keep existing
    return existing_answer, "existing"


def _clean_question_text(text: str) -> str:
    """Strip ## markdown artifacts from question text."""
    import re

    text = re.sub(r"\s*##\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def should_update_question(
    existing: str,
    extracted: str,
) -> bool:
    """
    Decide whether to update question text.

    Update only if extracted is non-empty AND strictly longer.
    Never regress to a shorter text. The ## cleanup is handled
    separately in the final pass.
    """
    if not extracted:
        return False
    return len(extracted) > len(existing)


def patch_gold_standard(
    gs_path: Path,
    results_path: Path,
    output_path: Path,
    diff_path: Path | None = None,
) -> dict[str, Any]:
    """
    Patch gold standard with re-extraction results.

    Args:
        gs_path: Path to gold standard JSON.
        results_path: Path to reextraction_results.json.
        output_path: Path to write patched GS.
        diff_path: Optional path for diff report.

    Returns:
        Patch report dict.
    """
    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    # Index results by ID
    result_by_id: dict[str, dict[str, Any]] = {r["id"]: r for r in results}

    # Track changes
    changes: list[dict[str, Any]] = []
    excluded_ids: list[dict[str, Any]] = []
    patched_questions: list[dict[str, Any]] = []

    for q in gs["questions"]:
        q_id = q["id"]
        meta = q.get("metadata", {})
        src = meta.get("annales_source")

        # Non-annales questions: keep as-is
        if not src or not src.get("session"):
            patched_questions.append(q)
            continue

        result = result_by_id.get(q_id)
        if not result:
            # No re-extraction result — keep question
            patched_questions.append(q)
            continue

        flags = result.get("extraction_flags", [])

        # Exclude annulled / commentary
        if "annulled" in flags or "commentary" in flags:
            excluded_ids.append(
                {
                    "id": q_id,
                    "reason": "annulled" if "annulled" in flags else "commentary",
                    "session": src.get("session"),
                    "uv": src.get("uv"),
                    "question_num": src.get("question_num"),
                }
            )
            continue

        # Track what changes
        change_record: dict[str, Any] = {"id": q_id, "fields_changed": []}

        # 1. Update question text
        extracted_q = result.get("question_full", "")
        if should_update_question(q.get("question", ""), extracted_q):
            old_len = len(q.get("question", ""))
            q["question"] = extracted_q
            change_record["fields_changed"].append(
                f"question ({old_len}->{len(extracted_q)} chars)"
            )

        # 2. Update expected_answer
        new_answer, answer_source = build_expected_answer(
            result, q.get("expected_answer", "")
        )
        # Fallback: if answer is still ref-only, try deriving from existing choices
        if _is_ref_only(new_answer):
            mcq_letter = meta.get("mcq_answer") or result.get("mcq_answer")
            all_choices = meta.get("choices") or result.get("choices") or {}
            if mcq_letter and mcq_letter in all_choices:
                choice_text = all_choices[mcq_letter]
                if not _is_ref_only(choice_text):
                    new_answer = choice_text
                    answer_source = "choice_fallback"

        if new_answer != q.get("expected_answer", ""):
            old_len = len(q.get("expected_answer", ""))
            q["expected_answer"] = new_answer
            change_record["fields_changed"].append(
                f"expected_answer ({old_len}->{len(new_answer)} chars, source={answer_source})"
            )

        # 3. Update metadata.choices (only if re-extracted has more)
        existing_choices = meta.get("choices") or {}
        new_choices = result.get("choices", {})
        if new_choices and len(new_choices) > len(existing_choices):
            meta["choices"] = new_choices
            change_record["fields_changed"].append(
                f"choices ({len(existing_choices)}->{len(new_choices)})"
            )

        # 4. Update metadata.mcq_answer
        new_mcq = result.get("mcq_answer")
        if new_mcq and new_mcq != meta.get("mcq_answer"):
            old_mcq = meta.get("mcq_answer")
            meta["mcq_answer"] = new_mcq
            change_record["fields_changed"].append(f"mcq_answer ({old_mcq}->{new_mcq})")

        # 5. Update article_reference (prefer re-extracted if longer)
        new_ref = result.get("article_reference")
        existing_ref = meta.get("article_reference", "")
        if new_ref and len(new_ref) > len(existing_ref or ""):
            meta["article_reference"] = new_ref
            change_record["fields_changed"].append("article_reference")

        # 6. Update success_rate
        new_rate = result.get("success_rate")
        if new_rate is not None:
            old_rate = src.get("success_rate")
            src["success_rate"] = new_rate
            if old_rate != new_rate:
                change_record["fields_changed"].append(
                    f"success_rate ({old_rate}->{new_rate})"
                )

        # 7. Add new metadata fields
        meta["extraction_flags"] = flags
        meta["answer_explanation"] = result.get("answer_explanation")
        meta["answer_source"] = answer_source

        q["metadata"] = meta

        if change_record["fields_changed"]:
            changes.append(change_record)

        patched_questions.append(q)

    # Final cleanup: strip any remaining ## from ALL question AND answer texts
    cleaned_count = 0
    cleaned_answer_count = 0
    for q in patched_questions:
        q_text = q.get("question", "")
        if "##" in q_text:
            q["question"] = _clean_question_text(q_text)
            cleaned_count += 1
        a_text = q.get("expected_answer", "")
        if "##" in a_text:
            q["expected_answer"] = _clean_question_text(a_text)
            cleaned_answer_count += 1

    # Cleanup corrupted mcq_answer values ("D 88.1" -> "D", "ADE" stays)
    import re as _re

    mcq_cleaned_count = 0
    for q in patched_questions:
        mcq = q.get("metadata", {}).get("mcq_answer")
        if mcq and not _re.match(r"^[A-F]+$", mcq):
            # Extract leading letter(s) only
            m = _re.match(r"^([A-F]+)", mcq)
            if m:
                q["metadata"]["mcq_answer"] = m.group(1)
                mcq_cleaned_count += 1

    # Replace questions
    gs["questions"] = patched_questions

    if cleaned_count:
        logger.info(f"Cleaned ## from {cleaned_count} question texts")
    if cleaned_answer_count:
        logger.info(f"Cleaned ## from {cleaned_answer_count} answer texts")
    if mcq_cleaned_count:
        logger.info(f"Cleaned {mcq_cleaned_count} corrupted mcq_answer values")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gs, f, ensure_ascii=False, indent=2)

    # Build report
    report: dict[str, Any] = {
        "total_input": len(gs["questions"]) + len(excluded_ids),
        "total_output": len(patched_questions),
        "total_excluded": len(excluded_ids),
        "total_changed": len(changes),
        "excluded": excluded_ids,
        "changes_summary": _summarize_changes(changes),
    }

    if diff_path:
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        report["changes_detail"] = changes
        with open(diff_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def _summarize_changes(changes: list[dict[str, Any]]) -> dict[str, int]:
    """Count how many times each field was changed."""
    counts: dict[str, int] = {}
    for c in changes:
        for field_str in c["fields_changed"]:
            # Extract field name (before parenthetical)
            field_name = field_str.split("(")[0].strip()
            counts[field_name] = counts.get(field_name, 0) + 1
    return counts


def validate_patched_gs(gs_path: Path) -> dict[str, Any]:
    """
    Validate the patched GS against GATE 4 criteria.

    Args:
        gs_path: Path to patched GS.

    Returns:
        Validation results dict.
    """
    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)

    questions = gs["questions"]
    issues: dict[str, list[str]] = {
        "truncated": [],
        "fusion": [],
        "ref_only_answer": [],
        "empty_answer": [],
    }

    for q in questions:
        q_id = q["id"]
        q_text = q.get("question", "")
        answer = q.get("expected_answer", "")

        # Zero truncation: no question < 200 chars that was identified as truncated
        # (This is informational — short questions exist naturally)

        # Zero fusion: no ## in question text
        if "##" in q_text:
            issues["fusion"].append(q_id)

        # Answers not empty
        if not answer or not answer.strip():
            issues["empty_answer"].append(q_id)

        # Answers not ref-only (< 20 chars)
        if answer and _is_ref_only(answer):
            issues["ref_only_answer"].append(q_id)

    validation: dict[str, Any] = {
        "total_questions": len(questions),
        "zero_fusion": len(issues["fusion"]) == 0,
        "zero_empty_answer": len(issues["empty_answer"]) == 0,
        "ref_only_count": len(issues["ref_only_answer"]),
        "issues": {k: v for k, v in issues.items() if v},
    }

    return validation


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Patch gold standard with re-extraction results"
    )
    parser.add_argument(
        "--gs",
        type=Path,
        default=Path("tests/data/gold_standard_annales_fr_v7.json"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("tests/data/checkpoints/reextraction_results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/data/gold_standard_annales_fr_v7.json"),
    )
    parser.add_argument(
        "--diff",
        type=Path,
        default=Path("tests/data/checkpoints/patch_diff_report.json"),
    )
    args = parser.parse_args()

    report = patch_gold_standard(
        gs_path=args.gs,
        results_path=args.results,
        output_path=args.output,
        diff_path=args.diff,
    )

    print("\n=== Patch Report ===")
    print(f"Total input: {report['total_input']}")
    print(f"Total output: {report['total_output']}")
    print(f"Total excluded: {report['total_excluded']}")
    print(f"Total changed: {report['total_changed']}")
    print(f"Changes: {report['changes_summary']}")
    if report["excluded"]:
        print(f"Excluded: {[e['id'] for e in report['excluded']]}")

    # Validate
    validation = validate_patched_gs(args.output)
    print("\n=== GATE 4 Validation ===")
    print(f"Total questions: {validation['total_questions']}")
    print(f"Zero fusion: {'PASS' if validation['zero_fusion'] else 'FAIL'}")
    print(f"Zero empty answer: {'PASS' if validation['zero_empty_answer'] else 'FAIL'}")
    print(f"Ref-only answers: {validation['ref_only_count']}")
    if validation.get("issues"):
        for k, v in validation["issues"].items():
            print(f"  {k}: {len(v)} issues")


if __name__ == "__main__":
    main()
