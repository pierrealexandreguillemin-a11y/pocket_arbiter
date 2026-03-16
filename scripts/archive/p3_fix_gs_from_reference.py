#!/usr/bin/env python3
"""
Fix Gold Standard MCQ coherence from Docling reference data.

Compares GS v7 questions with the Docling MCQ reference extracted by
p3_extract_docling_mcq.py and fixes:
- mcq_answer mismatches (wrong correct letter)
- expected_answer != choices[mcq_answer] (incoherent answer text)
- mcq_answer pointing to absent choice letter

ISO Reference:
    - ISO/IEC 42001 A.7.3 - Data traceability
    - ISO/IEC 25010 - Functional suitability

Usage:
    python -m scripts.p3_fix_gs_from_reference [--dry-run]
"""

import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_reference(ref_path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Load Docling MCQ reference and index by (session, uv, question_num)."""
    with open(ref_path, encoding="utf-8") as f:
        entries = json.load(f)

    lookup: dict[tuple[str, str, int], dict[str, Any]] = {}
    for entry in entries:
        key = (entry["session"], entry["uv"], entry["question_num"])
        # Keep entry with the best data (prefer one with correct_letter)
        if key not in lookup or (
            entry.get("correct_letter") and not lookup[key].get("correct_letter")
        ):
            lookup[key] = entry

    return lookup


def _process_question(
    q: dict[str, Any],
    ref_lookup: dict[tuple[str, str, int], dict[str, Any]],
    report: dict[str, Any],
    dry_run: bool,
) -> None:
    """Process a single GS question against the reference, updating report."""
    meta = q.get("metadata", {})
    source = meta.get("annales_source")

    if not source:
        report["human_questions"] += 1
        return

    report["annales_questions"] += 1

    session = source.get("session", "")
    uv = source.get("uv", "")
    q_num = source.get("question_num", 0)
    q_id = q.get("id", "")

    ref = ref_lookup.get((session, uv, q_num))
    if not ref:
        report["no_ref_match"] += 1
        return

    report["matched_with_ref"] += 1

    ref_letter = ref.get("correct_letter", "")
    gs_letter = meta.get("mcq_answer", "")
    gs_choices = meta.get("choices", {})
    gs_expected = q.get("expected_answer", "")

    if not ref_letter or len(ref_letter) != 1:
        report["remaining_issues"]["no_correct_letter_in_ref"] += 1
        return

    ref_letter = ref_letter.upper()

    if ref_letter not in gs_choices:
        report["remaining_issues"]["letter_not_in_choices"] += 1
        report["changes"].append(
            {
                "id": q_id,
                "session": session,
                "uv": uv,
                "q_num": q_num,
                "issue": f"ref_letter {ref_letter} not in choices {list(gs_choices.keys())}",
                "action": "SKIPPED",
            }
        )
        return

    correct_text = gs_choices[ref_letter]

    if gs_letter == ref_letter and gs_expected == correct_text:
        report["already_correct"] += 1
        return

    change: dict[str, Any] = {
        "id": q_id,
        "session": session,
        "uv": uv,
        "q_num": q_num,
    }

    if gs_letter != ref_letter:
        change["mcq_answer"] = {"old": gs_letter, "new": ref_letter}
        report["fixes_applied"]["mcq_answer_changed"] += 1

    if gs_expected != correct_text:
        change["expected_answer"] = {
            "old": gs_expected[:60] + ("..." if len(gs_expected) > 60 else ""),
            "new": correct_text[:60] + ("..." if len(correct_text) > 60 else ""),
        }
        report["fixes_applied"]["expected_answer_changed"] += 1

    report["changes"].append(change)

    if not dry_run:
        meta["mcq_answer"] = ref_letter
        q["expected_answer"] = correct_text


def fix_gs_from_reference(
    gs_path: Path,
    ref_path: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Compare GS with Docling reference and fix MCQ coherence issues.

    Returns a report dict with statistics and change details.
    """
    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)

    ref_lookup = load_reference(ref_path)

    report: dict[str, Any] = {
        "total_questions": len(gs["questions"]),
        "annales_questions": 0,
        "human_questions": 0,
        "matched_with_ref": 0,
        "no_ref_match": 0,
        "already_correct": 0,
        "fixes_applied": {
            "mcq_answer_changed": 0,
            "expected_answer_changed": 0,
            "choices_fixed": 0,
        },
        "remaining_issues": {
            "no_correct_letter_in_ref": 0,
            "letter_not_in_choices": 0,
        },
        "changes": [],
    }

    for q in gs["questions"]:
        _process_question(q, ref_lookup, report, dry_run)

    # Final validation pass
    coherence_ok = 0
    coherence_fail = 0
    for q in gs["questions"]:
        meta = q.get("metadata", {})
        choices = meta.get("choices", {})
        mcq = meta.get("mcq_answer", "")
        exp = q.get("expected_answer", "")
        if not choices or not mcq:
            continue
        if mcq in choices and choices[mcq] == exp:
            coherence_ok += 1
        else:
            coherence_fail += 1

    report["final_coherence"] = {
        "ok": coherence_ok,
        "fail": coherence_fail,
        "total_mcq": coherence_ok + coherence_fail,
    }

    # Save if not dry run
    if not dry_run:
        with open(gs_path, "w", encoding="utf-8") as f:
            json.dump(gs, f, ensure_ascii=False, indent=2)
            f.write("\n")
        logger.info(f"Saved corrected GS to {gs_path}")

    return report


def main() -> None:
    """Run GS fix from Docling reference."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix Gold Standard MCQ coherence from Docling reference"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing",
    )
    parser.add_argument(
        "--gs",
        type=Path,
        default=Path("tests/data/gold_standard_annales_fr_v7.json"),
        help="Gold Standard JSON file",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=Path("data/evaluation/annales/docling_mcq_reference.json"),
        help="Docling MCQ reference JSON",
    )
    args = parser.parse_args()

    if not args.ref.exists():
        logger.error(f"Reference file not found: {args.ref}")
        logger.info("Run p3_extract_docling_mcq.py first to generate reference data")
        return

    report = fix_gs_from_reference(args.gs, args.ref, dry_run=args.dry_run)

    print("\n=== P3 MCQ Fix Report ===")
    print(f"Total questions: {report['total_questions']}")
    print(f"Annales questions: {report['annales_questions']}")
    print(f"Human questions: {report['human_questions']}")
    print(f"Matched with reference: {report['matched_with_ref']}")
    print(f"No reference match: {report['no_ref_match']}")
    print(f"Already correct: {report['already_correct']}")

    fixes = report["fixes_applied"]
    print("\nFixes applied:")
    print(f"  mcq_answer changed: {fixes['mcq_answer_changed']}")
    print(f"  expected_answer changed: {fixes['expected_answer_changed']}")

    remaining = report["remaining_issues"]
    print("\nRemaining issues:")
    print(f"  No correct_letter in ref: {remaining['no_correct_letter_in_ref']}")
    print(f"  Letter not in choices: {remaining['letter_not_in_choices']}")

    coherence = report["final_coherence"]
    print("\nFinal coherence check:")
    print(f"  OK: {coherence['ok']}/{coherence['total_mcq']}")
    print(f"  FAIL: {coherence['fail']}/{coherence['total_mcq']}")

    if args.dry_run:
        print("\n[DRY RUN - no changes written]")

    # Show first 10 changes
    changes = report["changes"]
    if changes:
        print(f"\nSample changes ({len(changes)} total):")
        for ch in changes[:15]:
            parts = [f"  {ch['session']}/{ch['uv']}/Q{ch['q_num']}"]
            if "mcq_answer" in ch:
                parts.append(
                    f"letter: {ch['mcq_answer']['old']}->{ch['mcq_answer']['new']}"
                )
            if "expected_answer" in ch:
                parts.append("answer_text: updated")
            if "issue" in ch:
                parts.append(f"ISSUE: {ch['issue']}")
            print(" | ".join(parts))


if __name__ == "__main__":
    main()
