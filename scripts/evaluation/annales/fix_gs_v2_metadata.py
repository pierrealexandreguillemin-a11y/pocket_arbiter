"""
Safe GS v2 metadata corrections (Phase A: A1/A2/A3).

A1: Schema normalization - add missing priority_boost to unanswerable questions
A2: Safe cognitive reclassification - regex-based Understand/Remember -> Apply/Analyze
A3: Audit trail - tag corrections in audit.history

ISO Reference: ISO/IEC 42001 - AI quality management
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.pipeline.utils import get_date, load_json, save_json  # noqa: E402


@dataclass
class CorrectionRecord:
    """Single correction applied to a question."""

    question_id: str
    correction_type: str  # "A1_schema" | "A2_cognitive" | "A3_audit"
    field: str
    old_value: Any
    new_value: Any
    pattern_matched: str = ""  # regex pattern (A2 only)


# ---------------------------------------------------------------------------
# A2: Compiled regex patterns for cognitive reclassification
# ---------------------------------------------------------------------------

_APPLY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"que doit[ -]faire", re.IGNORECASE), "que doit[ -]faire"),
    (re.compile(r"que doit-on faire", re.IGNORECASE), "que doit-on faire"),
    (re.compile(r"comment doit", re.IGNORECASE), "comment doit"),
    (re.compile(r"que faire si", re.IGNORECASE), "que faire si"),
]

_ANALYZE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"quelle diff[eé]rence", re.IGNORECASE), "quelle diff[eé]rence"),
    (re.compile(r"comparez", re.IGNORECASE), "comparez"),
    (
        re.compile(r"pourquoi .+ plut[oô]t que", re.IGNORECASE),
        "pourquoi .+ plut[oô]t que",
    ),
]


# ---------------------------------------------------------------------------
# A1: Schema normalization
# ---------------------------------------------------------------------------


def normalize_schema(question: dict, date: str) -> list[CorrectionRecord]:
    """Add missing priority_boost to unanswerable questions.

    Only modifies questions where processing.priority_boost is absent.

    Args:
        question: Single GS question dict (mutated in place).
        date: Date string for audit trail.

    Returns:
        List of CorrectionRecords (0 or 1 record).
    """
    processing = question.get("processing", {})
    if "priority_boost" not in processing:
        processing["priority_boost"] = 0.0
        question["processing"] = processing
        return [
            CorrectionRecord(
                question_id=question["id"],
                correction_type="A1_schema",
                field="processing.priority_boost",
                old_value=None,
                new_value=0.0,
            ),
        ]
    return []


# ---------------------------------------------------------------------------
# A2: Safe cognitive reclassification
# ---------------------------------------------------------------------------


def safe_cognitive_reclassify(question: dict, date: str) -> list[CorrectionRecord]:
    """Reclassify cognitive_level using safe regex patterns.

    Only applies to answerable questions with cognitive_level in
    (Understand, Remember). Patterns determine target level (Apply/Analyze).

    Args:
        question: Single GS question dict (mutated in place).
        date: Date string for audit trail.

    Returns:
        List of CorrectionRecords (0 or 1 record).
    """
    # Only answerable questions
    if question.get("content", {}).get("is_impossible", False):
        return []

    classification = question.get("classification", {})
    current_level = classification.get("cognitive_level", "")

    # Only reclassify Remember or Understand
    if current_level not in ("Remember", "Understand"):
        return []

    text = question.get("content", {}).get("question", "")

    # Check Apply patterns
    for pattern, pat_str in _APPLY_PATTERNS:
        if pattern.search(text):
            classification["cognitive_level"] = "Apply"
            return [
                CorrectionRecord(
                    question_id=question["id"],
                    correction_type="A2_cognitive",
                    field="classification.cognitive_level",
                    old_value=current_level,
                    new_value="Apply",
                    pattern_matched=pat_str,
                ),
            ]

    # Check Analyze patterns
    for pattern, pat_str in _ANALYZE_PATTERNS:
        if pattern.search(text):
            classification["cognitive_level"] = "Analyze"
            return [
                CorrectionRecord(
                    question_id=question["id"],
                    correction_type="A2_cognitive",
                    field="classification.cognitive_level",
                    old_value=current_level,
                    new_value="Analyze",
                    pattern_matched=pat_str,
                ),
            ]

    return []


# ---------------------------------------------------------------------------
# A3: Audit trail
# ---------------------------------------------------------------------------


def update_audit_trail(
    question: dict,
    corrections: list[CorrectionRecord],
    date: str,
) -> None:
    """Append correction tags to audit.history.

    Args:
        question: Single GS question dict (mutated in place).
        corrections: List of corrections applied to this question.
        date: Date string for tags.
    """
    if not corrections:
        return

    audit = question.get("audit", {})
    history = audit.get("history", "")

    for corr in corrections:
        if corr.correction_type == "A1_schema":
            tag = f"[PHASE A] schema normalized on {date}"
        elif corr.correction_type == "A2_cognitive":
            tag = (
                f"[PHASE A] reclassified cognitive_level: "
                f"{corr.old_value} -> {corr.new_value} on {date}"
            )
        else:
            continue  # pragma: no cover

        if history:
            history = f"{history} | {tag}"
        else:
            history = tag

    audit["history"] = history
    question["audit"] = audit


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def apply_all_corrections(
    gs_data: dict,
    dry_run: bool = False,
) -> tuple[dict, dict]:
    """Apply all Phase A corrections to a GS dataset.

    Args:
        gs_data: Full GS JSON data with 'questions' key.
        dry_run: If True, work on a deep copy and return original unchanged.

    Returns:
        Tuple of (corrected_gs_data, report_dict).
    """
    date = get_date()

    if dry_run:
        working = copy.deepcopy(gs_data)
    else:
        working = gs_data

    all_corrections: list[CorrectionRecord] = []
    a1_count = 0
    a2_count = 0
    a2_by_pattern: dict[str, int] = {}
    cognitive_before: dict[str, int] = {}
    cognitive_after: dict[str, int] = {}

    questions = working.get("questions", [])

    # Count cognitive levels before
    for q in questions:
        level = q.get("classification", {}).get("cognitive_level", "unknown")
        cognitive_before[level] = cognitive_before.get(level, 0) + 1

    for q in questions:
        corrections: list[CorrectionRecord] = []

        # A1: Schema normalization
        a1_corrs = normalize_schema(q, date)
        corrections.extend(a1_corrs)
        a1_count += len(a1_corrs)

        # A2: Cognitive reclassification
        a2_corrs = safe_cognitive_reclassify(q, date)
        corrections.extend(a2_corrs)
        a2_count += len(a2_corrs)
        for corr in a2_corrs:
            a2_by_pattern[corr.pattern_matched] = (
                a2_by_pattern.get(corr.pattern_matched, 0) + 1
            )

        # A3: Audit trail
        update_audit_trail(q, corrections, date)
        all_corrections.extend(corrections)

    # Count cognitive levels after
    for q in questions:
        level = q.get("classification", {}).get("cognitive_level", "unknown")
        cognitive_after[level] = cognitive_after.get(level, 0) + 1

    report = {
        "date": date,
        "total_corrections": len(all_corrections),
        "a1_schema_normalized": a1_count,
        "a2_cognitive_reclassified": a2_count,
        "a2_by_pattern": a2_by_pattern,
        "cognitive_before": cognitive_before,
        "cognitive_after": cognitive_after,
        "dry_run": dry_run,
    }

    if dry_run:
        return gs_data, report
    return working, report


def format_correction_report(report: dict) -> str:
    """Format correction report as human-readable text.

    Args:
        report: Report dict from apply_all_corrections().

    Returns:
        Multi-line report string.
    """
    lines = [
        f"=== Phase A Corrections {'(DRY RUN)' if report['dry_run'] else ''} ===",
        f"Date: {report['date']}",
        f"Total corrections: {report['total_corrections']}",
        f"  A1 schema normalized: {report['a1_schema_normalized']}",
        f"  A2 cognitive reclassified: {report['a2_cognitive_reclassified']}",
    ]
    if report["a2_by_pattern"]:
        lines.append("  A2 by pattern:")
        for pat, count in sorted(report["a2_by_pattern"].items()):
            lines.append(f"    {pat!r}: {count}")
    lines.append(f"Cognitive before: {report['cognitive_before']}")
    lines.append(f"Cognitive after:  {report['cognitive_after']}")
    return "\n".join(lines)


def main() -> int:
    """CLI entry point for safe metadata corrections."""
    parser = argparse.ArgumentParser(
        description="Safe GS v2 metadata corrections (Phase A)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input GS JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output GS JSON file (default: overwrite input)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing",
    )
    args = parser.parse_args()

    gs_data = load_json(args.input)
    result_data, report = apply_all_corrections(gs_data, dry_run=args.dry_run)

    print(format_correction_report(report))

    if not args.dry_run:
        output = args.output or args.input
        save_json(result_data, output)
        print(f"\nSaved: {output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
