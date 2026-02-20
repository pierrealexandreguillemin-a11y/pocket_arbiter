"""
GS regression verification tool - Snapshot & Compare.

Creates baseline snapshots of GS files and compares them after
corrections to ensure no data is lost (IDs, scores, fields).

ISO Reference: ISO/IEC 29119 - Regression testing
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.evaluation.annales.quality_gates import count_schema_fields  # noqa: E402
from scripts.pipeline.utils import get_date, load_json, save_json  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class SnapshotData:
    """Snapshot of a GS file state at a given point in time."""

    date: str
    source_file: str
    total_questions: int
    answerable_count: int
    unanswerable_count: int
    question_ids: list[str]
    field_counts: dict[str, float]
    cognitive_level_dist: dict[str, int]
    question_type_dist: dict[str, int]
    difficulty_buckets: dict[str, int]
    reasoning_class_dist: dict[str, int]
    answer_type_dist: dict[str, int]
    chunk_match_scores: list[int]


@dataclass
class ComparisonResult:
    """Result of comparing a GS file against a baseline snapshot."""

    passed: bool
    ids_lost: list[str]
    ids_added: list[str]
    ids_preserved: int
    chunk_scores_valid: bool
    field_counts_valid: bool
    messages: list[str] = field(default_factory=list)


def _get_questions(gs_data: dict) -> list[dict]:
    """Extract questions list from GS data."""
    return gs_data.get("questions", [])


def create_snapshot(gs_data: dict, source_file: str = "") -> SnapshotData:
    """Create a snapshot from GS data.

    Args:
        gs_data: Full GS JSON data with 'questions' key.
        source_file: Path of the source file (for metadata).

    Returns:
        SnapshotData with all distributions computed.
    """
    questions = _get_questions(gs_data)

    answerable = [
        q for q in questions if not q.get("content", {}).get("is_impossible", False)
    ]
    unanswerable = [
        q for q in questions if q.get("content", {}).get("is_impossible", False)
    ]

    question_ids = [q["id"] for q in questions]

    # Field counts: min, max, avg
    if questions:
        field_vals = [count_schema_fields(q) for q in questions]
        field_counts = {
            "min": float(min(field_vals)),
            "max": float(max(field_vals)),
            "avg": round(sum(field_vals) / len(field_vals), 2),
        }
    else:
        field_counts = {"min": 0.0, "max": 0.0, "avg": 0.0}

    # Distributions
    cognitive_level_dist: dict[str, int] = {}
    question_type_dist: dict[str, int] = {}
    difficulty_buckets: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    reasoning_class_dist: dict[str, int] = {}
    answer_type_dist: dict[str, int] = {}
    chunk_match_scores: list[int] = []

    for q in questions:
        cls = q.get("classification", {})

        level = cls.get("cognitive_level", "unknown")
        cognitive_level_dist[level] = cognitive_level_dist.get(level, 0) + 1

        qtype = cls.get("question_type", "unknown")
        question_type_dist[qtype] = question_type_dist.get(qtype, 0) + 1

        diff = cls.get("difficulty", 0.5)
        if diff < 0.4:
            difficulty_buckets["easy"] += 1
        elif diff >= 0.7:
            difficulty_buckets["hard"] += 1
        else:
            difficulty_buckets["medium"] += 1

        rclass = cls.get("reasoning_class", "unknown")
        reasoning_class_dist[rclass] = reasoning_class_dist.get(rclass, 0) + 1

        atype = cls.get("answer_type", "unknown")
        answer_type_dist[atype] = answer_type_dist.get(atype, 0) + 1

        score = q.get("processing", {}).get("chunk_match_score", 0)
        chunk_match_scores.append(int(score))

    logger.info(
        "Snapshot: %dQ (ans=%d, unans=%d), fields min=%.0f/max=%.0f",
        len(questions),
        len(answerable),
        len(unanswerable),
        field_counts["min"],
        field_counts["max"],
    )

    return SnapshotData(
        date=get_date(),
        source_file=source_file,
        total_questions=len(questions),
        answerable_count=len(answerable),
        unanswerable_count=len(unanswerable),
        question_ids=question_ids,
        field_counts=field_counts,
        cognitive_level_dist=cognitive_level_dist,
        question_type_dist=question_type_dist,
        difficulty_buckets=difficulty_buckets,
        reasoning_class_dist=reasoning_class_dist,
        answer_type_dist=answer_type_dist,
        chunk_match_scores=chunk_match_scores,
    )


def save_snapshot(snapshot: SnapshotData, output_path: Path) -> None:
    """Save snapshot to JSON file."""
    save_json(asdict(snapshot), output_path)
    logger.debug("Snapshot saved to %s", output_path)


def load_snapshot(snapshot_path: Path) -> SnapshotData:
    """Load snapshot from JSON file."""
    data = load_json(snapshot_path)
    logger.debug("Snapshot loaded from %s", snapshot_path)
    return SnapshotData(**data)


def compare_snapshot(baseline: SnapshotData, current_gs: dict) -> ComparisonResult:
    """Compare current GS data against a baseline snapshot.

    Rules:
        - baseline IDs must be a subset of current IDs (no loss)
        - chunk_match_score must remain 100 for all questions
        - field_counts.min must not decrease

    Args:
        baseline: Previously saved snapshot.
        current_gs: Current GS JSON data.

    Returns:
        ComparisonResult with pass/fail and details.
    """
    current_snapshot = create_snapshot(current_gs)

    baseline_ids = set(baseline.question_ids)
    current_ids = set(current_snapshot.question_ids)

    ids_lost = sorted(baseline_ids - current_ids)
    ids_added = sorted(current_ids - baseline_ids)
    ids_preserved = len(baseline_ids & current_ids)

    messages: list[str] = []

    # Rule 1: No IDs lost
    ids_ok = len(ids_lost) == 0
    if not ids_ok:
        messages.append(
            f"FAIL: {len(ids_lost)} IDs lost: {ids_lost[:5]}{'...' if len(ids_lost) > 5 else ''}"
        )
    else:
        messages.append(f"PASS: All {ids_preserved} baseline IDs preserved")

    if ids_added:
        messages.append(f"INFO: {len(ids_added)} new IDs added")

    # Rule 2: chunk_match_score must be 100
    chunk_scores_valid = all(s == 100 for s in current_snapshot.chunk_match_scores)
    if not chunk_scores_valid:
        bad = [s for s in current_snapshot.chunk_match_scores if s != 100]
        messages.append(f"FAIL: {len(bad)} questions with chunk_match_score != 100")
    else:
        messages.append("PASS: All chunk_match_scores = 100")

    # Rule 3: field_counts.min must not decrease
    field_counts_valid = (
        current_snapshot.field_counts["min"] >= baseline.field_counts["min"]
    )
    if not field_counts_valid:
        messages.append(
            f"FAIL: field_counts.min decreased from "
            f"{baseline.field_counts['min']} to {current_snapshot.field_counts['min']}"
        )
    else:
        messages.append(
            f"PASS: field_counts.min maintained ({current_snapshot.field_counts['min']} "
            f">= {baseline.field_counts['min']})"
        )

    passed = ids_ok and chunk_scores_valid and field_counts_valid

    logger.info(
        "Comparison: %s (%d preserved, %d lost, %d added)",
        "PASS" if passed else "FAIL",
        ids_preserved,
        len(ids_lost),
        len(ids_added),
    )

    return ComparisonResult(
        passed=passed,
        ids_lost=ids_lost,
        ids_added=ids_added,
        ids_preserved=ids_preserved,
        chunk_scores_valid=chunk_scores_valid,
        field_counts_valid=field_counts_valid,
        messages=messages,
    )


def format_comparison_report(result: ComparisonResult) -> str:
    """Format comparison result as human-readable report.

    Args:
        result: ComparisonResult to format.

    Returns:
        Multi-line report string.
    """
    status = "PASS" if result.passed else "FAIL"
    lines = [
        f"=== Regression Check: {status} ===",
        f"IDs preserved: {result.ids_preserved}",
        f"IDs lost: {len(result.ids_lost)}",
        f"IDs added: {len(result.ids_added)}",
        f"Chunk scores valid: {result.chunk_scores_valid}",
        f"Field counts valid: {result.field_counts_valid}",
        "---",
    ]
    lines.extend(result.messages)
    return "\n".join(lines)


def main() -> int:
    """CLI entry point for snapshot and compare operations."""
    parser = argparse.ArgumentParser(
        description="GS regression verification (snapshot & compare)",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Create a baseline snapshot",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare GS against baseline snapshot",
    )
    parser.add_argument(
        "--gs",
        type=Path,
        required=True,
        help="Path to GS JSON file",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Path to baseline snapshot (required for --compare)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for snapshot or report",
    )
    args = parser.parse_args()

    if not args.snapshot and not args.compare:
        parser.error("Must specify --snapshot or --compare")

    gs_data = load_json(args.gs)

    if args.snapshot:
        snapshot = create_snapshot(gs_data, source_file=str(args.gs))
        output = args.output or Path(f"gs_snapshot_{get_date()}.json")
        save_snapshot(snapshot, output)
        print(f"Snapshot saved: {output}")
        print(f"  Total questions: {snapshot.total_questions}")
        print(f"  Answerable: {snapshot.answerable_count}")
        print(f"  Unanswerable: {snapshot.unanswerable_count}")
        return 0

    if args.compare:
        if not args.baseline:
            parser.error("--baseline required for --compare")
        baseline = load_snapshot(args.baseline)
        result = compare_snapshot(baseline, gs_data)
        report = format_comparison_report(result)
        print(report)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8")
        return 0 if result.passed else 1

    return 0  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
