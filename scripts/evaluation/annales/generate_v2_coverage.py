"""Phase B orchestrator: batch generation over uncovered chunks.

Generates answerable questions from uncovered chunks in batches,
with stop gates, distribution steering, and quality validation.

ISO Reference:
- ISO 42001 A.6.2.2: Provenance tracking
- ISO 29119-3: Test data coverage
- ISO 25010: Data quality

Usage:
    python -m scripts.evaluation.annales.generate_v2_coverage \
        --gs tests/data/gs_scratch_v1_step1.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --output data/gs_generation/phase_b_answerable.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.evaluation.annales.generate_real_questions import (  # noqa: E402
    generate_questions_from_chunk,
)
from scripts.evaluation.annales.stratify_corpus import (  # noqa: E402
    extract_covered_chunk_ids,
    filter_uncovered_chunks,
    load_chunks,
)
from scripts.pipeline.utils import get_date, load_json, save_json  # noqa: E402

BATCH_SIZE = 30
STOP_KEYWORD_THRESHOLD = 0.3
STOP_KEYWORD_RATIO = 0.20


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BatchReport:
    """Report for a single batch of generation."""

    batch_num: int
    chunks_processed: int
    questions_generated: int
    empty_chunks: int
    low_keyword_count: int = 0


@dataclass
class CumulativeStats:
    """Cumulative statistics across all batches."""

    total_chunks: int = 0
    total_questions: int = 0
    total_empty: int = 0
    reasoning_class: Counter = field(default_factory=Counter)
    cognitive_level: Counter = field(default_factory=Counter)
    question_type: Counter = field(default_factory=Counter)
    difficulty_buckets: Counter = field(default_factory=Counter)
    answer_type: Counter = field(default_factory=Counter)
    covered_chunks: set = field(default_factory=set)

    def fact_single_ratio(self) -> float:
        """Get current fact_single ratio."""
        total = sum(self.reasoning_class.values())
        if total == 0:
            return 0.0
        return self.reasoning_class.get("fact_single", 0) / total

    def hard_ratio(self) -> float:
        """Get current hard difficulty ratio."""
        total = sum(self.difficulty_buckets.values())
        if total == 0:
            return 0.0
        return self.difficulty_buckets.get("hard", 0) / total


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def validate_question(q: dict, chunk: dict) -> bool:
    """Validate a generated question against per-question gates.

    Gates (Plan §5.5):
    - G1-1: chunk_match_score = 100 (BY DESIGN invariant — set, not checked)
    - G1-2: answer extractable from chunk (keyword overlap >= 0.3)
    - G1-4: question ends with "?"

    Args:
        q: Generated question dict (flat format).
        chunk: Source chunk dict.

    Returns:
        True if the question passes all gates.
    """
    # G1-4: question must end with "?"
    question_text = q.get("question", "")
    if not question_text.rstrip().endswith("?"):
        return False

    # G1-2: answer must be extractable from chunk
    answer = q.get("expected_answer", "")
    chunk_text = chunk.get("text", "")
    if not answer or len(answer) < 6:
        return False
    if chunk_text:
        answer_words = {w.lower() for w in answer.split() if len(w) >= 4}
        chunk_words = {w.lower() for w in chunk_text.split()}
        if answer_words:
            overlap = len(answer_words & chunk_words) / len(answer_words)
            if overlap < STOP_KEYWORD_THRESHOLD:
                return False

    return True


def load_uncovered_chunks(
    chunks_path: Path,
    gs_path: Path,
    min_tokens: int = 50,
) -> list[dict]:
    """Load chunks not covered by existing GS, filtered by token count.

    Args:
        chunks_path: Path to chunks JSON.
        gs_path: Path to GS JSON.
        min_tokens: Minimum token count (default 50).

    Returns:
        List of uncovered, generatable chunks.
    """
    all_chunks = load_chunks(chunks_path)
    uncovered = filter_uncovered_chunks(all_chunks, gs_path)
    return [c for c in uncovered if c.get("tokens", 0) >= min_tokens]


def prioritize_chunks(chunks: list[dict]) -> list[dict]:
    """Sort chunks by generation priority.

    Priority order:
    1. Small docs with 0% coverage (Coupes, Championnats, Other)
    2. Large docs (LA-octobre2025)

    Within each group, sort by page number for locality.

    Args:
        chunks: List of chunk dicts.

    Returns:
        Sorted list.
    """

    def priority_key(c: dict) -> tuple[int, str, int]:
        source = c.get("source", "")
        page = c.get("page", 0)
        if "LA-octobre" in source or source.startswith("LA-"):
            return (2, source, page)  # Large doc = lower priority
        return (1, source, page)  # Small docs first

    return sorted(chunks, key=priority_key)


def generate_batch(
    chunks_batch: list[dict],
    batch_num: int,
    target_per_chunk: int = 2,
) -> tuple[list[dict], BatchReport]:
    """Generate questions for a batch of chunks.

    Args:
        chunks_batch: List of chunks to process.
        batch_num: Batch number for tracking.
        target_per_chunk: Target questions per chunk.

    Returns:
        Tuple of (questions, batch_report).
    """
    questions: list[dict] = []
    empty_chunks = 0
    low_keyword = 0

    for chunk in chunks_batch:
        qs = generate_questions_from_chunk(chunk, target_count=target_per_chunk)
        if not qs:
            empty_chunks += 1
            continue

        chunk_text = chunk.get("text", "")
        for q in qs:
            # Validate per-question gates (Plan §5.5: G1-1, G1-2, G1-4)
            if not validate_question(q, chunk):
                continue

            # Check keyword coverage (answer words in chunk)
            answer = q.get("expected_answer", "")
            if answer and chunk_text:
                answer_words = {w.lower() for w in answer.split() if len(w) >= 4}
                chunk_words = {w.lower() for w in chunk_text.split()}
                if answer_words:
                    kw_cov = len(answer_words & chunk_words) / len(answer_words)
                    if kw_cov < STOP_KEYWORD_THRESHOLD:
                        low_keyword += 1

            # Enrich with chunk metadata
            q["chunk_id"] = chunk["id"]
            q["source"] = chunk.get("source", "")
            q["pages"] = chunk.get("pages", [chunk.get("page", 0)])
            questions.append(q)

    report = BatchReport(
        batch_num=batch_num,
        chunks_processed=len(chunks_batch),
        questions_generated=len(questions),
        empty_chunks=empty_chunks,
        low_keyword_count=low_keyword,
    )

    return questions, report


def check_stop_gates(
    batch_report: BatchReport,
    cumulative: CumulativeStats,
) -> tuple[bool, str]:
    """Check if generation should stop.

    Args:
        batch_report: Current batch report.
        cumulative: Cumulative stats so far.

    Returns:
        Tuple of (should_stop, reason).
    """
    # Stop gate 1: keyword coverage degradation
    if batch_report.questions_generated > 0:
        low_ratio = batch_report.low_keyword_count / batch_report.questions_generated
        if low_ratio > STOP_KEYWORD_RATIO:
            return True, (
                f"STOP: keyword coverage < {STOP_KEYWORD_THRESHOLD} on "
                f"{low_ratio:.0%} of batch (threshold: {STOP_KEYWORD_RATIO:.0%})"
            )

    # Stop gate 2: empty chunk ratio too high
    if batch_report.chunks_processed > 0:
        empty_ratio = batch_report.empty_chunks / batch_report.chunks_processed
        if empty_ratio > 0.5:
            return True, (
                f"STOP: {empty_ratio:.0%} of chunks produced 0 questions "
                f"(>50% threshold)"
            )

    return False, ""


def update_cumulative(
    cumulative: CumulativeStats,
    questions: list[dict],
    batch_report: BatchReport,
) -> None:
    """Update cumulative statistics from a batch (mutates in place).

    Args:
        cumulative: Cumulative stats to update.
        questions: Questions from this batch.
        batch_report: Batch report.
    """
    cumulative.total_chunks += batch_report.chunks_processed
    cumulative.total_questions += batch_report.questions_generated
    cumulative.total_empty += batch_report.empty_chunks

    for q in questions:
        cumulative.reasoning_class[q.get("reasoning_class", "unknown")] += 1
        cumulative.cognitive_level[q.get("cognitive_level", "unknown")] += 1
        cumulative.question_type[q.get("question_type", "unknown")] += 1
        cumulative.answer_type[q.get("answer_type", "extractive")] += 1

        diff = q.get("difficulty", 0.5)
        if diff < 0.4:
            cumulative.difficulty_buckets["easy"] += 1
        elif diff >= 0.7:
            cumulative.difficulty_buckets["hard"] += 1
        else:
            cumulative.difficulty_buckets["medium"] += 1

        chunk_id = q.get("chunk_id", "")
        if chunk_id:
            cumulative.covered_chunks.add(chunk_id)


def format_progress(
    batch_num: int,
    total_batches: int,
    batch_report: BatchReport,
    cumulative: CumulativeStats,
) -> str:
    """Format a progress line.

    Args:
        batch_num: Current batch number.
        total_batches: Total number of batches.
        batch_report: Current batch report.
        cumulative: Cumulative stats.

    Returns:
        Progress string.
    """
    fs_ratio = cumulative.fact_single_ratio()
    hard_ratio = cumulative.hard_ratio()
    return (
        f"[Batch {batch_num}/{total_batches}] "
        f"+{batch_report.questions_generated}Q "
        f"({batch_report.empty_chunks} empty) | "
        f"Total: {cumulative.total_questions}Q, "
        f"{len(cumulative.covered_chunks)} chunks | "
        f"fact_single={fs_ratio:.1%} hard={hard_ratio:.1%}"
    )


# ---------------------------------------------------------------------------
# Phase B gates
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Result of a single gate check."""

    name: str
    passed: bool
    value: str
    threshold: str


def run_phase_b_gates(
    cumulative: CumulativeStats,
    total_corpus_chunks: int,
    existing_covered: int,
) -> list[GateResult]:
    """Run Phase B quality gates.

    Args:
        cumulative: Cumulative stats from generation.
        total_corpus_chunks: Total chunks in corpus.
        existing_covered: Chunks already covered before Phase B.

    Returns:
        List of GateResult.
    """
    results: list[GateResult] = []
    new_covered = len(cumulative.covered_chunks)
    total_covered = existing_covered + new_covered
    coverage = total_covered / total_corpus_chunks if total_corpus_chunks else 0

    results.append(
        GateResult(
            name="B-G1: coverage >= 80%",
            passed=coverage >= 0.80,
            value=f"{coverage:.1%} ({total_covered}/{total_corpus_chunks})",
            threshold=">= 80%",
        )
    )

    # Cognitive levels - check Apply and Analyze
    cl = cumulative.cognitive_level
    total_q = sum(cl.values())
    apply_pct = cl.get("Apply", 0) / total_q if total_q else 0
    analyze_pct = cl.get("Analyze", 0) / total_q if total_q else 0

    results.append(
        GateResult(
            name="B-G2: Apply >= 10%",
            passed=apply_pct >= 0.10,
            value=f"{apply_pct:.1%}",
            threshold=">= 10%",
        )
    )
    results.append(
        GateResult(
            name="B-G3: Analyze >= 10%",
            passed=analyze_pct >= 0.10,
            value=f"{analyze_pct:.1%}",
            threshold=">= 10%",
        )
    )

    results.append(
        GateResult(
            name="B-G4: hard >= 10%",
            passed=cumulative.hard_ratio() >= 0.10,
            value=f"{cumulative.hard_ratio():.1%}",
            threshold=">= 10%",
        )
    )

    qt = cumulative.question_type
    comp_pct = qt.get("comparative", 0) / total_q if total_q else 0
    results.append(
        GateResult(
            name="B-G5: comparative >= 5%",
            passed=comp_pct >= 0.05,
            value=f"{comp_pct:.1%}",
            threshold=">= 5%",
        )
    )

    return results


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_coverage_generation(
    chunks_path: Path,
    gs_path: Path,
    output_path: Path,
    batch_size: int = BATCH_SIZE,
    max_batches: int | None = None,
    batches_dir: Path | None = None,
) -> dict:
    """Run Phase B coverage generation.

    Args:
        chunks_path: Path to chunks JSON.
        gs_path: Path to current GS JSON.
        output_path: Path for output questions JSON.
        batch_size: Chunks per batch.
        max_batches: Optional limit on batch count.
        batches_dir: Optional directory for per-batch outputs.

    Returns:
        Generation report dict.
    """
    print("=" * 60)
    print("PHASE B: COVERAGE GENERATION")
    print("=" * 60)

    # Load all data once (DRY: single load per resource)
    print("\nLoading data...")
    all_chunks = load_chunks(chunks_path)
    total_corpus = len(all_chunks)
    gs_data = load_json(gs_path)
    existing_covered = extract_covered_chunk_ids(gs_data)

    # Filter to uncovered + generatable
    uncovered = [c for c in all_chunks if c["id"] not in existing_covered]
    chunks = [c for c in uncovered if c.get("tokens", 0) >= 50]
    print(f"  Generatable chunks: {len(chunks)}")
    print(f"  Existing covered: {len(existing_covered)}/{total_corpus}")

    # Prioritize
    chunks = prioritize_chunks(chunks)

    # Batch planning
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    print(f"  Planned batches: {total_batches} ({batch_size} chunks/batch)")

    if batches_dir:
        batches_dir.mkdir(parents=True, exist_ok=True)

    # Generate
    all_questions: list[dict] = []
    cumulative = CumulativeStats()
    batch_reports: list[dict] = []
    stopped_reason = ""

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, len(chunks))
        batch_chunks = chunks[start:end]

        if not batch_chunks:
            break

        questions, report = generate_batch(batch_chunks, batch_num)
        update_cumulative(cumulative, questions, report)
        all_questions.extend(questions)

        # TODO(Plan §5.4): Distribution steering — if fact_single > 55%,
        # adjust next batch prompt to force reasoning/Apply. Requires
        # LLM-in-the-loop prompt modification (Phase B execution).
        # TODO(Plan §5.4): Token budget tracking — stop if cumulative
        # tokens > 10M. Requires token counting per LLM call.

        progress = format_progress(batch_num + 1, total_batches, report, cumulative)
        print(progress)

        batch_reports.append(
            {
                "batch_num": report.batch_num,
                "chunks_processed": report.chunks_processed,
                "questions_generated": report.questions_generated,
                "empty_chunks": report.empty_chunks,
                "low_keyword_count": report.low_keyword_count,
            }
        )

        # Save per-batch output
        if batches_dir:
            batch_path = batches_dir / f"batch_{batch_num:03d}.json"
            save_json(questions, batch_path)

        # Check stop gates
        should_stop, reason = check_stop_gates(report, cumulative)
        if should_stop:
            print(f"\n{reason}")
            stopped_reason = reason
            break

    # Run Phase B gates
    print("\n" + "=" * 60)
    print("PHASE B GATES")
    print("=" * 60)

    gates = run_phase_b_gates(cumulative, total_corpus, len(existing_covered))
    for g in gates:
        status = "PASS" if g.passed else "FAIL"
        print(f"  [{status}] {g.name}: {g.value} (threshold: {g.threshold})")

    # Save output
    output = {
        "version": "1.0",
        "date": get_date(),
        "phase": "B",
        "total_questions": len(all_questions),
        "generation_stats": {
            "chunks_processed": cumulative.total_chunks,
            "chunks_covered_new": len(cumulative.covered_chunks),
            "chunks_empty": cumulative.total_empty,
            "questions_generated": cumulative.total_questions,
            "stopped": bool(stopped_reason),
            "stopped_reason": stopped_reason,
        },
        "distributions": {
            "reasoning_class": dict(cumulative.reasoning_class),
            "cognitive_level": dict(cumulative.cognitive_level),
            "question_type": dict(cumulative.question_type),
            "difficulty_buckets": dict(cumulative.difficulty_buckets),
            "answer_type": dict(cumulative.answer_type),
        },
        "gates": [
            {
                "name": g.name,
                "passed": g.passed,
                "value": g.value,
                "threshold": g.threshold,
            }
            for g in gates
        ],
        "batch_reports": batch_reports,
        "questions": all_questions,
    }

    save_json(output, output_path)
    print(f"\nSaved {len(all_questions)} questions to {output_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase B: batch generation over uncovered chunks",
    )
    parser.add_argument(
        "--gs",
        type=Path,
        default=Path("tests/data/gs_scratch_v1_step1.json"),
        help="Current GS JSON file",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("corpus/processed/chunks_mode_b_fr.json"),
        help="Chunks JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/gs_generation/phase_b_answerable.json"),
        help="Output questions JSON",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Chunks per batch (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit number of batches (for testing)",
    )
    parser.add_argument(
        "--batches-dir",
        type=Path,
        default=None,
        help="Directory for per-batch output files",
    )
    args = parser.parse_args()

    result = run_coverage_generation(
        chunks_path=args.chunks,
        gs_path=args.gs,
        output_path=args.output,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        batches_dir=args.batches_dir,
    )

    failed_gates = [g for g in result["gates"] if not g["passed"]]
    if failed_gates:
        print(f"\nWARNING: {len(failed_gates)} gates FAILED")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
