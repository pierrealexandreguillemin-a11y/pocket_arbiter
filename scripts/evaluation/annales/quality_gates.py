#!/usr/bin/env python3
"""
Quality Gates for Gold Standard BY DESIGN generation.

Defines all quality gates (G0-G5) with blocking and warning thresholds.
Functions return (passed: bool, errors: list[str]) tuples.
Exit code != 0 on gate failure.

ISO Reference:
- ISO 29119-3: Test data validation
- ISO 25010: Quality metrics
- ISO 42001 A.6.2.2: Provenance verification

Gates Reference:
- G0: Stratification gates
- G1: Generation gates (answerable)
- G2: Generation gates (unanswerable)
- G3: Anti-hallucination validation
- G4: Schema v2 enrichment
- G5: Distribution and deduplication
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@dataclass
class GateResult:
    """Result of a quality gate check."""

    gate_id: str
    name: str
    passed: bool
    blocking: bool
    value: Any
    threshold: Any
    message: str


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


# =============================================================================
# PHASE 0: STRATIFICATION GATES
# =============================================================================


def g0_1_strata_count(strata: dict, min_count: int = 5) -> GateResult:
    """
    G0-1: Minimum strata count.

    BLOCKING: At least 5 active strata required.
    """
    active = sum(1 for s in strata.values() if s.get("quota", 0) > 0)
    passed = active >= min_count
    return GateResult(
        gate_id="G0-1",
        name="strata_count",
        passed=passed,
        blocking=True,
        value=active,
        threshold=min_count,
        message=f"Active strata: {active} (min: {min_count})",
    )


def g0_2_document_coverage(coverage: dict, min_coverage: float = 0.80) -> GateResult:
    """
    G0-2: Minimum document coverage.

    WARNING: At least 80% of corpus documents should be represented.
    """
    ratio = coverage.get("coverage_ratio", 0)
    passed = ratio >= min_coverage
    return GateResult(
        gate_id="G0-2",
        name="document_coverage",
        passed=passed,
        blocking=False,  # Warning only
        value=ratio,
        threshold=min_coverage,
        message=f"Document coverage: {ratio:.1%} (min: {min_coverage:.0%})",
    )


# =============================================================================
# PHASE 1: ANSWERABLE GENERATION GATES
# =============================================================================


def g1_1_chunk_match_score(question: dict) -> GateResult:
    """
    G1-1: Chunk match score must be 100% for BY DESIGN generation.

    BLOCKING: score == 100 for each question.
    """
    score = question.get("processing", {}).get("chunk_match_score", 0)
    passed = score == 100
    return GateResult(
        gate_id="G1-1",
        name="chunk_match_score",
        passed=passed,
        blocking=True,
        value=score,
        threshold=100,
        message=f"Chunk match: {score}% (required: 100%)",
    )


def g1_2_answer_in_chunk(
    answer: str,
    chunk_text: str,
    semantic_sim: float | None = None,
    sim_threshold: float = 0.95,
) -> GateResult:
    """
    G1-2: Answer must be extractable from chunk.

    BLOCKING: verbatim match OR semantic_similarity >= 0.95
    """
    # Check verbatim match (case-insensitive)
    answer_norm = answer.lower().strip()
    chunk_norm = chunk_text.lower()

    if answer_norm in chunk_norm:
        return GateResult(
            gate_id="G1-2",
            name="answer_in_chunk",
            passed=True,
            blocking=True,
            value="verbatim",
            threshold="verbatim or sim>=0.95",
            message="Answer found verbatim in chunk",
        )

    # Check semantic similarity if provided
    if semantic_sim is not None and semantic_sim >= sim_threshold:
        return GateResult(
            gate_id="G1-2",
            name="answer_in_chunk",
            passed=True,
            blocking=True,
            value=f"semantic:{semantic_sim:.3f}",
            threshold=f"sim>={sim_threshold}",
            message=f"Answer semantically similar: {semantic_sim:.3f}",
        )

    return GateResult(
        gate_id="G1-2",
        name="answer_in_chunk",
        passed=False,
        blocking=True,
        value=semantic_sim or "no_match",
        threshold=f"verbatim or sim>={sim_threshold}",
        message=f"Answer not found in chunk (sim={semantic_sim})",
    )


def g1_3_fact_single_ratio(
    questions: list[dict], max_ratio: float = 0.60
) -> GateResult:
    """
    G1-3: fact_single should not dominate.

    WARNING: fact_single_ratio < 60% (project threshold to prevent dominance)
    """
    answerable = [
        q for q in questions if not q.get("content", {}).get("is_impossible", False)
    ]
    if not answerable:
        return GateResult(
            gate_id="G1-3",
            name="fact_single_ratio",
            passed=True,
            blocking=False,
            value=0,
            threshold=max_ratio,
            message="No answerable questions",
        )

    fact_single = sum(
        1
        for q in answerable
        if q.get("classification", {}).get("reasoning_class") == "fact_single"
    )
    ratio = fact_single / len(answerable)
    passed = ratio < max_ratio

    return GateResult(
        gate_id="G1-3",
        name="fact_single_ratio",
        passed=passed,
        blocking=False,
        value=ratio,
        threshold=max_ratio,
        message=f"fact_single ratio: {ratio:.1%} (max: {max_ratio:.0%})",
    )


def g1_4_question_format(question: str) -> GateResult:
    """
    G1-4: Question must end with '?'.

    BLOCKING: All questions must be properly formatted.
    """
    passed = question.strip().endswith("?")
    return GateResult(
        gate_id="G1-4",
        name="question_format",
        passed=passed,
        blocking=True,
        value=question[-10:] if question else "",
        threshold="ends with ?",
        message=f"Question {'ends' if passed else 'does not end'} with '?'",
    )


# =============================================================================
# PHASE 2: UNANSWERABLE GENERATION GATES
# =============================================================================


def g2_1_is_impossible_flag(question: dict) -> GateResult:
    """
    G2-1: Unanswerable questions must have is_impossible=true.

    BLOCKING: All Phase 2 questions must be flagged.
    """
    is_impossible = question.get("content", {}).get("is_impossible", False)
    return GateResult(
        gate_id="G2-1",
        name="is_impossible_flag",
        passed=is_impossible is True,
        blocking=True,
        value=is_impossible,
        threshold=True,
        message=f"is_impossible={is_impossible}",
    )


def g2_2_unanswerable_ratio(
    questions: list[dict],
    min_ratio: float = 0.25,
    max_ratio: float = 0.40,
) -> GateResult:
    """
    G2-2: Unanswerable ratio should be 25-40%.

    SQuAD 2.0 reference: train split ~33.4%, dev split ~50%.
    Range [25%, 40%] is between train and dev splits.

    WARNING: Ratio should be in target range.
    """
    total = len(questions)
    if total == 0:
        return GateResult(
            gate_id="G2-2",
            name="unanswerable_ratio",
            passed=False,
            blocking=False,
            value=0,
            threshold=f"[{min_ratio:.0%}, {max_ratio:.0%}]",
            message="No questions",
        )

    unanswerable = sum(
        1 for q in questions if q.get("content", {}).get("is_impossible", False)
    )
    ratio = unanswerable / total
    passed = min_ratio <= ratio <= max_ratio

    return GateResult(
        gate_id="G2-2",
        name="unanswerable_ratio",
        passed=passed,
        blocking=False,
        value=ratio,
        threshold=f"[{min_ratio:.0%}, {max_ratio:.0%}]",
        message=f"Unanswerable: {unanswerable}/{total} ({ratio:.1%})",
    )


def g2_3_hard_type_diversity(
    questions: list[dict],
    min_types: int = 4,
) -> GateResult:
    """
    G2-3: Unanswerable questions should have diverse hard_types.

    WARNING: At least 4 different hard_type categories (project-adapted from UAEval4RAG).
    """
    unanswerable = [
        q for q in questions if q.get("content", {}).get("is_impossible", False)
    ]
    hard_types = set(
        q.get("classification", {}).get("hard_type", "UNKNOWN") for q in unanswerable
    )
    hard_types.discard("UNKNOWN")
    hard_types.discard("ANSWERABLE")

    passed = len(hard_types) >= min_types

    return GateResult(
        gate_id="G2-3",
        name="hard_type_diversity",
        passed=passed,
        blocking=False,
        value=len(hard_types),
        threshold=min_types,
        message=f"Hard types: {len(hard_types)} ({', '.join(sorted(hard_types))})",
    )


# =============================================================================
# PHASE 3: ANTI-HALLUCINATION GATES
# =============================================================================


def g3_1_validation_passed(
    questions: list[dict],
    validation_results: dict[str, bool],
) -> GateResult:
    """
    G3-1: All questions must pass anti-hallucination validation.

    BLOCKING: 100% validation pass rate.
    """
    total = len(questions)
    passed_count = sum(1 for v in validation_results.values() if v)
    all_passed = passed_count == total

    return GateResult(
        gate_id="G3-1",
        name="validation_passed",
        passed=all_passed,
        blocking=True,
        value=f"{passed_count}/{total}",
        threshold="100%",
        message=f"Validation: {passed_count}/{total} passed",
    )


def g3_2_hallucination_count(rejected_ids: list[str]) -> GateResult:
    """
    G3-2: Zero hallucinations allowed.

    BLOCKING: No questions rejected for hallucination.
    """
    count = len(rejected_ids)
    passed = count == 0

    return GateResult(
        gate_id="G3-2",
        name="hallucination_count",
        passed=passed,
        blocking=True,
        value=count,
        threshold=0,
        message=f"Rejected for hallucination: {count}",
    )


# =============================================================================
# PHASE 4: SCHEMA V2 ENRICHMENT GATES
# =============================================================================


def count_schema_fields(question: dict) -> int:
    """Count populated fields in Schema v2 question."""
    count = 0

    # Root fields (2)
    if question.get("id"):
        count += 1
    if "legacy_id" in question:
        count += 1

    # Groups
    groups = [
        "content",
        "mcq",
        "provenance",
        "classification",
        "validation",
        "processing",
        "audit",
    ]

    for group in groups:
        group_data = question.get(group, {})
        if isinstance(group_data, dict):
            for key, value in group_data.items():
                if value is not None and value != "":
                    count += 1

    return count


def g4_1_schema_fields(question: dict, required_fields: int = 42) -> GateResult:
    """
    G4-1: All 42 top-level Schema v2 fields must be populated.

    Note: 42 fields at top level (not counting annales_source sub-fields).
    Full count with annales_source sub-fields would be 46.

    BLOCKING: Full schema compliance required.
    """
    field_count = count_schema_fields(question)
    passed = field_count >= required_fields

    return GateResult(
        gate_id="G4-1",
        name="schema_fields",
        passed=passed,
        blocking=True,
        value=field_count,
        threshold=required_fields,
        message=f"Fields: {field_count}/{required_fields}",
    )


def g4_2_chunk_match_method(question: dict) -> GateResult:
    """
    G4-2: BY DESIGN generation must use 'by_design_input' method.

    WARNING: Ensures proper provenance tracking.
    """
    method = question.get("processing", {}).get("chunk_match_method", "")
    passed = method == "by_design_input"

    return GateResult(
        gate_id="G4-2",
        name="chunk_match_method",
        passed=passed,
        blocking=False,
        value=method,
        threshold="by_design_input",
        message=f"Chunk match method: {method}",
    )


# =============================================================================
# PHASE 5: DISTRIBUTION AND DEDUPLICATION GATES
# =============================================================================


def g5_1_inter_question_similarity(
    embeddings: list[np.ndarray],
    threshold: float = 0.95,
) -> GateResult:
    """
    G5-1: No near-duplicate questions.

    WARNING: Inter-question similarity < 0.95
    """
    if len(embeddings) < 2:
        return GateResult(
            gate_id="G5-1",
            name="inter_question_similarity",
            passed=True,
            blocking=False,
            value=0,
            threshold=threshold,
            message="Less than 2 questions",
        )

    # Check all pairs
    max_sim = 0.0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            max_sim = max(max_sim, sim)

    passed = max_sim < threshold

    return GateResult(
        gate_id="G5-1",
        name="inter_question_similarity",
        passed=passed,
        blocking=False,
        value=max_sim,
        threshold=threshold,
        message=f"Max inter-question similarity: {max_sim:.3f}",
    )


def g5_2_anchor_independence(
    question_embedding: np.ndarray,
    chunk_embedding: np.ndarray,
    threshold: float = 0.90,
) -> GateResult:
    """
    G5-2: Question should not be too similar to its source chunk.

    BLOCKING: Anchor-positive similarity < 0.90 (project threshold for valid triplet training)
    """
    sim = _cosine_similarity(question_embedding, chunk_embedding)
    passed = sim < threshold

    return GateResult(
        gate_id="G5-2",
        name="anchor_independence",
        passed=passed,
        blocking=True,
        value=sim,
        threshold=threshold,
        message=f"Anchor-positive similarity: {sim:.3f} (max: {threshold})",
    )


def g5_3_final_fact_single_ratio(
    questions: list[dict],
    max_ratio: float = 0.60,
) -> GateResult:
    """
    G5-3: Final fact_single ratio check.

    WARNING: Same as G1-3, verified after balancing.
    """
    return g1_3_fact_single_ratio(questions, max_ratio)


def g5_4_hard_ratio(questions: list[dict], min_ratio: float = 0.10) -> GateResult:
    """
    G5-4: At least 10% of answerable questions should be hard (difficulty >= 0.7).

    BLOCKING: Ensures challenging evaluation set among real questions.
    Filters on answerable only — unanswerable difficulty values are LLM-generated
    and inflate the metric artificially.
    """
    answerable = [
        q for q in questions if not q.get("content", {}).get("is_impossible", False)
    ]
    if not answerable:
        return GateResult(
            gate_id="G5-4",
            name="hard_ratio",
            passed=False,
            blocking=True,
            value=0,
            threshold=min_ratio,
            message="No answerable questions",
        )

    hard = sum(
        1 for q in answerable if q.get("classification", {}).get("difficulty", 0) >= 0.7
    )
    ratio = hard / len(answerable)
    passed = ratio >= min_ratio

    return GateResult(
        gate_id="G5-4",
        name="hard_ratio",
        passed=passed,
        blocking=True,
        value=ratio,
        threshold=min_ratio,
        message=f"Hard answerable: {hard}/{len(answerable)} ({ratio:.1%})",
    )


def g5_5_final_unanswerable_ratio(
    questions: list[dict],
    min_ratio: float = 0.25,
    max_ratio: float = 0.40,
) -> GateResult:
    """
    G5-5: Final unanswerable ratio check.

    WARNING: Same as G2-2, verified after balancing.
    """
    return g2_2_unanswerable_ratio(questions, min_ratio, max_ratio)


def g5_6_cognitive_level_diversity(
    questions: list[dict], min_levels: int = 4
) -> GateResult:
    """
    G5-6: At least min_levels distinct Bloom cognitive levels.

    BLOCKING: Ensures pedagogical diversity (Remember, Understand, Apply, Analyze).
    """
    levels = {
        q.get("classification", {}).get("cognitive_level", "")
        for q in questions
        if q.get("classification", {}).get("cognitive_level")
    }
    passed = len(levels) >= min_levels

    return GateResult(
        gate_id="G5-6",
        name="cognitive_level_diversity",
        passed=passed,
        blocking=True,
        value=len(levels),
        threshold=min_levels,
        message=f"Cognitive levels: {len(levels)} ({', '.join(sorted(levels))})",
    )


def g5_7_question_type_diversity(
    questions: list[dict],
    required_types: set[str] | None = None,
) -> GateResult:
    """
    G5-7: Required question_type categories must be present.

    WARNING: Ensures diversity of question types (factual, procedural, scenario, comparative).
    """
    if required_types is None:
        required_types = {"factual", "procedural", "scenario", "comparative"}

    answerable = [
        q for q in questions if not q.get("content", {}).get("is_impossible", False)
    ]
    present_types = {
        q.get("classification", {}).get("question_type", "") for q in answerable
    }
    missing = required_types - present_types
    passed = len(missing) == 0

    return GateResult(
        gate_id="G5-7",
        name="question_type_diversity",
        passed=passed,
        blocking=False,
        value=len(present_types & required_types),
        threshold=len(required_types),
        message=f"Question types: {len(present_types & required_types)}/{len(required_types)}"
        + (f" (missing: {', '.join(sorted(missing))})" if missing else ""),
    )


def g5_8_chunk_coverage(
    covered_chunks: int, total_chunks: int, min_ratio: float = 0.80
) -> GateResult:
    """
    G5-8: At least min_ratio of corpus chunks should be covered by questions.

    WARNING: Ensures broad corpus coverage.
    """
    ratio = covered_chunks / total_chunks if total_chunks > 0 else 0.0
    passed = ratio >= min_ratio

    return GateResult(
        gate_id="G5-8",
        name="chunk_coverage",
        passed=passed,
        blocking=False,
        value=ratio,
        threshold=min_ratio,
        message=f"Chunk coverage: {covered_chunks}/{total_chunks} ({ratio:.1%})",
    )


# =============================================================================
# AGGREGATE VALIDATION
# =============================================================================


def validate_all_gates(
    questions: list[dict],
    strata: dict | None = None,
    coverage: dict | None = None,
    validation_results: dict[str, bool] | None = None,
    rejected_ids: list[str] | None = None,
    chunk_coverage_stats: tuple[int, int] | None = None,
) -> tuple[bool, list[GateResult]]:
    """
    Run all applicable quality gates.

    Args:
        questions: List of questions to validate
        strata: Stratification data (for G0 gates)
        coverage: Coverage statistics (for G0 gates)
        validation_results: Anti-hallucination results (for G3 gates)
        rejected_ids: IDs rejected for hallucination (for G3 gates)
        chunk_coverage_stats: Tuple (covered_chunks, total_chunks) for G5-8

    Returns:
        Tuple of (all_blocking_passed, list of GateResults)
    """
    results = []

    # G0 gates (if stratification data provided)
    if strata is not None:
        results.append(g0_1_strata_count(strata))
    if coverage is not None:
        results.append(g0_2_document_coverage(coverage))

    # G1 gates (per-question)
    for q in questions:
        if q.get("content", {}).get("is_impossible", False):
            continue  # Skip unanswerable for G1 gates
        results.append(g1_1_chunk_match_score(q))
        question_text = q.get("content", {}).get("question", "")
        results.append(g1_4_question_format(question_text))

    # G1-3 aggregate
    results.append(g1_3_fact_single_ratio(questions))

    # G2 gates (per-question for unanswerable)
    for q in questions:
        if q.get("content", {}).get("is_impossible", False):
            results.append(g2_1_is_impossible_flag(q))

    # G2 aggregate gates
    results.append(g2_2_unanswerable_ratio(questions))
    results.append(g2_3_hard_type_diversity(questions))

    # G3 gates (if validation data provided)
    if validation_results is not None:
        results.append(g3_1_validation_passed(questions, validation_results))
    if rejected_ids is not None:
        results.append(g3_2_hallucination_count(rejected_ids))

    # G4 gates (per-question)
    for q in questions:
        results.append(g4_1_schema_fields(q))
        results.append(g4_2_chunk_match_method(q))

    # G5 aggregate gates
    results.append(g5_3_final_fact_single_ratio(questions))
    results.append(g5_4_hard_ratio(questions))
    results.append(g5_5_final_unanswerable_ratio(questions))
    results.append(g5_6_cognitive_level_diversity(questions))
    results.append(g5_7_question_type_diversity(questions))

    # G5-8 chunk coverage (if stats provided)
    if chunk_coverage_stats is not None:
        covered, total = chunk_coverage_stats
        results.append(g5_8_chunk_coverage(covered, total))

    # Check if all blocking gates passed
    all_blocking_passed = all(r.passed for r in results if r.blocking)

    return all_blocking_passed, results


def format_gate_report(results: list[GateResult]) -> str:
    """Format gate results as a report."""
    lines = [
        "=" * 70,
        "QUALITY GATES REPORT",
        "=" * 70,
        "",
        "BLOCKING GATES:",
    ]

    blocking = [r for r in results if r.blocking]
    for r in blocking:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{status}] {r.gate_id}: {r.name}")
        lines.append(f"         {r.message}")

    lines.append("")
    lines.append("WARNING GATES:")

    warnings = [r for r in results if not r.blocking]
    for r in warnings:
        status = "PASS" if r.passed else "WARN"
        lines.append(f"  [{status}] {r.gate_id}: {r.name}")
        lines.append(f"         {r.message}")

    # Summary
    blocking_passed = sum(1 for r in blocking if r.passed)
    warning_passed = sum(1 for r in warnings if r.passed)

    lines.extend(
        [
            "",
            "=" * 70,
            "SUMMARY",
            "=" * 70,
            f"Blocking gates: {blocking_passed}/{len(blocking)} passed",
            f"Warning gates: {warning_passed}/{len(warnings)} passed",
            "",
        ]
    )

    all_blocking = blocking_passed == len(blocking)
    if all_blocking:
        lines.append("[OK] All blocking gates passed")
    else:
        lines.append("[FAIL] Some blocking gates failed - generation blocked")

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """CLI entry point for testing gates."""
    import argparse
    from pathlib import Path

    from scripts.pipeline.utils import load_json

    parser = argparse.ArgumentParser(description="Validate GS against quality gates")
    parser.add_argument(
        "--gs",
        "-g",
        type=Path,
        required=True,
        help="Gold Standard JSON file",
    )
    parser.add_argument(
        "--strata",
        "-s",
        type=Path,
        help="Stratification JSON file",
    )

    args = parser.parse_args()

    # Load data
    gs_data = load_json(args.gs)
    questions = gs_data.get("questions", [])

    strata = None
    coverage = None
    if args.strata and args.strata.exists():
        strata_data = load_json(args.strata)
        strata = strata_data.get("strata", {})
        coverage = strata_data.get("coverage", {})

    # Run validation
    all_passed, results = validate_all_gates(
        questions,
        strata=strata,
        coverage=coverage,
    )

    # Print report
    report = format_gate_report(results)
    print(report)

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
