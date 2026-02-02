"""Phase 4: Audit fixes post GS v8.0.

Execution order (sequential):
  Fix 3: Reclassify false positives and exam_name:* tags
  Fix 2: Unified taxonomy for requires_context_reason
  Fix 1: Add metadata.correct_answer field
  Fix 4: Strip ## markdown prefixes from expected_answer
  Fix 5: Add difficulty variance for 34 human questions

Quality gates verify all fixes + full regression suite.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GS_PATH = Path("tests/data/gold_standard_annales_fr_v7.json")
CHUNKS_PATH = Path("corpus/processed/chunks_mode_b_fr.json")

# --- Fix 3: Manual reclassifications ---

# 3 false positives: "position" keyword triggered requires_position_diagram
# but these questions are about administrative contexts, not chess diagrams.
FALSE_POSITIVE_RECLASSIFICATIONS: dict[str, str] = {
    "open:001": "insufficient_context",  # administrative criteria
    "open:005": "insufficient_context",  # training internship
    "rules:008": "insufficient_context",  # post-game analysis
}

# 9 exam_name:* questions — reviewer tag, not a reason.
# Each gets a proper reason + reviewer_tag extracted.
EXAM_NAME_RECLASSIFICATIONS: dict[str, str] = {
    "clubs:034": "insufficient_context",
    "clubs:035": "insufficient_context",
    "clubs:166": "insufficient_context",
    "open:012": "insufficient_context",
    "open:019": "numerical_reasoning",
    "open:025": "numerical_reasoning",
    "rules:055": "case_analysis",
    "rules:058": "case_analysis",
    "tournament:045": "numerical_reasoning",
}

# --- Fix 2: Taxonomy mapping old -> new ---
# Standards: Know Your RAG (COLING 2025), SQuAD 2.0, HotpotQA, RAGAS, ISO 5259

TAXONOMY_MAPPING: dict[str, str] = {
    "requires_calculation": "numerical_reasoning",
    "requires_position_diagram": "visual_reasoning",
    "requires_cross_reference": "multi_hop_reasoning",
    "answer_not_in_corpus": "answer_not_in_corpus",  # unchanged
    "chunk_insufficient_context": "insufficient_context",
    "requires_specific_context": "insufficient_context",
    "requires_competition_context": "insufficient_context",
    "requires_external_data": "external_knowledge",
    "failure_analysis": "case_analysis",
}

VALID_NEW_REASONS: set[str] = {
    "numerical_reasoning",
    "visual_reasoning",
    "multi_hop_reasoning",
    "answer_not_in_corpus",
    "insufficient_context",
    "external_knowledge",
    "case_analysis",
}

# --- Fix 5: Analytical keywords for difficulty variance ---

ANALYTICAL_KEYWORDS: set[str] = {
    "pourquoi",
    "expliquez",
    "comparez",
    "analysez",
    "justifiez",
    "argumentez",
    "commentez",
    "décrivez",
    "évaluez",
}


def load_json(path: Path) -> Any:
    """Load JSON with UTF-8."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    """Save JSON with consistent formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n")


def match_short_id(question_id: str, short_id: str) -> bool:
    """Match a full question ID against a short form like 'open:001'."""
    # Full ID: ffe:annales:open:001:hash or ffe:human:open:001:hash
    parts = question_id.split(":")
    # Short ID: category:number
    short_parts = short_id.split(":")
    if len(short_parts) != 2 or len(parts) < 4:
        return False
    # parts[2] = category (clubs/open/rules/tournament)
    # parts[3] = number (001, 034, etc.)
    return parts[2] == short_parts[0] and parts[3] == short_parts[1]


def find_question(
    questions: list[dict[str, Any]], short_id: str
) -> dict[str, Any] | None:
    """Find a question by short ID."""
    for q in questions:
        if match_short_id(q["id"], short_id):
            return q
    return None


def fix3_reclassify(
    questions: list[dict[str, Any]], verbose: bool = False
) -> dict[str, int]:
    """Fix 3: Reclassify false positives and exam_name:* tags."""
    stats = {
        "false_positives_fixed": 0,
        "exam_name_reclassified": 0,
        "reviewer_tags": 0,
    }

    # 3 false positives
    for short_id, new_reason in FALSE_POSITIVE_RECLASSIFICATIONS.items():
        q = find_question(questions, short_id)
        if q is None:
            print(f"  WARNING: {short_id} not found")
            continue
        md = q.get("metadata", {})
        old_reason = md.get("requires_context_reason", "")
        md["requires_context_reason"] = new_reason
        q["metadata"] = md
        stats["false_positives_fixed"] += 1
        if verbose:
            print(f"  FP: {q['id']} {old_reason} -> {new_reason}")

    # 9 exam_name:* questions
    for short_id, new_reason in EXAM_NAME_RECLASSIFICATIONS.items():
        q = find_question(questions, short_id)
        if q is None:
            print(f"  WARNING: {short_id} not found")
            continue
        md = q.get("metadata", {})
        old_reason = md.get("requires_context_reason", "")

        # Extract reviewer name from exam_name:Name
        if old_reason.startswith("exam_name:"):
            reviewer = old_reason.split(":", 1)[1]
            md["reviewer_tag"] = reviewer
            stats["reviewer_tags"] += 1

        md["requires_context_reason"] = new_reason
        q["metadata"] = md
        stats["exam_name_reclassified"] += 1
        if verbose:
            print(f"  EN: {q['id']} {old_reason} -> {new_reason}")

    return stats


def fix2_taxonomy(
    questions: list[dict[str, Any]], verbose: bool = False
) -> dict[str, int]:
    """Fix 2: Apply unified taxonomy mapping to requires_context_reason."""
    stats = {"mapped": 0, "already_new": 0, "unknown": 0}

    for q in questions:
        md = q.get("metadata", {})
        if not md.get("requires_context"):
            continue
        reason = md.get("requires_context_reason", "")
        if not reason:
            continue

        # Skip exam_name:* — should already be handled by Fix 3
        if reason.startswith("exam_name:"):
            stats["unknown"] += 1
            if verbose:
                print(f"  WARN: residual exam_name: {q['id']} -> {reason}")
            continue

        if reason in VALID_NEW_REASONS:
            stats["already_new"] += 1
            continue

        new_reason = TAXONOMY_MAPPING.get(reason)
        if new_reason is None:
            stats["unknown"] += 1
            if verbose:
                print(f"  UNKNOWN: {q['id']} -> {reason}")
            continue

        md["requires_context_reason"] = new_reason
        q["metadata"] = md
        stats["mapped"] += 1
        if verbose:
            print(f"  MAP: {q['id']} {reason} -> {new_reason}")

    return stats


def _resolve_mcq_correct_answer(
    md: dict[str, Any], q_id: str, verbose: bool
) -> tuple[str, str]:
    """Resolve correct answer for an annales MCQ question.

    Returns (correct_answer, stat_key).
    """
    choices = md.get("choices", {})
    mcq_answer = md.get("mcq_answer", "")

    # Single answer: direct lookup
    correct = choices.get(mcq_answer, "")
    if correct:
        return correct, "annales_set"

    # Multi-answer MCQ (e.g. "ADE" -> concat individual choices)
    if len(mcq_answer) > 1 and all(c in choices for c in mcq_answer):
        parts = [choices[c] for c in mcq_answer]
        if verbose:
            print(f"  MULTI: {q_id} {mcq_answer} -> {len(parts)} parts")
        return " | ".join(parts), "multi_answer"

    # Fallback to original_answer for incomplete choices data
    original = md.get("original_answer", "")
    if original:
        if verbose:
            print(f"  FALLBACK: {q_id} mcq={mcq_answer} -> original_answer")
        return original, "multi_answer"

    if verbose:
        print(f"  SKIP: {q_id} mcq_answer={mcq_answer} not resolvable")
    return "", "skipped"


def fix1_correct_answer(
    questions: list[dict[str, Any]], verbose: bool = False
) -> dict[str, int]:
    """Fix 1: Add metadata.correct_answer field."""
    stats = {"annales_set": 0, "human_set": 0, "multi_answer": 0, "skipped": 0}

    for q in questions:
        md = q.get("metadata", {})

        if md.get("choices") and md.get("mcq_answer"):
            answer, key = _resolve_mcq_correct_answer(md, q["id"], verbose)
            if answer:
                md["correct_answer"] = answer
            stats[key] += 1
        else:
            # Human: correct_answer = original_answer
            original = md.get("original_answer", "")
            if original:
                md["correct_answer"] = original
                stats["human_set"] += 1
            else:
                stats["skipped"] += 1
                if verbose:
                    print(f"  SKIP: {q['id']} no original_answer")

        q["metadata"] = md

    return stats


def fix4_markdown_cleanup(
    questions: list[dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    verbose: bool = False,
) -> dict[str, int]:
    """Fix 4: Strip ## markdown heading prefixes from expected_answer."""
    stats = {"stripped": 0, "rollback": 0, "not_needed": 0}
    heading_re = re.compile(r"^#{1,3}\s+")

    for q in questions:
        answer = q.get("expected_answer", "")
        if not heading_re.match(answer):
            stats["not_needed"] += 1
            continue

        cleaned = heading_re.sub("", answer)
        chunk_id = q.get("expected_chunk_id", "")
        chunk_text = chunks.get(chunk_id, {}).get("text", "")

        # Safety checks: CB-01 (answer in chunk) and F-04 (answer > 5 chars)
        if cleaned in chunk_text and len(cleaned) > 5:
            q["expected_answer"] = cleaned
            stats["stripped"] += 1
            if verbose:
                print(f"  STRIP: {q['id']} ({answer[:40]}...)")
        else:
            # Rollback: keep original to preserve CB-01 or F-04
            stats["rollback"] += 1
            if verbose:
                reason = "CB-01" if cleaned not in chunk_text else "F-04"
                print(f"  ROLLBACK: {q['id']} ({reason} would fail)")

    return stats


def fix5_difficulty_variance(
    questions: list[dict[str, Any]], verbose: bool = False
) -> dict[str, int]:
    """Fix 5: Add variance to human question difficulty values."""
    stats = {"adjusted": 0, "unchanged": 0}

    for q in questions:
        if not q["id"].startswith("ffe:human:"):
            continue
        md = q.get("metadata", {})
        base_difficulty = md.get("difficulty")
        if base_difficulty is None:
            continue

        question_text = q.get("question", "")
        answer_text = md.get("original_answer", "")
        delta = 0.0

        # Long question bonus
        if len(question_text) > 100:
            delta += 0.05

        # Long answer bonus
        if len(answer_text) > 200:
            delta += 0.05

        # Analytical keywords bonus
        q_lower = question_text.lower()
        if any(kw in q_lower for kw in ANALYTICAL_KEYWORDS):
            delta += 0.05

        if delta > 0:
            new_difficulty = min(1.0, max(0.0, base_difficulty + delta))
            md["difficulty"] = round(new_difficulty, 2)
            q["metadata"] = md
            stats["adjusted"] += 1
            if verbose:
                print(
                    f"  ADJ: {q['id']} {base_difficulty} -> {md['difficulty']} "
                    f"(+{delta:.2f})"
                )
        else:
            stats["unchanged"] += 1

    return stats


def _correct_answer_valid(q: dict[str, Any]) -> bool:
    """Check correct_answer is consistent with choices/mcq_answer."""
    md = q["metadata"]
    ca = md.get("correct_answer", "")
    choices = md.get("choices", {})
    mcq = md.get("mcq_answer", "")
    if not ca:
        return False
    if mcq in choices:
        return ca == choices[mcq]
    return len(ca) > 0


def _gate_fix1(questions: list[dict[str, Any]]) -> list[str]:
    """Gate for Fix 1: correct_answer field."""
    errors: list[str] = []
    total = len(questions)
    ca_present = sum(
        1 for q in questions if q.get("metadata", {}).get("correct_answer")
    )
    ca_nonempty = sum(
        1 for q in questions if q.get("metadata", {}).get("correct_answer", "").strip()
    )
    annales = [q for q in questions if q.get("metadata", {}).get("choices")]
    ca_valid = sum(1 for q in annales if _correct_answer_valid(q))

    print("\n--- Fix 1 - correct_answer ---")
    print(f"  correct_answer present:    {ca_present}/{total}")
    print(f"  correct_answer non-empty:  {ca_nonempty}/{total}")
    print(f"  annales valid:             {ca_valid}/{len(annales)}")
    if ca_present != total:
        errors.append(f"Fix1: correct_answer present {ca_present}/{total}")
    if ca_nonempty != total:
        errors.append(f"Fix1: correct_answer non-empty {ca_nonempty}/{total}")
    if ca_valid != len(annales):
        errors.append(f"Fix1: annales valid {ca_valid}/{len(annales)}")
    return errors


def _gate_fix2(
    rc_questions: list[dict[str, Any]],
) -> list[str]:
    """Gate for Fix 2: unified taxonomy."""
    errors: list[str] = []
    reasons = [
        q.get("metadata", {}).get("requires_context_reason", "") for q in rc_questions
    ]
    valid = sum(1 for r in reasons if r in VALID_NEW_REASONS)
    old = sum(
        1 for r in reasons if r in TAXONOMY_MAPPING and r not in VALID_NEW_REASONS
    )
    exam = sum(1 for r in reasons if r.startswith("exam_name:"))

    print("\n--- Fix 2 - taxonomy ---")
    print(f"  valid reasons (7 cats):    {valid}/{len(rc_questions)}")
    print(f"  old taxonomy remaining:    {old}")
    print(f"  exam_name remaining:       {exam}")
    if valid != len(rc_questions):
        errors.append(f"Fix2: valid reasons {valid}/{len(rc_questions)}")
    if old > 0:
        errors.append(f"Fix2: {old} old taxonomy values remaining")
    if exam > 0:
        errors.append(f"Fix2: {exam} exam_name values remaining")
    return errors


def _gate_fix3(questions: list[dict[str, Any]]) -> list[str]:
    """Gate for Fix 3: reclassification."""
    errors: list[str] = []
    fp = sum(
        1
        for sid in FALSE_POSITIVE_RECLASSIFICATIONS
        if (q := find_question(questions, sid)) is not None
        and q.get("metadata", {}).get("requires_context_reason")
        != "requires_position_diagram"
    )
    en = sum(
        1
        for sid in EXAM_NAME_RECLASSIFICATIONS
        if (q := find_question(questions, sid)) is not None
        and not q.get("metadata", {})
        .get("requires_context_reason", "")
        .startswith("exam_name:")
    )
    tags = sum(
        1
        for sid in EXAM_NAME_RECLASSIFICATIONS
        if (q := find_question(questions, sid)) is not None
        and q.get("metadata", {}).get("reviewer_tag")
    )

    print("\n--- Fix 3 - reclassification ---")
    print(f"  false positives fixed:     {fp}/3")
    print(f"  exam_name reclassified:    {en}/9")
    print(f"  reviewer_tag preserved:    {tags}/9")
    if fp != 3:
        errors.append(f"Fix3: false positives {fp}/3")
    if en != 9:
        errors.append(f"Fix3: exam_name {en}/9")
    if tags != 9:
        errors.append(f"Fix3: reviewer_tag {tags}/9")
    return errors


def _gate_fix45(
    questions: list[dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
) -> tuple[list[str], int]:
    """Gate for Fix 4 (markdown) and Fix 5 (difficulty). Returns (errors, cb01)."""
    errors: list[str] = []
    total = len(questions)

    heading_re = re.compile(r"^#{1,3}\s+")
    remaining = sum(
        1 for q in questions if heading_re.match(q.get("expected_answer", ""))
    )
    cb01 = sum(
        1
        for q in questions
        if q.get("expected_answer", "")
        in chunks.get(q.get("expected_chunk_id", ""), {}).get("text", "")
    )
    print("\n--- Fix 4 - markdown cleanup ---")
    print(f"  ## remaining:              {remaining}")
    print(f"  CB-01 post-cleanup:        {cb01}/{total}")
    if cb01 != total:
        errors.append(f"Fix4: CB-01 post-cleanup {cb01}/{total}")

    human_qs = [q for q in questions if q["id"].startswith("ffe:human:")]
    human_diffs = {q["metadata"]["difficulty"] for q in human_qs}
    all_ok = all(
        0.0 <= q.get("metadata", {}).get("difficulty", -1) <= 1.0 for q in questions
    )
    print("\n--- Fix 5 - difficulty ---")
    print(f"  distinct human values:     {len(human_diffs)} (was 4)")
    print(f"  all in [0,1]:              {all_ok}")
    if len(human_diffs) <= 4:
        errors.append(f"Fix5: distinct human values {len(human_diffs)} <= 4")
    if not all_ok:
        errors.append("Fix5: not all difficulties in [0,1]")

    return errors, cb01


def _check_regression(
    questions: list[dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    cb01_count: int,
    rc_questions: list[dict[str, Any]],
) -> list[str]:
    """Run full regression suite. Returns list of errors."""
    errors: list[str] = []
    total = len(questions)

    checks = {
        "CB-04": sum(
            1
            for q in questions
            if q.get("metadata", {}).get("chunk_match_method") == "manual_by_design"
        ),
        "F-01": sum(
            1 for q in questions if q.get("question", "").strip().endswith("?")
        ),
        "F-04": sum(1 for q in questions if len(q.get("expected_answer", "")) > 5),
        "CB-02": sum(1 for q in questions if q.get("expected_chunk_id", "") in chunks),
        "CB-03": sum(1 for q in questions if q.get("expected_chunk_id")),
        "CB-07": sum(1 for q in questions if q.get("expected_docs")),
        "M-01": sum(
            1 for q in questions if q.get("metadata", {}).get("difficulty") is not None
        ),
        "M-02": sum(
            1
            for q in questions
            if isinstance(q.get("metadata", {}).get("difficulty"), int | float)
            and 0 <= q["metadata"]["difficulty"] <= 1
        ),
        "M-03": sum(
            1 for q in questions if q.get("metadata", {}).get("cognitive_level")
        ),
        "M-04": sum(1 for q in questions if q.get("category")),
    }
    cb09 = sum(
        1 for q in rc_questions if q.get("metadata", {}).get("requires_context_reason")
    )

    print("\n=== REGRESSION TOTALE ===")
    print(f"  CB-01 answer in chunk:     {cb01_count}/{total}")
    for label, val in checks.items():
        print(f"  {label:5s}:                    {val}/{total}")
    print(f"  total:                     {total}")
    print(f"  CB-09 reason present:      {cb09}/{len(rc_questions)}")

    if cb01_count != total:
        errors.append(f"REGR: CB-01 {cb01_count}/{total}")
    for label, val in checks.items():
        if val != total:
            errors.append(f"REGR: {label} {val}/{total}")
    if total != 420:
        errors.append(f"REGR: total {total}/420")
    if cb09 != len(rc_questions):
        errors.append(f"REGR: CB-09 {cb09}/{len(rc_questions)}")

    return errors


def run_quality_gates(
    questions: list[dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Run all quality gates and regression checks."""
    rc_questions = [
        q for q in questions if q.get("metadata", {}).get("requires_context")
    ]
    errors: list[str] = []
    errors.extend(_gate_fix1(questions))
    errors.extend(_gate_fix2(rc_questions))
    errors.extend(_gate_fix3(questions))
    fix45_errors, cb01_count = _gate_fix45(questions, chunks)
    errors.extend(fix45_errors)
    errors.extend(_check_regression(questions, chunks, cb01_count, rc_questions))
    return len(errors) == 0, errors


def main() -> None:
    """Run Phase 4 audit fixes."""
    parser = argparse.ArgumentParser(description="Phase 4: Audit fixes post GS v8.0")
    parser.add_argument("--check", action="store_true", help="Dry-run mode")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    gs = load_json(GS_PATH)
    chunks_raw = load_json(CHUNKS_PATH)
    chunks_list = (
        chunks_raw.get("chunks", chunks_raw)
        if isinstance(chunks_raw, dict)
        else chunks_raw
    )
    chunks: dict[str, dict[str, Any]] = {c["id"]: c for c in chunks_list}
    questions = gs["questions"]

    print("=" * 60)
    print("Phase 4: Audit fixes post GS v8.0")
    print(f"Questions: {len(questions)}, Chunks: {len(chunks)}")
    print("=" * 60)

    # --- Fix 3: Reclassify (before taxonomy mapping) ---
    print("\n[Fix 3] Reclassify false positives + exam_name:*")
    s3 = fix3_reclassify(questions, verbose=args.verbose)
    print(f"  false_positives_fixed: {s3['false_positives_fixed']}")
    print(f"  exam_name_reclassified: {s3['exam_name_reclassified']}")
    print(f"  reviewer_tags: {s3['reviewer_tags']}")

    # --- Fix 2: Unified taxonomy ---
    print("\n[Fix 2] Unified taxonomy mapping")
    s2 = fix2_taxonomy(questions, verbose=args.verbose)
    print(f"  mapped: {s2['mapped']}")
    print(f"  already_new: {s2['already_new']}")
    print(f"  unknown: {s2['unknown']}")

    # --- Fix 1: correct_answer ---
    print("\n[Fix 1] Add metadata.correct_answer")
    s1 = fix1_correct_answer(questions, verbose=args.verbose)
    print(f"  annales_set: {s1['annales_set']}")
    print(f"  multi_answer: {s1['multi_answer']}")
    print(f"  human_set: {s1['human_set']}")
    print(f"  skipped: {s1['skipped']}")

    # --- Fix 4: Markdown cleanup ---
    print("\n[Fix 4] Strip ## markdown from expected_answer")
    s4 = fix4_markdown_cleanup(questions, chunks, verbose=args.verbose)
    print(f"  stripped: {s4['stripped']}")
    print(f"  rollback: {s4['rollback']}")
    print(f"  not_needed: {s4['not_needed']}")

    # --- Fix 5: Difficulty variance ---
    print("\n[Fix 5] Difficulty variance for human questions")
    s5 = fix5_difficulty_variance(questions, verbose=args.verbose)
    print(f"  adjusted: {s5['adjusted']}")
    print(f"  unchanged: {s5['unchanged']}")

    # --- Update methodology ---
    gs["methodology"]["p4_audit_fixes"] = {
        "date": datetime.now(timezone.utc).isoformat(),
        "fixes": [
            "correct_answer field",
            "unified taxonomy",
            "false positive reclassification",
            "markdown heading cleanup",
            "difficulty variance",
        ],
        "fix3_reclassified": s3["false_positives_fixed"] + s3["exam_name_reclassified"],
        "fix2_taxonomy_mapped": s2["mapped"],
        "fix1_correct_answer": s1["annales_set"] + s1["human_set"],
        "fix4_markdown_stripped": s4["stripped"],
        "fix5_difficulty_adjusted": s5["adjusted"],
    }

    # --- Quality Gates ---
    print("\n" + "=" * 60)
    print("=== GATE P4 ===")
    print("=" * 60)

    all_pass, gate_errors = run_quality_gates(questions, chunks)

    print("\n" + "=" * 60)
    if all_pass:
        print("GATE P4: PASS")
    else:
        print(f"GATE P4: FAIL ({len(gate_errors)} errors)")
        for err in gate_errors:
            print(f"  {err}")
    print("=" * 60)

    if not args.check and all_pass:
        save_json(gs, GS_PATH)
        print(f"\nSaved to {GS_PATH}")
    elif args.check:
        print("\nDry-run mode - no changes saved")
    else:
        print("\nGate FAIL - no changes saved")


if __name__ == "__main__":
    main()
