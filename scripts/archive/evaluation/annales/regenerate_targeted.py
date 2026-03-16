"""Targeted question regeneration for GS Phase A-P2.

Selects ~80 answerable questions and replaces them with higher-quality
alternatives matching 4 target profiles (hard/medium x Apply/Analyze).

ISO Reference: ISO/IEC 42001 - AI quality, ISO/IEC 25010 - Data quality
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.evaluation.annales.fix_gs_v2_metadata import (  # noqa: E402
    _sync_coverage_header,
)
from scripts.pipeline.utils import get_date, load_json, save_json  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict[str, Any]] = {
    "HARD_APPLY": {
        "cognitive_level": "Apply",
        "difficulty_min": 0.7,
        "difficulty_max": 1.0,
        "question_type": "scenario",
        "answer_type": "extractive",
    },
    "HARD_ANALYZE": {
        "cognitive_level": "Analyze",
        "difficulty_min": 0.7,
        "difficulty_max": 1.0,
        "question_type": "comparative",
        "answer_type": "inferential",
    },
    "MED_APPLY_INF": {
        "cognitive_level": "Apply",
        "difficulty_min": 0.5,
        "difficulty_max": 0.7,
        "question_type": "procedural",
        "answer_type": "inferential",
    },
    "MED_ANALYZE_COMP": {
        "cognitive_level": "Analyze",
        "difficulty_min": 0.4,
        "difficulty_max": 0.6,
        "question_type": "comparative",
        "answer_type": "extractive",
    },
}

N_PER_PROFILE = 20


@dataclass
class RegenTask:
    """A single question targeted for regeneration."""

    old_id: str
    chunk_id: str
    chunk_text: str
    profile: str


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _replaceability_score(q: dict) -> float:
    """Score how replaceable a question is (higher = better candidate).

    Args:
        q: GS question dict.

    Returns:
        Float score (higher = more replaceable).
    """
    score = 0.0
    cls = q.get("classification", {})

    level = cls.get("cognitive_level", "")
    if level == "Remember":
        score += 3.0
    elif level == "Understand":
        score += 2.0
    elif level == "Apply":
        score += 0.5

    qtype = cls.get("question_type", "")
    if qtype == "factual":
        score += 2.0
    elif qtype == "procedural":
        score += 1.0

    rclass = cls.get("reasoning_class", "")
    if rclass == "fact_single":
        score += 2.0
    elif rclass == "summary":
        score += 1.0

    diff = cls.get("difficulty", 0.5)
    score += (1.0 - diff) * 2.0

    return score


def select_candidates(
    gs_data: dict,
    chunk_index: dict[str, str],
    n_per_profile: int = N_PER_PROFILE,
) -> dict[str, list[RegenTask]]:
    """Select answerable questions for regeneration across 4 profiles.

    Prioritizes replaceable questions (low cognitive, easy difficulty,
    simple types) and maximizes chunk diversity.

    Args:
        gs_data: Full GS JSON data.
        chunk_index: Mapping chunk_id -> chunk_text.
        n_per_profile: Questions per profile (default 20).

    Returns:
        Dict mapping profile name to list of RegenTask.
    """
    questions = gs_data.get("questions", [])

    # Filter: answerable only, chunk exists with substantive text
    candidates = []
    for q in questions:
        if q.get("content", {}).get("is_impossible", False):
            continue
        chunk_id = q.get("provenance", {}).get("chunk_id", "")
        chunk_text = chunk_index.get(chunk_id, "")
        if len(chunk_text) <= 50:
            continue
        candidates.append(q)

    # Sort by replaceability (most replaceable first)
    candidates.sort(key=_replaceability_score, reverse=True)

    used_ids: set[str] = set()
    used_chunks: set[str] = set()
    result: dict[str, list[RegenTask]] = {p: [] for p in PROFILES}

    # Pass 1: unique chunks per profile
    for profile_name in PROFILES:
        for q in candidates:
            if len(result[profile_name]) >= n_per_profile:
                break
            qid = q["id"]
            chunk_id = q["provenance"]["chunk_id"]
            if qid in used_ids or chunk_id in used_chunks:
                continue
            result[profile_name].append(
                RegenTask(
                    old_id=qid,
                    chunk_id=chunk_id,
                    chunk_text=chunk_index[chunk_id],
                    profile=profile_name,
                )
            )
            used_ids.add(qid)
            used_chunks.add(chunk_id)

    # Pass 2: fill remaining (allow shared chunks)
    for profile_name in PROFILES:
        if len(result[profile_name]) >= n_per_profile:
            continue
        for q in candidates:
            if len(result[profile_name]) >= n_per_profile:
                break
            qid = q["id"]
            if qid in used_ids:
                continue
            chunk_id = q["provenance"]["chunk_id"]
            result[profile_name].append(
                RegenTask(
                    old_id=qid,
                    chunk_id=chunk_id,
                    chunk_text=chunk_index[chunk_id],
                    profile=profile_name,
                )
            )
            used_ids.add(qid)

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_CONTENT_KEYS = {"question", "expected_answer", "is_impossible"}
REQUIRED_TOP_KEYS = {"content", "classification", "provenance"}


def _validate_new_question(q: dict) -> list[str]:
    """Validate a replacement question has required structure.

    Args:
        q: New question dict to validate.

    Returns:
        List of error messages (empty if valid).
    """
    errs: list[str] = []
    for key in REQUIRED_TOP_KEYS:
        if key not in q or not isinstance(q[key], dict):
            errs.append(f"missing or invalid '{key}'")
    content = q.get("content", {})
    for key in REQUIRED_CONTENT_KEYS:
        if key not in content:
            errs.append(f"missing content.{key}")
    question_text = content.get("question", "")
    if not question_text.strip().endswith("?"):
        errs.append("question doesn't end with '?'")
    answer_text = content.get("expected_answer", "")
    if len(answer_text) <= 5:
        errs.append(f"expected_answer too short ({len(answer_text)} chars)")
    cls = q.get("classification", {})
    diff = cls.get("difficulty", -1)
    if not 0.0 <= diff <= 1.0:
        errs.append(f"difficulty out of range: {diff}")
    cog = cls.get("cognitive_level", "")
    if cog not in {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"}:
        errs.append(f"invalid cognitive_level: {cog!r}")
    return errs


# ---------------------------------------------------------------------------
# Replacement
# ---------------------------------------------------------------------------


def apply_replacements(
    gs_data: dict,
    replacements: list[dict],
    date: str | None = None,
) -> dict:
    """Apply replacement questions to the GS (mutates in place).

    Each replacement dict must contain:
        - old_id: ID of question to replace
        - profile: Target profile name
        - new_question: Complete Schema v2 question dict

    Args:
        gs_data: Full GS JSON data (mutated in place).
        replacements: List of replacement dicts.
        date: Date string (default: today).

    Returns:
        Report dict with keys: date, total_replacements, errors,
        total_questions.
    """
    if date is None:
        date = get_date()

    questions = gs_data.get("questions", [])
    q_by_id = {q["id"]: i for i, q in enumerate(questions)}

    replaced = 0
    errors: list[str] = []

    for repl in replacements:
        old_id = repl["old_id"]
        profile = repl["profile"]
        new_q = repl["new_question"]

        if old_id not in q_by_id:
            errors.append(f"ID not found: {old_id}")
            continue

        # Validate new question schema
        validation_errs = _validate_new_question(new_q)
        if validation_errs:
            errors.append(
                f"Invalid question for {old_id}: {'; '.join(validation_errs)}"
            )
            continue

        # Require chunk_match_score already set to 100
        cms = new_q.get("processing", {}).get("chunk_match_score")
        if cms != 100:
            errors.append(f"chunk_match_score={cms} for {old_id} (must be 100)")
            continue

        idx = q_by_id[old_id]

        # Keep original ID — content changes, identity stays
        new_q["id"] = old_id

        # Audit trail
        audit = new_q.get("audit", {})
        history = audit.get("history", "")
        tag = f"[PHASE A-P2] regenerated {profile} " f"on {date}"
        audit["history"] = f"{history} | {tag}" if history else tag
        new_q["audit"] = audit

        # Validation batch
        new_q.setdefault("validation", {})["batch"] = "gs_v1_step1_p2"

        questions[idx] = new_q
        replaced += 1

    _sync_coverage_header(gs_data)

    return {
        "date": date,
        "total_replacements": replaced,
        "errors": errors,
        "total_questions": len(questions),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point for targeted regeneration."""
    parser = argparse.ArgumentParser(
        description="Targeted GS question regeneration (Phase A-P2)",
    )
    parser.add_argument(
        "--gs",
        type=Path,
        required=True,
        help="GS JSON file",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        help="Chunks JSON file",
    )
    parser.add_argument(
        "--replacements",
        type=Path,
        help="Replacements JSON file",
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Select candidates and output JSON",
    )
    parser.add_argument(
        "--apply-only",
        action="store_true",
        help="Apply replacements from --replacements",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output GS file",
    )
    args = parser.parse_args()

    gs_data = load_json(args.gs)

    if args.select_only:
        if not args.chunks:
            parser.error("--chunks required for --select-only")
        chunks_data = load_json(args.chunks)
        chunk_idx = {c["id"]: c["text"] for c in chunks_data.get("chunks", [])}
        tasks = select_candidates(gs_data, chunk_idx)
        output = {p: [asdict(t) for t in tl] for p, tl in tasks.items()}
        print(json.dumps(output, indent=2, ensure_ascii=False))
        total = sum(len(v) for v in tasks.values())
        print(f"\n# Total: {total} candidates", file=sys.stderr)
        return 0

    if args.apply_only:
        if not args.replacements:
            parser.error("--replacements required for --apply-only")
        repls = load_json(args.replacements)
        if not isinstance(repls, list):
            repls = repls.get("replacements", [])
        report = apply_replacements(gs_data, repls)
        out = args.output or args.gs
        save_json(gs_data, out)
        print(f"Applied {report['total_replacements']} replacements")
        print(f"Total questions: {report['total_questions']}")
        if report["errors"]:
            print(f"Errors: {report['errors']}")
        return 1 if report["errors"] else 0

    parser.error("Must specify --select-only or --apply-only")
    return 1  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
