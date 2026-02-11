"""Phase 3: Metadata completion, version bump.

Tasks:
- 3a: M-01/M-02 difficulty for 34 human questions (from cognitive_level)
- 3b: CB-09 requires_context_reason for 42 questions
- 3c: Version bump to "8.0"

Difficulty mapping (from plan):
  Remember    -> 0.2-0.3
  Understand  -> 0.3-0.5
  Apply       -> 0.5-0.6
  Analyze     -> 0.6-0.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.pipeline.utils import load_json

GS_PATH = Path("tests/data/gold_standard_annales_fr_v7.json")
CHUNKS_PATH = Path("corpus/processed/chunks_mode_b_fr.json")

# Difficulty mapping based on cognitive level
DIFFICULTY_MAP: dict[str, float] = {
    "Remember": 0.25,
    "Understand": 0.40,
    "Apply": 0.55,
    "Analyze": 0.70,
}

# Requires context reason categories
# Analyzed from question content and chunk relationship
REASON_CATEGORIES: dict[str, str] = {
    "calculation": "requires_calculation",
    "diagram": "requires_position_diagram",
    "image": "image_dependent",
    "table": "requires_calculation_table",
    "position": "requires_position_diagram",
    "elo": "requires_calculation",
    "score": "requires_calculation",
    "classement": "requires_calculation",
    "buchholz": "requires_calculation",
    "departage": "requires_calculation",
    "points": "requires_calculation",
    "resultat": "requires_calculation",
    "nombre": "requires_calculation",
    "combien": "requires_calculation",
    "categorie": "requires_external_data",
    "age": "requires_external_data",
    "licence": "requires_external_data",
    "saison": "requires_external_data",
}


def save_json(data: Any, path: Path) -> None:
    """Save JSON with consistent formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n")


def infer_requires_context_reason(q: dict[str, Any]) -> str:
    """Infer why a question requires context beyond the chunk."""
    question = q.get("question", "").lower()
    answer = q.get("expected_answer", "").lower()
    md = q.get("metadata", {})

    # Check for calculation-related keywords
    calc_keywords = [
        "calcul",
        "elo",
        "buchholz",
        "departage",
        "départage",
        "score",
        "points",
        "resultat",
        "résultat",
        "combien",
        "nombre",
        "classement",
        "somme",
        "total",
        "coefficient",
    ]
    for kw in calc_keywords:
        if kw in question or kw in answer:
            return "requires_calculation"

    # Check for position/diagram
    if any(w in question for w in ["position", "diagramme", "échiquier", "pièce"]):
        return "requires_position_diagram"

    # Check for age/category
    if any(w in question for w in ["catégorie", "categorie", "âge", "age", "né le"]):
        return "requires_external_data"

    # Check for time/schedule
    if any(
        w in question
        for w in ["cadence", "temps", "minutes", "heures", "retard", "délai"]
    ):
        return "requires_specific_context"

    # Check for specific competition rules
    if any(
        w in question
        for w in [
            "entente",
            "équipe",
            "phase",
            "qualification",
            "division",
            "poule",
        ]
    ):
        return "requires_competition_context"

    # Check for regulation cross-references
    if (
        md.get("article_reference")
        and "chapitre" in md.get("article_reference", "").lower()
    ):
        return "requires_cross_reference"

    # Default: the chunk doesn't contain enough context
    return "chunk_insufficient_context"


def main() -> None:  # noqa: C901
    """Run Phase 3 metadata completion."""
    parser = argparse.ArgumentParser(description="Phase 3: Metadata completion")
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

    stats = {
        "difficulty_added": 0,
        "reason_added": 0,
    }

    # --- 3a: M-01/M-02 difficulty for questions without it ---
    for q in gs["questions"]:
        md = q.get("metadata", {})
        if md.get("difficulty") is None:
            cog = md.get("cognitive_level", "Remember")
            difficulty = DIFFICULTY_MAP.get(cog, 0.30)
            md["difficulty"] = difficulty
            stats["difficulty_added"] += 1
            if args.verbose:
                print(f"  difficulty: {q['id']} -> {difficulty} (from {cog})")
        q["metadata"] = md

    # --- 3b: CB-09 requires_context_reason ---
    for q in gs["questions"]:
        md = q.get("metadata", {})
        if md.get("requires_context") and not md.get("requires_context_reason"):
            reason = infer_requires_context_reason(q)
            md["requires_context_reason"] = reason
            stats["reason_added"] += 1
            if args.verbose:
                print(f"  reason: {q['id']} -> {reason} (Q: {q['question'][:50]})")
        q["metadata"] = md

    # --- 3c: Version bump ---
    old_version = gs.get("version")
    gs["version"] = "8.0"

    # --- Report ---
    print("=" * 60)
    print("Phase 3: Metadata completion")
    print("=" * 60)
    print(f"Difficulty added:         {stats['difficulty_added']}")
    print(f"Context reasons added:    {stats['reason_added']}")
    print(f"Version:                  {old_version} -> {gs['version']}")
    print()

    # Verify all criteria
    total = len(gs["questions"])
    m01 = sum(
        1
        for q in gs["questions"]
        if q.get("metadata", {}).get("difficulty") is not None
    )
    m02 = sum(
        1
        for q in gs["questions"]
        if q.get("metadata", {}).get("difficulty") is not None
        and 0 <= q["metadata"]["difficulty"] <= 1
    )
    m03 = sum(
        1 for q in gs["questions"] if q.get("metadata", {}).get("cognitive_level")
    )
    m04 = sum(1 for q in gs["questions"] if q.get("category"))
    req_ctx = [
        q for q in gs["questions"] if q.get("metadata", {}).get("requires_context")
    ]
    cb09 = sum(
        1 for q in req_ctx if q.get("metadata", {}).get("requires_context_reason")
    )
    cb01 = sum(
        1
        for q in gs["questions"]
        if q.get("expected_answer", "")
        in chunks.get(q.get("expected_chunk_id", ""), {}).get("text", "")
    )
    cb04 = sum(
        1
        for q in gs["questions"]
        if q.get("metadata", {}).get("chunk_match_method") == "manual_by_design"
    )
    f01 = sum(1 for q in gs["questions"] if q.get("question", "").strip().endswith("?"))
    f04 = sum(1 for q in gs["questions"] if len(q.get("expected_answer", "")) > 5)
    cb02 = sum(1 for q in gs["questions"] if q.get("expected_chunk_id", "") in chunks)
    cb03 = sum(1 for q in gs["questions"] if q.get("expected_chunk_id"))
    cb07 = sum(1 for q in gs["questions"] if q.get("expected_docs"))
    cb08 = sum(1 for q in gs["questions"] if q.get("expected_pages"))

    print("=== GATE P3 FINAL ===")
    print(
        f"M-01 (difficulty):        {m01}/{total}  {'PASS' if m01 == total else 'FAIL'}"
    )
    print(
        f"M-02 (difficulty [0,1]):  {m02}/{total}  {'PASS' if m02 == total else 'FAIL'}"
    )
    print(
        f"M-03 (cognitive_level):   {m03}/{total}  {'PASS' if m03 == total else 'FAIL'}"
    )
    print(
        f"M-04 (category):          {m04}/{total}  {'PASS' if m04 == total else 'FAIL'}"
    )
    print(
        f"CB-09 (req_ctx_reason):   {cb09}/{len(req_ctx)}  {'PASS' if cb09 == len(req_ctx) else 'FAIL'}"
    )
    print(
        f"Version:                  {gs['version']}  {'PASS' if gs['version'] == '8.0' else 'FAIL'}"
    )
    print()
    print("=== REGRESSION P2 ===")
    print(
        f"CB-01 (answer in chunk):  {cb01}/{total}  {'PASS' if cb01 == total else 'FAIL'}"
    )
    print(
        f"CB-04 (by_design):        {cb04}/{total}  {'PASS' if cb04 == total else 'FAIL'}"
    )
    print(
        f"F-01 (ends with ?):       {f01}/{total}  {'PASS' if f01 == total else 'FAIL'}"
    )
    print(
        f"F-04 (answer > 5):        {f04}/{total}  {'PASS' if f04 == total else 'FAIL'}"
    )
    print()
    print("=== REGRESSION P1 ===")
    print(f"Total questions:          {total}  {'PASS' if total == 420 else 'FAIL'}")
    print(
        f"CB-02 (chunk exists):     {cb02}/{total}  {'PASS' if cb02 == total else 'FAIL'}"
    )
    print(
        f"CB-03 (chunk non-null):   {cb03}/{total}  {'PASS' if cb03 == total else 'FAIL'}"
    )
    print(
        f"CB-07 (expected_docs):    {cb07}/{total}  {'PASS' if cb07 == total else 'FAIL'}"
    )
    print(f"CB-08 (expected_pages):   {cb08}/{total}")

    all_pass = (
        m01 == total
        and m02 == total
        and m03 == total
        and m04 == total
        and cb09 == len(req_ctx)
        and gs["version"] == "8.0"
        and cb01 == total
        and cb04 == total
        and f01 == total
        and f04 == total
        and total == 420
        and cb02 == total
        and cb03 == total
        and cb07 == total
    )
    print(f"\nGATE P3 FINAL: {'PASS' if all_pass else 'FAIL'}")

    if not args.check:
        save_json(gs, GS_PATH)
        print(f"Saved to {GS_PATH}")
    else:
        print("Dry-run mode — no changes saved")


if __name__ == "__main__":
    main()
