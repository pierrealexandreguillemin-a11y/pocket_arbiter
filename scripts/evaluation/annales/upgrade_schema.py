#!/usr/bin/env python3
# pragma: no cover
"""
[ARCHIVED] Upgrade Gold Standard v6 schema to industry standard (v5.30 compatible).

STATUS: ARCHIVED - One-shot migration script, no longer needed.
        Gold Standard is now at v6.7.0.
        Kept for reference only.

COVERAGE EXCLUSION RATIONALE (ISO 29119-4):
    This module is excluded from test coverage because:
    1. One-time migration script - not part of runtime application
    2. Already successfully executed - data migration complete
    3. No future execution expected - archived for audit trail only
    4. Testing would require recreating obsolete data formats

Original purpose:
    Adds missing fields required by GOLD_STANDARD_SPECIFICATION.md:
    - audit: date d'ajout
    - expected_chunk_id: provenance exacte (to be filled later)
    - metadata: hard_case, hard_type, corpus_truth, test_purpose

ISO 42001 A.6.2.2: Complete traceability required.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def map_to_hard_type(question: dict[str, Any]) -> str:
    """
    Map GS v6 fields to UAEval4RAG hard_type categories.

    Categories from arXiv:2412.12300:
    - ANSWERABLE: Standard answerable question
    - PARTIAL_INFO: Information incomplete in corpus
    - VOCABULARY_MISMATCH: Different terms (synonyms)
    - MULTI_HOP_IMPOSSIBLE: Requires unavailable inference
    - FALSE_PREMISE: Based on false premise
    - OUT_OF_SCOPE: Outside corpus scope
    """
    # Default: all annales questions are answerable (from official exams)
    hard_type = "ANSWERABLE"

    # Check if corpus verification failed
    corpus_verified = question.get("corpus_verified")
    if corpus_verified is False:
        hard_type = "PARTIAL_INFO"  # Article not in current corpus

    # Check reasoning type for multi-hop
    reasoning = question.get("reasoning_type", "")
    if reasoning == "multi-hop":
        # Could be MULTI_HOP_IMPOSSIBLE if too complex
        # For now, assume answerable multi-hop
        pass

    return hard_type


def upgrade_question_schema(question: dict[str, Any]) -> dict[str, Any]:
    """
    Upgrade a single question to industry standard schema.
    """
    # Add audit field
    session = question.get("annales_source", {}).get("session", "unknown")
    audit_date = datetime.now().strftime("%Y-%m-%d")
    question["audit"] = f"annales_{session}_imported_{audit_date}"

    # Add expected_chunk_id placeholder (to be filled by chunk mapping)
    if "expected_chunk_id" not in question:
        docs = question.get("expected_docs", [])
        pages = question.get("expected_pages", [])
        if docs and pages:
            # Create placeholder ID
            doc_base = docs[0].replace(".pdf", "").replace("_", "-")
            page_str = f"p{pages[0]}" if pages else "p0"
            question["expected_chunk_id"] = f"{doc_base}-{page_str}-pending"
        else:
            question["expected_chunk_id"] = "unmapped"

    # Add metadata object
    hard_type = map_to_hard_type(question)
    is_hard = hard_type != "ANSWERABLE"

    # Get existing fields to populate metadata
    article_ref = question.get("article_reference", "")
    answer_text = question.get("answer_text", "")
    q_type = question.get("question_type", "factual")

    question["metadata"] = {
        "type": q_type,
        "chapter": article_ref.split(".")[0] if "." in article_ref else "",
        "hard_case": is_hard,
        "hard_type": hard_type,
        "hard_reason": ""
        if not is_hard
        else f"corpus_verified={question.get('corpus_verified')}",
        "corpus_truth": answer_text[:200] if answer_text else "",
        "test_purpose": f"Evaluate retrieval for {q_type} question on article {article_ref}",
    }

    # Ensure validation object has required fields
    validation = question.get("validation", {})
    if "recall_actual" not in validation:
        validation["recall_actual"] = "pending"
    if "audit_note" not in validation:
        validation["audit_note"] = f"Imported from annales {session}"
    question["validation"] = validation

    return question


def upgrade_gold_standard(gs_data: dict[str, Any]) -> dict[str, Any]:
    """
    Upgrade entire Gold Standard to industry standard schema.
    """
    questions = gs_data.get("questions", [])

    stats = {
        "total": len(questions),
        "hard_cases": 0,
        "with_chunk_id": 0,
    }

    for q in questions:
        upgrade_question_schema(q)

        if q.get("metadata", {}).get("hard_case"):
            stats["hard_cases"] += 1
        if q.get("expected_chunk_id") and "pending" not in q["expected_chunk_id"]:
            stats["with_chunk_id"] += 1

    # Update version
    version = gs_data.get("version", {})
    version["number"] = "6.2.0"
    version["date"] = datetime.now().strftime("%Y-%m-%d")
    version["schema_version"] = "v5.30-compatible"
    version["changes"] = version.get("changes", []) + [
        "Added audit field for traceability",
        "Added expected_chunk_id placeholder",
        "Added metadata object (hard_type, corpus_truth, test_purpose)",
        "Schema now compatible with GOLD_STANDARD_SPECIFICATION.md",
    ]
    gs_data["version"] = version

    gs_data["schema_upgrade_stats"] = stats

    return gs_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upgrade GS v6 schema to industry standard"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tests/data/gold_standard_annales_fr.json"),
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

    # Upgrade
    gs_data = upgrade_gold_standard(gs_data)

    # Report
    stats = gs_data["schema_upgrade_stats"]
    print("=== Schema Upgrade Report ===")
    print(f"Total questions: {stats['total']}")
    print(f"Hard cases: {stats['hard_cases']}")
    print(f"With chunk_id: {stats['with_chunk_id']}")

    if args.dry_run:
        print("\n[DRY RUN] No changes written")

        # Show sample
        q = gs_data["questions"][0]
        print("\n=== Sample Upgraded Question ===")
        print(f"audit: {q.get('audit')}")
        print(f"expected_chunk_id: {q.get('expected_chunk_id')}")
        print(f"metadata: {json.dumps(q.get('metadata', {}), indent=2)}")
        return

    # Write
    output_path = args.output or args.input
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gs_data, f, ensure_ascii=False, indent=2)

    print(f"\nWritten to: {output_path}")


if __name__ == "__main__":
    main()
