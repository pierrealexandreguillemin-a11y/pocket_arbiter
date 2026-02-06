"""
Merge adversarial questions into Gold Standard with Schema v2.0 conversion.

Converts adversarial questions from legacy schema to GS Schema v2.0
(8 groups: content, mcq, provenance, classification, validation, processing, audit)

ISO Reference: ISO 42001 A.6.2.2 - Provenance tracking
"""

import hashlib
import json
from pathlib import Path


def convert_to_schema_v2(adv_q: dict) -> dict:
    """Convert adversarial question to Schema v2.0 format."""
    # Generate ID if needed
    if not adv_q.get("id"):
        q_hash = hashlib.sha256(adv_q["question"].encode()).hexdigest()[:8]
        adv_q["id"] = f"adversarial:human:rules:001:{q_hash}"

    metadata = adv_q.get("metadata", {})
    hard_type = metadata.get("hard_type", "OUT_OF_SCOPE")

    return {
        "id": adv_q["id"],
        "legacy_id": adv_q.get("legacy_id", ""),
        # Content group
        # All adversarial questions are unanswerable by definition
        "content": {
            "question": adv_q["question"],
            "expected_answer": adv_q.get("expected_answer", ""),
            "is_impossible": True,  # Forced for adversarial
        },
        # MCQ group (empty for adversarial)
        "mcq": {
            "original_question": adv_q["question"],
            "choices": {},
            "mcq_answer": "",
            "correct_answer": "",
            "original_answer": "",
        },
        # Provenance group
        "provenance": {
            "chunk_id": adv_q.get("expected_chunk_id", "") or None,
            "docs": adv_q.get("expected_docs", []),
            "pages": adv_q.get("expected_pages", []),
            "article_reference": "",
            "answer_explanation": metadata.get("corpus_truth", ""),
            "annales_source": None,
        },
        # Classification group
        "classification": {
            "category": adv_q.get("category", "adversarial"),
            "keywords": adv_q.get("keywords", []),
            "difficulty": 1.0,  # Adversarial = hard
            "question_type": "adversarial",
            "cognitive_level": metadata.get("cognitive_level", "ANALYZE"),
            "reasoning_type": metadata.get("reasoning_type", "adversarial"),
            "reasoning_class": "adversarial",
            "answer_type": metadata.get("answer_type", "unanswerable"),
            "hard_type": hard_type,  # UAEval4RAG category
        },
        # Validation group
        "validation": {
            "status": adv_q.get("validation", {}).get("status", "VALIDATED"),
            "method": adv_q.get("validation", {}).get("method", "adversarial_manual"),
            "reviewer": adv_q.get("validation", {}).get("reviewer", "human"),
            "answer_current": True,
            "verified_date": "2026-02-05",
            "pages_verified": True,
            "batch": "adversarial_merge",
        },
        # Processing group
        "processing": {
            "chunk_match_score": 0,
            "chunk_match_method": "not_applicable",
            "reasoning_class_method": "manual",
            "triplet_ready": False,  # Adversarial not for triplets
            "extraction_flags": ["adversarial"],
            "answer_source": "human_crafted",
            "quality_score": 1.0,
        },
        # Audit group
        "audit": {
            "history": "[ADVERSARIAL] Merged from adversarial_questions.json 2026-02-05",
            "qat_revalidation": None,
            "requires_inference": False,
        },
    }


def merge_adversarial_to_gs(gs_path: Path, adv_path: Path) -> dict:
    """Merge adversarial questions into Gold Standard."""
    # Load data
    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)

    with open(adv_path, encoding="utf-8") as f:
        adv = json.load(f)

    # Convert adversarial to Schema v2
    adv_questions = [convert_to_schema_v2(q) for q in adv["questions"]]

    # Check for duplicates
    gs_ids = {q["id"] for q in gs["questions"]}
    new_questions = [q for q in adv_questions if q["id"] not in gs_ids]

    print(f"GS questions: {len(gs['questions'])}")
    print(f"Adversarial questions: {len(adv_questions)}")
    print(f"New (non-duplicate): {len(new_questions)}")

    # Merge
    gs["questions"].extend(new_questions)

    # Update version
    total = len(gs["questions"])
    unanswerable = sum(
        1 for q in gs["questions"] if q.get("content", {}).get("is_impossible")
    )
    gs["version"] = {
        "number": "8.1.0",
        "date": "2026-02-05",
        "schema": "GS_SCHEMA_V2",
        "adversarial_merged": True,
        "unanswerable_ratio": f"{unanswerable}/{total} ({100*unanswerable/total:.1f}%)",
        "audit_note": "Merged 420 annales + adversarial questions",
    }

    # Update description
    gs["description"] = (
        f"Gold Standard v8.1 - 420 annales + {len(new_questions)} adversarial - Schema v2 (46 fields, 8 groups)"
    )

    # Stats
    from collections import Counter

    hard_types = Counter(
        q.get("classification", {}).get("hard_type", "ANSWERABLE")
        for q in gs["questions"]
    )

    print("\n=== MERGED GS STATS ===")
    print(f"Total: {total}")
    print(f"Answerable: {total - unanswerable}")
    print(f"Unanswerable: {unanswerable} ({100*unanswerable/total:.1f}%)")
    print("\nBy hard_type:")
    for ht, count in hard_types.most_common():
        print(f"  {ht}: {count}")

    return gs


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge adversarial questions into Gold Standard"
    )
    parser.add_argument(
        "--gs",
        "-g",
        type=Path,
        default=Path("tests/data/gold_standard_annales_fr_v7.json"),
        help="Input Gold Standard JSON",
    )
    parser.add_argument(
        "--adversarial",
        "-a",
        type=Path,
        default=Path("tests/data/adversarial_questions.json"),
        help="Adversarial questions JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("tests/data/gold_standard_annales_fr_v8_adversarial.json"),
        help="Output merged GS JSON",
    )

    args = parser.parse_args()

    print(f"Merging: {args.gs} + {args.adversarial}")
    merged = merge_adversarial_to_gs(args.gs, args.adversarial)

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
