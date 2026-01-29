#!/usr/bin/env python3
"""Migrate Gold Standard to unified schema (industry standards order).

This script migrates the gold standard questions to the unified schema
following industry standards (SQuAD 2.0, BEIR, Schema.org, HuggingFace).

Field order:
1. id                 - Unique identifier
2. question           - Query text
3. expected_answer    - Expected answer text (NEW)
4. is_impossible      - Unanswerable flag (NEW)
5. expected_chunk_id  - BEIR qrel
6. expected_docs      - Source documents
7. expected_pages     - Source pages
8. category           - Domain/theme
9. keywords           - Retrieval terms
10. validation        - Validation status
11. audit             - Audit trail
12. metadata          - Extensible metadata (always last)

Usage:
    python scripts/training/migrate_gs_schema.py
    python scripts/training/migrate_gs_schema.py --dry-run
    python scripts/training/migrate_gs_schema.py --input path/to/gs.json

ISO Compliance:
    - ISO 42001: Adds validation.reviewer for traceability
    - ISO 29119: Generates migration report
    - ISO 25010: Normalizes data quality
"""

import argparse
import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any


# Schema field order (industry standards)
FIELD_ORDER = [
    "id",
    "question",
    "expected_answer",
    "is_impossible",
    "expected_chunk_id",
    "expected_docs",
    "expected_pages",
    "category",
    "keywords",
    "validation",
    "audit",
    "metadata",
]

# Metadata field order
METADATA_ORDER = [
    # Core evaluation
    "answer_type",
    "reasoning_type",
    "cognitive_level",
    # Source reference
    "article_num",
    "article_reference",
    "section",
    # Annales-specific (optional)
    "difficulty",
    "annales_source",
    "question_type",
    # MCQ-specific (optional)
    "choices",
    "mcq_answer",
    "answer_explanation",
    # Quality metrics
    "quality_score",
    "chunk_match_score",
    "chunk_match_method",
    # Legacy fields
    "source",
    "hard_type",
    "hard_reason",
    "hard_case",
    "test_purpose",
    "corpus_truth",
    "expected_behavior",
]

# Reasoning type normalization (uppercase -> lowercase)
REASONING_TYPE_MAP = {
    "SINGLE_SENTENCE": "single-hop",
    "MULTI_SENTENCE": "multi-hop",
    "TEMPORAL": "temporal",
    "single-hop": "single-hop",
    "multi-hop": "multi-hop",
    "temporal": "temporal",
}


def normalize_reasoning_type(value: str | None) -> str:
    """Normalize reasoning_type to lowercase standard."""
    if not value:
        return "single-hop"
    return REASONING_TYPE_MAP.get(value, value.lower().replace("_", "-"))


def migrate_validation(validation: dict[str, Any] | None) -> dict[str, Any]:
    """Migrate validation object, adding reviewer field."""
    if not validation:
        return {
            "status": "PENDING",
            "method": "migration_default",
            "reviewer": "auto",
        }

    result = OrderedDict()
    result["status"] = validation.get("status", "PENDING")
    result["method"] = validation.get("method", "unknown")
    result["reviewer"] = validation.get("reviewer", "human")

    # Preserve other fields
    for k, v in validation.items():
        if k not in result:
            result[k] = v

    return dict(result)


def migrate_metadata(
    metadata: dict[str, Any] | None,
    question_data: dict[str, Any],
) -> dict[str, Any]:
    """Migrate metadata object to standard order.

    Pulls fields from question level into metadata if they belong there.
    """
    if not metadata:
        metadata = {}

    result = OrderedDict()

    # Fields that may be at question level but belong in metadata
    _question_level_fields = [
        "difficulty",
        "annales_source",
        "question_type",
        "choices",
        "mcq_answer",
        "answer_explanation",
        "quality_score",
        "chunk_match_score",
        "chunk_match_method",
        "article_reference",
        "answer_type",
        "reasoning_type",
        "cognitive_level",
    ]

    # Add fields in standard order
    for field in METADATA_ORDER:
        value = None
        # Check metadata first
        if field in metadata:
            value = metadata[field]
        # Then check question level
        elif field in question_data:
            value = question_data[field]

        if value is not None:
            # Normalize reasoning_type
            if field == "reasoning_type":
                value = normalize_reasoning_type(value)
            result[field] = value

    # Preserve any other metadata fields not in standard order
    for k, v in metadata.items():
        if k not in result and k not in ["audit_note"]:  # Skip deprecated fields
            result[k] = v

    return dict(result)


def migrate_question(
    question: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    """Migrate a single question to the unified schema."""
    result = OrderedDict()

    # 1. id (required)
    result["id"] = question.get("id", f"MISSING-{report['missing_ids']}")
    if "id" not in question:
        report["missing_ids"] += 1
        report["warnings"].append(f"Missing id, assigned: {result['id']}")

    # 2. question (required)
    result["question"] = question.get("question", "")
    if not result["question"]:
        report["errors"].append(f"{result['id']}: Missing question text")

    # 3. expected_answer (map from answer_text if exists)
    result["expected_answer"] = question.get("expected_answer") or question.get(
        "answer_text", ""
    )
    if not result["expected_answer"]:
        report["fields_added"]["expected_answer"] += 1
    elif "answer_text" in question and "expected_answer" not in question:
        report["fields_mapped"]["answer_text->expected_answer"] += 1

    # 4. is_impossible (NEW - required, default False)
    result["is_impossible"] = question.get("is_impossible", False)
    if "is_impossible" not in question:
        report["fields_added"]["is_impossible"] += 1

    # 5. expected_chunk_id (moved up)
    result["expected_chunk_id"] = question.get("expected_chunk_id", "")
    if not result["expected_chunk_id"]:
        report["warnings"].append(f"{result['id']}: Missing expected_chunk_id")

    # 6. expected_docs
    result["expected_docs"] = question.get("expected_docs", [])

    # 7. expected_pages
    result["expected_pages"] = question.get("expected_pages", [])

    # 8. category
    result["category"] = question.get("category", "unknown")

    # 9. keywords
    result["keywords"] = question.get("keywords", [])

    # 10. validation (with reviewer)
    result["validation"] = migrate_validation(question.get("validation"))
    if "reviewer" not in question.get("validation", {}):
        report["fields_added"]["validation.reviewer"] += 1

    # 11. audit
    result["audit"] = question.get("audit", "")

    # 12. metadata (always last)
    result["metadata"] = migrate_metadata(
        question.get("metadata", {}),
        question,
    )

    # Track reasoning_type normalization
    old_rt = question.get("metadata", {}).get("reasoning_type", "")
    new_rt = result["metadata"].get("reasoning_type", "")
    if old_rt and old_rt != new_rt:
        report["normalized"]["reasoning_type"] += 1

    return dict(result)


def migrate_gold_standard(
    data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Migrate entire gold standard file to unified schema."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "input_version": data.get("version", "unknown"),
        "output_version": "7.0.0",
        "total_questions": 0,
        "migrated": 0,
        "errors": [],
        "warnings": [],
        "missing_ids": 0,
        "fields_added": {
            "expected_answer": 0,
            "is_impossible": 0,
            "validation.reviewer": 0,
        },
        "fields_mapped": {
            "answer_text->expected_answer": 0,
        },
        "normalized": {
            "reasoning_type": 0,
        },
        "field_order_changes": True,
    }

    questions = data.get("questions", [])
    report["total_questions"] = len(questions)

    migrated_questions = []
    for q in questions:
        try:
            migrated = migrate_question(q, report)
            migrated_questions.append(migrated)
            report["migrated"] += 1
        except Exception as e:
            report["errors"].append(f"{q.get('id', 'UNKNOWN')}: {e}")

    # Build output with updated header
    output = OrderedDict()
    output["version"] = "7.0.0"
    output["schema"] = "unified-v1"
    output["description"] = (
        "Gold standard v7.0.0 - unified schema (SQuAD 2.0 + BEIR + ISO 42001)"
    )
    output["methodology"] = data.get("methodology", {})
    output["coverage"] = data.get("coverage", {})
    output["coverage"]["total_questions"] = len(migrated_questions)
    output["questions"] = migrated_questions

    return dict(output), report


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate Gold Standard to unified schema"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("tests/data/gold_standard_fr.json"),
        help="Input gold standard file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file (default: input with _v7 suffix)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show report without writing files",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file (creates backup)",
    )

    args = parser.parse_args()

    # Resolve paths
    input_path = args.input
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    # Determine output path
    if args.output:
        output_path = args.output
    elif args.in_place:
        output_path = input_path
        backup_path = input_path.with_suffix(".json.bak")
    else:
        output_path = input_path.with_name(input_path.stem + "_v7" + input_path.suffix)

    # Load input
    print(f"Loading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Migrate
    print("Migrating to unified schema...")
    migrated, report = migrate_gold_standard(data)

    # Print report
    print("\n" + "=" * 60)
    print("MIGRATION REPORT")
    print("=" * 60)
    print(f"Input version:  {report['input_version']}")
    print(f"Output version: {report['output_version']}")
    print(f"Total questions: {report['total_questions']}")
    print(f"Migrated: {report['migrated']}")
    print(f"Errors: {len(report['errors'])}")
    print(f"Warnings: {len(report['warnings'])}")
    print()
    print("Fields added:")
    for field, count in report["fields_added"].items():
        print(f"  {field}: {count}")
    print()
    print("Fields mapped:")
    for field, count in report["fields_mapped"].items():
        print(f"  {field}: {count}")
    print()
    print("Normalized:")
    for field, count in report["normalized"].items():
        print(f"  {field}: {count}")

    if report["errors"]:
        print("\nErrors:")
        for err in report["errors"][:10]:
            print(f"  - {err}")
        if len(report["errors"]) > 10:
            print(f"  ... and {len(report['errors']) - 10} more")

    if report["warnings"][:5]:
        print("\nWarnings (first 5):")
        for warn in report["warnings"][:5]:
            print(f"  - {warn}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        # Show sample migrated question
        if migrated["questions"]:
            print("\nSample migrated question:")
            print(json.dumps(migrated["questions"][0], indent=2, ensure_ascii=False))
        return 0

    # Write output
    if args.in_place:
        print(f"\nCreating backup: {backup_path}")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Writing: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(migrated, f, indent=2, ensure_ascii=False)

    # Write report
    report_path = output_path.with_name(output_path.stem + "_migration_report.json")
    print(f"Writing report: {report_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nMigration complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
