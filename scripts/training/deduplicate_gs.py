#!/usr/bin/env python3
"""Deduplicate Gold Standard questions.

Removes exact duplicate questions, keeping the one with:
1. Highest reasoning_type complexity (multi-hop > temporal > single-hop)
2. First ID alphabetically (for determinism)

Usage:
    python scripts/training/deduplicate_gs.py
    python scripts/training/deduplicate_gs.py --dry-run
    python scripts/training/deduplicate_gs.py --input path/to/gs.json
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Reasoning type priority (higher = keep)
RT_PRIORITY = {
    "multi-hop": 3,
    "temporal": 2,
    "single-hop": 1,
    "": 0,
}


def get_reasoning_type(question: dict[str, Any]) -> str:
    """Extract reasoning_type from question."""
    return question.get("metadata", {}).get("reasoning_type", "")


def deduplicate_questions(
    questions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deduplicate questions by exact text match.

    Returns:
        Tuple of (deduplicated questions, report dict)
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "input_count": len(questions),
        "duplicate_groups": 0,
        "removed_count": 0,
        "output_count": 0,
        "removed_ids": [],
        "kept_ids": [],
        "duplicates_detail": [],
    }

    # Group by normalized question text
    by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        normalized = q["question"].strip().lower()
        by_text[normalized].append(q)

    # Process each group
    deduplicated = []
    for text, group in by_text.items():
        if len(group) == 1:
            # No duplicate
            deduplicated.append(group[0])
        else:
            # Sort by reasoning_type priority (desc), then by id (asc)
            sorted_group = sorted(
                group,
                key=lambda q: (
                    -RT_PRIORITY.get(get_reasoning_type(q), 0),
                    q["id"],
                ),
            )

            # Keep first, remove rest
            keep = sorted_group[0]
            remove = sorted_group[1:]

            deduplicated.append(keep)
            report["duplicate_groups"] += 1
            report["removed_count"] += len(remove)
            report["kept_ids"].append(keep["id"])
            report["removed_ids"].extend([q["id"] for q in remove])
            report["duplicates_detail"].append(
                {
                    "question": text[:100],
                    "keep": {
                        "id": keep["id"],
                        "reasoning_type": get_reasoning_type(keep),
                    },
                    "remove": [
                        {
                            "id": q["id"],
                            "reasoning_type": get_reasoning_type(q),
                        }
                        for q in remove
                    ],
                }
            )

    report["output_count"] = len(deduplicated)

    # Sort by ID for consistent output
    deduplicated.sort(key=lambda q: q["id"])

    return deduplicated, report


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deduplicate Gold Standard questions")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("tests/data/gold_standard_annales_fr_v7.json"),
        help="Input gold standard file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file (default: input with _dedup suffix)",
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
        output_path = input_path.with_name(
            input_path.stem + "_dedup" + input_path.suffix
        )

    # Load input
    print(f"Loading: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])

    # Deduplicate
    print("Deduplicating...")
    deduplicated, report = deduplicate_questions(questions)

    # Print report
    print("\n" + "=" * 60)
    print("DEDUPLICATION REPORT")
    print("=" * 60)
    print(f"Input questions:    {report['input_count']}")
    print(f"Duplicate groups:   {report['duplicate_groups']}")
    print(f"Removed:            {report['removed_count']}")
    print(f"Output questions:   {report['output_count']}")
    print()

    # Stats by reasoning_type
    kept_rt = defaultdict(int)
    removed_rt = defaultdict(int)
    for d in report["duplicates_detail"]:
        kept_rt[d["keep"]["reasoning_type"]] += 1
        for r in d["remove"]:
            removed_rt[r["reasoning_type"]] += 1

    print("Kept by reasoning_type:")
    for rt, count in sorted(kept_rt.items(), key=lambda x: -RT_PRIORITY.get(x[0], 0)):
        print(f"  {rt or 'unknown'}: {count}")
    print()
    print("Removed by reasoning_type:")
    for rt, count in sorted(
        removed_rt.items(), key=lambda x: -RT_PRIORITY.get(x[0], 0)
    ):
        print(f"  {rt or 'unknown'}: {count}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        print("\nSample removed IDs (first 10):")
        for qid in report["removed_ids"][:10]:
            print(f"  - {qid}")
        return 0

    # Build output
    output = data.copy()
    output["questions"] = deduplicated
    if "coverage" in output:
        output["coverage"]["total_questions"] = len(deduplicated)

    # Update version
    old_version = output.get("version", "unknown")
    if "_dedup" not in str(old_version):
        output["version"] = f"{old_version}_dedup"
    output["deduplication"] = {
        "date": report["timestamp"],
        "removed_count": report["removed_count"],
        "method": "exact_match_keep_highest_reasoning_type",
    }

    # Write output
    if args.in_place:
        print(f"\nCreating backup: {backup_path}")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Writing: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Write report
    report_path = output_path.with_name(output_path.stem + "_dedup_report.json")
    print(f"Writing report: {report_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nDeduplication complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
