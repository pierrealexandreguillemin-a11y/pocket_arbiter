#!/usr/bin/env python3
"""Merge adversarial questions into gold standard files.

Usage:
    python scripts/training/merge_adversarial.py [--dry-run]
"""

import json
import argparse
from pathlib import Path


def load_json(path: Path) -> dict:
    """Load JSON file with UTF-8 encoding."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    """Save JSON file with UTF-8 encoding and formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def merge_adversarial(dry_run: bool = False) -> None:
    """Merge adversarial questions into gold standard files."""

    # Paths
    gs_fr_path = Path("tests/data/gold_standard_fr.json")
    gs_intl_path = Path("tests/data/gold_standard_intl.json")
    adv_path = Path("tests/data/adversarial_questions.json")

    # Load files
    print("Loading files...")
    gs_fr = load_json(gs_fr_path)
    gs_intl = load_json(gs_intl_path)
    adv = load_json(adv_path)

    # Stats before
    before_fr = len(gs_fr["questions"])
    before_intl = len(gs_intl["questions"])

    # Get existing IDs to avoid duplicates
    existing_fr_ids = {q["id"] for q in gs_fr["questions"]}
    existing_intl_ids = {q["id"] for q in gs_intl["questions"]}

    # Filter new questions
    new_fr = [q for q in adv["questions_fr"] if q["id"] not in existing_fr_ids]
    new_intl = [q for q in adv["questions_intl"] if q["id"] not in existing_intl_ids]

    print("\nBefore merge:")
    print(f"  FR: {before_fr} questions")
    print(f"  INTL: {before_intl} questions")
    print(f"  Total: {before_fr + before_intl}")

    print("\nNew adversarial questions:")
    print(
        f"  FR: {len(new_fr)} (skipped {len(adv['questions_fr']) - len(new_fr)} duplicates)"
    )
    print(
        f"  INTL: {len(new_intl)} (skipped {len(adv['questions_intl']) - len(new_intl)} duplicates)"
    )

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        return

    # Merge FR
    gs_fr["questions"].extend(new_fr)
    gs_fr["coverage"]["total_questions"] = len(gs_fr["questions"])
    gs_fr["statistics"]["total_questions"] = len(gs_fr["questions"])

    # Update version
    old_version = gs_fr.get("version", "5.26")
    new_version = f"{float(old_version) + 0.01:.2f}"
    gs_fr["version"] = new_version
    gs_fr["description"] = (
        f"Gold standard v{new_version} - {len(gs_fr['questions'])} questions "
        f"(+{len(new_fr)} adversarial UAEval4RAG/SQuAD2-CR)"
    )

    # Count adversarial
    adv_count = len(
        [
            q
            for q in gs_fr["questions"]
            if q.get("metadata", {}).get("hard_type") and not q.get("expected_pages")
        ]
    )
    gs_fr["statistics"]["adversarial_questions"] = adv_count
    gs_fr["statistics"]["adversarial_ratio"] = (
        f"{adv_count/len(gs_fr['questions'])*100:.1f}%"
    )

    # Merge INTL
    gs_intl["questions"].extend(new_intl)
    old_version_intl = gs_intl.get("version", "2.0")
    new_version_intl = f"{float(old_version_intl) + 0.1:.1f}"
    gs_intl["version"] = new_version_intl

    # Save
    print("\nSaving files...")
    save_json(gs_fr_path, gs_fr)
    save_json(gs_intl_path, gs_intl)

    # Stats after
    after_fr = len(gs_fr["questions"])
    after_intl = len(gs_intl["questions"])
    total = after_fr + after_intl

    # Count all unanswerable
    unanswerable_fr = len(
        [
            q
            for q in gs_fr["questions"]
            if q.get("metadata", {}).get("hard_type") and not q.get("expected_pages")
        ]
    )
    unanswerable_intl = len(
        [
            q
            for q in gs_intl["questions"]
            if q.get("metadata", {}).get("hard_type") and not q.get("expected_pages")
        ]
    )
    unanswerable_total = unanswerable_fr + unanswerable_intl

    print("\nAfter merge:")
    print(f"  FR: {after_fr} questions (+{after_fr - before_fr})")
    print(f"  INTL: {after_intl} questions (+{after_intl - before_intl})")
    print(f"  Total: {total}")
    print("\nAdversarial ratio:")
    print(
        f"  Unanswerable: {unanswerable_total}/{total} ({unanswerable_total/total*100:.1f}%)"
    )
    print(
        f"  Answerable: {total - unanswerable_total}/{total} ({(total-unanswerable_total)/total*100:.1f}%)"
    )

    # Conformance check
    ratio = unanswerable_total / total * 100
    if 25 <= ratio <= 35:
        print("\n[OK] CONFORME SQuAD 2.0 (25-33%)")
    else:
        print(f"\n[WARN] Ratio {ratio:.1f}% hors cible (25-33%)")

    print("\nFiles updated:")
    print(f"  {gs_fr_path}")
    print(f"  {gs_intl_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge adversarial questions into gold standard"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without saving"
    )
    args = parser.parse_args()

    merge_adversarial(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
