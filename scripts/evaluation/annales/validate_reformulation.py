#!/usr/bin/env python3
"""
Validate GS Annales v7.5.0 reformulation quality.

Checks:
1. All questions end with "?"
2. All reformulated questions have original_annales preserved
3. Questions maintain semantic meaning (basic checks)
4. No empty questions

ISO Reference: ISO 42001, ISO 25010
"""

import json
from pathlib import Path


def load_json(path: str) -> dict:
    """Load JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_reformulation(gs: dict) -> dict:
    """Validate reformulation quality."""
    results = {
        "version": gs.get("version", "unknown"),
        "total_questions": len(gs.get("questions", [])),
        "checks": {},
        "issues": [],
    }

    questions = gs.get("questions", [])

    # Check 1: All questions end with ?
    no_qmark = [q["id"] for q in questions if not q.get("question", "").strip().endswith("?")]
    results["checks"]["questions_end_with_qmark"] = {
        "passed": len(no_qmark) == 0,
        "count": len(questions) - len(no_qmark),
        "total": len(questions),
        "issues": no_qmark[:10],
    }

    # Check 2: No empty questions
    empty = [q["id"] for q in questions if not q.get("question", "").strip()]
    results["checks"]["no_empty_questions"] = {
        "passed": len(empty) == 0,
        "count": len(questions) - len(empty),
        "total": len(questions),
        "issues": empty,
    }

    # Check 3: Reformulated questions have original_annales
    reformulated = [q for q in questions if q.get("metadata", {}).get("reformulation_method") not in ["already_clean", "unchanged"]]
    missing_orig = [q["id"] for q in reformulated if "original_annales" not in q]
    results["checks"]["original_preserved"] = {
        "passed": len(missing_orig) == 0,
        "count": len(reformulated) - len(missing_orig),
        "total": len(reformulated),
        "issues": missing_orig[:10],
    }

    # Check 4: Method distribution
    methods = {}
    for q in questions:
        m = q.get("metadata", {}).get("reformulation_method", "none")
        methods[m] = methods.get(m, 0) + 1
    results["checks"]["method_distribution"] = {
        "passed": True,
        "methods": methods,
    }

    # Check 5: Basic semantic checks
    # - Questions should be at least 10 characters
    too_short = [q["id"] for q in questions if len(q.get("question", "").strip()) < 10]
    results["checks"]["minimum_length"] = {
        "passed": len(too_short) == 0,
        "count": len(questions) - len(too_short),
        "total": len(questions),
        "issues": too_short[:10],
    }

    # Overall result
    all_passed = all(
        c.get("passed", True) for c in results["checks"].values()
    )
    results["overall_passed"] = all_passed

    return results


def main():
    """Main validation pipeline."""
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    gs_path = Path("tests/data/gold_standard_annales_fr_v7.json")

    print("=" * 60)
    print("GS Annales v7 - Reformulation Validation")
    print("=" * 60)

    print(f"\nLoading {gs_path}...")
    gs = load_json(str(gs_path))

    print(f"Validating {len(gs['questions'])} questions...")
    results = validate_reformulation(gs)

    print(f"\nVersion: {results['version']}")
    print(f"Total questions: {results['total_questions']}")
    print()

    print("Validation results:")
    for check_name, check_result in results["checks"].items():
        status = "PASS" if check_result.get("passed", True) else "FAIL"
        if "count" in check_result:
            print(f"  [{status}] {check_name}: {check_result['count']}/{check_result['total']}")
        elif "methods" in check_result:
            print(f"  [{status}] {check_name}:")
            for method, count in sorted(check_result["methods"].items(), key=lambda x: -x[1]):
                print(f"       - {method}: {count}")
        else:
            print(f"  [{status}] {check_name}")

        if check_result.get("issues"):
            print(f"       Issues: {check_result['issues'][:5]}")

    print()
    if results["overall_passed"]:
        print("[OK] Validation reformulation v7.5.0 - All checks passed")
    else:
        print("[FAIL] Some validation checks failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
