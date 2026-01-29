"""Consolidate semantic audit results from all batches.

This script collects results from the 6 parallel audit agents
and generates a consolidated report.

ISO 42001 A.6.2.2: Provenance verification
ISO 29119-3: Test data validation
"""

import json
from pathlib import Path
from typing import Any


def load_batch_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load all batch result files."""
    batches = []
    for i in range(1, 7):
        result_file = results_dir / f"batch_{i}_results.json"
        if result_file.exists():
            with open(result_file, encoding="utf-8") as f:
                batches.append(json.load(f))
        else:
            print(f"Warning: {result_file} not found")
    return batches


def consolidate_results(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Consolidate all batch results into a single report."""
    all_results = []
    total_keep = 0
    total_wrong = 0
    total_partial = 0

    for batch in batches:
        results = batch.get("results", [])
        all_results.extend(results)

        summary = batch.get("summary", {})
        total_keep += summary.get("keep", 0)
        total_wrong += summary.get("wrong", 0)
        total_partial += summary.get("partial", 0)

    total = len(all_results)
    validation_rate = (total_keep / total * 100) if total > 0 else 0

    # Extract problematic questions
    wrong_questions = [r for r in all_results if r.get("action") == "WRONG"]
    partial_questions = [r for r in all_results if r.get("action") == "PARTIAL"]

    return {
        "audit_summary": {
            "total_questions": total,
            "keep": total_keep,
            "wrong": total_wrong,
            "partial": total_partial,
            "validation_rate": f"{validation_rate:.1f}%",
            "iso_compliant": validation_rate >= 95.0,
        },
        "problematic_questions": {
            "wrong": wrong_questions,
            "partial": partial_questions,
        },
        "all_results": all_results,
    }


def generate_correction_list(consolidated: dict[str, Any], output_file: Path) -> None:
    """Generate a list of questions needing correction."""
    wrong = consolidated["problematic_questions"]["wrong"]
    partial = consolidated["problematic_questions"]["partial"]

    corrections = []
    for q in wrong:
        corrections.append(
            {
                "id": q["id"],
                "action_needed": "REPLACE_CHUNK",
                "reason": q.get("reason", "Chunk does not answer question"),
                "priority": "HIGH",
            }
        )

    for q in partial:
        corrections.append(
            {
                "id": q["id"],
                "action_needed": "REVIEW_CHUNK",
                "reason": q.get("reason", "Partial answer"),
                "priority": "MEDIUM",
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "corrections_needed": len(corrections),
                "corrections": corrections,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    """Main function to consolidate audit results."""
    base_path = Path(__file__).parent.parent.parent
    results_dir = base_path / "data" / "audit_results"

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print("Loading batch results...")
    batches = load_batch_results(results_dir)

    if not batches:
        print("No batch results found. Ensure agents have completed.")
        return

    print(f"Loaded {len(batches)} batch files")

    print("\nConsolidating results...")
    consolidated = consolidate_results(batches)

    # Save consolidated report
    consolidated_file = results_dir / "consolidated.json"
    with open(consolidated_file, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, ensure_ascii=False, indent=2)
    print(f"Consolidated report saved to {consolidated_file}")

    # Generate correction list
    corrections_file = results_dir / "corrections_needed.json"
    generate_correction_list(consolidated, corrections_file)
    print(f"Corrections list saved to {corrections_file}")

    # Print summary
    summary = consolidated["audit_summary"]
    print("\n" + "=" * 50)
    print("AUDIT SUMMARY")
    print("=" * 50)
    print(f"Total questions audited: {summary['total_questions']}")
    print(f"  KEEP (valid):   {summary['keep']}")
    print(f"  WRONG:          {summary['wrong']}")
    print(f"  PARTIAL:        {summary['partial']}")
    print(f"\nValidation rate: {summary['validation_rate']}")
    print(f"ISO 25010 compliant (>=95%): {'YES' if summary['iso_compliant'] else 'NO'}")

    if summary["wrong"] > 0:
        print(f"\n[!] {summary['wrong']} questions need chunk replacement")
    if summary["partial"] > 0:
        print(f"[!] {summary['partial']} questions need review")


if __name__ == "__main__":
    main()
