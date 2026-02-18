"""
Validation Complete Gold Standard BY DESIGN - Pocket Arbiter

Script de validation sincere: execute TOUTES les quality gates (G0-G5)
via validate_all_gates() et genere un rapport honnete.

ISO Reference:
    - ISO/IEC 42001 A.6.2.2 - Validation complete
    - ISO/IEC 25010 FA-01 - Conformite specification
    - ISO/IEC 29119 - Tests validation

Usage:
    python -m scripts.evaluation.annales.validate_gs_complete \
        --gs tests/data/gs_scratch_v1.json \
        --chunks corpus/processed/chunks_mode_b_fr.json \
        --output data/gs_generation/validation_report_iso.json
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from scripts.evaluation.annales.quality_gates import (
    format_gate_report,
    validate_all_gates,
)
from scripts.pipeline.utils import load_json, save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_coverage_stats(questions: list[dict], chunks_data: dict) -> dict:
    """Compute document and chunk coverage statistics from GS questions.

    Args:
        questions: List of Schema v2 questions.
        chunks_data: Chunks corpus dict with "chunks" key.

    Returns:
        Dict with coverage_ratio (document) and chunk stats.
    """
    chunks = chunks_data.get("chunks", [])
    all_docs = {c["source"] for c in chunks}
    all_chunk_ids = {c["id"] for c in chunks}

    covered_docs: set[str] = set()
    covered_chunks: set[str] = set()

    for q in questions:
        prov = q.get("provenance", {})
        for doc in prov.get("docs", []):
            covered_docs.add(doc)
        chunk_id = prov.get("chunk_id", "")
        if chunk_id:
            covered_chunks.add(chunk_id)

    doc_ratio = len(covered_docs & all_docs) / len(all_docs) if all_docs else 0.0

    return {
        "coverage_ratio": doc_ratio,
        "covered_docs": len(covered_docs & all_docs),
        "total_docs": len(all_docs),
        "covered_chunks": len(covered_chunks & all_chunk_ids),
        "total_chunks": len(all_chunk_ids),
    }


def validate_gs_complete(
    gs_path: Path,
    chunks_path: Path | None = None,
) -> dict:
    """
    Run complete validation of Gold Standard BY DESIGN using all quality gates.

    Args:
        gs_path: Path to gold standard JSON (Schema v2).
        chunks_path: Path to chunks JSON (optional, for coverage gates).

    Returns:
        Complete validation report dict.
    """
    logger.info(f"Loading gold standard: {gs_path}")
    gs_data = load_json(gs_path)
    questions = gs_data.get("questions", [])
    total = len(questions)

    answerable = [
        q for q in questions if not q.get("content", {}).get("is_impossible", False)
    ]
    unanswerable = [
        q for q in questions if q.get("content", {}).get("is_impossible", False)
    ]

    # Compute coverage if chunks available
    coverage = None
    chunk_coverage_stats = None
    if chunks_path and chunks_path.exists():
        logger.info(f"Loading chunks: {chunks_path}")
        chunks_data = load_json(chunks_path)
        cov = compute_coverage_stats(questions, chunks_data)
        coverage = {"coverage_ratio": cov["coverage_ratio"]}
        chunk_coverage_stats = (cov["covered_chunks"], cov["total_chunks"])
    else:
        cov = {}

    # Run ALL quality gates
    logger.info("Running all quality gates...")
    all_blocking_passed, results = validate_all_gates(
        questions,
        coverage=coverage,
        chunk_coverage_stats=chunk_coverage_stats,
    )

    # Build gate results dict
    gate_results: dict[str, bool] = {}
    gate_details: list[dict] = []
    for r in results:
        key = f"{r.gate_id}_{r.name}"
        # Deduplicate per-question gates — only report aggregate pass/fail
        if key not in gate_results:
            gate_results[key] = r.passed
        else:
            # If any question fails a per-question gate, the gate fails
            gate_results[key] = gate_results[key] and r.passed

    # Build detailed list (unique gates only)
    seen_gates: set[str] = set()
    for r in results:
        key = f"{r.gate_id}_{r.name}"
        if key not in seen_gates:
            seen_gates.add(key)
            gate_details.append(
                {
                    "gate_id": r.gate_id,
                    "name": r.name,
                    "passed": gate_results[key],
                    "blocking": r.blocking,
                    "value": str(r.value),
                    "threshold": str(r.threshold),
                    "message": r.message,
                }
            )

    # Count results
    blocking_gates = [d for d in gate_details if d["blocking"]]
    warning_gates = [d for d in gate_details if not d["blocking"]]
    blocking_passed = sum(1 for d in blocking_gates if d["passed"])
    warning_passed = sum(1 for d in warning_gates if d["passed"])

    status = "VALIDATED" if all_blocking_passed else "FAILED"

    report = {
        "report_id": "VAL-GS-SCRATCH-003",
        "iso_reference": "ISO 29119-3",
        "generated_at": datetime.now().isoformat(),
        "gs_file": str(gs_path),
        "methodology": "BY DESIGN (chunk = INPUT)",
        "status": status,
        "coverage": {
            "total_questions": total,
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "unanswerable_ratio": round(len(unanswerable) / total, 2) if total else 0,
            "document_coverage": cov.get("coverage_ratio"),
            "chunk_coverage": (
                round(cov["covered_chunks"] / cov["total_chunks"], 3)
                if cov.get("total_chunks")
                else None
            ),
        },
        "validation_gates": {
            d["gate_id"] + "_" + d["name"]: d["passed"] for d in gate_details
        },
        "gate_details": gate_details,
        "summary": {
            "blocking_gates": f"{blocking_passed}/{len(blocking_gates)}",
            "warning_gates": f"{warning_passed}/{len(warning_gates)}",
            "all_blocking_passed": all_blocking_passed,
        },
    }

    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Complete validation of Gold Standard BY DESIGN"
    )

    parser.add_argument(
        "--gs",
        "-g",
        type=Path,
        required=True,
        help="Gold standard JSON file (Schema v2)",
    )
    parser.add_argument(
        "--chunks",
        "-c",
        type=Path,
        default=None,
        help="Chunks JSON file from corpus (optional, for coverage gates)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output report JSON (optional)",
    )

    args = parser.parse_args()

    report = validate_gs_complete(args.gs, args.chunks)

    # Print formatted report using quality_gates.format_gate_report
    gs_data = load_json(args.gs)
    questions = gs_data.get("questions", [])

    coverage = None
    chunk_coverage_stats = None
    if args.chunks and args.chunks.exists():
        chunks_data = load_json(args.chunks)
        cov = compute_coverage_stats(questions, chunks_data)
        coverage = {"coverage_ratio": cov["coverage_ratio"]}
        chunk_coverage_stats = (cov["covered_chunks"], cov["total_chunks"])

    _, results = validate_all_gates(
        questions,
        coverage=coverage,
        chunk_coverage_stats=chunk_coverage_stats,
    )
    print(format_gate_report(results))
    print()
    print(f"Status: {report['status']}")
    print(f"Blocking: {report['summary']['blocking_gates']}")
    print(f"Warning: {report['summary']['warning_gates']}")

    # Save report
    if args.output:
        save_json(report, args.output)
        logger.info(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
