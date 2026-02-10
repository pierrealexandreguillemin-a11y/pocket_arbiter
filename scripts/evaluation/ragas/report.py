"""Generate ISO-compliant RAGAS evaluation report.

RAGAS paper (arXiv:2309.15217): 4 metrics evaluation report.

ISO Reference: ISO 42001 A.7.3, ISO 25010 S4.2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.evaluation.ragas.run_evaluation import RAGAS_METRICS

# Path constants
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "evaluation" / "ragas"
RESULTS_DIR = DATA_DIR / "results"


def load_latest_ragas_result(corpus: str) -> dict[str, Any] | None:
    """Load the most recent RAGAS evaluation result.

    Args:
        corpus: Either 'fr' or 'intl'

    Returns:
        Evaluation result dict or None
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pattern = f"ragas_{corpus}_*.json"
    files = sorted(RESULTS_DIR.glob(pattern), reverse=True)

    if not files:
        return None

    with open(files[0], encoding="utf-8") as f:
        return json.load(f)


def generate_ragas_report(
    corpus: str = "fr",
    output_dir: Path | None = None,
    evaluation_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate ISO-compliant RAGAS evaluation report.

    Args:
        corpus: Either 'fr' or 'intl'
        output_dir: Output directory
        evaluation_result: Pre-computed results

    Returns:
        Report dict
    """
    if output_dir is None:
        output_dir = DATA_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    if evaluation_result is None:
        evaluation_result = load_latest_ragas_result(corpus)

    if evaluation_result is None:
        return _create_pending_report(corpus, output_dir)

    metrics_data = evaluation_result.get("metrics", {})

    # Build report
    report: dict[str, Any] = {
        "metadata": {
            "corpus": corpus,
            "report_version": "1.0",
            "framework": "RAGAS (arXiv:2309.15217)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "iso_references": ["ISO 42001 A.7.3", "ISO 25010 S4.2"],
        },
        "evaluation_type": "self_consistency_check",
        "evaluation_warning": (
            "Score computed on BY-DESIGN questions against their source chunks. "
            "This measures GS internal consistency, NOT RAG system performance."
        ),
    }

    # Add each metric
    for m in RAGAS_METRICS:
        data = metrics_data.get(m, {})
        report[m] = {
            "score": data.get("score", 0.0),
            "pass": data.get("pass", False),
            "n_samples": data.get("n_samples", 0),
            "threshold": 0.80,
        }

    # ISO compliance
    checks: dict[str, dict[str, Any]] = {}
    for m in RAGAS_METRICS:
        score = metrics_data.get(m, {}).get("score", 0.0)
        checks[f"{m}_threshold"] = {
            "requirement": f"{m} >= 80%",
            "value": score,
            "pass": score >= 0.80,
            "reference": "ISO 25010 S4.2",
        }

    report["iso_compliance"] = {
        "overall_pass": all(c["pass"] for c in checks.values()),
        "checks": checks,
    }

    # Recommendations
    recommendations = []
    for m in RAGAS_METRICS:
        score = metrics_data.get(m, {}).get("score", 0.0)
        if score < 0.80:
            recommendations.append(f"PRIORITY: {m} ({score:.1%}) below 80% threshold.")

    if not recommendations:
        recommendations.append("All RAGAS metrics within acceptable ranges.")

    report["recommendations"] = recommendations

    # Save
    report_path = output_dir / f"ragas_report_{corpus}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"RAGAS report saved to: {report_path}")
    _print_summary(report)

    return report


def _create_pending_report(corpus: str, output_dir: Path) -> dict[str, Any]:
    """Create pending report when no evaluation exists."""
    report = {
        "metadata": {
            "corpus": corpus,
            "report_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        },
        "message": "No RAGAS evaluation found. Run evaluation first.",
        "instructions": [
            f"1. python -m scripts.evaluation.ragas.convert_to_ragas --corpus {corpus}",
            f"2. python -m scripts.evaluation.ragas.run_evaluation --corpus {corpus}",
            f"3. python -m scripts.evaluation.ragas.report --corpus {corpus}",
        ],
    }

    report_path = output_dir / f"ragas_report_{corpus}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Pending report saved to: {report_path}")
    return report


def _print_summary(report: dict[str, Any]) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION REPORT SUMMARY (4 METRICS)")
    print("=" * 60)

    if report.get("evaluation_type") == "self_consistency_check":
        print("\n[WARNING] Self-consistency check - NOT RAG system evaluation")

    for m in RAGAS_METRICS:
        data = report.get(m, {})
        if data:
            score = data.get("score", 0)
            passed = data.get("pass", False)
            print(f"  {m}: {score:.1%} {'PASS' if passed else 'FAIL'}")

    iso = report.get("iso_compliance", {})
    print(f"\nISO Compliance: {'PASS' if iso.get('overall_pass') else 'FAIL'}")
    print("=" * 60)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate ISO-compliant RAGAS evaluation report"
    )
    parser.add_argument(
        "--corpus",
        choices=["fr", "intl"],
        default="fr",
    )
    args = parser.parse_args()

    generate_ragas_report(corpus=args.corpus)


if __name__ == "__main__":
    main()
