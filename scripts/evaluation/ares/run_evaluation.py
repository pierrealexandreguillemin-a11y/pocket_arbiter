"""Run ARES evaluation for context relevance.

ISO Reference: ISO 42001 A.7.3, ISO 25010 S4.2
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Path constants
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "evaluation" / "ares"
OUTPUT_DIR = DATA_DIR / "results"

# Default LLM judge configurations
LLM_CONFIGS = {
    "gpt-4o-mini": {
        "model": "gpt-4o-mini",
        "host": "openai",
        "estimated_cost_per_eval": 0.02,
    },
    "gpt-4o": {
        "model": "gpt-4o",
        "host": "openai",
        "estimated_cost_per_eval": 0.10,
    },
    "vllm:mistral-7b": {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "host": "vllm",
        "estimated_cost_per_eval": 0.0,
    },
    "vllm:llama-3-8b": {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "host": "vllm",
        "estimated_cost_per_eval": 0.0,
    },
}


def check_ares_available() -> bool:
    """Check if ARES is installed."""
    try:
        from ares import ARES  # noqa: F401

        return True
    except ImportError:
        return False


def check_openai_api_key() -> bool:
    """Check if OpenAI API key is configured."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def run_context_relevance_evaluation(
    corpus: str = "fr",
    llm: str = "gpt-4o-mini",
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run ARES context relevance evaluation.

    Uses LLM-as-judge with Prediction-Powered Inference (PPI)
    to provide calibrated confidence intervals.

    Args:
        corpus: Either 'fr' or 'intl'
        llm: LLM judge to use (see LLM_CONFIGS)
        output_dir: Directory for results
        dry_run: If True, validate config without running evaluation

    Returns:
        Evaluation results dict with:
        - context_relevance_score: float
        - ci_95_lower: float
        - ci_95_upper: float
        - n_samples: int
        - llm_used: str
        - timestamp: str
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate files exist
    gold_label_path = DATA_DIR / f"gold_label_{corpus}.tsv"
    few_shot_path = DATA_DIR / f"few_shot_{corpus}.tsv"
    unlabeled_path = DATA_DIR / f"unlabeled_{corpus}.tsv"

    missing_files = []
    for path in [gold_label_path, few_shot_path, unlabeled_path]:
        if not path.exists():
            missing_files.append(str(path))

    if missing_files:
        raise FileNotFoundError(
            f"Missing required files. Run convert_to_ares.py and "
            f"generate_few_shot.py first.\nMissing: {missing_files}"
        )

    # Validate LLM config
    if llm not in LLM_CONFIGS:
        raise ValueError(
            f"Unknown LLM: {llm}. Available: {list(LLM_CONFIGS.keys())}"
        )

    llm_config = LLM_CONFIGS[llm]

    if dry_run:
        return {
            "status": "dry_run",
            "corpus": corpus,
            "llm": llm,
            "files_validated": True,
            "gold_label_path": str(gold_label_path),
            "few_shot_path": str(few_shot_path),
            "unlabeled_path": str(unlabeled_path),
            "estimated_cost": _estimate_cost(unlabeled_path, llm_config),
        }

    # Check API key for OpenAI
    if llm_config["host"] == "openai" and not check_openai_api_key():
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it or use a local vLLM model."
        )

    # Check ARES availability
    if not check_ares_available():
        raise ImportError(
            "ARES not installed. Install with: pip install ares-ai"
        )

    # Import ARES (only when actually running)
    from ares import ARES

    # Build ARES configuration
    ppi_config = {
        "evaluation_datasets": [str(unlabeled_path)],
        "few_shot_examples_filepath": str(few_shot_path),
        "gold_label_path": str(gold_label_path),
        "model_choice": llm_config["model"],
        "labels": ["Context_Relevance_Label"],
        "debug_mode": False,
        "rag_type": "question_answering",
        "label_column": "Context_Relevance_Label",
        "text_column": "Document",
        "query_column": "Query",
        "answer_column": "Answer",
    }

    # Add vLLM specific config if needed
    if llm_config["host"] == "vllm":
        ppi_config["vllm"] = True
        ppi_config["host_url"] = os.environ.get(
            "VLLM_HOST_URL", "http://localhost:8000"
        )

    # Run evaluation
    print(f"Running ARES evaluation for corpus '{corpus}' with LLM '{llm}'...")

    ares = ARES(ppi=ppi_config)
    results = ares.evaluate_RAG()

    # Extract results
    context_relevance = results.get("Context_Relevance_Label", {})

    evaluation_result = {
        "corpus": corpus,
        "llm_used": llm,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "context_relevance": {
            "score": context_relevance.get("estimate", 0.0),
            "ci_95_lower": context_relevance.get("ci_lower", 0.0),
            "ci_95_upper": context_relevance.get("ci_upper", 0.0),
            "n_samples": context_relevance.get("n_samples", 0),
            "pass": context_relevance.get("estimate", 0.0) >= 0.80,
        },
        "raw_results": results,
        "config": ppi_config,
    }

    # Save results
    result_path = output_dir / f"evaluation_{corpus}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_result, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {result_path}")
    print(f"\nContext Relevance Score: {evaluation_result['context_relevance']['score']:.2%}")
    print(f"95% CI: [{evaluation_result['context_relevance']['ci_95_lower']:.2%}, "
          f"{evaluation_result['context_relevance']['ci_95_upper']:.2%}]")

    return evaluation_result


def _estimate_cost(unlabeled_path: Path, llm_config: dict[str, Any]) -> dict[str, Any]:
    """Estimate evaluation cost.

    Args:
        unlabeled_path: Path to unlabeled TSV
        llm_config: LLM configuration

    Returns:
        Cost estimate dict
    """
    import csv

    # Count TSV rows properly (handles multiline content in cells)
    with open(unlabeled_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # Skip header
        n_samples = sum(1 for _ in reader)

    cost_per_eval = llm_config.get("estimated_cost_per_eval", 0.0)
    total_cost = n_samples * cost_per_eval

    return {
        "n_samples": n_samples,
        "cost_per_eval_usd": cost_per_eval,
        "estimated_total_usd": total_cost,
        "llm": llm_config.get("model"),
    }


def run_mock_evaluation(corpus: str = "fr") -> dict[str, Any]:
    """Run mock evaluation for testing without LLM calls.

    Generates synthetic results based on the gold label distribution.

    Args:
        corpus: Either 'fr' or 'intl'

    Returns:
        Mock evaluation results
    """
    import csv
    import random

    gold_label_path = DATA_DIR / f"gold_label_{corpus}.tsv"
    if not gold_label_path.exists():
        raise FileNotFoundError(f"Gold label file not found: {gold_label_path}")

    # Read gold labels
    with open(gold_label_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        labels = [int(row["Context_Relevance_Label"]) for row in reader]

    # Calculate ground truth positive rate
    positive_rate = sum(labels) / len(labels) if labels else 0.5

    # Simulate LLM judge with some noise
    random.seed(42)
    noise = random.uniform(-0.05, 0.05)
    simulated_score = min(max(positive_rate + noise, 0.0), 1.0)

    # Simulate confidence interval (wider for smaller samples)
    ci_width = 0.10 / (len(labels) / 50) ** 0.5

    return {
        "corpus": corpus,
        "llm_used": "mock",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "context_relevance": {
            "score": simulated_score,
            "ci_95_lower": max(0.0, simulated_score - ci_width),
            "ci_95_upper": min(1.0, simulated_score + ci_width),
            "n_samples": len(labels),
            "pass": simulated_score >= 0.80,
        },
        "note": "Mock evaluation - no LLM calls made",
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run ARES context relevance evaluation"
    )
    parser.add_argument(
        "--corpus",
        choices=["fr", "intl"],
        default="fr",
        help="Corpus to evaluate (default: fr)",
    )
    parser.add_argument(
        "--llm",
        choices=list(LLM_CONFIGS.keys()),
        default="gpt-4o-mini",
        help="LLM judge to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running evaluation",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run mock evaluation without LLM calls",
    )
    args = parser.parse_args()

    if args.mock:
        results = run_mock_evaluation(corpus=args.corpus)
    else:
        results = run_context_relevance_evaluation(
            corpus=args.corpus, llm=args.llm, dry_run=args.dry_run
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
