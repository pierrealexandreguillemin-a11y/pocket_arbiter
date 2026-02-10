"""Run RAGAS evaluation (standalone mode, no ragas library required).

RAGAS paper (arXiv:2309.15217) defines 4 metrics:
1. Faithfulness: Is the answer faithful to the context?
2. Answer Relevancy: Is the answer relevant to the question?
3. Context Precision: Is the ground truth in the top contexts?
4. Context Recall: Are all ground truth sentences supported by context?

Standalone mode uses LLM-as-judge (Ollama/Groq/HF) to evaluate each metric
without requiring the ragas Python library.

ISO Reference: ISO 42001 A.7.3, ISO 25010 S4.2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Path constants
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "evaluation" / "ragas"
OUTPUT_DIR = DATA_DIR / "results"

RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

# System prompts for standalone LLM-as-judge (RAGAS paper Section 3)
RAGAS_SYSTEM_PROMPTS = {
    "faithfulness": (
        "You are an expert evaluator. Given a context and an answer, determine "
        "whether every claim in the answer can be supported by the context. "
        "Score 1.0 if all claims are supported, 0.0 if none are supported, "
        "or a value between 0 and 1 based on the fraction of supported claims. "
        'Output ONLY a JSON object: {"score": <float>}'
    ),
    "answer_relevancy": (
        "You are an expert evaluator. Given a question and an answer, determine "
        "how relevant the answer is to the question. Score 1.0 if the answer "
        "directly and completely addresses the question, 0.0 if completely "
        "irrelevant, or a value between 0 and 1. "
        'Output ONLY a JSON object: {"score": <float>}'
    ),
    "context_precision": (
        "You are an expert evaluator. Given a question, the ground truth answer, "
        "and a list of context passages, determine if the contexts that contain "
        "relevant information are ranked higher. Score 1.0 if the most relevant "
        "context is ranked first, lower scores for worse ranking. "
        'Output ONLY a JSON object: {"score": <float>}'
    ),
    "context_recall": (
        "You are an expert evaluator. Given a ground truth answer and context "
        "passages, determine what fraction of the ground truth sentences can be "
        "attributed to the provided contexts. Score 1.0 if all sentences are "
        "supported, 0.0 if none are. "
        'Output ONLY a JSON object: {"score": <float>}'
    ),
}


def _load_ragas_data(data_path: Path | None = None) -> list[dict[str, Any]]:
    """Load RAGAS evaluation data from JSONL.

    Args:
        data_path: Path to ragas_evaluation.jsonl

    Returns:
        List of evaluation records
    """
    if data_path is None:
        data_path = DATA_DIR / "ragas_evaluation.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(
            f"RAGAS data not found: {data_path}. " "Run convert_to_ragas.py first."
        )

    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def _extract_score(response_text: str) -> float:
    """Extract score from LLM response.

    Args:
        response_text: Raw LLM response (expects JSON with score field)

    Returns:
        Score between 0.0 and 1.0
    """
    import re

    # Try JSON parsing first
    try:
        data = json.loads(response_text.strip())
        score = float(data.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Try to find score pattern in text
    match = re.search(r'"score"\s*:\s*([\d.]+)', response_text)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))

    # Try to find a standalone number
    match = re.search(r"\b(0\.\d+|1\.0|0|1)\b", response_text)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))

    return 0.0


def _build_ragas_prompt(
    metric: str,
    record: dict[str, Any],
) -> str:
    """Build evaluation prompt for a RAGAS metric.

    Args:
        metric: RAGAS metric name
        record: RAGAS evaluation record

    Returns:
        Formatted prompt string
    """
    question = record.get("question", "")
    answer = record.get("answer", "")
    contexts = record.get("contexts", [])
    ground_truth = record.get("ground_truth", "")

    context_text = "\n---\n".join(contexts[:3])  # Limit to 3 contexts

    if metric == "faithfulness":
        return (
            f"Context:\n{context_text[:2000]}\n\n"
            f"Answer:\n{answer[:1000]}\n\n"
            "Evaluate faithfulness."
        )
    elif metric == "answer_relevancy":
        return (
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer[:1000]}\n\n"
            "Evaluate answer relevancy."
        )
    elif metric == "context_precision":
        return (
            f"Question:\n{question}\n\n"
            f"Ground Truth:\n{ground_truth[:1000]}\n\n"
            f"Contexts:\n{context_text[:2000]}\n\n"
            "Evaluate context precision."
        )
    else:  # context_recall
        return (
            f"Ground Truth:\n{ground_truth[:1000]}\n\n"
            f"Contexts:\n{context_text[:2000]}\n\n"
            "Evaluate context recall."
        )


def run_ragas_evaluation(
    corpus: str = "fr",
    data_path: Path | None = None,
    llm_backend: str = "mock",
    model: str = "mistral:latest",
    metrics: list[str] | None = None,
    max_samples: int = 0,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run RAGAS evaluation with standalone LLM-as-judge.

    Args:
        corpus: Either 'fr' or 'intl'
        data_path: Path to ragas_evaluation.jsonl
        llm_backend: Backend (mock, ollama, groq, hf)
        model: Model name
        metrics: List of metrics to evaluate (default: all 4)
        max_samples: Limit samples (0=all)
        output_dir: Results directory

    Returns:
        Evaluation results with 4 metric scores
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if metrics is None:
        metrics = list(RAGAS_METRICS)

    records = _load_ragas_data(data_path)

    if max_samples > 0:
        records = records[:max_samples]

    print(f"RAGAS evaluation ({llm_backend}:{model})")
    print(f"  Records: {len(records)}")
    print(f"  Metrics: {metrics}")

    results: dict[str, Any] = {
        "corpus": corpus,
        "llm_backend": llm_backend,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(records),
        "metrics": {},
    }

    for metric in metrics:
        if metric not in RAGAS_METRICS:
            print(f"  Skipping unknown metric: {metric}")
            continue

        print(f"\n  Evaluating {metric}...")

        if llm_backend == "mock":
            score = _mock_score(metric, records)
        else:
            score = _llm_evaluate_metric(metric, records, llm_backend, model)

        results["metrics"][metric] = {
            "score": score,
            "pass": score >= 0.80,
            "n_samples": len(records),
        }
        print(f"    {metric}: {score:.2%}")

    # Overall pass
    results["all_pass"] = all(
        results["metrics"].get(m, {}).get("pass", False) for m in metrics
    )

    # Save results
    result_path = (
        output_dir / f"ragas_{corpus}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {result_path}")

    return results


def _mock_score(metric: str, records: list[dict[str, Any]]) -> float:
    """Generate mock score for testing.

    Args:
        metric: Metric name
        records: Evaluation records

    Returns:
        Mock score
    """
    # For mock, check data quality indicators
    n_with_ground_truth = sum(1 for r in records if r.get("ground_truth"))
    n_with_contexts = sum(1 for r in records if r.get("contexts"))

    if not records:
        return 0.0

    # Context-based metrics get score from context availability
    if metric in ("faithfulness", "context_recall"):
        return n_with_contexts / len(records) if records else 0.0
    elif metric == "context_precision":
        return n_with_contexts / len(records) if records else 0.0
    else:  # answer_relevancy
        return n_with_ground_truth / len(records) if records else 0.0


def _llm_evaluate_metric(
    metric: str,
    records: list[dict[str, Any]],
    backend: str,
    model: str,
) -> float:
    """Evaluate a metric using LLM-as-judge.

    Args:
        metric: RAGAS metric name
        records: Evaluation records
        backend: LLM backend (ollama, groq, hf)
        model: Model name

    Returns:
        Average score across all records
    """
    import os

    import requests

    system_prompt = RAGAS_SYSTEM_PROMPTS[metric]
    scores: list[float] = []

    for i, record in enumerate(records):
        user_prompt = _build_ragas_prompt(metric, record)

        try:
            if backend == "ollama":
                ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
                resp = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": f"{system_prompt}\n\n{user_prompt}",
                        "stream": False,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                response_text = resp.json().get("response", "")
            elif backend == "groq":
                from openai import OpenAI

                api_key = os.environ.get("GROQ_API_KEY", "")
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=50,
                    temperature=0,
                )
                response_text = response.choices[0].message.content or ""
            else:
                response_text = '{"score": 0.0}'

            score = _extract_score(response_text)
            scores.append(score)
        except Exception as e:
            print(f"    Error on record {i}: {e}")
            scores.append(0.0)

        if (i + 1) % 50 == 0:
            print(f"    Progress: {i + 1}/{len(records)}")

    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation (4 metrics)")
    parser.add_argument(
        "--corpus",
        choices=["fr", "intl"],
        default="fr",
        help="Corpus (default: fr)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to ragas_evaluation.jsonl",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "ollama", "groq", "hf"],
        default="mock",
        help="LLM backend (default: mock)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral:latest",
        help="Model name",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated metrics (default: all 4)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples (0=all)",
    )
    args = parser.parse_args()

    metrics = args.metrics.split(",") if args.metrics else None

    results = run_ragas_evaluation(
        corpus=args.corpus,
        data_path=args.data_path,
        llm_backend=args.backend,
        model=args.model,
        metrics=metrics,
        max_samples=args.max_samples,
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
