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
    "ollama:mistral": {
        "model": "mistral:latest",
        "host": "ollama",
        "estimated_cost_per_eval": 0.0,
    },
    "ollama:qwen2.5": {
        "model": "qwen2.5:latest",
        "host": "ollama",
        "estimated_cost_per_eval": 0.0,
    },
    "ollama:gemma2": {
        "model": "gemma2:latest",
        "host": "ollama",
        "estimated_cost_per_eval": 0.0,
    },
    "ollama:llama3.2": {
        "model": "llama3.2:latest",
        "host": "ollama",
        "estimated_cost_per_eval": 0.0,
    },
    # Groq (free tier: 14,400 req/day)
    "groq:llama-3.3-70b": {
        "model": "llama-3.3-70b-versatile",
        "host": "groq",
        "estimated_cost_per_eval": 0.0,
    },
    "groq:llama-3.1-8b": {
        "model": "llama-3.1-8b-instant",
        "host": "groq",
        "estimated_cost_per_eval": 0.0,
    },
    "groq:mixtral-8x7b": {
        "model": "mixtral-8x7b-32768",
        "host": "groq",
        "estimated_cost_per_eval": 0.0,
    },
    # HuggingFace Inference API (free tier: ~30k req/month)
    "hf:mistral-7b": {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "host": "huggingface",
        "estimated_cost_per_eval": 0.0,
    },
    "hf:llama-3.2-3b": {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "host": "huggingface",
        "estimated_cost_per_eval": 0.0,
    },
    "hf:qwen2.5-72b": {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "host": "huggingface",
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


def _ppi_mean_ci(
    Y_labeled: list[int],
    Yhat_labeled: list[int],
    Yhat_unlabeled: list[int],
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """ARES-verbatim PPI mean estimation with confidence interval.

    Formula: θ̂_PP = θ̃_f - r̂
    Where:
        θ̃_f = mean of predictions on unlabeled set
        r̂ = mean prediction error on labeled set (Yhat - Y)

    CI: θ̂_PP ± z_(1-α/2) × √(σ²_f/N + σ²_r/n)

    Args:
        Y_labeled: Ground truth labels (0/1)
        Yhat_labeled: Predictions on labeled set (0/1)
        Yhat_unlabeled: Predictions on unlabeled set (0/1)
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    import math
    import statistics

    n = len(Y_labeled)
    N = len(Yhat_unlabeled)

    if n == 0 or N == 0:
        return 0.0, 0.0, 0.0

    # θ̃_f = mean of predictions on unlabeled set
    theta_f = sum(Yhat_unlabeled) / N

    # r̂ = mean prediction error on labeled set
    residuals = [yhat - y for y, yhat in zip(Y_labeled, Yhat_labeled)]
    r_hat = sum(residuals) / n

    # Point estimate
    theta_pp = theta_f - r_hat

    # Variance of predictions on unlabeled set
    if N > 1:
        var_f = statistics.variance(Yhat_unlabeled)
    else:
        var_f = 0.0

    # Variance of residuals on labeled set
    if n > 1:
        var_r = statistics.variance(residuals)
    else:
        var_r = 0.0

    # Standard error
    se = math.sqrt(var_f / N + var_r / n)

    # z critical value for (1 - alpha/2)
    # For alpha=0.05, z=1.96
    z = 1.96 if alpha == 0.05 else 2.576 if alpha == 0.01 else 1.645

    ci_lower = max(0.0, theta_pp - z * se)
    ci_upper = min(1.0, theta_pp + z * se)

    return theta_pp, ci_lower, ci_upper


def run_ollama_evaluation(
    corpus: str = "fr",
    model: str = "mistral:latest",
    output_dir: Path | None = None,
    max_samples: int = 0,
) -> dict[str, Any]:
    """ARES-VERBATIM LLM-as-judge evaluation using Ollama.

    PERFORMANCE: Full evaluation = ~6h. Use max_samples for quick tests.

    Implements EXACTLY:
    - ARES context_relevance_system_prompt
    - ARES few-shot examples format
    - ARES [[Yes]]/[[No]] response parsing
    - ARES PPI confidence intervals

    Args:
        corpus: Either 'fr' or 'intl'
        model: Ollama model to use
        output_dir: Directory for results

    Returns:
        Evaluation results dict (ARES-compatible format)
    """
    import csv
    import re
    import requests

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check Ollama is running
    ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ConnectionError(f"Ollama not available at {ollama_url}: {e}")

    # Load data files (ARES format)
    gold_label_path = DATA_DIR / f"gold_label_{corpus}.tsv"
    few_shot_path = DATA_DIR / f"few_shot_{corpus}.tsv"
    unlabeled_path = DATA_DIR / f"unlabeled_{corpus}.tsv"

    for path in [gold_label_path, few_shot_path, unlabeled_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    # Load few-shot examples (ARES format)
    few_shot_examples = []
    with open(few_shot_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row.get("Context_Relevance_Label", "")
            if label in ("0", "1"):
                few_shot_examples.append({
                    "query": row["Query"],
                    "document": row["Document"],
                    "label": "[[Yes]]" if label == "1" else "[[No]]",
                })

    # Load gold labeled samples (for PPI)
    gold_samples: list[dict[str, str | int]] = []
    with open(gold_label_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gold_samples.append({
                "query": str(row["Query"]),
                "document": str(row["Document"]),
                "gold_label": int(row["Context_Relevance_Label"]),
            })

    # Load unlabeled samples
    unlabeled_samples: list[dict[str, str]] = []
    with open(unlabeled_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            unlabeled_samples.append({
                "query": str(row["Query"]),
                "document": str(row["Document"]),
            })

    # Limit samples for quick testing
    if max_samples > 0:
        gold_samples = gold_samples[:max_samples]
        unlabeled_samples = unlabeled_samples[:max_samples]

    total_calls = len(gold_samples) + len(unlabeled_samples)
    est_minutes = total_calls * 5 / 60  # ~5s per call

    print(f"ARES-verbatim evaluation with Ollama ({model})")
    print(f"  Few-shot examples: {len(few_shot_examples)}")
    print(f"  Gold labeled: {len(gold_samples)}")
    print(f"  Unlabeled: {len(unlabeled_samples)}")
    print(f"  Total LLM calls: {total_calls}")
    print(f"  Estimated time: {est_minutes:.0f} min")

    # ARES-verbatim system prompt
    system_prompt = (
        'You are an expert dialogue agent. Your task is to analyze the provided '
        'document and determine whether it is relevant for responding to the dialogue. '
        'In your evaluation, you should consider the content of the document and how '
        'it relates to the provided dialogue. Output your final verdict by strictly '
        'following this format: "[[Yes]]" if the document is relevant and "[[No]]" '
        'if the document provided is not relevant. Do not provide any additional '
        'explanation for your decision.'
    )

    # Build few-shot prompt (ARES format)
    few_shot_text = ""
    for ex in few_shot_examples[:5]:  # ARES uses ~5 examples
        few_shot_text += f"\nQuestion: {ex['query']}\nDocument: {ex['document'][:500]}\nLabel: {ex['label']}\n"

    def evaluate_sample(query: str, document: str) -> int:
        """Evaluate a single sample, returns 1 (relevant) or 0 (not relevant)."""
        prompt = f"""{system_prompt}

{few_shot_text}

Question: {query}
Document: {document[:2000]}
Label: """

        try:
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            response_text = resp.json().get("response", "").strip()

            # ARES-verbatim parsing: [[Yes]] or [[No]]
            yes_match = re.search(r"\[\[Yes\]\]", response_text, re.IGNORECASE)
            no_match = re.search(r"\[\[No\]\]", response_text, re.IGNORECASE)

            if yes_match and not no_match:
                return 1
            elif no_match:
                return 0
            else:
                # Fallback
                return 1 if "yes" in response_text.lower()[:10] else 0
        except requests.RequestException:
            return 0

    # Evaluate gold labeled samples (for PPI rectification)
    print("\nPhase 1: Evaluating gold labeled samples...")
    Y_labeled: list[int] = []
    Yhat_labeled: list[int] = []
    for i, sample in enumerate(gold_samples):
        pred = evaluate_sample(str(sample["query"]), str(sample["document"]))
        Y_labeled.append(int(sample["gold_label"]))
        Yhat_labeled.append(pred)
        if (i + 1) % 10 == 0:
            print(f"  Gold: {i + 1}/{len(gold_samples)}")

    # Evaluate unlabeled samples
    print("\nPhase 2: Evaluating unlabeled samples...")
    Yhat_unlabeled: list[int] = []
    for j, unlabeled in enumerate(unlabeled_samples):
        pred = evaluate_sample(unlabeled["query"], unlabeled["document"])
        Yhat_unlabeled.append(pred)
        if (j + 1) % 10 == 0:
            print(f"  Unlabeled: {j + 1}/{len(unlabeled_samples)}")

    # ARES-verbatim PPI confidence interval
    estimate, ci_lower, ci_upper = _ppi_mean_ci(
        Y_labeled, Yhat_labeled, Yhat_unlabeled
    )

    # Calculate accuracy on gold set
    n_correct = sum(p == y for p, y in zip(Yhat_labeled, Y_labeled))
    accuracy = n_correct / len(Y_labeled) if Y_labeled else 0.0

    result = {
        "corpus": corpus,
        "llm_used": f"ollama:{model}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "Context_Relevance_Label": {
            "estimate": estimate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_labeled": len(Y_labeled),
            "n_unlabeled": len(Yhat_unlabeled),
        },
        "context_relevance": {
            "score": estimate,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "n_samples": len(Y_labeled) + len(Yhat_unlabeled),
            "pass": estimate >= 0.80,
        },
        "judge_accuracy_on_gold": accuracy,
        "method": "ARES-verbatim PPI",
    }

    # Save results
    result_path = output_dir / f"ares_ollama_{corpus}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {result_path}")
    print(f"\nContext Relevance (PPI): {estimate:.2%}")
    print(f"95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")
    print(f"Judge accuracy on gold: {accuracy:.2%}")
    print(f"Pass (>=80%): {'PASS' if estimate >= 0.80 else 'FAIL'}")

    return result


def run_groq_evaluation(
    corpus: str = "fr",
    model: str = "llama-3.3-70b-versatile",
    output_dir: Path | None = None,
    max_samples: int = 0,
) -> dict[str, Any]:
    """ARES-VERBATIM LLM-as-judge evaluation using Groq API.

    Groq free tier: 14,400 requests/day, ~0.5s/call.
    Full evaluation (3727 samples) in ~30 min.

    Args:
        corpus: Either 'fr' or 'intl'
        model: Groq model (llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768)
        output_dir: Directory for results
        max_samples: Limit samples (0=all)

    Returns:
        Evaluation results dict (ARES-compatible format)
    """
    import csv
    import re

    from openai import OpenAI

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Get free key at https://console.groq.com"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data files (ARES format)
    gold_label_path = DATA_DIR / f"gold_label_{corpus}.tsv"
    few_shot_path = DATA_DIR / f"few_shot_{corpus}.tsv"
    unlabeled_path = DATA_DIR / f"unlabeled_{corpus}.tsv"

    for path in [gold_label_path, few_shot_path, unlabeled_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    # Load few-shot examples
    few_shot_examples = []
    with open(few_shot_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row.get("Context_Relevance_Label", "")
            if label in ("0", "1"):
                few_shot_examples.append({
                    "query": row["Query"],
                    "document": row["Document"],
                    "label": "[[Yes]]" if label == "1" else "[[No]]",
                })

    # Load gold labeled samples
    gold_samples: list[dict[str, str | int]] = []
    with open(gold_label_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gold_samples.append({
                "query": str(row["Query"]),
                "document": str(row["Document"]),
                "gold_label": int(row["Context_Relevance_Label"]),
            })

    # Load unlabeled samples
    unlabeled_samples: list[dict[str, str]] = []
    with open(unlabeled_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            unlabeled_samples.append({
                "query": str(row["Query"]),
                "document": str(row["Document"]),
            })

    # Limit samples
    if max_samples > 0:
        gold_samples = gold_samples[:max_samples]
        unlabeled_samples = unlabeled_samples[:max_samples]

    total_calls = len(gold_samples) + len(unlabeled_samples)
    est_minutes = total_calls * 0.5 / 60  # ~0.5s per call with Groq

    print(f"ARES-verbatim evaluation with Groq ({model})")
    print(f"  Few-shot examples: {len(few_shot_examples)}")
    print(f"  Gold labeled: {len(gold_samples)}")
    print(f"  Unlabeled: {len(unlabeled_samples)}")
    print(f"  Total LLM calls: {total_calls}")
    print(f"  Estimated time: {est_minutes:.0f} min")

    # ARES-verbatim system prompt
    system_prompt = (
        'You are an expert dialogue agent. Your task is to analyze the provided '
        'document and determine whether it is relevant for responding to the dialogue. '
        'In your evaluation, you should consider the content of the document and how '
        'it relates to the provided dialogue. Output your final verdict by strictly '
        'following this format: "[[Yes]]" if the document is relevant and "[[No]]" '
        'if the document provided is not relevant. Do not provide any additional '
        'explanation for your decision.'
    )

    # Build few-shot messages
    few_shot_messages: list[dict[str, str]] = []
    for ex in few_shot_examples[:5]:
        few_shot_messages.append({
            "role": "user",
            "content": f"Question: {ex['query']}\nDocument: {ex['document'][:500]}",
        })
        few_shot_messages.append({
            "role": "assistant",
            "content": ex["label"],
        })

    def evaluate_sample(query: str, document: str) -> int:
        """Evaluate a single sample, returns 1 (relevant) or 0 (not relevant)."""
        messages = [
            {"role": "system", "content": system_prompt},
            *few_shot_messages,
            {"role": "user", "content": f"Question: {query}\nDocument: {document[:2000]}"},
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=10,
                temperature=0,
            )
            response_text = response.choices[0].message.content or ""

            # ARES-verbatim parsing
            yes_match = re.search(r"\[\[Yes\]\]", response_text, re.IGNORECASE)
            no_match = re.search(r"\[\[No\]\]", response_text, re.IGNORECASE)

            if yes_match and not no_match:
                return 1
            elif no_match:
                return 0
            else:
                return 1 if "yes" in response_text.lower()[:10] else 0
        except Exception as e:
            print(f"  Error: {e}")
            return 0

    # Evaluate gold labeled samples
    print("\nPhase 1: Evaluating gold labeled samples...")
    Y_labeled: list[int] = []
    Yhat_labeled: list[int] = []
    for i, sample in enumerate(gold_samples):
        pred = evaluate_sample(str(sample["query"]), str(sample["document"]))
        Y_labeled.append(int(sample["gold_label"]))
        Yhat_labeled.append(pred)
        if (i + 1) % 50 == 0:
            print(f"  Gold: {i + 1}/{len(gold_samples)}")

    # Evaluate unlabeled samples
    print("\nPhase 2: Evaluating unlabeled samples...")
    Yhat_unlabeled: list[int] = []
    for j, unlabeled in enumerate(unlabeled_samples):
        pred = evaluate_sample(unlabeled["query"], unlabeled["document"])
        Yhat_unlabeled.append(pred)
        if (j + 1) % 50 == 0:
            print(f"  Unlabeled: {j + 1}/{len(unlabeled_samples)}")

    # ARES-verbatim PPI confidence interval
    estimate, ci_lower, ci_upper = _ppi_mean_ci(
        Y_labeled, Yhat_labeled, Yhat_unlabeled
    )

    # Calculate accuracy on gold set
    n_correct = sum(p == y for p, y in zip(Yhat_labeled, Y_labeled))
    accuracy = n_correct / len(Y_labeled) if Y_labeled else 0.0

    result = {
        "corpus": corpus,
        "llm_used": f"groq:{model}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "Context_Relevance_Label": {
            "estimate": estimate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_labeled": len(Y_labeled),
            "n_unlabeled": len(Yhat_unlabeled),
        },
        "context_relevance": {
            "score": estimate,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "n_samples": len(Y_labeled) + len(Yhat_unlabeled),
            "pass": estimate >= 0.80,
        },
        "judge_accuracy_on_gold": accuracy,
        "method": "ARES-verbatim PPI",
    }

    # Save results
    result_path = output_dir / f"ares_groq_{corpus}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {result_path}")
    print(f"\nContext Relevance (PPI): {estimate:.2%}")
    print(f"95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")
    print(f"Judge accuracy on gold: {accuracy:.2%}")
    print(f"Pass (>=80%): {'PASS' if estimate >= 0.80 else 'FAIL'}")

    return result


def run_huggingface_evaluation(
    corpus: str = "fr",
    model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    output_dir: Path | None = None,
    max_samples: int = 0,
) -> dict[str, Any]:
    """ARES-VERBATIM LLM-as-judge evaluation using HuggingFace Inference API.

    HuggingFace free tier: ~30k requests/month.

    Args:
        corpus: Either 'fr' or 'intl'
        model: HuggingFace model ID
        output_dir: Directory for results
        max_samples: Limit samples (0=all)

    Returns:
        Evaluation results dict (ARES-compatible format)
    """
    import csv
    import re

    import requests

    api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not api_key:
        raise EnvironmentError(
            "HF_TOKEN not set. Get free token at https://huggingface.co/settings/tokens"
        )

    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data files
    gold_label_path = DATA_DIR / f"gold_label_{corpus}.tsv"
    few_shot_path = DATA_DIR / f"few_shot_{corpus}.tsv"
    unlabeled_path = DATA_DIR / f"unlabeled_{corpus}.tsv"

    for path in [gold_label_path, few_shot_path, unlabeled_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    # Load few-shot examples
    few_shot_examples = []
    with open(few_shot_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row.get("Context_Relevance_Label", "")
            if label in ("0", "1"):
                few_shot_examples.append({
                    "query": row["Query"],
                    "document": row["Document"],
                    "label": "[[Yes]]" if label == "1" else "[[No]]",
                })

    # Load gold labeled samples
    gold_samples: list[dict[str, str | int]] = []
    with open(gold_label_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gold_samples.append({
                "query": str(row["Query"]),
                "document": str(row["Document"]),
                "gold_label": int(row["Context_Relevance_Label"]),
            })

    # Load unlabeled samples
    unlabeled_samples: list[dict[str, str]] = []
    with open(unlabeled_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            unlabeled_samples.append({
                "query": str(row["Query"]),
                "document": str(row["Document"]),
            })

    # Limit samples
    if max_samples > 0:
        gold_samples = gold_samples[:max_samples]
        unlabeled_samples = unlabeled_samples[:max_samples]

    total_calls = len(gold_samples) + len(unlabeled_samples)
    est_minutes = total_calls * 1.5 / 60  # ~1.5s per call with HF

    print(f"ARES-verbatim evaluation with HuggingFace ({model})")
    print(f"  Few-shot examples: {len(few_shot_examples)}")
    print(f"  Gold labeled: {len(gold_samples)}")
    print(f"  Unlabeled: {len(unlabeled_samples)}")
    print(f"  Total LLM calls: {total_calls}")
    print(f"  Estimated time: {est_minutes:.0f} min")

    # Build few-shot prompt
    few_shot_text = ""
    for ex in few_shot_examples[:5]:
        few_shot_text += f"\nQuestion: {ex['query']}\nDocument: {ex['document'][:300]}\nRelevant: {ex['label']}\n"

    system_prompt = (
        'Determine if the document is relevant to answer the question. '
        'Reply ONLY with "[[Yes]]" or "[[No]]".'
    )

    def evaluate_sample(query: str, document: str) -> int:
        """Evaluate a single sample."""
        prompt = f"""<s>[INST] {system_prompt}

{few_shot_text}

Question: {query}
Document: {document[:1500]}
Relevant: [/INST]"""

        try:
            resp = requests.post(
                api_url,
                headers=headers,
                json={"inputs": prompt, "parameters": {"max_new_tokens": 10, "temperature": 0.01}},
                timeout=60,
            )
            if resp.status_code == 503:
                # Model loading, wait and retry
                import time
                time.sleep(20)
                resp = requests.post(
                    api_url,
                    headers=headers,
                    json={"inputs": prompt, "parameters": {"max_new_tokens": 10}},
                    timeout=60,
                )
            resp.raise_for_status()
            result = resp.json()
            response_text = result[0].get("generated_text", "") if isinstance(result, list) else ""

            # Parse response
            yes_match = re.search(r"\[\[Yes\]\]", response_text, re.IGNORECASE)
            no_match = re.search(r"\[\[No\]\]", response_text, re.IGNORECASE)

            if yes_match and not no_match:
                return 1
            elif no_match:
                return 0
            else:
                return 1 if "yes" in response_text.lower()[-50:] else 0
        except Exception as e:
            print(f"  Error: {e}")
            return 0

    # Evaluate gold labeled samples
    print("\nPhase 1: Evaluating gold labeled samples...")
    Y_labeled: list[int] = []
    Yhat_labeled: list[int] = []
    for i, sample in enumerate(gold_samples):
        pred = evaluate_sample(str(sample["query"]), str(sample["document"]))
        Y_labeled.append(int(sample["gold_label"]))
        Yhat_labeled.append(pred)
        if (i + 1) % 10 == 0:
            print(f"  Gold: {i + 1}/{len(gold_samples)}")

    # Evaluate unlabeled samples
    print("\nPhase 2: Evaluating unlabeled samples...")
    Yhat_unlabeled: list[int] = []
    for j, unlabeled in enumerate(unlabeled_samples):
        pred = evaluate_sample(unlabeled["query"], unlabeled["document"])
        Yhat_unlabeled.append(pred)
        if (j + 1) % 10 == 0:
            print(f"  Unlabeled: {j + 1}/{len(unlabeled_samples)}")

    # PPI confidence interval
    estimate, ci_lower, ci_upper = _ppi_mean_ci(
        Y_labeled, Yhat_labeled, Yhat_unlabeled
    )

    n_correct = sum(p == y for p, y in zip(Yhat_labeled, Y_labeled))
    accuracy = n_correct / len(Y_labeled) if Y_labeled else 0.0

    result = {
        "corpus": corpus,
        "llm_used": f"hf:{model}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "Context_Relevance_Label": {
            "estimate": estimate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_labeled": len(Y_labeled),
            "n_unlabeled": len(Yhat_unlabeled),
        },
        "context_relevance": {
            "score": estimate,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "n_samples": len(Y_labeled) + len(Yhat_unlabeled),
            "pass": estimate >= 0.80,
        },
        "judge_accuracy_on_gold": accuracy,
        "method": "ARES-verbatim PPI",
    }

    result_path = output_dir / f"ares_hf_{corpus}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {result_path}")
    print(f"\nContext Relevance (PPI): {estimate:.2%}")
    print(f"95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")
    print(f"Judge accuracy on gold: {accuracy:.2%}")
    print(f"Pass (>=80%): {'PASS' if estimate >= 0.80 else 'FAIL'}")

    return result


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
    parser.add_argument(
        "--ollama",
        type=str,
        metavar="MODEL",
        help="Run evaluation with Ollama model (e.g., mistral:latest)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples for quick testing (0=all, 50=~8min)",
    )
    parser.add_argument(
        "--groq",
        type=str,
        metavar="MODEL",
        help="Run with Groq API (e.g., llama-3.3-70b-versatile). Free: 14,400 req/day",
    )
    parser.add_argument(
        "--hf",
        type=str,
        metavar="MODEL",
        help="Run with HuggingFace API (e.g., mistralai/Mistral-7B-Instruct-v0.3). Free tier.",
    )
    args = parser.parse_args()

    if args.hf:
        results = run_huggingface_evaluation(
            corpus=args.corpus,
            model=args.hf,
            max_samples=args.max_samples,
        )
    elif args.groq:
        results = run_groq_evaluation(
            corpus=args.corpus,
            model=args.groq,
            max_samples=args.max_samples,
        )
    elif args.ollama:
        results = run_ollama_evaluation(
            corpus=args.corpus,
            model=args.ollama,
            max_samples=args.max_samples,
        )
    elif args.mock:
        results = run_mock_evaluation(corpus=args.corpus)
    else:
        results = run_context_relevance_evaluation(
            corpus=args.corpus, llm=args.llm, dry_run=args.dry_run
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
