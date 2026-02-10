"""Run ARES evaluation for 3 metrics: context relevance, answer faithfulness, answer relevance.

ARES paper (arXiv:2311.09476) Section 3.2: All 3 metrics use LLM-as-judge + PPI.
Only the system prompts and label columns differ.

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

# ARES metrics and their label column names
ARES_METRICS = ["context_relevance", "answer_faithfulness", "answer_relevance"]

ARES_LABEL_COLUMNS = {
    "context_relevance": "Context_Relevance_Label",
    "answer_faithfulness": "Answer_Faithfulness_Label",
    "answer_relevance": "Answer_Relevance_Label",
}

# ARES-verbatim system prompts (arXiv:2311.09476, Section 3.2)
ARES_SYSTEM_PROMPTS = {
    "context_relevance": (
        "You are an expert dialogue agent. Your task is to analyze the provided "
        "document and determine whether it is relevant for responding to the dialogue. "
        "In your evaluation, you should consider the content of the document and how "
        "it relates to the provided dialogue. Output your final verdict by strictly "
        'following this format: "[[Yes]]" if the document is relevant and "[[No]]" '
        "if the document provided is not relevant. Do not provide any additional "
        "explanation for your decision."
    ),
    "answer_faithfulness": (
        "You are an expert dialogue agent. Your task is to analyze the provided "
        "document and answer, then determine whether the answer is faithful to "
        "the contents of the document. The answer must not include information "
        "that cannot be supported by the document. Output your final verdict by "
        'strictly following this format: "[[Yes]]" if the answer is faithful to '
        'the document and "[[No]]" if the answer contains information not supported '
        "by the document. Do not provide any additional explanation for your decision."
    ),
    "answer_relevance": (
        "You are an expert dialogue agent. Your task is to analyze the provided "
        "question and answer, then determine whether the answer is relevant to "
        "the question asked. The answer should directly address what the question "
        "is asking. Output your final verdict by strictly following this format: "
        '"[[Yes]]" if the answer is relevant to the question and "[[No]]" if the '
        "answer does not address the question. Do not provide any additional "
        "explanation for your decision."
    ),
}

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


def _parse_yes_no(response_text: str) -> int:
    """Parse ARES [[Yes]]/[[No]] response.

    Args:
        response_text: Raw LLM response

    Returns:
        1 for Yes, 0 for No
    """
    import re

    yes_match = re.search(r"\[\[Yes\]\]", response_text, re.IGNORECASE)
    no_match = re.search(r"\[\[No\]\]", response_text, re.IGNORECASE)

    if yes_match and not no_match:
        return 1
    elif no_match:
        return 0

    # Fallback: search whole text for yes/no (first match wins)
    text_lower = response_text.lower().strip()
    if text_lower.startswith("yes") or " yes" in text_lower:
        return 1
    return 0


def _build_user_prompt(metric: str, query: str, document: str, answer: str) -> str:
    """Build user prompt content for a given metric.

    Args:
        metric: ARES metric name
        query: The question
        document: The context document
        answer: The answer text

    Returns:
        Formatted user prompt
    """
    if metric == "context_relevance":
        return f"Question: {query}\nDocument: {document[:2000]}"
    elif metric == "answer_faithfulness":
        return f"Document: {document[:2000]}\n" f"Answer: {answer[:1000]}"
    else:  # answer_relevance
        return f"Question: {query}\n" f"Answer: {answer[:1000]}"


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
    """Run ARES context relevance evaluation via ARES library.

    Uses LLM-as-judge with Prediction-Powered Inference (PPI)
    to provide calibrated confidence intervals.

    Args:
        corpus: Either 'fr' or 'intl'
        llm: LLM judge to use (see LLM_CONFIGS)
        output_dir: Directory for results
        dry_run: If True, validate config without running evaluation

    Returns:
        Evaluation results dict
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
        raise ValueError(f"Unknown LLM: {llm}. Available: {list(LLM_CONFIGS.keys())}")

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
        raise OSError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it or use a local vLLM model."
        )

    # Check ARES availability
    if not check_ares_available():
        raise ImportError("ARES not installed. Install with: pip install ares-ai")

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
    result_path = (
        output_dir
        / f"evaluation_{corpus}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_result, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {result_path}")
    print(
        f"\nContext Relevance Score: {evaluation_result['context_relevance']['score']:.2%}"
    )
    print(
        f"95% CI: [{evaluation_result['context_relevance']['ci_95_lower']:.2%}, "
        f"{evaluation_result['context_relevance']['ci_95_upper']:.2%}]"
    )

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

    Formula: theta_PP = theta_f - r_hat
    Where:
        theta_f = mean of predictions on unlabeled set
        r_hat = mean prediction error on labeled set (Yhat - Y)

    CI: theta_PP +/- z_(1-alpha/2) * sqrt(var_f/N + var_r/n)

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

    # theta_f = mean of predictions on unlabeled set
    theta_f = sum(Yhat_unlabeled) / N

    # r_hat = mean prediction error on labeled set
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


def _load_ares_data(
    corpus: str,
    metric: str,
    max_samples: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    """Load ARES data files for a given metric.

    Args:
        corpus: Either 'fr' or 'intl'
        metric: ARES metric name
        max_samples: Limit samples (0=all)

    Returns:
        (few_shot_examples, gold_samples, unlabeled_samples)
    """
    import csv

    gold_label_path = DATA_DIR / f"gold_label_{corpus}.tsv"
    few_shot_path = DATA_DIR / f"few_shot_{corpus}.tsv"
    unlabeled_path = DATA_DIR / f"unlabeled_{corpus}.tsv"

    for path in [gold_label_path, few_shot_path, unlabeled_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")

    label_col = ARES_LABEL_COLUMNS[metric]

    # Load few-shot examples
    few_shot_examples: list[dict[str, Any]] = []
    with open(few_shot_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row.get(label_col, "")
            if label in ("0", "1"):
                few_shot_examples.append(
                    {
                        "query": row["Query"],
                        "document": row["Document"],
                        "answer": row.get("Answer", ""),
                        "label": "[[Yes]]" if label == "1" else "[[No]]",
                    }
                )

    # Load gold labeled samples
    gold_samples: list[dict[str, Any]] = []
    with open(gold_label_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if label_col in row:
                gold_samples.append(
                    {
                        "query": str(row["Query"]),
                        "document": str(row["Document"]),
                        "answer": str(row.get("Answer", "")),
                        "gold_label": int(row[label_col]),
                    }
                )

    # Load unlabeled samples
    unlabeled_samples: list[dict[str, str]] = []
    with open(unlabeled_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            unlabeled_samples.append(
                {
                    "query": str(row["Query"]),
                    "document": str(row["Document"]),
                    "answer": str(row.get("Answer", "")),
                }
            )

    if max_samples > 0:
        gold_samples = gold_samples[:max_samples]
        unlabeled_samples = unlabeled_samples[:max_samples]

    return few_shot_examples, gold_samples, unlabeled_samples


def _run_llm_evaluation(
    evaluate_fn: Any,  # Callable[[str, str, str], int]
    corpus: str,
    metric: str,
    llm_name: str,
    output_dir: Path,
    max_samples: int = 0,
    progress_interval: int = 10,
) -> dict[str, Any]:
    """Shared ARES evaluation loop: gold eval + unlabeled eval + PPI.

    Args:
        evaluate_fn: Callable(query, document, answer) -> 0/1
        corpus: Either 'fr' or 'intl'
        metric: ARES metric name
        llm_name: LLM identifier for results (e.g. "ollama:mistral")
        output_dir: Directory for results
        max_samples: Limit samples (0=all)
        progress_interval: Print progress every N samples

    Returns:
        Evaluation results dict (ARES-compatible format)
    """
    few_shot_examples, gold_samples, unlabeled_samples = _load_ares_data(
        corpus, metric, max_samples
    )

    total_calls = len(gold_samples) + len(unlabeled_samples)
    print(f"  Gold labeled: {len(gold_samples)}")
    print(f"  Unlabeled: {len(unlabeled_samples)}")
    print(f"  Total LLM calls: {total_calls}")

    # Phase 1: Evaluate gold labeled samples (for PPI rectification)
    print("\nPhase 1: Evaluating gold labeled samples...")
    Y_labeled: list[int] = []
    Yhat_labeled: list[int] = []
    for i, sample in enumerate(gold_samples):
        pred = evaluate_fn(
            str(sample["query"]), str(sample["document"]), str(sample["answer"])
        )
        Y_labeled.append(int(sample["gold_label"]))
        Yhat_labeled.append(pred)
        if (i + 1) % progress_interval == 0:
            print(f"  Gold: {i + 1}/{len(gold_samples)}")

    # Phase 2: Evaluate unlabeled samples
    print("\nPhase 2: Evaluating unlabeled samples...")
    Yhat_unlabeled: list[int] = []
    for j, unlabeled in enumerate(unlabeled_samples):
        pred = evaluate_fn(
            unlabeled["query"], unlabeled["document"], unlabeled["answer"]
        )
        Yhat_unlabeled.append(pred)
        if (j + 1) % progress_interval == 0:
            print(f"  Unlabeled: {j + 1}/{len(unlabeled_samples)}")

    # ARES PPI confidence interval
    estimate, ci_lower, ci_upper = _ppi_mean_ci(Y_labeled, Yhat_labeled, Yhat_unlabeled)

    n_correct = sum(p == y for p, y in zip(Yhat_labeled, Y_labeled))
    accuracy = n_correct / len(Y_labeled) if Y_labeled else 0.0

    label_col = ARES_LABEL_COLUMNS[metric]
    result = {
        "corpus": corpus,
        "llm_used": llm_name,
        "metric": metric,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        label_col: {
            "estimate": estimate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_labeled": len(Y_labeled),
            "n_unlabeled": len(Yhat_unlabeled),
        },
        metric: {
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
    backend_tag = llm_name.split(":")[0]
    result_path = (
        output_dir
        / f"ares_{backend_tag}_{metric}_{corpus}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {result_path}")
    print(f"\n{metric} (PPI): {estimate:.2%}")
    print(f"95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")
    print(f"Judge accuracy on gold: {accuracy:.2%}")
    print(f"Pass (>=80%): {'PASS' if estimate >= 0.80 else 'FAIL'}")

    return result


def _build_few_shot_text(
    metric: str,
    few_shot_examples: list[dict[str, Any]],
    max_doc_len: int = 500,
) -> str:
    """Build few-shot prompt text from examples.

    Args:
        metric: ARES metric name
        few_shot_examples: List of few-shot example dicts
        max_doc_len: Max document length per example

    Returns:
        Formatted few-shot text
    """
    text = ""
    for ex in few_shot_examples[:5]:
        user_content = _build_user_prompt(
            metric, ex["query"], ex["document"][:max_doc_len], ex.get("answer", "")
        )
        text += f"\n{user_content}\nLabel: {ex['label']}\n"
    return text


def run_ollama_evaluation(
    corpus: str = "fr",
    model: str = "mistral:latest",
    output_dir: Path | None = None,
    max_samples: int = 0,
    metric: str = "context_relevance",
) -> dict[str, Any]:
    """ARES LLM-as-judge evaluation using Ollama.

    Args:
        corpus: Either 'fr' or 'intl'
        model: Ollama model to use
        output_dir: Directory for results
        max_samples: Limit samples (0=all)
        metric: ARES metric (context_relevance, answer_faithfulness, answer_relevance)

    Returns:
        Evaluation results dict (ARES-compatible format)
    """
    import requests

    if metric not in ARES_METRICS:
        raise ValueError(f"Unknown metric: {metric}. Available: {ARES_METRICS}")

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ConnectionError(f"Ollama not available at {ollama_url}: {e}")

    few_shot_examples, _, _ = _load_ares_data(corpus, metric, max_samples)
    system_prompt = ARES_SYSTEM_PROMPTS[metric]
    few_shot_text = _build_few_shot_text(metric, few_shot_examples)

    print(f"ARES {metric} evaluation with Ollama ({model})")

    def evaluate_fn(query: str, document: str, answer: str) -> int:
        user_content = _build_user_prompt(metric, query, document, answer)
        prompt = f"{system_prompt}\n{few_shot_text}\n{user_content}\nLabel: "
        try:
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            return _parse_yes_no(resp.json().get("response", "").strip())
        except requests.RequestException:
            return 0

    return _run_llm_evaluation(
        evaluate_fn,
        corpus,
        metric,
        f"ollama:{model}",
        output_dir,
        max_samples,
        progress_interval=10,
    )


def run_groq_evaluation(
    corpus: str = "fr",
    model: str = "llama-3.3-70b-versatile",
    output_dir: Path | None = None,
    max_samples: int = 0,
    metric: str = "context_relevance",
) -> dict[str, Any]:
    """ARES LLM-as-judge evaluation using Groq API.

    Args:
        corpus: Either 'fr' or 'intl'
        model: Groq model
        output_dir: Directory for results
        max_samples: Limit samples (0=all)
        metric: ARES metric (context_relevance, answer_faithfulness, answer_relevance)

    Returns:
        Evaluation results dict (ARES-compatible format)
    """
    from openai import OpenAI

    if metric not in ARES_METRICS:
        raise ValueError(f"Unknown metric: {metric}. Available: {ARES_METRICS}")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise OSError("GROQ_API_KEY not set. Get free key at https://console.groq.com")

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    few_shot_examples, _, _ = _load_ares_data(corpus, metric, max_samples)
    system_prompt = ARES_SYSTEM_PROMPTS[metric]

    # Build few-shot as chat messages
    few_shot_messages: list[dict[str, str]] = []
    for ex in few_shot_examples[:5]:
        user_content = _build_user_prompt(
            metric, ex["query"], ex["document"][:500], ex.get("answer", "")
        )
        few_shot_messages.append({"role": "user", "content": user_content})
        few_shot_messages.append({"role": "assistant", "content": ex["label"]})

    print(f"ARES {metric} evaluation with Groq ({model})")

    def evaluate_fn(query: str, document: str, answer: str) -> int:
        user_content = _build_user_prompt(metric, query, document, answer)
        messages = [
            {"role": "system", "content": system_prompt},
            *few_shot_messages,
            {"role": "user", "content": user_content},
        ]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=10,
                temperature=0,
            )
            return _parse_yes_no(response.choices[0].message.content or "")
        except Exception as e:
            print(f"  Error: {e}")
            return 0

    return _run_llm_evaluation(
        evaluate_fn,
        corpus,
        metric,
        f"groq:{model}",
        output_dir,
        max_samples,
        progress_interval=50,
    )


def run_huggingface_evaluation(
    corpus: str = "fr",
    model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    output_dir: Path | None = None,
    max_samples: int = 0,
    metric: str = "context_relevance",
) -> dict[str, Any]:
    """ARES LLM-as-judge evaluation using HuggingFace Inference API.

    Args:
        corpus: Either 'fr' or 'intl'
        model: HuggingFace model ID
        output_dir: Directory for results
        max_samples: Limit samples (0=all)
        metric: ARES metric (context_relevance, answer_faithfulness, answer_relevance)

    Returns:
        Evaluation results dict (ARES-compatible format)
    """
    import requests

    if metric not in ARES_METRICS:
        raise ValueError(f"Unknown metric: {metric}. Available: {ARES_METRICS}")

    api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not api_key:
        raise OSError(
            "HF_TOKEN not set. Get free token at https://huggingface.co/settings/tokens"
        )

    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    few_shot_examples, _, _ = _load_ares_data(corpus, metric, max_samples)
    system_prompt = ARES_SYSTEM_PROMPTS[metric]
    few_shot_text = _build_few_shot_text(metric, few_shot_examples, max_doc_len=300)

    print(f"ARES {metric} evaluation with HuggingFace ({model})")

    def evaluate_fn(query: str, document: str, answer: str) -> int:
        user_content = _build_user_prompt(metric, query, document, answer)
        prompt = (
            f"<s>[INST] {system_prompt}\n{few_shot_text}\n"
            f"{user_content}\nLabel: [/INST]"
        )
        try:
            resp = requests.post(
                api_url,
                headers=headers,
                json={
                    "inputs": prompt,
                    "parameters": {"max_new_tokens": 10, "temperature": 0.01},
                },
                timeout=60,
            )
            if resp.status_code == 503:
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
            text = (
                result[0].get("generated_text", "") if isinstance(result, list) else ""
            )
            return _parse_yes_no(text)
        except Exception as e:
            print(f"  Error: {e}")
            return 0

    return _run_llm_evaluation(
        evaluate_fn,
        corpus,
        metric,
        f"hf:{model}",
        output_dir,
        max_samples,
        progress_interval=10,
    )


def run_mock_evaluation(
    corpus: str = "fr",
    metric: str = "context_relevance",
) -> dict[str, Any]:
    """Run mock evaluation for testing without LLM calls.

    Generates synthetic results based on the gold label distribution.

    Args:
        corpus: Either 'fr' or 'intl'
        metric: ARES metric name

    Returns:
        Mock evaluation results
    """
    import csv
    import hashlib
    import random

    if metric not in ARES_LABEL_COLUMNS:
        raise ValueError(f"Unknown metric: {metric}. Available: {ARES_METRICS}")

    label_col = ARES_LABEL_COLUMNS[metric]

    gold_label_path = DATA_DIR / f"gold_label_{corpus}.tsv"
    if not gold_label_path.exists():
        raise FileNotFoundError(f"Gold label file not found: {gold_label_path}")

    # Read gold labels — strict: use only the exact label column for this metric
    with open(gold_label_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        labels = []
        has_label_col = False
        for row in reader:
            if label_col in row:
                has_label_col = True
                labels.append(int(row[label_col]))

    if not has_label_col:
        raise KeyError(
            f"Column '{label_col}' not found in {gold_label_path}. "
            f"Re-run convert_to_ares.py to generate all label columns."
        )

    # Calculate ground truth positive rate
    positive_rate = sum(labels) / len(labels) if labels else 0.5

    # Metric-dependent seed so different metrics get different noise
    metric_seed = int(
        hashlib.md5(metric.encode(), usedforsecurity=False).hexdigest()[:8], 16
    )
    random.seed(metric_seed)
    noise = random.uniform(-0.05, 0.05)
    simulated_score = min(max(positive_rate + noise, 0.0), 1.0)

    # Simulate confidence interval (wider for smaller samples)
    ci_width = 0.10 / (len(labels) / 50) ** 0.5

    metric_result = {
        "score": simulated_score,
        "ci_95_lower": max(0.0, simulated_score - ci_width),
        "ci_95_upper": min(1.0, simulated_score + ci_width),
        "n_samples": len(labels),
        "pass": simulated_score >= 0.80,
    }

    result: dict[str, Any] = {
        "corpus": corpus,
        "llm_used": "mock",
        "metric": metric,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        metric: metric_result,
        "note": "Mock evaluation - no LLM calls made",
    }

    # Backward-compat: always include context_relevance key
    if metric != "context_relevance":
        result["context_relevance"] = metric_result
    else:
        result["context_relevance"] = metric_result

    return result


def run_all_metrics(
    backend: str = "mock",
    corpus: str = "fr",
    model: str = "",
    max_samples: int = 0,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run all 3 ARES metrics sequentially.

    Args:
        backend: Backend to use (mock, ollama, groq, hf)
        corpus: Either 'fr' or 'intl'
        model: Model name for the backend
        max_samples: Limit samples (0=all)
        output_dir: Directory for results

    Returns:
        Combined results dict with all 3 metrics
    """
    results: dict[str, Any] = {
        "corpus": corpus,
        "backend": backend,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {},
    }

    for m in ARES_METRICS:
        print(f"\n{'='*60}")
        print(f"Running metric: {m}")
        print(f"{'='*60}")

        if backend == "mock":
            r = run_mock_evaluation(corpus=corpus, metric=m)
        elif backend == "ollama":
            r = run_ollama_evaluation(
                corpus=corpus,
                model=model,
                output_dir=output_dir,
                max_samples=max_samples,
                metric=m,
            )
        elif backend == "groq":
            r = run_groq_evaluation(
                corpus=corpus,
                model=model,
                output_dir=output_dir,
                max_samples=max_samples,
                metric=m,
            )
        elif backend == "hf":
            r = run_huggingface_evaluation(
                corpus=corpus,
                model=model,
                output_dir=output_dir,
                max_samples=max_samples,
                metric=m,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        results["metrics"][m] = r.get(m, r.get("context_relevance", {}))

    # Summary
    all_pass = all(
        results["metrics"].get(m, {}).get("pass", False) for m in ARES_METRICS
    )
    results["all_pass"] = all_pass

    print(f"\n{'='*60}")
    print("ALL METRICS SUMMARY")
    print(f"{'='*60}")
    for m in ARES_METRICS:
        score = results["metrics"].get(m, {}).get("score", 0.0)
        passed = results["metrics"].get(m, {}).get("pass", False)
        print(f"  {m}: {score:.2%} {'PASS' if passed else 'FAIL'}")
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")

    return results


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run ARES evaluation (3 metrics: context_relevance, answer_faithfulness, answer_relevance)"
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
        "--metric",
        choices=["context_relevance", "answer_faithfulness", "answer_relevance", "all"],
        default="context_relevance",
        help="ARES metric to evaluate (default: context_relevance)",
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

    if args.metric == "all":
        # Determine backend
        if args.mock:
            backend = "mock"
            model = ""
        elif args.ollama:
            backend = "ollama"
            model = args.ollama
        elif args.groq:
            backend = "groq"
            model = args.groq
        elif args.hf:
            backend = "hf"
            model = args.hf
        else:
            backend = "mock"
            model = ""

        results = run_all_metrics(
            backend=backend,
            corpus=args.corpus,
            model=model,
            max_samples=args.max_samples,
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if args.hf:
        results = run_huggingface_evaluation(
            corpus=args.corpus,
            model=args.hf,
            max_samples=args.max_samples,
            metric=args.metric,
        )
    elif args.groq:
        results = run_groq_evaluation(
            corpus=args.corpus,
            model=args.groq,
            max_samples=args.max_samples,
            metric=args.metric,
        )
    elif args.ollama:
        results = run_ollama_evaluation(
            corpus=args.corpus,
            model=args.ollama,
            max_samples=args.max_samples,
            metric=args.metric,
        )
    elif args.mock:
        results = run_mock_evaluation(corpus=args.corpus, metric=args.metric)
    else:
        results = run_context_relevance_evaluation(
            corpus=args.corpus, llm=args.llm, dry_run=args.dry_run
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
