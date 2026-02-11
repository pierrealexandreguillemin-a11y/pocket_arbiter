"""Validate Gold Standard items by LLM-as-judge (3 independent criteria).

Each GS item (question, chunk, expected_answer) is evaluated on:
1. Context Relevance (CR): Is the chunk relevant for answering this question?
2. Answer Faithfulness (AF): Is the expected answer grounded in the chunk?
3. Answer Relevance (AR): Does the expected answer actually address the question?

References:
- "A Survey on LLM-as-a-Judge" (arXiv:2411.15594, Nov 2024)
- "LLM Judge for Legal RAG with Gwet's AC" (arXiv:2509.12382, 2025)
- ARES (arXiv:2311.09476) Section 3.2: system prompts
- ISO 29119: exhaustive evaluation when N <= 1000

ISO Reference: ISO 42001 A.7.3, ISO 29119
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Path constants
BASE_DIR = Path(__file__).parent.parent.parent
TESTS_DATA_DIR = BASE_DIR / "tests" / "data"
CORPUS_DIR = BASE_DIR / "corpus" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "evaluation" / "gs_validation"

# The 3 validation criteria
CRITERIA = ["context_relevance", "answer_faithfulness", "answer_relevance"]

# System prompts for each criterion (ARES-style, adapted for GS validation)
JUDGE_PROMPTS: dict[str, str] = {
    "context_relevance": (
        "You are an expert evaluator for a chess arbitration knowledge base. "
        "Given a QUESTION and a CHUNK (text passage), determine whether the chunk "
        "contains information relevant to answering the question.\n\n"
        "Output your verdict as [[PASS]] if the chunk is relevant, or [[FAIL]] "
        "if the chunk is not relevant. Follow with a brief justification (1 sentence)."
    ),
    "answer_faithfulness": (
        "You are an expert evaluator for a chess arbitration knowledge base. "
        "Given a CHUNK (text passage) and an EXPECTED ANSWER, determine whether "
        "the answer is faithfully grounded in the chunk without adding information "
        "that cannot be found or inferred from the chunk.\n\n"
        "Output your verdict as [[PASS]] if the answer is faithful to the chunk, "
        "or [[FAIL]] if the answer contains claims not supported by the chunk. "
        "Follow with a brief justification (1 sentence)."
    ),
    "answer_relevance": (
        "You are an expert evaluator for a chess arbitration knowledge base. "
        "Given a QUESTION and an EXPECTED ANSWER, determine whether the answer "
        "directly and adequately addresses the question.\n\n"
        "Output your verdict as [[PASS]] if the answer is relevant to the question, "
        "or [[FAIL]] if the answer does not address the question. "
        "Follow with a brief justification (1 sentence)."
    ),
}


def _load_gs_items(
    corpus: str = "fr",
    gs_path: Path | None = None,
) -> list[dict[str, str]]:
    """Load GS items with their associated chunk text.

    Reads Schema V2 (gs_scratch_v1.json) and chunks_mode_b_{corpus}.json.
    Filters to ANSWERABLE + VALIDATED items with a valid chunk.

    Args:
        corpus: Either 'fr' or 'intl'
        gs_path: Override path to GS file

    Returns:
        List of dicts with gs_id, question, chunk_text, expected_answer
    """
    if gs_path is None:
        gs_path = TESTS_DATA_DIR / "gs_scratch_v1.json"
    if not gs_path.exists():
        raise FileNotFoundError(f"Gold standard not found: {gs_path}")

    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)

    # Load chunks
    chunks_path = CORPUS_DIR / f"chunks_mode_b_{corpus}.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks not found: {chunks_path}")

    with open(chunks_path, encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks = {c["id"]: c["text"] for c in chunks_data.get("chunks", [])}

    items: list[dict[str, str]] = []
    for q in gs.get("questions", []):
        # Schema V2: nested fields
        chunk_id = q.get("provenance", {}).get("chunk_id", "")
        status = q.get("validation", {}).get("status", "")
        hard_type = q.get("classification", {}).get("hard_type", "")
        question = q.get("content", {}).get("question", "")
        expected_answer = q.get("content", {}).get("expected_answer", "")

        if not chunk_id or status != "VALIDATED" or hard_type != "ANSWERABLE":
            continue

        # Skip items with empty question or answer (would produce meaningless eval)
        if not question or not expected_answer:
            continue

        chunk_text = chunks.get(chunk_id, "")
        if not chunk_text:
            continue

        gs_id = q.get("id", "")
        if not gs_id:
            continue

        items.append(
            {
                "gs_id": gs_id,
                "question": question,
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "expected_answer": expected_answer,
            }
        )

    return items


def _build_judge_prompt(
    criterion: str,
    question: str,
    chunk_text: str,
    expected_answer: str,
) -> str:
    """Build user prompt for a given criterion.

    Args:
        criterion: One of CRITERIA
        question: The GS question
        chunk_text: The source chunk text
        expected_answer: The expected answer from GS

    Returns:
        Formatted user prompt
    """
    if criterion == "context_relevance":
        return f"QUESTION: {question}\n\n" f"CHUNK: {chunk_text[:2000]}"
    elif criterion == "answer_faithfulness":
        return (
            f"CHUNK: {chunk_text[:2000]}\n\n"
            f"EXPECTED ANSWER: {expected_answer[:1000]}"
        )
    else:  # answer_relevance
        return f"QUESTION: {question}\n\n" f"EXPECTED ANSWER: {expected_answer[:1000]}"


def _parse_verdict(response_text: str) -> tuple[bool, str]:
    """Parse [[PASS]]/[[FAIL]] verdict from LLM response.

    Args:
        response_text: Raw LLM response

    Returns:
        (is_pass, reason)
    """
    pass_match = re.search(r"\[\[PASS\]\]", response_text, re.IGNORECASE)
    fail_match = re.search(r"\[\[FAIL\]\]", response_text, re.IGNORECASE)

    if pass_match and not fail_match:
        is_pass = True
    elif fail_match:
        is_pass = False
    else:
        # Fallback: look for pass/fail keywords
        lower = response_text.lower().strip()
        is_pass = "pass" in lower and "fail" not in lower

    # Extract reason (text after the verdict marker, case-insensitive)
    reason = response_text.strip()
    marker_match = re.search(r"\[\[(PASS|FAIL)\]\]", reason, re.IGNORECASE)
    if marker_match:
        reason = reason[marker_match.end() :].strip()

    # Truncate long reasons
    if len(reason) > 200:
        reason = reason[:197] + "..."

    return is_pass, reason


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    backend: str,
    model: str,
) -> str:
    """Call LLM backend and return raw response text.

    Args:
        system_prompt: System message
        user_prompt: User message
        backend: 'mock', 'ollama', or 'groq'
        model: Model name

    Returns:
        Raw response text
    """
    if backend == "mock":
        return "[[PASS]] Mock evaluation - no LLM call made."

    if backend == "ollama":
        import os

        import requests

        ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        resp = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()

    if backend == "groq":
        import os

        from openai import OpenAI

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise OSError("GROQ_API_KEY not set")

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
            max_tokens=250,
            temperature=0,
        )
        return (response.choices[0].message.content or "").strip()

    raise ValueError(f"Unsupported backend: {backend}. Available: mock, ollama, groq")


def _judge_item(
    question: str,
    chunk_text: str,
    expected_answer: str,
    criterion: str,
    backend: str = "mock",
    model: str = "mistral:latest",
) -> tuple[bool, str]:
    """Evaluate one GS item on one criterion.

    Args:
        question: The GS question
        chunk_text: The source chunk text
        expected_answer: The expected answer
        criterion: One of CRITERIA
        backend: LLM backend
        model: Model name

    Returns:
        (is_pass, reason)
    """
    if criterion not in CRITERIA:
        raise ValueError(f"Unknown criterion: {criterion}. Available: {CRITERIA}")

    system_prompt = JUDGE_PROMPTS[criterion]
    user_prompt = _build_judge_prompt(criterion, question, chunk_text, expected_answer)
    response = _call_llm(system_prompt, user_prompt, backend, model)
    return _parse_verdict(response)


def _compute_agreement(
    judge_labels: list[bool],
) -> dict[str, float]:
    """Compute inter-annotator agreement metrics.

    GS creator labels are assumed all True (creator believes all items correct).
    Judge labels come from the LLM evaluation.

    Computes: raw agreement, Cohen's Kappa, Gwet's AC1.

    Note: AC1 (not AC2) is the correct name for the binary nominal formula
    P_e = 2*pi*(1-pi). AC2 uses ordinal quadratic weights.

    Args:
        judge_labels: List of boolean verdicts from the judge

    Returns:
        Dict with raw_agreement, cohens_kappa, gwets_ac1
    """
    if not judge_labels:
        return {"raw_agreement": 0.0, "cohens_kappa": 0.0, "gwets_ac1": 0.0}

    n = len(judge_labels)
    gs_labels = [True] * n  # GS creator assumes all items are correct

    # Raw agreement
    agree = sum(1 for g, j in zip(gs_labels, judge_labels) if g == j)
    raw_agreement = agree / n

    # Marginal proportions for Cohen's Kappa
    # GS: all True -> p_gs_true = 1.0, p_gs_false = 0.0
    p_judge_true = sum(judge_labels) / n
    p_judge_false = 1.0 - p_judge_true

    # Expected agreement by chance (Cohen's Kappa)
    # p_e = p(both True by chance) + p(both False by chance)
    # = (1.0 * p_judge_true) + (0.0 * p_judge_false)
    p_e = 1.0 * p_judge_true + 0.0 * p_judge_false

    if abs(1.0 - p_e) < 1e-10:
        cohens_kappa = 1.0 if raw_agreement == 1.0 else 0.0
    else:
        cohens_kappa = (raw_agreement - p_e) / (1.0 - p_e)

    # Gwet's AC1 (more robust to prevalence paradox than Kappa)
    # Binary nominal case: P_e = 2*pi*(1-pi)
    # Pi = overall proportion of True across both raters
    pi = (1.0 + p_judge_true) / 2.0  # average of gs=1.0 and judge rate
    p_e_ac1 = 2.0 * pi * (1.0 - pi)

    if abs(1.0 - p_e_ac1) < 1e-10:
        gwets_ac1 = 1.0 if raw_agreement == 1.0 else 0.0
    else:
        gwets_ac1 = (raw_agreement - p_e_ac1) / (1.0 - p_e_ac1)

    return {
        "raw_agreement": round(raw_agreement, 4),
        "cohens_kappa": round(cohens_kappa, 4),
        "gwets_ac1": round(gwets_ac1, 4),
    }


def _generate_report(
    results: list[dict[str, Any]],
    corpus: str,
    llm_backend: str,
    model: str,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write CSV per-item results and JSON aggregate report.

    Args:
        results: List of per-item result dicts
        corpus: Corpus name
        llm_backend: Backend used
        model: Model used
        output_dir: Directory for output files

    Returns:
        (csv_path, json_path, report_dict)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-item CSV
    csv_path = output_dir / f"validation_results_{corpus}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "gs_id",
                "question",
                "chunk_id",
                "cr_pass",
                "cr_reason",
                "af_pass",
                "af_reason",
                "ar_pass",
                "ar_reason",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["gs_id"],
                    r["question"][:200],
                    r["chunk_id"],
                    int(r["cr_pass"]),
                    r["cr_reason"],
                    int(r["af_pass"]),
                    r["af_reason"],
                    int(r["ar_pass"]),
                    r["ar_reason"],
                ]
            )

    # Aggregate statistics
    n = len(results)
    cr_labels = [r["cr_pass"] for r in results]
    af_labels = [r["af_pass"] for r in results]
    ar_labels = [r["ar_pass"] for r in results]

    cr_pass_rate = sum(cr_labels) / n if n else 0.0
    af_pass_rate = sum(af_labels) / n if n else 0.0
    ar_pass_rate = sum(ar_labels) / n if n else 0.0

    all_pass = [r["cr_pass"] and r["af_pass"] and r["ar_pass"] for r in results]
    overall_pass_rate = sum(all_pass) / n if n else 0.0

    flagged = [
        r["gs_id"]
        for r in results
        if not (r["cr_pass"] and r["af_pass"] and r["ar_pass"])
    ]

    # Compute agreement once per criterion (not 9 times)
    cr_agreement = _compute_agreement(cr_labels)
    af_agreement = _compute_agreement(af_labels)
    ar_agreement = _compute_agreement(ar_labels)

    report: dict[str, Any] = {
        "corpus": corpus,
        "n_items": n,
        "llm_judge": f"{llm_backend}:{model}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "context_relevance": {
                "pass_rate": round(cr_pass_rate, 4),
                "n_fail": n - sum(cr_labels),
            },
            "answer_faithfulness": {
                "pass_rate": round(af_pass_rate, 4),
                "n_fail": n - sum(af_labels),
            },
            "answer_relevance": {
                "pass_rate": round(ar_pass_rate, 4),
                "n_fail": n - sum(ar_labels),
            },
        },
        "overall_pass_rate": round(overall_pass_rate, 4),
        "agreement": {
            "cohens_kappa": {
                "cr": cr_agreement["cohens_kappa"],
                "af": af_agreement["cohens_kappa"],
                "ar": ar_agreement["cohens_kappa"],
            },
            "gwets_ac1": {
                "cr": cr_agreement["gwets_ac1"],
                "af": af_agreement["gwets_ac1"],
                "ar": ar_agreement["gwets_ac1"],
            },
            "raw_agreement": {
                "cr": cr_agreement["raw_agreement"],
                "af": af_agreement["raw_agreement"],
                "ar": ar_agreement["raw_agreement"],
            },
        },
        "flagged_items": flagged,
    }

    json_path = output_dir / f"validation_report_{corpus}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return csv_path, json_path, report


def validate_gs(
    corpus: str = "fr",
    llm_backend: str = "mock",
    model: str = "mistral:latest",
    max_items: int = 0,
    output_dir: Path | None = None,
    gs_path: Path | None = None,
) -> dict[str, Any]:
    """Validate GS items using LLM-as-judge on 3 criteria.

    Args:
        corpus: Either 'fr' or 'intl'
        llm_backend: Backend ('mock', 'ollama', 'groq')
        model: Model name
        max_items: Limit items (0=all)
        output_dir: Output directory
        gs_path: Override GS file path

    Returns:
        Validation report dict
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    items = _load_gs_items(corpus, gs_path)
    if max_items > 0:
        items = items[:max_items]

    print(f"GS Validation ({llm_backend}:{model})")
    print(f"  Items: {len(items)}")
    print(f"  Criteria: {CRITERIA}")
    print(f"  Total LLM calls: {len(items) * 3}")

    results: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        cr_pass, cr_reason = _judge_item(
            item["question"],
            item["chunk_text"],
            item["expected_answer"],
            "context_relevance",
            llm_backend,
            model,
        )
        af_pass, af_reason = _judge_item(
            item["question"],
            item["chunk_text"],
            item["expected_answer"],
            "answer_faithfulness",
            llm_backend,
            model,
        )
        ar_pass, ar_reason = _judge_item(
            item["question"],
            item["chunk_text"],
            item["expected_answer"],
            "answer_relevance",
            llm_backend,
            model,
        )

        results.append(
            {
                "gs_id": item["gs_id"],
                "question": item["question"],
                "chunk_id": item["chunk_id"],
                "cr_pass": cr_pass,
                "cr_reason": cr_reason,
                "af_pass": af_pass,
                "af_reason": af_reason,
                "ar_pass": ar_pass,
                "ar_reason": ar_reason,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(items)}")

    csv_path, json_path, report = _generate_report(
        results,
        corpus,
        llm_backend,
        model,
        output_dir,
    )

    print("\nResults saved:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")

    # Print summary
    print(f"\n{'='*50}")
    print("GS VALIDATION SUMMARY")
    print(f"{'='*50}")
    for c in CRITERIA:
        pr = report["metrics"][c]["pass_rate"]
        nf = report["metrics"][c]["n_fail"]
        print(f"  {c}: {pr:.1%} pass ({nf} failures)")
    print(f"  Overall: {report['overall_pass_rate']:.1%}")
    print(f"  Flagged items: {len(report['flagged_items'])}")

    return report


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Validate Gold Standard by LLM-as-judge (3 criteria)"
    )
    parser.add_argument(
        "--corpus",
        choices=["fr", "intl"],
        default="fr",
        help="Corpus (default: fr)",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "ollama", "groq"],
        default="mock",
        help="LLM backend (default: mock)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral:latest",
        help="Model name (default: mistral:latest)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Limit items for testing (0=all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    args = parser.parse_args()

    report = validate_gs(
        corpus=args.corpus,
        llm_backend=args.backend,
        model=args.model,
        max_items=args.max_items,
        output_dir=args.output_dir,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
