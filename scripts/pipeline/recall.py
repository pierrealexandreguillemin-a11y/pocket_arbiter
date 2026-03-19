"""Recall measurement: page-level recall@k and MRR on GS questions."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from scripts.pipeline.recall_report import write_json, write_markdown

logger = logging.getLogger(__name__)


def load_gs(gs_path: Path | str) -> list[dict]:
    """Load GS and return answerable questions with normalized fields.

    Args:
        gs_path: Path to gold_standard JSON file.

    Returns:
        List of dicts with keys: id, question, expected_docs,
        expected_pages, expected_pairs, reasoning_class, difficulty.
    """
    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)

    questions = []
    for q in gs["questions"]:
        if q["content"]["is_impossible"]:
            continue
        prov = q["provenance"]
        clf = q["classification"]
        expected = list(zip(prov["docs"], prov["pages"], strict=True))
        questions.append(
            {
                "id": q["id"],
                "question": q["content"]["question"],
                "expected_docs": prov["docs"],
                "expected_pages": prov["pages"],
                "expected_pairs": expected,
                "reasoning_class": clf.get("reasoning_class", "unknown"),
                "difficulty": clf.get("difficulty", 0.5),
            }
        )
    return questions


def page_match(
    expected: list[tuple[str, int]],
    retrieved: list[tuple[str, int | None]],
) -> bool:
    """Check if any retrieved (source, page) matches expected.

    Args:
        expected: [(source, page), ...] from GS provenance.
        retrieved: [(source, page), ...] from search results.

    Returns:
        True if at least one match.
    """
    expected_set = set(expected)
    return any(pair in expected_set for pair in retrieved)


def evaluate_question(
    question: dict,
    retrieved_pages: list[list[tuple[str, int | None]]],
) -> dict:
    """Evaluate a single question against retrieved contexts.

    Args:
        question: GS question with expected_pairs, reasoning_class, difficulty.
        retrieved_pages: For each context (ranked), list of (source, page) pairs.

    Returns:
        Dict with hit@1, hit@3, hit@5, hit@10, rank, reasoning_class, difficulty.
    """
    expected = question["expected_pairs"]
    rank = 0
    for i, context_pages in enumerate(retrieved_pages):
        if page_match(expected, context_pages):
            rank = i + 1
            break

    return {
        "id": question["id"],
        "hit@1": rank == 1,
        "hit@3": 1 <= rank <= 3,
        "hit@5": 1 <= rank <= 5,
        "hit@10": 1 <= rank <= 10,
        "rank": rank,
        "reasoning_class": question["reasoning_class"],
        "difficulty": question["difficulty"],
    }


def _difficulty_bucket(difficulty: float) -> str:
    """Map difficulty float to bucket name."""
    if difficulty < 0.33:
        return "easy"
    if difficulty < 0.66:
        return "medium"
    return "hard"


def _aggregate(results: list[dict]) -> dict:
    """Aggregate hit rates and MRR from a list of eval results."""
    if not results:
        return {
            "count": 0,
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "mrr": 0.0,
        }
    n = len(results)
    return {
        "count": n,
        "recall@1": sum(r["hit@1"] for r in results) / n,
        "recall@3": sum(r["hit@3"] for r in results) / n,
        "recall@5": sum(r["hit@5"] for r in results) / n,
        "recall@10": sum(r["hit@10"] for r in results) / n,
        "mrr": sum((1 / r["rank"] if r["rank"] > 0 else 0) for r in results) / n,
    }


def compute_metrics(results: list[dict]) -> dict:
    """Compute global and segmented recall metrics.

    Args:
        results: List of evaluate_question outputs.

    Returns:
        Dict with global, segments.reasoning_class, segments.difficulty.
    """
    global_metrics = _aggregate(results)

    by_class: dict[str, list[dict]] = {}
    for r in results:
        by_class.setdefault(r["reasoning_class"], []).append(r)

    by_diff: dict[str, list[dict]] = {}
    for r in results:
        bucket = _difficulty_bucket(r["difficulty"])
        by_diff.setdefault(bucket, []).append(r)

    return {
        "global": global_metrics,
        "segments": {
            "reasoning_class": {k: _aggregate(v) for k, v in sorted(by_class.items())},
            "difficulty": {k: _aggregate(v) for k, v in sorted(by_diff.items())},
        },
    }


def error_analysis(
    results: list[dict],
    questions: list[dict],
    n: int = 20,
) -> list[dict]:
    """Extract top-n failure cases for diagnostic.

    Args:
        results: List of evaluate_question outputs.
        questions: Original question dicts (for text/expected).
        n: Number of failures to return.

    Returns:
        List of failure dicts with question text and expected info.
    """
    q_by_id = {q["id"]: q for q in questions}
    failures = [r for r in results if not r["hit@10"]][:n]
    return [
        {
            "id": r["id"],
            "question": q_by_id[r["id"]]["question"][:80],
            "expected_docs": q_by_id[r["id"]]["expected_docs"],
            "expected_pages": q_by_id[r["id"]]["expected_pages"],
            "reasoning_class": r["reasoning_class"],
            "difficulty": r["difficulty"],
            "hit@10": False,
        }
        for r in failures
    ]


def _get_context_pages(
    conn: sqlite3.Connection,
    matched_ids: list[str],
) -> list[tuple[str, int | None]]:
    """Lookup (source, page) for child or table_summary IDs."""
    pages = []
    for mid in matched_ids:
        row = conn.execute(
            "SELECT source, page FROM children WHERE id = ?", (mid,)
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT source, page FROM table_summaries WHERE id = ?", (mid,)
            ).fetchone()
        if row:
            pages.append((row[0], row[1]))
    return pages


def run_recall(
    db_path: Path | str,
    gs_path: Path | str,
    output_dir: Path | str = "data/benchmarks",
) -> dict:
    """Run full recall measurement and write reports.

    Args:
        db_path: Path to corpus_v2_fr.db.
        gs_path: Path to gold_standard JSON.
        output_dir: Directory for output files.

    Returns:
        Full results dict (same as JSON output).
    """
    from scripts.pipeline.indexer import DEFAULT_MODEL_ID, load_model
    from scripts.pipeline.search import search

    output_dir = Path(output_dir)
    questions = load_gs(gs_path)
    logger.info("Loaded %d answerable questions", len(questions))

    model = load_model()
    logger.info("Model loaded: %s", DEFAULT_MODEL_ID)

    conn = sqlite3.connect(str(db_path))
    results = []

    try:
        for i, q in enumerate(questions):
            sr = search(db_path, q["question"], model=model)

            retrieved_pages = []
            for ctx in sr.contexts:
                ctx_pages = _get_context_pages(conn, ctx.children_matched)
                retrieved_pages.append(ctx_pages)

            result = evaluate_question(q, retrieved_pages)
            results.append(result)

            if (i + 1) % 50 == 0:
                logger.info("Progress: %d/%d", i + 1, len(questions))
    finally:
        conn.close()

    logger.info("Evaluation complete: %d questions", len(results))

    metrics = compute_metrics(results)
    errors = error_analysis(results, questions, n=20)

    data = {
        "metadata": {
            "generated": datetime.now(tz=timezone.utc).isoformat(),
            "pipeline": "hybrid cosine+BM25 RRF, adaptive-k largest-gap",
            "model": DEFAULT_MODEL_ID,
            "db": str(Path(db_path).name),
            "gs_version": "9.0.0",
            "match_level": "page",
            "settings": {"min_score": 0.005, "max_k": 10, "rrf_k": 60},
            "questions_total": len(questions),
        },
        **metrics,
        "errors": errors,
        "per_question": results,
    }

    write_json(data, output_dir / "recall_baseline.json")
    write_markdown(data, output_dir / "recall_baseline.md")
    logger.info("Reports written to %s", output_dir)

    r5 = metrics["global"]["recall@5"]
    logger.info("recall@5 = %.1f%%", r5 * 100)

    return data
