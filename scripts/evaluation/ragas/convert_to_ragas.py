"""Convert gold standard data to RAGAS evaluation format.

RAGAS paper (arXiv:2309.15217): question, answer, contexts, ground_truth.

ISO Reference: ISO 42001 A.7.3 - Traceability
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Path constants
BASE_DIR = Path(__file__).parent.parent.parent.parent
TESTS_DATA_DIR = BASE_DIR / "tests" / "data"
CORPUS_DIR = BASE_DIR / "corpus" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "evaluation" / "ragas"


def load_gold_standard(path: Path | None = None) -> dict[str, Any]:
    """Load gold standard questions (Schema v2).

    Args:
        path: Path to GS file. Defaults to gs_scratch_v1.json.

    Returns:
        Gold standard data dict
    """
    if path is None:
        path = TESTS_DATA_DIR / "gs_scratch_v1.json"

    if not path.exists():
        # Fallback to gold_standard_fr.json
        path = TESTS_DATA_DIR / "gold_standard_fr.json"

    if not path.exists():
        raise FileNotFoundError(f"Gold standard not found: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_chunks(corpus: str = "fr") -> dict[str, dict[str, Any]]:
    """Load chunks indexed by ID.

    Args:
        corpus: Either 'fr' or 'intl'

    Returns:
        Dict mapping chunk_id to chunk data
    """
    path = CORPUS_DIR / f"chunks_mode_b_{corpus}.json"
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return {chunk["id"]: chunk for chunk in data.get("chunks", [])}


def convert_to_ragas(
    gs_path: Path | None = None,
    corpus: str = "fr",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Convert gold standard to RAGAS JSONL format.

    Creates ragas_evaluation.jsonl with:
    - question: The question text
    - answer: Empty (to be filled by RAG system)
    - contexts: [chunk_text] from expected_chunk_id
    - ground_truth: expected_answer from GS

    Args:
        gs_path: Path to GS file
        corpus: Corpus for chunks
        output_dir: Output directory

    Returns:
        Conversion stats
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    gs = load_gold_standard(gs_path)
    chunks = load_chunks(corpus)
    questions = gs.get("questions", [])

    # Filter to answerable, validated questions
    # Support both Schema V2 (nested) and V1 (flat) field names
    answerable = []
    for q in questions:
        chunk_id = q.get("provenance", {}).get("chunk_id") or q.get("expected_chunk_id")
        status = q.get("validation", {}).get("status", "")
        hard_type = q.get("classification", {}).get("hard_type") or q.get(
            "metadata", {}
        ).get("hard_type", "ANSWERABLE")
        if chunk_id and status == "VALIDATED" and hard_type == "ANSWERABLE":
            answerable.append(q)

    output_path = output_dir / "ragas_evaluation.jsonl"
    n_written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for q in answerable:
            chunk_id = q.get("provenance", {}).get("chunk_id") or q["expected_chunk_id"]
            chunk = chunks.get(chunk_id)
            if not chunk:
                continue

            # ground_truth = expected answer, NOT the question
            # Schema V2: content.expected_answer; V1: expected_answer
            expected_answer = q.get("content", {}).get("expected_answer") or q.get(
                "expected_answer", ""
            )
            if not expected_answer:
                expected_answer = q.get("metadata", {}).get("expected_answer", "")

            # Schema V2: content.question; V1: question
            question = q.get("content", {}).get("question") or q.get("question", "")

            record = {
                "question": question,
                "answer": "",  # To be filled by RAG system
                "contexts": [chunk["text"]],
                "ground_truth": expected_answer,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Converted {n_written} questions to RAGAS format: {output_path}")

    return {
        "output_path": str(output_path),
        "total_questions": len(questions),
        "answerable": len(answerable),
        "written": n_written,
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Convert gold standard to RAGAS evaluation format"
    )
    parser.add_argument(
        "--gs-path",
        type=Path,
        default=None,
        help="Path to gold standard file (default: tests/data/gs_scratch_v1.json)",
    )
    parser.add_argument(
        "--corpus",
        choices=["fr", "intl"],
        default="fr",
        help="Corpus for chunks (default: fr)",
    )
    args = parser.parse_args()

    convert_to_ragas(gs_path=args.gs_path, corpus=args.corpus)


if __name__ == "__main__":
    main()
