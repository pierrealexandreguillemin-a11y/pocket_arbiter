"""Prepare semantic audit batches for gold standard validation.

This script extracts chunk text for each question with expected_chunk_id
and divides them into batches for parallel validation by haiku agents.

ISO 42001 A.6.2.2: Provenance verification
ISO 29119-3: Test data validation
"""

import json
import sqlite3
from pathlib import Path
from typing import Any


def load_gold_standard(path: Path) -> list[dict[str, Any]]:
    """Load gold standard questions from JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", [])


def get_chunk_text(db_path: Path, chunk_id: str) -> str | None:
    """Retrieve chunk text from SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT text FROM chunks WHERE id = ?", (chunk_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def extract_answerable_questions(
    questions: list[dict[str, Any]], db_path: Path, corpus: str
) -> list[dict[str, Any]]:
    """Extract questions with expected_chunk_id and fetch chunk text."""
    results = []

    for q in questions:
        chunk_id = q.get("expected_chunk_id")
        if not chunk_id:
            continue

        # Skip adversarial questions (they don't have valid chunk_id)
        hard_type = q.get("metadata", {}).get("hard_type", "ANSWERABLE")
        if hard_type != "ANSWERABLE":
            continue

        chunk_text = get_chunk_text(db_path, chunk_id)
        if not chunk_text:
            print(f"Warning: Chunk not found for {q['id']}: {chunk_id}")
            continue

        results.append(
            {
                "id": q["id"],
                "question": q["question"],
                "expected_chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "corpus": corpus,
                "expected_docs": q.get("expected_docs", []),
                "keywords": q.get("keywords", []),
            }
        )

    return results


def divide_into_batches(
    fr_questions: list[dict[str, Any]],
    intl_questions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Divide questions into 6 batches for parallel processing."""
    batches = {}

    # FR batches (5 batches of ~40-53 questions each)
    fr_sorted = sorted(fr_questions, key=lambda x: x["id"])
    _batch_size = 40  # noqa: F841 — documents intended slice size

    batches["batch_1"] = fr_sorted[0:40]  # FR-Q01 to ~FR-Q40
    batches["batch_2"] = fr_sorted[40:80]  # FR-Q41 to ~FR-Q80
    batches["batch_3"] = fr_sorted[80:120]  # FR-Q81 to ~FR-Q120
    batches["batch_4"] = fr_sorted[120:160]  # FR-Q121 to ~FR-Q160
    batches["batch_5"] = fr_sorted[160:]  # FR-Q161 to end

    # INTL batch (1 batch for all INTL questions)
    batches["batch_6"] = sorted(intl_questions, key=lambda x: x["id"])

    return batches


def main() -> None:
    """Main function to prepare audit batches."""
    base_path = Path(__file__).parent.parent.parent

    # Paths
    fr_gold = base_path / "tests" / "data" / "gold_standard_fr.json"
    intl_gold = base_path / "tests" / "data" / "gold_standard_intl.json"
    fr_db = base_path / "corpus" / "processed" / "corpus_mode_b_fr.db"
    intl_db = base_path / "corpus" / "processed" / "corpus_mode_b_intl.db"
    output_dir = base_path / "data" / "audit_batches"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and extract questions
    print("Loading gold standard files...")
    fr_questions = load_gold_standard(fr_gold)
    intl_questions = load_gold_standard(intl_gold)

    print(f"FR: {len(fr_questions)} total questions")
    print(f"INTL: {len(intl_questions)} total questions")

    # Extract answerable questions with chunk text
    print("\nExtracting answerable questions with chunk text...")
    fr_answerable = extract_answerable_questions(fr_questions, fr_db, "fr")
    intl_answerable = extract_answerable_questions(intl_questions, intl_db, "intl")

    print(f"FR ANSWERABLE with chunk_id: {len(fr_answerable)}")
    print(f"INTL ANSWERABLE with chunk_id: {len(intl_answerable)}")

    # Divide into batches
    print("\nDividing into 6 batches...")
    batches = divide_into_batches(fr_answerable, intl_answerable)

    # Save batches
    summary = {}
    for batch_name, batch_data in batches.items():
        output_file = output_dir / f"{batch_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "batch_name": batch_name,
                    "count": len(batch_data),
                    "questions": batch_data,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        summary[batch_name] = {
            "count": len(batch_data),
            "first_id": batch_data[0]["id"] if batch_data else None,
            "last_id": batch_data[-1]["id"] if batch_data else None,
        }
        print(f"  {batch_name}: {len(batch_data)} questions -> {output_file}")

    # Save summary
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_questions": len(fr_answerable) + len(intl_answerable),
                "batches": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSummary saved to {summary_file}")
    print(f"Total questions for audit: {len(fr_answerable) + len(intl_answerable)}")


if __name__ == "__main__":
    main()
