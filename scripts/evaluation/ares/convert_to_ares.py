"""Convert gold standard data to ARES TSV format.

ISO Reference: ISO 42001 A.7.3 - Traceability
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


# Path constants
BASE_DIR = Path(__file__).parent.parent.parent.parent
TESTS_DATA_DIR = BASE_DIR / "tests" / "data"
CORPUS_DIR = BASE_DIR / "corpus" / "processed"
DATA_TRAINING_DIR = BASE_DIR / "data" / "training"
OUTPUT_DIR = BASE_DIR / "data" / "evaluation" / "ares"


def load_gold_standard(corpus: str) -> dict[str, Any]:
    """Load gold standard questions for a corpus.

    Args:
        corpus: Either 'fr' or 'intl'

    Returns:
        Gold standard data dict
    """
    path = TESTS_DATA_DIR / f"gold_standard_{corpus}.json"
    if not path.exists():
        raise FileNotFoundError(f"Gold standard not found: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_chunks(corpus: str) -> dict[str, dict[str, Any]]:
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


def load_triplets(corpus: str) -> list[dict[str, str]]:
    """Load gold triplets for positive answers.

    Args:
        corpus: Either 'fr' for mode_b triplets

    Returns:
        List of triplet dicts with anchor, positive, negative
    """
    path = DATA_TRAINING_DIR / "gold_triplets_mode_b.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Triplets file not found: {path}")

    triplets = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                triplets.append(json.loads(line))

    return triplets


def _get_triplet_by_question(
    triplets: list[dict[str, str]], question: str
) -> dict[str, str] | None:
    """Find triplet matching a question.

    Args:
        triplets: List of triplet dicts
        question: Question text to match

    Returns:
        Matching triplet or None
    """
    for triplet in triplets:
        if triplet.get("anchor") == question:
            return triplet
    return None


def _get_negative_chunks(
    chunks: dict[str, dict[str, Any]],
    positive_chunk_id: str,
    n: int = 3,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Select random negative chunks from different pages.

    Args:
        chunks: Dict of all chunks
        positive_chunk_id: ID of the positive chunk to exclude
        n: Number of negatives to select
        seed: Random seed for reproducibility

    Returns:
        List of negative chunk dicts
    """
    if seed is not None:
        random.seed(seed)

    positive_chunk = chunks.get(positive_chunk_id)
    if not positive_chunk:
        return []

    positive_source = positive_chunk.get("source", "")
    positive_pages = set(positive_chunk.get("pages", []))

    # Filter chunks: different page or different source
    candidates = [
        c
        for cid, c in chunks.items()
        if cid != positive_chunk_id
        and (
            c.get("source") != positive_source
            or not set(c.get("pages", [])).intersection(positive_pages)
        )
    ]

    if len(candidates) <= n:
        return candidates

    return random.sample(candidates, n)


def convert_gold_standard_to_ares(
    corpus: str = "fr",
    negative_ratio: float = 0.30,
    output_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Path]:
    """Convert gold standard to ARES TSV format.

    Creates three TSV files:
    - gold_label_{corpus}.tsv: Labeled validation set (50+ samples)
    - unlabeled_{corpus}.tsv: Full evaluation set without labels
    - mapping_{corpus}.json: Traceability mapping (ISO 42001)

    Args:
        corpus: Either 'fr' or 'intl'
        negative_ratio: Ratio of negative samples to add (0.30 = 30%)
        output_dir: Output directory (defaults to data/evaluation/ares/)
        seed: Random seed for reproducibility

    Returns:
        Dict with paths to created files
    """
    random.seed(seed)

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    gold_standard = load_gold_standard(corpus)
    chunks = load_chunks(corpus)
    triplets = load_triplets(corpus) if corpus == "fr" else []

    questions = gold_standard.get("questions", [])

    # Filter to answerable questions with expected_chunk_id
    answerable = [
        q
        for q in questions
        if q.get("expected_chunk_id")
        and q.get("validation", {}).get("status") == "VALIDATED"
        and q.get("metadata", {}).get("hard_type", "ANSWERABLE") == "ANSWERABLE"
    ]

    print(f"Found {len(answerable)} answerable questions with expected_chunk_id")

    # Build positive samples
    positive_samples = []
    traceability = []

    for q in answerable:
        chunk_id = q["expected_chunk_id"]
        chunk = chunks.get(chunk_id)
        if not chunk:
            print(f"Warning: chunk {chunk_id} not found, skipping {q['id']}")
            continue

        # Get answer from triplet if available
        triplet = _get_triplet_by_question(triplets, q["question"])
        answer = triplet["positive"] if triplet else chunk["text"]

        positive_samples.append(
            {
                "Query": q["question"],
                "Document": chunk["text"],
                "Answer": answer,
                "Context_Relevance_Label": 1,
                "gs_id": q["id"],
                "chunk_id": chunk_id,
            }
        )

        traceability.append(
            {
                "gs_id": q["id"],
                "chunk_id": chunk_id,
                "source": chunk.get("source"),
                "pages": chunk.get("pages"),
                "label": 1,
            }
        )

    print(f"Created {len(positive_samples)} positive samples")

    # Add negative samples (30% of total, minimum 1 if we have positives)
    n_negatives = int(len(positive_samples) * negative_ratio / (1 - negative_ratio))
    if len(positive_samples) > 0 and n_negatives == 0:
        n_negatives = 1  # Ensure at least 1 negative for small datasets
    negative_samples = []

    # Select random questions and pair with random unrelated chunks
    for _ in range(n_negatives):
        q = random.choice(answerable)
        chunk_id = q["expected_chunk_id"]

        neg_chunks = _get_negative_chunks(chunks, chunk_id, n=1, seed=None)
        if not neg_chunks:
            continue

        neg_chunk = neg_chunks[0]

        negative_samples.append(
            {
                "Query": q["question"],
                "Document": neg_chunk["text"],
                "Answer": "",  # No valid answer for negative
                "Context_Relevance_Label": 0,
                "gs_id": f"{q['id']}_neg",
                "chunk_id": neg_chunk["id"],
            }
        )

        traceability.append(
            {
                "gs_id": f"{q['id']}_neg",
                "chunk_id": neg_chunk["id"],
                "source": neg_chunk.get("source"),
                "pages": neg_chunk.get("pages"),
                "label": 0,
            }
        )

    print(f"Created {len(negative_samples)} negative samples")

    # Combine and shuffle
    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)

    # Write gold_label TSV (labeled validation set - first 50+)
    gold_label_path = output_dir / f"gold_label_{corpus}.tsv"
    _write_tsv(
        gold_label_path,
        all_samples[:60],  # At least 50 for statistical significance
        include_label=True,
    )

    # Write unlabeled TSV (full evaluation set)
    unlabeled_path = output_dir / f"unlabeled_{corpus}.tsv"
    _write_tsv(unlabeled_path, all_samples, include_label=False)

    # Write traceability mapping (ISO 42001)
    mapping_path = output_dir / f"mapping_{corpus}.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "corpus": corpus,
                "version": gold_standard.get("version", "unknown"),
                "total_samples": len(all_samples),
                "positive_count": len(positive_samples),
                "negative_count": len(negative_samples),
                "negative_ratio": len(negative_samples) / len(all_samples)
                if all_samples
                else 0,
                "samples": traceability,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Created files:")
    print(f"  - {gold_label_path}")
    print(f"  - {unlabeled_path}")
    print(f"  - {mapping_path}")

    return {
        "gold_label": gold_label_path,
        "unlabeled": unlabeled_path,
        "mapping": mapping_path,
    }


def _write_tsv(
    path: Path, samples: list[dict[str, Any]], include_label: bool = True
) -> None:
    """Write samples to TSV file.

    Args:
        path: Output path
        samples: List of sample dicts
        include_label: Whether to include Context_Relevance_Label column
    """
    if not samples:
        return

    fieldnames = ["Query", "Document", "Answer"]
    if include_label:
        fieldnames.append("Context_Relevance_Label")

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for sample in samples:
            row = {k: sample[k] for k in fieldnames if k in sample}
            writer.writerow(row)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Convert gold standard to ARES TSV format"
    )
    parser.add_argument(
        "--corpus",
        choices=["fr", "intl"],
        default="fr",
        help="Corpus to convert (default: fr)",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.30,
        help="Ratio of negative samples (default: 0.30)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    convert_gold_standard_to_ares(
        corpus=args.corpus, negative_ratio=args.negative_ratio, seed=args.seed
    )


if __name__ == "__main__":
    main()
