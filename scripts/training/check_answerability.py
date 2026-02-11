"""
Check answerability of synthetic questions via embedding similarity.

ISO 42001: Zero hallucination - each question must be answerable from its chunk.

Principle:
- Encode question and chunk with embedding model
- Compute cosine similarity
- Flag questions with low similarity (< threshold)

Usage:
    python scripts/pipeline/check_answerability.py
    python scripts/pipeline/check_answerability.py --threshold 0.5 --batch-size 64
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.pipeline.utils import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer

    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("Warning: sentence-transformers not installed. Install with:")
    print("  pip install sentence-transformers")


@dataclass
class AnswerabilityReport:
    """Answerability check report."""

    total_questions: int
    checked: int
    passed: int
    failed: int
    pass_rate: float
    avg_similarity: float
    min_similarity: float
    max_similarity: float
    low_similarity_questions: list[dict]


def load_questions(path: str) -> list[dict]:
    """Load synthetic questions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_chunks(path: str) -> dict[str, str]:
    """Load chunks indexed by ID."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, dict) and "chunks" in data:
        chunks = data["chunks"]
    elif isinstance(data, list):
        chunks = data
    else:
        raise ValueError(f"Unknown chunks format: {type(data)}")

    return {c["id"]: c["text"] for c in chunks}


def check_answerability(
    questions: list[dict],
    chunks: dict[str, str],
    model: "SentenceTransformer",
    threshold: float = 0.5,
    batch_size: int = 32,
    max_questions: int | None = None,
) -> AnswerabilityReport:
    """
    Check if questions are answerable from their source chunks.

    Args:
        questions: List of questions with chunk_id
        chunks: Dict mapping chunk_id to chunk text
        model: SentenceTransformer model for encoding
        threshold: Minimum similarity to consider answerable
        batch_size: Batch size for encoding
        max_questions: Limit number of questions (for testing)

    Returns:
        AnswerabilityReport with statistics and flagged questions
    """
    if max_questions:
        questions = questions[:max_questions]

    print(f"Checking {len(questions)} questions...")
    print(f"Threshold: {threshold}")
    try:
        model_name = model.get_config_dict().get("model_name_or_path", "unknown")
    except AttributeError:
        model_name = str(type(model).__name__)
    print(f"Model: {model_name}")

    # Prepare pairs (question, chunk)
    pairs = []
    for q in questions:
        chunk_id = q.get("chunk_id")
        if chunk_id and chunk_id in chunks:
            pairs.append(
                {
                    "question": q["question"],
                    "chunk_text": chunks[chunk_id],
                    "chunk_id": chunk_id,
                    "category": q.get("category", "unknown"),
                    "difficulty": q.get("difficulty", "unknown"),
                }
            )

    if not pairs:
        print("Error: No valid question-chunk pairs found")
        return None

    print(f"Valid pairs: {len(pairs)}")

    # Extract texts for batch encoding
    question_texts = [p["question"] for p in pairs]
    chunk_texts = [p["chunk_text"] for p in pairs]

    # Encode in batches
    print("\nEncoding questions...")
    start_time = time.time()
    question_embeddings = model.encode(
        question_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("Encoding chunks...")
    chunk_embeddings = model.encode(
        chunk_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    encode_time = time.time() - start_time
    print(f"Encoding time: {encode_time:.1f}s")

    # Compute similarities
    print("\nComputing similarities...")
    similarities = []
    low_similarity = []

    for i, pair in enumerate(pairs):
        sim = cosine_similarity(question_embeddings[i], chunk_embeddings[i])
        similarities.append(sim)
        pair["similarity"] = sim

        if sim < threshold:
            low_similarity.append(pair)

    # Statistics
    similarities = np.array(similarities)
    passed = int(np.sum(similarities >= threshold))
    failed = len(pairs) - passed
    pass_rate = passed / len(pairs) * 100

    report = AnswerabilityReport(
        total_questions=len(questions),
        checked=len(pairs),
        passed=passed,
        failed=failed,
        pass_rate=pass_rate,
        avg_similarity=float(np.mean(similarities)),
        min_similarity=float(np.min(similarities)),
        max_similarity=float(np.max(similarities)),
        low_similarity_questions=sorted(low_similarity, key=lambda x: x["similarity"]),
    )

    return report


def print_report(report: AnswerabilityReport, show_failed: int = 10) -> None:
    """Print answerability report."""
    print("\n" + "=" * 60)
    print("ANSWERABILITY CHECK REPORT")
    print("=" * 60)

    print("\nStatistics:")
    print(f"  Total questions: {report.total_questions}")
    print(f"  Checked: {report.checked}")
    print(f"  Passed: {report.passed}")
    print(f"  Failed: {report.failed}")
    print(f"  Pass rate: {report.pass_rate:.1f}%")

    print("\nSimilarity scores:")
    print(f"  Average: {report.avg_similarity:.3f}")
    print(f"  Min: {report.min_similarity:.3f}")
    print(f"  Max: {report.max_similarity:.3f}")

    if report.low_similarity_questions:
        print(
            f"\nLowest similarity questions ({min(show_failed, len(report.low_similarity_questions))}):"
        )
        for i, q in enumerate(report.low_similarity_questions[:show_failed]):
            print(f"\n  [{i + 1}] Similarity: {q['similarity']:.3f}")
            print(f"      Category: {q['category']}")
            print(f"      Q: {q['question'][:100]}...")
            print(f"      Chunk: {q['chunk_id'][:40]}...")

    # Conformity check
    print("\n" + "=" * 60)
    print("CONFORMITE ISO 42001")
    print("=" * 60)

    if report.pass_rate >= 95:
        status = "OK"
        msg = "Excellent - >95% answerable"
    elif report.pass_rate >= 90:
        status = "OK"
        msg = "Bon - >90% answerable"
    elif report.pass_rate >= 80:
        status = "WARN"
        msg = "Acceptable - revue manuelle recommandee"
    else:
        status = "FAIL"
        msg = "Insuffisant - filtrage necessaire"

    print(f"  [{status}] Answerability: {report.pass_rate:.1f}% - {msg}")


def save_report(
    report: AnswerabilityReport,
    output_path: str,
) -> None:
    """Save report to JSON."""
    data = {
        "total_questions": report.total_questions,
        "checked": report.checked,
        "passed": report.passed,
        "failed": report.failed,
        "pass_rate": report.pass_rate,
        "avg_similarity": report.avg_similarity,
        "min_similarity": report.min_similarity,
        "max_similarity": report.max_similarity,
        "low_similarity_questions": report.low_similarity_questions,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check answerability via embedding similarity"
    )
    parser.add_argument(
        "--questions",
        default="data/synthetic_triplets/synthetic_triplets_ffe_final.json",
        help="Path to questions JSON",
    )
    parser.add_argument(
        "--chunks", default="data/chunks/chunks_ffe_v2.json", help="Path to chunks JSON"
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model name",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum similarity threshold (default: 0.5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for encoding"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions (for testing)",
    )
    parser.add_argument(
        "--output",
        default="data/synthetic_triplets/answerability_report.json",
        help="Output report path",
    )

    args = parser.parse_args()

    if not HAS_ST:
        print("Error: sentence-transformers required")
        return

    # Check chunks file exists
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        # Try alternative path
        alt_path = Path("corpus/processed/chunks_for_embedding_fr.json")
        if alt_path.exists():
            args.chunks = str(alt_path)
        else:
            print(f"Error: Chunks file not found: {args.chunks}")
            print("Looking for chunks...")
            from glob import glob

            found = glob("**/chunks*.json", recursive=True)
            if found:
                print(f"Found: {found[:5]}")
            return

    # Load data
    print("Loading questions...")
    questions = load_questions(args.questions)

    print("Loading chunks...")
    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks")

    # Load model
    print(f"\nLoading model: {args.model}")
    model = SentenceTransformer(args.model)

    # Run check
    report = check_answerability(
        questions=questions,
        chunks=chunks,
        model=model,
        threshold=args.threshold,
        batch_size=args.batch_size,
        max_questions=args.max_questions,
    )

    if report:
        print_report(report)
        save_report(report, args.output)


if __name__ == "__main__":
    main()
