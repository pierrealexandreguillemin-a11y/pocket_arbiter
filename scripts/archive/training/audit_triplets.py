"""
Audit synthetic triplets for quality and conformity.

ISO 42001: Zero hallucination tolerance
ISO 29119: Test coverage

Validation layers:
1. Statistical (automatic) - duplicates, distribution, empty
2. Human-in-the-loop (10% sample) - manual review

Usage:
    python scripts/pipeline/audit_triplets.py
    python scripts/pipeline/audit_triplets.py --sample 100
"""

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AuditReport:
    """Audit report structure."""

    total_questions: int
    unique_questions: int
    duplicates: int
    categories: dict[str, int]
    difficulties: dict[str, int]
    chunks_covered: int
    avg_question_length: float
    short_questions: list[dict]  # < 20 chars
    long_questions: list[dict]  # > 200 chars
    missing_fields: list[dict]
    sample_for_review: list[dict]


def load_triplets(path: str) -> list[dict]:
    """Load synthetic triplets from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def detect_duplicates(questions: list[dict]) -> tuple[int, list[str]]:
    """Detect duplicate questions."""
    seen = {}
    duplicates = []

    for q in questions:
        text = q.get("question", "").strip().lower()
        if text in seen:
            duplicates.append(text)
        else:
            seen[text] = True

    return len(duplicates), duplicates[:10]  # Return first 10


def analyze_distribution(questions: list[dict]) -> tuple[dict, dict]:
    """Analyze category and difficulty distribution."""
    categories = Counter(q.get("category", "unknown") for q in questions)
    difficulties = Counter(q.get("difficulty", "unknown") for q in questions)
    return dict(categories), dict(difficulties)


def find_quality_issues(questions: list[dict]) -> tuple[list, list, list]:
    """Find questions with potential quality issues."""
    short = []
    long = []
    missing = []

    for q in questions:
        text = q.get("question", "")

        # Short questions (< 20 chars)
        if len(text) < 20:
            short.append(q)

        # Long questions (> 200 chars)
        if len(text) > 200:
            long.append(q)

        # Missing required fields
        if not q.get("question") or not q.get("category") or not q.get("chunk_id"):
            missing.append(q)

    return short, long, missing


def sample_for_review(
    questions: list[dict], sample_size: int = 100, stratified: bool = True
) -> list[dict]:
    """
    Sample questions for human review.

    Args:
        questions: All questions
        sample_size: Number to sample
        stratified: If True, sample proportionally from each category
    """
    if not stratified:
        return random.sample(questions, min(sample_size, len(questions)))

    # Stratified sampling by category
    by_category = {}
    for q in questions:
        cat = q.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(q)

    # Calculate proportions
    total = len(questions)
    sample = []

    for cat, cat_questions in by_category.items():
        proportion = len(cat_questions) / total
        n_sample = max(1, int(sample_size * proportion))
        sample.extend(random.sample(cat_questions, min(n_sample, len(cat_questions))))

    return sample[:sample_size]


def run_audit(
    triplets_path: str,
    sample_size: int = 100,
    output_dir: str = "data/synthetic_triplets",
) -> AuditReport:
    """
    Run full audit on synthetic triplets.

    Args:
        triplets_path: Path to triplets JSON
        sample_size: Number of questions to sample for review
        output_dir: Directory for audit outputs
    """
    print("=" * 60)
    print("AUDIT TRIPLETS SYNTHETIQUES")
    print("=" * 60)

    # Load data
    questions = load_triplets(triplets_path)
    print(f"\nTotal questions: {len(questions)}")

    # 1. Duplicates
    n_duplicates, dup_examples = detect_duplicates(questions)
    unique = len(questions) - n_duplicates
    print(f"\nDuplicates: {n_duplicates} ({n_duplicates / len(questions) * 100:.1f}%)")
    if dup_examples:
        print(f"  Exemples: {dup_examples[:3]}")

    # 2. Distribution
    categories, difficulties = analyze_distribution(questions)
    print("\nCategories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count / len(questions) * 100:.1f}%)")

    print("\nDifficultes:")
    for diff, count in sorted(difficulties.items(), key=lambda x: -x[1]):
        print(f"  {diff}: {count} ({count / len(questions) * 100:.1f}%)")

    # 3. Chunks coverage
    chunk_ids = set(q.get("chunk_id") for q in questions)
    print(f"\nChunks couverts: {len(chunk_ids)}")

    # 4. Quality issues
    short, long, missing = find_quality_issues(questions)
    avg_len = sum(len(q.get("question", "")) for q in questions) / len(questions)
    print("\nQualite:")
    print(f"  Longueur moyenne: {avg_len:.1f} chars")
    print(f"  Questions courtes (<20): {len(short)}")
    print(f"  Questions longues (>200): {len(long)}")
    print(f"  Champs manquants: {len(missing)}")

    # 5. Sample for human review
    sample = sample_for_review(questions, sample_size, stratified=True)
    print(f"\nEchantillon pour review: {len(sample)} questions")

    # Save sample for review
    output_path = Path(output_dir)
    sample_path = output_path / "audit_sample_review.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"  Sauvegarde: {sample_path}")

    # Save quality issues
    issues_path = output_path / "audit_quality_issues.json"
    with open(issues_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "short_questions": short[:50],
                "long_questions": long[:50],
                "missing_fields": missing[:50],
                "duplicate_examples": dup_examples,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  Issues: {issues_path}")

    # Create report
    report = AuditReport(
        total_questions=len(questions),
        unique_questions=unique,
        duplicates=n_duplicates,
        categories=categories,
        difficulties=difficulties,
        chunks_covered=len(chunk_ids),
        avg_question_length=avg_len,
        short_questions=short,
        long_questions=long,
        missing_fields=missing,
        sample_for_review=sample,
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("RESUME AUDIT")
    print("=" * 60)

    # Quality score (simple heuristic)
    issues_count = n_duplicates + len(short) + len(missing)
    quality_score = max(0, 100 - (issues_count / len(questions) * 100))
    print(f"Score qualite: {quality_score:.1f}%")

    # Conformity checks
    print("\nConformite ISO 42001:")
    print(
        f"  [{'OK' if n_duplicates < len(questions) * 0.05 else 'XX'}] Duplicates < 5%"
    )
    print(f"  [{'OK' if len(missing) == 0 else 'XX'}] Tous les champs requis")
    print(f"  [{'OK' if len(categories) >= 3 else 'XX'}] Diversite categories (>=3)")
    print("  [--] Answerability (necessite embedding check)")
    print("  [--] Human review (echantillon pret)")

    return report


def print_sample_for_review(sample: list[dict], limit: int = 10) -> None:
    """Print sample questions for quick visual review."""
    print(f"\n{'=' * 60}")
    print(f"ECHANTILLON REVIEW ({limit} questions)")
    print("=" * 60)

    for i, q in enumerate(sample[:limit]):
        print(f"\n[{i + 1}] {q.get('category', '?')} | {q.get('difficulty', '?')}")
        print(f"    Q: {q.get('question', '?')}")
        print(f"    Chunk: {q.get('chunk_id', '?')[:30]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audit synthetic triplets")
    parser.add_argument(
        "--input",
        default="data/synthetic_triplets/synthetic_triplets_ffe_final.json",
        help="Path to triplets JSON",
    )
    parser.add_argument(
        "--sample", type=int, default=100, help="Sample size for human review"
    )
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Print sample questions for quick review",
    )

    args = parser.parse_args()

    # Run audit
    report = run_audit(args.input, args.sample)

    # Optionally show sample
    if args.show_sample:
        print_sample_for_review(report.sample_for_review, limit=10)
