"""
Filter weak questions from synthetic triplets.

Based on answerability analysis:
- 49.1% are questions on numeric tables (low semantic match)
- 8.6% are potential hallucinations (answer not in chunk)
- 1.8% are vague questions

This script filters and classifies weak questions for quality improvement.

Usage:
    python scripts/pipeline/filter_weak_questions.py
    python scripts/pipeline/filter_weak_questions.py --threshold 0.35
"""

import json
from collections import Counter


def load_data():
    """Load questions and answerability report."""
    with open(
        "data/synthetic_triplets/synthetic_triplets_ffe_final.json", encoding="utf-8"
    ) as f:
        questions = json.load(f)

    with open(
        "data/synthetic_triplets/answerability_report.json", encoding="utf-8"
    ) as f:
        report = json.load(f)

    with open("corpus/processed/chunks_for_embedding_fr.json", encoding="utf-8") as f:
        data = json.load(f)
    chunks = {c["id"]: c["text"] for c in data["chunks"]}

    return questions, report, chunks


def classify_weak_question(question: dict, chunk_text: str) -> str:
    """
    Classify a weak question by problem type.

    Returns:
        'table_numeric': Question on numeric table
        'hallucination': Keywords missing from chunk
        'vague': Too vague
        'example_names': Based on fictional example
        'ok': Actually seems OK
    """
    q_text = question["question"].lower()

    # Check for numeric tables
    if "|" in chunk_text:
        lines = chunk_text.split("\n")
        table_lines = [line for line in lines if "|" in line]
        # If >50% table content and contains numbers
        if len(table_lines) > len(lines) * 0.3:
            digit_count = sum(1 for c in chunk_text if c.isdigit())
            if digit_count > 50:
                return "table_numeric"

    # Check for example with fictional names
    names = ["eric", "fatou", "claude", "isidore", "guy", "béatrice", "beatrice"]
    if any(name in chunk_text.lower() for name in names):
        if any(name in q_text for name in names):
            return "example_names"

    # Check for vague questions
    vague_patterns = ["que se passe-t-il", "comment faire", "qu'est-ce qui"]
    if any(p in q_text for p in vague_patterns) and len(q_text) < 50:
        return "vague"

    # Check for potential hallucination (keywords missing)
    keywords = [w for w in q_text.split() if len(w) > 5]
    stopwords = {
        "quelle",
        "quels",
        "quelles",
        "comment",
        "pourquoi",
        "combien",
        "joueur",
        "joueuse",
        "partie",
        "tournoi",
        "arbitre",
    }
    keywords = [k for k in keywords if k not in stopwords]

    if keywords:
        chunk_lower = chunk_text.lower()
        matches = sum(1 for kw in keywords if kw in chunk_lower)
        if len(keywords) >= 3 and matches / len(keywords) < 0.2:
            return "hallucination"

    return "ok"


def filter_questions(
    questions: list[dict],
    report: dict,
    chunks: dict[str, str],
    similarity_threshold: float = 0.35,
) -> tuple[list[dict], dict]:
    """
    Filter questions based on answerability and classification.

    Returns:
        Tuple of (filtered_questions, stats)
    """
    # Index weak questions by chunk_id + question
    weak_set = {
        (q["chunk_id"], q["question"]) for q in report["low_similarity_questions"]
    }

    # Get similarity scores
    weak_sims = {
        (q["chunk_id"], q["question"]): q["similarity"]
        for q in report["low_similarity_questions"]
    }

    kept = []
    removed = {
        "table_numeric": [],
        "hallucination": [],
        "vague": [],
        "example_names": [],
        "low_similarity": [],
    }

    for q in questions:
        key = (q["chunk_id"], q["question"])

        if key in weak_set:
            sim = weak_sims.get(key, 0.5)
            chunk_text = chunks.get(q["chunk_id"], "")

            # Very low similarity -> always remove
            if sim < similarity_threshold:
                classification = classify_weak_question(q, chunk_text)

                if classification == "ok":
                    classification = "low_similarity"

                removed[classification].append(
                    {
                        "question": q["question"],
                        "chunk_id": q["chunk_id"],
                        "similarity": sim,
                        "category": q.get("category"),
                    }
                )
                continue

        kept.append(q)

    stats = {
        "original": len(questions),
        "kept": len(kept),
        "removed_total": len(questions) - len(kept),
        "removed_by_type": {k: len(v) for k, v in removed.items()},
        "removal_rate": (len(questions) - len(kept)) / len(questions) * 100,
    }

    return kept, stats, removed


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Filter weak questions")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Similarity threshold (default: 0.35)",
    )
    parser.add_argument(
        "--output",
        default="data/synthetic_triplets/synthetic_triplets_filtered.json",
        help="Output file path",
    )

    args = parser.parse_args()

    print("Loading data...")
    questions, report, chunks = load_data()

    print(f"Original questions: {len(questions)}")
    print(f"Similarity threshold: {args.threshold}")

    print("\nFiltering...")
    kept, stats, removed = filter_questions(questions, report, chunks, args.threshold)

    print("\n" + "=" * 60)
    print("FILTERING RESULTS")
    print("=" * 60)
    print(f"Original: {stats['original']}")
    print(f"Kept: {stats['kept']}")
    print(f"Removed: {stats['removed_total']} ({stats['removal_rate']:.1f}%)")

    print("\nRemoved by type:")
    for type_name, count in sorted(
        stats["removed_by_type"].items(), key=lambda x: -x[1]
    ):
        if count > 0:
            print(f"  {type_name}: {count}")

    # Save filtered questions
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(f"\nFiltered questions saved: {args.output}")

    # Save removed questions for review
    removed_path = args.output.replace(".json", "_removed.json")
    with open(removed_path, "w", encoding="utf-8") as f:
        json.dump(removed, f, ensure_ascii=False, indent=2)
    print(f"Removed questions saved: {removed_path}")

    # Category distribution after filtering
    print("\n" + "=" * 60)
    print("CATEGORY DISTRIBUTION (after filtering)")
    print("=" * 60)
    cats = Counter(q["category"] for q in kept)
    for cat, count in cats.most_common():
        print(f"  {cat}: {count} ({count / len(kept) * 100:.1f}%)")


if __name__ == "__main__":
    main()
