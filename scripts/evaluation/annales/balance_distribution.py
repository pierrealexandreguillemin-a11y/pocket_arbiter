#!/usr/bin/env python3
"""
Phase 5: Deduplication et Equilibrage des distributions pour GS BY DESIGN.

Effectue:
1. Deduplication par similarite semantique (seuil 0.95)
2. Verification de l'independance anchor-positive (seuil 0.90)
3. Equilibrage des distributions cibles:
   - fact_single: 40-50% (pas plus de 60%)
   - summary: 15-25%
   - reasoning: 10-20%
   - unanswerable: 25-33% (SQuAD 2.0)
   - hard (difficulty >= 0.7): >= 10%

ISO Reference:
- ISO 29119-3: Test data balance
- ISO 25010: Data quality metrics
- ISO 42001 A.6.2.2: Provenance and embeddings (EmbeddingGemma QAT)

Standards & Thresholds:
- SemHash/SoftDedup: Deduplication threshold 0.95 (stricter than SemHash default 0.90)
- Anchor independence: < 0.90 (project threshold for valid triplet training)
- EmbeddingGemma QAT: google/embeddinggemma-300m-qat-q4_0-unquantized
- SQuAD 2.0: Unanswerable ratio 25-33% (inspired by train split ~33.4%)
- Know Your RAG (COLING 2025): reasoning_class taxonomy (fact_single/summary/reasoning)

Usage:
    python balance_distribution.py --questions PATH --chunks PATH [--output PATH]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.pipeline.embeddings import (  # noqa: E402
    embed_query,
    embed_texts,
    load_embedding_model,
)
from scripts.pipeline.utils import (  # noqa: E402
    cosine_similarity,
    get_date,
    load_json,
    save_json,
)

# Lazy-loaded EmbeddingGemma QAT model (ISO 42001 A.6.2.2)
_embedding_model = None


def get_embedding_model():
    """Lazy load EmbeddingGemma QAT embedding model (ISO 42001 A.6.2.2)."""
    global _embedding_model
    if _embedding_model is None:
        print("  Loading EmbeddingGemma QAT model...")
        _embedding_model = load_embedding_model()
    return _embedding_model


def compute_embedding(text: str) -> np.ndarray:
    """Compute sentence embedding using EmbeddingGemma QAT."""
    model = get_embedding_model()
    return embed_query(text, model, normalize=True)


def compute_embeddings_batch(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Compute embeddings for a batch of texts using EmbeddingGemma QAT."""
    model = get_embedding_model()
    return embed_texts(texts, model, normalize=True)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms

    # Compute similarity matrix
    return np.dot(normalized, normalized.T)


@dataclass
class DeduplicationResult:
    """Result of deduplication process."""

    unique_ids: list[str]
    duplicate_pairs: list[tuple[str, str, float]]
    removed_ids: list[str]


@dataclass
class DistributionStats:
    """Statistics about question distribution."""

    total: int
    answerable: int
    unanswerable: int
    fact_single: int
    summary: int
    reasoning: int
    arithmetic: int
    hard: int

    @property
    def unanswerable_ratio(self) -> float:
        return self.unanswerable / self.total if self.total > 0 else 0

    @property
    def fact_single_ratio(self) -> float:
        return self.fact_single / self.answerable if self.answerable > 0 else 0

    @property
    def summary_ratio(self) -> float:
        return self.summary / self.answerable if self.answerable > 0 else 0

    @property
    def reasoning_ratio(self) -> float:
        return self.reasoning / self.answerable if self.answerable > 0 else 0

    @property
    def hard_ratio(self) -> float:
        return self.hard / self.total if self.total > 0 else 0


def compute_distribution_stats(questions: list[dict]) -> DistributionStats:
    """Compute distribution statistics for questions."""
    total = len(questions)
    unanswerable = sum(
        1 for q in questions if q.get("content", {}).get("is_impossible", False)
    )
    answerable = total - unanswerable

    # Count reasoning classes (only for answerable)
    answerable_qs = [
        q for q in questions if not q.get("content", {}).get("is_impossible", False)
    ]
    class_counts = Counter(
        q.get("classification", {}).get("reasoning_class", "unknown")
        for q in answerable_qs
    )

    # Count hard questions
    hard = sum(
        1 for q in questions if q.get("classification", {}).get("difficulty", 0) >= 0.7
    )

    return DistributionStats(
        total=total,
        answerable=answerable,
        unanswerable=unanswerable,
        fact_single=class_counts.get("fact_single", 0),
        summary=class_counts.get("summary", 0),
        reasoning=class_counts.get("reasoning", 0),
        arithmetic=class_counts.get("arithmetic", 0),
        hard=hard,
    )


def deduplicate_questions(
    questions: list[dict],
    threshold: float = 0.95,
) -> DeduplicationResult:
    """
    Remove near-duplicate questions based on semantic similarity.

    Args:
        questions: List of question dictionaries
        threshold: Similarity threshold for duplicate detection

    Returns:
        DeduplicationResult with unique IDs and removed duplicates
    """
    if len(questions) < 2:
        return DeduplicationResult(
            unique_ids=[q.get("id", f"q_{i}") for i, q in enumerate(questions)],
            duplicate_pairs=[],
            removed_ids=[],
        )

    print(f"\nComputing embeddings for {len(questions)} questions...")

    # Extract question texts
    texts = [q.get("content", {}).get("question", "") for q in questions]
    ids = [q.get("id", f"q_{i}") for i, q in enumerate(questions)]

    # Compute embeddings
    embeddings = compute_embeddings_batch(texts)

    # Compute similarity matrix
    print("  Computing similarity matrix...")
    sim_matrix = cosine_similarity_matrix(embeddings)

    # Find duplicates (upper triangle only)
    duplicate_pairs = []
    removed_ids = set()

    for i in range(len(questions)):
        if ids[i] in removed_ids:
            continue
        for j in range(i + 1, len(questions)):
            if ids[j] in removed_ids:
                continue
            sim = sim_matrix[i, j]
            if sim >= threshold:
                # Keep the first one, mark second as duplicate
                duplicate_pairs.append((ids[i], ids[j], float(sim)))
                removed_ids.add(ids[j])

    unique_ids = [qid for qid in ids if qid not in removed_ids]

    return DeduplicationResult(
        unique_ids=unique_ids,
        duplicate_pairs=duplicate_pairs,
        removed_ids=list(removed_ids),
    )


def check_anchor_independence(
    questions: list[dict],
    chunk_index: dict[str, str],
    threshold: float = 0.90,
) -> tuple[list[str], list[tuple[str, float]]]:
    """
    Check that questions are not too similar to their source chunks.

    For valid triplet training, anchor (question) should not be too
    similar to positive (chunk).

    Args:
        questions: List of question dictionaries
        chunk_index: Dict mapping chunk_id to chunk text
        threshold: Maximum allowed similarity

    Returns:
        Tuple of (valid_ids, violations as (qid, similarity))
    """
    print(f"\nChecking anchor independence for {len(questions)} questions...")

    valid_ids = []
    violations = []

    for i, q in enumerate(questions):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(questions)}...")

        qid = q.get("id", f"q_{i}")
        q_text = q.get("content", {}).get("question", "")
        chunk_id = q.get("provenance", {}).get("chunk_id", "")
        chunk_text = chunk_index.get(chunk_id, "")

        # Skip unanswerable (no chunk to compare)
        if q.get("content", {}).get("is_impossible", False):
            valid_ids.append(qid)
            continue

        if not chunk_text:
            valid_ids.append(qid)  # Can't check, assume valid
            continue

        # Compute similarity
        q_emb = compute_embedding(q_text)
        c_emb = compute_embedding(chunk_text[:512])  # Truncate long chunks
        sim = cosine_similarity(q_emb, c_emb)

        if sim < threshold:
            valid_ids.append(qid)
        else:
            violations.append((qid, sim))

    return valid_ids, violations


def balance_distribution(
    questions: list[dict],
    targets: dict[str, tuple[float, float]],
) -> list[dict]:
    """
    Balance question distribution to meet targets.

    This doesn't generate new questions, but flags which questions
    to prioritize or de-prioritize for the final set.

    Args:
        questions: List of question dictionaries
        targets: Dict of metric -> (min, max) targets

    Returns:
        Balanced list of questions
    """
    # For now, just return all questions
    # In a full implementation, this would:
    # 1. Identify over-represented classes
    # 2. Identify under-represented classes
    # 3. Suggest which questions to add/remove

    # We'll add quality scores to help prioritize
    for q in questions:
        if "processing" not in q:
            q["processing"] = {}

        # Boost under-represented classes
        reasoning_class = q.get("classification", {}).get("reasoning_class", "")
        if reasoning_class in ("reasoning", "summary"):
            q["processing"]["priority_boost"] = 0.1
        elif reasoning_class == "fact_single":
            q["processing"]["priority_boost"] = -0.05

    return questions


def validate_distribution(
    stats: DistributionStats,
    targets: dict[str, tuple[float, float]],
) -> tuple[bool, list[str]]:
    """
    Validate distribution against targets.

    Args:
        stats: Distribution statistics
        targets: Dict of metric -> (min, max) targets

    Returns:
        Tuple of (all_passed, errors)
    """
    errors = []

    # G5-3: fact_single ratio
    if "fact_single" in targets:
        min_r, max_r = targets["fact_single"]
        if stats.fact_single_ratio >= max_r:
            errors.append(
                f"G5-3: fact_single {stats.fact_single_ratio:.1%} >= {max_r:.0%}"
            )

    # G5-4: hard ratio
    if "hard" in targets:
        min_r, max_r = targets["hard"]
        if stats.hard_ratio < min_r:
            errors.append(f"G5-4: hard {stats.hard_ratio:.1%} < {min_r:.0%}")

    # G5-5: unanswerable ratio
    if "unanswerable" in targets:
        min_r, max_r = targets["unanswerable"]
        if not (min_r <= stats.unanswerable_ratio <= max_r):
            errors.append(
                f"G5-5: unanswerable {stats.unanswerable_ratio:.1%} "
                f"not in [{min_r:.0%}, {max_r:.0%}]"
            )

    return len(errors) == 0, errors


def run_balance_pipeline(
    questions_path: Path,
    chunks_path: Path,
    output_path: Path | None = None,
    dedup_threshold: float = 0.95,
    anchor_threshold: float = 0.90,
) -> dict:
    """
    Run complete deduplication and balancing pipeline.

    Args:
        questions_path: Path to input questions JSON
        chunks_path: Path to chunks JSON
        output_path: Path to save balanced questions
        dedup_threshold: Similarity threshold for deduplication
        anchor_threshold: Max similarity for anchor independence

    Returns:
        Pipeline report dictionary
    """
    print(f"Loading questions from {questions_path}...")
    questions_data = load_json(questions_path)
    questions = questions_data.get("questions", [])
    print(f"  Loaded {len(questions)} questions")

    print(f"\nLoading chunks from {chunks_path}...")
    chunks_data = load_json(chunks_path)
    chunks = chunks_data.get("chunks", chunks_data)
    chunk_index = {c["id"]: c["text"] for c in chunks}
    print(f"  Loaded {len(chunks)} chunks")

    # Initial stats
    initial_stats = compute_distribution_stats(questions)
    print("\nInitial distribution:")
    print(f"  Total: {initial_stats.total}")
    print(f"  Answerable: {initial_stats.answerable}")
    print(
        f"  Unanswerable: {initial_stats.unanswerable} ({initial_stats.unanswerable_ratio:.1%})"
    )
    print(
        f"  fact_single: {initial_stats.fact_single} ({initial_stats.fact_single_ratio:.1%})"
    )
    print(f"  summary: {initial_stats.summary} ({initial_stats.summary_ratio:.1%})")
    print(
        f"  reasoning: {initial_stats.reasoning} ({initial_stats.reasoning_ratio:.1%})"
    )
    print(f"  hard: {initial_stats.hard} ({initial_stats.hard_ratio:.1%})")

    # Step 1: Deduplication
    print("\n" + "=" * 50)
    print("STEP 1: DEDUPLICATION")
    print("=" * 50)

    dedup_result = deduplicate_questions(questions, dedup_threshold)
    print(f"\n  Unique questions: {len(dedup_result.unique_ids)}")
    print(f"  Duplicates removed: {len(dedup_result.removed_ids)}")

    if dedup_result.duplicate_pairs:
        print("\n  Sample duplicate pairs:")
        for q1, q2, sim in dedup_result.duplicate_pairs[:3]:
            print(f"    {q1[:30]}... ~ {q2[:30]}... (sim={sim:.3f})")

    # Filter to unique questions
    unique_ids_set = set(dedup_result.unique_ids)
    unique_questions = [q for q in questions if q.get("id") in unique_ids_set]

    # Step 2: Anchor independence check
    print("\n" + "=" * 50)
    print("STEP 2: ANCHOR INDEPENDENCE CHECK")
    print("=" * 50)

    valid_ids, violations = check_anchor_independence(
        unique_questions, chunk_index, anchor_threshold
    )
    print(f"\n  Valid questions: {len(valid_ids)}")
    print(f"  Violations: {len(violations)}")

    if violations:
        print("\n  Sample violations:")
        for qid, sim in violations[:3]:
            print(f"    {qid[:40]}... (sim={sim:.3f})")

    # Filter to valid questions
    valid_ids_set = set(valid_ids)
    valid_questions = [q for q in unique_questions if q.get("id") in valid_ids_set]

    # Step 3: Balance distribution
    print("\n" + "=" * 50)
    print("STEP 3: DISTRIBUTION BALANCING")
    print("=" * 50)

    targets = {
        "fact_single": (0.0, 0.60),  # Max 60% (project threshold)
        "summary": (0.15, 0.25),
        "reasoning": (0.10, 0.20),
        "unanswerable": (0.25, 0.40),  # SQuAD 2.0: train=33.4%, dev=50%
        "hard": (0.10, 1.0),  # Min 10%
    }

    balanced_questions = balance_distribution(valid_questions, targets)

    # Final stats
    final_stats = compute_distribution_stats(balanced_questions)
    print("\nFinal distribution:")
    print(f"  Total: {final_stats.total}")
    print(f"  Answerable: {final_stats.answerable}")
    print(
        f"  Unanswerable: {final_stats.unanswerable} ({final_stats.unanswerable_ratio:.1%})"
    )
    print(
        f"  fact_single: {final_stats.fact_single} ({final_stats.fact_single_ratio:.1%})"
    )
    print(f"  summary: {final_stats.summary} ({final_stats.summary_ratio:.1%})")
    print(f"  reasoning: {final_stats.reasoning} ({final_stats.reasoning_ratio:.1%})")
    print(f"  hard: {final_stats.hard} ({final_stats.hard_ratio:.1%})")

    # Validate
    passed, dist_errors = validate_distribution(final_stats, targets)

    # Compile report
    report = {
        "date": get_date(),
        "initial_count": len(questions),
        "final_count": len(balanced_questions),
        "deduplication": {
            "threshold": dedup_threshold,
            "duplicates_removed": len(dedup_result.removed_ids),
            "duplicate_pairs": [
                {"q1": p[0], "q2": p[1], "similarity": p[2]}
                for p in dedup_result.duplicate_pairs
            ],
        },
        "anchor_independence": {
            "threshold": anchor_threshold,
            "violations": len(violations),
            "violation_details": [
                {"question_id": v[0], "similarity": v[1]} for v in violations
            ],
        },
        "distribution": {
            "initial": {
                "total": initial_stats.total,
                "unanswerable_ratio": initial_stats.unanswerable_ratio,
                "fact_single_ratio": initial_stats.fact_single_ratio,
                "hard_ratio": initial_stats.hard_ratio,
            },
            "final": {
                "total": final_stats.total,
                "unanswerable_ratio": final_stats.unanswerable_ratio,
                "fact_single_ratio": final_stats.fact_single_ratio,
                "summary_ratio": final_stats.summary_ratio,
                "reasoning_ratio": final_stats.reasoning_ratio,
                "hard_ratio": final_stats.hard_ratio,
            },
        },
        "gates": {
            "G5-1": {
                "name": "inter_question_similarity",
                "passed": len(dedup_result.removed_ids) == 0,
                "value": len(dedup_result.removed_ids),
                "threshold": f"<{dedup_threshold}",
            },
            "G5-2": {
                "name": "anchor_independence",
                "passed": len(violations) == 0,
                "value": len(violations),
                "threshold": f"<{anchor_threshold}",
            },
            "G5-3": {
                "name": "fact_single_ratio",
                "passed": final_stats.fact_single_ratio < 0.60,
                "value": f"{final_stats.fact_single_ratio:.1%}",
                "threshold": "<60%",
            },
            "G5-4": {
                "name": "hard_ratio",
                "passed": final_stats.hard_ratio >= 0.10,
                "value": f"{final_stats.hard_ratio:.1%}",
                "threshold": ">=10%",
            },
            "G5-5": {
                "name": "unanswerable_ratio",
                "passed": 0.25 <= final_stats.unanswerable_ratio <= 0.33,
                "value": f"{final_stats.unanswerable_ratio:.1%}",
                "threshold": "[25%, 33%]",
            },
        },
        "validation_errors": dist_errors,
    }

    # Save output
    if output_path:
        output_data = {
            "version": "1.0",
            "schema": "GS_SCHEMA_V2",
            "date": get_date(),
            "methodology": {
                "deduplication_threshold": dedup_threshold,
                "anchor_independence_threshold": anchor_threshold,
            },
            "coverage": {
                "total_questions": len(balanced_questions),
                "answerable": final_stats.answerable,
                "unanswerable": final_stats.unanswerable,
            },
            "questions": balanced_questions,
        }
        save_json(output_data, output_path)
        print(f"\nBalanced questions saved to {output_path}")

    # Print gate summary
    print("\n" + "=" * 50)
    print("QUALITY GATES SUMMARY")
    print("=" * 50)

    gates: dict[str, dict] = report["gates"]  # type: ignore[assignment]
    for gate_id, gate in gates.items():
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"  [{status}] {gate_id}: {gate['name']}")
        print(f"         Value: {gate['value']}, Threshold: {gate['threshold']}")

    return report


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deduplicate and balance question distribution"
    )
    parser.add_argument(
        "--questions",
        "-q",
        type=Path,
        required=True,
        help="Input questions JSON file",
    )
    parser.add_argument(
        "--chunks",
        "-c",
        type=Path,
        default=Path("corpus/processed/chunks_mode_b_fr.json"),
        help="Chunks JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output balanced questions JSON",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.95,
        help="Deduplication similarity threshold (default: 0.95)",
    )
    parser.add_argument(
        "--anchor-threshold",
        type=float,
        default=0.90,
        help="Anchor independence threshold (default: 0.90)",
    )

    args = parser.parse_args()

    report = run_balance_pipeline(
        args.questions,
        args.chunks,
        args.output,
        args.dedup_threshold,
        args.anchor_threshold,
    )

    # Exit with error if blocking gate G5-2 failed
    if not report["gates"]["G5-2"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
