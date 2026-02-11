#!/usr/bin/env python3
"""
Phase 3: Anti-Hallucination Validation for BY DESIGN generated questions.

Validates that each question's expected_answer is actually extractable
from the source chunk. Uses three validation methods:
1. Verbatim match (exact substring)
2. Keyword coverage >= 80%
3. Semantic similarity >= 0.90 (using EmbeddingGemma QAT)

ISO Reference:
- ISO 42001 A.6.2.2: Provenance verification
- ISO 25010: Accuracy metrics

Usage:
    python validate_anti_hallucination.py --questions PATH --chunks PATH [--output PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
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

# Lazy-loaded embedding model (uses EmbeddingGemma QAT per ISO 42001)
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


def compute_embeddings_batch(texts: list[str]) -> np.ndarray:
    """Compute embeddings for a batch of texts."""
    model = get_embedding_model()
    return embed_texts(texts, model, normalize=True)


def normalize_text_for_matching(text: str) -> str:
    """
    Normalize text for matching purposes.

    - Lowercase
    - Remove accents via Unicode NFKD
    - Normalize whitespace
    """
    text = text.lower()

    # Unicode normalization to remove accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_keywords(text: str, min_length: int = 4) -> list[str]:
    """
    Extract meaningful keywords from text.

    Args:
        text: Input text
        min_length: Minimum keyword length

    Returns:
        List of keywords
    """
    # French stopwords
    stopwords = {
        "pour",
        "dans",
        "avec",
        "cette",
        "celui",
        "celle",
        "sont",
        "etre",
        "avoir",
        "fait",
        "faire",
        "peut",
        "doit",
        "tous",
        "tout",
        "plus",
        "moins",
        "avant",
        "apres",
        "entre",
        "sous",
        "sans",
        "vers",
        "comme",
        "donc",
        "alors",
        "aussi",
        "mais",
        "meme",
        "ainsi",
        "encore",
        "toujours",
        "jamais",
        "rien",
        "bien",
        "tres",
        "quand",
        "quoi",
        "comment",
        "pourquoi",
        "partir",
        "selon",
        "depuis",
        "pendant",
        "jusqu",
        "contre",
        "chez",
        "article",
        "articles",
        "alinea",
        "regle",
        "regles",
        "chapitre",
    }

    text_norm = normalize_text_for_matching(text)
    words = re.findall(r"\b[a-z]+\b", text_norm)

    keywords = []
    for word in words:
        if len(word) >= min_length and word not in stopwords:
            keywords.append(word)

    return keywords


@dataclass
class ValidationResult:
    """Result of anti-hallucination validation for one question."""

    question_id: str
    passed: bool
    method: str  # "verbatim" | "keyword" | "semantic" | "REJECTED"
    verbatim_match: bool
    keyword_coverage: float
    semantic_similarity: float | None
    details: str


def validate_verbatim(answer: str, chunk_text: str) -> bool:
    """
    Check for verbatim match (exact substring).

    Args:
        answer: Expected answer text
        chunk_text: Source chunk text

    Returns:
        True if answer is verbatim substring of chunk
    """
    answer_norm = normalize_text_for_matching(answer)
    chunk_norm = normalize_text_for_matching(chunk_text)

    # Direct substring match
    if answer_norm in chunk_norm:
        return True

    # Try with some flexibility (word boundaries)
    answer_words = answer_norm.split()
    if len(answer_words) >= 3:
        # Check if 80% of answer words appear in sequence
        chunk_words = chunk_norm.split()

        # Sliding window
        window_size = len(answer_words)
        for i in range(len(chunk_words) - window_size + 1):
            window = " ".join(chunk_words[i : i + window_size])
            if answer_norm in window or window in answer_norm:
                return True

    return False


def validate_keyword_coverage(
    answer: str,
    chunk_text: str,
    threshold: float = 0.80,
) -> tuple[bool, float]:
    """
    Check keyword coverage.

    Args:
        answer: Expected answer text
        chunk_text: Source chunk text
        threshold: Minimum coverage ratio

    Returns:
        Tuple of (passed, coverage_ratio)
    """
    keywords = extract_keywords(answer)
    if not keywords:
        return True, 1.0  # No keywords = trivially covered

    chunk_norm = normalize_text_for_matching(chunk_text)

    found = sum(1 for kw in keywords if kw in chunk_norm)
    coverage = found / len(keywords)

    return coverage >= threshold, coverage


def validate_semantic(
    answer: str,
    chunk_text: str,
    threshold: float = 0.90,
) -> tuple[bool, float]:
    """
    Check semantic similarity using EmbeddingGemma QAT.

    Args:
        answer: Expected answer text
        chunk_text: Source chunk text
        threshold: Minimum similarity

    Returns:
        Tuple of (passed, similarity)
    """
    answer_emb = compute_embedding(answer)
    chunk_emb = compute_embedding(chunk_text[:512])  # Truncate for efficiency

    similarity = cosine_similarity(answer_emb, chunk_emb)

    return similarity >= threshold, similarity


def validate_question(
    question: dict,
    chunk_text: str,
    use_semantic: bool = True,
    keyword_threshold: float = 0.80,
    semantic_threshold: float = 0.90,
) -> ValidationResult:
    """
    Validate a single question against anti-hallucination criteria.

    Validation order (first passing method wins):
    1. Verbatim match
    2. Keyword coverage >= 80%
    3. Semantic similarity >= 0.90

    Args:
        question: Question dict with content.expected_answer
        chunk_text: Source chunk text
        use_semantic: Whether to compute semantic similarity
        keyword_threshold: Minimum keyword coverage
        semantic_threshold: Minimum semantic similarity

    Returns:
        ValidationResult with pass/fail status and method
    """
    qid = question.get("id", "unknown")
    answer = question.get("content", {}).get("expected_answer", "")

    # Skip validation for unanswerable questions FIRST
    # (they may not have expected_answer)
    if question.get("content", {}).get("is_impossible", False):
        return ValidationResult(
            question_id=qid,
            passed=True,
            method="unanswerable_skip",
            verbatim_match=False,
            keyword_coverage=0.0,
            semantic_similarity=None,
            details="Unanswerable question - validation skipped",
        )

    if not answer:
        return ValidationResult(
            question_id=qid,
            passed=False,
            method="REJECTED",
            verbatim_match=False,
            keyword_coverage=0.0,
            semantic_similarity=None,
            details="No expected_answer provided",
        )

    # Test 1: Verbatim match
    verbatim = validate_verbatim(answer, chunk_text)
    if verbatim:
        return ValidationResult(
            question_id=qid,
            passed=True,
            method="verbatim",
            verbatim_match=True,
            keyword_coverage=1.0,
            semantic_similarity=None,
            details="Answer found verbatim in chunk",
        )

    # Test 2: Keyword coverage
    kw_passed, kw_coverage = validate_keyword_coverage(
        answer, chunk_text, keyword_threshold
    )
    if kw_passed:
        return ValidationResult(
            question_id=qid,
            passed=True,
            method="keyword",
            verbatim_match=False,
            keyword_coverage=kw_coverage,
            semantic_similarity=None,
            details=f"Keyword coverage: {kw_coverage:.1%}",
        )

    # Test 3: Semantic similarity
    semantic_sim = None
    if use_semantic:
        sem_passed, semantic_sim = validate_semantic(
            answer, chunk_text, semantic_threshold
        )
        if sem_passed:
            return ValidationResult(
                question_id=qid,
                passed=True,
                method="semantic",
                verbatim_match=False,
                keyword_coverage=kw_coverage,
                semantic_similarity=semantic_sim,
                details=f"Semantic similarity: {semantic_sim:.3f}",
            )

    # All tests failed
    return ValidationResult(
        question_id=qid,
        passed=False,
        method="REJECTED",
        verbatim_match=False,
        keyword_coverage=kw_coverage,
        semantic_similarity=semantic_sim,
        details=f"No match: verbatim=False, keyword={kw_coverage:.1%}, "
        f"semantic={f'{semantic_sim:.3f}' if semantic_sim is not None else 'N/A'}",
    )


def validate_questions(
    questions: list[dict],
    chunk_index: dict[str, str],
    use_semantic: bool = True,
    keyword_threshold: float = 0.80,
    semantic_threshold: float = 0.90,
) -> dict:
    """
    Validate all questions against anti-hallucination criteria.

    Args:
        questions: List of question dicts
        chunk_index: Dict mapping chunk_id to chunk text
        use_semantic: Whether to compute semantic similarity
        keyword_threshold: Minimum keyword coverage
        semantic_threshold: Minimum semantic similarity

    Returns:
        Validation report dict
    """
    results = []
    passed_count = 0
    rejected_ids = []

    method_counts: dict[str, int] = {
        "verbatim": 0,
        "keyword": 0,
        "semantic": 0,
        "unanswerable_skip": 0,
        "REJECTED": 0,
    }

    print(f"\nValidating {len(questions)} questions...")

    for i, question in enumerate(questions):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(questions)}...")

        chunk_id = question.get("provenance", {}).get("chunk_id", "")
        chunk_text = chunk_index.get(chunk_id, "")

        if not chunk_text and not question.get("content", {}).get("is_impossible"):
            result = ValidationResult(
                question_id=question.get("id", "unknown"),
                passed=False,
                method="REJECTED",
                verbatim_match=False,
                keyword_coverage=0.0,
                semantic_similarity=None,
                details=f"Chunk not found: {chunk_id}",
            )
        else:
            result = validate_question(
                question,
                chunk_text,
                use_semantic=use_semantic,
                keyword_threshold=keyword_threshold,
                semantic_threshold=semantic_threshold,
            )

        results.append(result)
        method_counts[result.method] = method_counts.get(result.method, 0) + 1

        if result.passed:
            passed_count += 1
        else:
            rejected_ids.append(result.question_id)

    # Compile report
    total = len(questions)
    report = {
        "validation_date": get_date(),
        "total_questions": total,
        "passed": passed_count,
        "rejected": len(rejected_ids),
        "pass_rate": passed_count / total if total > 0 else 0,
        "thresholds": {
            "keyword_coverage": keyword_threshold,
            "semantic_similarity": semantic_threshold,
        },
        "method_distribution": method_counts,
        "rejected_ids": rejected_ids,
        "details": [
            {
                "question_id": r.question_id,
                "passed": r.passed,
                "method": r.method,
                "verbatim": r.verbatim_match,
                "keyword_coverage": r.keyword_coverage,
                "semantic_similarity": r.semantic_similarity,
                "details": r.details,
            }
            for r in results
        ],
        "gates": {
            "G3-1": {
                "name": "validation_passed",
                "passed": passed_count == total,
                "value": f"{passed_count}/{total}",
                "threshold": "100%",
            },
            "G3-2": {
                "name": "hallucination_count",
                "passed": len(rejected_ids) == 0,
                "value": len(rejected_ids),
                "threshold": 0,
            },
        },
    }

    return report


def format_validation_report(report: dict) -> str:
    """Format validation report for display."""
    lines = [
        "=" * 70,
        "PHASE 3: ANTI-HALLUCINATION VALIDATION REPORT",
        "=" * 70,
        "",
        f"Date: {report['validation_date']}",
        f"Total questions: {report['total_questions']}",
        f"Passed: {report['passed']} ({report['pass_rate']:.1%})",
        f"Rejected: {report['rejected']}",
        "",
        "VALIDATION METHODS:",
    ]

    for method, count in report["method_distribution"].items():
        lines.append(f"  {method}: {count}")

    lines.extend(
        [
            "",
            "THRESHOLDS:",
            f"  Keyword coverage: {report['thresholds']['keyword_coverage']:.0%}",
            f"  Semantic similarity: {report['thresholds']['semantic_similarity']:.2f}",
            "",
            "QUALITY GATES:",
        ]
    )

    for gate_id, gate in report["gates"].items():
        status = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"  [{status}] {gate_id}: {gate['name']}")
        lines.append(f"         Value: {gate['value']}, Threshold: {gate['threshold']}")

    if report["rejected_ids"]:
        lines.extend(
            [
                "",
                f"REJECTED QUESTIONS ({len(report['rejected_ids'])}):",
            ]
        )
        for qid in report["rejected_ids"][:10]:
            lines.append(f"  - {qid}")
        if len(report["rejected_ids"]) > 10:
            lines.append(f"  ... and {len(report['rejected_ids']) - 10} more")

    return "\n".join(lines)


def run_validation(
    questions_path: Path,
    chunks_path: Path,
    output_path: Path | None = None,
    use_semantic: bool = True,
) -> dict:
    """
    Run complete anti-hallucination validation.

    Args:
        questions_path: Path to questions JSON (GS format)
        chunks_path: Path to chunks JSON
        output_path: Path to save validation report
        use_semantic: Whether to use semantic validation

    Returns:
        Validation report dict
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

    # Run validation
    report = validate_questions(
        questions,
        chunk_index,
        use_semantic=use_semantic,
    )

    # Print report
    print("\n" + format_validation_report(report))

    # Save if output path provided
    if output_path:
        save_json(report, output_path)
        print(f"\nReport saved to {output_path}")

    return report


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate questions against anti-hallucination criteria"
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
        help="Output validation report JSON",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip semantic similarity validation (faster)",
    )

    args = parser.parse_args()

    report = run_validation(
        args.questions,
        args.chunks,
        args.output,
        use_semantic=not args.no_semantic,
    )

    # Exit with error if validation failed
    if not report["gates"]["G3-1"]["passed"] or not report["gates"]["G3-2"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
