"""
Tests de recall et validation anti-hallucination

ISO Reference:
    - ISO/IEC 25010 S4.2 - Performance efficiency (Recall >= 80%)
    - ISO/IEC 42001 - AI traceability (0% hallucination)
    - ISO/IEC 29119 - Test execution

Structure ISO:
    - Unit tests: TestComputeRecall (pas de modele, rapide)
    - Integration: benchmark_recall() via CLI ou script separe
    - NO fixtures de modeles ML (overhead inacceptable)

Usage CLI:
    python -m scripts.pipeline.tests.test_recall --db corpus/processed/corpus_mode_b_fr.db
"""

from pathlib import Path

import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).parents[3]
DATA_DIR = PROJECT_ROOT / "tests" / "data"
CORPUS_DIR = PROJECT_ROOT / "corpus" / "processed"


# =============================================================================
# Benchmark Functions (ISO 25010 S4.2)
# =============================================================================


def compute_recall_at_k(
    retrieved_pages: list[int],
    expected_pages: list[int],
    k: int = 5,
    tolerance: int = 0,
) -> float:
    """
    Calcule le recall@k pour une question.

    Args:
        retrieved_pages: Pages des chunks retrieves (top-k).
        expected_pages: Pages attendues (gold standard).
        k: Nombre de resultats consideres.
        tolerance: Tolerance de pages (±N) pour fuzzy matching.

    Returns:
        Recall entre 0.0 et 1.0.
    """
    if not expected_pages:
        return 1.0

    retrieved_set = set(retrieved_pages[:k])
    expected_set = set(expected_pages)

    if tolerance == 0:
        found = len(retrieved_set & expected_set)
    else:
        found = 0
        for expected_page in expected_set:
            for retrieved_page in retrieved_set:
                if abs(retrieved_page - expected_page) <= tolerance:
                    found += 1
                    break

    return found / len(expected_set)


def _parse_question(q: dict) -> tuple[str, list[int], bool]:
    """Extract question text, expected pages, and impossibility.

    Supports two formats:
        - V2 (nested): q["content"]["question"], q["provenance"]["pages"]
        - Legacy (flat): q["question"], q["expected_pages"]

    Required keys per format:
        - V2: content.question, provenance.pages
        - Legacy: question, expected_pages
    """
    if "content" in q:
        return (
            q["content"]["question"],
            q["provenance"]["pages"],
            q["content"].get("is_impossible", False),
        )
    return q["question"], q["expected_pages"], q.get("is_impossible", False)


def _build_recall_result(
    recalls: list[float],
    results: list[dict],
    failed: list[dict],
    top_k: int,
    use_hybrid: bool,
    tolerance: int,
) -> dict:
    """Build the recall benchmark result dict."""
    recall_mean = 0.0 if not recalls else float(np.mean(recalls))

    return {
        "recall_mean": recall_mean,
        "recall_percent": recall_mean * 100,
        "total_questions": len(results),
        "questions_detail": results,
        "failed_questions": failed,
        "config": {
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "tolerance": tolerance,
        },
    }


def benchmark_recall(
    db_path: Path,
    questions_file: Path,
    model,
    top_k: int = 5,
    use_hybrid: bool = False,
    tolerance: int = 0,
) -> dict:
    """
    Benchmark le recall sur un ensemble de questions gold standard.

    ISO 25010 S4.2: Recall@5 >= 80% requis.

    Args:
        db_path: Base SqliteVectorStore.
        questions_file: Fichier JSON questions gold standard.
        model: Modele d'embeddings (pre-charge).
        top_k: Nombre de resultats finaux.
        use_hybrid: Recherche hybride (vector + BM25).
        tolerance: Tolerance pages (±N).

    Returns:
        Dict avec recall_mean, recall_percent, questions_detail, failed_questions.
    """
    from scripts.pipeline.embeddings import embed_query
    from scripts.pipeline.export_search import (
        retrieve_hybrid,
        smart_retrieve,
    )
    from scripts.pipeline.utils import load_json

    questions_data = load_json(questions_file)
    questions = questions_data["questions"]

    results = []
    failed = []

    for q in questions:
        question_text, expected_pages, is_impossible = _parse_question(q)

        if is_impossible or not expected_pages:
            continue

        query_emb = embed_query(question_text, model)

        if use_hybrid:
            retrieved = retrieve_hybrid(db_path, query_emb, question_text, top_k=top_k)
        else:
            # smart_retrieve: auto source_filter based on specific patterns
            retrieved = smart_retrieve(db_path, query_emb, question_text, top_k=top_k)

        retrieved_pages = [r["page"] for r in retrieved]

        recall = compute_recall_at_k(
            retrieved_pages, expected_pages, k=top_k, tolerance=tolerance
        )

        result = {
            "id": q["id"],
            "question": question_text,
            "expected_pages": expected_pages,
            "retrieved_pages": retrieved_pages,
            "recall": recall,
        }
        results.append(result)

        if recall < 1.0:
            failed.append(result)

    recalls = [r["recall"] for r in results]
    return _build_recall_result(recalls, results, failed, top_k, use_hybrid, tolerance)


# =============================================================================
# Unit Tests (pytest - NO model fixtures)
# =============================================================================


class TestComputeRecall:
    """Tests unitaires compute_recall_at_k() - ISO 29119."""

    def test_perfect_recall(self):
        assert compute_recall_at_k([41, 42, 50], [41, 42], k=5) == 1.0

    def test_partial_recall(self):
        assert compute_recall_at_k([41, 50, 60], [41, 42], k=5) == 0.5

    def test_zero_recall(self):
        assert compute_recall_at_k([50, 60, 70], [41, 42], k=5) == 0.0

    def test_empty_expected(self):
        assert compute_recall_at_k([41, 42], [], k=5) == 1.0

    def test_k_limits_results(self):
        assert compute_recall_at_k([41, 50, 42], [41, 42], k=2) == 0.5

    def test_tolerance_adjacent_page(self):
        assert compute_recall_at_k([56], [55], k=5, tolerance=2) == 1.0

    def test_tolerance_zero_no_fuzzy(self):
        assert compute_recall_at_k([56], [55], k=5, tolerance=0) == 0.0

    def test_tolerance_exact_boundary(self):
        assert compute_recall_at_k([57], [55], k=5, tolerance=2) == 1.0
        assert compute_recall_at_k([58], [55], k=5, tolerance=2) == 0.0


class TestParseQuestion:
    """Tests _parse_question() - Schema V2 vs legacy."""

    def test_v2_nested_format(self):
        q = {
            "content": {"question": "Q?", "is_impossible": False},
            "provenance": {"pages": [1, 2]},
        }
        text, pages, impossible = _parse_question(q)
        assert text == "Q?"
        assert pages == [1, 2]
        assert impossible is False

    def test_v2_impossible(self):
        q = {
            "content": {"question": "Q?", "is_impossible": True},
            "provenance": {"pages": []},
        }
        _, _, impossible = _parse_question(q)
        assert impossible is True

    def test_legacy_flat_format(self):
        q = {"question": "Q?", "expected_pages": [3]}
        text, pages, impossible = _parse_question(q)
        assert text == "Q?"
        assert pages == [3]
        assert impossible is False

    def test_legacy_with_is_impossible(self):
        q = {"question": "Q?", "expected_pages": [], "is_impossible": True}
        _, _, impossible = _parse_question(q)
        assert impossible is True

    def test_v2_missing_is_impossible_defaults_false(self):
        q = {
            "content": {"question": "Q?"},
            "provenance": {"pages": [1]},
        }
        _, _, impossible = _parse_question(q)
        assert impossible is False


class TestSyntheticRecall:
    """Tests synthetiques - ISO 29119."""

    def test_recall_computation_correct(self):
        test_cases = [
            ([1, 2, 3], [1, 2], 1.0),
            ([1, 3, 4], [1, 2], 0.5),
            ([3, 4, 5], [1, 2], 0.0),
            ([1, 2, 3, 4, 5], [1, 2, 3], 1.0),
        ]
        for retrieved, expected, expected_recall in test_cases:
            assert compute_recall_at_k(retrieved, expected, k=5) == expected_recall

    def test_mean_recall_computation(self):
        recalls = [1.0, 0.5, 0.5, 1.0, 0.0]
        assert abs(np.mean(recalls) - 0.6) < 0.001


class TestBuildRecallResult:
    """Tests _build_recall_result() helper."""

    def test_empty_recalls_returns_zero(self):
        result = _build_recall_result(
            [], [], [], top_k=5, use_hybrid=False, tolerance=0
        )
        assert result["recall_mean"] == 0.0
        assert result["recall_percent"] == 0.0
        assert result["total_questions"] == 0

    def test_normal_recalls(self):
        recalls = [1.0, 0.5]
        results = [{"recall": 1.0}, {"recall": 0.5}]
        failed = [{"recall": 0.5}]
        result = _build_recall_result(
            recalls, results, failed, top_k=5, use_hybrid=True, tolerance=1
        )
        assert result["recall_mean"] == 0.75
        assert result["recall_percent"] == 75.0
        assert result["total_questions"] == 2
        assert result["failed_questions"] == failed
        assert result["config"]["use_hybrid"] is True


# =============================================================================
# CLI Benchmark (ISO 25010 validation)
# =============================================================================


def _build_corpora(corpus_choice: str) -> list[tuple[str, Path, Path]]:
    """Build list of (name, db_path, questions_file) based on CLI choice."""
    corpora: list[tuple[str, Path, Path]] = []
    if corpus_choice in ["fr", "both"]:
        corpora.append(
            (
                "FR",
                CORPUS_DIR / "corpus_mode_b_fr.db",
                DATA_DIR / "gold_standard_fr.json",
            )
        )
    if corpus_choice in ["intl", "both"]:
        corpora.append(
            (
                "INTL",
                CORPUS_DIR / "corpus_mode_a_intl.db",
                DATA_DIR / "gold_standard_intl.json",
            )
        )
    return corpora


def _run_single_benchmark(
    name: str,
    db_path: Path,
    questions_file: Path,
    model: object,
    args: object,
) -> bool | None:
    """Run benchmark on a single corpus. Returns True/False for pass/fail, None if skipped."""
    if not db_path.exists():
        print(f"\n[SKIP] {name}: DB not found {db_path}")
        return None
    if not questions_file.exists():
        print(f"\n[SKIP] {name}: Questions not found {questions_file}")
        return None

    print(f"\n=== {name} ===")
    result = benchmark_recall(
        db_path,
        questions_file,
        model,
        top_k=args.top_k,
        use_hybrid=args.hybrid,
        tolerance=args.tolerance,
    )

    recall_pct = result["recall_percent"]
    iso_pass = recall_pct >= 80

    print(f"Recall@{args.top_k}: {recall_pct:.2f}%")
    print(f"ISO 25010 (>=80%): {'PASS' if iso_pass else 'FAIL'}")
    print(f"Target (>=90%): {'PASS' if recall_pct >= 90 else 'FAIL'}")
    print(
        f"Questions: {result['total_questions']}, Failed: {len(result['failed_questions'])}"
    )

    if args.verbose and result["failed_questions"]:
        for q in result["failed_questions"]:
            print(
                f"  {q['id']}: {q['recall'] * 100:.0f}% - expected {q['expected_pages']}, got {q['retrieved_pages']}"
            )

    return iso_pass


def main() -> int:
    """CLI benchmark recall - ISO 25010 S4.2."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Recall ISO 25010")
    parser.add_argument("--corpus", choices=["fr", "intl", "both"], default="both")
    parser.add_argument("--hybrid", action="store_true", help="Hybrid search")
    parser.add_argument("--tolerance", type=int, default=2, help="Page tolerance")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    from scripts.pipeline.embeddings import MODEL_ID, load_embedding_model

    print(f"Loading embedding model: {MODEL_ID}")
    model = load_embedding_model(MODEL_ID)

    corpora = _build_corpora(args.corpus)

    print("\n" + "=" * 60)
    print("BENCHMARK RECALL - ISO 25010 S4.2")
    print("=" * 60)

    all_pass = True
    for name, db_path, questions_file in corpora:
        passed = _run_single_benchmark(name, db_path, questions_file, model, args)
        if passed is False:
            all_pass = False

    print("\n" + "=" * 60)
    print(f"RESULT: {'ALL PASS' if all_pass else 'FAIL'}")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
