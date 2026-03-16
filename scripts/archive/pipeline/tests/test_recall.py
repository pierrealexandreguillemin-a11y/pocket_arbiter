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


def compute_recall_at_k_chunk(
    retrieved_chunks: list[dict],
    expected_chunk_id: str,
    k: int = 5,
) -> float:
    """
    Recall@k by exact chunk ID match.

    Args:
        retrieved_chunks: Top-k retrieved chunks (each has "id" key).
        expected_chunk_id: Expected chunk ID from gold standard.
        k: Number of results to consider.

    Returns:
        1.0 if expected chunk found in top-k, 0.0 otherwise.
    """
    if not expected_chunk_id:
        return 1.0
    retrieved_ids = {c["id"] for c in retrieved_chunks[:k]}
    return 1.0 if expected_chunk_id in retrieved_ids else 0.0


def compute_mrr(
    retrieved_chunks: list[dict],
    expected_chunk_id: str,
) -> float:
    """
    Mean Reciprocal Rank for a single query.

    Args:
        retrieved_chunks: Retrieved chunks (ordered by score).
        expected_chunk_id: Expected chunk ID.

    Returns:
        1/rank if found, 0.0 otherwise.
    """
    if not expected_chunk_id:
        return 1.0
    for i, chunk in enumerate(retrieved_chunks):
        if chunk["id"] == expected_chunk_id:
            return 1.0 / (i + 1)
    return 0.0


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


def _parse_question_v8(q: dict) -> dict:
    """Extract full v8 metadata from a GS question.

    Returns dict with: question, chunk_id, pages, is_impossible,
    reasoning_class, difficulty, source_session, source_uuid,
    requires_context.
    """
    content = q.get("content", {})
    provenance = q.get("provenance", {})
    classification = q.get("classification", {})
    audit = q.get("audit", {})

    annales = provenance.get("annales_source") or {}
    docs = provenance.get("docs") or []

    return {
        "question": content.get("question", ""),
        "chunk_id": provenance.get("chunk_id", ""),
        "pages": provenance.get("pages", []),
        "is_impossible": content.get("is_impossible", False),
        "reasoning_class": classification.get("reasoning_class", "unknown"),
        "difficulty": classification.get("difficulty", 0.0),
        "source_session": annales.get("session", "unknown"),
        "source_uuid": docs[0] if docs else "unknown",
        "requires_context": "requires_context" in audit.get("history", ""),
    }


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


def _aggregate_group(
    group_results: list[dict], k_values: list[int]
) -> dict[str, int | float]:
    """Compute recall@K stats for a group of results."""
    n = len(group_results)
    seg: dict[str, int | float] = {"count": n}
    for k in k_values:
        sk = str(k)
        seg[f"recall@{k}"] = round(
            sum(r["chunk_recall"][sk] for r in group_results) / n, 4
        )
    return seg


def _difficulty_band(d: float) -> str:
    """Map difficulty score to band name."""
    if d < 0.33:
        return "easy"
    return "medium" if d <= 0.66 else "hard"


def _compute_segments(
    results: list[dict],
    k_values: list[int],
) -> dict:
    """Aggregate recall by reasoning_class, difficulty band, source_session, source_uuid."""
    from collections import defaultdict

    segment_keys = ["reasoning_class", "source_session", "source_uuid"]
    segments: dict[str, dict] = {}

    for key in segment_keys:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            groups[r[key]].append(r)
        segments[key] = {
            name: _aggregate_group(grp, k_values)
            for name, grp in sorted(groups.items())
        }

    diff_groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        diff_groups[_difficulty_band(r["difficulty"])].append(r)

    segments["difficulty"] = {
        band: _aggregate_group(diff_groups[band], k_values)
        for band in ["easy", "medium", "hard"]
        if diff_groups.get(band)
    }

    return segments


def _evaluate_single_question(
    q: dict,
    db_path: Path,
    model: object,
    top_k: int,
    use_hybrid: bool,
    tolerance: int,
    k_values: list[int],
) -> dict | None:
    """Evaluate a single GS question. Returns detail dict or None if skipped."""
    from scripts.pipeline.embeddings import embed_query
    from scripts.pipeline.export_search import retrieve_hybrid, smart_retrieve

    parsed = _parse_question_v8(q)
    if parsed["is_impossible"] or not parsed["chunk_id"] or parsed["requires_context"]:
        return None

    query_emb = embed_query(parsed["question"], model)
    retriever = retrieve_hybrid if use_hybrid else smart_retrieve
    retrieved = retriever(db_path, query_emb, parsed["question"], top_k=top_k)

    chunk_recalls = {
        str(k): compute_recall_at_k_chunk(retrieved, parsed["chunk_id"], k=k)
        for k in k_values
    }
    page_recalls = {
        str(k): compute_recall_at_k(
            [r["page"] for r in retrieved[:k]],
            parsed["pages"],
            k=k,
            tolerance=tolerance,
        )
        for k in k_values
    }

    return {
        "id": q["id"],
        "question": parsed["question"],
        "expected_chunk_id": parsed["chunk_id"],
        "expected_pages": parsed["pages"],
        "retrieved_chunks": [
            {
                "id": r["id"],
                "score": round(r.get("score", r.get("hybrid_score", 0)), 4),
                "page": r["page"],
            }
            for r in retrieved[:top_k]
        ],
        "chunk_recall": chunk_recalls,
        "page_recall": page_recalls,
        "mrr": compute_mrr(retrieved, parsed["chunk_id"]),
        "reasoning_class": parsed["reasoning_class"],
        "difficulty": parsed["difficulty"],
        "source_session": parsed["source_session"],
        "source_uuid": parsed["source_uuid"],
    }


def _aggregate_recall(
    results_detail: list[dict], k_values: list[int]
) -> tuple[dict[str, float], dict[str, float], float]:
    """Aggregate chunk/page recall@K and MRR from detailed results."""
    n = len(results_detail)
    if n == 0:
        empty: dict[str, float] = {str(k): 0.0 for k in k_values}
        return empty, empty, 0.0
    chunk = {
        str(k): sum(r["chunk_recall"][str(k)] for r in results_detail) / n
        for k in k_values
    }
    page = {
        str(k): sum(r["page_recall"][str(k)] for r in results_detail) / n
        for k in k_values
    }
    mrr = sum(r["mrr"] for r in results_detail) / n
    return chunk, page, mrr


def benchmark_recall_v8(
    db_path: Path,
    questions_file: Path,
    model: object,
    top_k: int = 10,
    use_hybrid: bool = False,
    tolerance: int = 2,
) -> dict:
    """Benchmark recall on GS with chunk-level and page-level metrics.

    Evaluates recall@1,3,5,10 (chunk + page), MRR, segmented by metadata.
    Skips is_impossible and requires_context questions.
    """
    from scripts.pipeline.utils import load_json

    questions = load_json(questions_file)["questions"]
    k_values = [1, 3, 5, 10]

    results_detail = [
        detail
        for q in questions
        if (
            detail := _evaluate_single_question(
                q, db_path, model, top_k, use_hybrid, tolerance, k_values
            )
        )
        is not None
    ]

    chunk_recall, page_recall, mrr_mean = _aggregate_recall(results_detail, k_values)
    failed = [r for r in results_detail if r["chunk_recall"]["5"] < 1.0]

    return {
        "total_questions": len(results_detail),
        "recall_at_k": {"chunk": chunk_recall, "page": page_recall},
        "mrr_mean": round(mrr_mean, 4),
        "failed_count": len(failed),
        "failed_questions": failed,
        "segments": _compute_segments(results_detail, k_values),
        "config": {
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "tolerance": tolerance,
            "model": "EmbeddingGemma-300M-QAT",
            "gs_version": "v9.0.0",
        },
        "questions_detail": results_detail,
    }


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


class TestComputeRecallChunk:
    """Tests compute_recall_at_k_chunk() - chunk-level recall."""

    def test_exact_match(self):
        retrieved = [{"id": "doc-p01-c00"}, {"id": "doc-p02-c00"}]
        assert compute_recall_at_k_chunk(retrieved, "doc-p01-c00", k=5) == 1.0

    def test_no_match(self):
        retrieved = [{"id": "doc-p02-c00"}, {"id": "doc-p03-c00"}]
        assert compute_recall_at_k_chunk(retrieved, "doc-p01-c00", k=5) == 0.0

    def test_k_limits(self):
        retrieved = [{"id": "other"}, {"id": "target"}]
        assert compute_recall_at_k_chunk(retrieved, "target", k=1) == 0.0
        assert compute_recall_at_k_chunk(retrieved, "target", k=2) == 1.0

    def test_empty_retrieved(self):
        assert compute_recall_at_k_chunk([], "target", k=5) == 0.0

    def test_empty_expected(self):
        retrieved = [{"id": "doc-p01-c00"}]
        assert compute_recall_at_k_chunk(retrieved, "", k=5) == 1.0


class TestComputeMRR:
    """Tests compute_mrr() - Mean Reciprocal Rank."""

    def test_first_position(self):
        retrieved = [{"id": "target"}, {"id": "other"}]
        assert compute_mrr(retrieved, "target") == 1.0

    def test_second_position(self):
        retrieved = [{"id": "other"}, {"id": "target"}]
        assert compute_mrr(retrieved, "target") == 0.5

    def test_not_found(self):
        retrieved = [{"id": "a"}, {"id": "b"}]
        assert compute_mrr(retrieved, "target") == 0.0

    def test_empty(self):
        assert compute_mrr([], "target") == 0.0

    def test_empty_expected(self):
        assert compute_mrr([{"id": "a"}], "") == 1.0


class TestParseQuestionV8:
    """Tests _parse_question_v8() - full v8 metadata extraction."""

    def test_answerable_question(self):
        q = {
            "content": {"question": "Q?", "is_impossible": False},
            "provenance": {
                "chunk_id": "doc-p01-c00",
                "docs": ["doc.pdf"],
                "pages": [1],
                "annales_source": {"session": "dec2024", "uv": "clubs"},
            },
            "classification": {
                "reasoning_class": "summary",
                "difficulty": 0.3,
            },
            "audit": {"history": "some history"},
        }
        parsed = _parse_question_v8(q)
        assert parsed["question"] == "Q?"
        assert parsed["chunk_id"] == "doc-p01-c00"
        assert parsed["pages"] == [1]
        assert parsed["is_impossible"] is False
        assert parsed["reasoning_class"] == "summary"
        assert parsed["difficulty"] == 0.3
        assert parsed["source_session"] == "dec2024"
        assert parsed["source_uuid"] == "doc.pdf"
        assert parsed["requires_context"] is False

    def test_requires_context_flagged(self):
        q = {
            "content": {"question": "Q?", "is_impossible": False},
            "provenance": {
                "chunk_id": "doc-p01-c00",
                "docs": ["doc.pdf"],
                "pages": [1],
            },
            "classification": {"reasoning_class": "fact_single", "difficulty": 0.5},
            "audit": {"history": "requires_context:exam_name(Olivier)"},
        }
        parsed = _parse_question_v8(q)
        assert parsed["requires_context"] is True

    def test_impossible_question(self):
        q = {
            "content": {"question": "Q?", "is_impossible": True},
            "provenance": {"chunk_id": "", "docs": [], "pages": []},
            "classification": {"reasoning_class": "summary", "difficulty": 0.0},
            "audit": {"history": ""},
        }
        parsed = _parse_question_v8(q)
        assert parsed["is_impossible"] is True
        assert parsed["chunk_id"] == ""


class TestComputeSegments:
    """Tests _compute_segments() aggregation."""

    def test_segments_by_reasoning_class(self):
        results = [
            {
                "chunk_recall": {"5": 1.0},
                "reasoning_class": "summary",
                "difficulty": 0.3,
                "source_session": "s1",
                "source_uuid": "doc.pdf",
            },
            {
                "chunk_recall": {"5": 0.0},
                "reasoning_class": "summary",
                "difficulty": 0.5,
                "source_session": "s1",
                "source_uuid": "doc.pdf",
            },
            {
                "chunk_recall": {"5": 1.0},
                "reasoning_class": "fact_single",
                "difficulty": 0.2,
                "source_session": "s2",
                "source_uuid": "other.pdf",
            },
        ]
        segments = _compute_segments(results, [5])
        assert segments["reasoning_class"]["summary"]["count"] == 2
        assert segments["reasoning_class"]["summary"]["recall@5"] == 0.5
        assert segments["reasoning_class"]["fact_single"]["recall@5"] == 1.0

    def test_segments_by_difficulty_band(self):
        results = [
            {
                "chunk_recall": {"5": 1.0},
                "reasoning_class": "summary",
                "difficulty": 0.1,
                "source_session": "s1",
                "source_uuid": "doc.pdf",
            },
            {
                "chunk_recall": {"5": 0.0},
                "reasoning_class": "summary",
                "difficulty": 0.5,
                "source_session": "s1",
                "source_uuid": "doc.pdf",
            },
            {
                "chunk_recall": {"5": 1.0},
                "reasoning_class": "fact_single",
                "difficulty": 0.8,
                "source_session": "s2",
                "source_uuid": "other.pdf",
            },
        ]
        segments = _compute_segments(results, [5])
        assert segments["difficulty"]["easy"]["count"] == 1
        assert segments["difficulty"]["easy"]["recall@5"] == 1.0
        assert segments["difficulty"]["medium"]["count"] == 1
        assert segments["difficulty"]["medium"]["recall@5"] == 0.0
        assert segments["difficulty"]["hard"]["count"] == 1
        assert segments["difficulty"]["hard"]["recall@5"] == 1.0

    def test_segments_by_source_session(self):
        results = [
            {
                "chunk_recall": {"5": 1.0},
                "reasoning_class": "summary",
                "difficulty": 0.3,
                "source_session": "dec2024",
                "source_uuid": "doc.pdf",
            },
            {
                "chunk_recall": {"5": 0.0},
                "reasoning_class": "summary",
                "difficulty": 0.5,
                "source_session": "jun2025",
                "source_uuid": "doc.pdf",
            },
        ]
        segments = _compute_segments(results, [5])
        assert segments["source_session"]["dec2024"]["count"] == 1
        assert segments["source_session"]["jun2025"]["count"] == 1

    def test_segments_empty_band_skipped(self):
        results = [
            {
                "chunk_recall": {"5": 1.0},
                "reasoning_class": "summary",
                "difficulty": 0.1,
                "source_session": "s1",
                "source_uuid": "doc.pdf",
            },
        ]
        segments = _compute_segments(results, [5])
        assert "easy" in segments["difficulty"]
        assert "medium" not in segments["difficulty"]
        assert "hard" not in segments["difficulty"]

    def test_segments_multi_k(self):
        results = [
            {
                "chunk_recall": {"1": 0.0, "5": 1.0},
                "reasoning_class": "summary",
                "difficulty": 0.3,
                "source_session": "s1",
                "source_uuid": "doc.pdf",
            },
        ]
        segments = _compute_segments(results, [1, 5])
        assert segments["reasoning_class"]["summary"]["recall@1"] == 0.0
        assert segments["reasoning_class"]["summary"]["recall@5"] == 1.0


class TestBenchmarkRecallV8:
    """Test benchmark_recall_v8 with mocked retrieval."""

    def test_basic_flow(self, tmp_path):
        import json
        from unittest.mock import patch

        gs_data = {
            "version": {"schema": "GS_SCHEMA_V2"},
            "questions": [
                {
                    "id": "q1",
                    "content": {"question": "Test?", "is_impossible": False},
                    "provenance": {
                        "chunk_id": "doc-p01-c00",
                        "docs": ["doc.pdf"],
                        "pages": [1],
                        "annales_source": {"session": "dec2024", "uv": "clubs"},
                    },
                    "classification": {
                        "reasoning_class": "summary",
                        "difficulty": 0.3,
                    },
                    "audit": {"history": ""},
                },
                {
                    "id": "q2",
                    "content": {"question": "Impossible?", "is_impossible": True},
                    "provenance": {"chunk_id": "", "docs": [], "pages": []},
                    "classification": {
                        "reasoning_class": "summary",
                        "difficulty": 0.0,
                    },
                    "audit": {"history": ""},
                },
            ],
        }
        gs_file = tmp_path / "gs.json"
        gs_file.write_text(json.dumps(gs_data), encoding="utf-8")

        mock_results = [
            {
                "id": "doc-p01-c00",
                "text": "chunk text",
                "source": "doc.pdf",
                "page": 1,
                "score": 0.95,
            },
            {
                "id": "other-c01",
                "text": "other",
                "source": "doc.pdf",
                "page": 2,
                "score": 0.80,
            },
        ]

        with (
            patch("scripts.pipeline.embeddings.embed_query", return_value="fake_emb"),
            patch(
                "scripts.pipeline.export_search.smart_retrieve",
                return_value=mock_results,
            ),
        ):
            result = benchmark_recall_v8(
                db_path=tmp_path / "fake.db",
                questions_file=gs_file,
                model=None,
                top_k=10,
                use_hybrid=False,
            )

        assert result["total_questions"] == 1  # q2 skipped (impossible)
        assert result["recall_at_k"]["chunk"]["1"] == 1.0
        assert result["recall_at_k"]["chunk"]["5"] == 1.0
        assert result["mrr_mean"] == 1.0
        assert result["failed_count"] == 0
        assert "segments" in result
        assert result["config"]["gs_version"] == "v9.0.0"

    def test_hybrid_mode(self, tmp_path):
        import json
        from unittest.mock import patch

        gs_data = {
            "questions": [
                {
                    "id": "q1",
                    "content": {"question": "Test?", "is_impossible": False},
                    "provenance": {
                        "chunk_id": "doc-p01-c00",
                        "docs": ["doc.pdf"],
                        "pages": [1],
                        "annales_source": {"session": "dec2024", "uv": "clubs"},
                    },
                    "classification": {
                        "reasoning_class": "summary",
                        "difficulty": 0.3,
                    },
                    "audit": {"history": ""},
                },
            ],
        }
        gs_file = tmp_path / "gs.json"
        gs_file.write_text(json.dumps(gs_data), encoding="utf-8")

        mock_results = [
            {
                "id": "doc-p01-c00",
                "text": "chunk",
                "source": "doc.pdf",
                "page": 1,
                "hybrid_score": 0.9,
            },
        ]

        with (
            patch("scripts.pipeline.embeddings.embed_query", return_value="fake_emb"),
            patch(
                "scripts.pipeline.export_search.retrieve_hybrid",
                return_value=mock_results,
            ),
        ):
            result = benchmark_recall_v8(
                db_path=tmp_path / "fake.db",
                questions_file=gs_file,
                model=None,
                top_k=10,
                use_hybrid=True,
            )

        assert result["total_questions"] == 1
        assert result["config"]["use_hybrid"] is True

    def test_requires_context_skipped(self, tmp_path):
        import json
        from unittest.mock import patch

        gs_data = {
            "questions": [
                {
                    "id": "q1",
                    "content": {"question": "Test?", "is_impossible": False},
                    "provenance": {
                        "chunk_id": "doc-p01-c00",
                        "docs": ["doc.pdf"],
                        "pages": [1],
                        "annales_source": {"session": "dec2024", "uv": "clubs"},
                    },
                    "classification": {
                        "reasoning_class": "summary",
                        "difficulty": 0.3,
                    },
                    "audit": {"history": "requires_context:exam_name(Test)"},
                },
            ],
        }
        gs_file = tmp_path / "gs.json"
        gs_file.write_text(json.dumps(gs_data), encoding="utf-8")

        with (
            patch("scripts.pipeline.embeddings.embed_query", return_value="fake_emb"),
            patch("scripts.pipeline.export_search.smart_retrieve", return_value=[]),
        ):
            result = benchmark_recall_v8(
                db_path=tmp_path / "fake.db",
                questions_file=gs_file,
                model=None,
            )

        assert result["total_questions"] == 0

    def test_failed_questions_tracked(self, tmp_path):
        import json
        from unittest.mock import patch

        gs_data = {
            "questions": [
                {
                    "id": "q1",
                    "content": {"question": "Test?", "is_impossible": False},
                    "provenance": {
                        "chunk_id": "doc-p01-c00",
                        "docs": ["doc.pdf"],
                        "pages": [1],
                        "annales_source": {"session": "dec2024", "uv": "clubs"},
                    },
                    "classification": {
                        "reasoning_class": "summary",
                        "difficulty": 0.3,
                    },
                    "audit": {"history": ""},
                },
            ],
        }
        gs_file = tmp_path / "gs.json"
        gs_file.write_text(json.dumps(gs_data), encoding="utf-8")

        # Return chunks that do NOT match expected chunk_id
        mock_results = [
            {
                "id": "wrong-c01",
                "text": "wrong",
                "source": "doc.pdf",
                "page": 5,
                "score": 0.8,
            },
        ]

        with (
            patch("scripts.pipeline.embeddings.embed_query", return_value="fake_emb"),
            patch(
                "scripts.pipeline.export_search.smart_retrieve",
                return_value=mock_results,
            ),
        ):
            result = benchmark_recall_v8(
                db_path=tmp_path / "fake.db",
                questions_file=gs_file,
                model=None,
            )

        assert result["total_questions"] == 1
        assert result["failed_count"] == 1
        assert result["recall_at_k"]["chunk"]["5"] == 0.0
        assert result["mrr_mean"] == 0.0


class TestPrintBaselineReport:
    """Test _print_baseline_report doesn't crash."""

    def test_prints_without_error(self, capsys):
        result = {
            "total_questions": 2,
            "recall_at_k": {
                "chunk": {"1": 0.5, "3": 0.5, "5": 1.0, "10": 1.0},
                "page": {"1": 0.5, "3": 1.0, "5": 1.0, "10": 1.0},
            },
            "mrr_mean": 0.75,
            "failed_count": 1,
            "failed_questions": [
                {
                    "id": "q1",
                    "question": "Test?",
                    "expected_chunk_id": "doc-p01-c00",
                    "retrieved_chunks": [
                        {"id": "other-c00", "score": 0.8, "page": 2},
                    ],
                    "chunk_recall": {"5": 0.0},
                }
            ],
            "segments": {
                "reasoning_class": {
                    "summary": {"count": 2, "recall@5": 0.5},
                },
                "difficulty": {
                    "easy": {"count": 2, "recall@5": 0.5},
                },
                "source_session": {},
                "source_uuid": {},
            },
            "config": {
                "top_k": 10,
                "use_hybrid": False,
                "model": "test",
                "gs_version": "v8",
            },
        }
        _print_baseline_report(result, "Vector-Only")
        captured = capsys.readouterr()
        assert "BASELINE RECALL v8" in captured.out
        assert "Vector-Only" in captured.out


class TestExportBaselineJson:
    """Tests _export_baseline_json() export."""

    def test_exports_json_file(self, tmp_path):
        result_vector = {
            "recall_at_k": {"chunk": {"5": 0.8}},
            "config": {"model": "test"},
        }
        result_hybrid = {
            "recall_at_k": {"chunk": {"5": 0.9}},
            "config": {"model": "test"},
        }
        output_path = _export_baseline_json(result_vector, result_hybrid, tmp_path)
        assert output_path.exists()
        import json

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert "vector_only" in data
        assert "hybrid" in data
        assert "timestamp" in data

    def test_exports_without_hybrid(self, tmp_path):
        result_vector = {
            "recall_at_k": {"chunk": {"5": 0.8}},
            "config": {"model": "test"},
        }
        output_path = _export_baseline_json(result_vector, None, tmp_path)
        import json

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert "vector_only" in data
        assert "hybrid" not in data


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


def _print_baseline_report(result: dict, variant_name: str) -> None:
    """Print formatted baseline recall report to console."""
    print(f"\n{'=' * 60}")
    print(f"BASELINE RECALL v8 — {variant_name}")
    print(f"{'=' * 60}")
    print(
        f"{result['total_questions']} questions | "
        f"{result['config']['model']} | {result['config']['gs_version']}"
    )
    print()

    chunk = result["recall_at_k"]["chunk"]
    page = result["recall_at_k"]["page"]

    print(f"{'Metric':<22} {'Chunk':>10} {'Page':>10}")
    print("-" * 44)
    for k in ["1", "3", "5", "10"]:
        if k in chunk:
            print(
                f"recall@{k:<16} "
                f"{chunk[k] * 100:>9.1f}% "
                f"{page.get(k, 0) * 100:>9.1f}%"
            )
    print(f"{'MRR':<22} {result['mrr_mean']:>10.4f}")
    print()

    # ISO gate
    r5 = chunk.get("5", 0)
    print(
        f"ISO 25010 (recall@5 chunk >= 80%): "
        f"{'PASS' if r5 >= 0.8 else 'FAIL'} ({r5 * 100:.1f}%)"
    )
    print()

    # Segments
    for seg_name in ["reasoning_class", "difficulty"]:
        seg_data = result["segments"].get(seg_name, {})
        if not seg_data:
            continue
        print(f"--- By {seg_name} ---")
        for group, stats in sorted(seg_data.items()):
            r5_seg = stats.get("recall@5", 0)
            print(f"  {group:<20} {stats['count']:>4}Q  recall@5={r5_seg * 100:.1f}%")
        print()

    # Top failed questions
    failed = result["failed_questions"]
    if failed:
        print(f"--- Failed questions ({len(failed)}) — top 20 ---")
        for fq in failed[:20]:
            print(f"  {fq['id']}")
            print(f"    Q: {fq['question'][:80]}...")
            print(f"    Expected: {fq['expected_chunk_id']}")
            top_retrieved = fq.get("retrieved_chunks", [])[:3]
            for i, rc in enumerate(top_retrieved):
                print(f"    #{i + 1}: {rc['id']} (score={rc['score']:.4f})")
        print()


def _export_baseline_json(
    result_vector: dict,
    result_hybrid: dict | None,
    output_dir: Path,
) -> Path:
    """Export baseline results to JSON."""
    import json
    from datetime import datetime, timezone

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    report: dict = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "vector_only": result_vector,
    }
    if result_hybrid:
        report["hybrid"] = result_hybrid

    output_path = output_dir / f"baseline_v8_{timestamp}.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path


def _run_baseline_v8(model: object, tolerance: int) -> int:
    """Run full baseline benchmark (vector + hybrid, multi-K, segmented)."""
    db_path = CORPUS_DIR / "corpus_mode_b_fr.db"
    gs_file = DATA_DIR / "gold_standard_annales_fr_v8_adversarial.json"

    if not db_path.exists():
        print(f"[ERROR] DB not found: {db_path}")
        return 1
    if not gs_file.exists():
        print(f"[ERROR] GS not found: {gs_file}")
        return 1

    print("\n>>> Running vector-only benchmark...")
    result_vector = benchmark_recall_v8(
        db_path, gs_file, model, top_k=10, use_hybrid=False, tolerance=tolerance
    )
    _print_baseline_report(result_vector, "Vector-Only")

    print("\n>>> Running hybrid benchmark...")
    result_hybrid = benchmark_recall_v8(
        db_path, gs_file, model, top_k=10, use_hybrid=True, tolerance=tolerance
    )
    _print_baseline_report(result_hybrid, "Hybrid (BM25+Vector)")

    output_dir = PROJECT_ROOT / "data" / "benchmarks"
    output_path = _export_baseline_json(result_vector, result_hybrid, output_dir)
    print(f"\n[EXPORT] Results saved to {output_path}")

    best = max(
        result_vector["recall_at_k"]["chunk"]["5"],
        result_hybrid["recall_at_k"]["chunk"]["5"],
    )
    _print_decision(best)
    return 0 if best >= 0.8 else 1


def _print_decision(best_recall: float) -> None:
    """Print recall decision threshold message."""
    if best_recall >= 0.8:
        msg = "recall@5 >= 80% — RAG pur + prompt engineering suffisant"
    elif best_recall >= 0.6:
        msg = "recall@5 60-80% — optimisations retrieval recommandees"
    else:
        msg = "recall@5 < 60% — fine-tuning embeddings justifie"
    print(f"\n>> DECISION: {msg}")


def main() -> int:
    """CLI benchmark recall - ISO 25010 S4.2."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Recall ISO 25010")
    parser.add_argument("--corpus", choices=["fr", "intl", "both"], default="both")
    parser.add_argument("--hybrid", action="store_true", help="Hybrid search")
    parser.add_argument("--tolerance", type=int, default=2, help="Page tolerance")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--baseline-v8",
        action="store_true",
        help="Run full baseline (vector + hybrid, multi-K, segmented)",
    )
    args = parser.parse_args()

    from scripts.pipeline.embeddings import MODEL_ID, load_embedding_model

    print(f"Loading embedding model: {MODEL_ID}")
    model = load_embedding_model(MODEL_ID)

    if args.baseline_v8:
        return _run_baseline_v8(model, args.tolerance)

    # Legacy benchmark (existing behavior unchanged)
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
