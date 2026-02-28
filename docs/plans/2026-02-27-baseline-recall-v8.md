# Baseline Recall v8 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish a reliable retrieval recall baseline on the v8 GS (328 testable questions) with both vector-only and hybrid search, including per-chunk recall, multi-K metrics, segmented analysis, and detailed error diagnostics.

**Architecture:** Extend `scripts/pipeline/tests/test_recall.py` with new functions for chunk-level recall, multi-K evaluation, MRR, segmented reporting, and error analysis. Keep the existing page-level recall intact. Add a new CLI command `--baseline-v8` that runs the full evaluation and exports JSON + console report.

**Tech Stack:** Python 3.10+, numpy, existing `export_search.py` (smart_retrieve, retrieve_hybrid), existing `embeddings.py` (embed_query, load_embedding_model)

---

## Key Facts (from codebase exploration)

- **GS file**: `tests/data/gold_standard_annales_fr_v8_adversarial.json` (525Q, UTF-8)
- **Structure**: Schema V2 nested — `q["provenance"]["chunk_id"]` is the expected chunk, `q["content"]["is_impossible"]` for filtering
- **420 answerable** (all have `chunk_id`), **105 unanswerable** (no `chunk_id`)
- **62 questions** have `requires_context` in `audit.history` — these should be skipped (328 testable = 420 - 92, but actual count from audit.history is 62, so re-derive: test all 420 with chunk_id, flag the 62 requires_context separately)
- **Retrieved chunks** return dict with keys: `id`, `text`, `source`, `page`, `score` (vector) or `hybrid_score` (hybrid)
- **Chunk IDs** format: `LA-octobre2025.pdf-p010-parent024-child00`
- **DB**: `corpus/processed/corpus_mode_b_fr.db`
- **Embeddings model**: `google/embeddinggemma-300m-qat-q4_0-unquantized` (768D)
- **Existing functions**: `compute_recall_at_k()` (page-level), `_parse_question()`, `benchmark_recall()`, `smart_retrieve()`, `retrieve_hybrid()`

---

### Task 1: Add `compute_recall_at_k_chunk()` function

**Files:**
- Modify: `scripts/pipeline/tests/test_recall.py` (after line 67)
- Test: same file, new `TestComputeRecallChunk` class

**Step 1: Write the failing tests**

Add after `TestBuildRecallResult` class (line 304):

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest scripts/pipeline/tests/test_recall.py::TestComputeRecallChunk -v`
Expected: FAIL — `NameError: name 'compute_recall_at_k_chunk' is not defined`

**Step 3: Write implementation**

Add after `compute_recall_at_k()` (after line 67):

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest scripts/pipeline/tests/test_recall.py::TestComputeRecallChunk -v`
Expected: all 5 PASS

**Step 5: Commit**

```bash
git add scripts/pipeline/tests/test_recall.py
git commit -m "feat(recall): add compute_recall_at_k_chunk for chunk-level recall"
```

---

### Task 2: Add `compute_mrr()` function

**Files:**
- Modify: `scripts/pipeline/tests/test_recall.py`

**Step 1: Write the failing tests**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest scripts/pipeline/tests/test_recall.py::TestComputeMRR -v`
Expected: FAIL

**Step 3: Write implementation**

```python
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
```

**Step 4: Run tests**

Run: `python -m pytest scripts/pipeline/tests/test_recall.py::TestComputeMRR -v`
Expected: all 5 PASS

**Step 5: Commit**

```bash
git add scripts/pipeline/tests/test_recall.py
git commit -m "feat(recall): add compute_mrr for reciprocal rank metric"
```

---

### Task 3: Add `_parse_question_v8()` helper

**Files:**
- Modify: `scripts/pipeline/tests/test_recall.py`

The existing `_parse_question()` returns `(text, pages, is_impossible)`. We need a v8-aware parser that also extracts `chunk_id`, `reasoning_class`, `difficulty`, `source_uuid`, `source_session`, and `requires_context`.

**Step 1: Write the failing tests**

```python
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
```

**Step 2: Run test to verify failure**

Run: `python -m pytest scripts/pipeline/tests/test_recall.py::TestParseQuestionV8 -v`

**Step 3: Write implementation**

```python
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

    annales = provenance.get("annales_source", {})
    docs = provenance.get("docs", [])

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
```

**Step 4: Run tests**

Run: `python -m pytest scripts/pipeline/tests/test_recall.py::TestParseQuestionV8 -v`
Expected: all 3 PASS

**Step 5: Commit**

```bash
git add scripts/pipeline/tests/test_recall.py
git commit -m "feat(recall): add _parse_question_v8 for full metadata extraction"
```

---

### Task 4: Add `benchmark_recall_v8()` core function

**Files:**
- Modify: `scripts/pipeline/tests/test_recall.py`

This is the main function. It calls `smart_retrieve()` and `retrieve_hybrid()` for each testable question, computes chunk-level recall at multiple K values, MRR, and collects error details.

**Step 1: Write the failing test**

One integration-style unit test using mocks (no model needed):

```python
from unittest.mock import patch


class TestBenchmarkRecallV8:
    """Test benchmark_recall_v8 with mocked retrieval."""

    def test_basic_flow(self, tmp_path):
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
        import json

        gs_file.write_text(json.dumps(gs_data), encoding="utf-8")

        mock_results = [
            {"id": "doc-p01-c00", "text": "chunk text", "source": "doc.pdf", "page": 1, "score": 0.95},
            {"id": "other-c01", "text": "other", "source": "doc.pdf", "page": 2, "score": 0.80},
        ]

        with patch("scripts.pipeline.tests.test_recall.embed_query", return_value="fake_emb"):
            with patch("scripts.pipeline.tests.test_recall.smart_retrieve", return_value=mock_results):
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
```

**Step 2: Run test to verify failure**

Run: `python -m pytest scripts/pipeline/tests/test_recall.py::TestBenchmarkRecallV8 -v`

**Step 3: Write implementation**

```python
def benchmark_recall_v8(
    db_path: Path,
    questions_file: Path,
    model: object,
    top_k: int = 10,
    use_hybrid: bool = False,
    tolerance: int = 2,
) -> dict:
    """
    Benchmark recall on v8 GS with chunk-level and page-level metrics.

    Evaluates recall@1,3,5,10 (chunk + page), MRR, segmented by metadata.
    Skips is_impossible and requires_context questions.

    Args:
        db_path: SQLite vector store path.
        questions_file: v8 GS JSON file.
        model: Embedding model (pre-loaded).
        top_k: Max results to retrieve (default 10 for multi-K).
        use_hybrid: Use hybrid search (BM25 + vector).
        tolerance: Page tolerance for page-level recall.

    Returns:
        Dict with recall_at_k, mrr_mean, segmented results, error analysis.
    """
    from scripts.pipeline.embeddings import embed_query
    from scripts.pipeline.export_search import retrieve_hybrid, smart_retrieve
    from scripts.pipeline.utils import load_json

    gs_data = load_json(questions_file)
    questions = gs_data["questions"]

    k_values = [1, 3, 5, 10]
    results_detail: list[dict] = []

    for q in questions:
        parsed = _parse_question_v8(q)

        if parsed["is_impossible"] or not parsed["chunk_id"]:
            continue
        if parsed["requires_context"]:
            continue

        query_emb = embed_query(parsed["question"], model)

        if use_hybrid:
            retrieved = retrieve_hybrid(db_path, query_emb, parsed["question"], top_k=top_k)
        else:
            retrieved = smart_retrieve(db_path, query_emb, parsed["question"], top_k=top_k)

        # Compute metrics at each K
        chunk_recalls = {}
        page_recalls = {}
        for k in k_values:
            chunk_recalls[str(k)] = compute_recall_at_k_chunk(retrieved, parsed["chunk_id"], k=k)
            retrieved_pages = [r["page"] for r in retrieved[:k]]
            page_recalls[str(k)] = compute_recall_at_k(
                retrieved_pages, parsed["pages"], k=k, tolerance=tolerance
            )

        mrr = compute_mrr(retrieved, parsed["chunk_id"])

        detail = {
            "id": q["id"],
            "question": parsed["question"],
            "expected_chunk_id": parsed["chunk_id"],
            "expected_pages": parsed["pages"],
            "retrieved_chunks": [
                {"id": r["id"], "score": round(r.get("score", r.get("hybrid_score", 0)), 4), "page": r["page"]}
                for r in retrieved[:top_k]
            ],
            "chunk_recall": chunk_recalls,
            "page_recall": page_recalls,
            "mrr": mrr,
            "reasoning_class": parsed["reasoning_class"],
            "difficulty": parsed["difficulty"],
            "source_session": parsed["source_session"],
            "source_uuid": parsed["source_uuid"],
        }
        results_detail.append(detail)

    # Aggregate
    n = len(results_detail)
    recall_at_k_chunk = {}
    recall_at_k_page = {}
    for k in k_values:
        sk = str(k)
        recall_at_k_chunk[sk] = sum(r["chunk_recall"][sk] for r in results_detail) / n if n else 0.0
        recall_at_k_page[sk] = sum(r["page_recall"][sk] for r in results_detail) / n if n else 0.0

    mrr_mean = sum(r["mrr"] for r in results_detail) / n if n else 0.0

    # Failed questions (chunk recall@5 < 1.0)
    failed = [r for r in results_detail if r["chunk_recall"]["5"] < 1.0]

    # Segmented stats
    segments = _compute_segments(results_detail, k_values)

    return {
        "total_questions": n,
        "recall_at_k": {"chunk": recall_at_k_chunk, "page": recall_at_k_page},
        "mrr_mean": round(mrr_mean, 4),
        "failed_count": len(failed),
        "failed_questions": failed,
        "segments": segments,
        "config": {
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "tolerance": tolerance,
            "model": "EmbeddingGemma-300M-QAT",
            "gs_version": "v8.1.0",
        },
        "questions_detail": results_detail,
    }
```

**Step 4: Run tests**

Run: `python -m pytest scripts/pipeline/tests/test_recall.py::TestBenchmarkRecallV8 -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/pipeline/tests/test_recall.py
git commit -m "feat(recall): add benchmark_recall_v8 with multi-K chunk/page metrics"
```

---

### Task 5: Add `_compute_segments()` helper

**Files:**
- Modify: `scripts/pipeline/tests/test_recall.py`

**Step 1: Write the failing test**

```python
class TestComputeSegments:
    """Tests _compute_segments() aggregation."""

    def test_segments_by_reasoning_class(self):
        results = [
            {"chunk_recall": {"5": 1.0}, "reasoning_class": "summary", "difficulty": 0.3, "source_session": "s1", "source_uuid": "doc.pdf"},
            {"chunk_recall": {"5": 0.0}, "reasoning_class": "summary", "difficulty": 0.5, "source_session": "s1", "source_uuid": "doc.pdf"},
            {"chunk_recall": {"5": 1.0}, "reasoning_class": "fact_single", "difficulty": 0.2, "source_session": "s2", "source_uuid": "other.pdf"},
        ]
        segments = _compute_segments(results, [5])
        assert segments["reasoning_class"]["summary"]["count"] == 2
        assert segments["reasoning_class"]["summary"]["recall@5"] == 0.5
        assert segments["reasoning_class"]["fact_single"]["recall@5"] == 1.0
```

**Step 2: Run test to verify failure**

Run: `python -m pytest scripts/pipeline/tests/test_recall.py::TestComputeSegments -v`

**Step 3: Write implementation**

```python
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

        segments[key] = {}
        for group_name, group_results in sorted(groups.items()):
            n = len(group_results)
            seg = {"count": n}
            for k in k_values:
                sk = str(k)
                seg[f"recall@{k}"] = round(
                    sum(r["chunk_recall"][sk] for r in group_results) / n, 4
                )
            segments[key][group_name] = seg

    # Difficulty bands: easy (<0.33), medium (0.33-0.66), hard (>0.66)
    diff_groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        d = r["difficulty"]
        band = "easy" if d < 0.33 else ("medium" if d <= 0.66 else "hard")
        diff_groups[band].append(r)

    segments["difficulty"] = {}
    for band in ["easy", "medium", "hard"]:
        group_results = diff_groups.get(band, [])
        n = len(group_results)
        if n == 0:
            continue
        seg = {"count": n}
        for k in k_values:
            sk = str(k)
            seg[f"recall@{k}"] = round(
                sum(r["chunk_recall"][sk] for r in group_results) / n, 4
            )
        segments["difficulty"][band] = seg

    return segments
```

**Step 4: Run tests**

Run: `python -m pytest scripts/pipeline/tests/test_recall.py::TestComputeSegments -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/pipeline/tests/test_recall.py
git commit -m "feat(recall): add _compute_segments for segmented recall analysis"
```

---

### Task 6: Add `_print_baseline_report()` console output

**Files:**
- Modify: `scripts/pipeline/tests/test_recall.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify failure**

Run: `python -m pytest scripts/pipeline/tests/test_recall.py::TestPrintBaselineReport -v`

**Step 3: Write implementation**

```python
def _print_baseline_report(result: dict, variant_name: str) -> None:
    """Print formatted baseline recall report to console."""
    print(f"\n{'=' * 60}")
    print(f"BASELINE RECALL v8 — {variant_name}")
    print(f"{'=' * 60}")
    print(f"{result['total_questions']} questions | {result['config']['model']} | {result['config']['gs_version']}")
    print()

    chunk = result["recall_at_k"]["chunk"]
    page = result["recall_at_k"]["page"]

    print(f"{'Metric':<22} {'Chunk':>10} {'Page':>10}")
    print("-" * 44)
    for k in ["1", "3", "5", "10"]:
        if k in chunk:
            print(f"recall@{k:<16} {chunk[k]*100:>9.1f}% {page.get(k, 0)*100:>9.1f}%")
    print(f"{'MRR':<22} {result['mrr_mean']:>10.4f}")
    print()

    # ISO gate
    r5 = chunk.get("5", 0)
    print(f"ISO 25010 (recall@5 chunk >= 80%): {'PASS' if r5 >= 0.8 else 'FAIL'} ({r5*100:.1f}%)")
    print()

    # Segments
    for seg_name in ["reasoning_class", "difficulty"]:
        seg_data = result["segments"].get(seg_name, {})
        if not seg_data:
            continue
        print(f"--- By {seg_name} ---")
        for group, stats in sorted(seg_data.items()):
            r5_seg = stats.get("recall@5", 0)
            print(f"  {group:<20} {stats['count']:>4}Q  recall@5={r5_seg*100:.1f}%")
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
                print(f"    #{i+1}: {rc['id']} (score={rc['score']:.4f})")
        print()
```

**Step 4: Run tests**

Run: `python -m pytest scripts/pipeline/tests/test_recall.py::TestPrintBaselineReport -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/pipeline/tests/test_recall.py
git commit -m "feat(recall): add _print_baseline_report for formatted console output"
```

---

### Task 7: Add CLI `--baseline-v8` command and JSON export

**Files:**
- Modify: `scripts/pipeline/tests/test_recall.py` — update `main()` function

**Step 1: Write the failing test**

```python
class TestBaselineV8CLI:
    """Test CLI --baseline-v8 flag integration."""

    def test_baseline_v8_flag_exists(self):
        """Verify argparse accepts --baseline-v8."""
        import argparse
        parser = argparse.ArgumentParser()
        # This mimics what main() should set up
        parser.add_argument("--baseline-v8", action="store_true")
        args = parser.parse_args(["--baseline-v8"])
        assert args.baseline_v8 is True
```

**Step 2: Run test to verify it passes** (this is just an argparse test, it should pass immediately)

Run: `python -m pytest scripts/pipeline/tests/test_recall.py::TestBaselineV8CLI -v`

**Step 3: Update `main()` function**

Replace the existing `main()` (lines 378-415) with an updated version that supports `--baseline-v8`:

```python
def _export_baseline_json(result_vector: dict, result_hybrid: dict | None, output_dir: Path) -> Path:
    """Export baseline results to JSON."""
    import json
    from datetime import datetime, timezone

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    report = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "vector_only": result_vector,
    }
    if result_hybrid:
        report["hybrid"] = result_hybrid

    output_path = output_dir / f"baseline_v8_{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


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
        help="Run full v8 baseline (vector + hybrid, multi-K, segmented)",
    )
    args = parser.parse_args()

    from scripts.pipeline.embeddings import MODEL_ID, load_embedding_model

    print(f"Loading embedding model: {MODEL_ID}")
    model = load_embedding_model(MODEL_ID)

    if args.baseline_v8:
        db_path = CORPUS_DIR / "corpus_mode_b_fr.db"
        gs_file = DATA_DIR / "gold_standard_annales_fr_v8_adversarial.json"

        if not db_path.exists():
            print(f"[ERROR] DB not found: {db_path}")
            return 1
        if not gs_file.exists():
            print(f"[ERROR] GS not found: {gs_file}")
            return 1

        # Vector-only
        print("\n>>> Running vector-only benchmark...")
        result_vector = benchmark_recall_v8(
            db_path, gs_file, model, top_k=10, use_hybrid=False, tolerance=args.tolerance
        )
        _print_baseline_report(result_vector, "Vector-Only")

        # Hybrid
        print("\n>>> Running hybrid benchmark...")
        result_hybrid = benchmark_recall_v8(
            db_path, gs_file, model, top_k=10, use_hybrid=True, tolerance=args.tolerance
        )
        _print_baseline_report(result_hybrid, "Hybrid (BM25+Vector)")

        # Export JSON
        output_dir = PROJECT_ROOT / "data" / "benchmarks"
        output_path = _export_baseline_json(result_vector, result_hybrid, output_dir)
        print(f"\n[EXPORT] Results saved to {output_path}")

        # Decision
        r5_vec = result_vector["recall_at_k"]["chunk"]["5"]
        r5_hyb = result_hybrid["recall_at_k"]["chunk"]["5"]
        best = max(r5_vec, r5_hyb)
        if best >= 0.8:
            print("\n>> DECISION: recall@5 >= 80% — RAG pur + prompt engineering suffisant")
        elif best >= 0.6:
            print("\n>> DECISION: recall@5 60-80% — optimisations retrieval recommandees")
        else:
            print("\n>> DECISION: recall@5 < 60% — fine-tuning embeddings justifie")

        return 0 if best >= 0.8 else 1

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
```

**Step 4: Run all tests**

Run: `python -m pytest scripts/pipeline/tests/test_recall.py -v`
Expected: ALL PASS (existing + new)

**Step 5: Run ruff + mypy**

Run: `python -m ruff check scripts/pipeline/tests/test_recall.py`
Run: `python -m mypy scripts/pipeline/tests/test_recall.py`

**Step 6: Commit**

```bash
git add scripts/pipeline/tests/test_recall.py
git commit -m "feat(recall): add --baseline-v8 CLI with JSON export and decision logic"
```

---

### Task 8: Run the actual baseline

**Prerequisite:** Tasks 1-7 committed and tests green.

**Step 1: Verify DB and GS exist**

```bash
ls -la corpus/processed/corpus_mode_b_fr.db
ls -la tests/data/gold_standard_annales_fr_v8_adversarial.json
```

**Step 2: Run baseline**

```bash
python -m scripts.pipeline.tests.test_recall --baseline-v8
```

**Step 3: Review output**

Check console report and `data/benchmarks/baseline_v8_*.json`.

**Step 4: Commit results**

```bash
git add data/benchmarks/
git commit -m "docs(recall): baseline recall v8 results — vector-only and hybrid"
```

---

### Task 9: Final verification

**Step 1: Run full test suite with coverage**

```bash
python -m pytest scripts/ --cov=scripts --cov-config=.coveragerc --cov-fail-under=80 -v
```

**Step 2: Run pre-commit hooks**

```bash
python -m pre_commit run --all-files
```

**Step 3: Verify no regressions**

Expected: all existing tests still pass, coverage >= 80%.

**Step 4: Final commit if any fixups needed**

---

## File Summary

| File | Action |
|------|--------|
| `scripts/pipeline/tests/test_recall.py` | Extend with 7 new functions + 6 test classes |
| `data/benchmarks/baseline_v8_*.json` | New (generated output) |
| `docs/plans/2026-02-27-baseline-recall-v8-design.md` | Already created |
| `docs/plans/2026-02-27-baseline-recall-v8.md` | This plan |
