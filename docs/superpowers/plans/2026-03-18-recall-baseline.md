# Recall Baseline Measurement — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure page-level recall@1/3/5/10 and MRR on 298 answerable GS questions with the current pipeline (default settings), output JSON + auto-generated Markdown report.

**Architecture:** Single script `recall.py` with pure functions: load GS → run search per question → page-level match → aggregate metrics → segment → error analysis → write JSON + Markdown. Tests mock `search()` to avoid model loading. One slow integration test on real DB.

**Tech Stack:** Python 3.10+, sqlite3, json, existing `scripts.pipeline.search.search()`, pytest

**Spec:** `docs/superpowers/specs/2026-03-18-recall-baseline-design.md`

---

## File Map

| File | Action | Responsibility | Est. lines |
|------|--------|---------------|------------|
| `scripts/pipeline/recall.py` | CREATE | Load GS, run search, page-match, metrics, reports | ~200 |
| `scripts/pipeline/tests/test_recall.py` | CREATE | Unit tests (mock search), edge cases | ~150 |

---

## Task 1: Unit tests for GS loading and page-level matching

**Files:**
- Create: `scripts/pipeline/tests/test_recall.py`
- Create: `scripts/pipeline/recall.py` (stubs)

- [ ] **Step 1: Write test_recall.py with tests for load_gs and page_match**

```python
# scripts/pipeline/tests/test_recall.py
"""Unit tests for recall measurement (no model loading)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.pipeline.recall import (
    load_gs,
    page_match,
)

GS_PATH = Path("tests/data/gold_standard_annales_fr_v8_adversarial.json")


class TestLoadGs:
    """Test GS loading and filtering."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_gs(self) -> None:
        if not GS_PATH.exists():
            pytest.skip("GS file not available")

    def test_loads_answerable_only(self) -> None:
        questions = load_gs(GS_PATH)
        assert len(questions) == 298

    def test_question_has_required_fields(self) -> None:
        questions = load_gs(GS_PATH)
        q = questions[0]
        assert "question" in q
        assert "expected_pages" in q
        assert "expected_docs" in q
        assert "reasoning_class" in q
        assert "difficulty" in q
        assert "id" in q

    def test_no_impossible_questions(self) -> None:
        questions = load_gs(GS_PATH)
        # All should be answerable — is_impossible filtered out
        for q in questions:
            assert len(q["expected_pages"]) >= 1
            assert len(q["expected_docs"]) >= 1


class TestPageMatch:
    """Test page-level matching logic."""

    def test_hit_when_page_matches(self) -> None:
        expected = [("doc.pdf", 10), ("doc.pdf", 11)]
        retrieved = [("doc.pdf", 10)]
        assert page_match(expected, retrieved) is True

    def test_miss_when_no_page_matches(self) -> None:
        expected = [("doc.pdf", 10)]
        retrieved = [("doc.pdf", 20), ("other.pdf", 10)]
        assert page_match(expected, retrieved) is False

    def test_miss_when_empty_retrieved(self) -> None:
        expected = [("doc.pdf", 10)]
        assert page_match(expected, []) is False

    def test_hit_with_multiple_expected(self) -> None:
        expected = [("doc.pdf", 10), ("doc.pdf", 11)]
        retrieved = [("doc.pdf", 11)]
        assert page_match(expected, retrieved) is True
```

- [ ] **Step 2: Write recall.py stubs to make imports pass**

```python
# scripts/pipeline/recall.py
"""Recall measurement: page-level recall@k and MRR on GS questions."""

from __future__ import annotations

import json
from pathlib import Path


def load_gs(gs_path: Path | str) -> list[dict]:
    """Load GS and return answerable questions with normalized fields.

    Args:
        gs_path: Path to gold_standard JSON file.

    Returns:
        List of dicts with keys: id, question, expected_docs,
        expected_pages, reasoning_class, difficulty.
    """
    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)

    questions = []
    for q in gs["questions"]:
        if q["content"]["is_impossible"]:
            continue
        prov = q["provenance"]
        clf = q["classification"]
        expected = list(zip(prov["docs"], prov["pages"], strict=True))
        questions.append({
            "id": q["id"],
            "question": q["content"]["question"],
            "expected_docs": prov["docs"],
            "expected_pages": prov["pages"],
            "expected_pairs": expected,
            "reasoning_class": clf.get("reasoning_class", "unknown"),
            "difficulty": clf.get("difficulty", 0.5),
        })
    return questions


def page_match(
    expected: list[tuple[str, int]],
    retrieved: list[tuple[str, int | None]],
) -> bool:
    """Check if any retrieved (source, page) matches expected.

    Args:
        expected: [(source, page), ...] from GS provenance.
        retrieved: [(source, page), ...] from search results.

    Returns:
        True if at least one match.
    """
    expected_set = set(expected)
    return any(pair in expected_set for pair in retrieved)
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest scripts/pipeline/tests/test_recall.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/pipeline/recall.py scripts/pipeline/tests/test_recall.py
git commit -m "test(pipeline): add recall measurement tests and stubs"
```

---

## Task 2: Metrics computation (evaluate + compute_metrics)

**Files:**
- Modify: `scripts/pipeline/tests/test_recall.py`
- Modify: `scripts/pipeline/recall.py`

- [ ] **Step 1: Add tests for evaluate_question and compute_metrics**

Append to `test_recall.py`:

```python
from scripts.pipeline.recall import compute_metrics, evaluate_question


class TestEvaluateQuestion:
    """Test single question evaluation."""

    def test_hit_at_rank_1(self) -> None:
        question = {
            "id": "q1",
            "expected_pairs": [("doc.pdf", 10)],
            "reasoning_class": "fact_single",
            "difficulty": 0.2,
        }
        # Simulate contexts: first context has matching page
        retrieved_pages = [
            [("doc.pdf", 10)],  # context 0 → hit
            [("doc.pdf", 20)],  # context 1
        ]
        result = evaluate_question(question, retrieved_pages)
        assert result["hit@1"] is True
        assert result["hit@3"] is True
        assert result["rank"] == 1

    def test_hit_at_rank_3(self) -> None:
        question = {
            "id": "q2",
            "expected_pairs": [("doc.pdf", 10)],
            "reasoning_class": "summary",
            "difficulty": 0.5,
        }
        retrieved_pages = [
            [("doc.pdf", 20)],  # context 0 → miss
            [("doc.pdf", 30)],  # context 1 → miss
            [("doc.pdf", 10)],  # context 2 → hit
        ]
        result = evaluate_question(question, retrieved_pages)
        assert result["hit@1"] is False
        assert result["hit@3"] is True
        assert result["hit@5"] is True
        assert result["rank"] == 3

    def test_miss(self) -> None:
        question = {
            "id": "q3",
            "expected_pairs": [("doc.pdf", 10)],
            "reasoning_class": "summary",
            "difficulty": 0.8,
        }
        retrieved_pages = [
            [("other.pdf", 5)],
        ]
        result = evaluate_question(question, retrieved_pages)
        assert result["hit@1"] is False
        assert result["hit@10"] is False
        assert result["rank"] == 0


class TestComputeMetrics:
    """Test metrics aggregation."""

    def test_perfect_recall(self) -> None:
        results = [
            {"hit@1": True, "hit@3": True, "hit@5": True, "hit@10": True,
             "rank": 1, "reasoning_class": "fact_single", "difficulty": 0.2},
            {"hit@1": True, "hit@3": True, "hit@5": True, "hit@10": True,
             "rank": 1, "reasoning_class": "summary", "difficulty": 0.5},
        ]
        metrics = compute_metrics(results)
        assert metrics["global"]["recall@1"] == 1.0
        assert metrics["global"]["mrr"] == 1.0

    def test_partial_recall(self) -> None:
        results = [
            {"hit@1": True, "hit@3": True, "hit@5": True, "hit@10": True,
             "rank": 1, "reasoning_class": "fact_single", "difficulty": 0.2},
            {"hit@1": False, "hit@3": False, "hit@5": False, "hit@10": False,
             "rank": 0, "reasoning_class": "summary", "difficulty": 0.8},
        ]
        metrics = compute_metrics(results)
        assert metrics["global"]["recall@1"] == 0.5
        assert metrics["global"]["recall@5"] == 0.5

    def test_segments_by_reasoning_class(self) -> None:
        results = [
            {"hit@1": True, "hit@3": True, "hit@5": True, "hit@10": True,
             "rank": 1, "reasoning_class": "fact_single", "difficulty": 0.2},
            {"hit@1": False, "hit@3": False, "hit@5": False, "hit@10": False,
             "rank": 0, "reasoning_class": "summary", "difficulty": 0.8},
        ]
        metrics = compute_metrics(results)
        assert metrics["segments"]["reasoning_class"]["fact_single"]["recall@1"] == 1.0
        assert metrics["segments"]["reasoning_class"]["summary"]["recall@1"] == 0.0

    def test_segments_by_difficulty(self) -> None:
        results = [
            {"hit@1": True, "hit@3": True, "hit@5": True, "hit@10": True,
             "rank": 1, "reasoning_class": "fact_single", "difficulty": 0.1},
            {"hit@1": False, "hit@3": False, "hit@5": False, "hit@10": False,
             "rank": 0, "reasoning_class": "summary", "difficulty": 0.9},
        ]
        metrics = compute_metrics(results)
        assert metrics["segments"]["difficulty"]["easy"]["recall@1"] == 1.0
        assert metrics["segments"]["difficulty"]["hard"]["recall@1"] == 0.0
```

- [ ] **Step 2: Implement evaluate_question and compute_metrics**

Add to `recall.py`:

```python
def evaluate_question(
    question: dict,
    retrieved_pages: list[list[tuple[str, int | None]]],
) -> dict:
    """Evaluate a single question against retrieved contexts.

    Args:
        question: GS question with expected_pairs, reasoning_class, difficulty.
        retrieved_pages: For each context (ranked), list of (source, page) pairs.

    Returns:
        Dict with hit@1, hit@3, hit@5, hit@10, rank, reasoning_class, difficulty.
    """
    expected = question["expected_pairs"]
    rank = 0
    for i, context_pages in enumerate(retrieved_pages):
        if page_match(expected, context_pages):
            rank = i + 1
            break

    return {
        "id": question["id"],
        "hit@1": rank == 1,
        "hit@3": 1 <= rank <= 3,
        "hit@5": 1 <= rank <= 5,
        "hit@10": 1 <= rank <= 10,
        "rank": rank,
        "reasoning_class": question["reasoning_class"],
        "difficulty": question["difficulty"],
    }


def _difficulty_bucket(difficulty: float) -> str:
    """Map difficulty float to bucket name."""
    if difficulty < 0.33:
        return "easy"
    if difficulty < 0.66:
        return "medium"
    return "hard"


def _aggregate(results: list[dict]) -> dict:
    """Aggregate hit rates and MRR from a list of eval results."""
    if not results:
        return {"count": 0, "recall@1": 0.0, "recall@3": 0.0,
                "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
    n = len(results)
    return {
        "count": n,
        "recall@1": sum(r["hit@1"] for r in results) / n,
        "recall@3": sum(r["hit@3"] for r in results) / n,
        "recall@5": sum(r["hit@5"] for r in results) / n,
        "recall@10": sum(r["hit@10"] for r in results) / n,
        "mrr": sum((1 / r["rank"] if r["rank"] > 0 else 0) for r in results) / n,
    }


def compute_metrics(results: list[dict]) -> dict:
    """Compute global and segmented recall metrics.

    Args:
        results: List of evaluate_question outputs.

    Returns:
        Dict with global, segments.reasoning_class, segments.difficulty.
    """
    global_metrics = _aggregate(results)

    # Segment by reasoning_class
    by_class: dict[str, list[dict]] = {}
    for r in results:
        by_class.setdefault(r["reasoning_class"], []).append(r)

    # Segment by difficulty bucket
    by_diff: dict[str, list[dict]] = {}
    for r in results:
        bucket = _difficulty_bucket(r["difficulty"])
        by_diff.setdefault(bucket, []).append(r)

    return {
        "global": global_metrics,
        "segments": {
            "reasoning_class": {k: _aggregate(v) for k, v in sorted(by_class.items())},
            "difficulty": {k: _aggregate(v) for k, v in sorted(by_diff.items())},
        },
    }
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest scripts/pipeline/tests/test_recall.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/pipeline/recall.py scripts/pipeline/tests/test_recall.py
git commit -m "feat(pipeline): add recall evaluation and metrics computation"
```

---

## Task 3: Report generation (JSON + Markdown)

**Files:**
- Modify: `scripts/pipeline/tests/test_recall.py`
- Modify: `scripts/pipeline/recall.py`

- [ ] **Step 1: Add tests for error_analysis and write_reports**

Append to `test_recall.py`:

```python
from scripts.pipeline.recall import error_analysis, write_markdown, write_json


class TestErrorAnalysis:
    """Test error case extraction."""

    def test_returns_top_n_failures(self) -> None:
        results = [
            {"id": f"q{i}", "hit@10": i < 5, "rank": i + 1 if i < 5 else 0,
             "reasoning_class": "summary", "difficulty": 0.5}
            for i in range(10)
        ]
        questions = [
            {"id": f"q{i}", "question": f"Question {i}",
             "expected_docs": ["d.pdf"], "expected_pages": [i],
             "expected_pairs": [("d.pdf", i)],
             "reasoning_class": "summary", "difficulty": 0.5}
            for i in range(10)
        ]
        errors = error_analysis(results, questions, n=3)
        assert len(errors) == 3
        # All should be failures (hit@10 == False)
        assert all(e["hit@10"] is False for e in errors)

    def test_returns_empty_if_no_failures(self) -> None:
        results = [
            {"id": "q1", "hit@10": True, "rank": 1,
             "reasoning_class": "fact_single", "difficulty": 0.2}
        ]
        questions = [
            {"id": "q1", "question": "Q1", "expected_docs": ["d.pdf"],
             "expected_pages": [1], "expected_pairs": [("d.pdf", 1)],
             "reasoning_class": "fact_single", "difficulty": 0.2}
        ]
        errors = error_analysis(results, questions, n=20)
        assert len(errors) == 0


class TestWriteReports:
    """Test report file generation."""

    def test_write_json_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "test.json"
        data = {"global": {"recall@5": 0.75}, "metadata": {}}
        write_json(data, out)
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["global"]["recall@5"] == 0.75

    def test_write_markdown_has_yaml_header(self, tmp_path: Path) -> None:
        out = tmp_path / "test.md"
        data = {
            "metadata": {"generated": "2026-03-18", "model": "test"},
            "global": {"recall@1": 0.5, "recall@3": 0.6,
                       "recall@5": 0.7, "recall@10": 0.8, "mrr": 0.55},
            "segments": {"reasoning_class": {}, "difficulty": {}},
            "errors": [],
        }
        write_markdown(data, out)
        content = out.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert "recall@5" in content
```

- [ ] **Step 2: Implement error_analysis, write_json, write_markdown**

Add to `recall.py`:

```python
from datetime import datetime, timezone


def error_analysis(
    results: list[dict],
    questions: list[dict],
    n: int = 20,
) -> list[dict]:
    """Extract top-n failure cases for diagnostic.

    Args:
        results: List of evaluate_question outputs.
        questions: Original question dicts (for text/expected).
        n: Number of failures to return.

    Returns:
        List of failure dicts with question text and expected info.
    """
    q_by_id = {q["id"]: q for q in questions}
    failures = [r for r in results if not r["hit@10"]]
    failures = failures[:n]
    return [
        {
            "id": r["id"],
            "question": q_by_id[r["id"]]["question"][:80],
            "expected_docs": q_by_id[r["id"]]["expected_docs"],
            "expected_pages": q_by_id[r["id"]]["expected_pages"],
            "reasoning_class": r["reasoning_class"],
            "difficulty": r["difficulty"],
            "hit@10": False,
        }
        for r in failures
    ]


def write_json(data: dict, path: Path | str) -> None:
    """Write recall results to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_markdown(data: dict, path: Path | str) -> None:
    """Write recall results to Markdown with YAML header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = data["metadata"]
    g = data["global"]

    lines = [
        "---",
        *[f"{k}: {v}" for k, v in meta.items()],
        "---",
        "",
        "# Recall Baseline — Pipeline v2",
        "",
        "## Global",
        "",
        "| Metrique | Score |",
        "|----------|-------|",
        *[f"| {k} | {v:.1%} |" for k, v in g.items() if k.startswith("recall")],
        f"| MRR | {g['mrr']:.3f} |",
        "",
    ]

    # Segments
    for seg_name, seg_data in data["segments"].items():
        lines.append(f"## Par {seg_name}")
        lines.append("")
        lines.append("| Bucket | Count | R@1 | R@3 | R@5 | R@10 |")
        lines.append("|--------|-------|-----|-----|-----|------|")
        for bucket, vals in seg_data.items():
            lines.append(
                f"| {bucket} | {vals['count']} "
                f"| {vals['recall@1']:.1%} | {vals['recall@3']:.1%} "
                f"| {vals['recall@5']:.1%} | {vals['recall@10']:.1%} |"
            )
        lines.append("")

    # Errors
    errors = data.get("errors", [])
    if errors:
        lines.append("## Top echecs (recall@10 = 0)")
        lines.append("")
        lines.append("| # | Question | Expected | Class |")
        lines.append("|---|----------|----------|-------|")
        for i, e in enumerate(errors, 1):
            q_short = e["question"][:50]
            pages = e["expected_pages"]
            lines.append(f"| {i} | {q_short} | {e['expected_docs'][0]} p{pages} | {e['reasoning_class']} |")
        lines.append("")

    # Decision
    r5 = g["recall@5"]
    if r5 >= 0.8:
        decision = "Prompt engineering suffisant"
    elif r5 >= 0.6:
        decision = "Optimisations retrieval necessaires"
    else:
        decision = "Fine-tuning embeddings justifie"
    lines.append("## Decision")
    lines.append("")
    lines.append(f"recall@5 = {r5:.1%} → **{decision}**")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest scripts/pipeline/tests/test_recall.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/pipeline/recall.py scripts/pipeline/tests/test_recall.py
git commit -m "feat(pipeline): add recall report generation (JSON + Markdown)"
```

---

## Task 4: Main entry point + run on real corpus

**Files:**
- Modify: `scripts/pipeline/recall.py`
- Modify: `scripts/pipeline/tests/test_recall.py`

- [ ] **Step 1: Implement run_recall main function**

Add to `recall.py`:

```python
import logging
import sqlite3

logger = logging.getLogger(__name__)


def _get_context_pages(
    conn: sqlite3.Connection,
    children_matched: list[str],
) -> list[tuple[str, int | None]]:
    """Lookup (source, page) for a list of child_ids."""
    pages = []
    for cid in children_matched:
        row = conn.execute(
            "SELECT source, page FROM children WHERE id = ?", (cid,)
        ).fetchone()
        if row:
            pages.append((row[0], row[1]))
    return pages


def run_recall(
    db_path: Path | str,
    gs_path: Path | str,
    output_dir: Path | str = "data/benchmarks",
) -> dict:
    """Run full recall measurement and write reports.

    Args:
        db_path: Path to corpus_v2_fr.db.
        gs_path: Path to gold_standard JSON.
        output_dir: Directory for output files.

    Returns:
        Full results dict (same as JSON output).
    """
    from scripts.pipeline.indexer import DEFAULT_MODEL_ID, load_model
    from scripts.pipeline.search import search

    output_dir = Path(output_dir)
    questions = load_gs(gs_path)
    logger.info("Loaded %d answerable questions", len(questions))

    model = load_model()
    logger.info("Model loaded: %s", DEFAULT_MODEL_ID)

    conn = sqlite3.connect(str(db_path))
    results = []

    try:
        for i, q in enumerate(questions):
            sr = search(db_path, q["question"], model=model)

            # For each context, get (source, page) of its children
            retrieved_pages = []
            for ctx in sr.contexts:
                ctx_pages = _get_context_pages(conn, ctx.children_matched)
                retrieved_pages.append(ctx_pages)

            result = evaluate_question(q, retrieved_pages)
            results.append(result)

            if (i + 1) % 50 == 0:
                logger.info("Progress: %d/%d", i + 1, len(questions))
    finally:
        conn.close()

    logger.info("Evaluation complete: %d questions", len(results))

    metrics = compute_metrics(results)
    errors = error_analysis(results, questions, n=20)

    data = {
        "metadata": {
            "generated": datetime.now(tz=timezone.utc).isoformat(),
            "pipeline": "hybrid cosine+BM25 RRF, adaptive-k largest-gap",
            "model": DEFAULT_MODEL_ID,
            "db": str(Path(db_path).name),
            "gs_version": "9.0.0",
            "match_level": "page",
            "settings": {"min_score": 0.005, "max_k": 10, "rrf_k": 60},
            "questions_total": len(questions),
        },
        **metrics,
        "errors": errors,
        "per_question": results,
    }

    write_json(data, output_dir / "recall_baseline.json")
    write_markdown(data, output_dir / "recall_baseline.md")
    logger.info("Reports written to %s", output_dir)

    r5 = metrics["global"]["recall@5"]
    logger.info("recall@5 = %.1f%%", r5 * 100)

    return data
```

- [ ] **Step 2: Add slow integration test**

Append to `test_recall.py`:

```python
from scripts.pipeline.recall import run_recall


@pytest.mark.slow
class TestRunRecall:
    """Integration test on real corpus."""

    DB_PATH = Path("corpus/processed/corpus_v2_fr.db")

    @pytest.fixture(autouse=True)
    def _skip_if_no_db(self) -> None:
        if not self.DB_PATH.exists():
            pytest.skip("corpus_v2_fr.db not available")
        if not GS_PATH.exists():
            pytest.skip("GS file not available")

    def test_run_produces_reports(self, tmp_path: Path) -> None:
        data = run_recall(self.DB_PATH, GS_PATH, output_dir=tmp_path)
        assert data["metadata"]["questions_total"] == 298
        assert 0.0 <= data["global"]["recall@5"] <= 1.0
        assert (tmp_path / "recall_baseline.json").exists()
        assert (tmp_path / "recall_baseline.md").exists()
        md = (tmp_path / "recall_baseline.md").read_text(encoding="utf-8")
        assert md.startswith("---\n")
```

- [ ] **Step 3: Run fast tests only**

```bash
python -m pytest scripts/pipeline/tests/test_recall.py -m "not slow" -v
```

Expected: all PASS.

- [ ] **Step 4: Ruff + mypy**

```bash
python -m ruff check scripts/pipeline/recall.py scripts/pipeline/tests/test_recall.py
python -m mypy scripts/pipeline/recall.py --ignore-missing-imports
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/recall.py scripts/pipeline/tests/test_recall.py
git commit -m "feat(pipeline): add run_recall entry point with integration test"
```

---

## Task 5: Run baseline measurement on real corpus

- [ ] **Step 1: Run recall measurement**

```bash
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from scripts.pipeline.recall import run_recall
data = run_recall(
    'corpus/processed/corpus_v2_fr.db',
    'tests/data/gold_standard_annales_fr_v8_adversarial.json',
)
r5 = data['global']['recall@5']
print(f'recall@5 = {r5:.1%}')
"
```

This takes ~10-15 min (298 queries × embedding + search). Model loading is one-shot, embedding cache kicks in after first query.

- [ ] **Step 2: Verify outputs**

```bash
python -c "
import json
with open('data/benchmarks/recall_baseline.json', encoding='utf-8') as f:
    d = json.load(f)
g = d['global']
print(f'recall@1  = {g[\"recall@1\"]:.1%}')
print(f'recall@3  = {g[\"recall@3\"]:.1%}')
print(f'recall@5  = {g[\"recall@5\"]:.1%}')
print(f'recall@10 = {g[\"recall@10\"]:.1%}')
print(f'MRR       = {g[\"mrr\"]:.3f}')
print(f'Errors    = {len(d[\"errors\"])}')
"
```

- [ ] **Step 3: Read Markdown report**

```bash
head -40 data/benchmarks/recall_baseline.md
```

Verify: YAML header present, tables formatted, decision line at bottom.

- [ ] **Step 4: Update CLAUDE.md with results**

Add recall results to the "Ce qui fonctionne" section.

- [ ] **Step 5: Commit results + docs**

```bash
git add data/benchmarks/recall_baseline.json data/benchmarks/recall_baseline.md CLAUDE.md
git commit -m "feat(pipeline): baseline recall measurement on 298 GS questions

recall@5 = X.X% (page-level, default settings)
Pipeline: hybrid cosine+BM25 RRF, adaptive-k largest-gap
Model: embeddinggemma-300m-qat-q4_0-unquantized"
```

---

## Definition of Done

- [ ] `scripts/pipeline/recall.py` written and tested (<=300 lines)
- [ ] `scripts/pipeline/tests/test_recall.py` written (<=300 lines)
- [ ] Fast tests PASS
- [ ] Ruff + mypy clean
- [ ] `data/benchmarks/recall_baseline.json` generated
- [ ] `data/benchmarks/recall_baseline.md` generated with YAML header
- [ ] CLAUDE.md updated with recall results
- [ ] Decision documented: prompt eng / retrieval optim / fine-tune
- [ ] Conventional commits
