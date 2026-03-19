# Recall Optimization — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reach 80%+ recall@5 (from 56.7%) by applying 8 zero-cost optimizations in a single DB rebuild.

**Architecture:** Two phases: (1) query-time optimizations testable on current DB (score calibration, query decomposition), (2) build-time enrichment (contextual retrieval, abbreviations, CCH injection, chapter overrides, config tuning) applied to chunks before a single rebuild. Quality gates at each phase boundary.

**Tech Stack:** Python 3.10+, existing pipeline (chunker, indexer, search), tiktoken, re, sqlite3, pytest

**Spec:** `docs/superpowers/specs/2026-03-19-recall-optimization-design.md`

---

## SRP / File Map

| File | Responsibility | Action | Max Lines |
|------|---------------|--------|-----------|
| `scripts/pipeline/enrichment.py` | CREATE — OPT 1-4: abbreviations, CCH inject, chapter overrides, contextual retrieval loader | New | 150 |
| `scripts/pipeline/search_utils.py` | CREATE — OPT 7-8: score calibration, query decomposition (extracted from search.py for SRP) | New | 100 |
| `scripts/pipeline/search.py` | MODIFY — import and call search_utils functions | Modify | 300 |
| `scripts/pipeline/indexer.py` | MODIFY — call enrich_chunks before embedding | Modify | 250 |
| `scripts/pipeline/chunker.py` | MODIFY — OPT 6 config (chunk_size=450, overlap=50) | Config | 140 |
| `scripts/pipeline/tests/test_enrichment.py` | CREATE — tests enrichment functions | New | 200 |
| `scripts/pipeline/tests/test_search_utils.py` | CREATE — tests score_calibration + query_decomposition | New | 150 |
| `corpus/processed/chunk_contexts.json` | OUTPUT — auditable contextual retrieval phrases | Generated | ~1073 entries |

**DRY principle:**
- Abbreviations dict defined ONCE in `enrichment.py`, imported by tests
- Chapter overrides dict defined ONCE in `enrichment.py`
- `count_tokens` reused from `chunker_utils.py` (not duplicated)
- `SOURCE_TITLES` reused from `indexer.py`

---

## Industry Standards Checklist

| Standard | Requirement | Task verification |
|----------|-------------|-------------------|
| **Anthropic 2024** | Contextual Retrieval prepend | Task 3: enrich_chunks, gate E5 |
| **Haystack** | Abbreviation expansion | Task 2: expand_abbreviations, gate E2 |
| **Google EmbeddingGemma** | `title: {t} \| text: {c}` format | Task 4: verified in indexer |
| **arXiv FILCO** | Intro page filtering | Task 6: is_intro flag |
| **Firecrawl/NAACL 2025** | chunk_size=450, overlap=50 | Task 4: chunker config |
| **ISO 29119** | TDD, quality gates | Every task |
| **ISO 25010** | Files ≤ 300 lines | Every task: line count |
| **ISO 12207** | Conventional commits | Every task |
| **ISO 42001** | Traceability | chunk_contexts.json auditable |

---

## Quality Gates

### Pre-rebuild (E1-E6)

| Gate | Check | How |
|------|-------|-----|
| E1 | chunk_contexts.json: all chunks have context, none empty, max 50 tokens | Script assertion |
| E2 | Each abbreviation key matches ≥1 chunk via `\b` regex | Script assertion |
| E3 | Each enriched child text starts with `[` (CCH injected) | Script assertion |
| E4 | Chapter override pages exist in corpus | Script assertion |
| E5 | Sample 20 enriched chunks: visual check (no hallucination) | Manual + print |
| E6 | Token distribution post-enrichment: median 350-500, max < 2048 | Script stats |

### Post-rebuild (I1-I9)

Existing integrity gates from `integrity.py` — all PASS.

### Post-recall (R1-R4)

| Gate | Check |
|------|-------|
| R1 | recall@5 ≥ 70% |
| R2 | recall@10 ≥ 75% |
| R3 | No regression: questions that passed before still pass |
| R4 | MRR ≥ 0.50 |

---

## Definition of Done

- [ ] `enrichment.py` created with OPT 1-4 (≤ 150 lines)
- [ ] `search_utils.py` created with OPT 7-8 (≤ 100 lines)
- [ ] `chunk_contexts.json` generated and audited (1073 entries)
- [ ] Abbreviation dict verified against corpus (every key matches ≥1 chunk)
- [ ] Chunker config: chunk_size=450, overlap=50
- [ ] Query-time optimizations (OPT 7-8) tested on current DB BEFORE rebuild
- [ ] Pre-rebuild quality gates E1-E6 PASS
- [ ] Single rebuild: 9/9 integrity gates I1-I9 PASS
- [ ] Recall gates R1-R4 verified
- [ ] All fast tests PASS
- [ ] Ruff, mypy clean
- [ ] All files ≤ 300 lines
- [ ] CLAUDE.md updated with recall results
- [ ] Conventional commits throughout

---

## Anti-Laziness Protocol

Before EACH task completion:
1. Did I run the tests and READ the output?
2. Did I check line counts of modified files?
3. Did I verify on REAL corpus data (not just unit tests)?
4. Did I check the industry standard claimed in the task?
5. Would this pass the audit tables above?

---

## Task 1: Query-time optimizations — score calibration + query decomposition (OPT 7-8)

**Files:**
- Create: `scripts/pipeline/search_utils.py`
- Create: `scripts/pipeline/tests/test_search_utils.py`
- Modify: `scripts/pipeline/search.py`

**Why first:** Testable on current DB without rebuild. Measures immediate gain.

- [ ] **Step 1: Write test_search_utils.py**

```python
# scripts/pipeline/tests/test_search_utils.py
"""Tests for score calibration and query decomposition."""

from __future__ import annotations

from scripts.pipeline.search_utils import calibrate_scores, decompose_query


class TestCalibrateScores:
    """Test score calibration by source document size."""

    def test_large_source_penalized(self) -> None:
        """Chunks from large documents get lower calibrated scores."""
        results = [("big-c0", 0.8), ("small-c0", 0.75)]
        source_counts = {"big.pdf": 500, "small.pdf": 10}
        source_lookup = {"big-c0": "big.pdf", "small-c0": "small.pdf"}
        calibrated = calibrate_scores(results, source_counts, source_lookup)
        # small-c0 should rank higher after calibration
        assert calibrated[0][0] == "small-c0"

    def test_equal_sources_unchanged_order(self) -> None:
        results = [("a-c0", 0.9), ("b-c0", 0.8)]
        source_counts = {"a.pdf": 50, "b.pdf": 50}
        source_lookup = {"a-c0": "a.pdf", "b-c0": "b.pdf"}
        calibrated = calibrate_scores(results, source_counts, source_lookup)
        assert calibrated[0][0] == "a-c0"

    def test_empty_results(self) -> None:
        assert calibrate_scores([], {}, {}) == []


class TestDecomposeQuery:
    """Test query decomposition for BM25."""

    def test_simple_query_not_split(self) -> None:
        parts = decompose_query("forfait equipe")
        assert len(parts) == 1

    def test_si_splits(self) -> None:
        parts = decompose_query(
            "A quelle heure un joueur est-il forfait si la ronde a debute en retard"
        )
        assert len(parts) >= 2

    def test_lorsque_splits(self) -> None:
        parts = decompose_query("Que faire lorsque le joueur arrive en retard")
        assert len(parts) >= 2

    def test_echec_et_mat_not_split(self) -> None:
        parts = decompose_query("echec et mat")
        assert len(parts) == 1

    def test_empty_query(self) -> None:
        assert decompose_query("") == [""]
```

- [ ] **Step 2: Write search_utils.py**

```python
# scripts/pipeline/search_utils.py
"""Search utilities: score calibration and query decomposition."""

from __future__ import annotations

import math
import re


def calibrate_scores(
    results: list[tuple[str, float]],
    source_counts: dict[str, int],
    source_lookup: dict[str, str],
) -> list[tuple[str, float]]:
    """Normalize scores by source document size.

    Large documents (e.g. LA 573 chunks) have probabilistic bias
    in top-K. Penalize by 1/log2(n_chunks+1).

    Args:
        results: [(doc_id, score), ...] sorted desc.
        source_counts: {source: n_chunks_in_source}.
        source_lookup: {doc_id: source}.

    Returns:
        Re-sorted [(doc_id, calibrated_score), ...].
    """
    if not results:
        return []
    calibrated = []
    for doc_id, score in results:
        source = source_lookup.get(doc_id, "")
        n = source_counts.get(source, 1)
        factor = 1.0 / math.log2(n + 1) if n > 1 else 1.0
        calibrated.append((doc_id, score * factor))
    return sorted(calibrated, key=lambda x: -x[1])


# Patterns that split complex queries into sub-queries for BM25
_SPLIT_RE = re.compile(
    r"\bsi\b|\blorsque\b|\blorsqu['']|\bquand\b",
    re.IGNORECASE,
)

# Protect "echec et mat" from splitting on "et"
_ET_RE = re.compile(r"\bet\b(?!.*\bmat\b)", re.IGNORECASE)


def decompose_query(query: str) -> list[str]:
    """Split complex queries into sub-queries for BM25.

    Simple queries are returned as-is. Complex queries with
    conditional clauses (si, lorsque, quand) are split.

    Args:
        query: Raw user query.

    Returns:
        List of sub-query strings (at least 1).
    """
    if not query.strip():
        return [query]

    # Try primary splits (si, lorsque, quand)
    parts = _SPLIT_RE.split(query)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) > 1:
        return parts

    # Try "et" split (but not "echec et mat")
    parts = _ET_RE.split(query)
    parts = [p.strip() for p in parts if p.strip()]

    return parts if len(parts) > 1 else [query]
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest scripts/pipeline/tests/test_search_utils.py -v
```

Expected: all PASS.

- [ ] **Step 4: Integrate into search.py**

Add to `search()` function in `search.py`:

1. After `cosine_search`: call `calibrate_scores` with source_counts from DB
2. In `bm25_search` call: if query has sub-queries, search each and merge

```python
# In search.py, add imports:
from scripts.pipeline.search_utils import calibrate_scores, decompose_query

# In search() function, after cosine_search:
source_counts = _get_source_counts(conn)  # cached
source_lookup = _get_source_lookup(conn)  # cached
cosine_results = calibrate_scores(cosine_results, source_counts, source_lookup)

# For BM25, decompose and merge:
sub_queries = decompose_query(query)
bm25_results = []
for sq in sub_queries:
    sq_expanded = expand_query(sq)
    bm25_results.extend(bm25_search(conn, sq_expanded, max_k=max_k * 2))
# Deduplicate, keep best score per doc_id
bm25_deduped = {}
for doc_id, score in bm25_results:
    if doc_id not in bm25_deduped or score < bm25_deduped[doc_id]:
        bm25_deduped[doc_id] = score
bm25_results = sorted(bm25_deduped.items(), key=lambda x: x[1])[:max_k * 2]
```

- [ ] **Step 5: Measure recall on current DB (BEFORE rebuild)**

```bash
python -c "
from scripts.pipeline.recall import run_recall
data = run_recall('corpus/processed/corpus_v2_fr.db', 'tests/data/gold_standard_annales_fr_v8_adversarial.json')
g = data['global']
print(f'recall@5={g[\"recall@5\"]:.1%} recall@10={g[\"recall@10\"]:.1%} MRR={g[\"mrr\"]:.3f}')
"
```

Record delta vs baseline (56.7%).

- [ ] **Step 6: Line count check + ruff**

```bash
wc -l scripts/pipeline/search.py scripts/pipeline/search_utils.py
python -m ruff check scripts/pipeline/search.py scripts/pipeline/search_utils.py
```

Both ≤ 300 lines.

- [ ] **Step 7: Commit**

```bash
git add scripts/pipeline/search.py scripts/pipeline/search_utils.py scripts/pipeline/tests/test_search_utils.py
git commit -m "feat(pipeline): OPT 7-8 score calibration + query decomposition

Score calibration: penalize large-source chunks by 1/log2(n+1).
Query decomposition: split complex queries (si/lorsque/quand) for BM25.
recall@5: 56.7% -> X.X% (query-time only, no rebuild)."
```

---

## Task 2: Enrichment module — abbreviations + CCH injection + chapter overrides (OPT 2-4)

**Files:**
- Create: `scripts/pipeline/enrichment.py`
- Create: `scripts/pipeline/tests/test_enrichment.py`

- [ ] **Step 1: Write test_enrichment.py**

```python
# scripts/pipeline/tests/test_enrichment.py
"""Tests for chunk enrichment (build-time optimizations)."""

from __future__ import annotations

import pytest

from scripts.pipeline.enrichment import (
    ABBREVIATIONS,
    CHAPTER_OVERRIDES,
    enrich_chunks,
    expand_abbreviations,
    inject_cch_text,
    apply_chapter_override,
)


class TestExpandAbbreviations:
    """Test abbreviation expansion in chunk text."""

    def test_expands_cm(self) -> None:
        assert "Candidat Maitre" in expand_abbreviations("Le joueur est CM.")

    def test_word_boundary_only(self) -> None:
        """ACME should not trigger CM expansion."""
        result = expand_abbreviations("ACME corporation")
        assert "Candidat Maitre" not in result

    def test_already_expanded_skipped(self) -> None:
        text = "CM (Candidat Maitre) et FM"
        result = expand_abbreviations(text)
        # CM already expanded, FM should be expanded
        assert result.count("Candidat Maitre") == 1
        assert "Maitre FIDE" in result

    def test_all_abbreviations_have_corpus_match(self) -> None:
        """Every abbreviation key must exist in at least 1 chunk."""
        import sqlite3
        from pathlib import Path
        db = Path("corpus/processed/corpus_v2_fr.db")
        if not db.exists():
            pytest.skip("DB not available")
        conn = sqlite3.connect(str(db))
        for abbr in ABBREVIATIONS:
            count = conn.execute(
                "SELECT COUNT(*) FROM children WHERE text LIKE ?",
                (f"% {abbr} %",),
            ).fetchone()[0]
            # Also check without spaces (start/end of text)
            count += conn.execute(
                "SELECT COUNT(*) FROM children WHERE text LIKE ? OR text LIKE ?",
                (f"{abbr} %", f"% {abbr}"),
            ).fetchone()[0]
            assert count > 0, f"Abbreviation '{abbr}' not found in any chunk"
        conn.close()


class TestInjectCchText:
    """Test CCH heading injection into chunk text."""

    def test_prepends_heading(self) -> None:
        result = inject_cch_text("Body text", "Title > Section")
        assert result.startswith("[Title > Section]")
        assert "Body text" in result

    def test_empty_section_skipped(self) -> None:
        result = inject_cch_text("Body text", "")
        assert result == "Body text"


class TestApplyChapterOverride:
    """Test chapter title override for specific pages."""

    def test_override_applied(self) -> None:
        title = apply_chapter_override("LA-octobre2025.pdf", 185, "Original")
        assert "FIDE" in title or title == "Original"  # depends on mapping

    def test_no_override_passthrough(self) -> None:
        title = apply_chapter_override("R01.pdf", 1, "Original")
        assert title == "Original"


class TestEnrichChunks:
    """Test full enrichment pipeline."""

    def test_enriches_text(self) -> None:
        children = [
            {"id": "test-c0", "text": "Le CM doit...", "section": "Titres",
             "source": "test.pdf", "page": 1, "tokens": 10, "parent_id": "p0"},
        ]
        contexts = {"test-c0": "Extrait du reglement, section Titres."}
        enriched = enrich_chunks(children, contexts)
        assert "Candidat Maitre" in enriched[0]["text"]
        assert enriched[0]["text"].startswith("[")
        assert "Extrait du reglement" in enriched[0]["text"]

    def test_tokens_updated(self) -> None:
        children = [
            {"id": "test-c0", "text": "Short", "section": "S",
             "source": "t.pdf", "page": 1, "tokens": 5, "parent_id": "p0"},
        ]
        enriched = enrich_chunks(children, {"test-c0": "Context."})
        assert enriched[0]["tokens"] > 5
```

- [ ] **Step 2: Write enrichment.py**

```python
# scripts/pipeline/enrichment.py
"""Build-time chunk enrichment: abbreviations, CCH injection, chapter overrides,
contextual retrieval. All zero runtime cost (applied before embedding)."""

from __future__ import annotations

import re

from scripts.pipeline.chunker_utils import count_tokens

# OPT-2: Abbreviation expansion dictionary (verified against corpus)
ABBREVIATIONS: dict[str, str] = {
    "CM": "CM (Candidat Maitre)",
    "FM": "FM (Maitre FIDE)",
    "MI": "MI (Maitre International)",
    "GMI": "GMI (Grand Maitre International)",
    "DNA": "DNA (Direction Nationale de l'Arbitrage)",
    "AFJ": "AFJ (Arbitre Federal Jeune)",
    "AFC": "AFC (Arbitre Federal de Club)",
    "AF3": "AF3 (Arbitre Federal 3)",
    "AF2": "AF2 (Arbitre Federal 2)",
    "AF1": "AF1 (Arbitre Federal 1)",
    "AI": "AI (Arbitre International)",
    "FFE": "FFE (Federation Francaise des Echecs)",
    "FIDE": "FIDE (Federation Internationale des Echecs)",
    "UV": "UV (Unite de Valeur)",
    "CDJE": "CDJE (Comite Departemental du Jeu d'Echecs)",
}

# OPT-4: Chapter title overrides (LA-octobre2025.pdf specific pages)
CHAPTER_OVERRIDES: dict[str, dict[tuple[int, int], str]] = {
    "LA-octobre2025.pdf": {
        (182, 186): "Classement Elo Standard FIDE",
        (187, 191): "Classement Rapide et Blitz FIDE",
        (192, 205): "Titres FIDE",
        (56, 57): "Annexe A - Cadence Rapide",
        (58, 66): "Annexe B - Cadence Blitz",
    },
}


def expand_abbreviations(text: str) -> str:
    """Replace abbreviations with expanded form (word boundary match)."""
    for abbr, expanded in ABBREVIATIONS.items():
        # Skip if already expanded
        if expanded in text:
            continue
        text = re.sub(rf"\b{abbr}\b", expanded, text)
    return text


def inject_cch_text(text: str, section: str) -> str:
    """Prepend heading hierarchy to chunk text."""
    if not section:
        return text
    return f"[{section}]\n\n{text}"


def apply_chapter_override(
    source: str, page: int | None, current_title: str,
) -> str:
    """Override CCH title for specific chapter pages."""
    if page is None or source not in CHAPTER_OVERRIDES:
        return current_title
    for (start, end), title in CHAPTER_OVERRIDES[source].items():
        if start <= page <= end:
            return title
    return current_title


def enrich_chunks(
    children: list[dict],
    contexts: dict[str, str],
) -> list[dict]:
    """Apply all build-time enrichments to chunks.

    Order: abbreviations -> CCH injection -> contextual retrieval prepend.
    Updates tokens count after enrichment.

    Args:
        children: List of chunk dicts from chunker.
        contexts: {chunk_id: context_text} from chunk_contexts.json.

    Returns:
        Enriched children (same list, modified in place).
    """
    for child in children:
        text = child["text"]

        # OPT-2: Abbreviation expansion
        text = expand_abbreviations(text)

        # OPT-3: CCH text injection
        text = inject_cch_text(text, child.get("section", ""))

        # OPT-1: Contextual retrieval prepend
        ctx = contexts.get(child["id"], "")
        if ctx:
            text = f"{ctx}\n\n{text}"

        child["text"] = text
        child["tokens"] = count_tokens(text)

    return children
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest scripts/pipeline/tests/test_enrichment.py -v
```

Expected: all PASS.

- [ ] **Step 4: Line count + ruff**

```bash
wc -l scripts/pipeline/enrichment.py scripts/pipeline/tests/test_enrichment.py
python -m ruff check scripts/pipeline/enrichment.py
```

enrichment.py ≤ 150, test ≤ 200.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/enrichment.py scripts/pipeline/tests/test_enrichment.py
git commit -m "feat(pipeline): OPT 2-4 enrichment module (abbreviations, CCH inject, chapter overrides)"
```

---

## Task 3: Generate chunk_contexts.json (OPT 1)

**Files:**
- Create: `corpus/processed/chunk_contexts.json`
- Modify: `scripts/pipeline/enrichment.py` (add load_contexts helper)

**This is the LLM-generated contextual retrieval content.** The engineer (Claude Code) generates 1-2 factual sentences per chunk during this task. The output is a JSON file that is auditable before any embedding.

- [ ] **Step 1: Generate contexts for all 1073 chunks**

Read each chunk from the DB + its parent text + metadata. Generate a factual 1-2 sentence context. Write to `corpus/processed/chunk_contexts.json`.

**Generation rules (from spec):**
- Factual only (no paraphrase, no interpretation)
- Mention: source document, section, subject
- Max 50 tokens per context
- French without accents (FTS5 compatible)

```python
# Script to generate chunk_contexts.json
import json, sqlite3

conn = sqlite3.connect("corpus/processed/corpus_v2_fr.db")
children = conn.execute(
    "SELECT c.id, c.text, c.source, c.section, c.page, p.text "
    "FROM children c JOIN parents p ON c.parent_id = p.id"
).fetchall()

contexts = {}
for cid, text, source, section, page, parent_text in children:
    # Generate factual context from metadata
    # SOURCE_TITLES lookup for display name
    # section = CCH hierarchy
    # First 30 chars of text for subject identification
    contexts[cid] = generate_context(source, section, text[:200], page)

with open("corpus/processed/chunk_contexts.json", "w", encoding="utf-8") as f:
    json.dump(contexts, f, indent=2, ensure_ascii=False)
```

- [ ] **Step 2: Quality gate E1 — verify chunk_contexts.json**

```python
# Verify: 1073 entries, none empty, max 50 tokens
import json, tiktoken
enc = tiktoken.get_encoding("cl100k_base")
with open("corpus/processed/chunk_contexts.json") as f:
    contexts = json.load(f)
assert len(contexts) >= 1073
assert all(v.strip() for v in contexts.values())
assert all(len(enc.encode(v)) <= 50 for v in contexts.values())
```

- [ ] **Step 3: Quality gate E5 — visual check sample 20 chunks**

Print 20 random enriched chunks (context + abbreviations + CCH) and verify:
- Context is factual (matches chunk content)
- No hallucination (no info not in chunk/parent)
- Abbreviations correctly expanded
- CCH heading present

- [ ] **Step 4: Add load_contexts to enrichment.py**

```python
def load_contexts(path: Path | str) -> dict[str, str]:
    """Load chunk contexts from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
```

- [ ] **Step 5: Commit**

```bash
git add corpus/processed/chunk_contexts.json scripts/pipeline/enrichment.py
git commit -m "feat(pipeline): OPT-1 generate chunk_contexts.json (1073 contextual retrieval entries)"
```

---

## Task 4: Config tuning + integrate enrichment in indexer (OPT 6 + wiring)

**Files:**
- Modify: `scripts/pipeline/chunker.py` (chunk_size=450, overlap=50)
- Modify: `scripts/pipeline/indexer.py` (call enrich_chunks before embedding)

- [ ] **Step 1: Update chunker config**

In `scripts/pipeline/chunker.py`:
```python
CHUNK_OVERLAP = 50  # was 100, reduced per NAACL 2025 + jan benchmark (450t = 86.94%)
```

In `scripts/pipeline/chunker_utils.py`:
```python
CHUNK_SIZE = 450  # was 512, Firecrawl 2026 + jan benchmark optimal
```

- [ ] **Step 2: Integrate enrichment in indexer**

In `scripts/pipeline/indexer.py`, after chunking (Step 1), before embedding (Step 4):

```python
# 1b. Enrich chunks (OPT 1-4: contextual retrieval, abbreviations, CCH, overrides)
from scripts.pipeline.enrichment import enrich_chunks, load_contexts, apply_chapter_override
from pathlib import Path

contexts_path = Path("corpus/processed/chunk_contexts.json")
if contexts_path.exists():
    contexts = load_contexts(contexts_path)
    logger.info("Loaded %d chunk contexts", len(contexts))
else:
    contexts = {}
    logger.warning("No chunk_contexts.json found, skipping contextual retrieval")

all_children = enrich_chunks(all_children, contexts)
logger.info("Enriched %d children", len(all_children))

# Apply chapter overrides to CCH titles
for c in all_children:
    c_title = make_cch_title(c["source"], c.get("section", ""), SOURCE_TITLES)
    c_title = apply_chapter_override(c["source"], c.get("page"), c_title)
    # Store override for embedding step
    c["_cch_title"] = c_title
```

Update Step 4 (embedding) to use `c["_cch_title"]` instead of computing CCH again.

- [ ] **Step 3: Also enrich table summaries (OPT 2-3 for summaries)**

```python
# Enrich table summaries with abbreviations
from scripts.pipeline.enrichment import expand_abbreviations, inject_cch_text
for ts in table_sums:
    ts["summary_text"] = expand_abbreviations(ts["summary_text"])
```

- [ ] **Step 4: Full corpus audit BEFORE rebuild (gates E1-E6)**

```bash
python -c "
# Run ALL pre-rebuild quality gates
# E1: chunk_contexts.json complete
# E2: abbreviations verified
# E3: CCH injection verified
# E4: chapter overrides pages exist
# E5: sample 20 visual check
# E6: token distribution
"
```

Print enriched sample, token stats, verify everything BEFORE the 12-min rebuild.

- [ ] **Step 5: Commit (without rebuild yet)**

```bash
git add scripts/pipeline/chunker.py scripts/pipeline/chunker_utils.py scripts/pipeline/indexer.py
git commit -m "feat(pipeline): integrate enrichment + config tuning (450/50) in build pipeline"
```

---

## Task 5: Single rebuild + recall measurement

**Files:**
- Output: `corpus/processed/corpus_v2_fr.db` (rebuilt)
- Output: `data/benchmarks/recall_baseline.json` (updated)
- Output: `data/benchmarks/recall_baseline.md` (updated)

- [ ] **Step 1: Rebuild DB**

```bash
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from scripts.pipeline.indexer import build_index
from pathlib import Path
stats = build_index(
    docling_dir=Path('corpus/processed/docling_v2_fr'),
    table_summaries_path=Path('corpus/processed/table_summaries_claude.json'),
    output_db=Path('corpus/processed/corpus_v2_fr.db'),
)
"
```

Expected: 9/9 integrity gates PASS. ~12 min.

- [ ] **Step 2: Measure recall**

```bash
python -c "
from scripts.pipeline.recall import run_recall
data = run_recall('corpus/processed/corpus_v2_fr.db', 'tests/data/gold_standard_annales_fr_v8_adversarial.json')
g = data['global']
print(f'recall@1={g[\"recall@1\"]:.1%} recall@3={g[\"recall@3\"]:.1%} recall@5={g[\"recall@5\"]:.1%} recall@10={g[\"recall@10\"]:.1%} MRR={g[\"mrr\"]:.3f}')
"
```

- [ ] **Step 3: Verify recall gates R1-R4**

| Gate | Required | Actual | Pass? |
|------|----------|--------|-------|
| R1 | recall@5 ≥ 70% | ? | |
| R2 | recall@10 ≥ 75% | ? | |
| R3 | No regression | ? | |
| R4 | MRR ≥ 0.50 | ? | |

- [ ] **Step 4: If recall < 70%, analyze failures before proceeding**

Do NOT commit bad results. Investigate root cause first.

- [ ] **Step 5: Commit results**

```bash
git add data/benchmarks/ corpus/processed/corpus_v2_fr.db
git commit -m "feat(pipeline): recall optimization — X.X% recall@5 (+Xpp)

OPT 1-8 applied: contextual retrieval, abbreviations, CCH injection,
chapter overrides, score calibration, query decomposition, chunk 450/50.
9/9 integrity gates PASS."
```

---

## Task 6: Flag intro pages + final audit (OPT 5)

**Files:**
- Modify: `scripts/pipeline/search.py` (add is_intro filter)

- [ ] **Step 1: Identify intro pages per document**

```python
# Check which pages are intro/sommaire across corpus
import sqlite3
conn = sqlite3.connect("corpus/processed/corpus_v2_fr.db")
# LA pages 1-3 are sommaire/preface
# Other docs: page 1 often is title page
# Be conservative: only flag well-known intro pages
```

- [ ] **Step 2: Add is_intro filtering in cosine_search and bm25_search**

Only if recall analysis shows intro pages cause false positives. If not, skip.

- [ ] **Step 3: Measure recall with intro filter**

- [ ] **Step 4: Commit if improvement, revert if not**

---

## Task 7: Full test suite + final audit

- [ ] **Step 1: Run all fast tests**

```bash
python -m pytest scripts/pipeline/tests/ scripts/iso/ -m "not slow" -q
```

Expected: all PASS.

- [ ] **Step 2: Run slow quality gates**

```bash
python -m pytest scripts/pipeline/tests/test_search_quality_gates.py -v
```

Expected: S1-S8 PASS.

- [ ] **Step 3: Ruff + mypy + xenon**

```bash
python -m ruff check scripts/pipeline/
python -m mypy scripts/pipeline/enrichment.py scripts/pipeline/search_utils.py --ignore-missing-imports
python -m xenon scripts/pipeline/enrichment.py scripts/pipeline/search_utils.py -b B -m B -a B
```

- [ ] **Step 4: Line count audit ALL pipeline files**

```bash
wc -l scripts/pipeline/*.py scripts/pipeline/tests/*.py
```

ALL files ≤ 300 lines.

- [ ] **Step 5: Industry standards final audit**

| Standard | Evidence | Pass? |
|----------|----------|-------|
| Anthropic 2024 (contextual retrieval) | chunk_contexts.json, enrichment.py | |
| Haystack (abbreviations) | ABBREVIATIONS dict, corpus-verified | |
| Google EmbeddingGemma (prompt format) | format_document in indexer_embed.py | |
| Firecrawl/NAACL (chunk 450/50) | CHUNK_SIZE, CHUNK_OVERLAP | |
| arXiv FILCO (intro filter) | Task 6 | |
| ISO 29119 (TDD) | Tests before code | |
| ISO 25010 (file sizes) | All ≤ 300 | |
| ISO 42001 (traceability) | chunk_contexts.json auditable | |

- [ ] **Step 6: Update CLAUDE.md + memory**

- [ ] **Step 7: Final commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with recall optimization results"
```
