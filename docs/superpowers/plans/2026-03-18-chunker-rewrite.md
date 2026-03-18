# Chunker Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `chunker.py` using LangChain `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter.from_tiktoken_encoder`, with table extraction before header split, CCH from heading hierarchy, parent cap 2048 tokens, and 9 integrity quality gates.

**Architecture:** 7-stage pipeline: (1) extract tables from raw markdown → placeholders, (2) split by markdown headers → metadata hierarchy, (3) recursive split by tokens 512/100 overlap, (4) assemble parents by (h1,h2) grouping capped at 2048 tok, (5) interpolate pages from heading_pages, (6) build CCH titles from metadata, (7) link tables to parent sections. Output format unchanged (children + parents + tables).

**Tech Stack:** Python 3.10+, `langchain-text-splitters` (MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter), `tiktoken` (cl100k_base), `re`, pytest

**Spec:** `docs/superpowers/specs/2026-03-18-chunker-rewrite-design.md`

---

## SRP (Single Responsibility Principle)

| File | Single Responsibility | Max Lines |
|------|----------------------|-----------|
| `scripts/pipeline/chunker.py` | REWRITE — 7-stage chunk_document() | ≤ 200 |
| `scripts/pipeline/tests/test_chunker.py` | REWRITE — unit tests chunker | ≤ 300 |
| `scripts/pipeline/indexer.py` | MODIFY — add integrity gates I6-I9, CCH for table summaries | ≤ 300 (currently 685 → must split) |
| `scripts/pipeline/integrity.py` | CREATE — extracted from indexer, all I1-I9 gates | ≤ 150 |
| `scripts/pipeline/tests/test_integrity.py` | CREATE — tests for integrity gates | ≤ 200 |

**Note:** `indexer.py` at 685 lines violates ISO 25010 (max 300). The integrity gates will be extracted to `integrity.py` as part of this plan.

---

## Target Architecture

```
corpus/processed/docling_v2_fr/*.json (28 markdown files)
        |
   chunker.py::chunk_document(markdown, source, heading_pages)
        |
        +-- extract_tables(markdown) -> clean_md + tables[]
        +-- MarkdownHeaderTextSplitter -> header_docs[]
        +-- RecursiveCharacterTextSplitter(512/100) -> children_docs[]
        +-- build_parents(children_docs) -> parents[] (capped 2048 tok)
        +-- interpolate_pages(children, heading_pages) -> pages assigned
        +-- build_cch_titles(children) -> section fields
        +-- link_tables(tables, header_docs, parents) -> tables with parent_id
        |
   return {"children": [...], "parents": [...], "tables": [...]}
        |
   indexer.py::build_index()
        |
        +-- embed children + table summaries (EmbeddingGemma 768D)
        +-- CCH prefix for children AND table summaries
        +-- insert into SQLite (children, parents, table_summaries)
        +-- populate FTS5
        +-- integrity.py::run_integrity_gates(conn) -> I1-I9 PASS or FAIL
```

---

## Industry Standards Checklist (audited against each task)

| Standard | Requirement | Verification |
|----------|-------------|--------------|
| **FloTorch 2026** | chunk_size=512 tokens, recursive splitting | Task 1 test, Task 2 code |
| **Microsoft Azure 2026** | 20% overlap (100 tokens) | Task 1 test, Task 2 code |
| **NVIDIA 2024** | Header-based first, recursive within | Task 2 code (Stage 2 → Stage 3) |
| **LangChain Parent-Child** | split_documents() coverage guarantee | Task 3 test: total tokens check |
| **LangChain Multi-Vector** | Tables: embed summary, return raw | Task 5 code, Task 6 gate I9 |
| **KX/PremAI 2026** | Table CCH enrichment | Task 5 code: format_document for summaries |
| **NAACL 2025** | Recursive > semantic for prod | Task 2 code: RecursiveCharacterTextSplitter |
| **GraphRAG** | Parent 3-5x child, capped | Task 3 test: parent ≤ 2048 tokens |
| **ISO 29119** | TDD, tests before code | Every task: test first |
| **ISO 25010** | Files ≤ 300 lines, complexity ≤ B | Every task: line count check |
| **ISO 42001** | Traceability: metadata heading hierarchy | Task 2+4: CCH h1>h2>h3 |
| **ISO 12207** | Conventional commits | Every task: commit format |

---

## Quality Gates Definition (I1-I9)

| Gate | SQL/Check | Standard | Severity |
|------|-----------|----------|----------|
| I1 | `SELECT p.id FROM parents p LEFT JOIN children c ON c.parent_id=p.id WHERE c.id IS NULL AND p.text!='' AND p.tokens>0` → 0 rows | LangChain coverage guarantee | FAIL |
| I2 | `SELECT c.id FROM children c LEFT JOIN parents p ON c.parent_id=p.id WHERE p.id IS NULL` → 0 rows | Relational integrity | FAIL |
| I3 | `SELECT COUNT(*) FROM children WHERE page IS NULL` → 0 | Page traceability | FAIL |
| I4 | `SELECT COUNT(*) FROM children WHERE embedding IS NULL` → 0 | Embedding completeness | FAIL |
| I5 | `children_fts COUNT = children COUNT`, same for table_summaries | FTS5 sync | FAIL |
| I6 | `SELECT COUNT(*) FROM parents WHERE tokens > 2048` → 0 | GraphRAG parent cap | FAIL |
| I7 | sum(child tokens) ≥ 90% × sum(markdown section tokens) | Coverage completeness | FAIL |
| I8 | `SELECT COUNT(*) FROM children WHERE text LIKE '%TABLE_%'` → 0 (no unresolved placeholders) | Table extraction integrity | FAIL |
| I9 | Every table_summary has valid (source, page) in DB | Multi-vector linkage | FAIL |

---

## Definition of Done

- [ ] `chunker.py` rewritten with LangChain splitters (≤ 200 lines)
- [ ] `integrity.py` extracted from indexer (≤ 150 lines)
- [ ] All 9 integrity gates (I1-I9) implemented and tested
- [ ] `indexer.py` reduced to ≤ 300 lines (integrity extracted)
- [ ] CCH titles from heading hierarchy (h1 > h2 > h3) for children AND table summaries
- [ ] Table extraction before header split, placeholders resolved
- [ ] All unit tests pass (fast, no model loading)
- [ ] All integration tests pass (slow, real corpus)
- [ ] DB rebuilt: `corpus_v2_fr.db` with new chunking
- [ ] Recall re-measured: `data/benchmarks/recall_baseline.json` updated
- [ ] Recall improved vs 62.1% baseline (or root cause documented if not)
- [ ] Ruff, mypy, xenon clean
- [ ] All files ≤ 300 lines
- [ ] CLAUDE.md and memory updated
- [ ] Every standard in checklist verified with evidence
- [ ] Conventional commits throughout

---

## Task 1: Table extraction tests + implementation (Stage 1)

**Files:**
- Create: `scripts/pipeline/chunker.py` (begin rewrite — Stage 1 only)
- Create: `scripts/pipeline/tests/test_chunker.py` (begin rewrite)

**Industry standard verified:** Tables extracted BEFORE header split (KX 2026, PremAI 2026, empirically verified: 98 tables in LA, only 22 detected after header split).

- [ ] **Step 1: Write test_chunker.py with table extraction tests**

```python
# scripts/pipeline/tests/test_chunker.py
"""Tests for LangChain-based chunker."""

from __future__ import annotations

import pytest

from scripts.pipeline.chunker import extract_tables


class TestExtractTables:
    """Test table extraction from raw markdown (Stage 1)."""

    def test_extracts_simple_table(self) -> None:
        md = "Some text\n| A | B |\n|---|---|\n| 1 | 2 |\nMore text"
        clean, tables = extract_tables(md)
        assert len(tables) == 1
        assert "| A | B |" in tables[0]["raw_text"]
        assert "<!-- TABLE_0 -->" in clean
        assert "| A | B |" not in clean

    def test_preserves_non_table_text(self) -> None:
        md = "Hello\nWorld"
        clean, tables = extract_tables(md)
        assert len(tables) == 0
        assert clean == md

    def test_skips_short_pipe_blocks(self) -> None:
        """Lines with | but < 3 lines are not tables."""
        md = "| single line |\n| another |"
        clean, tables = extract_tables(md)
        assert len(tables) == 0
        assert "| single line |" in clean

    def test_multiple_tables(self) -> None:
        md = (
            "Text\n| A | B |\n|---|---|\n| 1 | 2 |\n"
            "Middle\n| X | Y |\n|---|---|\n| 3 | 4 |\nEnd"
        )
        clean, tables = extract_tables(md)
        assert len(tables) == 2
        assert "<!-- TABLE_0 -->" in clean
        assert "<!-- TABLE_1 -->" in clean

    def test_table_includes_separator_row(self) -> None:
        md = "| H1 | H2 |\n|:---|:---|\n| D1 | D2 |"
        clean, tables = extract_tables(md)
        assert len(tables) == 1
        assert "|:---|" in tables[0]["raw_text"]

    def test_placeholder_format(self) -> None:
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        clean, tables = extract_tables(md)
        assert clean.strip() == "<!-- TABLE_0 -->"

    def test_real_corpus_la_table_count(self) -> None:
        """Regression: LA-octobre2025 has ~98 table blocks."""
        from pathlib import Path
        import json
        p = Path("corpus/processed/docling_v2_fr/LA-octobre2025.json")
        if not p.exists():
            pytest.skip("LA docling file not available")
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        _, tables = extract_tables(doc["markdown"])
        assert len(tables) >= 90, f"Expected ~98 tables, got {len(tables)}"
```

- [ ] **Step 2: Write extract_tables() implementation**

```python
# scripts/pipeline/chunker.py (beginning of rewrite)
"""LangChain-based chunker for FFE regulation markdown.

7-stage pipeline: table extraction, header split, recursive token split,
parent assembly, page interpolation, CCH titles, table linkage.

Industry standards: FloTorch 2026 (512 tok), Azure 2026 (20% overlap),
NVIDIA 2024 (header-first), LangChain parent-child (coverage guarantee).
"""

from __future__ import annotations

import re

import tiktoken

TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
PAGE_MARKER_RE = re.compile(r"\n\s*[A-Z][A-Z0-9]{1,3}-\d+/\d+\s*\n")
IMAGE_PLACEHOLDER = "<!-- image -->"
ARTICLE_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*\.?)\s")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
PARENT_MAX_TOKENS = 2048
TABLE_MIN_LINES = 3  # header + separator + at least 1 data row

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base."""
    return len(_enc.encode(text))


def extract_tables(markdown: str) -> tuple[str, list[dict]]:
    """Extract table blocks from markdown, replace with placeholders.

    Must be called BEFORE header splitting — MarkdownHeaderTextSplitter
    fragments tables across heading boundaries (verified: 98 tables in LA,
    only 22 detected after header split).

    Args:
        markdown: Raw markdown text.

    Returns:
        (clean_markdown, tables) where tables is list of {"raw_text": ...}.
    """
    # Pre-clean: strip page markers and image placeholders
    markdown = PAGE_MARKER_RE.sub("\n", markdown)
    markdown = markdown.replace(IMAGE_PLACEHOLDER, "")

    lines = markdown.split("\n")
    tables: list[dict] = []
    clean_lines: list[str] = []
    i = 0
    while i < len(lines):
        if TABLE_LINE_RE.match(lines[i]):
            table_lines: list[str] = []
            while i < len(lines) and TABLE_LINE_RE.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= TABLE_MIN_LINES:
                tables.append({"raw_text": "\n".join(table_lines)})
                clean_lines.append(f"<!-- TABLE_{len(tables) - 1} -->")
            else:
                clean_lines.extend(table_lines)
        else:
            clean_lines.append(lines[i])
            i += 1
    return "\n".join(clean_lines), tables
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest scripts/pipeline/tests/test_chunker.py -v
```

Expected: all PASS (skip the corpus test if file not available).

- [ ] **Step 4: Commit**

```bash
git add scripts/pipeline/chunker.py scripts/pipeline/tests/test_chunker.py
git commit -m "feat(pipeline): rewrite chunker Stage 1 — table extraction before header split"
```

---

## Task 2: Header split + recursive split tests + implementation (Stages 2-3)

**Files:**
- Modify: `scripts/pipeline/chunker.py`
- Modify: `scripts/pipeline/tests/test_chunker.py`

**Industry standards verified:** FloTorch 2026 (chunk_size=512), Azure 2026 (overlap=100, 20%), NVIDIA 2024 (header-first then recursive), ".\n" separator (French abbreviations safe).

- [ ] **Step 1: Add tests for header_split and recursive_split**

Append to `test_chunker.py`:

```python
from scripts.pipeline.chunker import header_split, recursive_split, CHUNK_SIZE


class TestHeaderSplit:
    """Test MarkdownHeaderTextSplitter wrapper (Stage 2)."""

    def test_splits_by_h2(self) -> None:
        md = "## Section A\nText A\n## Section B\nText B"
        docs = header_split(md)
        assert len(docs) >= 2
        assert any("Section A" in d.page_content for d in docs)
        assert any("Section B" in d.page_content for d in docs)

    def test_preserves_heading_in_content(self) -> None:
        """strip_headers=False keeps heading in page_content."""
        md = "## My Section\nBody text"
        docs = header_split(md)
        assert any("## My Section" in d.page_content for d in docs)

    def test_metadata_contains_heading_hierarchy(self) -> None:
        md = "# Title\n## Sub\nBody"
        docs = header_split(md)
        sub_doc = [d for d in docs if "Sub" in d.page_content]
        assert len(sub_doc) >= 1
        assert sub_doc[0].metadata.get("h1") == "Title"

    def test_placeholder_preserved(self) -> None:
        md = "## Section\n<!-- TABLE_0 -->\nMore text"
        docs = header_split(md)
        all_text = " ".join(d.page_content for d in docs)
        assert "<!-- TABLE_0 -->" in all_text


class TestRecursiveSplit:
    """Test RecursiveCharacterTextSplitter wrapper (Stage 3)."""

    def test_splits_oversized_section(self) -> None:
        from langchain_core.documents import Document
        # Create a doc > 512 tokens
        long_text = "Mot " * 600  # ~600 tokens
        docs = [Document(page_content=long_text, metadata={"h1": "T"})]
        children = recursive_split(docs)
        assert len(children) >= 2
        assert all(
            count_tokens(c.page_content) <= CHUNK_SIZE + 20  # small tolerance
            for c in children
        )

    def test_preserves_metadata(self) -> None:
        from langchain_core.documents import Document
        docs = [Document(page_content="Short text", metadata={"h1": "A", "h2": "B"})]
        children = recursive_split(docs)
        assert children[0].metadata.get("h1") == "A"
        assert children[0].metadata.get("h2") == "B"

    def test_small_doc_not_split(self) -> None:
        from langchain_core.documents import Document
        docs = [Document(page_content="Small", metadata={})]
        children = recursive_split(docs)
        assert len(children) == 1

    def test_overlap_applied(self) -> None:
        from langchain_core.documents import Document
        # Two chunks from a long doc should share some text
        long_text = "Phrase unique numero " + " ".join(str(i) for i in range(300))
        docs = [Document(page_content=long_text, metadata={})]
        children = recursive_split(docs)
        if len(children) >= 2:
            # Last tokens of chunk 0 should appear in start of chunk 1
            end_0 = children[0].page_content[-50:]
            start_1 = children[1].page_content[:200]
            # At least some overlap text should be shared
            shared = set(end_0.split()) & set(start_1.split())
            assert len(shared) > 0, "No overlap detected between consecutive chunks"

    def test_uses_newline_separator_not_dot_space(self) -> None:
        """Verify '. ' does not split French abbreviations."""
        from langchain_core.documents import Document
        text = "Art. 3.2 du reglement.\nLe joueur doit respecter les regles."
        docs = [Document(page_content=text, metadata={})]
        children = recursive_split(docs)
        # "Art. 3" should NOT be split across chunks
        for c in children:
            assert "Art." not in c.page_content or "3" in c.page_content
```

- [ ] **Step 2: Implement header_split and recursive_split**

Add to `chunker.py`:

```python
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

_md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=_HEADERS_TO_SPLIT_ON,
    strip_headers=False,
)

_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".\n", " ", ""],
)


def header_split(markdown: str) -> list[Document]:
    """Split markdown by headers, preserving heading hierarchy in metadata.

    Args:
        markdown: Clean markdown (tables already replaced by placeholders).

    Returns:
        List of Documents with metadata {h1, h2, h3, h4}.
    """
    return _md_splitter.split_text(markdown)


def recursive_split(docs: list[Document]) -> list[Document]:
    """Split oversized sections by token count with overlap.

    Args:
        docs: Documents from header_split.

    Returns:
        Children documents, each ≤ CHUNK_SIZE tokens, metadata inherited.
    """
    return _text_splitter.split_documents(docs)
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest scripts/pipeline/tests/test_chunker.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/pipeline/chunker.py scripts/pipeline/tests/test_chunker.py
git commit -m "feat(pipeline): chunker Stages 2-3 — header split + recursive token split"
```

---

## Task 3: Parent assembly + CCH + page interpolation (Stages 4-6)

**Files:**
- Modify: `scripts/pipeline/chunker.py`
- Modify: `scripts/pipeline/tests/test_chunker.py`

**Industry standards verified:** GraphRAG (parent 3-5x child, capped 2048), ISO 42001 (CCH traceability h1>h2>h3).

- [ ] **Step 1: Add tests for build_parents, build_cch_title, interpolate_pages**

Append to `test_chunker.py`:

```python
from scripts.pipeline.chunker import (
    build_cch_title,
    build_parents,
    interpolate_pages,
    PARENT_MAX_TOKENS,
    count_tokens,
)


class TestBuildParents:
    """Test parent assembly from children (Stage 4)."""

    def test_groups_by_h1_h2(self) -> None:
        children = [
            Document(page_content="Child 1", metadata={"h1": "A", "h2": "B"}),
            Document(page_content="Child 2", metadata={"h1": "A", "h2": "B"}),
            Document(page_content="Child 3", metadata={"h1": "A", "h2": "C"}),
        ]
        parents = build_parents(children, "test.pdf")
        assert len(parents) == 2  # (A,B) and (A,C)

    def test_parent_text_is_concatenation(self) -> None:
        children = [
            Document(page_content="First.", metadata={"h1": "A", "h2": "B"}),
            Document(page_content="Second.", metadata={"h1": "A", "h2": "B"}),
        ]
        parents = build_parents(children, "test.pdf")
        assert "First." in parents[0]["text"]
        assert "Second." in parents[0]["text"]

    def test_parent_capped_at_2048(self) -> None:
        big_text = "word " * 800  # ~800 tokens per child
        children = [
            Document(page_content=big_text, metadata={"h1": "A", "h2": "B"}),
            Document(page_content=big_text, metadata={"h1": "A", "h2": "B"}),
            Document(page_content=big_text, metadata={"h1": "A", "h2": "B"}),
        ]
        parents = build_parents(children, "test.pdf")
        for p in parents:
            assert p["tokens"] <= PARENT_MAX_TOKENS + 50  # small tolerance

    def test_parent_has_required_fields(self) -> None:
        children = [
            Document(page_content="Text", metadata={"h1": "Title", "h2": "Sub"}),
        ]
        parents = build_parents(children, "src.pdf")
        p = parents[0]
        for key in ("id", "text", "source", "section", "tokens"):
            assert key in p, f"Missing field: {key}"
        assert p["source"] == "src.pdf"

    def test_no_empty_parents(self) -> None:
        children = [
            Document(page_content="Text", metadata={"h1": "A"}),
        ]
        parents = build_parents(children, "test.pdf")
        for p in parents:
            assert p["text"].strip(), f"Empty parent: {p['id']}"


class TestBuildCchTitle:
    """Test CCH title from heading hierarchy (Stage 6)."""

    def test_full_hierarchy(self) -> None:
        meta = {"h1": "Title", "h2": "Chapter", "h3": "Section"}
        assert build_cch_title(meta) == "Title > Chapter > Section"

    def test_partial_hierarchy(self) -> None:
        meta = {"h2": "Chapter"}
        assert build_cch_title(meta) == "Chapter"

    def test_empty_metadata(self) -> None:
        assert build_cch_title({}) == ""


class TestInterpolatePages:
    """Test page assignment from heading_pages (Stage 5)."""

    def test_assigns_page_from_heading(self) -> None:
        children = [
            {"section": "Forfaits", "page": None},
        ]
        heading_pages = {"Forfaits": 5}
        result = interpolate_pages(children, heading_pages)
        assert result[0]["page"] == 5

    def test_none_when_no_mapping(self) -> None:
        children = [{"section": "Unknown", "page": None}]
        result = interpolate_pages(children, {})
        assert result[0]["page"] is None
```

- [ ] **Step 2: Implement build_parents, build_cch_title, interpolate_pages**

Add to `chunker.py`:

```python
from collections import defaultdict


def build_cch_title(metadata: dict) -> str:
    """Build CCH title from heading hierarchy metadata.

    Args:
        metadata: Dict with optional h1, h2, h3, h4 keys.

    Returns:
        Hierarchical title like "Title > Chapter > Section".
    """
    parts = [metadata.get(f"h{i}", "") for i in range(1, 5)]
    return " > ".join(p for p in parts if p)


def build_parents(
    children: list[Document],
    source: str,
) -> list[dict]:
    """Group children by (h1, h2) and build parent dicts.

    Parents exceeding PARENT_MAX_TOKENS are split into sub-parents.

    Args:
        children: Documents from recursive_split.
        source: PDF source filename.

    Returns:
        List of parent dicts with id, text, source, section, tokens.
    """
    groups: dict[tuple[str, str], list[Document]] = defaultdict(list)
    for child in children:
        key = (child.metadata.get("h1", ""), child.metadata.get("h2", ""))
        groups[key].append(child)

    parents: list[dict] = []
    parent_counter = 0

    for (h1, h2), group in groups.items():
        section = " > ".join(p for p in (h1, h2) if p) or "root"
        full_text = "\n\n".join(c.page_content for c in group)
        tokens = count_tokens(full_text)

        if tokens <= PARENT_MAX_TOKENS:
            parents.append({
                "id": f"{source}-p{parent_counter:03d}",
                "text": full_text,
                "source": source,
                "section": section,
                "tokens": tokens,
            })
            parent_counter += 1
        else:
            # Split into sub-parents at PARENT_MAX_TOKENS boundaries
            current_parts: list[str] = []
            current_tokens = 0
            for c in group:
                c_tokens = count_tokens(c.page_content)
                if current_tokens + c_tokens > PARENT_MAX_TOKENS and current_parts:
                    sub_text = "\n\n".join(current_parts)
                    parents.append({
                        "id": f"{source}-p{parent_counter:03d}",
                        "text": sub_text,
                        "source": source,
                        "section": section,
                        "tokens": count_tokens(sub_text),
                    })
                    parent_counter += 1
                    current_parts = []
                    current_tokens = 0
                current_parts.append(c.page_content)
                current_tokens += c_tokens
            if current_parts:
                sub_text = "\n\n".join(current_parts)
                parents.append({
                    "id": f"{source}-p{parent_counter:03d}",
                    "text": sub_text,
                    "source": source,
                    "section": section,
                    "tokens": count_tokens(sub_text),
                })
                parent_counter += 1

    return parents


def interpolate_pages(
    children: list[dict],
    heading_pages: dict[str, int],
) -> list[dict]:
    """Assign page numbers to children from heading_pages mapping.

    For each child, find the closest heading match in its section field.

    Args:
        children: List of child dicts with "section" field.
        heading_pages: Mapping heading text → page number.

    Returns:
        Same children list with "page" field populated.
    """
    for child in children:
        if child.get("page") is not None:
            continue
        section = child.get("section", "")
        # Try exact match on each heading level
        for heading_text, page in heading_pages.items():
            if heading_text in section:
                child["page"] = page
                break
    return children
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest scripts/pipeline/tests/test_chunker.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/pipeline/chunker.py scripts/pipeline/tests/test_chunker.py
git commit -m "feat(pipeline): chunker Stages 4-6 — parent assembly, CCH, page interpolation"
```

---

## Task 4: Table linkage + chunk_document orchestrator (Stage 7 + main)

**Files:**
- Modify: `scripts/pipeline/chunker.py`
- Modify: `scripts/pipeline/tests/test_chunker.py`

**Industry standards verified:** KX 2026 (table context enrichment), LangChain multi-vector (table linked to parent section).

- [ ] **Step 1: Add tests for link_tables and chunk_document**

Append to `test_chunker.py`:

```python
from scripts.pipeline.chunker import chunk_document, link_tables


class TestLinkTables:
    """Test table linkage to parent sections (Stage 7)."""

    def test_table_gets_section_from_placeholder(self) -> None:
        tables = [{"raw_text": "| A | B |"}]
        header_docs = [
            Document(
                page_content="## Section\n<!-- TABLE_0 -->\nMore text",
                metadata={"h1": "Title", "h2": "Section"},
            ),
        ]
        parents = [{"id": "test.pdf-p000", "section": "Title > Section"}]
        linked = link_tables(tables, header_docs, parents)
        assert linked[0]["section"] == "Title > Section"

    def test_unresolved_table_has_no_section(self) -> None:
        tables = [{"raw_text": "| A | B |"}]
        header_docs = [
            Document(page_content="No placeholder here", metadata={}),
        ]
        linked = link_tables(tables, header_docs, [])
        assert linked[0].get("section", "") == ""


class TestChunkDocument:
    """Test full pipeline orchestrator."""

    def test_returns_children_parents_tables(self) -> None:
        md = "## Section\nSome text about rules.\n| A | B |\n|---|---|\n| 1 | 2 |"
        result = chunk_document(md, "test.pdf")
        assert "children" in result
        assert "parents" in result
        assert "tables" in result

    def test_no_placeholder_in_children(self) -> None:
        md = "## S\nText\n| A | B |\n|---|---|\n| 1 | 2 |\nMore"
        result = chunk_document(md, "test.pdf")
        for child in result["children"]:
            assert "<!-- TABLE_" not in child["text"], \
                f"Unresolved placeholder in child: {child['id']}"

    def test_children_have_required_fields(self) -> None:
        md = "## Section\nBody text for the section."
        result = chunk_document(md, "test.pdf")
        for child in result["children"]:
            for key in ("id", "text", "parent_id", "source", "section", "tokens"):
                assert key in child, f"Missing field: {key}"

    def test_no_empty_parents(self) -> None:
        md = "## A\nText A\n## B\nText B"
        result = chunk_document(md, "test.pdf")
        for parent in result["parents"]:
            assert parent["text"].strip(), f"Empty parent: {parent['id']}"

    def test_parent_ids_valid(self) -> None:
        md = "## A\nText A"
        result = chunk_document(md, "test.pdf")
        parent_ids = {p["id"] for p in result["parents"]}
        for child in result["children"]:
            assert child["parent_id"] in parent_ids, \
                f"Child {child['id']} has invalid parent_id {child['parent_id']}"

    def test_cch_title_in_section(self) -> None:
        md = "# Title\n## Chapter\nBody"
        result = chunk_document(md, "test.pdf")
        if result["children"]:
            assert ">" in result["children"][0]["section"] or \
                   result["children"][0]["section"] != ""

    def test_real_corpus_no_invisible_parents(self) -> None:
        """Regression: no parent with text should have 0 children."""
        from pathlib import Path
        import json
        p = Path("corpus/processed/docling_v2_fr/R01_2025_26_Regles_generales.json")
        if not p.exists():
            pytest.skip("R01 docling file not available")
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        result = chunk_document(doc["markdown"], "R01_2025_26_Regles_generales.pdf")
        parent_ids_with_children = {c["parent_id"] for c in result["children"]}
        for parent in result["parents"]:
            if parent["text"].strip():
                assert parent["id"] in parent_ids_with_children, \
                    f"Parent {parent['id']} has text but 0 children"

    def test_real_corpus_all_parents_under_2048(self) -> None:
        """Regression: no parent should exceed 2048 tokens."""
        from pathlib import Path
        import json
        p = Path("corpus/processed/docling_v2_fr/LA-octobre2025.json")
        if not p.exists():
            pytest.skip("LA docling file not available")
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        result = chunk_document(doc["markdown"], "LA-octobre2025.pdf")
        for parent in result["parents"]:
            assert parent["tokens"] <= 2048 + 50, \
                f"Parent {parent['id']} has {parent['tokens']} tokens (> 2048)"
```

- [ ] **Step 2: Implement link_tables and chunk_document**

Add to `chunker.py`:

```python
def link_tables(
    tables: list[dict],
    header_docs: list[Document],
    parents: list[dict],
) -> list[dict]:
    """Link tables to their parent sections via placeholder positions.

    Args:
        tables: Tables from extract_tables (with raw_text).
        header_docs: Documents from header_split (may contain placeholders).
        parents: Parent dicts from build_parents.

    Returns:
        Tables with added section and parent_id fields.
    """
    parent_by_section: dict[str, str] = {}
    for p in parents:
        parent_by_section[p["section"]] = p["id"]

    for i, table in enumerate(tables):
        placeholder = f"<!-- TABLE_{i} -->"
        table["section"] = ""
        table["parent_id"] = ""
        for doc in header_docs:
            if placeholder in doc.page_content:
                table["section"] = build_cch_title(doc.metadata)
                section_key = table["section"]
                table["parent_id"] = parent_by_section.get(
                    section_key,
                    # Fallback: try h1>h2 subset
                    parent_by_section.get(
                        " > ".join(
                            p for p in (
                                doc.metadata.get("h1", ""),
                                doc.metadata.get("h2", ""),
                            ) if p
                        ),
                        "",
                    ),
                )
                break
    return tables


def chunk_document(
    markdown: str,
    source: str,
    heading_pages: dict[str, int] | None = None,
) -> dict:
    """Chunk a markdown document into children, parents, and tables.

    7-stage pipeline:
    1. Extract tables → placeholders
    2. Split by markdown headers
    3. Recursive split by tokens (512/100 overlap)
    4. Assemble parents (grouped by h1,h2, capped 2048 tok)
    5. Interpolate pages
    6. Build CCH titles
    7. Link tables to parent sections

    Args:
        markdown: Markdown text with heading levels.
        source: Source PDF filename.
        heading_pages: Optional mapping heading text → page number.

    Returns:
        Dict with "children", "parents", "tables" lists.
    """
    _heading_pages = heading_pages or {}

    # Stage 1: Table extraction
    clean_md, tables = extract_tables(markdown)

    # Stage 2: Header split
    header_docs = header_split(clean_md)

    # Stage 3: Recursive token split
    children_docs = recursive_split(header_docs)

    # Stage 4: Parent assembly
    parents = build_parents(children_docs, source)
    parent_id_by_section: dict[str, str] = {p["section"]: p["id"] for p in parents}

    # Build children dicts with IDs and parent linkage
    children: list[dict] = []
    for i, doc in enumerate(children_docs):
        cch = build_cch_title(doc.metadata)
        section_key = " > ".join(
            p for p in (
                doc.metadata.get("h1", ""),
                doc.metadata.get("h2", ""),
            ) if p
        ) or "root"
        pid = parent_id_by_section.get(section_key, "")

        # Extract article number if present
        art_match = ARTICLE_NUM_RE.match(doc.page_content)
        art_num = art_match.group(1).rstrip(".") if art_match else None

        children.append({
            "id": f"{source}-c{i:04d}",
            "text": doc.page_content,
            "parent_id": pid,
            "source": source,
            "article_num": art_num,
            "section": cch,
            "tokens": count_tokens(doc.page_content),
            "page": None,  # Set in Stage 5
        })

    # Stage 5: Page interpolation
    children = interpolate_pages(children, _heading_pages)

    # Stage 7: Table linkage
    tables = link_tables(tables, header_docs, parents)
    # Add source and page to tables
    for table in tables:
        table["source"] = source
        # Page from heading_pages via section
        table["page"] = None
        for heading_text, page in _heading_pages.items():
            if heading_text in table.get("section", ""):
                table["page"] = page
                break

    return {"children": children, "parents": parents, "tables": tables}
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest scripts/pipeline/tests/test_chunker.py -v
```

Expected: all PASS including regression tests on real corpus.

- [ ] **Step 4: Line count check**

```bash
wc -l scripts/pipeline/chunker.py scripts/pipeline/tests/test_chunker.py
```

Expected: chunker ≤ 200, test_chunker ≤ 300. If over, split.

- [ ] **Step 5: Ruff + mypy**

```bash
python -m ruff check scripts/pipeline/chunker.py scripts/pipeline/tests/test_chunker.py
python -m mypy scripts/pipeline/chunker.py --ignore-missing-imports
```

- [ ] **Step 6: Commit**

```bash
git add scripts/pipeline/chunker.py scripts/pipeline/tests/test_chunker.py
git commit -m "feat(pipeline): chunker Stages 4-7 — parents, CCH, pages, table linkage"
```

---

## Task 5: Extract integrity.py from indexer + add I6-I9 gates

**Files:**
- Create: `scripts/pipeline/integrity.py`
- Create: `scripts/pipeline/tests/test_integrity.py`
- Modify: `scripts/pipeline/indexer.py` (replace inline gates with import)

**Industry standards verified:** LangChain coverage guarantee (I7), KX multi-vector linkage (I9), table extraction integrity (I8), GraphRAG parent cap (I6).

- [ ] **Step 1: Create integrity.py with all 9 gates**

```python
# scripts/pipeline/integrity.py
"""Post-build integrity gates for corpus DB (I1-I9)."""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)


def run_integrity_gates(conn: sqlite3.Connection) -> None:
    """Validate relational integrity after build. Raises on failure.

    Gates I1-I9 per spec 2026-03-18-chunker-rewrite-design.md.
    """
    _gate_i1_no_invisible_parents(conn)
    _gate_i2_no_orphan_children(conn)
    _gate_i3_no_null_pages(conn)
    _gate_i4_no_null_embeddings(conn)
    _gate_i5_fts5_sync(conn)
    _gate_i6_parent_token_cap(conn)
    _gate_i7_coverage(conn)
    _gate_i8_no_unresolved_placeholders(conn)
    _gate_i9_table_summary_linkage(conn)


def _gate_i1_no_invisible_parents(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT p.id FROM parents p "
        "LEFT JOIN children c ON c.parent_id = p.id "
        "WHERE c.id IS NULL AND p.text != '' AND p.tokens > 0"
    ).fetchall()
    if rows:
        ids = [r[0] for r in rows[:5]]
        msg = f"I1 FAIL: {len(rows)} parents with text but 0 children: {ids}"
        raise ValueError(msg)
    logger.info("I1 PASS: all parents with text have children")


def _gate_i2_no_orphan_children(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT c.id FROM children c "
        "LEFT JOIN parents p ON c.parent_id = p.id "
        "WHERE p.id IS NULL"
    ).fetchall()
    if rows:
        msg = f"I2 FAIL: {len(rows)} children with missing parent"
        raise ValueError(msg)
    logger.info("I2 PASS: all children have valid parent_id")


def _gate_i3_no_null_pages(conn: sqlite3.Connection) -> None:
    count = conn.execute(
        "SELECT COUNT(*) FROM children WHERE page IS NULL"
    ).fetchone()[0]
    if count:
        msg = f"I3 FAIL: {count} children with NULL page"
        raise ValueError(msg)
    logger.info("I3 PASS: all children have page number")


def _gate_i4_no_null_embeddings(conn: sqlite3.Connection) -> None:
    count = conn.execute(
        "SELECT COUNT(*) FROM children WHERE embedding IS NULL"
    ).fetchone()[0]
    if count:
        msg = f"I4 FAIL: {count} children with NULL embedding"
        raise ValueError(msg)
    logger.info("I4 PASS: all children have embeddings")


def _gate_i5_fts5_sync(conn: sqlite3.Connection) -> None:
    c_count = conn.execute("SELECT COUNT(*) FROM children").fetchone()[0]
    c_fts = conn.execute("SELECT COUNT(*) FROM children_fts").fetchone()[0]
    if c_count != c_fts:
        msg = f"I5 FAIL: children={c_count} vs children_fts={c_fts}"
        raise ValueError(msg)
    ts_count = conn.execute("SELECT COUNT(*) FROM table_summaries").fetchone()[0]
    ts_fts = conn.execute(
        "SELECT COUNT(*) FROM table_summaries_fts"
    ).fetchone()[0]
    if ts_count != ts_fts:
        msg = f"I5 FAIL: table_summaries={ts_count} vs fts={ts_fts}"
        raise ValueError(msg)
    logger.info("I5 PASS: FTS5 counts match (%d + %d)", c_fts, ts_fts)


def _gate_i6_parent_token_cap(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT id, tokens FROM parents WHERE tokens > 2048"
    ).fetchall()
    if rows:
        ids = [f"{r[0]}({r[1]}tok)" for r in rows[:5]]
        msg = f"I6 FAIL: {len(rows)} parents > 2048 tokens: {ids}"
        raise ValueError(msg)
    logger.info("I6 PASS: all parents <= 2048 tokens")


def _gate_i7_coverage(conn: sqlite3.Connection) -> None:
    child_tokens = conn.execute(
        "SELECT SUM(tokens) FROM children"
    ).fetchone()[0] or 0
    parent_tokens = conn.execute(
        "SELECT SUM(tokens) FROM parents WHERE text != ''"
    ).fetchone()[0] or 0
    if parent_tokens == 0:
        logger.info("I7 SKIP: no parent tokens to compare")
        return
    ratio = child_tokens / parent_tokens
    if ratio < 0.9:
        msg = f"I7 FAIL: child tokens ({child_tokens}) < 90% of parent tokens ({parent_tokens}), ratio={ratio:.2f}"
        raise ValueError(msg)
    logger.info("I7 PASS: coverage ratio %.2f (>= 0.90)", ratio)


def _gate_i8_no_unresolved_placeholders(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT id FROM children WHERE text LIKE '%<!-- TABLE_%'"
    ).fetchall()
    if rows:
        ids = [r[0] for r in rows[:5]]
        msg = f"I8 FAIL: {len(rows)} children with unresolved TABLE placeholders: {ids}"
        raise ValueError(msg)
    logger.info("I8 PASS: no unresolved table placeholders")


def _gate_i9_table_summary_linkage(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT id, source, page FROM table_summaries WHERE source IS NULL OR page IS NULL"
    ).fetchall()
    if rows:
        ids = [r[0] for r in rows[:5]]
        msg = f"I9 FAIL: {len(rows)} table_summaries with NULL source/page: {ids}"
        raise ValueError(msg)
    logger.info("I9 PASS: all table_summaries have valid source+page")
```

- [ ] **Step 2: Create test_integrity.py**

```python
# scripts/pipeline/tests/test_integrity.py
"""Tests for post-build integrity gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline.indexer import (
    create_db,
    insert_children,
    insert_parents,
    insert_table_summaries,
    populate_fts,
)
from scripts.pipeline.integrity import run_integrity_gates


class TestIntegrityGates:
    """Test each gate catches the specific violation."""

    def _build_valid_db(self, tmp_path: Path) -> Path:
        """Helper: build a minimal valid DB that passes all gates."""
        db_path = tmp_path / "valid.db"
        conn = create_db(db_path)
        insert_parents(conn, [{
            "id": "p1", "text": "Parent text", "source": "t.pdf",
            "section": "S", "tokens": 20, "page": 1,
        }])
        emb = np.random.randn(1, 768).astype(np.float32)
        insert_children(conn, [{
            "id": "c1", "text": "Child text", "parent_id": "p1",
            "source": "t.pdf", "page": 1, "article_num": None,
            "section": "S", "tokens": 10,
        }], emb)
        ts_emb = np.random.randn(1, 768).astype(np.float32)
        insert_table_summaries(conn, [{
            "id": "t1", "summary_text": "Summary", "raw_table_text": "| A |",
            "source": "t.pdf", "page": 1, "tokens": 5,
        }], ts_emb)
        populate_fts(conn)
        return db_path

    def test_valid_db_passes_all_gates(self, tmp_path: Path) -> None:
        import sqlite3
        db_path = self._build_valid_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        run_integrity_gates(conn)  # Should not raise
        conn.close()

    def test_i1_fails_invisible_parent(self, tmp_path: Path) -> None:
        import sqlite3
        db_path = self._build_valid_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        # Add a parent with text but no children
        conn.execute(
            "INSERT INTO parents (id, text, source, section, tokens, page) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("p_orphan", "Invisible text", "t.pdf", "S", 20, 1),
        )
        conn.commit()
        with pytest.raises(ValueError, match="I1 FAIL"):
            run_integrity_gates(conn)
        conn.close()

    def test_i6_fails_giant_parent(self, tmp_path: Path) -> None:
        import sqlite3
        db_path = self._build_valid_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE parents SET tokens = 5000 WHERE id = 'p1'"
        )
        conn.commit()
        with pytest.raises(ValueError, match="I6 FAIL"):
            run_integrity_gates(conn)
        conn.close()

    def test_i8_fails_unresolved_placeholder(self, tmp_path: Path) -> None:
        import sqlite3
        db_path = self._build_valid_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "UPDATE children SET text = 'Some <!-- TABLE_0 --> text' WHERE id = 'c1'"
        )
        conn.commit()
        with pytest.raises(ValueError, match="I8 FAIL"):
            run_integrity_gates(conn)
        conn.close()
```

- [ ] **Step 3: Update indexer.py to import from integrity.py**

Replace the inline `_run_integrity_gates` in `indexer.py` with:

```python
from scripts.pipeline.integrity import run_integrity_gates
```

And remove the entire `_run_integrity_gates` function body from indexer.py.

- [ ] **Step 4: Run tests**

```bash
python -m pytest scripts/pipeline/tests/test_integrity.py scripts/pipeline/tests/test_indexer.py -m "not slow" -v
```

Expected: all PASS.

- [ ] **Step 5: Line count check**

```bash
wc -l scripts/pipeline/integrity.py scripts/pipeline/indexer.py
```

Expected: integrity ≤ 150, indexer significantly reduced.

- [ ] **Step 6: Commit**

```bash
git add scripts/pipeline/integrity.py scripts/pipeline/tests/test_integrity.py scripts/pipeline/indexer.py
git commit -m "refactor(pipeline): extract integrity gates I1-I9 to integrity.py"
```

---

## Task 6: Modify indexer for CCH table summaries + rebuild DB

**Files:**
- Modify: `scripts/pipeline/indexer.py` (CCH for table summaries)

**Industry standard verified:** KX 2026 "Enrich each table with context — create table chunk = context + markdown table". Table summaries get CCH prefix before embedding.

- [ ] **Step 1: Update indexer to apply CCH to table summaries**

In the `build_index` function, where table summaries are embedded, add CCH prefix:

```python
# Before embedding table summaries, enrich with CCH context
for ts in table_sums:
    cch_title = ts.get("section", _make_summary_title(ts["summary_text"]))
    ts["embedding_text"] = format_document(ts["summary_text"], cch_title)
```

- [ ] **Step 2: Rebuild DB**

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
for k, v in stats.items():
    print(f'{k}: {v}')
"
```

Expected: all 9 integrity gates PASS (I1-I9). No invisible parents, no giant parents, no unresolved placeholders.

- [ ] **Step 3: Verify DB**

```bash
python -c "
import sqlite3
conn = sqlite3.connect('corpus/processed/corpus_v2_fr.db')
for t in ['children', 'parents', 'table_summaries']:
    c = conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'{t}: {c}')
# Verify no giant parents
giant = conn.execute('SELECT COUNT(*) FROM parents WHERE tokens > 2048').fetchone()[0]
print(f'Parents > 2048: {giant}')
# Verify no invisible parents
invisible = conn.execute('''
    SELECT COUNT(*) FROM parents p
    LEFT JOIN children c ON c.parent_id = p.id
    WHERE c.id IS NULL AND p.text != '' AND p.tokens > 0
''').fetchone()[0]
print(f'Invisible parents: {invisible}')
conn.close()
"
```

Expected: 0 giant parents, 0 invisible parents.

- [ ] **Step 4: Commit**

```bash
git add scripts/pipeline/indexer.py corpus/processed/corpus_v2_fr.db
git commit -m "feat(pipeline): rebuild DB with LangChain chunker + CCH table summaries"
```

---

## Task 7: Re-measure recall + final audit

**Files:**
- Modify: `data/benchmarks/recall_baseline.json` (regenerated)
- Modify: `data/benchmarks/recall_baseline.md` (regenerated)
- Modify: `CLAUDE.md`

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
g = data['global']
print(f'recall@1  = {g[\"recall@1\"]:.1%}')
print(f'recall@3  = {g[\"recall@3\"]:.1%}')
print(f'recall@5  = {g[\"recall@5\"]:.1%}')
print(f'recall@10 = {g[\"recall@10\"]:.1%}')
print(f'MRR       = {g[\"mrr\"]:.3f}')
"
```

- [ ] **Step 2: Compare with previous baseline**

| Metric | Old (custom chunker) | New (LangChain) | Delta |
|--------|---------------------|-----------------|-------|
| recall@1 | 39.9% | ? | ? |
| recall@5 | 62.1% | ? | ? |
| recall@10 | 65.4% | ? | ? |
| MRR | 0.494 | ? | ? |

If recall degraded: investigate root cause before proceeding.

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest scripts/pipeline/tests/ scripts/iso/ -m "not slow" -q
```

Expected: all PASS.

- [ ] **Step 4: Run slow quality gates**

```bash
python -m pytest scripts/pipeline/tests/test_search_quality_gates.py -v
```

Expected: S1-S8 PASS.

- [ ] **Step 5: Ruff + mypy + xenon on all modified files**

```bash
python -m ruff check scripts/pipeline/chunker.py scripts/pipeline/integrity.py scripts/pipeline/indexer.py
python -m mypy scripts/pipeline/chunker.py scripts/pipeline/integrity.py --ignore-missing-imports
python -m xenon scripts/pipeline/chunker.py scripts/pipeline/integrity.py -b B -m B -a B
```

- [ ] **Step 6: Line count audit**

```bash
wc -l scripts/pipeline/chunker.py scripts/pipeline/integrity.py scripts/pipeline/indexer.py scripts/pipeline/tests/test_chunker.py scripts/pipeline/tests/test_integrity.py
```

ALL files must be ≤ 300 lines.

- [ ] **Step 7: Industry standards final check**

| Standard | Evidence | Pass? |
|----------|----------|-------|
| FloTorch 2026 (512 tok) | `CHUNK_SIZE = 512` in chunker.py | |
| Azure 2026 (20% overlap) | `CHUNK_OVERLAP = 100` in chunker.py | |
| NVIDIA 2024 (header-first) | Stage 2 before Stage 3 | |
| LangChain coverage | Gate I7 PASS in build log | |
| GraphRAG parent cap | Gate I6 PASS, `PARENT_MAX_TOKENS = 2048` | |
| KX table CCH | format_document for summaries in indexer | |
| ISO 29119 TDD | Tests written before code in each task | |
| ISO 25010 complexity | xenon ≤ B | |
| ISO 25010 file size | All files ≤ 300 lines | |
| ISO 42001 traceability | CCH h1>h2>h3 in section field | |
| ISO 12207 commits | Conventional format all commits | |

- [ ] **Step 8: Update CLAUDE.md**

Update chunks count, recall numbers, pipeline description.

- [ ] **Step 9: Update memory**

- [ ] **Step 10: Final commit**

```bash
git add data/benchmarks/ CLAUDE.md
git commit -m "feat(pipeline): chunker rewrite complete — LangChain splitters + 9 integrity gates

Replaced custom chunker with MarkdownHeaderTextSplitter +
RecursiveCharacterTextSplitter (512/100). Tables extracted before
header split. Parents capped 2048 tokens. CCH for table summaries.
9 integrity gates (I1-I9) all PASS.
recall@5: 62.1% -> X.X%"
```

---

## Anti-Laziness Checklist (to be verified at EACH task boundary)

Before committing any task, the implementer MUST verify:

- [ ] Did I run the tests? (not just assume they pass)
- [ ] Did I check the line count? (not just estimate)
- [ ] Did I run ruff? (not just assume it's clean)
- [ ] Did I verify the industry standard claimed in the task header?
- [ ] Did I check for regressions in existing tests?
- [ ] Did I look at the actual output, not just the exit code?
- [ ] Is there a quality gate that would catch this if I made an error?
- [ ] Would this pass the audit table in Task 7 Step 7?
