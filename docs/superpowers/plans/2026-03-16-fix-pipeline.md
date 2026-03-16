# Fix Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruire le pipeline retrieval avec vrais heading levels, chunks 400-512 tokens, parents pour contexte LLM, et table summaries dans l'index.

**Architecture:** 4 scripts sequentiels : extract (docling + hierarchical post-processor) → chunker (structure-aware) → indexer (EmbeddingGemma-300M → SQLite) → search (cosine brute-force → parents). TDD, un commit par tache.

**Tech Stack:** Python 3.10+, docling 2.68+, docling-hierarchical-pdf, tiktoken, sentence-transformers, numpy, sqlite3

**Spec:** `docs/superpowers/specs/2026-03-16-fix-pipeline-design.md`

---

## Task 0: Setup

**Files:**
- Modify: `requirements.txt`
- Create: `scripts/pipeline/__init__.py`
- Create: `scripts/pipeline/tests/__init__.py`
- Create: `scripts/pipeline/tests/conftest.py`

- [ ] **Step 1: Install docling-hierarchical-pdf**

```bash
pip install git+https://github.com/krrome/docling-hierarchical-pdf.git
```

- [ ] **Step 2: Verify install**

```bash
python -c "from hierarchical.postprocessor import ResultPostprocessor; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Create pipeline package structure**

```bash
mkdir -p scripts/pipeline/tests
touch scripts/pipeline/__init__.py
touch scripts/pipeline/tests/__init__.py
```

- [ ] **Step 4: Create conftest.py with shared fixtures**

Create `scripts/pipeline/tests/conftest.py`:

```python
"""Shared fixtures for pipeline tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


CORPUS_FR_DIR = Path("corpus/fr")
DOCLING_FR_DIR = Path("corpus/processed/docling_fr")
TABLE_SUMMARIES_PATH = Path("corpus/processed/table_summaries_claude.json")


@pytest.fixture
def sample_markdown_hierarchical() -> str:
    """Markdown with real heading levels (post-hierarchical-pdf)."""
    return (
        "# REGLES GENERALES\n\n"
        "## 1. Licences\n\n"
        "Les joueurs doivent etre licencies.\n\n"
        "### 1.1. Licence A\n\n"
        "Pour cadence >= 60 min.\n\n"
        "### 1.2. Licence B\n\n"
        "Pour cadence < 60 min.\n\n"
        "## 2. Statut\n\n"
        "### 2.1. Nationalite\n\n"
        "En cas de reserve sur la nationalite, le club justifie dans 15 jours.\n\n"
        "## 3. Forfaits\n\n"
    )


@pytest.fixture
def sample_markdown_flat() -> str:
    """Markdown with all ## headings (current docling output, fallback)."""
    return (
        "## REGLES GENERALES\n\n"
        "## 1. Licences\n\n"
        "Les joueurs doivent etre licencies.\n\n"
        "## 1.1. Licence A\n\n"
        "Pour cadence >= 60 min.\n\n"
        "## 1.2. Licence B\n\n"
        "Pour cadence < 60 min.\n\n"
        "## 2. Statut\n\n"
        "## 2.1. Nationalite\n\n"
        "En cas de reserve sur la nationalite, le club justifie dans 15 jours.\n\n"
        "## 3. Forfaits\n\n"
    )
```

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/__init__.py scripts/pipeline/tests/__init__.py scripts/pipeline/tests/conftest.py
git commit -m "feat(pipeline): setup package structure and test fixtures"
```

---

## Task 1: Extract — Re-extraction avec heading levels

**Files:**
- Create: `scripts/pipeline/extract.py`
- Create: `scripts/pipeline/tests/test_extract.py`

- [ ] **Step 1: Write test — verify extraction produces multi-level headings**

Create `scripts/pipeline/tests/test_extract.py`:

```python
"""Tests for PDF extraction with hierarchical headings."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.pipeline.extract import extract_pdf, extract_corpus


class TestExtractPdf:
    """Test single PDF extraction."""

    @pytest.mark.slow
    def test_r01_has_multiple_heading_levels(self):
        """R01 should produce at least 2 different heading levels."""
        pdf_path = Path("corpus/fr/Compétitions/R01_2025_26_Regles_generales.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        md = result["markdown"]
        levels = set(re.findall(r"^(#{1,6}) ", md, re.MULTILINE))
        assert len(levels) >= 2, f"Expected multiple heading levels, got: {levels}"

    @pytest.mark.slow
    def test_r01_text_faithful(self):
        """Extracted text should contain known R01 content."""
        pdf_path = Path("corpus/fr/Compétitions/R01_2025_26_Regles_generales.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        md = result["markdown"]
        assert "Licences" in md
        assert "Forfait" in md or "forfait" in md
        assert "Commission Technique" in md

    @pytest.mark.slow
    def test_r01_tables_extracted(self):
        """R01 has 2 tables (categories + cadences)."""
        pdf_path = Path("corpus/fr/Compétitions/R01_2025_26_Regles_generales.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        assert len(result["tables"]) >= 2

    @pytest.mark.slow
    def test_heading_levels_match_structure(self):
        """Article numbers should map to correct heading levels.

        Expected: top-level articles (1., 2., 3.) at h2,
        sub-articles (2.1, 2.2) at h3.
        """
        pdf_path = Path("corpus/fr/Compétitions/R01_2025_26_Regles_generales.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF not available")
        result = extract_pdf(pdf_path)
        md = result["markdown"]
        # Top-level articles should be h1 or h2
        assert re.search(r"^#{1,2} .*1\. Licences", md, re.MULTILINE)
        # Sub-articles should be deeper
        sub = re.search(r"^(#{1,6}) .*2\.1", md, re.MULTILINE)
        top = re.search(r"^(#{1,6}) .*2\. Statut", md, re.MULTILINE)
        if sub and top:
            assert len(sub.group(1)) > len(top.group(1)), \
                f"Sub-article 2.1 (h{len(sub.group(1))}) should be deeper than 2. (h{len(top.group(1))})"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest scripts/pipeline/tests/test_extract.py -v -m "not slow" 2>/dev/null; echo "---"
python -m pytest scripts/pipeline/tests/test_extract.py -v -k "test_r01_has_multiple" 2>&1 | tail -5
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.pipeline.extract'`

- [ ] **Step 3: Implement extract.py**

Create `scripts/pipeline/extract.py`:

```python
"""PDF extraction with hierarchical heading levels.

Uses docling + docling-hierarchical-pdf to produce markdown
with real heading levels (#, ##, ###) from PDF structure.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

# Try hierarchical post-processor, fallback gracefully
try:
    from hierarchical.postprocessor import ResultPostprocessor
    HAS_HIERARCHICAL = True
except ImportError:
    HAS_HIERARCHICAL = False
    logger.warning("docling-hierarchical-pdf not installed, headings will be flat")


def extract_pdf(pdf_path: Path) -> dict:
    """Extract a single PDF with hierarchical headings.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with keys: markdown, tables, source, pages.
    """
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    if HAS_HIERARCHICAL:
        ResultPostprocessor(result).process()

    doc = result.document
    markdown = doc.export_to_markdown()

    # Extract tables
    tables = []
    for table_ix, table in enumerate(doc.tables):
        table_md = table.export_to_markdown()
        tables.append({
            "id": f"{pdf_path.stem}-table{table_ix}",
            "source": pdf_path.name,
            "text": table_md,
        })

    return {
        "markdown": markdown,
        "tables": tables,
        "source": pdf_path.name,
    }


def extract_corpus(corpus_dir: Path, output_dir: Path) -> list[dict]:
    """Extract all PDFs in corpus_dir (recursively).

    Args:
        corpus_dir: Root directory containing PDFs.
        output_dir: Where to save JSON extractions.

    Returns:
        List of extraction results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(corpus_dir.rglob("*.pdf"))
    results = []

    for pdf_path in pdfs:
        logger.info("Extracting %s", pdf_path.name)
        try:
            result = extract_pdf(pdf_path)
            out_path = output_dir / f"{pdf_path.stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            results.append(result)
        except Exception:
            logger.exception("Failed to extract %s", pdf_path.name)

    logger.info("Extracted %d/%d PDFs", len(results), len(pdfs))
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    corpus_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("corpus/fr")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("corpus/processed/docling_v2_fr")
    extract_corpus(corpus_dir, output_dir)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest scripts/pipeline/tests/test_extract.py -v -k "test_r01_has_multiple" 2>&1 | tail -10
```

Expected: PASS (if docling-hierarchical-pdf works) or FAIL with heading level detail (if it doesn't — fallback path).

- [ ] **Step 5: Audit heading levels against PDF**

Read the extracted markdown for R01 and LA (sample). Manually verify that heading levels match the visual structure of the PDFs. Document findings.

This is a **manual verification step** — read the output, compare with PDFs, note any discrepancies.

If hierarchical post-processor produces bad results: remove it and implement fallback (parse article numbering from flat `##` headings). Update spec accordingly.

- [ ] **Step 6: Run full extraction on corpus FR (if audit passes)**

```bash
python scripts/pipeline/extract.py corpus/fr corpus/processed/docling_v2_fr
```

Expected: 28-29 JSON files in `corpus/processed/docling_v2_fr/`.

- [ ] **Step 7: Commit**

```bash
git add scripts/pipeline/extract.py scripts/pipeline/tests/test_extract.py
git commit -m "feat(pipeline): add PDF extraction with hierarchical headings"
```

---

## Task 2: Chunker — Structure-aware chunking

**Files:**
- Create: `scripts/pipeline/chunker.py`
- Create: `scripts/pipeline/tests/test_chunker.py`

- [ ] **Step 1: Write tests**

Create `scripts/pipeline/tests/test_chunker.py`:

```python
"""Tests for structure-aware chunker."""
from __future__ import annotations

import pytest

from scripts.pipeline.chunker import (
    parse_sections,
    build_hierarchy,
    chunk_document,
)


class TestParseSections:
    """Test markdown section parsing."""

    def test_splits_on_headings(self, sample_markdown_hierarchical):
        sections = parse_sections(sample_markdown_hierarchical)
        headings = [s["heading"] for s in sections]
        assert "REGLES GENERALES" in headings[0]
        assert "1. Licences" in headings[1]

    def test_captures_level(self, sample_markdown_hierarchical):
        sections = parse_sections(sample_markdown_hierarchical)
        levels = {s["heading"]: s["level"] for s in sections}
        assert levels["REGLES GENERALES"] == 1
        assert levels["1. Licences"] == 2
        assert levels["1.1. Licence A"] == 3

    def test_captures_body_text(self, sample_markdown_hierarchical):
        sections = parse_sections(sample_markdown_hierarchical)
        licences = [s for s in sections if "1. Licences" in s["heading"]][0]
        assert "licencies" in licences["body"]

    def test_empty_heading_has_no_body(self, sample_markdown_hierarchical):
        sections = parse_sections(sample_markdown_hierarchical)
        forfaits = [s for s in sections if "3. Forfaits" in s["heading"]][0]
        assert forfaits["body"].strip() == ""

    def test_strips_page_markers(self):
        md = "## 1. Test\n\nSome text.\n\nR01-1/6\n\n## 2. Next\n\nMore text.\n"
        sections = parse_sections(md)
        assert "R01-1/6" not in sections[0]["body"]


class TestBuildHierarchy:
    """Test parent-child hierarchy from heading levels."""

    def test_parent_grouping(self, sample_markdown_hierarchical):
        sections = parse_sections(sample_markdown_hierarchical)
        hierarchy = build_hierarchy(sections)
        # "1. Licences" should be parent of "1.1" and "1.2"
        licences_parent = [h for h in hierarchy if "1. Licences" in h["heading"]][0]
        child_headings = [c["heading"] for c in licences_parent["children"]]
        assert any("1.1" in h for h in child_headings)
        assert any("1.2" in h for h in child_headings)

    def test_empty_heading_becomes_parent_label(self, sample_markdown_hierarchical):
        sections = parse_sections(sample_markdown_hierarchical)
        hierarchy = build_hierarchy(sections)
        # "3. Forfaits" has no body, should still be a parent node
        forfaits = [h for h in hierarchy if "3. Forfaits" in h["heading"]]
        assert len(forfaits) == 1


class TestChunkDocument:
    """Test full chunking pipeline."""

    def test_produces_children_and_parents(self, sample_markdown_hierarchical):
        result = chunk_document(sample_markdown_hierarchical, source="test.pdf")
        assert len(result["children"]) > 0
        assert len(result["parents"]) > 0

    def test_children_have_required_fields(self, sample_markdown_hierarchical):
        result = chunk_document(sample_markdown_hierarchical, source="test.pdf")
        child = result["children"][0]
        assert "id" in child
        assert "text" in child
        assert "parent_id" in child
        assert "source" in child
        assert "tokens" in child

    def test_parent_ids_valid(self, sample_markdown_hierarchical):
        result = chunk_document(sample_markdown_hierarchical, source="test.pdf")
        parent_ids = {p["id"] for p in result["parents"]}
        for child in result["children"]:
            assert child["parent_id"] in parent_ids, \
                f"Child {child['id']} references unknown parent {child['parent_id']}"

    def test_no_empty_children(self, sample_markdown_hierarchical):
        result = chunk_document(sample_markdown_hierarchical, source="test.pdf")
        for child in result["children"]:
            assert len(child["text"].strip()) > 10, \
                f"Child {child['id']} has near-empty text"

    def test_children_under_512_tokens(self, sample_markdown_hierarchical):
        result = chunk_document(sample_markdown_hierarchical, source="test.pdf")
        for child in result["children"]:
            assert child["tokens"] <= 550, \
                f"Child {child['id']} has {child['tokens']} tokens (max 512+margin)"

    def test_flat_markdown_fallback(self, sample_markdown_flat):
        """Fallback: flat ## headings with article numbering."""
        result = chunk_document(sample_markdown_flat, source="test.pdf")
        assert len(result["children"]) > 0
        assert len(result["parents"]) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest scripts/pipeline/tests/test_chunker.py -v 2>&1 | tail -5
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement chunker.py**

Create `scripts/pipeline/chunker.py`:

```python
"""Structure-aware chunker for FFE regulation markdown.

Parses markdown with heading levels, builds parent-child hierarchy,
and produces chunks of 400-512 tokens respecting article boundaries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")
PAGE_MARKER_RE = re.compile(r"\n\s*[A-Z]\d{2}-\d+/\d+\s*\n")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
ARTICLE_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*\.?)\s")
MAX_TOKENS = 512
MERGE_THRESHOLD = 100
SPLIT_TARGET = 450
SPLIT_OVERLAP = 50


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base."""
    return len(_enc.encode(text))


def parse_sections(markdown: str) -> list[dict]:
    """Parse markdown into sections based on headings.

    Returns list of dicts with: heading, level, body, start_pos.
    """
    # Strip page markers
    markdown = PAGE_MARKER_RE.sub("\n", markdown)
    # Strip image placeholders
    markdown = markdown.replace("<!-- image -->", "")

    sections: list[dict] = []
    matches = list(HEADING_RE.finditer(markdown))

    for i, match in enumerate(matches):
        hashes = match.group(1)
        heading = match.group(2).strip()
        level = len(hashes)
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[body_start:body_end].strip()

        sections.append({
            "heading": heading,
            "level": level,
            "body": body,
        })

    return sections


def _infer_level_from_numbering(heading: str) -> int | None:
    """Infer heading level from article numbering (fallback for flat ##)."""
    m = ARTICLE_NUM_RE.match(heading)
    if not m:
        return None
    num = m.group(1).rstrip(".")
    depth = num.count(".") + 1  # "2" -> 1, "2.1" -> 2, "2.1.1" -> 3
    return depth + 1  # offset: top articles = level 2


def build_hierarchy(sections: list[dict]) -> list[dict]:
    """Build parent-child hierarchy from sections.

    Uses heading levels if available (multi-level markdown).
    Falls back to article numbering if all levels are the same.
    """
    # Check if we have real heading levels
    levels = {s["level"] for s in sections}
    all_flat = len(levels) <= 1

    if all_flat:
        # Fallback: infer levels from article numbering
        for s in sections:
            inferred = _infer_level_from_numbering(s["heading"])
            if inferred is not None:
                s["level"] = inferred
            # else keep original level (title headings stay at 1 or 2)

    # Build tree using a stack
    root: list[dict] = []
    stack: list[dict] = []  # stack of (node, level)

    for s in sections:
        node = {
            "heading": s["heading"],
            "level": s["level"],
            "body": s["body"],
            "children": [],
        }

        # Pop stack until we find a parent with lower level
        while stack and stack[-1]["level"] >= s["level"]:
            stack.pop()

        if stack:
            stack[-1]["children"].append(node)
        else:
            root.append(node)

        stack.append(node)

    return root


def _split_long_text(text: str, heading: str) -> list[str]:
    """Split text > MAX_TOKENS into ~SPLIT_TARGET token chunks with overlap."""
    tokens = _enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + SPLIT_TARGET
        chunk_tokens = tokens[start:end]
        chunk_text = _enc.decode(chunk_tokens)
        # Prepend heading for context
        if start > 0:
            chunk_text = f"{heading} (suite)\n\n{chunk_text}"
        chunks.append(chunk_text)
        start = end - SPLIT_OVERLAP
    return chunks


def _is_table_section(body: str) -> bool:
    """Check if body is primarily a table (many | characters)."""
    lines = body.strip().split("\n")
    if not lines:
        return False
    pipe_lines = sum(1 for l in lines if "|" in l)
    return pipe_lines > len(lines) * 0.5


def chunk_document(
    markdown: str,
    source: str,
) -> dict:
    """Chunk a markdown document into children and parents.

    Args:
        markdown: Markdown text with heading levels.
        source: Source PDF filename.

    Returns:
        Dict with "children" and "parents" lists.
    """
    sections = parse_sections(markdown)
    hierarchy = build_hierarchy(sections)

    children: list[dict] = []
    parents: list[dict] = []
    child_counter = 0

    def _process_node(node: dict, parent_id: str | None) -> None:
        nonlocal child_counter

        has_children = len(node["children"]) > 0
        has_body = len(node["body"].strip()) > 10

        # If this node has sub-sections, it's a parent
        if has_children:
            # Build parent text = heading + body + all children text
            parts = [f"{node['heading']}\n\n{node['body']}".strip()]
            for child_node in node["children"]:
                parts.append(f"{child_node['heading']}\n\n{child_node['body']}".strip())
            parent_text = "\n\n".join(parts)

            my_parent_id = f"{source}-p{len(parents)}"
            parents.append({
                "id": my_parent_id,
                "text": parent_text,
                "source": source,
                "section": node["heading"],
                "tokens": count_tokens(parent_text),
            })

            # If this parent also has its own body, make it a child too
            if has_body:
                _add_child(node["heading"], node["body"], source, my_parent_id)

            # Process child sections
            for child_node in node["children"]:
                _process_node(child_node, my_parent_id)

        elif has_body:
            # Leaf section with body → child
            pid = parent_id or f"{source}-p-root"
            _add_child(node["heading"], node["body"], source, pid)

        # Empty heading with no children → skip (label only)

    def _add_child(heading: str, body: str, src: str, pid: str) -> None:
        nonlocal child_counter
        full_text = f"{heading}\n\n{body}".strip()
        tokens = count_tokens(full_text)

        if tokens > MAX_TOKENS and not _is_table_section(body):
            # Split long prose
            for chunk_text in _split_long_text(full_text, heading):
                _emit_child(chunk_text, src, pid, heading)
        else:
            # Single child (even if > 512 for tables)
            _emit_child(full_text, src, pid, heading)

    def _emit_child(text: str, src: str, pid: str, section: str) -> None:
        nonlocal child_counter
        # Extract article number if present
        art_match = ARTICLE_NUM_RE.match(section)
        art_num = art_match.group(1).rstrip(".") if art_match else None

        children.append({
            "id": f"{src}-c{child_counter:04d}",
            "text": text,
            "parent_id": pid,
            "source": src,
            "article_num": art_num,
            "section": section,
            "tokens": count_tokens(text),
        })
        child_counter += 1

    # Ensure root parent exists
    parents.append({
        "id": f"{source}-p-root",
        "text": "",
        "source": source,
        "section": "root",
        "tokens": 0,
    })

    for node in hierarchy:
        _process_node(node, f"{source}-p-root")

    # Merge consecutive small children under same parent
    children = _merge_small_children(children)

    return {"children": children, "parents": parents}


def _merge_small_children(children: list[dict]) -> list[dict]:
    """Merge consecutive children under same parent if both < MERGE_THRESHOLD tokens."""
    if not children:
        return children

    merged: list[dict] = [children[0]]

    for child in children[1:]:
        prev = merged[-1]
        if (
            prev["parent_id"] == child["parent_id"]
            and prev["tokens"] < MERGE_THRESHOLD
            and child["tokens"] < MERGE_THRESHOLD
            and prev["tokens"] + child["tokens"] <= MAX_TOKENS
        ):
            # Merge
            prev["text"] = prev["text"] + "\n\n" + child["text"]
            prev["tokens"] = count_tokens(prev["text"])
            prev["section"] = prev["section"] + " + " + child["section"]
            if child.get("article_num") and prev.get("article_num"):
                prev["article_num"] = prev["article_num"] + "+" + child["article_num"]
        else:
            merged.append(child)

    return merged
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest scripts/pipeline/tests/test_chunker.py -v
```

Expected: all PASS

- [ ] **Step 5: Test on real R01 extraction**

```bash
python -c "
from scripts.pipeline.chunker import chunk_document
from pathlib import Path
import json

# Use existing docling extraction as test (flat ##)
with open('corpus/processed/docling_fr/R01_2025_26_Regles_generales.json') as f:
    doc = json.load(f)
result = chunk_document(doc['markdown'], source='R01_2025_26_Regles_generales.pdf')
print(f'Children: {len(result[\"children\"])}')
print(f'Parents: {len(result[\"parents\"])}')
for c in result['children'][:5]:
    print(f'  {c[\"id\"]}: {c[\"tokens\"]}tok - {c[\"section\"][:50]}')
"
```

Verify: children have reasonable token counts, parents group articles correctly.

- [ ] **Step 6: Commit**

```bash
git add scripts/pipeline/chunker.py scripts/pipeline/tests/test_chunker.py
git commit -m "feat(pipeline): add structure-aware chunker with hierarchy"
```

---

## Task 3: Indexer — Embeddings + SQLite DB

**Files:**
- Create: `scripts/pipeline/indexer.py`
- Create: `scripts/pipeline/tests/test_indexer.py`

- [ ] **Step 1: Write tests**

Create `scripts/pipeline/tests/test_indexer.py`:

```python
"""Tests for indexer (embeddings + SQLite)."""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline.indexer import (
    create_db,
    insert_children,
    insert_parents,
    insert_table_summaries,
    embed_texts,
    load_table_summaries,
)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def sample_children():
    return [
        {"id": "test-c0000", "text": "Article 1 content", "parent_id": "test-p0",
         "source": "test.pdf", "article_num": "1", "section": "1. Test",
         "tokens": 10},
        {"id": "test-c0001", "text": "Article 2 content", "parent_id": "test-p0",
         "source": "test.pdf", "article_num": "2", "section": "2. Test",
         "tokens": 10},
    ]


@pytest.fixture
def sample_parents():
    return [
        {"id": "test-p0", "text": "Full parent text", "source": "test.pdf",
         "section": "Root", "tokens": 20},
    ]


class TestCreateDb:
    def test_creates_tables(self, db_path):
        create_db(db_path)
        conn = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {t[0] for t in tables}
        assert "children" in names
        assert "parents" in names
        assert "table_summaries" in names
        conn.close()


class TestInsertData:
    def test_insert_children(self, db_path, sample_children):
        create_db(db_path)
        # Fake embeddings
        embeddings = {c["id"]: np.random.randn(768).astype(np.float32) for c in sample_children}
        insert_children(db_path, sample_children, embeddings)

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT id, source FROM children").fetchall()
        assert len(rows) == 2
        conn.close()

    def test_insert_parents(self, db_path, sample_parents):
        create_db(db_path)
        insert_parents(db_path, sample_parents)

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT id FROM parents").fetchall()
        assert len(rows) == 1
        conn.close()

    def test_embedding_roundtrip(self, db_path, sample_children):
        create_db(db_path)
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        embeddings = {sample_children[0]["id"]: emb, sample_children[1]["id"]: emb}
        insert_children(db_path, sample_children, embeddings)

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT embedding FROM children WHERE id=?",
                          (sample_children[0]["id"],)).fetchone()
        recovered = np.frombuffer(row[0], dtype=np.float32)
        np.testing.assert_array_equal(recovered, emb)
        conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest scripts/pipeline/tests/test_indexer.py -v 2>&1 | tail -5
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement indexer.py**

Create `scripts/pipeline/indexer.py`:

```python
"""Index children and table summaries into SQLite with embeddings.

Embeds text with EmbeddingGemma-300M (768D) via sentence-transformers,
stores in SQLite for brute-force cosine retrieval.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_ID = "google/embeddinggemma-300m"
EMBEDDING_DIM = 768

SCHEMA = """
CREATE TABLE IF NOT EXISTS children (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    parent_id TEXT NOT NULL,
    source TEXT NOT NULL,
    pages TEXT,
    article_num TEXT,
    section TEXT,
    tokens INTEGER
);

CREATE TABLE IF NOT EXISTS parents (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    section TEXT,
    tokens INTEGER
);

CREATE TABLE IF NOT EXISTS table_summaries (
    id TEXT PRIMARY KEY,
    summary_text TEXT NOT NULL,
    raw_table_text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    source TEXT NOT NULL,
    page INTEGER,
    tokens INTEGER
);
"""


def create_db(db_path: Path) -> None:
    """Create SQLite database with schema."""
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.close()


def _to_blob(arr: np.ndarray) -> bytes:
    """Convert numpy float32 array to bytes."""
    return arr.astype(np.float32).tobytes()


def embed_texts(texts: list[str], model_id: str = EMBEDDING_MODEL_ID) -> np.ndarray:
    """Embed texts using sentence-transformers.

    Args:
        texts: List of strings to embed.
        model_id: HuggingFace model ID.

    Returns:
        np.ndarray of shape (len(texts), EMBEDDING_DIM).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def insert_parents(db_path: Path, parents: list[dict]) -> None:
    """Insert parent records."""
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT OR REPLACE INTO parents (id, text, source, section, tokens) VALUES (?, ?, ?, ?, ?)",
        [(p["id"], p["text"], p["source"], p.get("section"), p.get("tokens")) for p in parents],
    )
    conn.commit()
    conn.close()


def insert_children(
    db_path: Path,
    children: list[dict],
    embeddings: dict[str, np.ndarray],
) -> None:
    """Insert children with embeddings."""
    conn = sqlite3.connect(db_path)
    rows = []
    for c in children:
        emb = embeddings[c["id"]]
        rows.append((
            c["id"], c["text"], _to_blob(emb), c["parent_id"],
            c["source"], json.dumps(c.get("pages")),
            c.get("article_num"), c.get("section"), c.get("tokens"),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO children (id, text, embedding, parent_id, source, pages, article_num, section, tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def insert_table_summaries(
    db_path: Path,
    summaries: list[dict],
    embeddings: dict[str, np.ndarray],
) -> None:
    """Insert table summaries with embeddings."""
    conn = sqlite3.connect(db_path)
    rows = []
    for s in summaries:
        emb = embeddings[s["id"]]
        rows.append((
            s["id"], s["summary_text"], s["raw_table_text"],
            _to_blob(emb), s["source"], s.get("page"), s.get("tokens"),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO table_summaries (id, summary_text, raw_table_text, embedding, source, page, tokens) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def load_table_summaries(path: Path) -> list[dict]:
    """Load table summaries from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summaries = []
    for key, summary_text in data["summaries"].items():
        # key format: "filename-tableN"
        source = key.rsplit("-table", 1)[0] + ".pdf"
        summaries.append({
            "id": key,
            "summary_text": summary_text,
            "raw_table_text": summary_text,  # TODO: link to raw table if available
            "source": source,
        })
    return summaries


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    chunks_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("corpus/processed/chunks_v2_fr")
    db_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("corpus/processed/corpus_v2_fr.db")
    table_summaries_path = Path("corpus/processed/table_summaries_claude.json")

    # Load chunks
    children_all, parents_all = [], []
    for f in sorted(chunks_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
            children_all.extend(data["children"])
            parents_all.extend(data["parents"])

    logger.info("Total children: %d, parents: %d", len(children_all), len(parents_all))

    # Embed children
    child_texts = [c["text"] for c in children_all]
    child_embs = embed_texts(child_texts)
    child_emb_dict = {c["id"]: child_embs[i] for i, c in enumerate(children_all)}

    # Load and embed table summaries
    summaries = load_table_summaries(table_summaries_path)
    summary_texts = [s["summary_text"] for s in summaries]
    summary_embs = embed_texts(summary_texts)
    summary_emb_dict = {s["id"]: summary_embs[i] for i, s in enumerate(summaries)}

    # Build DB
    create_db(db_path)
    insert_parents(db_path, parents_all)
    insert_children(db_path, children_all, child_emb_dict)
    insert_table_summaries(db_path, summaries, summary_emb_dict)

    logger.info("DB written to %s", db_path)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest scripts/pipeline/tests/test_indexer.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/indexer.py scripts/pipeline/tests/test_indexer.py
git commit -m "feat(pipeline): add indexer with EmbeddingGemma + SQLite"
```

---

## Task 4: Search — Cosine brute-force → parents

**Files:**
- Create: `scripts/pipeline/search.py`
- Create: `scripts/pipeline/tests/test_search.py`

- [ ] **Step 1: Write tests**

Create `scripts/pipeline/tests/test_search.py`:

```python
"""Tests for search (cosine retrieval + parent lookup)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from scripts.pipeline.indexer import create_db, insert_children, insert_parents, insert_table_summaries
from scripts.pipeline.search import search, load_index


@pytest.fixture
def populated_db(tmp_path):
    """Create a small DB with known embeddings for testing."""
    db_path = tmp_path / "test.db"
    create_db(db_path)

    parents = [
        {"id": "p0", "text": "Parent about licences", "source": "test.pdf",
         "section": "Licences", "tokens": 20},
        {"id": "p1", "text": "Parent about forfaits", "source": "test.pdf",
         "section": "Forfaits", "tokens": 20},
    ]
    insert_parents(db_path, parents)

    # Create children with distinct embeddings
    emb_licence = np.zeros(768, dtype=np.float32)
    emb_licence[0] = 1.0  # points in dimension 0

    emb_forfait = np.zeros(768, dtype=np.float32)
    emb_forfait[1] = 1.0  # points in dimension 1

    children = [
        {"id": "c0", "text": "Licence A", "parent_id": "p0",
         "source": "test.pdf", "article_num": "1", "section": "1. Licences",
         "tokens": 10},
        {"id": "c1", "text": "Forfait sportif", "parent_id": "p1",
         "source": "test.pdf", "article_num": "3", "section": "3. Forfaits",
         "tokens": 10},
    ]
    embeddings = {"c0": emb_licence, "c1": emb_forfait}
    insert_children(db_path, children, embeddings)

    return db_path


class TestSearch:
    def test_returns_results(self, populated_db):
        index = load_index(populated_db)
        # Query close to emb_licence (dimension 0)
        query_emb = np.zeros(768, dtype=np.float32)
        query_emb[0] = 1.0
        results = search(index, query_emb, k=2)
        assert len(results) > 0

    def test_nearest_is_correct(self, populated_db):
        index = load_index(populated_db)
        query_emb = np.zeros(768, dtype=np.float32)
        query_emb[0] = 1.0
        results = search(index, query_emb, k=1)
        assert results[0]["child_id"] == "c0"

    def test_returns_parent_text(self, populated_db):
        index = load_index(populated_db)
        query_emb = np.zeros(768, dtype=np.float32)
        query_emb[0] = 1.0
        results = search(index, query_emb, k=1)
        assert "Parent about licences" in results[0]["parent_text"]

    def test_deduplicates_parents(self, populated_db):
        index = load_index(populated_db)
        # Query that matches both children (they have different parents)
        query_emb = np.ones(768, dtype=np.float32)
        results = search(index, query_emb, k=2)
        parent_ids = [r["parent_id"] for r in results]
        assert len(parent_ids) == len(set(parent_ids))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest scripts/pipeline/tests/test_search.py -v 2>&1 | tail -5
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement search.py**

Create `scripts/pipeline/search.py`:

```python
"""Cosine brute-force search over children + table summaries.

Searches children and table summary embeddings, returns parent text
for LLM context (small-to-big retrieval).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchIndex:
    """In-memory search index loaded from SQLite."""

    child_ids: list[str]
    child_embeddings: np.ndarray  # (N, 768)
    child_parent_ids: list[str]
    child_metadata: list[dict]

    summary_ids: list[str]
    summary_embeddings: np.ndarray  # (M, 768)
    summary_metadata: list[dict]

    parents: dict[str, dict]  # parent_id -> {text, source, section}


def load_index(db_path: Path) -> SearchIndex:
    """Load all embeddings and parents from SQLite into memory."""
    conn = sqlite3.connect(db_path)

    # Load children
    rows = conn.execute(
        "SELECT id, embedding, parent_id, source, article_num, section, tokens FROM children"
    ).fetchall()

    child_ids = [r[0] for r in rows]
    child_embeddings = np.array(
        [np.frombuffer(r[1], dtype=np.float32) for r in rows]
    ) if rows else np.empty((0, 768), dtype=np.float32)
    child_parent_ids = [r[2] for r in rows]
    child_metadata = [
        {"source": r[3], "article_num": r[4], "section": r[5], "tokens": r[6]}
        for r in rows
    ]

    # Load table summaries
    srows = conn.execute(
        "SELECT id, embedding, summary_text, raw_table_text, source, page FROM table_summaries"
    ).fetchall()

    summary_ids = [r[0] for r in srows]
    summary_embeddings = np.array(
        [np.frombuffer(r[1], dtype=np.float32) for r in srows]
    ) if srows else np.empty((0, 768), dtype=np.float32)
    summary_metadata = [
        {"summary_text": r[2], "raw_table_text": r[3], "source": r[4], "page": r[5]}
        for r in srows
    ]

    # Load parents
    prows = conn.execute("SELECT id, text, source, section FROM parents").fetchall()
    parents_dict = {r[0]: {"text": r[1], "source": r[2], "section": r[3]} for r in prows}

    conn.close()

    return SearchIndex(
        child_ids=child_ids,
        child_embeddings=child_embeddings,
        child_parent_ids=child_parent_ids,
        child_metadata=child_metadata,
        summary_ids=summary_ids,
        summary_embeddings=summary_embeddings,
        summary_metadata=summary_metadata,
        parents=parents_dict,
    )


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between query vector and matrix of vectors."""
    if matrix.shape[0] == 0:
        return np.array([])
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norm @ query_norm


def search(
    index: SearchIndex,
    query_embedding: np.ndarray,
    k: int = 10,
) -> list[dict]:
    """Search for top-k children + summaries, return parents.

    Args:
        index: Pre-loaded search index.
        query_embedding: Query vector (768D).
        k: Number of top results.

    Returns:
        List of result dicts with: child_id, score, parent_id, parent_text,
        source, section, result_type ("child" or "table_summary").
    """
    results = []

    # Score children
    child_scores = _cosine_similarity(query_embedding, index.child_embeddings)
    for i, score in enumerate(child_scores):
        results.append({
            "child_id": index.child_ids[i],
            "score": float(score),
            "parent_id": index.child_parent_ids[i],
            "result_type": "child",
            **index.child_metadata[i],
        })

    # Score table summaries
    summary_scores = _cosine_similarity(query_embedding, index.summary_embeddings)
    for i, score in enumerate(summary_scores):
        results.append({
            "child_id": index.summary_ids[i],
            "score": float(score),
            "parent_id": None,
            "result_type": "table_summary",
            **index.summary_metadata[i],
        })

    # Sort by score descending, take top-k
    results.sort(key=lambda r: r["score"], reverse=True)
    results = results[:k]

    # Attach parent text (deduplicated)
    seen_parents = set()
    for r in results:
        if r["result_type"] == "child" and r["parent_id"]:
            parent = index.parents.get(r["parent_id"], {})
            r["parent_text"] = parent.get("text", "")
            seen_parents.add(r["parent_id"])
        elif r["result_type"] == "table_summary":
            r["parent_text"] = r.get("raw_table_text", "")

    return results
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest scripts/pipeline/tests/test_search.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/search.py scripts/pipeline/tests/test_search.py
git commit -m "feat(pipeline): add cosine brute-force search with parent lookup"
```

---

## Task 5: Integration — Build corpus_v2_fr.db

- [ ] **Step 1: Run full extraction (if not done in Task 1)**

```bash
python scripts/pipeline/extract.py corpus/fr corpus/processed/docling_v2_fr
```

- [ ] **Step 2: Chunk all extracted documents**

```bash
python -c "
from scripts.pipeline.chunker import chunk_document
from pathlib import Path
import json

input_dir = Path('corpus/processed/docling_v2_fr')
output_dir = Path('corpus/processed/chunks_v2_fr')
output_dir.mkdir(exist_ok=True)

total_children, total_parents = 0, 0
for f in sorted(input_dir.glob('*.json')):
    with open(f) as fh:
        doc = json.load(fh)
    result = chunk_document(doc['markdown'], source=f.stem + '.pdf')
    with open(output_dir / f.name, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    total_children += len(result['children'])
    total_parents += len(result['parents'])
    print(f'{f.name}: {len(result[\"children\"])} children, {len(result[\"parents\"])} parents')

print(f'\nTotal: {total_children} children, {total_parents} parents')
"
```

Expected: ~600-800 children total, ~200-400 parents.

- [ ] **Step 3: Build DB with embeddings**

```bash
python scripts/pipeline/indexer.py corpus/processed/chunks_v2_fr corpus/processed/corpus_v2_fr.db
```

This will take a few minutes (embedding ~700 texts with EmbeddingGemma-300M).

- [ ] **Step 4: Verify DB**

```bash
python -c "
import sqlite3
conn = sqlite3.connect('corpus/processed/corpus_v2_fr.db')
for table in ['children', 'parents', 'table_summaries']:
    count = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
    print(f'{table}: {count} rows')
conn.close()
"
```

Expected: children ~600-800, parents ~200-400, table_summaries 111.

- [ ] **Step 5: Quick search test**

```bash
python -c "
from scripts.pipeline.search import load_index, search
from scripts.pipeline.indexer import embed_texts
import numpy as np

index = load_index('corpus/processed/corpus_v2_fr.db')
query = 'Quelle est la composition du jury d appel ?'
query_emb = embed_texts([query])[0]
results = search(index, query_emb, k=5)
for r in results:
    print(f'{r[\"score\"]:.3f} | {r[\"child_id\"]} | {r[\"section\"][:60]}')
    if r.get('parent_text'):
        print(f'  Parent: {r[\"parent_text\"][:100]}...')
"
```

Verify: results make sense, parent text provides context.

- [ ] **Step 6: Commit**

```bash
git add scripts/pipeline/
git commit -m "feat(pipeline): complete pipeline v2 — extract, chunk, index, search"
```

---

## Task 6: Update config and docs

- [ ] **Step 1: Update .coveragerc**

Add `scripts/pipeline/` back to source:

```ini
[run]
source =
    scripts/iso/
    scripts/pipeline/
omit =
    scripts/archive/**
    */conftest.py
    */tests/*
```

- [ ] **Step 2: Update CLAUDE.md**

Update the "Ce qui est casse" section to reflect pipeline rebuild. Update commands section.

- [ ] **Step 3: Run all tests**

```bash
python -m pytest scripts/iso/ scripts/pipeline/tests/ -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add .coveragerc CLAUDE.md
git commit -m "docs: update config and CLAUDE.md for pipeline v2"
```
