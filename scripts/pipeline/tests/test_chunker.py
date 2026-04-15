"""Tests for LangChain-based chunker."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from scripts.pipeline.chunker import chunk_document, header_split, recursive_split
from scripts.pipeline.chunker_utils import (
    PARENT_MAX_TOKENS,
    build_cch_title,
    build_parents,
    extract_tables,
    interpolate_pages,
    link_tables,
)

# === Stage 1: Table extraction ===


class TestExtractTables:
    """Test table extraction from raw markdown."""

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
        md = "| single line |\n| another |"
        clean, tables = extract_tables(md)
        assert len(tables) == 0

    def test_multiple_tables(self) -> None:
        md = (
            "Text\n| A | B |\n|---|---|\n| 1 | 2 |\n"
            "Middle\n| X | Y |\n|---|---|\n| 3 | 4 |\nEnd"
        )
        clean, tables = extract_tables(md)
        assert len(tables) == 2
        assert "<!-- TABLE_0 -->" in clean
        assert "<!-- TABLE_1 -->" in clean

    def test_strips_page_markers(self) -> None:
        md = "Text\nR01-1/6\nMore"
        clean, _ = extract_tables(md)
        assert "R01-1/6" not in clean

    def test_real_corpus_la_table_count(self) -> None:
        p = Path("corpus/processed/docling_v2_fr/LA-octobre2025.json")
        if not p.exists():
            pytest.skip("LA docling file not available")
        import json

        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        _, tables = extract_tables(doc["markdown"])
        assert len(tables) >= 90, f"Expected ~98 tables, got {len(tables)}"


# === Stage 2: Header split ===


class TestHeaderSplit:
    """Test MarkdownHeaderTextSplitter wrapper."""

    def test_splits_by_h2(self) -> None:
        md = "## Section A\nText A\n## Section B\nText B"
        docs = header_split(md)
        assert len(docs) >= 2

    def test_preserves_heading_in_content(self) -> None:
        md = "## My Section\nBody text"
        docs = header_split(md)
        assert any("My Section" in d.page_content for d in docs)

    def test_metadata_hierarchy(self) -> None:
        md = "# Title\n## Sub\nBody"
        docs = header_split(md)
        sub = [d for d in docs if "Body" in d.page_content]
        assert len(sub) >= 1
        assert sub[0].metadata.get("h1") == "Title"

    def test_placeholder_preserved(self) -> None:
        md = "## Section\n<!-- TABLE_0 -->\nMore text"
        docs = header_split(md)
        all_text = " ".join(d.page_content for d in docs)
        assert "<!-- TABLE_0 -->" in all_text


# === Stage 3: Recursive split ===


class TestRecursiveSplit:
    """Test RecursiveCharacterTextSplitter wrapper."""

    def test_splits_oversized(self) -> None:
        long_text = "Mot " * 600
        docs = [Document(page_content=long_text, metadata={"h1": "T"})]
        children = recursive_split(docs)
        assert len(children) >= 2

    def test_preserves_metadata(self) -> None:
        docs = [Document(page_content="Short", metadata={"h1": "A", "h2": "B"})]
        children = recursive_split(docs)
        assert children[0].metadata.get("h1") == "A"

    def test_small_not_split(self) -> None:
        docs = [Document(page_content="Small", metadata={})]
        assert len(recursive_split(docs)) == 1


# === Stage 4: Parent assembly ===


class TestBuildParents:
    """Test parent assembly from children."""

    def test_groups_by_h1_h2(self) -> None:
        children = [
            Document(page_content="C1", metadata={"h1": "A", "h2": "B"}),
            Document(page_content="C2", metadata={"h1": "A", "h2": "B"}),
            Document(page_content="C3", metadata={"h1": "A", "h2": "C"}),
        ]
        parents, _ = build_parents(children, "test.pdf")
        assert len(parents) == 2

    def test_parent_capped_at_2048(self) -> None:
        big = "word " * 800
        children = [
            Document(page_content=big, metadata={"h1": "A", "h2": "B"}),
            Document(page_content=big, metadata={"h1": "A", "h2": "B"}),
            Document(page_content=big, metadata={"h1": "A", "h2": "B"}),
        ]
        parents, _ = build_parents(children, "test.pdf")
        for p in parents:
            assert p["tokens"] <= PARENT_MAX_TOKENS + 50

    def test_no_empty_parents(self) -> None:
        children = [Document(page_content="Text", metadata={"h1": "A"})]
        parents, _ = build_parents(children, "test.pdf")
        for p in parents:
            assert p["text"].strip()

    def test_has_required_fields(self) -> None:
        children = [Document(page_content="T", metadata={"h1": "X"})]
        parents, _ = build_parents(children, "src.pdf")
        for key in ("id", "text", "source", "section", "tokens"):
            assert key in parents[0]


# === Stage 6: CCH title ===


class TestBuildCchTitle:
    """Test CCH title from heading hierarchy."""

    def test_full(self) -> None:
        assert build_cch_title({"h1": "A", "h2": "B", "h3": "C"}) == "A > B > C"

    def test_partial(self) -> None:
        assert build_cch_title({"h2": "B"}) == "B"

    def test_empty(self) -> None:
        assert build_cch_title({}) == ""


# === Stage 5: Page interpolation ===


class TestInterpolatePages:
    """Test page assignment from heading_pages."""

    def test_assigns_page(self) -> None:
        children = [{"section": "Forfaits", "page": None}]
        result = interpolate_pages(children, {"Forfaits": 5})
        assert result[0]["page"] == 5

    def test_fallback_page_1_when_no_mapping(self) -> None:
        """Empty heading_pages → fallback page 1."""
        children = [{"section": "Unknown", "page": None}]
        assert interpolate_pages(children, {})[0]["page"] == 1


# === Stage 7: Table linkage ===


class TestLinkTables:
    """Test table linkage to parent sections."""

    def test_links_via_placeholder(self) -> None:
        tables = [{"raw_text": "| A | B |"}]
        header_docs = [
            Document(
                page_content="Text\n<!-- TABLE_0 -->",
                metadata={"h1": "T", "h2": "S"},
            ),
        ]
        parents = [{"id": "p0", "section": "T > S"}]
        linked = link_tables(tables, header_docs, parents)
        assert linked[0]["section"] == "T > S"
        assert linked[0]["parent_id"] == "p0"


# === Orchestrator ===


class TestChunkDocument:
    """Test full 7-stage pipeline."""

    def test_returns_children_parents_tables(self) -> None:
        md = "## Section\nText.\n| A | B |\n|---|---|\n| 1 | 2 |"
        result = chunk_document(md, "test.pdf")
        assert "children" in result
        assert "parents" in result
        assert "tables" in result

    def test_no_placeholder_in_children(self) -> None:
        md = "## S\nText\n| A | B |\n|---|---|\n| 1 | 2 |\nMore"
        result = chunk_document(md, "test.pdf")
        for child in result["children"]:
            assert "<!-- TABLE_" not in child["text"]

    def test_children_have_fields(self) -> None:
        md = "## Section\nBody text."
        result = chunk_document(md, "test.pdf")
        for child in result["children"]:
            for key in ("id", "text", "parent_id", "source", "section", "tokens"):
                assert key in child

    def test_no_empty_parents(self) -> None:
        md = "## A\nText A\n## B\nText B"
        result = chunk_document(md, "test.pdf")
        for p in result["parents"]:
            assert p["text"].strip()

    def test_parent_ids_valid(self) -> None:
        md = "## A\nText"
        result = chunk_document(md, "test.pdf")
        pids = {p["id"] for p in result["parents"]}
        for child in result["children"]:
            assert child["parent_id"] in pids or child["parent_id"] == ""

    def test_real_corpus_no_invisible_parents(self) -> None:
        p = Path("corpus/processed/docling_v2_fr/R01_2025_26_Regles_generales.json")
        if not p.exists():
            pytest.skip("R01 not available")
        import json

        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        result = chunk_document(doc["markdown"], "R01.pdf")
        pids_with_kids = {c["parent_id"] for c in result["children"]}
        for parent in result["parents"]:
            if parent["text"].strip():
                assert parent["id"] in pids_with_kids, (
                    f"Invisible parent: {parent['id']}"
                )

    def test_real_corpus_parents_under_2048(self) -> None:
        p = Path("corpus/processed/docling_v2_fr/LA-octobre2025.json")
        if not p.exists():
            pytest.skip("LA not available")
        import json

        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        result = chunk_document(doc["markdown"], "LA.pdf")
        for parent in result["parents"]:
            assert parent["tokens"] <= PARENT_MAX_TOKENS + 50, (
                f"{parent['id']}: {parent['tokens']} tokens"
            )
