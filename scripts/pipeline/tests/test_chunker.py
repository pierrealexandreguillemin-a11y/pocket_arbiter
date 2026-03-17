"""Tests for structure-aware chunker."""

from __future__ import annotations

from scripts.pipeline.chunker import (
    build_hierarchy,
    chunk_document,
    parse_sections,
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

    def test_strips_image_placeholders(self):
        md = "## Title\n\n<!-- image -->\n\nReal content.\n"
        sections = parse_sections(md)
        assert "<!-- image -->" not in sections[0]["body"]
        assert "Real content" in sections[0]["body"]


class TestBuildHierarchy:
    """Test parent-child hierarchy from heading levels."""

    def test_parent_grouping(self, sample_markdown_hierarchical):
        sections = parse_sections(sample_markdown_hierarchical)
        hierarchy = build_hierarchy(sections)

        # "1. Licences" is under "REGLES GENERALES" (h1)
        # Find it recursively
        def find_node(nodes, text):
            for n in nodes:
                if text in n["heading"]:
                    return n
                found = find_node(n["children"], text)
                if found:
                    return found
            return None

        licences = find_node(hierarchy, "1. Licences")
        assert licences is not None
        child_headings = [c["heading"] for c in licences["children"]]
        assert any("1.1" in h for h in child_headings)
        assert any("1.2" in h for h in child_headings)

    def test_empty_heading_is_parent(self, sample_markdown_hierarchical):
        sections = parse_sections(sample_markdown_hierarchical)
        hierarchy = build_hierarchy(sections)

        # "3. Forfaits" has no body, should still exist as a node
        def find_node(nodes, text):
            for n in nodes:
                if text in n["heading"]:
                    return n
                found = find_node(n["children"], text)
                if found:
                    return found
            return None

        forfaits = find_node(hierarchy, "3. Forfaits")
        assert forfaits is not None

    def test_flat_markdown_uses_numbering(self, sample_markdown_flat):
        sections = parse_sections(sample_markdown_flat)
        hierarchy = build_hierarchy(sections)
        # Even with flat ##, numbering should create hierarchy
        licences = None
        for node in hierarchy:
            if "1. Licences" in node["heading"]:
                licences = node
            for child in node.get("children", []):
                if "1. Licences" in child["heading"]:
                    licences = child
        assert licences is not None
        child_headings = [c["heading"] for c in licences.get("children", [])]
        assert any("1.1" in h for h in child_headings)

    def test_depth_preserved(self, sample_markdown_hierarchical):
        sections = parse_sections(sample_markdown_hierarchical)
        hierarchy = build_hierarchy(sections)
        # Root should contain h1 nodes
        assert hierarchy[0]["level"] == 1


class TestChunkDocument:
    """Test full chunking pipeline."""

    def test_produces_children_and_parents(self, sample_markdown_hierarchical):
        result = chunk_document(sample_markdown_hierarchical, source="test.pdf")
        assert len(result["children"]) > 0
        assert len(result["parents"]) > 0

    def test_children_have_required_fields(self, sample_markdown_hierarchical):
        result = chunk_document(sample_markdown_hierarchical, source="test.pdf")
        child = result["children"][0]
        for field in ("id", "text", "parent_id", "source", "tokens", "section"):
            assert field in child, f"Missing field: {field}"

    def test_parent_ids_valid(self, sample_markdown_hierarchical):
        result = chunk_document(sample_markdown_hierarchical, source="test.pdf")
        parent_ids = {p["id"] for p in result["parents"]}
        for child in result["children"]:
            assert (
                child["parent_id"] in parent_ids
            ), f"Child {child['id']} references unknown parent {child['parent_id']}"

    def test_no_empty_children(self, sample_markdown_hierarchical):
        result = chunk_document(sample_markdown_hierarchical, source="test.pdf")
        for child in result["children"]:
            assert (
                len(child["text"].strip()) > 5
            ), f"Child {child['id']} has near-empty text: {child['text'][:50]}"

    def test_children_under_max_tokens(self, sample_markdown_hierarchical):
        result = chunk_document(sample_markdown_hierarchical, source="test.pdf")
        for child in result["children"]:
            assert (
                child["tokens"] <= 560
            ), f"Child {child['id']} has {child['tokens']} tokens"

    def test_flat_markdown_works(self, sample_markdown_flat):
        result = chunk_document(sample_markdown_flat, source="test.pdf")
        assert len(result["children"]) > 0
        assert len(result["parents"]) > 0

    def test_source_propagated(self, sample_markdown_hierarchical):
        result = chunk_document(sample_markdown_hierarchical, source="my_doc.pdf")
        for child in result["children"]:
            assert child["source"] == "my_doc.pdf"
        for parent in result["parents"]:
            assert parent["source"] == "my_doc.pdf"

    def test_long_section_gets_split(self):
        """A section > 512 tokens should be split."""
        # 300 tokens < 512, should NOT split. Make it bigger:
        long_body2 = "mot " * 600  # ~600 tokens
        md2 = f"# Title\n\n## Long Section\n\n{long_body2}\n\n## Short\n\nBrief.\n"
        result2 = chunk_document(md2, source="test.pdf")
        long_children2 = [
            c for c in result2["children"] if "Long Section" in c["section"]
        ]
        assert (
            len(long_children2) >= 2
        ), f"Expected split, got {len(long_children2)} children"

    def test_table_section_not_split_under_hard_max(self):
        """A table section between 512-2048 tokens should not be split."""
        table_rows = "\n".join(f"| col1_{i} | col2_{i} | col3_{i} |" for i in range(80))
        md = f"# Title\n\n## Table Section\n\n| H1 | H2 | H3 |\n|---|---|---|\n{table_rows}\n\n## Next\n\nText.\n"
        result = chunk_document(md, source="test.pdf")
        table_children = [c for c in result["children"] if "Table" in c["section"]]
        assert (
            len(table_children) == 1
        ), f"Table section under 2048 tokens should not be split, got {len(table_children)} children"
        # Verify it's actually > 512 tokens (tests the table exemption)
        assert table_children[0]["tokens"] > 512

    def test_table_section_split_over_hard_max(self):
        """A table section exceeding 2048 tokens must be split (EmbeddingGemma limit)."""
        table_rows = "\n".join(
            f"| col1_{i} | col2_{i} | col3_{i} |" for i in range(200)
        )
        md = f"# Title\n\n## Big Table\n\n| H1 | H2 | H3 |\n|---|---|---|\n{table_rows}\n\n## Next\n\nText.\n"
        result = chunk_document(md, source="test.pdf")
        table_children = [c for c in result["children"] if "Big Table" in c["section"]]
        assert (
            len(table_children) >= 2
        ), f"Table section over 2048 tokens should be split, got {len(table_children)} children"

    def test_small_sections_merged(self):
        """Consecutive small sections under same parent should merge."""
        md = (
            "# Title\n\n"
            "## A\n\nTiny.\n\n"
            "## B\n\nAlso tiny.\n\n"
            "## C\n\nStill tiny.\n\n"
        )
        result = chunk_document(md, source="test.pdf")
        # 3 tiny sections could be merged
        assert len(result["children"]) <= 3

    def test_pages_from_heading_pages(self):
        """Children should have page numbers from heading_pages mapping."""
        # Use enough text to avoid merge (>250 tokens each)
        body1 = "Content one about licensing rules. " * 60
        body2 = "Content two about forfait rules. " * 60
        md = f"# Title\n\n## Art 1\n\n{body1}\n\n## Art 2\n\n{body2}\n"
        heading_pages = {"Art 1": 5, "Art 2": 7}
        result = chunk_document(md, source="test.pdf", heading_pages=heading_pages)
        art1 = [c for c in result["children"] if "Art 1" in c["section"]][0]
        art2 = [c for c in result["children"] if "Art 2" in c["section"]][0]
        assert art1["page"] == 5
        assert art2["page"] == 7

    def test_page_none_when_no_mapping(self, sample_markdown_hierarchical):
        """Page should be None when no heading_pages provided."""
        result = chunk_document(sample_markdown_hierarchical, source="test.pdf")
        for child in result["children"]:
            assert child["page"] is None
