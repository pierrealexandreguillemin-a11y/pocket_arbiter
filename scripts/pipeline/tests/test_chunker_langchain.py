"""
Tests for chunker_langchain.py (Mode B: LangChain + Gemma).

ISO Reference:
    - ISO/IEC 29119 - Software Testing
    - ISO/IEC 42001 A.6.2.2 - AI traceability (100% page coverage)
"""

import pytest


class TestBuildPageMapFromDocling:
    """Tests for build_page_map_from_docling function."""

    def test_empty_docling_doc_returns_empty_tuple(self):
        """Empty docling_document returns empty tuple."""
        from scripts.pipeline.chunker_langchain import build_page_map_from_docling

        ordered_texts, text_to_page = build_page_map_from_docling({})
        assert ordered_texts == []
        assert text_to_page == {}

    def test_builds_mapping_from_texts(self):
        """Builds text -> page mapping from texts with provenance."""
        from scripts.pipeline.chunker_langchain import build_page_map_from_docling

        docling_doc = {
            "texts": [
                {
                    "text": "This is a sample text that is long enough for matching algorithm",
                    "prov": [{"page_no": 5}],
                }
            ]
        }

        ordered_texts, text_to_page = build_page_map_from_docling(docling_doc)
        assert len(ordered_texts) > 0
        assert ordered_texts[0]["page_no"] == 5
        assert len(text_to_page) > 0
        assert 5 in text_to_page.values()

    def test_ordered_texts_preserves_order(self):
        """ordered_texts maintains document order."""
        from scripts.pipeline.chunker_langchain import build_page_map_from_docling

        docling_doc = {
            "texts": [
                {"text": "First text paragraph long enough", "prov": [{"page_no": 1}]},
                {"text": "Second text paragraph long enough", "prov": [{"page_no": 2}]},
                {"text": "Third text paragraph long enough", "prov": [{"page_no": 3}]},
            ]
        }

        ordered_texts, _ = build_page_map_from_docling(docling_doc)
        assert len(ordered_texts) == 3
        assert ordered_texts[0]["page_no"] == 1
        assert ordered_texts[1]["page_no"] == 2
        assert ordered_texts[2]["page_no"] == 3


class TestNormalizeForMatch:
    """Tests for normalize_for_match function."""

    def test_removes_markdown_headers(self):
        """Removes markdown header symbols."""
        from scripts.pipeline.chunker_langchain import normalize_for_match

        result = normalize_for_match("## Header Title")
        assert "#" not in result
        assert "header title" in result

    def test_collapses_whitespace(self):
        """Collapses multiple whitespace to single space."""
        from scripts.pipeline.chunker_langchain import normalize_for_match

        result = normalize_for_match("word1   word2\n\nword3")
        assert "  " not in result
        assert "\n" not in result

    def test_lowercase(self):
        """Converts text to lowercase."""
        from scripts.pipeline.chunker_langchain import normalize_for_match

        result = normalize_for_match("UPPERCASE Text")
        assert result == "uppercase text"


class TestFindPageFromMap:
    """Tests for find_page_from_map function."""

    def test_empty_map_with_prev_page_returns_prev(self):
        """Empty mapping with prev_page returns prev_page (context propagation)."""
        from scripts.pipeline.chunker_langchain import find_page_from_map

        result = find_page_from_map("some text", {}, prev_page=5)
        assert result == 5

    def test_empty_map_without_prev_page_returns_one(self):
        """Empty mapping without prev_page returns 1 (ISO 42001 fallback)."""
        from scripts.pipeline.chunker_langchain import find_page_from_map

        result = find_page_from_map("some text", {})
        assert result == 1

    def test_short_text_returns_prev_page(self):
        """Short text returns prev_page via context propagation."""
        from scripts.pipeline.chunker_langchain import find_page_from_map

        result = find_page_from_map("short", {"short": 5}, prev_page=3)
        assert result == 3

    def test_finds_matching_page(self):
        """Finds page for matching text."""
        from scripts.pipeline.chunker_langchain import find_page_from_map

        text_to_page = {
            "this is a long enough text for the matching algorithm to work": 3
        }
        result = find_page_from_map(
            "This is a long enough text for the matching algorithm to work properly",
            text_to_page,
        )
        assert result == 3

    def test_always_returns_positive_page(self):
        """Always returns page >= 1 (ISO 42001 A.6.2.2)."""
        from scripts.pipeline.chunker_langchain import find_page_from_map

        # Even with no match and no prev_page, should return 1
        result = find_page_from_map(
            "This is some text that might not match anything specific",
            {},
        )
        assert result >= 1


class TestChunkMarkdownLangchain:
    """Tests for chunk_markdown_langchain function (Parent-Child architecture)."""

    def test_empty_markdown_returns_empty_tuples(self):
        """Empty markdown returns empty parent and child lists."""
        from scripts.pipeline.chunker_langchain import chunk_markdown_langchain

        parents, children = chunk_markdown_langchain("", "test.pdf", "fr", [], {})
        assert parents == []
        assert children == []

    def test_returns_tuple_of_lists(self):
        """Returns tuple (parents, children)."""
        from scripts.pipeline.chunker_langchain import chunk_markdown_langchain

        markdown = """## Section 1

This is the content of section 1. It has enough text to create a meaningful chunk.
The text continues with more information about the topic being discussed.

## Section 2

This is the content of section 2. Another paragraph with sufficient content.
More text to ensure we have enough tokens for chunking.
"""
        result = chunk_markdown_langchain(markdown, "test.pdf", "fr", [], {})
        assert isinstance(result, tuple)
        assert len(result) == 2
        parents, children = result
        assert isinstance(parents, list)
        assert isinstance(children, list)

    def test_parent_chunk_has_required_fields(self):
        """Parent chunk has required fields."""
        from scripts.pipeline.chunker_langchain import chunk_markdown_langchain

        markdown = """## Test Section

This is a test section with enough content to create a chunk.
The content needs to be long enough to exceed the minimum token threshold.
Here is some more text to make sure we have sufficient content.
Additional sentences help ensure the chunk is large enough.
"""
        parents, children = chunk_markdown_langchain(markdown, "test.pdf", "fr", [], {})

        if parents:
            parent = parents[0]
            assert "id" in parent
            assert "text" in parent
            assert "source" in parent
            assert "page" in parent
            assert "corpus" in parent
            assert "chunk_type" in parent
            assert parent["chunk_type"] == "parent"

    def test_child_chunk_has_parent_id(self):
        """Child chunk has parent_id field."""
        from scripts.pipeline.chunker_langchain import chunk_markdown_langchain

        markdown = """## Test Section

This is a test section with enough content to create a chunk.
The content needs to be long enough to exceed the minimum token threshold.
Here is some more text to make sure we have sufficient content.
Additional sentences help ensure the chunk is large enough.
"""
        parents, children = chunk_markdown_langchain(markdown, "test.pdf", "fr", [], {})

        if children:
            child = children[0]
            assert "parent_id" in child
            assert child["chunk_type"] == "child"

    def test_all_chunks_have_page_ge_1(self):
        """All chunks have page >= 1 (ISO 42001 A.6.2.2)."""
        from scripts.pipeline.chunker_langchain import chunk_markdown_langchain

        markdown = """## Section 1

This is a test section with a lot of content to create multiple chunks.
We need enough text to exceed the minimum token threshold and generate chunks.
Here is more text to ensure we have sufficient content for testing.
Additional sentences help ensure we have multiple chunks generated.

## Section 2

Another section with content. This section also has enough text.
More content here to make sure we test the page propagation feature.
The algorithm should assign page numbers to all chunks.
"""
        parents, children = chunk_markdown_langchain(markdown, "test.pdf", "fr", [], {})

        for parent in parents:
            assert parent["page"] >= 1, f"Parent {parent['id']} has page < 1"
        for child in children:
            assert child["page"] >= 1, f"Child {child['id']} has page < 1"


class TestModuleConstants:
    """Tests for module constants (Parent-Child architecture)."""

    def test_parent_chunk_size_defined(self):
        """PARENT_CHUNK_SIZE constant is defined (1024 tokens for LLM context)."""
        from scripts.pipeline.chunker_langchain import PARENT_CHUNK_SIZE

        assert PARENT_CHUNK_SIZE == 1024

    def test_parent_chunk_overlap_defined(self):
        """PARENT_CHUNK_OVERLAP constant is defined (15% NVIDIA optimal)."""
        from scripts.pipeline.chunker_langchain import PARENT_CHUNK_OVERLAP

        assert PARENT_CHUNK_OVERLAP == 154

    def test_child_chunk_size_defined(self):
        """CHILD_CHUNK_SIZE constant is defined (450 tokens for embedding)."""
        from scripts.pipeline.chunker_langchain import CHILD_CHUNK_SIZE

        assert CHILD_CHUNK_SIZE == 450

    def test_child_chunk_overlap_defined(self):
        """CHILD_CHUNK_OVERLAP constant is defined (15% NVIDIA optimal)."""
        from scripts.pipeline.chunker_langchain import CHILD_CHUNK_OVERLAP

        assert CHILD_CHUNK_OVERLAP == 68

    def test_min_chunk_tokens_defined(self):
        """MIN_CHUNK_TOKENS constant is defined."""
        from scripts.pipeline.chunker_langchain import MIN_CHUNK_TOKENS

        assert MIN_CHUNK_TOKENS == 30
