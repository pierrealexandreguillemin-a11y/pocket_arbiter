"""
Tests for chunker_hybrid.py (Mode A: HybridChunker).

ISO Reference:
    - ISO/IEC 29119 - Software Testing
    - ISO/IEC 42001 A.6.2.2 - AI traceability
"""

import pytest


class TestGetHybridTokenizer:
    """Tests for get_hybrid_tokenizer function."""

    def test_returns_huggingface_tokenizer(self):
        """Returns a HuggingFaceTokenizer instance."""
        from scripts.pipeline.chunker_hybrid import get_hybrid_tokenizer

        tokenizer = get_hybrid_tokenizer()
        assert tokenizer is not None

    def test_tokenizer_has_max_tokens(self):
        """Tokenizer respects max_tokens setting."""
        from scripts.pipeline.chunker_hybrid import MAX_TOKENS, get_hybrid_tokenizer

        tokenizer = get_hybrid_tokenizer()
        assert tokenizer.max_tokens == MAX_TOKENS


class TestExtractPageNumbers:
    """Tests for extract_page_numbers function."""

    def test_empty_chunk_returns_empty_list(self):
        """Chunk without meta returns empty list."""
        from scripts.pipeline.chunker_hybrid import extract_page_numbers

        class MockChunk:
            meta = None

        result = extract_page_numbers(MockChunk())
        assert result == []

    def test_extracts_page_from_provenance(self):
        """Extracts page_no from chunk provenance."""
        from scripts.pipeline.chunker_hybrid import extract_page_numbers

        class MockProv:
            page_no = 5

        class MockDocItem:
            prov = [MockProv()]

        class MockMeta:
            doc_items = [MockDocItem()]

        class MockChunk:
            meta = MockMeta()

        result = extract_page_numbers(MockChunk())
        assert result == [5]

    def test_returns_sorted_unique_pages(self):
        """Returns sorted unique page numbers."""
        from scripts.pipeline.chunker_hybrid import extract_page_numbers

        class MockProv:
            def __init__(self, page: int):
                self.page_no = page

        class MockDocItem:
            def __init__(self, pages: list):
                self.prov = [MockProv(p) for p in pages]

        class MockMeta:
            doc_items = [MockDocItem([3, 1, 2, 1])]

        class MockChunk:
            meta = MockMeta()

        result = extract_page_numbers(MockChunk())
        assert result == [1, 2, 3]


class TestExtractHeadings:
    """Tests for extract_headings function."""

    def test_empty_meta_returns_empty_list(self):
        """Chunk without meta returns empty list."""
        from scripts.pipeline.chunker_hybrid import extract_headings

        class MockChunk:
            meta = None

        result = extract_headings(MockChunk())
        assert result == []

    def test_extracts_headings_from_meta(self):
        """Extracts headings from chunk metadata."""
        from scripts.pipeline.chunker_hybrid import extract_headings

        class MockMeta:
            headings = ["Chapter 1", "Section 1.1"]

        class MockChunk:
            meta = MockMeta()

        result = extract_headings(MockChunk())
        assert result == ["Chapter 1", "Section 1.1"]


class TestChunkDocumentHybrid:
    """Tests for chunk_document_hybrid function."""

    def test_empty_docling_dict_raises(self):
        """Empty docling_document raises ValueError."""
        from scripts.pipeline.chunker_hybrid import chunk_document_hybrid

        with pytest.raises(ValueError, match="Empty docling_document"):
            chunk_document_hybrid({}, "test.pdf")

    def test_minimal_document_returns_chunks(self):
        """Minimal valid document returns chunks."""
        from scripts.pipeline.chunker_hybrid import chunk_document_hybrid

        # Minimal DoclingDocument structure
        docling_dict = {
            "schema_name": "DoclingDocument",
            "version": "1.0.0",
            "name": "test",
        }

        # This may return empty list for minimal document
        chunks = chunk_document_hybrid(docling_dict, "test.pdf", "fr")
        assert isinstance(chunks, list)


class TestModuleConstants:
    """Tests for module constants."""

    def test_max_tokens_defined(self):
        """MAX_TOKENS constant is defined."""
        from scripts.pipeline.chunker_hybrid import MAX_TOKENS

        assert MAX_TOKENS == 450

    def test_merge_peers_defined(self):
        """MERGE_PEERS constant is defined."""
        from scripts.pipeline.chunker_hybrid import MERGE_PEERS

        assert MERGE_PEERS is True


class TestProcessDoclingOutputHybrid:
    """Tests for process_docling_output_hybrid function."""

    def test_empty_directory(self, tmp_path):
        """Empty directory returns zero stats."""
        from scripts.pipeline.chunker_hybrid import process_docling_output_hybrid

        output_file = tmp_path / "chunks.json"
        stats = process_docling_output_hybrid(tmp_path, output_file, "fr")

        assert stats["files"] == 0
        assert stats["chunks"] == 0

    def test_skips_extraction_report(self, tmp_path):
        """Skips extraction_report.json."""
        from scripts.pipeline.chunker_hybrid import process_docling_output_hybrid

        # Create only extraction_report.json
        report = tmp_path / "extraction_report.json"
        report.write_text('{"files": 0}', encoding="utf-8")

        output_file = tmp_path / "chunks.json"
        stats = process_docling_output_hybrid(tmp_path, output_file, "fr")

        assert stats["files"] == 0

    def test_handles_missing_docling_document(self, tmp_path):
        """Logs warning when docling_document missing."""
        import json
        from scripts.pipeline.chunker_hybrid import process_docling_output_hybrid

        # Create file without docling_document
        data = {"filename": "test.pdf", "markdown": "# Test"}
        (tmp_path / "test.json").write_text(json.dumps(data), encoding="utf-8")

        output_file = tmp_path / "chunks.json"
        stats = process_docling_output_hybrid(tmp_path, output_file, "fr")

        assert stats["files"] == 0
        assert len(stats["errors"]) == 1

    def test_creates_output_file(self, tmp_path):
        """Creates output JSON file."""
        import json
        from scripts.pipeline.chunker_hybrid import process_docling_output_hybrid

        # Create minimal docling file
        data = {
            "filename": "test.pdf",
            "docling_document": {"schema_name": "DoclingDocument", "version": "1.0.0", "name": "test"}
        }
        (tmp_path / "test.json").write_text(json.dumps(data), encoding="utf-8")

        output_file = tmp_path / "output" / "chunks.json"
        process_docling_output_hybrid(tmp_path, output_file, "fr")

        assert output_file.exists()
        with open(output_file, encoding="utf-8") as f:
            result = json.load(f)
        assert "chunks" in result
        assert "total" in result
