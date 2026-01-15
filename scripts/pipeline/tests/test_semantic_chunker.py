"""
Tests for semantic_chunker.py - LangChain SemanticChunker

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO/IEC 25010 - Quality requirements
"""

from unittest.mock import Mock, patch


class TestChunkDocumentSemantic:
    """Tests for chunk_document_semantic function."""

    def test_empty_text_returns_empty_list(self):
        """Empty text should return empty list."""
        from scripts.pipeline.semantic_chunker import chunk_document_semantic

        mock_chunker = Mock()
        result = chunk_document_semantic(
            text="",
            chunker=mock_chunker,
            source="test.pdf",
            page=1,
        )

        assert result == []
        mock_chunker.split_text.assert_not_called()

    def test_short_text_returns_empty_list(self):
        """Text shorter than min_size should return empty list."""
        from scripts.pipeline.semantic_chunker import chunk_document_semantic

        mock_chunker = Mock()
        result = chunk_document_semantic(
            text="Short",
            chunker=mock_chunker,
            source="test.pdf",
            page=1,
            min_size=100,
        )

        assert result == []

    def test_successful_chunking(self):
        """Should chunk text and return list with metadata."""
        from scripts.pipeline.semantic_chunker import chunk_document_semantic

        mock_chunker = Mock()
        mock_chunker.split_text.return_value = [
            "First semantic chunk about chess rules.",
            "Second semantic chunk about time controls.",
        ]

        result = chunk_document_semantic(
            text="First semantic chunk about chess rules. Second semantic chunk about time controls.",
            chunker=mock_chunker,
            source="test.pdf",
            page=5,
            min_size=10,
        )

        assert len(result) == 2
        assert result[0]["source"] == "test.pdf"
        assert result[0]["page"] == 5
        assert result[0]["chunk_index"] == "0"
        assert result[1]["chunk_index"] == "1"

    def test_fallback_on_chunker_error(self):
        """Should fallback to simple split on chunker error."""
        from scripts.pipeline.semantic_chunker import chunk_document_semantic

        mock_chunker = Mock()
        mock_chunker.split_text.side_effect = Exception("Model error")

        test_text = "A" * 500  # 500 chars
        result = chunk_document_semantic(
            text=test_text,
            chunker=mock_chunker,
            source="test.pdf",
            page=1,
            min_size=50,
            max_size=200,
        )

        # Should have chunks from fallback split
        assert len(result) >= 1

    def test_splits_large_chunks(self):
        """Chunks larger than max_size should be split."""
        from scripts.pipeline.semantic_chunker import chunk_document_semantic

        mock_chunker = Mock()
        mock_chunker.split_text.return_value = ["A" * 3000]  # Large chunk

        result = chunk_document_semantic(
            text="A" * 3000,
            chunker=mock_chunker,
            source="test.pdf",
            page=1,
            min_size=50,
            max_size=500,
        )

        # Should be split into multiple sub-chunks
        assert len(result) > 1
        for chunk in result:
            assert len(chunk["text"]) <= 500


class TestCreateSemanticChunker:
    """Tests for create_semantic_chunker function."""

    @patch("scripts.pipeline.semantic_chunker.HuggingFaceEmbeddings")
    @patch("scripts.pipeline.semantic_chunker.SemanticChunker")
    def test_creates_chunker_with_defaults(self, mock_sc_class, mock_hf_class):
        """Should create chunker with default parameters."""
        from scripts.pipeline.semantic_chunker import (
            create_semantic_chunker,
            DEFAULT_MODEL,
            DEFAULT_THRESHOLD_TYPE,
            DEFAULT_THRESHOLD_AMOUNT,
        )

        mock_embeddings = Mock()
        mock_hf_class.return_value = mock_embeddings

        create_semantic_chunker()

        mock_hf_class.assert_called_once_with(
            model_name=DEFAULT_MODEL,
            model_kwargs={"device": "cpu"},
        )
        mock_sc_class.assert_called_once_with(
            embeddings=mock_embeddings,
            breakpoint_threshold_type=DEFAULT_THRESHOLD_TYPE,
            breakpoint_threshold_amount=DEFAULT_THRESHOLD_AMOUNT,
        )

    @patch("scripts.pipeline.semantic_chunker.HuggingFaceEmbeddings")
    @patch("scripts.pipeline.semantic_chunker.SemanticChunker")
    def test_creates_chunker_with_custom_params(self, mock_sc_class, mock_hf_class):
        """Should create chunker with custom parameters."""
        from scripts.pipeline.semantic_chunker import create_semantic_chunker

        mock_embeddings = Mock()
        mock_hf_class.return_value = mock_embeddings

        create_semantic_chunker(
            model_name="custom/model",
            threshold_type="standard_deviation",
            threshold_amount=1.5,
        )

        mock_hf_class.assert_called_once_with(
            model_name="custom/model",
            model_kwargs={"device": "cpu"},
        )
        mock_sc_class.assert_called_once_with(
            embeddings=mock_embeddings,
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=1.5,
        )
