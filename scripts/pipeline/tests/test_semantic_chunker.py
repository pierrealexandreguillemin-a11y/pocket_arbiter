"""
Tests for semantic_chunker.py - LangChain SemanticChunker (Token-Aware)

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO/IEC 25010 - Quality requirements
"""

import tiktoken
from unittest.mock import Mock, patch


class TestChunkDocumentSemantic:
    """Tests for chunk_document_semantic function."""

    def test_empty_text_returns_empty_list(self):
        """Empty text should return empty list."""
        from scripts.pipeline.semantic_chunker import (
            chunk_document_semantic,
            get_tokenizer,
        )

        mock_chunker = Mock()
        tokenizer = get_tokenizer()
        result = chunk_document_semantic(
            text="",
            chunker=mock_chunker,
            source="test.pdf",
            page=1,
            tokenizer=tokenizer,
        )

        assert result == []
        mock_chunker.split_text.assert_not_called()

    def test_short_text_returns_empty_list(self):
        """Text shorter than min_tokens should return empty list."""
        from scripts.pipeline.semantic_chunker import (
            chunk_document_semantic,
            get_tokenizer,
        )

        mock_chunker = Mock()
        tokenizer = get_tokenizer()
        result = chunk_document_semantic(
            text="Short",
            chunker=mock_chunker,
            source="test.pdf",
            page=1,
            tokenizer=tokenizer,
            min_tokens=100,
        )

        assert result == []

    def test_successful_chunking(self):
        """Should chunk text and return list with metadata."""
        from scripts.pipeline.semantic_chunker import (
            chunk_document_semantic,
            get_tokenizer,
        )

        mock_chunker = Mock()
        # Return chunks with enough tokens
        mock_chunker.split_text.return_value = [
            "First semantic chunk about chess rules and tournament regulations for competitive play.",
            "Second semantic chunk about time controls and increment settings for official games.",
        ]

        tokenizer = get_tokenizer()
        long_text = "First semantic chunk about chess rules. " * 20
        result = chunk_document_semantic(
            text=long_text,
            chunker=mock_chunker,
            source="test.pdf",
            page=5,
            tokenizer=tokenizer,
            min_tokens=5,
        )

        assert len(result) == 2
        assert result[0]["source"] == "test.pdf"
        assert result[0]["page"] == 5
        assert result[0]["chunk_index"] == "0"
        assert result[1]["chunk_index"] == "1"
        assert "tokens" in result[0]

    def test_fallback_on_chunker_error(self):
        """Should fallback to simple split on chunker error."""
        from scripts.pipeline.semantic_chunker import (
            chunk_document_semantic,
            get_tokenizer,
        )

        mock_chunker = Mock()
        mock_chunker.split_text.side_effect = Exception("Model error")

        tokenizer = get_tokenizer()
        # Create text with enough tokens (~500 tokens)
        test_text = "This is a test sentence about chess rules. " * 100
        result = chunk_document_semantic(
            text=test_text,
            chunker=mock_chunker,
            source="test.pdf",
            page=1,
            tokenizer=tokenizer,
            min_tokens=50,
            max_tokens=200,
        )

        # Should have chunks from fallback split
        assert len(result) >= 1

    def test_splits_large_chunks(self):
        """Chunks larger than max_tokens should be split."""
        from scripts.pipeline.semantic_chunker import (
            chunk_document_semantic,
            get_tokenizer,
        )

        mock_chunker = Mock()
        # Create a large chunk (~1000 tokens)
        large_chunk = (
            "This is a test sentence about chess rules and regulations. " * 150
        )
        mock_chunker.split_text.return_value = [large_chunk]

        tokenizer = get_tokenizer()
        result = chunk_document_semantic(
            text=large_chunk,
            chunker=mock_chunker,
            source="test.pdf",
            page=1,
            tokenizer=tokenizer,
            min_tokens=50,
            max_tokens=200,
        )

        # Should be split into multiple sub-chunks
        assert len(result) > 1
        for chunk in result:
            assert chunk["tokens"] <= 200


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


class TestTokenizer:
    """Tests for token counting functions."""

    def test_get_tokenizer_returns_encoding(self):
        """Should return tiktoken encoding."""
        from scripts.pipeline.semantic_chunker import get_tokenizer

        tokenizer = get_tokenizer()
        assert isinstance(tokenizer, tiktoken.Encoding)

    def test_count_tokens(self):
        """Should count tokens correctly."""
        from scripts.pipeline.semantic_chunker import get_tokenizer, _count_tokens

        tokenizer = get_tokenizer()

        assert _count_tokens("Hello world", tokenizer) > 0
        assert _count_tokens("", tokenizer) == 0
        # French text
        assert _count_tokens("Bonjour le monde", tokenizer) > 0
