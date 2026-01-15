"""
Tests for sentence_chunker.py - LlamaIndex SentenceSplitter (Token-Aware)

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO/IEC 25010 - Quality requirements
"""

import tiktoken
from unittest.mock import Mock, patch


class TestChunkDocumentSentence:
    """Tests for chunk_document_sentence function."""

    def test_empty_text_returns_empty_list(self):
        """Empty text should return empty list."""
        from scripts.pipeline.sentence_chunker import (
            chunk_document_sentence,
            get_tokenizer,
        )

        mock_splitter = Mock()
        tokenizer = get_tokenizer()
        result = chunk_document_sentence(
            text="",
            splitter=mock_splitter,
            source="test.pdf",
            page=1,
            tokenizer=tokenizer,
        )

        assert result == []
        mock_splitter.split_text.assert_not_called()

    def test_short_text_returns_empty_list(self):
        """Text shorter than min_tokens should return empty list."""
        from scripts.pipeline.sentence_chunker import (
            chunk_document_sentence,
            get_tokenizer,
        )

        mock_splitter = Mock()
        tokenizer = get_tokenizer()
        result = chunk_document_sentence(
            text="Hi",
            splitter=mock_splitter,
            source="test.pdf",
            page=1,
            tokenizer=tokenizer,
            min_tokens=50,
        )

        assert result == []

    def test_successful_chunking(self):
        """Should chunk text and return list with metadata."""
        from scripts.pipeline.sentence_chunker import (
            chunk_document_sentence,
            get_tokenizer,
        )

        mock_splitter = Mock()
        # Return chunks with enough tokens (each ~10 tokens)
        mock_splitter.split_text.return_value = [
            "First sentence chunk about chess rules and regulations for tournament play.",
            "Second sentence chunk about time controls and increment settings for games.",
        ]

        tokenizer = get_tokenizer()
        result = chunk_document_sentence(
            text="First sentence chunk. Second sentence chunk.",
            splitter=mock_splitter,
            source="test.pdf",
            page=3,
            tokenizer=tokenizer,
            min_tokens=5,
        )

        assert len(result) == 2
        assert result[0]["source"] == "test.pdf"
        assert result[0]["page"] == 3
        assert result[0]["chunk_index"] == "0"
        assert "First sentence" in result[0]["text"]
        assert "tokens" in result[0]

    def test_filters_small_chunks(self):
        """Chunks smaller than min_tokens should be filtered."""
        from scripts.pipeline.sentence_chunker import (
            chunk_document_sentence,
            get_tokenizer,
        )

        mock_splitter = Mock()
        mock_splitter.split_text.return_value = [
            "Hi",  # Too small (1 token)
            "This is a longer chunk that should be kept because it has many tokens.",
        ]

        tokenizer = get_tokenizer()
        result = chunk_document_sentence(
            text="Short. This is a longer chunk.",
            splitter=mock_splitter,
            source="test.pdf",
            page=1,
            tokenizer=tokenizer,
            min_tokens=10,
        )

        assert len(result) == 1
        assert "longer chunk" in result[0]["text"]


class TestCreateSentenceSplitter:
    """Tests for create_sentence_splitter function."""

    @patch("scripts.pipeline.sentence_chunker.SentenceSplitter")
    def test_creates_splitter_with_defaults(self, mock_ss_class):
        """Should create splitter with default parameters."""
        from scripts.pipeline.sentence_chunker import (
            create_sentence_splitter,
            DEFAULT_CHUNK_SIZE_TOKENS,
            DEFAULT_CHUNK_OVERLAP_TOKENS,
        )

        create_sentence_splitter()

        mock_ss_class.assert_called_once()
        call_kwargs = mock_ss_class.call_args[1]
        assert call_kwargs["chunk_size"] == DEFAULT_CHUNK_SIZE_TOKENS
        assert call_kwargs["chunk_overlap"] == DEFAULT_CHUNK_OVERLAP_TOKENS

    @patch("scripts.pipeline.sentence_chunker.SentenceSplitter")
    def test_creates_splitter_with_custom_params(self, mock_ss_class):
        """Should create splitter with custom parameters."""
        from scripts.pipeline.sentence_chunker import create_sentence_splitter

        create_sentence_splitter(
            chunk_size=1024,
            chunk_overlap=256,
        )

        call_kwargs = mock_ss_class.call_args[1]
        assert call_kwargs["chunk_size"] == 1024
        assert call_kwargs["chunk_overlap"] == 256


class TestTokenizer:
    """Tests for token counting functions."""

    def test_get_tokenizer_returns_encoding(self):
        """Should return tiktoken encoding."""
        from scripts.pipeline.sentence_chunker import get_tokenizer

        tokenizer = get_tokenizer()
        assert isinstance(tokenizer, tiktoken.Encoding)

    def test_create_token_counter(self):
        """Should create callable token counter."""
        from scripts.pipeline.sentence_chunker import (
            get_tokenizer,
            create_token_counter,
        )

        tokenizer = get_tokenizer()
        counter = create_token_counter(tokenizer)

        assert callable(counter)
        assert counter("Hello world") > 0
        assert counter("") == 0
