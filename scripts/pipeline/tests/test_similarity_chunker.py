"""
Tests for similarity_chunker.py - Sentence Transformers (Token-Aware)

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO/IEC 25010 - Quality requirements
"""

import numpy as np
import tiktoken
from unittest.mock import Mock, patch


class TestSplitSentences:
    """Tests for split_sentences function."""

    def test_splits_on_period(self):
        """Should split text on sentence boundaries."""
        from scripts.pipeline.similarity_chunker import split_sentences, get_tokenizer

        tokenizer = get_tokenizer()
        text = "First sentence. Second sentence. Third sentence."
        result = split_sentences(text, tokenizer, min_tokens=1)

        assert len(result) >= 1

    def test_filters_short_sentences(self):
        """Should filter sentences shorter than min_tokens."""
        from scripts.pipeline.similarity_chunker import split_sentences, get_tokenizer

        tokenizer = get_tokenizer()
        text = "Hi. This is a much longer sentence that should be kept."
        result = split_sentences(text, tokenizer, min_tokens=5)

        assert len(result) >= 1
        # All results should have at least min_tokens
        for s in result:
            assert len(tokenizer.encode(s)) >= 5

    def test_handles_empty_text(self):
        """Should handle empty text."""
        from scripts.pipeline.similarity_chunker import split_sentences, get_tokenizer

        tokenizer = get_tokenizer()
        result = split_sentences("", tokenizer, min_tokens=1)
        assert result == []


class TestComputeSimilarityBreaks:
    """Tests for compute_similarity_breaks function."""

    def test_empty_sentences_returns_empty(self):
        """Empty sentences should return no breaks."""
        from scripts.pipeline.similarity_chunker import compute_similarity_breaks

        mock_model = Mock()
        result = compute_similarity_breaks([], mock_model, threshold=0.5)

        assert result == []

    def test_single_sentence_returns_empty(self):
        """Single sentence should return no breaks."""
        from scripts.pipeline.similarity_chunker import compute_similarity_breaks

        mock_model = Mock()
        result = compute_similarity_breaks(["One sentence"], mock_model, threshold=0.5)

        assert result == []

    def test_finds_breaks_below_threshold(self):
        """Should find breaks where similarity is below threshold."""
        from scripts.pipeline.similarity_chunker import compute_similarity_breaks

        mock_model = Mock()
        # Create embeddings where sentence 1-2 are similar, 2-3 are different
        embeddings = np.array(
            [
                [1.0, 0.0],  # Sentence 1
                [0.95, 0.05],  # Sentence 2 (similar to 1)
                [0.0, 1.0],  # Sentence 3 (different)
            ]
        )
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        mock_model.encode.return_value = embeddings

        sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]
        result = compute_similarity_breaks(sentences, mock_model, threshold=0.5)

        # Should find break between sentence 2 and 3
        assert 2 in result


class TestMergeSmallChunks:
    """Tests for merge_small_chunks function."""

    def test_merges_small_chunks(self):
        """Should merge chunks smaller than min_tokens."""
        from scripts.pipeline.similarity_chunker import (
            merge_small_chunks,
            get_tokenizer,
        )

        tokenizer = get_tokenizer()
        chunks = [
            "Hi",
            "Hello",
            "This is a longer chunk that meets minimum requirements.",
        ]
        result = merge_small_chunks(chunks, tokenizer, min_tokens=10, max_tokens=500)

        # Small chunks should be merged
        assert len(result) <= len(chunks)

    def test_splits_large_chunks(self):
        """Should split chunks larger than max_tokens."""
        from scripts.pipeline.similarity_chunker import (
            merge_small_chunks,
            get_tokenizer,
        )

        tokenizer = get_tokenizer()
        # Create a chunk with ~200 tokens
        chunks = ["This is a test sentence about chess rules. " * 50]
        result = merge_small_chunks(chunks, tokenizer, min_tokens=10, max_tokens=50)

        # Should be split
        assert len(result) > 1
        for chunk in result:
            assert len(tokenizer.encode(chunk)) <= 50


class TestChunkDocumentSimilarity:
    """Tests for chunk_document_similarity function."""

    def test_empty_text_returns_empty(self):
        """Empty text should return empty list."""
        from scripts.pipeline.similarity_chunker import (
            chunk_document_similarity,
            get_tokenizer,
        )

        mock_model = Mock()
        tokenizer = get_tokenizer()
        result = chunk_document_similarity(
            text="",
            model=mock_model,
            source="test.pdf",
            page=1,
            tokenizer=tokenizer,
        )

        assert result == []

    def test_short_text_returns_empty(self):
        """Short text should return empty list."""
        from scripts.pipeline.similarity_chunker import (
            chunk_document_similarity,
            get_tokenizer,
        )

        mock_model = Mock()
        tokenizer = get_tokenizer()
        result = chunk_document_similarity(
            text="Short",
            model=mock_model,
            source="test.pdf",
            page=1,
            tokenizer=tokenizer,
            min_tokens=100,
        )

        assert result == []

    @patch("scripts.pipeline.similarity_chunker.compute_similarity_breaks")
    @patch("scripts.pipeline.similarity_chunker.split_sentences")
    def test_returns_chunks_with_metadata(self, mock_split, mock_breaks):
        """Should return chunks with correct metadata."""
        from scripts.pipeline.similarity_chunker import (
            chunk_document_similarity,
            get_tokenizer,
        )

        mock_split.return_value = [
            "First sentence about chess rules and tournament regulations.",
            "Second sentence about time controls and increment settings.",
        ]
        mock_breaks.return_value = [1]  # Break after first sentence

        mock_model = Mock()
        tokenizer = get_tokenizer()
        result = chunk_document_similarity(
            text="First sentence about chess. Second sentence about rules.",
            model=mock_model,
            source="test.pdf",
            page=5,
            tokenizer=tokenizer,
            min_tokens=5,
        )

        assert len(result) >= 1
        assert result[0]["source"] == "test.pdf"
        assert result[0]["page"] == 5
        assert "tokens" in result[0]


class TestTokenizer:
    """Tests for token counting functions."""

    def test_get_tokenizer_returns_encoding(self):
        """Should return tiktoken encoding."""
        from scripts.pipeline.similarity_chunker import get_tokenizer

        tokenizer = get_tokenizer()
        assert isinstance(tokenizer, tiktoken.Encoding)

    def test_count_tokens(self):
        """Should count tokens correctly."""
        from scripts.pipeline.similarity_chunker import get_tokenizer, _count_tokens

        tokenizer = get_tokenizer()

        assert _count_tokens("Hello world", tokenizer) > 0
        assert _count_tokens("", tokenizer) == 0
        # French text
        assert _count_tokens("Bonjour le monde", tokenizer) > 0
