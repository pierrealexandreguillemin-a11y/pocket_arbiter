"""
Tests for similarity_chunker.py - Sentence Transformers

ISO Reference:
    - ISO/IEC 29119 - Software testing
    - ISO/IEC 25010 - Quality requirements
"""

import numpy as np
from unittest.mock import Mock, patch


class TestSplitSentences:
    """Tests for split_sentences function."""

    def test_splits_on_period(self):
        """Should split text on sentence boundaries."""
        from scripts.pipeline.similarity_chunker import split_sentences

        text = "First sentence. Second sentence. Third sentence."
        result = split_sentences(text, min_length=5)

        assert len(result) >= 1

    def test_filters_short_sentences(self):
        """Should filter sentences shorter than min_length."""
        from scripts.pipeline.similarity_chunker import split_sentences

        text = "Hi. This is a much longer sentence that should be kept."
        result = split_sentences(text, min_length=20)

        assert len(result) >= 1
        for s in result:
            assert len(s) >= 20

    def test_handles_empty_text(self):
        """Should handle empty text."""
        from scripts.pipeline.similarity_chunker import split_sentences

        result = split_sentences("", min_length=10)
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
        """Should merge chunks smaller than min_length."""
        from scripts.pipeline.similarity_chunker import merge_small_chunks

        chunks = ["Hi", "Hello", "This is a longer chunk that meets minimum."]
        result = merge_small_chunks(chunks, min_length=30, max_length=500)

        # Small chunks should be merged
        assert len(result) <= len(chunks)

    def test_splits_large_chunks(self):
        """Should split chunks larger than max_length."""
        from scripts.pipeline.similarity_chunker import merge_small_chunks

        chunks = ["A" * 1000]  # Large chunk
        result = merge_small_chunks(chunks, min_length=10, max_length=200)

        # Should be split
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 200


class TestChunkDocumentSimilarity:
    """Tests for chunk_document_similarity function."""

    def test_empty_text_returns_empty(self):
        """Empty text should return empty list."""
        from scripts.pipeline.similarity_chunker import chunk_document_similarity

        mock_model = Mock()
        result = chunk_document_similarity(
            text="",
            model=mock_model,
            source="test.pdf",
            page=1,
        )

        assert result == []

    def test_short_text_returns_empty(self):
        """Short text should return empty list."""
        from scripts.pipeline.similarity_chunker import chunk_document_similarity

        mock_model = Mock()
        result = chunk_document_similarity(
            text="Short",
            model=mock_model,
            source="test.pdf",
            page=1,
            min_length=100,
        )

        assert result == []

    @patch("scripts.pipeline.similarity_chunker.compute_similarity_breaks")
    @patch("scripts.pipeline.similarity_chunker.split_sentences")
    def test_returns_chunks_with_metadata(self, mock_split, mock_breaks):
        """Should return chunks with correct metadata."""
        from scripts.pipeline.similarity_chunker import chunk_document_similarity

        mock_split.return_value = [
            "First sentence about chess.",
            "Second sentence about rules.",
        ]
        mock_breaks.return_value = [1]  # Break after first sentence

        mock_model = Mock()
        result = chunk_document_similarity(
            text="First sentence about chess. Second sentence about rules.",
            model=mock_model,
            source="test.pdf",
            page=5,
            min_length=10,
        )

        assert len(result) >= 1
        assert result[0]["source"] == "test.pdf"
        assert result[0]["page"] == 5
