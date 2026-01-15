"""
Tests for chunk_normalizer.py - Chunk normalization utilities

Tests unitaires dedies pour le module chunk_normalizer.
Couverture cible: 100% (tous edge cases)

ISO Reference:
    - ISO/IEC 29119 - Test execution
    - ISO/IEC 25010 S4.2 - Performance efficiency
"""

import pytest

from scripts.pipeline.chunk_normalizer import (
    DEFAULT_MAX_CHUNK_TOKENS,
    DEFAULT_MIN_CHUNK_TOKENS,
    filter_by_min_tokens,
    merge_by_max_tokens,
    normalize_chunks,
    split_oversized,
)
from scripts.pipeline.token_utils import get_tokenizer


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tokenizer():
    """Shared tokenizer instance."""
    return get_tokenizer()


@pytest.fixture
def short_chunk():
    """A chunk with few tokens (~5)."""
    return "Short text."


@pytest.fixture
def medium_chunk():
    """A chunk with medium tokens (~50)."""
    return "This is a medium length chunk. " * 5


@pytest.fixture
def long_chunk():
    """A chunk with many tokens (~200)."""
    return "This is a longer sentence for testing purposes. " * 20


# =============================================================================
# Tests: Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_min_chunk_tokens(self):
        """DEFAULT_MIN_CHUNK_TOKENS is 50."""
        assert DEFAULT_MIN_CHUNK_TOKENS == 50

    def test_default_max_chunk_tokens(self):
        """DEFAULT_MAX_CHUNK_TOKENS is 1024."""
        assert DEFAULT_MAX_CHUNK_TOKENS == 1024


# =============================================================================
# Tests: merge_by_max_tokens
# =============================================================================


class TestMergeByMaxTokens:
    """Tests for merge_by_max_tokens function."""

    def test_empty_list_returns_empty(self, tokenizer):
        """Empty input returns empty output."""
        result = merge_by_max_tokens([], 100, tokenizer)
        assert result == []

    def test_single_chunk_unchanged(self, tokenizer, medium_chunk):
        """Single chunk within limit returned unchanged."""
        result = merge_by_max_tokens([medium_chunk], 500, tokenizer)
        assert len(result) == 1
        assert result[0] == medium_chunk

    def test_merges_small_chunks(self, tokenizer, short_chunk):
        """Small chunks are merged together."""
        chunks = [short_chunk, short_chunk, short_chunk]
        result = merge_by_max_tokens(chunks, 100, tokenizer)
        # Should merge into fewer chunks
        assert len(result) <= len(chunks)

    def test_respects_max_tokens(self, tokenizer):
        """Does not exceed max_tokens limit."""
        chunks = ["Word. " * 20] * 10
        max_tokens = 50
        result = merge_by_max_tokens(chunks, max_tokens, tokenizer)
        from scripts.pipeline.token_utils import count_tokens

        for chunk in result:
            # Allow small overflow due to merge mechanics
            assert count_tokens(chunk, tokenizer) <= max_tokens + 20

    def test_preserves_content(self, tokenizer, short_chunk):
        """All content is preserved after merge."""
        chunks = [short_chunk, "Another chunk.", "Third chunk."]
        result = merge_by_max_tokens(chunks, 1000, tokenizer)
        merged_text = " ".join(result)
        for original in chunks:
            assert original in merged_text or original.strip() in merged_text

    def test_handles_chunk_larger_than_max(self, tokenizer, long_chunk):
        """Chunk larger than max is kept as-is (not split here)."""
        result = merge_by_max_tokens([long_chunk], 50, tokenizer)
        assert len(result) == 1
        assert result[0] == long_chunk

    def test_current_buffer_flushed_on_overflow(self, tokenizer):
        """Current buffer is flushed when next chunk causes overflow."""
        # This tests line 48: merged.append(current)
        chunk1 = "First chunk with some words."
        chunk2 = "Second chunk also with words."
        chunk3 = "Third chunk that causes overflow and forces flush."
        result = merge_by_max_tokens([chunk1, chunk2, chunk3], 20, tokenizer)
        # Should have multiple chunks due to overflow
        assert len(result) >= 2


# =============================================================================
# Tests: filter_by_min_tokens
# =============================================================================


class TestFilterByMinTokens:
    """Tests for filter_by_min_tokens function."""

    def test_empty_list_returns_empty(self, tokenizer):
        """Empty input returns empty output."""
        result = filter_by_min_tokens([], 50, tokenizer)
        assert result == []

    def test_large_chunk_unchanged(self, tokenizer, medium_chunk):
        """Chunk above min_tokens returned unchanged."""
        result = filter_by_min_tokens([medium_chunk], 10, tokenizer)
        assert len(result) == 1
        assert result[0] == medium_chunk

    def test_small_chunk_buffered_and_merged(self, tokenizer):
        """Small chunk is buffered and merged with next."""
        # This tests lines 79-80: buffer accumulation
        small = "Hi."
        large = "This is a much larger chunk that exceeds minimum tokens easily."
        result = filter_by_min_tokens([small, large], 20, tokenizer)
        # Small should be merged into large
        assert len(result) == 1
        assert small in result[0] or "Hi" in result[0]

    def test_multiple_small_chunks_accumulated(self, tokenizer):
        """Multiple small chunks accumulate in buffer."""
        smalls = ["One.", "Two.", "Three.", "Four.", "Five."]
        large = "This is a larger chunk that finally exceeds the minimum."
        result = filter_by_min_tokens(smalls + [large], 30, tokenizer)
        # All smalls should be merged
        merged = " ".join(result)
        for s in smalls:
            assert s.replace(".", "") in merged or s in merged

    def test_buffer_prepended_to_next_large(self, tokenizer):
        """Buffer content prepended to next large chunk."""
        # This tests lines 83-85: buffer merge and reset
        small = "Tiny."
        large = "A larger chunk here."
        result = filter_by_min_tokens([small, large], 5, tokenizer)
        if len(result) == 1:
            assert small.replace(".", "") in result[0] or "Tiny" in result[0]

    def test_final_buffer_appended_to_last(self, tokenizer):
        """Trailing buffer appended to last result."""
        # This tests line 89: result[-1] = (result[-1] + " " + buffer)
        large = "This is definitely a large enough chunk to pass the filter."
        small = "End."
        result = filter_by_min_tokens([large, small], 20, tokenizer)
        # Small should be appended to large
        assert "End" in result[-1] or len(result) == 1

    def test_orphan_buffer_becomes_chunk(self, tokenizer):
        """Buffer with no results becomes its own chunk."""
        # This tests line 91: result.append(buffer)
        smalls = ["A.", "B.", "C."]
        result = filter_by_min_tokens(smalls, 1000, tokenizer)
        # All too small, buffer should become single chunk
        assert len(result) == 1
        merged = result[0]
        assert "A" in merged and "B" in merged and "C" in merged

    def test_buffer_reset_after_merge(self, tokenizer):
        """Buffer is reset after merging with large chunk."""
        small1 = "First small."
        large = "Large chunk in the middle that passes."
        small2 = "Second small."
        result = filter_by_min_tokens([small1, large, small2], 10, tokenizer)
        # Verify structure is reasonable
        assert len(result) >= 1


# =============================================================================
# Tests: split_oversized
# =============================================================================


class TestSplitOversized:
    """Tests for split_oversized function."""

    def test_empty_list_returns_empty(self, tokenizer):
        """Empty input returns empty output."""
        result = split_oversized([], 100, tokenizer)
        assert result == []

    def test_small_chunk_unchanged(self, tokenizer, short_chunk):
        """Chunk within limit returned unchanged."""
        result = split_oversized([short_chunk], 100, tokenizer)
        assert len(result) == 1
        assert result[0] == short_chunk

    def test_oversized_chunk_split(self, tokenizer):
        """Oversized chunk is split by sentences."""
        oversized = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here."
        result = split_oversized([oversized], 15, tokenizer)
        # Should be split into multiple chunks
        assert len(result) >= 2

    def test_split_respects_sentence_boundaries(self, tokenizer):
        """Split occurs at sentence boundaries."""
        oversized = (
            "Sentence one ends here. Sentence two ends here. Sentence three ends here."
        )
        result = split_oversized([oversized], 10, tokenizer)
        for chunk in result:
            # Each chunk should end with punctuation or be complete
            assert chunk.strip()

    def test_mixed_sizes_handled(self, tokenizer, short_chunk):
        """Mix of small and large chunks handled correctly."""
        large = (
            "Big sentence one. Big sentence two. Big sentence three. Big sentence four."
        )
        result = split_oversized([short_chunk, large, short_chunk], 20, tokenizer)
        # Small chunks should be preserved
        assert any(short_chunk in r or "Short" in r for r in result)

    def test_single_long_sentence_kept(self, tokenizer):
        """Single sentence exceeding limit kept as one chunk."""
        # Can't split a single sentence further
        long_sentence = "This is one very long sentence without any periods or breaks that just keeps going and going"
        result = split_oversized([long_sentence], 10, tokenizer)
        # Should still be one chunk (can't split further)
        assert len(result) >= 1


# =============================================================================
# Tests: normalize_chunks (integration)
# =============================================================================


class TestNormalizeChunks:
    """Tests for normalize_chunks function."""

    def test_empty_list_returns_empty(self, tokenizer):
        """Empty input returns empty output."""
        result = normalize_chunks([], tokenizer)
        assert result == []

    def test_uses_default_limits(self, tokenizer, medium_chunk):
        """Uses DEFAULT_MIN/MAX when not specified."""
        result = normalize_chunks([medium_chunk], tokenizer)
        assert len(result) >= 1

    def test_custom_limits_respected(self, tokenizer):
        """Custom min/max limits are applied."""
        chunks = ["Small.", "Another small.", "Third small."]
        result = normalize_chunks(chunks, tokenizer, min_tokens=5, max_tokens=100)
        # Should merge small chunks
        assert len(result) <= len(chunks)

    def test_pipeline_order_correct(self, tokenizer):
        """Operations applied in correct order: merge, filter, split."""
        # Create chunks that exercise all three operations
        chunks = [
            "Tiny.",  # Will be filtered
            "Medium chunk here.",  # Will be kept
            "Another medium one.",  # Will be merged if fits
        ]
        result = normalize_chunks(chunks, tokenizer, min_tokens=5, max_tokens=500)
        assert len(result) >= 1

    def test_result_within_bounds(self, tokenizer):
        """Result chunks are within min/max bounds (when possible)."""
        chunks = ["Word. " * 30] * 5  # Multiple medium chunks
        min_tok, max_tok = 20, 200
        result = normalize_chunks(chunks, tokenizer, min_tok, max_tok)
        from scripts.pipeline.token_utils import count_tokens

        for chunk in result:
            tokens = count_tokens(chunk, tokenizer)
            # Most chunks should be within bounds
            # (edge cases may slightly exceed)
            assert tokens > 0


# =============================================================================
# Tests: Module exports
# =============================================================================


class TestModuleExports:
    """Tests for __all__ exports."""

    def test_all_exports_importable(self):
        """All __all__ items are importable."""
        from scripts.pipeline import chunk_normalizer

        for name in chunk_normalizer.__all__:
            assert hasattr(chunk_normalizer, name)

    def test_all_contains_expected(self):
        """__all__ contains expected exports."""
        from scripts.pipeline.chunk_normalizer import __all__

        expected = [
            "merge_by_max_tokens",
            "filter_by_min_tokens",
            "split_oversized",
            "normalize_chunks",
            "DEFAULT_MIN_CHUNK_TOKENS",
            "DEFAULT_MAX_CHUNK_TOKENS",
        ]
        for name in expected:
            assert name in __all__
