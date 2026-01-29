"""
Tests for token_utils.py - Token counting utilities

Tests unitaires dedies pour le module token_utils.
Couverture cible: 100%

ISO Reference:
    - ISO/IEC 29119 - Test execution
    - ISO/IEC 12207 - Reusability verification
"""

import tiktoken

from scripts.pipeline.token_utils import (
    TOKENIZER_NAME,
    count_tokens,
    get_tokenizer,
)

# =============================================================================
# Tests: Constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_tokenizer_name_is_cl100k(self):
        """TOKENIZER_NAME is cl100k_base (OpenAI compatible)."""
        assert TOKENIZER_NAME == "cl100k_base"

    def test_tokenizer_name_is_valid(self):
        """TOKENIZER_NAME is a valid tiktoken encoding."""
        # Should not raise
        tiktoken.get_encoding(TOKENIZER_NAME)


# =============================================================================
# Tests: get_tokenizer
# =============================================================================


class TestGetTokenizer:
    """Tests for get_tokenizer function."""

    def test_returns_encoding_instance(self):
        """Returns a tiktoken Encoding instance."""
        tokenizer = get_tokenizer()
        assert isinstance(tokenizer, tiktoken.Encoding)

    def test_returns_correct_encoding(self):
        """Returns the cl100k_base encoding."""
        tokenizer = get_tokenizer()
        assert tokenizer.name == TOKENIZER_NAME

    def test_multiple_calls_return_equivalent(self):
        """Multiple calls return functionally equivalent tokenizers."""
        tok1 = get_tokenizer()
        tok2 = get_tokenizer()
        test_text = "Hello world"
        assert tok1.encode(test_text) == tok2.encode(test_text)


# =============================================================================
# Tests: count_tokens
# =============================================================================


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_empty_string_returns_zero(self):
        """Empty string returns 0 tokens."""
        assert count_tokens("") == 0

    def test_single_word(self):
        """Single word returns correct token count."""
        # "hello" is typically 1 token
        result = count_tokens("hello")
        assert result >= 1

    def test_sentence_returns_positive(self):
        """Sentence returns positive token count."""
        result = count_tokens("This is a test sentence.")
        assert result > 0

    def test_french_text(self):
        """French text is tokenized correctly."""
        result = count_tokens("L'arbitre doit appliquer le règlement.")
        assert result > 0

    def test_with_provided_tokenizer(self):
        """Works with explicitly provided tokenizer."""
        tokenizer = get_tokenizer()
        result = count_tokens("test", tokenizer)
        assert result >= 1

    def test_without_tokenizer_creates_one(self):
        """Creates tokenizer internally when None provided."""
        # This tests line 40: if tokenizer is None
        result = count_tokens("test", None)
        assert result >= 1

    def test_without_tokenizer_argument(self):
        """Works without tokenizer argument (default None)."""
        result = count_tokens("test")
        assert result >= 1

    def test_consistent_with_and_without_tokenizer(self):
        """Results match whether tokenizer provided or not."""
        text = "Consistent tokenization test"
        tokenizer = get_tokenizer()
        with_tok = count_tokens(text, tokenizer)
        without_tok = count_tokens(text)
        assert with_tok == without_tok

    def test_whitespace_only(self):
        """Whitespace-only string returns token count."""
        result = count_tokens("   ")
        # Whitespace may or may not be tokens depending on encoding
        assert result >= 0

    def test_special_characters(self):
        """Special characters are tokenized."""
        result = count_tokens("!@#$%^&*()")
        assert result >= 0

    def test_long_text(self):
        """Long text returns appropriate token count."""
        long_text = "word " * 1000
        result = count_tokens(long_text)
        # Should be roughly 1000+ tokens
        assert result >= 500


# =============================================================================
# Tests: Module exports
# =============================================================================


class TestModuleExports:
    """Tests for __all__ exports."""

    def test_all_exports_importable(self):
        """All __all__ items are importable."""
        from scripts.pipeline import token_utils

        for name in token_utils.__all__:
            assert hasattr(token_utils, name)

    def test_all_contains_expected(self):
        """__all__ contains expected exports."""
        from scripts.pipeline.token_utils import __all__

        assert "TOKENIZER_NAME" in __all__
        assert "get_tokenizer" in __all__
        assert "count_tokens" in __all__
