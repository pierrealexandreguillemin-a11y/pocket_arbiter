"""
Token Utilities - Pocket Arbiter

Fonctions utilitaires pour comptage de tokens via tiktoken et EmbeddingGemma.
Module partage entre similarity_chunker, sentence_chunker, chunker.

ISO Reference:
    - ISO/IEC 12207 - Reusability
    - ISO/IEC 25010 - Maintainability
    - ISO/IEC 42001 A.6.2.2 - AI traceability (tokenizer consistency)
"""

from typing import Any

import tiktoken

# Tokenizer compatible OpenAI/LLM (default)
TOKENIZER_NAME = "cl100k_base"

# EmbeddingGemma tokenizer for HybridChunker (ISO 42001)
# QAT model = pipeline unique (chunking → encoding → fine-tuning → deployment)
EMBED_MODEL_ID = "google/embeddinggemma-300m-qat-q4_0-unquantized"

# Singleton for EmbeddingGemma tokenizer (lazy load)
_gemma_tokenizer: Any = None


def get_tokenizer() -> tiktoken.Encoding:
    """
    Get tiktoken tokenizer for token counting.

    Returns:
        tiktoken Encoding instance.
    """
    return tiktoken.get_encoding(TOKENIZER_NAME)


def get_gemma_tokenizer() -> Any:
    """
    Get EmbeddingGemma tokenizer (lazy singleton).

    Uses same tokenizer as embedding model for consistency (ISO 42001).

    Returns:
        HuggingFace PreTrainedTokenizer instance.
    """
    global _gemma_tokenizer
    if _gemma_tokenizer is None:
        from transformers import AutoTokenizer
        _gemma_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    return _gemma_tokenizer


def count_tokens(text: str, tokenizer: tiktoken.Encoding | None = None) -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for.
        tokenizer: Optional tokenizer instance. Created if None.

    Returns:
        Number of tokens.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))


def count_tokens_gemma(text: str) -> int:
    """
    Count tokens using EmbeddingGemma tokenizer.

    Uses same tokenizer as embedding model for consistency (ISO 42001 A.6.2.2).

    Args:
        text: Text to count tokens for.

    Returns:
        Number of tokens.
    """
    tokenizer = get_gemma_tokenizer()
    return len(tokenizer.encode(text))


__all__ = [
    "TOKENIZER_NAME",
    "EMBED_MODEL_ID",
    "get_tokenizer",
    "get_gemma_tokenizer",
    "count_tokens",
    "count_tokens_gemma",
]
