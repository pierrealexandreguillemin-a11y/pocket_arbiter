"""
Token Utilities - Pocket Arbiter

Fonctions utilitaires pour comptage de tokens via tiktoken.
Module partage entre similarity_chunker, sentence_chunker, chunker.

ISO Reference:
    - ISO/IEC 12207 - Reusability
    - ISO/IEC 25010 - Maintainability
"""

import tiktoken

# Tokenizer compatible OpenAI/LLM
TOKENIZER_NAME = "cl100k_base"


def get_tokenizer() -> tiktoken.Encoding:
    """
    Get tiktoken tokenizer for token counting.

    Returns:
        tiktoken Encoding instance.
    """
    return tiktoken.get_encoding(TOKENIZER_NAME)


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


__all__ = ["TOKENIZER_NAME", "get_tokenizer", "count_tokens"]
