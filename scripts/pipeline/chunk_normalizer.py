"""
Chunk Normalizer - Pocket Arbiter

Fonctions de normalisation des chunks (merge/split par tokens).
Utilisees par similarity_chunker et autres chunkers.

ISO Reference:
    - ISO/IEC 25010 S4.2 - Performance efficiency
    - ISO/IEC 12207 - Reusability
"""

import re

import tiktoken

from scripts.pipeline.token_utils import count_tokens

# Default token limits
DEFAULT_MIN_CHUNK_TOKENS = 50
DEFAULT_MAX_CHUNK_TOKENS = 1024


def merge_by_max_tokens(
    chunks: list[str], max_tokens: int, tokenizer: tiktoken.Encoding
) -> list[str]:
    """
    Merge consecutive chunks up to max_tokens.

    Args:
        chunks: List of chunk texts.
        max_tokens: Maximum tokens per merged chunk.
        tokenizer: Tokenizer for counting.

    Returns:
        Merged chunks.
    """
    merged = []
    current = ""
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk, tokenizer)
        if current_tokens + chunk_tokens <= max_tokens:
            current = (current + " " + chunk).strip() if current else chunk
            current_tokens += chunk_tokens
        else:
            if current:
                merged.append(current)
            current = chunk
            current_tokens = chunk_tokens

    if current:
        merged.append(current)

    return merged


def filter_by_min_tokens(
    chunks: list[str], min_tokens: int, tokenizer: tiktoken.Encoding
) -> list[str]:
    """
    Filter and merge chunks smaller than min_tokens.

    Args:
        chunks: List of chunk texts.
        min_tokens: Minimum tokens per chunk.
        tokenizer: Tokenizer for counting.

    Returns:
        Filtered chunks with small ones merged.
    """
    result = []
    buffer = ""
    buffer_tokens = 0

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk, tokenizer)
        if chunk_tokens < min_tokens:
            buffer = (buffer + " " + chunk).strip() if buffer else chunk
            buffer_tokens += chunk_tokens
        else:
            if buffer:
                chunk = (buffer + " " + chunk).strip()
                buffer = ""
                buffer_tokens = 0
            result.append(chunk)

    if buffer and result:
        result[-1] = (result[-1] + " " + buffer).strip()
    elif buffer:
        result.append(buffer)

    return result


def split_oversized(
    chunks: list[str], max_tokens: int, tokenizer: tiktoken.Encoding
) -> list[str]:
    """
    Split chunks that exceed max_tokens by sentences.

    Args:
        chunks: List of chunk texts.
        max_tokens: Maximum tokens per chunk.
        tokenizer: Tokenizer for counting.

    Returns:
        Chunks with oversized ones split.
    """
    final = []

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk, tokenizer)
        if chunk_tokens > max_tokens:
            # Split by sentences to preserve semantic boundaries
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            current = ""
            current_tokens = 0

            for sentence in sentences:
                sentence_tokens = count_tokens(sentence, tokenizer)
                if current_tokens + sentence_tokens <= max_tokens:
                    current = (
                        (current + " " + sentence).strip() if current else sentence
                    )
                    current_tokens += sentence_tokens
                else:
                    if current:
                        final.append(current)
                    current = sentence
                    current_tokens = sentence_tokens

            if current:
                final.append(current)
        else:
            final.append(chunk)

    return final


def normalize_chunks(
    chunks: list[str],
    tokenizer: tiktoken.Encoding,
    min_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
    max_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
) -> list[str]:
    """
    Normalize chunks: merge small, split large (token-aware).

    Args:
        chunks: List of chunk texts.
        tokenizer: Tokenizer for counting.
        min_tokens: Minimum tokens per chunk.
        max_tokens: Maximum tokens per chunk.

    Returns:
        Normalized chunks.
    """
    merged = merge_by_max_tokens(chunks, max_tokens, tokenizer)
    filtered = filter_by_min_tokens(merged, min_tokens, tokenizer)
    return split_oversized(filtered, max_tokens, tokenizer)


__all__ = [
    "merge_by_max_tokens",
    "filter_by_min_tokens",
    "split_oversized",
    "normalize_chunks",
    "DEFAULT_MIN_CHUNK_TOKENS",
    "DEFAULT_MAX_CHUNK_TOKENS",
]
