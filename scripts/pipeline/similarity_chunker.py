"""
Similarity Chunker - Pocket Arbiter (Sentence Transformers)

Utilise Sentence Transformers pour chunking basé sur la similarité cosinus.
Détecte les points de rupture sémantique via embeddings directs.

IMPORTANT: Utilise des TOKENS (via tiktoken), pas des caractères.
           512 tokens ≈ 2048 caractères (ratio moyen 1:4)

ISO Reference:
    - ISO/IEC 25010 S4.2 - Performance efficiency (Recall >= 80%)
    - ISO/IEC 42001 - AI traceability

Usage:
    python similarity_chunker.py --input corpus/processed/raw_fr --output corpus/processed/chunks_similarity_fr.json
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Constants (TOKENS, not characters) ---

DEFAULT_MODEL = "intfloat/multilingual-e5-base"  # Best for French semantic similarity
DEFAULT_THRESHOLD = 0.5  # Cosine similarity threshold for breaks
MIN_SENTENCE_TOKENS = 5  # Minimum tokens per sentence
MIN_CHUNK_TOKENS = 50  # Minimum tokens per chunk
MAX_CHUNK_TOKENS = 1024  # Maximum tokens per chunk
TOKENIZER_NAME = "cl100k_base"  # Compatible OpenAI/LLM


def get_tokenizer() -> tiktoken.Encoding:
    """Get tiktoken tokenizer for token counting."""
    return tiktoken.get_encoding(TOKENIZER_NAME)


def _count_tokens(text: str, tokenizer: tiktoken.Encoding) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text))


def split_sentences(
    text: str,
    tokenizer: tiktoken.Encoding,
    min_tokens: int = MIN_SENTENCE_TOKENS,
) -> list[str]:
    """
    Split text into sentences using common delimiters.

    Args:
        text: Input text.
        tokenizer: Tokenizer for token counting.
        min_tokens: Minimum tokens per sentence.

    Returns:
        List of sentences.
    """
    # Split on sentence boundaries
    pattern = r"(?<=[.!?])\s+(?=[A-ZÀ-Ü])|(?<=\n)\s*(?=\S)"
    raw_sentences = re.split(pattern, text)

    # Filter and clean by token count
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if _count_tokens(s, tokenizer) >= min_tokens:
            sentences.append(s)

    return sentences


def compute_similarity_breaks(
    sentences: list[str],
    model: SentenceTransformer,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[int]:
    """
    Find semantic break points using cosine similarity.

    Args:
        sentences: List of sentences.
        model: Sentence transformer model.
        threshold: Similarity threshold (lower = more breaks).

    Returns:
        List of break indices.
    """
    if len(sentences) < 2:
        return []

    # Generate embeddings for all sentences
    embeddings = model.encode(
        sentences, normalize_embeddings=True, show_progress_bar=False
    )

    # Compute cosine similarity between consecutive sentences
    breaks = []
    for i in range(len(embeddings) - 1):
        similarity = np.dot(embeddings[i], embeddings[i + 1])
        if similarity < threshold:
            breaks.append(i + 1)  # Break after sentence i

    return breaks


def _merge_by_max_tokens(
    chunks: list[str], max_tokens: int, tokenizer: tiktoken.Encoding
) -> list[str]:
    """Merge consecutive chunks up to max_tokens."""
    merged = []
    current = ""
    current_tokens = 0

    for chunk in chunks:
        chunk_tokens = _count_tokens(chunk, tokenizer)
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


def _filter_by_min_tokens(
    chunks: list[str], min_tokens: int, tokenizer: tiktoken.Encoding
) -> list[str]:
    """Filter and merge chunks smaller than min_tokens."""
    result = []
    buffer = ""
    buffer_tokens = 0

    for chunk in chunks:
        chunk_tokens = _count_tokens(chunk, tokenizer)
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


def _split_oversized(
    chunks: list[str], max_tokens: int, tokenizer: tiktoken.Encoding
) -> list[str]:
    """Split chunks that exceed max_tokens by sentences."""
    final = []

    for chunk in chunks:
        chunk_tokens = _count_tokens(chunk, tokenizer)
        if chunk_tokens > max_tokens:
            # Split by sentences to preserve semantic boundaries
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            current = ""
            current_tokens = 0

            for sentence in sentences:
                sentence_tokens = _count_tokens(sentence, tokenizer)
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


def merge_small_chunks(
    chunks: list[str],
    tokenizer: tiktoken.Encoding,
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[str]:
    """
    Merge chunks that are too small, split those too large (token-aware).

    Args:
        chunks: List of chunk texts.
        tokenizer: Tokenizer for counting.
        min_tokens: Minimum tokens per chunk.
        max_tokens: Maximum tokens per chunk.

    Returns:
        Merged/split chunks.
    """
    merged = _merge_by_max_tokens(chunks, max_tokens, tokenizer)
    filtered = _filter_by_min_tokens(merged, min_tokens, tokenizer)
    return _split_oversized(filtered, max_tokens, tokenizer)


def _create_chunks_from_breaks(sentences: list[str], breaks: list[int]) -> list[str]:
    """Create chunk texts from sentence breaks."""
    chunks_text = []
    start = 0

    for break_idx in breaks:
        chunk = " ".join(sentences[start:break_idx])
        if chunk.strip():
            chunks_text.append(chunk.strip())
        start = break_idx

    if start < len(sentences):
        chunk = " ".join(sentences[start:])
        if chunk.strip():
            chunks_text.append(chunk.strip())

    return chunks_text if chunks_text else [" ".join(sentences)]


def _build_similarity_chunks(
    chunks_text: list[str], source: str, page: int, tokenizer: tiktoken.Encoding
) -> list[dict]:
    """Build final chunk dicts with metadata and token counts."""
    return [
        {
            "text": chunk_text,
            "source": source,
            "page": page,
            "chunk_index": str(i),
            "tokens": _count_tokens(chunk_text, tokenizer),
        }
        for i, chunk_text in enumerate(chunks_text)
    ]


def chunk_document_similarity(
    text: str,
    model: SentenceTransformer,
    source: str,
    page: int,
    tokenizer: tiktoken.Encoding,
    threshold: float = DEFAULT_THRESHOLD,
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[dict]:
    """
    Chunk document using semantic similarity (token-aware).

    Args:
        text: Document text.
        model: Sentence transformer model.
        source: Source filename.
        page: Page number.
        tokenizer: Tokenizer for counting.
        threshold: Similarity threshold.
        min_tokens: Minimum tokens per chunk.
        max_tokens: Maximum tokens per chunk.

    Returns:
        List of chunks with metadata.
    """
    text_tokens = _count_tokens(text, tokenizer)
    if not text or text_tokens < min_tokens:
        return []

    sentences = split_sentences(text, tokenizer)
    if not sentences:
        return [
            {
                "text": text.strip(),
                "source": source,
                "page": page,
                "chunk_index": "0",
                "tokens": text_tokens,
            }
        ]

    breaks = _get_similarity_breaks(sentences, model, threshold, source, page)
    chunks_text = _create_chunks_from_breaks(sentences, breaks)
    chunks_text = merge_small_chunks(chunks_text, tokenizer, min_tokens, max_tokens)
    return _build_similarity_chunks(chunks_text, source, page, tokenizer)


def _get_similarity_breaks(
    sentences: list[str],
    model: SentenceTransformer,
    threshold: float,
    source: str,
    page: int,
) -> list[int]:
    """Get similarity breaks with error handling."""
    try:
        return compute_similarity_breaks(sentences, model, threshold)
    except Exception as e:
        logger.warning(f"Similarity computation failed for {source} p{page}: {e}")
        return []


def process_corpus_similarity(
    input_dir: Path,
    output_file: Path,
    corpus: str = "fr",
    model_name: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """
    Process entire corpus with similarity-based chunking (token-aware).

    Args:
        input_dir: Directory with extraction JSON files.
        output_file: Output JSON file.
        corpus: Corpus code.
        model_name: Sentence transformer model.
        threshold: Similarity threshold.

    Returns:
        Processing report.
    """
    # Initialize tokenizer
    tokenizer = get_tokenizer()

    # Load model
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(
        f"Model loaded: {model_name} ({model.get_sentence_embedding_dimension()}D)"
    )

    # Find extraction files
    extraction_files = sorted(input_dir.glob("*.json"))
    logger.info(f"Found {len(extraction_files)} extraction files in {input_dir}")

    all_chunks = []
    total_pages = 0

    for ext_file in extraction_files:
        logger.info(f"Processing: {ext_file.name}")

        with open(ext_file, encoding="utf-8") as f:
            data = json.load(f)

        source = data.get("source", ext_file.stem + ".pdf")
        pages = data.get("pages", [])

        for page_data in pages:
            page_num = page_data.get("page_num", page_data.get("page", 0))
            text = page_data.get("text", "")

            if not text.strip():
                continue

            total_pages += 1

            # Token-aware similarity-based chunking
            page_chunks = chunk_document_similarity(
                text=text,
                model=model,
                source=source,
                page=page_num,
                tokenizer=tokenizer,
                threshold=threshold,
            )

            # Add IDs and metadata
            for chunk in page_chunks:
                chunk_id = f"{corpus}-{source}-p{page_num}-c{chunk['chunk_index']}"
                chunk["id"] = chunk_id
                chunk["metadata"] = {
                    "corpus": corpus,
                    "chunker": "similarity",
                    "model": model_name,
                    "threshold": threshold,
                    "tokenizer": TOKENIZER_NAME,
                    "min_tokens": MIN_CHUNK_TOKENS,
                    "max_tokens": MAX_CHUNK_TOKENS,
                }
                all_chunks.append(chunk)

    # Save
    output_data = {
        "corpus": corpus,
        "config": {
            "chunker": "similarity",
            "model": model_name,
            "threshold": threshold,
            "tokenizer": TOKENIZER_NAME,
            "min_chunk_tokens": MIN_CHUNK_TOKENS,
            "max_chunk_tokens": MAX_CHUNK_TOKENS,
        },
        "total_chunks": len(all_chunks),
        "total_pages": total_pages,
        "chunks": all_chunks,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(all_chunks)} similarity chunks to {output_file}")

    # Stats
    tokens = [c["tokens"] for c in all_chunks]
    report = {
        "corpus": corpus,
        "chunker": "similarity",
        "model": model_name,
        "threshold": threshold,
        "tokenizer": TOKENIZER_NAME,
        "total_chunks": len(all_chunks),
        "total_pages": total_pages,
        "avg_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0,
        "min_tokens": min(tokens) if tokens else 0,
        "max_tokens": max(tokens) if tokens else 0,
        "chunks_below_100": sum(1 for t in tokens if t < 100),
        "chunks_above_400": sum(1 for t in tokens if t >= 400),
    }

    return report


def main() -> None:
    """CLI for similarity chunking (token-aware)."""
    parser = argparse.ArgumentParser(
        description="Similarity Chunker - Pocket Arbiter (Sentence Transformers, token-aware)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input directory with extraction JSON files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output chunks JSON file",
    )
    parser.add_argument(
        "--corpus",
        "-c",
        choices=["fr", "intl"],
        default="fr",
        help="Corpus code (default: fr)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Sentence transformer model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Similarity threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    report = process_corpus_similarity(
        input_dir=args.input,
        output_file=args.output,
        corpus=args.corpus,
        model_name=args.model,
        threshold=args.threshold,
    )

    print("\n=== Similarity Chunking Report (Token-Aware) ===")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
