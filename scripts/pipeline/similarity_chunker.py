"""
Similarity Chunker - Pocket Arbiter (Sentence Transformers)

Utilise Sentence Transformers pour chunking basé sur la similarité cosinus.
Détecte les points de rupture sémantique via embeddings directs.

ISO Reference:
    - ISO/IEC 25010 S4.2 - Performance efficiency (Recall >= 80%)
    - ISO/IEC 42001 - AI traceability

Usage:
    python similarity_chunker.py --input corpus/processed/raw_fr --output corpus/processed/chunks_similarity_fr.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_THRESHOLD = 0.5  # Cosine similarity threshold for breaks
DEFAULT_SENTENCE_MIN_LENGTH = 20  # Minimum chars per sentence
DEFAULT_CHUNK_MIN_LENGTH = 100  # Minimum chars per chunk
DEFAULT_CHUNK_MAX_LENGTH = 2000  # Maximum chars per chunk


def split_sentences(
    text: str, min_length: int = DEFAULT_SENTENCE_MIN_LENGTH
) -> list[str]:
    """
    Split text into sentences using common delimiters.

    Args:
        text: Input text.
        min_length: Minimum sentence length.

    Returns:
        List of sentences.
    """
    import re

    # Split on sentence boundaries
    pattern = r"(?<=[.!?])\s+(?=[A-ZÀ-Ü])|(?<=\n)\s*(?=\S)"
    raw_sentences = re.split(pattern, text)

    # Filter and clean
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) >= min_length:
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


def merge_small_chunks(
    chunks: list[str],
    min_length: int = DEFAULT_CHUNK_MIN_LENGTH,
    max_length: int = DEFAULT_CHUNK_MAX_LENGTH,
) -> list[str]:
    """
    Merge chunks that are too small, split those too large.

    Args:
        chunks: List of chunk texts.
        min_length: Minimum chunk length.
        max_length: Maximum chunk length.

    Returns:
        Merged/split chunks.
    """
    merged = []
    current = ""

    for chunk in chunks:
        if len(current) + len(chunk) <= max_length:
            current = (current + " " + chunk).strip() if current else chunk
        else:
            if current:
                merged.append(current)
            current = chunk

    if current:
        merged.append(current)

    # Filter by minimum length
    result = []
    buffer = ""
    for chunk in merged:
        if len(chunk) < min_length:
            buffer = (buffer + " " + chunk).strip() if buffer else chunk
        else:
            if buffer:
                chunk = (buffer + " " + chunk).strip()
                buffer = ""
            result.append(chunk)

    if buffer and result:
        result[-1] = (result[-1] + " " + buffer).strip()
    elif buffer:
        result.append(buffer)

    # Split chunks that are too large
    final = []
    for chunk in result:
        if len(chunk) > max_length:
            # Split at max_length boundaries
            for i in range(0, len(chunk), max_length):
                sub = chunk[i : i + max_length].strip()
                if sub:
                    final.append(sub)
        else:
            final.append(chunk)

    return final


def chunk_document_similarity(
    text: str,
    model: SentenceTransformer,
    source: str,
    page: int,
    threshold: float = DEFAULT_THRESHOLD,
    min_length: int = DEFAULT_CHUNK_MIN_LENGTH,
    max_length: int = DEFAULT_CHUNK_MAX_LENGTH,
) -> list[dict]:
    """
    Chunk document using semantic similarity.

    Args:
        text: Document text.
        model: Sentence transformer model.
        source: Source filename.
        page: Page number.
        threshold: Similarity threshold.
        min_length: Minimum chunk length.
        max_length: Maximum chunk length.

    Returns:
        List of chunks with metadata.
    """
    if not text or len(text.strip()) < min_length:
        return []

    # Split into sentences
    sentences = split_sentences(text)
    if not sentences:
        return [
            {"text": text.strip(), "source": source, "page": page, "chunk_index": "0"}
        ]

    # Find semantic breaks
    try:
        breaks = compute_similarity_breaks(sentences, model, threshold)
    except Exception as e:
        logger.warning(f"Similarity computation failed for {source} p{page}: {e}")
        breaks = []

    # Create chunks from breaks
    chunks_text = []
    start = 0
    for break_idx in breaks:
        chunk = " ".join(sentences[start:break_idx])
        if chunk.strip():
            chunks_text.append(chunk.strip())
        start = break_idx

    # Add last chunk
    if start < len(sentences):
        chunk = " ".join(sentences[start:])
        if chunk.strip():
            chunks_text.append(chunk.strip())

    # If no breaks found, use all sentences as one chunk
    if not chunks_text:
        chunks_text = [" ".join(sentences)]

    # Merge/split to respect size constraints
    chunks_text = merge_small_chunks(chunks_text, min_length, max_length)

    # Build result
    chunks = []
    for i, chunk_text in enumerate(chunks_text):
        chunks.append(
            {
                "text": chunk_text,
                "source": source,
                "page": page,
                "chunk_index": str(i),
            }
        )

    return chunks


def process_corpus_similarity(
    input_dir: Path,
    output_file: Path,
    corpus: str = "fr",
    model_name: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """
    Process entire corpus with similarity-based chunking.

    Args:
        input_dir: Directory with extraction JSON files.
        output_file: Output JSON file.
        corpus: Corpus code.
        model_name: Sentence transformer model.
        threshold: Similarity threshold.

    Returns:
        Processing report.
    """
    import tiktoken

    # Load model
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(
        f"Model loaded: {model_name} ({model.get_sentence_embedding_dimension()}D)"
    )

    tokenizer = tiktoken.get_encoding("cl100k_base")

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

            # Similarity-based chunking
            page_chunks = chunk_document_similarity(
                text=text,
                model=model,
                source=source,
                page=page_num,
                threshold=threshold,
            )

            # Add IDs and token counts
            for chunk in page_chunks:
                chunk_id = f"{corpus}-{source}-p{page_num}-c{chunk['chunk_index']}"
                chunk["id"] = chunk_id
                chunk["tokens"] = len(tokenizer.encode(chunk["text"]))
                chunk["metadata"] = {
                    "corpus": corpus,
                    "chunker": "similarity",
                    "model": model_name,
                    "threshold": threshold,
                }
                all_chunks.append(chunk)

    # Save
    output_data = {
        "corpus": corpus,
        "chunker": "similarity",
        "model": model_name,
        "threshold": threshold,
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
        "total_chunks": len(all_chunks),
        "total_pages": total_pages,
        "avg_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0,
        "min_tokens": min(tokens) if tokens else 0,
        "max_tokens": max(tokens) if tokens else 0,
    }

    return report


def main():
    """CLI for similarity chunking."""
    parser = argparse.ArgumentParser(
        description="Similarity Chunker - Pocket Arbiter (Sentence Transformers)",
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

    print("\n=== Similarity Chunking Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
