"""
Semantic Chunker - Pocket Arbiter

Utilise LangChain SemanticChunker pour chunking basé sur la similarité sémantique.
+70% accuracy sur documents réglementaires (source: LangCopilot 2025).

IMPORTANT: Utilise des TOKENS (via tiktoken), pas des caractères.
           512 tokens ≈ 2048 caractères (ratio moyen 1:4)

ISO Reference:
    - ISO/IEC 25010 S4.2 - Performance efficiency (Recall >= 80%)
    - ISO/IEC 42001 - AI traceability

Usage:
    python semantic_chunker.py --input corpus/processed/raw_fr --output corpus/processed/chunks_semantic_fr.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Literal

import tiktoken
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from scripts.pipeline.token_utils import (
    TOKENIZER_NAME,
    count_tokens,
    get_tokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Constants (TOKENS, not characters) ---

DEFAULT_MODEL = "intfloat/multilingual-e5-base"  # Best for French semantic similarity
DEFAULT_THRESHOLD_TYPE = "percentile"
DEFAULT_THRESHOLD_AMOUNT = 90  # Higher = fewer breaks = larger chunks
MIN_CHUNK_TOKENS = 50  # Minimum tokens per chunk
MAX_CHUNK_TOKENS = (
    1024  # Maximum tokens per chunk (allow larger for semantic coherence)
)


BreakpointType = Literal["percentile", "standard_deviation", "interquartile", "gradient"]


def _count_tokens(text: str, tokenizer: tiktoken.Encoding) -> int:
    """Count tokens in text using shared utility."""
    return count_tokens(text, tokenizer)


def create_semantic_chunker(
    model_name: str = DEFAULT_MODEL,
    threshold_type: BreakpointType = DEFAULT_THRESHOLD_TYPE,  # type: ignore[assignment]
    threshold_amount: float = DEFAULT_THRESHOLD_AMOUNT,
) -> SemanticChunker:
    """
    Crée un SemanticChunker LangChain configuré.

    Args:
        model_name: Modèle d'embeddings HuggingFace.
        threshold_type: 'percentile', 'standard_deviation', 'interquartile'.
        threshold_amount: Seuil de coupure (plus haut = moins de coupures).

    Returns:
        SemanticChunker configuré.
    """
    logger.info(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )

    logger.info(
        f"Creating SemanticChunker (threshold={threshold_type}:{threshold_amount})"
    )
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=threshold_type,
        breakpoint_threshold_amount=threshold_amount,
    )


def _build_chunk_metadata(
    text: str, source: str, page: int, index: str, tokens: int
) -> dict:
    """Build chunk dict with metadata."""
    return {
        "text": text.strip(),
        "source": source,
        "page": page,
        "chunk_index": index,
        "tokens": tokens,
    }


def _split_large_chunk(
    chunk_text: str,
    chunk_idx: int,
    source: str,
    page: int,
    min_tokens: int,
    max_tokens: int,
    tokenizer: tiktoken.Encoding,
) -> list[dict]:
    """Split a chunk that exceeds max_tokens into sub-chunks."""
    # Split by sentences first to preserve semantic boundaries
    import re

    sentences = re.split(r"(?<=[.!?])\s+", chunk_text)

    sub_chunks = []
    current_chunk = ""
    current_tokens = 0
    sub_idx = 0

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence, tokenizer)

        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk = (
                (current_chunk + " " + sentence).strip() if current_chunk else sentence
            )
            current_tokens += sentence_tokens
        else:
            # Save current chunk if substantial
            if current_tokens >= min_tokens:
                sub_chunks.append(
                    _build_chunk_metadata(
                        current_chunk,
                        source,
                        page,
                        f"{chunk_idx}.{sub_idx}",
                        current_tokens,
                    )
                )
                sub_idx += 1
            current_chunk = sentence
            current_tokens = sentence_tokens

    # Don't forget the last chunk
    if current_chunk and current_tokens >= min_tokens:
        sub_chunks.append(
            _build_chunk_metadata(
                current_chunk, source, page, f"{chunk_idx}.{sub_idx}", current_tokens
            )
        )

    return sub_chunks


def chunk_document_semantic(
    text: str,
    chunker: SemanticChunker,
    source: str,
    page: int,
    tokenizer: tiktoken.Encoding,
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[dict]:
    """
    Chunk un document avec SemanticChunker (token-aware).

    Args:
        text: Texte du document.
        chunker: SemanticChunker configuré.
        source: Nom du fichier source.
        page: Numéro de page.
        tokenizer: Tokenizer pour comptage.
        min_tokens: Nombre minimum de tokens par chunk.
        max_tokens: Nombre maximum de tokens par chunk.

    Returns:
        Liste de chunks avec métadonnées.
    """
    text_tokens = _count_tokens(text, tokenizer)
    if not text or text_tokens < min_tokens:
        return []

    raw_chunks = _get_raw_chunks(text, chunker, source, page, tokenizer, max_tokens)
    return _process_raw_chunks(
        raw_chunks, source, page, min_tokens, max_tokens, tokenizer
    )


def _get_raw_chunks(
    text: str,
    chunker: SemanticChunker,
    source: str,
    page: int,
    tokenizer: tiktoken.Encoding,
    max_tokens: int,
) -> list[str]:
    """Get raw chunks from chunker with fallback."""
    try:
        return chunker.split_text(text)
    except Exception as e:
        logger.warning(f"SemanticChunker failed for {source} p{page}: {e}")
        # Fallback: split by estimated character count (token * 4)
        max_chars = max_tokens * 4
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _process_raw_chunks(
    raw_chunks: list[str],
    source: str,
    page: int,
    min_tokens: int,
    max_tokens: int,
    tokenizer: tiktoken.Encoding,
) -> list[dict]:
    """Process raw chunks into final chunk list with token-aware filtering."""
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        token_count = _count_tokens(chunk_text, tokenizer)

        if token_count < min_tokens:
            continue

        if token_count > max_tokens:
            chunks.extend(
                _split_large_chunk(
                    chunk_text, i, source, page, min_tokens, max_tokens, tokenizer
                )
            )
        else:
            chunks.append(
                _build_chunk_metadata(chunk_text, source, page, str(i), token_count)
            )
    return chunks


def _process_extraction_file_semantic(
    ext_file: Path,
    chunker: SemanticChunker,
    tokenizer: tiktoken.Encoding,
    corpus: str,
    threshold_type: str,
    threshold_amount: float,
) -> tuple[list[dict], int]:
    """Process a single extraction file and return chunks with page count."""
    with open(ext_file, encoding="utf-8") as f:
        data = json.load(f)

    source = data.get("source", ext_file.stem + ".pdf")
    pages = data.get("pages", [])
    chunks = []
    pages_processed = 0

    for page_data in pages:
        page_num = page_data.get("page_num", page_data.get("page", 0))
        text = page_data.get("text", "")

        if not text.strip():
            continue

        pages_processed += 1
        page_chunks = chunk_document_semantic(
            text=text,
            chunker=chunker,
            source=source,
            page=page_num,
            tokenizer=tokenizer,
        )

        for chunk in page_chunks:
            chunk["id"] = f"{corpus}-{source}-p{page_num}-c{chunk['chunk_index']}"
            chunk["metadata"] = {
                "corpus": corpus,
                "chunker": "semantic",
                "threshold": f"{threshold_type}:{threshold_amount}",
                "tokenizer": TOKENIZER_NAME,
                "min_tokens": MIN_CHUNK_TOKENS,
                "max_tokens": MAX_CHUNK_TOKENS,
            }
            chunks.append(chunk)

    return chunks, pages_processed


def process_corpus_semantic(
    input_dir: Path,
    output_file: Path,
    corpus: str = "fr",
    threshold_type: BreakpointType = DEFAULT_THRESHOLD_TYPE,  # type: ignore[assignment]
    threshold_amount: float = DEFAULT_THRESHOLD_AMOUNT,
) -> dict[str, Any]:
    """
    Traite un corpus complet avec SemanticChunker (token-aware).

    Args:
        input_dir: Répertoire des extractions JSON.
        output_file: Fichier JSON de sortie.
        corpus: Code corpus (fr, intl).
        threshold_type: Type de seuil.
        threshold_amount: Valeur du seuil.

    Returns:
        Rapport de traitement.
    """
    tokenizer = get_tokenizer()
    chunker = create_semantic_chunker(
        threshold_type=threshold_type, threshold_amount=threshold_amount
    )

    extraction_files = sorted(input_dir.glob("*.json"))
    logger.info(f"Found {len(extraction_files)} extraction files in {input_dir}")

    all_chunks = []
    total_pages = 0

    for ext_file in extraction_files:
        logger.info(f"Processing: {ext_file.name}")
        chunks, pages = _process_extraction_file_semantic(
            ext_file, chunker, tokenizer, corpus, threshold_type, threshold_amount
        )
        all_chunks.extend(chunks)
        total_pages += pages

    # Save
    output_data = {
        "corpus": corpus,
        "config": {
            "chunker": "semantic",
            "threshold_type": threshold_type,
            "threshold_amount": threshold_amount,
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

    logger.info(f"Saved {len(all_chunks)} semantic chunks to {output_file}")

    # Stats
    tokens = [c["tokens"] for c in all_chunks]
    report = {
        "corpus": corpus,
        "chunker": "semantic",
        "threshold_type": threshold_type,
        "threshold_amount": threshold_amount,
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
    """CLI pour semantic chunking (token-aware)."""
    parser = argparse.ArgumentParser(
        description="Semantic Chunker - Pocket Arbiter (LangChain, token-aware)",
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
        "--threshold-type",
        choices=["percentile", "standard_deviation", "interquartile"],
        default=DEFAULT_THRESHOLD_TYPE,
        help=f"Threshold type (default: {DEFAULT_THRESHOLD_TYPE})",
    )
    parser.add_argument(
        "--threshold-amount",
        type=float,
        default=DEFAULT_THRESHOLD_AMOUNT,
        help=f"Threshold amount (default: {DEFAULT_THRESHOLD_AMOUNT})",
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

    report = process_corpus_semantic(
        input_dir=args.input,
        output_file=args.output,
        corpus=args.corpus,
        threshold_type=args.threshold_type,
        threshold_amount=args.threshold_amount,
    )

    print("\n=== Semantic Chunking Report (Token-Aware) ===")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
