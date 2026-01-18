"""
Recursive Chunker - Pocket Arbiter (LangChain RecursiveCharacterTextSplitter)

Optimisé pour documents réglementaires structurés (articles, sections, listes).
Respecte la hiérarchie: sections > paragraphes > phrases > mots.

Config: 450 tokens, 100 overlap (22%) - Best practice 2025-2026 pour RAG normatif.

ISO Reference:
    - ISO/IEC 25010 S4.2 - Performance efficiency (Recall >= 80%)
    - ISO/IEC 42001 - AI traceability (chunking optimisé règlements)
    - ISO/IEC 12207 S7.3.3 - Implementation (RecursiveCharacterTextSplitter)

Changelog:
    - 2026-01-18: Switch SentenceSplitter → RecursiveCharacterTextSplitter
                  450 tokens / 100 overlap (était 512/128)

Usage:
    python sentence_chunker.py --input corpus/processed/raw_fr --output corpus/processed/chunks_recursive_fr.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from scripts.pipeline.token_utils import (
    TOKENIZER_NAME,
    count_tokens,
    get_tokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Constants (TOKENS, not characters) ---

DEFAULT_CHUNK_SIZE_TOKENS = 450  # Sweet-spot pour embeddings (2025-2026)
DEFAULT_CHUNK_OVERLAP_TOKENS = 100  # ~22% overlap (recommandé: 20-25%)
MIN_CHUNK_TOKENS = 50  # Minimum tokens per chunk

# Séparateurs hiérarchiques pour règlements (sections > paragraphes > phrases)
REGULATORY_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""]


def create_token_counter(tokenizer: tiktoken.Encoding) -> Callable[[str], int]:
    """
    Create a token counter function for LlamaIndex SentenceSplitter.

    Args:
        tokenizer: tiktoken Encoding instance.

    Returns:
        Callable that counts tokens in a string.
    """
    return lambda text: count_tokens(text, tokenizer)


def create_sentence_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    tokenizer: tiktoken.Encoding | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Crée un RecursiveCharacterTextSplitter avec comptage de TOKENS.

    Args:
        chunk_size: Taille cible des chunks en TOKENS.
        chunk_overlap: Chevauchement entre chunks en TOKENS.
        tokenizer: Tokenizer tiktoken (optionnel, créé si non fourni).

    Returns:
        RecursiveCharacterTextSplitter configuré.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    token_counter = create_token_counter(tokenizer)

    logger.info(
        f"Creating RecursiveCharacterTextSplitter (size={chunk_size} tokens, "
        f"overlap={chunk_overlap} tokens, tokenizer={TOKENIZER_NAME})"
    )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=REGULATORY_SEPARATORS,
        length_function=token_counter,
        keep_separator=True,
    )


def chunk_document_sentence(
    text: str,
    splitter: RecursiveCharacterTextSplitter,
    source: str,
    page: int,
    tokenizer: tiktoken.Encoding,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> list[dict]:
    """
    Chunk un document avec RecursiveCharacterTextSplitter.

    Args:
        text: Texte du document.
        splitter: RecursiveCharacterTextSplitter configuré.
        source: Nom du fichier source.
        page: Numéro de page.
        tokenizer: Tokenizer pour comptage.
        min_tokens: Nombre minimum de tokens par chunk.

    Returns:
        Liste de chunks avec métadonnées.
    """
    if not text or len(text.strip()) < 20:  # Minimum 20 chars
        return []

    try:
        # Split avec hiérarchie séparateurs (sections > paragraphes > phrases)
        raw_chunks = splitter.split_text(text)
    except Exception as e:
        logger.warning(f"RecursiveSplitter failed for {source} p{page}: {e}")
        # Fallback: no chunking, return whole text if substantial
        token_count = len(tokenizer.encode(text))
        if token_count >= min_tokens:
            return [
                {
                    "text": text.strip(),
                    "source": source,
                    "page": page,
                    "chunk_index": "0",
                }
            ]
        return []

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        # Check token count (not character count)
        token_count = len(tokenizer.encode(chunk_text))
        if token_count < min_tokens:
            continue

        chunks.append(
            {
                "text": chunk_text,
                "source": source,
                "page": page,
                "chunk_index": str(i),
                "tokens": token_count,
            }
        )

    return chunks


def _process_extraction_file_sentence(
    ext_file: Path,
    splitter: RecursiveCharacterTextSplitter,
    tokenizer: tiktoken.Encoding,
    corpus: str,
    chunk_size: int,
    chunk_overlap: int,
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
        page_chunks = chunk_document_sentence(
            text=text,
            splitter=splitter,
            source=source,
            page=page_num,
            tokenizer=tokenizer,
        )

        for chunk in page_chunks:
            chunk["id"] = f"{corpus}-{source}-p{page_num}-c{chunk['chunk_index']}"
            chunk["metadata"] = {
                "corpus": corpus,
                "chunker": "recursive",
                "chunk_size_tokens": chunk_size,
                "chunk_overlap_tokens": chunk_overlap,
                "tokenizer": TOKENIZER_NAME,
            }
            chunks.append(chunk)

    return chunks, pages_processed


def process_corpus_sentence(
    input_dir: Path,
    output_file: Path,
    corpus: str = "fr",
    chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> dict[str, Any]:
    """
    Traite un corpus complet avec SentenceSplitter (token-aware).

    Args:
        input_dir: Répertoire des extractions JSON.
        output_file: Fichier JSON de sortie.
        corpus: Code corpus (fr, intl).
        chunk_size: Taille des chunks en TOKENS.
        chunk_overlap: Chevauchement en TOKENS.

    Returns:
        Rapport de traitement.
    """
    tokenizer = get_tokenizer()
    splitter = create_sentence_splitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=tokenizer
    )

    extraction_files = sorted(input_dir.glob("*.json"))
    logger.info(f"Found {len(extraction_files)} extraction files in {input_dir}")

    all_chunks = []
    total_pages = 0

    for ext_file in extraction_files:
        logger.info(f"Processing: {ext_file.name}")
        chunks, pages = _process_extraction_file_sentence(
            ext_file, splitter, tokenizer, corpus, chunk_size, chunk_overlap
        )
        all_chunks.extend(chunks)
        total_pages += pages

    # Save
    output_data = {
        "corpus": corpus,
        "config": {
            "chunker": "recursive",
            "chunk_size_tokens": chunk_size,
            "chunk_overlap_tokens": chunk_overlap,
            "tokenizer": TOKENIZER_NAME,
            "min_chunk_tokens": MIN_CHUNK_TOKENS,
        },
        "total_chunks": len(all_chunks),
        "total_pages": total_pages,
        "chunks": all_chunks,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(all_chunks)} sentence chunks to {output_file}")

    # Stats
    tokens = [c["tokens"] for c in all_chunks]
    report = {
        "corpus": corpus,
        "chunker": "recursive",
        "chunk_size_tokens": chunk_size,
        "chunk_overlap_tokens": chunk_overlap,
        "overlap_pct": round(chunk_overlap / chunk_size * 100, 1),
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
    """CLI pour sentence chunking (token-aware)."""
    parser = argparse.ArgumentParser(
        description="Sentence Chunker - Pocket Arbiter (LlamaIndex, token-aware)",
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
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE_TOKENS,
        help=f"Chunk size in TOKENS (default: {DEFAULT_CHUNK_SIZE_TOKENS})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_TOKENS,
        help=f"Chunk overlap in TOKENS (default: {DEFAULT_CHUNK_OVERLAP_TOKENS})",
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

    report = process_corpus_sentence(
        input_dir=args.input,
        output_file=args.output,
        corpus=args.corpus,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print("\n=== Sentence Chunking Report (Token-Aware) ===")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
