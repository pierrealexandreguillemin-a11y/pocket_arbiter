"""
Sentence Chunker - Pocket Arbiter (LlamaIndex)

Utilise LlamaIndex SentenceSplitter pour chunking basé sur les phrases.
Meilleure préservation du contexte sémantique pour documents réglementaires.

IMPORTANT: Utilise des TOKENS (via tiktoken), pas des caractères.
           512 tokens ≈ 2048 caractères (ratio moyen 1:4)

ISO Reference:
    - ISO/IEC 25010 S4.2 - Performance efficiency (Recall >= 80%)
    - ISO/IEC 42001 - AI traceability

Usage:
    python sentence_chunker.py --input corpus/processed/raw_fr --output corpus/processed/chunks_sentence_fr.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable

import tiktoken
from llama_index.core.node_parser import SentenceSplitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Constants (TOKENS, not characters) ---

DEFAULT_CHUNK_SIZE_TOKENS = 512  # Target: 512 tokens (ISO 25010 optimal)
DEFAULT_CHUNK_OVERLAP_TOKENS = 128  # 25% overlap (research: 20-25% optimal)
MIN_CHUNK_TOKENS = 50  # Minimum tokens per chunk
TOKENIZER_NAME = "cl100k_base"  # Compatible OpenAI/LLM (same as chunker.py)
DEFAULT_SEPARATOR = " "
DEFAULT_PARAGRAPH_SEPARATOR = "\n\n"


def get_tokenizer() -> tiktoken.Encoding:
    """Get tiktoken tokenizer for token counting."""
    return tiktoken.get_encoding(TOKENIZER_NAME)


def create_token_counter(tokenizer: tiktoken.Encoding) -> Callable[[str], int]:
    """
    Create a token counter function for LlamaIndex SentenceSplitter.

    Args:
        tokenizer: tiktoken Encoding instance.

    Returns:
        Callable that counts tokens in a string.
    """
    return lambda text: len(tokenizer.encode(text))


def create_sentence_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    separator: str = DEFAULT_SEPARATOR,
    paragraph_separator: str = DEFAULT_PARAGRAPH_SEPARATOR,
    tokenizer: tiktoken.Encoding | None = None,
) -> SentenceSplitter:
    """
    Crée un SentenceSplitter LlamaIndex configuré avec comptage de TOKENS.

    Args:
        chunk_size: Taille cible des chunks en TOKENS.
        chunk_overlap: Chevauchement entre chunks en TOKENS.
        separator: Séparateur de mots.
        paragraph_separator: Séparateur de paragraphes.
        tokenizer: Tokenizer tiktoken (optionnel, créé si non fourni).

    Returns:
        SentenceSplitter configuré avec tokenizer.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    token_counter = create_token_counter(tokenizer)

    logger.info(
        f"Creating SentenceSplitter (size={chunk_size} tokens, "
        f"overlap={chunk_overlap} tokens, tokenizer={TOKENIZER_NAME})"
    )

    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator,
        paragraph_separator=paragraph_separator,
        secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",  # Sentence boundaries
        tokenizer=token_counter,  # Use token counting instead of character counting
    )


def chunk_document_sentence(
    text: str,
    splitter: SentenceSplitter,
    source: str,
    page: int,
    tokenizer: tiktoken.Encoding,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> list[dict]:
    """
    Chunk un document avec SentenceSplitter.

    Args:
        text: Texte du document.
        splitter: SentenceSplitter configuré.
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
        # Split by sentences with token-aware chunking
        raw_chunks = splitter.split_text(text)
    except Exception as e:
        logger.warning(f"SentenceSplitter failed for {source} p{page}: {e}")
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
    splitter: SentenceSplitter,
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
                "chunker": "sentence",
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
            "chunker": "sentence",
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
        "chunker": "sentence",
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
