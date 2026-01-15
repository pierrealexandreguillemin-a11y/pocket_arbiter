"""
Sentence Chunker - Pocket Arbiter (LlamaIndex)

Utilise LlamaIndex SentenceSplitter pour chunking basé sur les phrases.
Meilleure préservation du contexte sémantique pour documents réglementaires.

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
from typing import Any

from llama_index.core.node_parser import SentenceSplitter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---

DEFAULT_CHUNK_SIZE = 512  # Characters
DEFAULT_CHUNK_OVERLAP = 128  # 25% overlap
DEFAULT_SEPARATOR = " "
DEFAULT_PARAGRAPH_SEPARATOR = "\n\n"


def create_sentence_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separator: str = DEFAULT_SEPARATOR,
    paragraph_separator: str = DEFAULT_PARAGRAPH_SEPARATOR,
) -> SentenceSplitter:
    """
    Crée un SentenceSplitter LlamaIndex configuré.

    Args:
        chunk_size: Taille cible des chunks en caractères.
        chunk_overlap: Chevauchement entre chunks.
        separator: Séparateur de mots.
        paragraph_separator: Séparateur de paragraphes.

    Returns:
        SentenceSplitter configuré.
    """
    logger.info(
        f"Creating SentenceSplitter (size={chunk_size}, overlap={chunk_overlap})"
    )
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator,
        paragraph_separator=paragraph_separator,
        secondary_chunking_regex="[^,.;。？！]+[,.;。？！]?",  # Sentence boundaries
    )


def chunk_document_sentence(
    text: str,
    splitter: SentenceSplitter,
    source: str,
    page: int,
    min_size: int = 50,
) -> list[dict]:
    """
    Chunk un document avec SentenceSplitter.

    Args:
        text: Texte du document.
        splitter: SentenceSplitter configuré.
        source: Nom du fichier source.
        page: Numéro de page.
        min_size: Taille minimum d'un chunk.

    Returns:
        Liste de chunks avec métadonnées.
    """
    if not text or len(text.strip()) < min_size:
        return []

    try:
        # Split by sentences
        raw_chunks = splitter.split_text(text)
    except Exception as e:
        logger.warning(f"SentenceSplitter failed for {source} p{page}: {e}")
        # Fallback to simple split
        raw_chunks = [
            text[i : i + DEFAULT_CHUNK_SIZE]
            for i in range(0, len(text), DEFAULT_CHUNK_SIZE)
        ]

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        # Skip too small
        if len(chunk_text.strip()) < min_size:
            continue

        chunks.append(
            {
                "text": chunk_text.strip(),
                "source": source,
                "page": page,
                "chunk_index": str(i),
            }
        )

    return chunks


def process_corpus_sentence(
    input_dir: Path,
    output_file: Path,
    corpus: str = "fr",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    """
    Traite un corpus complet avec SentenceSplitter.

    Args:
        input_dir: Répertoire des extractions JSON.
        output_file: Fichier JSON de sortie.
        corpus: Code corpus (fr, intl).
        chunk_size: Taille des chunks.
        chunk_overlap: Chevauchement.

    Returns:
        Rapport de traitement.
    """
    import tiktoken

    # Initialize
    splitter = create_sentence_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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

            # Sentence-based chunking
            page_chunks = chunk_document_sentence(
                text=text,
                splitter=splitter,
                source=source,
                page=page_num,
            )

            # Add IDs and token counts
            for chunk in page_chunks:
                chunk_id = f"{corpus}-{source}-p{page_num}-c{chunk['chunk_index']}"
                chunk["id"] = chunk_id
                chunk["tokens"] = len(tokenizer.encode(chunk["text"]))
                chunk["metadata"] = {
                    "corpus": corpus,
                    "chunker": "sentence",
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
                all_chunks.append(chunk)

    # Save
    output_data = {
        "corpus": corpus,
        "chunker": "sentence",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
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
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "overlap_pct": round(chunk_overlap / chunk_size * 100, 1),
        "total_chunks": len(all_chunks),
        "total_pages": total_pages,
        "avg_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0,
        "min_tokens": min(tokens) if tokens else 0,
        "max_tokens": max(tokens) if tokens else 0,
    }

    return report


def main():
    """CLI pour sentence chunking."""
    parser = argparse.ArgumentParser(
        description="Sentence Chunker - Pocket Arbiter (LlamaIndex)",
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
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in characters (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap in characters (default: {DEFAULT_CHUNK_OVERLAP})",
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

    print("\n=== Sentence Chunking Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
