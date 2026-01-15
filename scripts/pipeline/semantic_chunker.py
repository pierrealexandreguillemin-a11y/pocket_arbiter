"""
Semantic Chunker - Pocket Arbiter

Utilise LangChain SemanticChunker pour chunking basé sur la similarité sémantique.
+70% accuracy sur documents réglementaires (source: LangCopilot 2025).

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
from typing import Any

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_THRESHOLD_TYPE = "percentile"
DEFAULT_THRESHOLD_AMOUNT = 90  # Higher = fewer breaks = larger chunks
MIN_CHUNK_SIZE = 100  # Minimum characters per chunk
MAX_CHUNK_SIZE = 2000  # Maximum characters per chunk


def create_semantic_chunker(
    model_name: str = DEFAULT_MODEL,
    threshold_type: str = DEFAULT_THRESHOLD_TYPE,
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


def chunk_document_semantic(
    text: str,
    chunker: SemanticChunker,
    source: str,
    page: int,
    min_size: int = MIN_CHUNK_SIZE,
    max_size: int = MAX_CHUNK_SIZE,
) -> list[dict]:
    """
    Chunk un document avec SemanticChunker.

    Args:
        text: Texte du document.
        chunker: SemanticChunker configuré.
        source: Nom du fichier source.
        page: Numéro de page.
        min_size: Taille minimum d'un chunk.
        max_size: Taille maximum d'un chunk.

    Returns:
        Liste de chunks avec métadonnées.
    """
    if not text or len(text.strip()) < min_size:
        return []

    try:
        # Split semantically
        raw_chunks = chunker.split_text(text)
    except Exception as e:
        logger.warning(f"SemanticChunker failed for {source} p{page}: {e}")
        # Fallback to simple split
        raw_chunks = [text[i : i + max_size] for i in range(0, len(text), max_size)]

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        # Skip too small
        if len(chunk_text.strip()) < min_size:
            continue

        # Split too large
        if len(chunk_text) > max_size:
            sub_chunks = [
                chunk_text[j : j + max_size]
                for j in range(0, len(chunk_text), max_size)
            ]
            for k, sub in enumerate(sub_chunks):
                if len(sub.strip()) >= min_size:
                    chunks.append(
                        {
                            "text": sub.strip(),
                            "source": source,
                            "page": page,
                            "chunk_index": f"{i}.{k}",
                        }
                    )
        else:
            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "source": source,
                    "page": page,
                    "chunk_index": str(i),
                }
            )

    return chunks


def process_corpus_semantic(
    input_dir: Path,
    output_file: Path,
    corpus: str = "fr",
    threshold_type: str = DEFAULT_THRESHOLD_TYPE,
    threshold_amount: float = DEFAULT_THRESHOLD_AMOUNT,
) -> dict[str, Any]:
    """
    Traite un corpus complet avec SemanticChunker.

    Args:
        input_dir: Répertoire des extractions JSON.
        output_file: Fichier JSON de sortie.
        corpus: Code corpus (fr, intl).
        threshold_type: Type de seuil.
        threshold_amount: Valeur du seuil.

    Returns:
        Rapport de traitement.
    """
    import tiktoken

    # Initialize
    chunker = create_semantic_chunker(
        threshold_type=threshold_type,
        threshold_amount=threshold_amount,
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

            # Semantic chunking
            page_chunks = chunk_document_semantic(
                text=text,
                chunker=chunker,
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
                    "chunker": "semantic",
                    "threshold": f"{threshold_type}:{threshold_amount}",
                }
                all_chunks.append(chunk)

    # Save
    output_data = {
        "corpus": corpus,
        "chunker": "semantic",
        "threshold_type": threshold_type,
        "threshold_amount": threshold_amount,
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
        "total_chunks": len(all_chunks),
        "total_pages": total_pages,
        "avg_tokens": sum(tokens) / len(tokens) if tokens else 0,
        "min_tokens": min(tokens) if tokens else 0,
        "max_tokens": max(tokens) if tokens else 0,
    }

    return report


def main():
    """CLI pour semantic chunking."""
    parser = argparse.ArgumentParser(
        description="Semantic Chunker - Pocket Arbiter (LangChain)",
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

    print("\n=== Semantic Chunking Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
