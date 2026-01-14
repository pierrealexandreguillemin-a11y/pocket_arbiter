"""
Segmentation en chunks - Pocket Arbiter

Ce module segmente le texte extrait en chunks de taille optimale
pour le retrieval RAG (256 tokens avec overlap de 50).

ISO Reference:
    - ISO/IEC 12207 §7.3.3 - Implementation
    - ISO 82045 - Document metadata

Dependencies:
    - tiktoken >= 0.5.0

Usage:
    python chunker.py --input corpus/processed/raw_fr --output corpus/processed/chunks_fr.json
    python chunker.py --input corpus/processed/raw_intl --output corpus/processed/chunks_intl.json

Example:
    >>> from scripts.pipeline.chunker import chunk_text
    >>> chunks = chunk_text("Long text...", max_tokens=256, overlap_tokens=50)
    >>> print(len(chunks))
    5
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import tiktoken

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --- Constants ---

DEFAULT_MAX_TOKENS = 256
DEFAULT_OVERLAP_TOKENS = 50
TOKENIZER_NAME = "cl100k_base"  # Compatible OpenAI/LLM


# --- Main Functions ---


def _build_chunk_dict(
    chunk_text: str, chunk_tokens: int, metadata: Optional[dict]
) -> dict:
    """Build a chunk dictionary from text and metadata."""
    from scripts.pipeline.utils import get_date

    return {
        "id": "",  # Will be set by caller
        "text": chunk_text,
        "source": metadata.get("source", "") if metadata else "",
        "page": metadata.get("page", 0) if metadata else 0,
        "tokens": chunk_tokens,
        "metadata": {
            "section": metadata.get("section") if metadata else None,
            "corpus": metadata.get("corpus", "fr") if metadata else "fr",
            "extraction_date": get_date(),
            "version": "1.0",
        },
    }


def _enforce_iso_limits(
    chunk_text: str, remaining: str, encoder: "tiktoken.Encoding"
) -> tuple[str, str, int]:
    """Enforce ISO 82045 max 512 tokens limit."""
    chunk_tokens = len(encoder.encode(chunk_text))
    if chunk_tokens > 512:
        hard_tokens = encoder.encode(chunk_text)[:512]
        overflow = encoder.decode(encoder.encode(chunk_text)[512:])
        chunk_text = encoder.decode(hard_tokens)
        remaining = overflow + " " + remaining if remaining else overflow
        chunk_tokens = len(encoder.encode(chunk_text))
    return chunk_text, remaining, chunk_tokens


def chunk_text(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """
    Segmente le texte en chunks avec overlap.

    Cette fonction divise un texte en segments de taille maximale
    definie en tokens, avec chevauchement entre les segments pour
    preserver le contexte aux frontieres.

    Args:
        text: Texte brut a segmenter.
        max_tokens: Taille maximale par chunk en tokens (default 256).
        overlap_tokens: Chevauchement entre chunks en tokens (default 50).
        metadata: Metadonnees a propager vers chaque chunk:
            - source (str): Fichier PDF source
            - page (int): Numero de page
            - corpus (str): fr ou intl

    Returns:
        Liste de chunks conformes au schema CHUNK_SCHEMA.md.
        Chaque chunk contient:
            - id (str): Identifiant unique (format: {corpus}-{doc}-{page}-{seq})
            - text (str): Contenu textuel
            - source (str): Fichier source
            - page (int): Numero de page
            - tokens (int): Nombre de tokens
            - metadata (dict): Metadonnees supplementaires

    Raises:
        ValueError: Si max_tokens <= overlap_tokens.
        ValueError: Si le texte est vide.

    Example:
        >>> chunks = chunk_text(
        ...     "Article 4.1 - Le toucher-jouer...",
        ...     max_tokens=256,
        ...     metadata={"source": "LA.pdf", "page": 15, "corpus": "fr"}
        ... )
        >>> chunks[0]["tokens"] <= 256
        True
    """
    if max_tokens <= overlap_tokens:
        raise ValueError(
            f"max_tokens ({max_tokens}) must be greater than overlap_tokens ({overlap_tokens})"
        )

    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    from scripts.pipeline.utils import normalize_text

    text = normalize_text(text)
    encoder = tiktoken.get_encoding(TOKENIZER_NAME)

    chunks = []
    remaining = text
    seq = 0
    prev_overlap = ""  # Text to prepend from previous chunk

    while remaining:
        # Prepend overlap from previous chunk
        working_text = prev_overlap + remaining if prev_overlap else remaining
        tokens = encoder.encode(working_text)

        if len(tokens) <= max_tokens:
            # Last chunk
            chunk_text = working_text
            remaining = ""
            prev_overlap = ""
        else:
            # Split at sentence boundary
            chunk_text, remaining = split_at_sentence_boundary(
                working_text, max_tokens, tolerance=20
            )

            # Calculate overlap for next chunk
            if remaining and overlap_tokens > 0:
                chunk_token_list = encoder.encode(chunk_text)
                overlap_start = max(0, len(chunk_token_list) - overlap_tokens)
                prev_overlap = encoder.decode(chunk_token_list[overlap_start:])
            else:
                prev_overlap = ""

        # Enforce ISO 82045 limits
        chunk_text, remaining, chunk_tokens = _enforce_iso_limits(
            chunk_text, remaining, encoder
        )

        # ISO 82045: Skip chunks with text < 50 chars
        if len(chunk_text) < 50:
            continue

        chunks.append(_build_chunk_dict(chunk_text, chunk_tokens, metadata))
        seq += 1

    return chunks


def count_tokens(text: str) -> int:
    """
    Compte le nombre de tokens dans un texte.

    Utilise tiktoken avec l'encodeur cl100k_base pour compatibilite
    avec les modeles OpenAI et LLM modernes.

    Args:
        text: Texte a analyser.

    Returns:
        Nombre de tokens.

    Example:
        >>> count_tokens("Hello world")
        2
    """
    encoder = tiktoken.get_encoding(TOKENIZER_NAME)
    return len(encoder.encode(text))


def split_at_sentence_boundary(
    text: str, target_tokens: int, tolerance: int = 20
) -> tuple[str, str]:
    """
    Coupe le texte a une frontiere de phrase proche du nombre de tokens cible.

    Cette fonction evite de couper au milieu d'une phrase en cherchant
    le point de coupure optimal (., !, ?) proche du nombre de tokens cible.

    Args:
        text: Texte a couper.
        target_tokens: Nombre de tokens cible pour la premiere partie.
        tolerance: Tolerance en tokens pour trouver une frontiere (default 20).

    Returns:
        Tuple (premiere_partie, reste).

    Example:
        >>> first, rest = split_at_sentence_boundary("First sentence. Second.", 10)
        >>> first.endswith(".")
        True
    """
    import re

    encoder = tiktoken.get_encoding(TOKENIZER_NAME)

    # Find sentence boundaries
    sentence_ends = list(re.finditer(r"[.!?]\s+", text))

    if not sentence_ends:
        # No sentence boundary, split at target
        tokens = encoder.encode(text)
        if len(tokens) <= target_tokens:
            return text, ""
        first_tokens = tokens[:target_tokens]
        rest_tokens = tokens[target_tokens:]
        return encoder.decode(first_tokens), encoder.decode(rest_tokens)

    # Find best split point near target_tokens
    best_pos = 0
    best_diff = float("inf")

    for match in sentence_ends:
        pos = match.end()
        first_part = text[:pos]
        tokens_count = len(encoder.encode(first_part))

        diff = abs(tokens_count - target_tokens)
        if diff < best_diff and tokens_count <= target_tokens + tolerance:
            best_diff = diff
            best_pos = pos

    if best_pos == 0:
        # Fallback: use first sentence boundary
        best_pos = sentence_ends[0].end()

    return text[:best_pos].strip(), text[best_pos:].strip()


def generate_chunk_id(corpus: str, doc_num: int, page: int, seq: int) -> str:
    """
    Genere un identifiant unique pour un chunk.

    Format: {CORPUS}-{DOC_NUM:03d}-{PAGE:03d}-{SEQ:02d}

    Args:
        corpus: Code corpus (fr ou intl).
        doc_num: Numero du document (1-999).
        page: Numero de page (1-999).
        seq: Numero de sequence dans la page (0-99).

    Returns:
        Identifiant unique du chunk.

    Example:
        >>> generate_chunk_id("fr", 1, 15, 1)
        "FR-001-015-01"
    """
    corpus_upper = corpus.upper()
    if corpus_upper not in ("FR", "INTL"):
        raise ValueError(f"Invalid corpus: {corpus}")

    return f"{corpus_upper}-{doc_num:03d}-{page:03d}-{seq:02d}"


def chunk_document(
    extracted_data: dict,
    corpus: str,
    doc_num: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[dict]:
    """
    Chunke un document extrait complet.

    Prend les donnees extraites par extract_pdf et genere tous les chunks
    pour le document avec IDs uniques et metadonnees.

    Args:
        extracted_data: Donnees du document (output de extract_pdf).
        corpus: Code corpus (fr ou intl).
        doc_num: Numero du document dans le corpus.
        max_tokens: Taille max par chunk (default 256).
        overlap_tokens: Overlap entre chunks (default 50).

    Returns:
        Liste de tous les chunks du document.
    """
    all_chunks = []

    for page_data in extracted_data.get("pages", []):
        page_num = page_data.get("page_num", 0)
        text = page_data.get("text", "")
        section = page_data.get("section")

        if not text or len(text.strip()) < 50:
            continue

        metadata = {
            "source": extracted_data.get("filename", ""),
            "page": page_num,
            "section": section,
            "corpus": corpus,
        }

        page_chunks = chunk_text(text, max_tokens, overlap_tokens, metadata)

        # Assign IDs
        for seq, chunk in enumerate(page_chunks):
            chunk["id"] = generate_chunk_id(corpus, doc_num, page_num, seq)

        all_chunks.extend(page_chunks)

    return all_chunks


def chunk_corpus(
    input_dir: Path,
    output_file: Path,
    corpus: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> dict:
    """
    Chunke tous les documents d'un corpus.

    Args:
        input_dir: Dossier contenant les extractions JSON.
        output_file: Fichier de sortie (chunks_fr.json ou chunks_intl.json).
        corpus: Code corpus (fr ou intl).
        max_tokens: Taille max par chunk.
        overlap_tokens: Overlap entre chunks.

    Returns:
        Rapport de chunking avec statistiques.
    """
    from scripts.pipeline.utils import get_timestamp, load_json, save_json

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != "extraction_report.json"]

    logger.info(f"Found {len(json_files)} extraction files in {input_dir}")

    all_chunks = []
    doc_num = 1

    for json_file in json_files:
        logger.info(f"Chunking: {json_file.name}")
        extracted_data = load_json(json_file)

        doc_chunks = chunk_document(
            extracted_data, corpus, doc_num, max_tokens, overlap_tokens
        )
        all_chunks.extend(doc_chunks)
        doc_num += 1

    # Build output structure
    output_data = {
        "metadata": {
            "corpus": corpus,
            "generated": get_timestamp(),
            "total_chunks": len(all_chunks),
            "schema_version": "1.0",
        },
        "chunks": all_chunks,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_data, output_file)

    report = {
        "corpus": corpus,
        "documents_processed": len(json_files),
        "total_chunks": len(all_chunks),
        "avg_tokens": sum(c["tokens"] for c in all_chunks) / len(all_chunks)
        if all_chunks
        else 0,
        "output_file": str(output_file),
        "timestamp": get_timestamp(),
    }

    logger.info(f"Generated {len(all_chunks)} chunks -> {output_file}")

    return report


# --- CLI ---


def main() -> None:
    """Point d'entree CLI pour le chunking."""
    parser = argparse.ArgumentParser(
        description="Segmentation en chunks pour Pocket Arbiter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python chunker.py --input corpus/processed/raw_fr --output corpus/processed/chunks_fr.json
    python chunker.py --input corpus/processed/raw_intl --output corpus/processed/chunks_intl.json --max-tokens 512
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Dossier contenant les extractions JSON",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Fichier de sortie JSON",
    )

    parser.add_argument(
        "--corpus",
        "-c",
        type=str,
        choices=["fr", "intl"],
        default="fr",
        help="Code corpus (default: fr)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Taille max par chunk (default: {DEFAULT_MAX_TOKENS})",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP_TOKENS,
        help=f"Overlap entre chunks (default: {DEFAULT_OVERLAP_TOKENS})",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Afficher les logs detailles",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Chunking corpus {args.corpus}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Max tokens: {args.max_tokens}, Overlap: {args.overlap}")

    report = chunk_corpus(
        args.input, args.output, args.corpus, args.max_tokens, args.overlap
    )

    logger.info(f"Documents: {report['documents_processed']}")
    logger.info(f"Chunks: {report['total_chunks']}")
    logger.info(f"Avg tokens: {report['avg_tokens']:.1f}")


if __name__ == "__main__":
    main()
