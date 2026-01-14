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

# TODO: Uncomment when implementing
# import tiktoken

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --- Constants ---

DEFAULT_MAX_TOKENS = 256
DEFAULT_OVERLAP_TOKENS = 50
TOKENIZER_NAME = "cl100k_base"  # Compatible OpenAI/LLM


# --- Main Functions ---


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

    # TODO: Implement with tiktoken
    raise NotImplementedError("chunk_text not yet implemented - Phase 1A")


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
    # TODO: Implement with tiktoken
    raise NotImplementedError("count_tokens not yet implemented - Phase 1A")


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
    # TODO: Implement
    raise NotImplementedError("split_at_sentence_boundary not yet implemented")


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
    # TODO: Implement
    raise NotImplementedError("chunk_document not yet implemented - Phase 1A")


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
    # TODO: Implement
    raise NotImplementedError("chunk_corpus not yet implemented - Phase 1A")


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

    # TODO: Call chunk_corpus
    logger.error("chunker.py not yet implemented - Phase 1A")


if __name__ == "__main__":
    main()
