"""
Segmentation en chunks - Pocket Arbiter v2.0

Ce module segmente le texte extrait en chunks semantiques optimaux
pour le retrieval RAG, basé sur la structure des Articles/Sections.

ISO Reference:
    - ISO/IEC 12207 §7.3.3 - Implementation
    - ISO 82045 - Document metadata
    - ISO 25010 - Functional Suitability (Recall >= 80%)

Strategy v2.0 (2026-01-15):
    - Semantic chunking par Article/Section (vs fixed-size)
    - 512 tokens max (vs 256)
    - Hybrid: Article-aware + sentence boundary fallback
    - Ref: docs/CHUNKING_STRATEGY.md

Dependencies:
    - tiktoken >= 0.5.0

Usage:
    python chunker.py --input corpus/processed/raw_fr --output corpus/processed/chunks_fr.json
    python chunker.py --input corpus/processed/raw_intl --output corpus/processed/chunks_intl.json
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import tiktoken

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --- Constants v2.0 ---

DEFAULT_MAX_TOKENS = 512  # v2.0: Increased from 256
DEFAULT_OVERLAP_TOKENS = 128  # v2.1: 25% overlap (research: 20-25% optimal)
MIN_CHUNK_TOKENS = 100  # v2.0: Avoid tiny fragments
MAX_CHUNK_TOKENS = 1024  # v2.0: Allow large Articles
TOKENIZER_NAME = "cl100k_base"  # Compatible OpenAI/LLM

# Article/Section detection patterns (French + English)
ARTICLE_PATTERNS = [
    # French patterns - ordered by specificity
    r"^(Article\s+\d+(?:\.\d+)*(?:\.\d+)?)",  # Article 4, Article 4.1, Article 4.1.2
    r"^(Chapitre\s+\d+(?:\.\d+)?)",  # Chapitre 2, Chapitre 2.1
    r"^(Section\s+\d+)",  # Section 1
    r"^(Annexe\s+[A-Z])",  # Annexe A
    r"^(Titre\s+[IVX]+)",  # Titre I, Titre II (Roman numerals)
    r"^(Partie\s+\d+)",  # Partie 1, Partie 2
    r"^(TITRE\s+[IVX]+)",  # TITRE I (uppercase)
    r"^(STATUTS)",  # STATUTS header
    r"^(REGLEMENT)",  # REGLEMENT header
    # Numeric patterns (most common in FFE docs)
    r"^(\d+\.\d+\.\d+\.?\s)",  # 4.1.2 (3-level)
    r"^(\d+\.\d+\.?\s)",  # 4.1 (2-level)
    r"^(\d+\.\d+)$",  # 5.5 alone on line
    r"^(\d+\.\s+[A-Z])",  # 4. Le toucher (single digit + text)
    # Lettered subsections
    r"^([a-z]\)\s)",  # a) , b) , c)
    r"^(\([a-z]\)\s)",  # (a), (b), (c)
    r"^([A-Z]\.\s)",  # A. , B. , C.
    # English patterns (FIDE)
    r"^(Article\s+\d+(?:\.\d+)*)",  # Article 4.1
    r"^(Chapter\s+\d+)",  # Chapter 2
    r"^(Appendix\s+[A-Z])",  # Appendix A
    r"^(Part\s+[IVX]+)",  # Part I, Part II
    r"^(Rule\s+\d+)",  # Rule 1, Rule 2
    r"^(Preface)",  # Preface
    r"^(Introduction)",  # Introduction
]


# --- Semantic Chunking Functions ---


def detect_article_boundaries(text: str) -> list[dict]:
    """
    Detecte les frontieres d'Articles/Sections dans le texte.

    Identifie les debuts d'Articles (4.1, 4.2, etc.) pour permettre
    un chunking semantique qui preserve le contexte reglementaire.

    Args:
        text: Texte complet d'une page ou document.

    Returns:
        Liste de segments avec:
            - start (int): Position de debut
            - end (int): Position de fin
            - article (str): Identifiant Article detecte (ou None)
            - text (str): Contenu du segment

    Example:
        >>> segments = detect_article_boundaries("Article 4.1 Toucher...")
        >>> segments[0]["article"]
        "Article 4.1"
    """
    if not text:
        return []

    segments: list[dict[str, str | None]] = []
    lines = text.split("\n")
    current_article: str | None = None
    current_lines: list[str] = []

    for line in lines:
        line_stripped = line.strip()
        article_match: str | None = None

        # Check if line starts an Article
        for pattern in ARTICLE_PATTERNS:
            match = re.match(pattern, line_stripped, re.IGNORECASE)
            if match:
                article_match = match.group(1).strip()
                break

        if article_match:
            # Save previous segment if it has content
            if current_lines:
                segment_text = "\n".join(current_lines)
                if len(segment_text.strip()) > 50:
                    segments.append(
                        {
                            "article": current_article,
                            "text": segment_text.strip(),
                        }
                    )

            # Start new segment
            current_article = article_match
            current_lines = [line]
        else:
            current_lines.append(line)

    # Don't forget last segment
    if current_lines:
        segment_text = "\n".join(current_lines)
        if len(segment_text.strip()) > 50:
            segments.append(
                {
                    "article": current_article,
                    "text": segment_text.strip(),
                }
            )

    return segments


def chunk_by_article(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """
    Chunking semantique base sur les Articles/Sections.

    Strategie v2.0:
    1. Detecter les frontieres d'Articles
    2. Si Article <= max_tokens: garder entier
    3. Si Article > max_tokens: split sur phrases
    4. Fallback: chunking par phrases si pas d'Articles

    Args:
        text: Texte a segmenter.
        max_tokens: Taille max par chunk (default 512).
        overlap_tokens: Chevauchement entre chunks (default 64).
        metadata: Metadonnees (source, page, corpus).

    Returns:
        Liste de chunks avec metadata enrichie (article detecte).
    """
    from scripts.pipeline.utils import normalize_text

    text = normalize_text(text)
    encoder = tiktoken.get_encoding(TOKENIZER_NAME)

    # Detect article boundaries
    segments = detect_article_boundaries(text)

    # If no articles detected, fall back to sentence-based chunking
    if not segments:
        return chunk_text_legacy(text, max_tokens, overlap_tokens, metadata)

    chunks = []

    for segment in segments:
        segment_text = segment["text"]
        segment_article = segment["article"]
        segment_tokens = len(encoder.encode(segment_text))

        # Enrich metadata with article info
        enriched_metadata = dict(metadata) if metadata else {}
        enriched_metadata["article"] = segment_article

        if segment_tokens <= max_tokens:
            # Article fits in one chunk - keep it whole
            chunk = _build_chunk_dict(segment_text, segment_tokens, enriched_metadata)
            chunks.append(chunk)
        else:
            # Article too long - split on sentence boundaries
            sub_chunks = chunk_text_legacy(
                segment_text, max_tokens, overlap_tokens, enriched_metadata
            )
            chunks.extend(sub_chunks)

    return chunks


def chunk_text_legacy(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """
    Chunking par phrases (fallback si pas d'Articles).

    Methode v1.0 preservee comme fallback pour textes non structures.
    """
    if max_tokens <= overlap_tokens:
        raise ValueError(
            f"max_tokens ({max_tokens}) must be greater than overlap_tokens"
        )

    if not text or not text.strip():
        return []

    from scripts.pipeline.utils import normalize_text

    text = normalize_text(text)
    encoder = tiktoken.get_encoding(TOKENIZER_NAME)

    chunks = []
    remaining = text
    prev_overlap = ""

    while remaining:
        working_text = prev_overlap + remaining if prev_overlap else remaining
        tokens = encoder.encode(working_text)

        if len(tokens) <= max_tokens:
            chunk_text_str = working_text
            remaining = ""
            prev_overlap = ""
        else:
            chunk_text_str, remaining = split_at_sentence_boundary(
                working_text, max_tokens, tolerance=30
            )

            if remaining and overlap_tokens > 0:
                chunk_token_list = encoder.encode(chunk_text_str)
                overlap_start = max(0, len(chunk_token_list) - overlap_tokens)
                prev_overlap = encoder.decode(chunk_token_list[overlap_start:])
            else:
                prev_overlap = ""

        # Enforce limits
        chunk_text_str, remaining, chunk_tokens = _enforce_iso_limits(
            chunk_text_str, remaining, encoder, max_tokens
        )

        # Skip tiny chunks
        if chunk_tokens < MIN_CHUNK_TOKENS and len(chunk_text_str) < 100:
            continue

        chunks.append(_build_chunk_dict(chunk_text_str, chunk_tokens, metadata))

    return chunks


# Keep old function name for compatibility
def chunk_text(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """
    Segmente le texte en chunks semantiques.

    v2.0: Utilise chunk_by_article pour chunking semantique.
    Fallback sur chunk_text_legacy si pas d'Articles detectes.

    Args:
        text: Texte brut a segmenter.
        max_tokens: Taille maximale par chunk en tokens (default 512).
        overlap_tokens: Chevauchement entre chunks en tokens (default 64).
        metadata: Metadonnees a propager vers chaque chunk.

    Returns:
        Liste de chunks conformes au schema.
    """
    return chunk_by_article(text, max_tokens, overlap_tokens, metadata)


# --- Helper Functions ---


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
            "article": metadata.get("article") if metadata else None,
            "corpus": metadata.get("corpus", "fr") if metadata else "fr",
            "extraction_date": get_date(),
            "version": "2.0",
        },
    }


def _enforce_iso_limits(
    chunk_text: str,
    remaining: str,
    encoder: "tiktoken.Encoding",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[str, str, int]:
    """Enforce ISO 25010 strict max_tokens limit."""
    chunk_tokens = len(encoder.encode(chunk_text))
    if chunk_tokens > max_tokens:
        hard_tokens = encoder.encode(chunk_text)[:max_tokens]
        overflow = encoder.decode(encoder.encode(chunk_text)[max_tokens:])
        chunk_text = encoder.decode(hard_tokens)
        remaining = overflow + " " + remaining if remaining else overflow
        chunk_tokens = len(encoder.encode(chunk_text))
    return chunk_text, remaining, chunk_tokens


def count_tokens(text: str) -> int:
    """Compte le nombre de tokens dans un texte."""
    encoder = tiktoken.get_encoding(TOKENIZER_NAME)
    return len(encoder.encode(text))


def split_at_sentence_boundary(
    text: str, target_tokens: int, tolerance: int = 30
) -> tuple[str, str]:
    """
    Coupe le texte a une frontiere de phrase.

    v2.0: Tolerance augmentee a 30 tokens pour meilleure coherence.
    """
    encoder = tiktoken.get_encoding(TOKENIZER_NAME)

    # Find sentence boundaries
    sentence_ends = list(re.finditer(r"[.!?]\s+", text))

    if not sentence_ends:
        tokens = encoder.encode(text)
        if len(tokens) <= target_tokens:
            return text, ""
        first_tokens = tokens[:target_tokens]
        rest_tokens = tokens[target_tokens:]
        return encoder.decode(first_tokens), encoder.decode(rest_tokens)

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
        best_pos = sentence_ends[0].end()

    return text[:best_pos].strip(), text[best_pos:].strip()


def generate_chunk_id(corpus: str, doc_num: int, page: int, seq: int) -> str:
    """
    Genere un identifiant unique pour un chunk.

    Format: {CORPUS}-{DOC_NUM:03d}-{PAGE:03d}-{SEQ:02d}
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
    Chunke un document extrait complet avec chunking semantique.

    v2.0: Utilise chunk_by_article pour preserver les Articles entiers.
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

    v2.0: Config chunking sauvegardee dans metadata pour tracabilite.
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

    # Build output structure with v2.0 config
    output_data = {
        "metadata": {
            "corpus": corpus,
            "generated": get_timestamp(),
            "total_chunks": len(all_chunks),
            "schema_version": "2.0",
        },
        "config": {
            "strategy": "semantic_article",
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
            "min_chunk_tokens": MIN_CHUNK_TOKENS,
            "overlap_percent": round(overlap_tokens / max_tokens * 100, 1),
        },
        "chunks": all_chunks,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_data, output_file)

    # Count articles detected
    articles_detected = sum(
        1 for c in all_chunks if c.get("metadata", {}).get("article")
    )

    report = {
        "corpus": corpus,
        "documents_processed": len(json_files),
        "total_chunks": len(all_chunks),
        "articles_detected": articles_detected,
        "avg_tokens": sum(c["tokens"] for c in all_chunks) / len(all_chunks)
        if all_chunks
        else 0,
        "output_file": str(output_file),
        "timestamp": get_timestamp(),
    }

    logger.info(f"Generated {len(all_chunks)} chunks -> {output_file}")
    logger.info(f"Articles detected: {articles_detected}")

    return report


# --- CLI ---


def main() -> None:
    """Point d'entree CLI pour le chunking."""
    parser = argparse.ArgumentParser(
        description="Segmentation semantique en chunks pour Pocket Arbiter v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python chunker.py -i corpus/processed/raw_fr -o corpus/processed/chunks_fr.json
    python chunker.py -i corpus/processed/raw_intl -o corpus/processed/chunks_intl.json -c intl

v2.0 Changes:
    - Semantic chunking by Article/Section
    - Default max_tokens: 512 (was 256)
    - Articles kept whole when possible
    - Config saved in output metadata
        """,
    )

    parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Input JSON extractions dir"
    )

    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output chunks JSON file"
    )

    parser.add_argument(
        "--corpus",
        "-c",
        type=str,
        choices=["fr", "intl"],
        default="fr",
        help="Corpus code (default: fr)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per chunk (default: {DEFAULT_MAX_TOKENS})",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP_TOKENS,
        help=f"Overlap tokens (default: {DEFAULT_OVERLAP_TOKENS})",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Chunking v2.0 - Semantic Article-based")
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Max tokens: {args.max_tokens}, Overlap: {args.overlap}")

    report = chunk_corpus(
        args.input, args.output, args.corpus, args.max_tokens, args.overlap
    )

    logger.info(f"Documents: {report['documents_processed']}")
    logger.info(f"Chunks: {report['total_chunks']}")
    logger.info(f"Articles detected: {report['articles_detected']}")
    logger.info(f"Avg tokens: {report['avg_tokens']:.1f}")


if __name__ == "__main__":
    main()
