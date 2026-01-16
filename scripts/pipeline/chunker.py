"""
Segmentation en chunks - Pocket Arbiter v2.0

Ce module segmente le texte extrait en chunks semantiques optimaux
pour le retrieval RAG, basé sur la structure des Articles/Sections.

ISO Reference:
    - ISO/IEC 12207 S7.3.3 - Implementation
    - ISO 82045 - Document metadata
    - ISO 25010 - Functional Suitability (Recall >= 80%)

Usage:
    python chunker.py --input corpus/processed/raw_fr --output corpus/processed/chunks_fr.json
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import tiktoken

from scripts.pipeline.chunker_article import detect_article_boundaries
from scripts.pipeline.token_utils import (
    TOKENIZER_NAME,
    count_tokens as _count_tokens_shared,
    get_tokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Constants v2.0 ---

DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 128
MIN_CHUNK_TOKENS = 100
MAX_CHUNK_TOKENS = 1024


# --- Semantic Chunking Functions ---


def chunk_by_article(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """Chunking semantique base sur les Articles/Sections."""
    from scripts.pipeline.utils import normalize_text

    text = normalize_text(text)
    encoder = get_tokenizer()
    segments = detect_article_boundaries(text)

    if not segments:
        return chunk_text_legacy(text, max_tokens, overlap_tokens, metadata)

    chunks = []
    for segment in segments:
        segment_text = segment["text"]
        segment_article = segment["article"]
        segment_tokens = len(encoder.encode(segment_text))

        enriched_metadata = dict(metadata) if metadata else {}
        enriched_metadata["article"] = segment_article

        if segment_tokens <= max_tokens:
            chunks.append(
                _build_chunk_dict(segment_text, segment_tokens, enriched_metadata)
            )
        else:
            sub_chunks = chunk_text_legacy(
                segment_text, max_tokens, overlap_tokens, enriched_metadata
            )
            chunks.extend(sub_chunks)

    return chunks


def _compute_overlap(
    chunk_text_str: str, remaining: str, overlap_tokens: int, encoder: tiktoken.Encoding
) -> str:
    """Compute overlap text from end of chunk."""
    if not remaining or overlap_tokens <= 0:
        return ""
    chunk_token_list = encoder.encode(chunk_text_str)
    overlap_start = max(0, len(chunk_token_list) - overlap_tokens)
    return encoder.decode(chunk_token_list[overlap_start:])


def _process_chunk_iteration(
    working_text: str, max_tokens: int, encoder: tiktoken.Encoding
) -> tuple[str, str]:
    """Process one iteration of chunking."""
    tokens = encoder.encode(working_text)
    if len(tokens) <= max_tokens:
        return working_text, ""
    return split_at_sentence_boundary(working_text, max_tokens, tolerance=30)


def chunk_text_legacy(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """Chunking par phrases (fallback si pas d'Articles)."""
    if max_tokens <= overlap_tokens:
        raise ValueError(
            f"max_tokens ({max_tokens}) must be greater than overlap_tokens"
        )

    if not text or not text.strip():
        return []

    from scripts.pipeline.utils import normalize_text

    text = normalize_text(text)
    encoder = get_tokenizer()

    chunks = []
    remaining = text
    prev_overlap = ""

    while remaining:
        working_text = prev_overlap + remaining if prev_overlap else remaining
        chunk_text_str, remaining = _process_chunk_iteration(
            working_text, max_tokens, encoder
        )
        prev_overlap = _compute_overlap(
            chunk_text_str, remaining, overlap_tokens, encoder
        )

        chunk_text_str, remaining, chunk_tokens = _enforce_iso_limits(
            chunk_text_str, remaining, encoder, max_tokens
        )

        if chunk_tokens < MIN_CHUNK_TOKENS:
            continue

        chunks.append(_build_chunk_dict(chunk_text_str, chunk_tokens, metadata))

    return chunks


def chunk_text(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """Segmente le texte en chunks semantiques."""
    return chunk_by_article(text, max_tokens, overlap_tokens, metadata)


# --- Helper Functions ---


def _build_chunk_dict(
    chunk_text: str, chunk_tokens: int, metadata: Optional[dict]
) -> dict:
    """Build a chunk dictionary from text and metadata."""
    from scripts.pipeline.utils import get_date

    return {
        "id": "",
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
    encoder: tiktoken.Encoding,
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
    """Compte le nombre de tokens dans un texte.

    Utilise le module partage token_utils pour coherence (ISO 12207).
    """
    return _count_tokens_shared(text)


def split_at_sentence_boundary(
    text: str, target_tokens: int, tolerance: int = 30
) -> tuple[str, str]:
    """Coupe le texte a une frontiere de phrase."""
    encoder = get_tokenizer()
    sentence_ends = list(re.finditer(r"[.!?]\s+", text))

    if not sentence_ends:
        tokens = encoder.encode(text)
        if len(tokens) <= target_tokens:
            return text, ""
        return encoder.decode(tokens[:target_tokens]), encoder.decode(
            tokens[target_tokens:]
        )

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
    """Genere un identifiant unique pour un chunk."""
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
    """Chunke un document extrait complet avec chunking semantique."""
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
    """Chunke tous les documents d'un corpus."""
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
    return report


# --- CLI ---


def main() -> None:
    """Point d'entree CLI pour le chunking."""
    parser = argparse.ArgumentParser(
        description="Segmentation semantique en chunks v2.0"
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
        help="Corpus code",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP_TOKENS,
        help=f"Overlap (default: {DEFAULT_OVERLAP_TOKENS})",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Chunking v2.0 - Corpus: {args.corpus}, Input: {args.input}")
    report = chunk_corpus(
        args.input, args.output, args.corpus, args.max_tokens, args.overlap
    )
    logger.info(
        f"Chunks: {report['total_chunks']}, Articles: {report['articles_detected']}"
    )


if __name__ == "__main__":
    main()
