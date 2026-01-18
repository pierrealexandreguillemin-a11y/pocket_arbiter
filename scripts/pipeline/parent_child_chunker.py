"""
Parent-Child Chunker - Pocket Arbiter

Implements Parent-Document Retrieval pattern:
- Child chunks (300 tokens): embedded for precise semantic search
- Parent chunks (800 tokens): returned as context for LLM

The child chunks are what get embedded and searched against.
When a child matches, we return its parent for richer context.

ISO Reference:
    - ISO/IEC 25010 S4.2 - Performance efficiency (Recall >= 90%)
    - ISO/IEC 42001 - AI traceability
    - ISO/IEC 12207 S7.3.3 - Implementation

Changelog:
    - 2026-01-18: Initial implementation (Step 2 chunking strategy)

Usage:
    python -m scripts.pipeline.parent_child_chunker \
        --input corpus/processed/raw_fr \
        --output corpus/processed/chunks_parent_child_fr.json
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from scripts.pipeline.token_utils import (
    TOKENIZER_NAME,
    count_tokens,
    get_tokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Constants (TOKENS) ---

# Parent chunks: rich context for LLM response
PARENT_CHUNK_SIZE = 800
PARENT_CHUNK_OVERLAP = 100

# Child chunks: precise semantic units for search
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 60

MIN_CHUNK_TOKENS = 30  # Minimum for child chunks

# Separateurs hierarchiques pour reglements
REGULATORY_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""]


def create_splitter(
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: tiktoken.Encoding,
) -> RecursiveCharacterTextSplitter:
    """Create a RecursiveCharacterTextSplitter with token counting."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=REGULATORY_SEPARATORS,
        length_function=lambda text: count_tokens(text, tokenizer),
        keep_separator=True,
    )


def extract_article_number(text: str) -> str | None:
    """
    Extract article number from text.

    Patterns:
        - Article 1.2.3
        - Art. 5
        - §3.4
    """
    patterns = [
        r"(?:Article|Art\.?)\s*(\d+(?:\.\d+)*)",
        r"§\s*(\d+(?:\.\d+)*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_section_header(text: str) -> str | None:
    """
    Extract section header from text.

    Patterns:
        - Chapitre X
        - TITRE Y
        - Section Z
    """
    patterns = [
        r"(Chapitre\s+\w+)",
        r"(TITRE\s+\w+)",
        r"(Section\s+\w+)",
        r"(ANNEXE\s+\w+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def chunk_document_parent_child(
    text: str,
    parent_splitter: RecursiveCharacterTextSplitter,
    child_splitter: RecursiveCharacterTextSplitter,
    source: str,
    page: int,
    tokenizer: tiktoken.Encoding,
) -> tuple[list[dict], list[dict]]:
    """
    Chunk document with parent-child pattern.

    Args:
        text: Document text.
        parent_splitter: Splitter for parent chunks (800 tokens).
        child_splitter: Splitter for child chunks (300 tokens).
        source: Source filename.
        page: Page number.
        tokenizer: Tokenizer for counting.

    Returns:
        Tuple of (parent_chunks, child_chunks).
    """
    if not text or len(text.strip()) < 20:
        return [], []

    # Step 1: Create parent chunks
    try:
        parent_texts = parent_splitter.split_text(text)
    except Exception as e:
        logger.warning(f"Parent split failed for {source} p{page}: {e}")
        return [], []

    parent_chunks = []
    child_chunks = []

    for parent_idx, parent_text in enumerate(parent_texts):
        parent_text = parent_text.strip()
        if not parent_text:
            continue

        parent_token_count = count_tokens(parent_text, tokenizer)
        parent_id = f"{source}-p{page}-parent{parent_idx}"

        # Extract metadata from parent text
        article_num = extract_article_number(parent_text)
        section = extract_section_header(parent_text)

        parent_chunk = {
            "id": parent_id,
            "text": parent_text,
            "source": source,
            "page": page,
            "chunk_type": "parent",
            "tokens": parent_token_count,
            "article_num": article_num,
            "section": section,
        }
        parent_chunks.append(parent_chunk)

        # Step 2: Create child chunks from this parent
        try:
            child_texts = child_splitter.split_text(parent_text)
        except Exception as e:
            logger.warning(f"Child split failed for {parent_id}: {e}")
            continue

        for child_idx, child_text in enumerate(child_texts):
            child_text = child_text.strip()
            if not child_text:
                continue

            child_token_count = count_tokens(child_text, tokenizer)
            if child_token_count < MIN_CHUNK_TOKENS:
                continue

            child_id = f"{source}-p{page}-parent{parent_idx}-child{child_idx}"

            child_chunk = {
                "id": child_id,
                "text": child_text,
                "source": source,
                "page": page,
                "parent_id": parent_id,  # Link to parent
                "chunk_type": "child",
                "tokens": child_token_count,
                "article_num": article_num,  # Inherit from parent
                "section": section,  # Inherit from parent
            }
            child_chunks.append(child_chunk)

    return parent_chunks, child_chunks


def process_corpus_parent_child(
    input_dir: Path,
    output_file: Path,
    corpus: str = "fr",
) -> dict[str, Any]:
    """
    Process corpus with parent-child chunking strategy.

    Args:
        input_dir: Directory with extraction JSON files.
        output_file: Output JSON file.
        corpus: Corpus code (fr, intl).

    Returns:
        Processing report.
    """
    tokenizer = get_tokenizer()

    logger.info(
        f"Creating parent splitter (size={PARENT_CHUNK_SIZE}, "
        f"overlap={PARENT_CHUNK_OVERLAP})"
    )
    parent_splitter = create_splitter(
        PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, tokenizer
    )

    logger.info(
        f"Creating child splitter (size={CHILD_CHUNK_SIZE}, "
        f"overlap={CHILD_CHUNK_OVERLAP})"
    )
    child_splitter = create_splitter(
        CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP, tokenizer
    )

    extraction_files = sorted(input_dir.glob("*.json"))
    logger.info(f"Found {len(extraction_files)} extraction files in {input_dir}")

    all_parents = []
    all_children = []
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

            parents, children = chunk_document_parent_child(
                text=text,
                parent_splitter=parent_splitter,
                child_splitter=child_splitter,
                source=source,
                page=page_num,
                tokenizer=tokenizer,
            )

            # Add corpus metadata
            for chunk in parents + children:
                chunk["corpus"] = corpus

            all_parents.extend(parents)
            all_children.extend(children)

    # Create parent lookup (id -> text)
    parent_lookup = {p["id"]: p["text"] for p in all_parents}

    # Save output
    output_data = {
        "corpus": corpus,
        "config": {
            "chunker": "parent_child",
            "parent_size_tokens": PARENT_CHUNK_SIZE,
            "parent_overlap_tokens": PARENT_CHUNK_OVERLAP,
            "child_size_tokens": CHILD_CHUNK_SIZE,
            "child_overlap_tokens": CHILD_CHUNK_OVERLAP,
            "tokenizer": TOKENIZER_NAME,
            "min_chunk_tokens": MIN_CHUNK_TOKENS,
        },
        "total_parents": len(all_parents),
        "total_children": len(all_children),
        "total_pages": total_pages,
        "parents": all_parents,
        "children": all_children,  # These get embedded
        "parent_lookup": parent_lookup,  # For retrieval
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    n_parents = len(all_parents)
    n_children = len(all_children)
    logger.info(f"Saved {n_parents} parents + {n_children} children to {output_file}")

    # Stats
    parent_tokens = [p["tokens"] for p in all_parents]
    child_tokens = [c["tokens"] for c in all_children]

    avg_parent = (
        round(sum(parent_tokens) / len(parent_tokens), 1) if parent_tokens else 0
    )
    avg_child = (
        round(sum(child_tokens) / len(child_tokens), 1) if child_tokens else 0
    )
    children_ratio = (
        round(n_children / n_parents, 2) if n_parents else 0
    )

    report = {
        "corpus": corpus,
        "chunker": "parent_child",
        "parent_size_tokens": PARENT_CHUNK_SIZE,
        "parent_overlap_tokens": PARENT_CHUNK_OVERLAP,
        "child_size_tokens": CHILD_CHUNK_SIZE,
        "child_overlap_tokens": CHILD_CHUNK_OVERLAP,
        "tokenizer": TOKENIZER_NAME,
        "total_parents": n_parents,
        "total_children": n_children,
        "total_pages": total_pages,
        "avg_parent_tokens": avg_parent,
        "avg_child_tokens": avg_child,
        "children_per_parent": children_ratio,
    }

    return report


def main() -> None:
    """CLI for parent-child chunking."""
    parser = argparse.ArgumentParser(
        description="Parent-Child Chunker - Pocket Arbiter",
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
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    report = process_corpus_parent_child(
        input_dir=args.input,
        output_file=args.output,
        corpus=args.corpus,
    )

    print("\n=== Parent-Child Chunking Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
