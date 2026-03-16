"""
Chunker Mode A: HybridChunker (Docling natif) avec tokenizer EmbeddingGemma.

Pipeline natif Docling pour chunking avec provenance page 100%.
Utilise le meme tokenizer que le modele d'embedding pour consistance.

ISO Reference:
    - ISO/IEC 42001 A.6.2.2 - AI traceability (page provenance 100%)
    - ISO/IEC 25010 S4.2 - Performance efficiency (Recall >= 90%)
    - ISO/IEC 12207 S7.3.3 - Implementation

Sources:
    - https://docling-project.github.io/docling/examples/hybrid_chunking/
    - https://github.com/docling-project/docling/discussions/1012

Usage:
    python -m scripts.pipeline.chunker_hybrid \
        --input corpus/processed/docling_fr \
        --output corpus/processed/chunks_mode_a_fr.json
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

# Workaround for Windows symlink permission issue with HuggingFace
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types import DoclingDocument
from transformers import AutoTokenizer

from scripts.pipeline.token_utils import EMBED_MODEL_ID

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# HybridChunker configuration (ISO 25010 PR-01, Chroma 2025)
MAX_TOKENS = 450  # Child chunk size for search
MERGE_PEERS = True  # Merge undersized chunks


def get_hybrid_tokenizer() -> HuggingFaceTokenizer:
    """
    Get HybridChunker tokenizer with EmbeddingGemma.

    Uses same tokenizer as embedding model for consistency (ISO 42001).

    Returns:
        HuggingFaceTokenizer instance for HybridChunker.
    """
    hf_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    return HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=MAX_TOKENS,
    )


def extract_page_numbers(chunk: object) -> list[int]:
    """
    Extract page numbers from chunk provenance.

    Source: Docling Discussion #1012
    Path: chunk.meta.doc_items[].prov[].page_no

    Args:
        chunk: HybridChunker chunk object.

    Returns:
        Sorted list of unique page numbers.
    """
    pages: set[int] = set()

    if not hasattr(chunk, "meta") or chunk.meta is None:
        return []

    doc_items = getattr(chunk.meta, "doc_items", [])
    for item in doc_items:
        prov_list = getattr(item, "prov", [])
        for prov in prov_list:
            page_no = getattr(prov, "page_no", None)
            if page_no is not None and page_no > 0:
                pages.add(page_no)

    return sorted(pages)


def extract_headings(chunk: object) -> list[str]:
    """
    Extract section headings from chunk metadata.

    Args:
        chunk: HybridChunker chunk object.

    Returns:
        List of heading strings.
    """
    if not hasattr(chunk, "meta") or chunk.meta is None:
        return []
    return getattr(chunk.meta, "headings", []) or []


def chunk_document_hybrid(
    docling_dict: dict,
    source: str,
    corpus: str = "fr",
) -> list[dict]:
    """
    Chunk document with HybridChunker (Mode A).

    Args:
        docling_dict: DoclingDocument dict from export_to_dict().
        source: Source filename.
        corpus: Corpus name (fr/intl).

    Returns:
        List of chunks with 100% page provenance.

    Raises:
        ValueError: If docling_dict is empty or invalid.
    """
    if not docling_dict:
        raise ValueError(f"Empty docling_document for {source}")

    # Load DoclingDocument from dict
    doc = DoclingDocument.model_validate(docling_dict)

    # Create HybridChunker with EmbeddingGemma tokenizer
    tokenizer = get_hybrid_tokenizer()
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=MERGE_PEERS)

    # Get raw tokenizer for token counting
    hf_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

    chunks = []
    for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
        pages = extract_page_numbers(chunk)
        headings = extract_headings(chunk)

        # Primary page (first) or 0 if no provenance
        primary_page = pages[0] if pages else 0

        # Build section from headings
        section = headings[0] if headings else ""

        # Count tokens
        tokens = len(hf_tokenizer.encode(chunk.text))

        chunks.append(
            {
                "id": f"{source}-hybrid-{i:04d}",
                "text": chunk.text,
                "source": source,
                "page": primary_page,
                "pages": pages,  # All pages (multi-page chunks)
                "section": section,
                "article_num": None,  # Can be extracted from headings if needed
                "tokens": tokens,
                "corpus": corpus,
                "chunk_type": "hybrid",
            }
        )

    return chunks


def process_docling_output_hybrid(
    input_dir: Path,
    output_file: Path,
    corpus: str = "fr",
) -> dict[str, Any]:
    """
    Process all Docling files with HybridChunker (Mode A).

    Args:
        input_dir: Directory containing Docling JSON files.
        output_file: Output chunks JSON file.
        corpus: Corpus name.

    Returns:
        Processing report dict.
    """
    json_files = sorted(input_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != "extraction_report.json"]

    logger.info(f"Found {len(json_files)} Docling files in {input_dir}")

    all_chunks: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "files": 0,
        "chunks": 0,
        "with_page": 0,
        "without_page": 0,
        "errors": [],
    }

    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            docling_dict = data.get("docling_document", {})
            source = data.get("filename", json_file.stem + ".pdf")

            if not docling_dict:
                error_msg = f"No docling_document in {json_file.name}"
                logger.warning(error_msg)
                stats["errors"].append(error_msg)
                continue

            chunks = chunk_document_hybrid(docling_dict, source, corpus)

            stats["files"] += 1
            stats["chunks"] += len(chunks)
            stats["with_page"] += sum(1 for c in chunks if c.get("page", 0) > 0)
            stats["without_page"] += sum(1 for c in chunks if c.get("page", 0) == 0)

            all_chunks.extend(chunks)
            logger.info(f"  {source}: {len(chunks)} chunks")

        except Exception as e:
            error_msg = f"{json_file.name}: {e}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)

    # Save chunks
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {"chunks": all_chunks, "total": len(all_chunks)},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Log results
    pct_page = 100 * stats["with_page"] / max(1, stats["chunks"])
    logger.info(f"Saved {stats['chunks']} chunks to {output_file}")
    logger.info(f"  with_page: {stats['with_page']} ({pct_page:.1f}%)")
    logger.info(f"  without_page: {stats['without_page']}")

    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HybridChunker Mode A - Pocket Arbiter (Docling native)"
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Docling output directory"
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output chunks JSON"
    )
    parser.add_argument(
        "--corpus", "-c", type=str, default="fr", help="Corpus name (fr/intl)"
    )

    args = parser.parse_args()

    stats = process_docling_output_hybrid(args.input, args.output, args.corpus)

    print(
        f"\nMode A (HybridChunker): {stats['files']} files -> {stats['chunks']} chunks"
    )
    pct = 100 * stats["with_page"] / max(1, stats["chunks"])
    print(f"  Page coverage: {pct:.1f}%")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")


if __name__ == "__main__":
    main()
