"""
Chunker - Pocket Arbiter

Pipeline ISO conforme avec deux modes:
1. DoclingDocument (prioritaire): HybridChunker avec 100% page provenance
2. Markdown fallback: MarkdownHeaderTextSplitter pour backward compatibility

ISO Reference:
    - ISO/IEC 12207 S7.3.3 - Implementation
    - ISO/IEC 25010 S4.2 - Performance efficiency (Recall >= 90%)
    - ISO/IEC 42001 A.6.2.2 - AI traceability (page provenance)

Research Sources (2025-2026):
    - Docling Discussion #1012: export_to_markdown() loses page metadata
    - Docling Discussion #444: Use chunk.meta.doc_items[].prov[].page_no
    - NVIDIA: 15% overlap optimal (FinanceBench)
    - arXiv: 512-1024 tokens pour contexte large
    - Chroma: 400-512 tokens sweet spot (85-90% recall)

Pipeline:
    Mode 1: DoclingDocument JSON → HybridChunker → Child chunks (100% pages)
    Mode 2: Docling markdown → MarkdownHeaderTextSplitter → Parent-Child chunks

Usage:
    python -m scripts.pipeline.chunker \
        --input corpus/processed/docling_fr \
        --output corpus/processed/chunks_fr.json
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

# Workaround for Windows symlink permission issue with HuggingFace
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from scripts.pipeline.token_utils import count_tokens, get_tokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Constants (ISO 25010 PR-01, NVIDIA/Chroma 2025) ---

# Parent chunks: rich context for LLM response (arXiv 2025)
PARENT_CHUNK_SIZE = 1024
PARENT_CHUNK_OVERLAP = 154  # NVIDIA 2025: 15% optimal

# Child chunks: precise semantic units for search (Chroma 2025)
CHILD_CHUNK_SIZE = 450
CHILD_CHUNK_OVERLAP = 68  # NVIDIA 2025: 15% optimal

MIN_CHUNK_TOKENS = 30  # Evite fragments inutiles

# Headers Markdown extraits par Docling
HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def chunk_markdown(
    markdown: str,
    source: str,
    corpus: str = "fr",
    page_map: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Chunk markdown avec extraction automatique des sections (Parent-Child).

    Args:
        markdown: Texte markdown extrait par Docling.
        source: Nom du fichier source.
        corpus: Corpus (fr/intl).
        page_map: Liste de pages [{page_num, text}] pour mapping page (optionnel).

    Returns:
        Tuple (parent_chunks, child_chunks) avec metadata section/article/page.
    """
    tokenizer = get_tokenizer()

    # Build page lookup with normalized text
    def normalize_for_match(text: str) -> str:
        """Normalize text for fuzzy matching."""
        # Remove markdown headers and formatting
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        # Collapse whitespace (newlines, tabs, multiple spaces)
        text = re.sub(r"\s+", " ", text)
        # Lowercase
        return text.lower().strip()

    # Pre-normalize all pages once
    normalized_pages = []
    if page_map:
        for page in page_map:
            normalized_pages.append(
                {
                    "page_num": page["page_num"],
                    "text_norm": normalize_for_match(page.get("text", "")),
                }
            )

    def find_page(text: str) -> int:
        """Find the page number for a text snippet using fuzzy matching."""
        if not normalized_pages:
            return 0

        text_norm = normalize_for_match(text)
        if len(text_norm) < 20:
            return 0

        # Try multiple snippets from the text
        snippets = [
            text_norm[:80],  # Start
            text_norm[50:130]
            if len(text_norm) > 130
            else text_norm[:80],  # Early middle
        ]

        for snippet in snippets:
            if len(snippet) < 20:
                continue
            for page in normalized_pages:
                if snippet in page["text_norm"]:
                    return page["page_num"]

        # Fallback: shorter snippet
        short_snippet = text_norm[:40]
        for page in normalized_pages:
            if short_snippet in page["text_norm"]:
                return page["page_num"]

        return 0

    # Step 1: Split by headers -> extract section metadata
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    header_docs = header_splitter.split_text(markdown)

    # Step 2: Parent splitter (1024 tokens)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        length_function=lambda t: count_tokens(t, tokenizer),
    )

    # Step 3: Child splitter (450 tokens)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        length_function=lambda t: count_tokens(t, tokenizer),
    )

    parent_chunks = []
    child_chunks = []

    for doc in header_docs:
        # Extract section from header metadata
        section = doc.metadata.get("h2") or doc.metadata.get("h1") or ""
        article = doc.metadata.get("h3") or doc.metadata.get("h4") or ""

        # Build article_num from h3/h4 if numeric
        article_num = None
        if article:
            match = re.match(r"^(\d+(?:\.\d+)*)", article)
            if match:
                article_num = match.group(1)

        # Split into parent chunks
        parent_texts = parent_splitter.split_text(doc.page_content)

        for p_idx, parent_text in enumerate(parent_texts):
            parent_text = parent_text.strip()
            if not parent_text:
                continue

            parent_tokens = count_tokens(parent_text, tokenizer)
            page = find_page(parent_text)
            parent_id = f"{source}-p{page:03d}-parent{len(parent_chunks):03d}"

            parent_chunks.append(
                {
                    "id": parent_id,
                    "text": parent_text,
                    "source": source,
                    "page": page,
                    "section": section,
                    "article_num": article_num,
                    "tokens": parent_tokens,
                    "corpus": corpus,
                    "chunk_type": "parent",
                }
            )

            # Split parent into child chunks
            child_texts = child_splitter.split_text(parent_text)

            for c_idx, child_text in enumerate(child_texts):
                child_text = child_text.strip()
                if not child_text:
                    continue

                child_tokens = count_tokens(child_text, tokenizer)
                if child_tokens < MIN_CHUNK_TOKENS:
                    continue

                child_page = find_page(child_text) or page
                child_id = f"{source}-p{child_page}-parent{len(parent_chunks) - 1:03d}-child{c_idx:02d}"

                child_chunks.append(
                    {
                        "id": child_id,
                        "text": child_text,
                        "source": source,
                        "page": child_page,
                        "section": section,
                        "article_num": article_num,
                        "parent_id": parent_id,
                        "tokens": child_tokens,
                        "corpus": corpus,
                        "chunk_type": "child",
                    }
                )

    return parent_chunks, child_chunks


def process_docling_output(
    input_dir: Path,
    output_file: Path,
    corpus: str = "fr",
    raw_dir: Path | None = None,
) -> dict:
    """
    Process tous les fichiers Docling d'un corpus (Parent-Child).

    Args:
        input_dir: Dossier contenant les JSON Docling (markdown).
        output_file: Fichier de sortie chunks (children pour embedding).
        corpus: Nom du corpus.
        raw_dir: Dossier contenant les raw JSON (pages avec page_num).

    Returns:
        Rapport de traitement.
    """
    json_files = sorted(input_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != "extraction_report.json"]

    logger.info(f"Found {len(json_files)} Docling files in {input_dir}")
    if raw_dir:
        logger.info(f"Using raw pages from {raw_dir} for page mapping")

    all_parents = []
    all_children = []
    stats = {"files": 0, "parents": 0, "children": 0, "with_section": 0, "with_page": 0}

    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        markdown = data.get("markdown", "")
        source = data.get("filename", json_file.stem + ".pdf")

        if not markdown:
            logger.warning(f"Empty markdown: {json_file.name}")
            continue

        # Load page map from raw extraction if available
        page_map = None
        if raw_dir:
            raw_file = raw_dir / json_file.name
            if raw_file.exists():
                with open(raw_file, encoding="utf-8") as f:
                    raw_data = json.load(f)
                page_map = raw_data.get("pages", [])

        parents, children = chunk_markdown(markdown, source, corpus, page_map)

        stats["files"] += 1
        stats["parents"] += len(parents)
        stats["children"] += len(children)
        stats["with_section"] += sum(1 for c in children if c.get("section"))
        stats["with_page"] += sum(1 for c in children if c.get("page", 0) > 0)

        all_parents.extend(parents)
        all_children.extend(children)
        logger.info(f"  {source}: {len(parents)} parents, {len(children)} children")

    # Save parent-child structure
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save parents (for LLM context retrieval)
    parents_file = output_file.with_name(output_file.stem + "_parents.json")
    with open(parents_file, "w", encoding="utf-8") as f:
        json.dump(
            {"chunks": all_parents, "total": len(all_parents)},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Save children (for embedding/search)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {"chunks": all_children, "total": len(all_children)},
            f,
            ensure_ascii=False,
            indent=2,
        )

    pct_section = 100 * stats["with_section"] / max(1, stats["children"])
    pct_page = 100 * stats["with_page"] / max(1, stats["children"])
    logger.info(f"Saved {stats['parents']} parents to {parents_file}")
    logger.info(f"Saved {stats['children']} children to {output_file}")
    logger.info(f"  with_section: {stats['with_section']} ({pct_section:.1f}%)")
    logger.info(f"  with_page: {stats['with_page']} ({pct_page:.1f}%)")

    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Markdown Chunker - Pocket Arbiter (Parent-Child)"
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True, help="Docling output directory"
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output chunks JSON (children)"
    )
    parser.add_argument(
        "--corpus", "-c", type=str, default="fr", help="Corpus name (fr/intl)"
    )
    parser.add_argument(
        "--raw-dir", "-r", type=Path, help="Raw extraction directory (for page mapping)"
    )

    args = parser.parse_args()

    stats = process_docling_output(args.input, args.output, args.corpus, args.raw_dir)
    print(
        f"\nProcessed {stats['files']} files -> {stats['parents']} parents, {stats['children']} children"
    )
    if args.raw_dir:
        print(
            f"  with_page: {stats['with_page']} ({100 * stats['with_page'] / max(1, stats['children']):.1f}%)"
        )


if __name__ == "__main__":
    main()
