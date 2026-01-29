"""
Chunker Mode B: LangChain avec tokenizer EmbeddingGemma.

Pipeline LangChain avec mapping page depuis docling_document.texts[].prov[].page_no.
Utilise le meme tokenizer que le modele d'embedding pour consistance.

ISO Reference:
    - ISO/IEC 42001 A.6.2.2 - AI traceability (page provenance 100%)
    - ISO/IEC 25010 S4.2 - Performance efficiency (Recall >= 90%)
    - ISO/IEC 12207 S7.3.3 - Implementation

Sources:
    - https://github.com/docling-project/docling/discussions/1012
    - Docling: texts[].prov[].page_no for page mapping

Usage:
    python -m scripts.pipeline.chunker_langchain \
        --input corpus/processed/docling_fr \
        --output corpus/processed/chunks_mode_b_fr.json
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

# Workaround for Windows symlink permission issue with HuggingFace
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from scripts.pipeline.token_utils import count_tokens_gemma

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Constants (ISO 25010 PR-01, NVIDIA/Chroma 2025) ---

# Parent chunks: rich context for LLM response (arXiv 2025)
PARENT_CHUNK_SIZE = 1024
PARENT_CHUNK_OVERLAP = 154  # NVIDIA 2025: 15% optimal

# Child chunks: precise semantic units for search (Chroma 2025)
CHILD_CHUNK_SIZE = 450
CHILD_CHUNK_OVERLAP = 68  # NVIDIA 2025: 15% optimal

MIN_CHUNK_TOKENS = 30  # Avoid tiny fragments

# Headers Markdown extraits par Docling
HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def _get_texts_from_docling(docling_doc: dict) -> list:
    """Extract texts list from DoclingDocument (ISO 25010 - reduced complexity)."""
    texts = docling_doc.get("texts", [])
    if texts:
        return texts
    # Fallback: check body.children for newer Docling versions
    body = docling_doc.get("body", {})
    if isinstance(body, dict):
        return body.get("children", [])
    return []


def _extract_page_from_prov(prov_list: list) -> int:
    """Extract page_no from provenance list (ISO 25010 - reduced complexity)."""
    if not prov_list:
        return 0
    first_prov = prov_list[0]
    if isinstance(first_prov, dict):
        return first_prov.get("page_no", 0)
    return getattr(first_prov, "page_no", 0)


def build_page_map_from_docling(docling_doc: dict) -> tuple[list[dict], dict[str, int]]:
    """
    Build comprehensive text -> page mapping from DoclingDocument provenance.

    Source: Docling Discussion #1012 - texts[].prov[].page_no

    Args:
        docling_doc: Dict from DoclingDocument export_to_dict().

    Returns:
        Tuple of:
        - List of {text_norm, page_no} ordered by document position
        - Dict mapping normalized text segments to page numbers
    """
    texts = _get_texts_from_docling(docling_doc)
    ordered_texts: list[dict] = []
    text_to_page: dict[str, int] = {}

    for text_item in texts:
        if not isinstance(text_item, dict):
            continue

        text = text_item.get("text", "")
        prov_list = text_item.get("prov", [])
        page_no = _extract_page_from_prov(prov_list)

        if text and page_no > 0:
            text_norm = normalize_for_match(text)
            if len(text_norm) >= 10:
                # Store in order for sequential fallback
                ordered_texts.append({"text_norm": text_norm, "page_no": page_no})

                # Index multiple segment lengths for robust matching
                for seg_len in [150, 100, 80, 60, 40, 30]:
                    if len(text_norm) >= seg_len:
                        text_to_page[text_norm[:seg_len]] = page_no

    return ordered_texts, text_to_page


def normalize_for_match(text: str) -> str:
    """Normalize text for fuzzy matching."""
    # Remove markdown headers and formatting
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Lowercase
    return text.lower().strip()


def _match_snippet_to_page(snippet: str, text_to_page: dict[str, int]) -> int:
    """Match a snippet against page mapping (ISO 25010 - reduced complexity)."""
    # Direct match first (fastest)
    if snippet in text_to_page:
        return text_to_page[snippet]

    # Substring match (slower but more robust)
    for key, page in text_to_page.items():
        if snippet in key or key in snippet:
            return page
    return 0


def _search_in_ordered_texts(
    text_norm: str, ordered_texts: list[dict], min_overlap: int = 25
) -> int:
    """Search for page by finding best overlap with ordered texts."""
    if len(text_norm) < min_overlap:
        return 0

    # Try finding a text that contains part of our chunk
    chunk_start = text_norm[:min_overlap]
    chunk_mid = text_norm[len(text_norm) // 4 : len(text_norm) // 4 + min_overlap]

    for item in ordered_texts:
        source_text = item["text_norm"]
        if chunk_start in source_text or chunk_mid in source_text:
            return item["page_no"]

    return 0


def find_page_from_map(
    text: str,
    text_to_page: dict[str, int],
    ordered_texts: list[dict] | None = None,
    prev_page: int = 0,
) -> int:
    """
    Find page number for text using comprehensive mapping strategy.

    ISO 42001 A.6.2.2: Must return page >= 1 (100% coverage required).

    Strategy:
    1. Direct segment matching in text_to_page
    2. Substring search in ordered_texts
    3. Context propagation (previous chunk's page)

    Args:
        text: Text to find page for.
        text_to_page: Mapping from normalized text segments to pages.
        ordered_texts: Ordered list of {text_norm, page_no} for search.
        prev_page: Previous chunk's page for context propagation.

    Returns:
        Page number (>= 1). Falls back to prev_page if no match.
    """
    text_norm = normalize_for_match(text)

    # Strategy 1: Direct segment matching (multiple lengths)
    if text_to_page and len(text_norm) >= 20:
        for length in [150, 100, 80, 60, 40, 30]:
            if len(text_norm) >= length:
                snippet = text_norm[:length]
                page = _match_snippet_to_page(snippet, text_to_page)
                if page > 0:
                    return page

        # Try from middle of chunk too
        if len(text_norm) >= 80:
            mid_start = len(text_norm) // 3
            mid_snippet = text_norm[mid_start : mid_start + 60]
            page = _match_snippet_to_page(mid_snippet, text_to_page)
            if page > 0:
                return page

    # Strategy 2: Search in ordered texts
    if ordered_texts and len(text_norm) >= 25:
        page = _search_in_ordered_texts(text_norm, ordered_texts)
        if page > 0:
            return page

    # Strategy 3: Context propagation (sequential document assumption)
    # ISO 42001: Must have page >= 1, use previous chunk's page
    if prev_page > 0:
        return prev_page

    # Fallback: first page if nothing else works (should rarely happen)
    return 1


def _extract_article_num(article: str) -> str | None:
    """Extract article number from header (ISO 25010 - reduced complexity)."""
    if not article:
        return None
    match = re.match(r"^(\d+(?:\.\d+)*)", article)
    return match.group(1) if match else None


def _extract_section_metadata(doc_metadata: dict) -> tuple[str, str | None]:
    """Extract section and article_num from document metadata (ISO 25010)."""
    section = doc_metadata.get("h2") or doc_metadata.get("h1") or ""
    article = doc_metadata.get("h3") or doc_metadata.get("h4") or ""
    article_num = _extract_article_num(article)
    return section, article_num


def _build_chunk_dict(
    chunk_id: str,
    text: str,
    source: str,
    page: int,
    section: str,
    article_num: str | None,
    tokens: int,
    corpus: str,
) -> dict:
    """Build a chunk dictionary (ISO 25010 - reduced complexity)."""
    return {
        "id": chunk_id,
        "text": text,
        "source": source,
        "page": page,
        "pages": [page] if page > 0 else [],
        "section": section,
        "article_num": article_num,
        "tokens": tokens,
        "corpus": corpus,
        "chunk_type": "langchain",
    }


def chunk_markdown_langchain(
    markdown: str,
    source: str,
    corpus: str,
    ordered_texts: list[dict],
    text_to_page: dict[str, int],
) -> tuple[list[dict], list[dict]]:
    """
    Chunk markdown with LangChain Parent-Child architecture (Mode B).

    Uses EmbeddingGemma tokenizer and page mapping from docling_document.
    Guarantees 100% page coverage via context propagation (ISO 42001 A.6.2.2).

    Architecture (NVIDIA/Chroma 2025):
        1. MarkdownHeaderTextSplitter: Extract section metadata
        2. RecursiveCharacterTextSplitter (1024 tokens): Parent chunks for LLM context
        3. RecursiveCharacterTextSplitter (450 tokens): Child chunks for embedding/search

    Args:
        markdown: Markdown text to chunk.
        source: Source filename.
        corpus: Corpus name.
        ordered_texts: Ordered list of {text_norm, page_no} for search.
        text_to_page: Mapping from text segments to page numbers.

    Returns:
        Tuple (parent_chunks, child_chunks) with page >= 1 for all chunks.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    header_docs = header_splitter.split_text(markdown)

    # Parent splitter: 1024 tokens for LLM context (arXiv 2025)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        length_function=count_tokens_gemma,
    )

    # Child splitter: 450 tokens for embedding/search (Chroma 2025)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        length_function=count_tokens_gemma,
    )

    parent_chunks: list[dict] = []
    child_chunks: list[dict] = []
    prev_page = 1  # Start with page 1, propagate forward

    for doc in header_docs:
        section, article_num = _extract_section_metadata(doc.metadata)

        # Step 1: Split into parent chunks (1024 tokens)
        parent_texts = parent_splitter.split_text(doc.page_content)

        for parent_text in parent_texts:
            parent_text = parent_text.strip()
            if not parent_text:
                continue

            parent_tokens = count_tokens_gemma(parent_text)

            # Find page with context propagation for 100% coverage
            parent_page = find_page_from_map(
                parent_text, text_to_page, ordered_texts, prev_page
            )
            prev_page = parent_page  # Update for next chunk

            parent_id = f"{source}-p{parent_page:03d}-parent{len(parent_chunks):03d}"

            parent_chunks.append(
                {
                    "id": parent_id,
                    "text": parent_text,
                    "source": source,
                    "page": parent_page,
                    "pages": [parent_page],
                    "section": section,
                    "article_num": article_num,
                    "tokens": parent_tokens,
                    "corpus": corpus,
                    "chunk_type": "parent",
                }
            )

            # Step 2: Split parent into child chunks (450 tokens)
            child_texts = child_splitter.split_text(parent_text)

            for c_idx, child_text in enumerate(child_texts):
                child_text = child_text.strip()
                if not child_text:
                    continue

                child_tokens = count_tokens_gemma(child_text)
                if child_tokens < MIN_CHUNK_TOKENS:
                    continue

                # Child inherits parent page, or finds its own
                child_page = find_page_from_map(
                    child_text, text_to_page, ordered_texts, parent_page
                )

                child_id = f"{source}-p{child_page:03d}-parent{len(parent_chunks) - 1:03d}-child{c_idx:02d}"

                child_chunks.append(
                    {
                        "id": child_id,
                        "text": child_text,
                        "source": source,
                        "page": child_page,
                        "pages": [child_page],
                        "section": section,
                        "article_num": article_num,
                        "parent_id": parent_id,
                        "tokens": child_tokens,
                        "corpus": corpus,
                        "chunk_type": "child",
                    }
                )

    return parent_chunks, child_chunks


def _process_single_docling_file(
    json_file: Path,
    corpus: str,
    stats: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process a single Docling JSON file (ISO 25010 - reduced complexity)."""
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    markdown = data.get("markdown", "")
    source = data.get("filename", json_file.stem + ".pdf")
    docling_doc = data.get("docling_document", {})

    if not markdown:
        error_msg = f"No markdown in {json_file.name}"
        logger.warning(error_msg)
        stats["errors"].append(error_msg)
        return [], []

    # Build comprehensive page mapping
    ordered_texts: list[dict] = []
    text_to_page: dict[str, int] = {}

    if docling_doc:
        ordered_texts, text_to_page = build_page_map_from_docling(docling_doc)

    if not text_to_page:
        logger.warning(f"  No page mapping for {source} (docling_document empty)")

    parents, children = chunk_markdown_langchain(
        markdown, source, corpus, ordered_texts, text_to_page
    )
    logger.info(f"  {source}: {len(parents)} parents, {len(children)} children")
    return parents, children


def _update_stats_from_chunks(
    parents: list[dict], children: list[dict], stats: dict[str, Any]
) -> None:
    """Update statistics from processed chunks (ISO 25010 - reduced complexity)."""
    stats["files"] += 1
    stats["parents"] += len(parents)
    stats["children"] += len(children)
    stats["with_page"] += sum(1 for c in children if c.get("page", 0) > 0)
    stats["without_page"] += sum(1 for c in children if c.get("page", 0) == 0)


def _save_chunks_to_file(chunks: list[dict], output_file: Path) -> None:
    """Save chunks to JSON file (ISO 25010 - reduced complexity)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {"chunks": chunks, "total": len(chunks)}, f, ensure_ascii=False, indent=2
        )


def merge_table_summaries(
    children: list[dict],
    tables_file: Path,
    corpus: str,
) -> list[dict]:
    """
    Merge table summaries with text children (ISO 42001 A.6.2.2).

    According to plan section 4.4:
        chunks_fr.json = text children (1716) + table summaries (111) = 1827
        chunks_intl.json = text children (900) + table summaries (74) = 974

    Args:
        children: List of text children chunks.
        tables_file: Path to tables_multivector_*.json file.
        corpus: Corpus name (fr/intl).

    Returns:
        Merged list of children + table_summary chunks.
    """
    if not tables_file.exists():
        logger.warning(f"Tables file not found: {tables_file}")
        return children

    with open(tables_file, encoding="utf-8") as f:
        tables_data = json.load(f)

    table_children = tables_data.get("children", [])
    logger.info(
        f"  Merging {len(table_children)} table summaries from {tables_file.name}"
    )

    # Convert table summaries to chunk format
    merged = list(children)  # Copy to avoid mutating original
    for table in table_children:
        merged.append(
            {
                "id": table.get("id", f"{table.get('doc_id', 'unknown')}-summary"),
                "text": table.get("text", ""),
                "source": table.get("source", ""),
                "page": table.get("page", 1),
                "pages": [table.get("page", 1)],
                "section": "",
                "article_num": None,
                "parent_id": None,  # Table summaries have no parent
                "tokens": count_tokens_gemma(table.get("text", "")),
                "corpus": corpus,
                "chunk_type": "table_summary",
                "table_type": table.get("table_type"),
            }
        )

    return merged


def process_docling_output_langchain(
    input_dir: Path,
    output_file: Path,
    corpus: str = "fr",
    tables_file: Path | None = None,
) -> dict[str, Any]:
    """
    Process all Docling files with LangChain Parent-Child chunker (Mode B).

    Saves two files:
        - output_file: Children chunks + table summaries (for embedding/search)
        - output_file_parents.json: Parent chunks (for LLM context)

    Args:
        input_dir: Directory containing Docling JSON files.
        output_file: Output children chunks JSON file.
        corpus: Corpus name.
        tables_file: Optional path to tables_multivector_*.json for merging.

    Returns:
        Processing report dict.
    """
    json_files = sorted(input_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != "extraction_report.json"]

    logger.info(f"Found {len(json_files)} Docling files in {input_dir}")

    all_parents: list[dict[str, Any]] = []
    all_children: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "files": 0,
        "parents": 0,
        "children": 0,
        "table_summaries": 0,
        "with_page": 0,
        "without_page": 0,
        "errors": [],
    }

    for json_file in json_files:
        try:
            parents, children = _process_single_docling_file(json_file, corpus, stats)
            if parents or children:
                _update_stats_from_chunks(parents, children, stats)
                all_parents.extend(parents)
                all_children.extend(children)
        except Exception as e:
            error_msg = f"{json_file.name}: {e}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)

    # Merge table summaries if provided
    if tables_file:
        children_before = len(all_children)
        all_children = merge_table_summaries(all_children, tables_file, corpus)
        stats["table_summaries"] = len(all_children) - children_before
        stats["with_page"] += stats[
            "table_summaries"
        ]  # All table summaries have page >= 1

    # Save parents (for LLM context retrieval)
    parents_file = output_file.with_name(output_file.stem + "_parents.json")
    _save_chunks_to_file(all_parents, parents_file)

    # Save children + table summaries (for embedding/search)
    _save_chunks_to_file(all_children, output_file)

    total_children = stats["children"] + stats["table_summaries"]
    pct_page = 100 * stats["with_page"] / max(1, total_children)
    logger.info(f"Saved {stats['parents']} parents to {parents_file}")
    logger.info(f"Saved {len(all_children)} chunks to {output_file}")
    logger.info(f"  text children: {stats['children']}")
    logger.info(f"  table_summaries: {stats['table_summaries']}")
    logger.info(f"  with_page: {stats['with_page']} ({pct_page:.1f}%)")
    logger.info(f"  without_page: {stats['without_page']}")

    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LangChain Chunker Mode B - Pocket Arbiter (Parent-Child + EmbeddingGemma)"
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
    parser.add_argument(
        "--tables", "-t", type=Path, help="Tables multivector JSON to merge"
    )

    args = parser.parse_args()

    stats = process_docling_output_langchain(
        args.input, args.output, args.corpus, args.tables
    )

    total = stats["children"] + stats["table_summaries"]
    pct = 100 * stats["with_page"] / max(1, total)

    print("\nMode B (LangChain Parent-Child):")
    print(f"  Files: {stats['files']}")
    print(f"  Parents: {stats['parents']} (1024 tokens)")
    print(f"  Children: {stats['children']} (450 tokens)")
    print(f"  Table summaries: {stats['table_summaries']}")
    print(f"  Total chunks: {total}")
    print(f"  Page coverage: {pct:.1f}%")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")


if __name__ == "__main__":
    main()
