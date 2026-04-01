"""LangChain-based chunker: 6-stage pipeline. CHUNK_SIZE=450, 11% overlap."""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from scripts.pipeline.chunker_utils import (
    ARTICLE_NUM_RE,
    CHUNK_SIZE,
    build_cch_title,
    build_parents,
    count_tokens,
    extract_tables,
    interpolate_pages,
    link_tables,
    merge_small_children,
)

CHUNK_OVERLAP = 50  # 11% overlap (tested: 0/50/100 on sample, 50 = best compromise)

_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

_md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=_HEADERS_TO_SPLIT_ON,
    strip_headers=False,
)

_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".\n", " ", ""],
)


def header_split(markdown: str) -> list[Document]:
    """Split markdown by headers, preserving hierarchy in metadata."""
    return _md_splitter.split_text(markdown)


def recursive_split(docs: list[Document]) -> list[Document]:
    """Split oversized sections by token count with overlap."""
    return _text_splitter.split_documents(docs)


def chunk_document(
    markdown: str,
    source: str,
    heading_pages: dict[str, int] | None = None,
    text_pages: list[tuple[str, int]] | None = None,
) -> dict:
    """Chunk a markdown document into children, parents, and tables.

    6-stage pipeline:
    1. Extract tables (before header split)
    2. Split by markdown headers
    3. Recursive split by tokens (450/50) + build children dicts with CCH
    4. Assemble parents (capped 2048 tok)
    5. Interpolate pages + merge small children
    6. Link tables to parent sections

    Args:
        markdown: Full markdown text of a document.
        source: PDF source filename.
        heading_pages: Optional heading -> page mapping for page interpolation.
        text_pages: Optional ordered (text[:80], page_no) from docling for
            dense page tracking (fixes off-by-1 gaps between headings).

    Returns:
        Dict with "children", "parents", "tables" lists.
    """
    _heading_pages = heading_pages or {}

    # Stage 1: Table extraction (BEFORE header split)
    clean_md, tables = extract_tables(markdown)

    # Stage 2: Header split
    header_docs = header_split(clean_md)

    # Stage 3: Recursive token split + build children dicts with CCH
    children_docs = recursive_split(header_docs)

    # Pre-filter: remove heading-only and placeholder-only Documents
    def _has_body(doc: Document) -> bool:
        text = re.sub(r"\s*<!-- TABLE_\d+ -->\s*", "", doc.page_content).strip()
        body = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE).strip()
        return bool(body)

    children_docs = [d for d in children_docs if _has_body(d)]

    # Stage 4: Parent assembly
    parents, child_to_parent = build_parents(children_docs, source)
    children = _build_children_dicts(children_docs, child_to_parent, source)

    # Stage 5: Page interpolation + merge
    children = interpolate_pages(children, _heading_pages, markdown, text_pages)
    children = merge_small_children(children)
    for idx, child in enumerate(children):
        child["id"] = f"{source}-c{idx:04d}"

    # Stage 6: Table linkage
    tables = link_tables(tables, header_docs, parents)
    _assign_table_metadata(tables, source, _heading_pages)

    return {"children": children, "parents": parents, "tables": tables}


def _build_children_dicts(
    docs: list[Document],
    child_to_parent: dict[int, str],
    source: str,
) -> list[dict]:
    """Convert LangChain Documents to children dicts."""
    children: list[dict] = []
    for i, doc in enumerate(docs):
        clean_text = re.sub(r"\s*<!-- TABLE_\d+ -->\s*", "\n", doc.page_content).strip()
        if not clean_text:
            continue
        art_match = ARTICLE_NUM_RE.match(doc.page_content)
        children.append(
            {
                "id": f"{source}-c{i:04d}",
                "text": clean_text,
                "parent_id": child_to_parent.get(i, ""),
                "source": source,
                "article_num": art_match.group(1).rstrip(".") if art_match else None,
                "section": build_cch_title(doc.metadata),
                "tokens": count_tokens(clean_text),
                "page": None,
            }
        )
    return children


def _assign_table_metadata(
    tables: list[dict],
    source: str,
    heading_pages: dict[str, int],
) -> None:
    """Set source and page on tables from heading_pages."""
    for table in tables:
        table["source"] = source
        table.setdefault("page", None)
        for heading_text, page in heading_pages.items():
            if heading_text in table.get("section", ""):
                table["page"] = page
                break
