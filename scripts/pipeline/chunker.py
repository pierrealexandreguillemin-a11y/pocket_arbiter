"""LangChain-based chunker: 7-stage pipeline. FloTorch 512, Azure 20% overlap."""

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
) -> dict:
    """Chunk a markdown document into children, parents, and tables.

    7-stage pipeline:
    1. Extract tables (before header split)
    2. Split by markdown headers
    3. Recursive split by tokens (512/100)
    4. Assemble parents (capped 2048 tok)
    5. Interpolate pages + merge small children
    6. Build CCH titles
    7. Link tables to parent sections

    Returns:
        Dict with "children", "parents", "tables" lists.
    """
    _heading_pages = heading_pages or {}

    # Stage 1: Table extraction (BEFORE header split)
    clean_md, tables = extract_tables(markdown)

    # Stage 2: Header split
    header_docs = header_split(clean_md)

    # Stage 3: Recursive token split
    children_docs = recursive_split(header_docs)

    # Pre-filter: remove heading-only and placeholder-only Documents
    def _has_body(doc: Document) -> bool:
        text = re.sub(r"\s*<!-- TABLE_\d+ -->\s*", "", doc.page_content).strip()
        body = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE).strip()
        return bool(body)

    children_docs = [d for d in children_docs if _has_body(d)]

    # Stage 4: Parent assembly
    parents, child_to_parent = build_parents(children_docs, source)

    # Build children dicts
    children: list[dict] = []
    for i, doc in enumerate(children_docs):
        cch = build_cch_title(doc.metadata)
        pid = child_to_parent.get(i, "")
        art_match = ARTICLE_NUM_RE.match(doc.page_content)
        art_num = art_match.group(1).rstrip(".") if art_match else None
        clean_text = re.sub(r"\s*<!-- TABLE_\d+ -->\s*", "\n", doc.page_content).strip()
        if not clean_text:
            continue
        children.append(
            {
                "id": f"{source}-c{i:04d}",
                "text": clean_text,
                "parent_id": pid,
                "source": source,
                "article_num": art_num,
                "section": cch,
                "tokens": count_tokens(clean_text),
                "page": None,
            }
        )

    # Stage 5: Page interpolation + merge
    children = interpolate_pages(children, _heading_pages, markdown)
    children = merge_small_children(children)
    for idx, child in enumerate(children):
        child["id"] = f"{source}-c{idx:04d}"

    # Stage 7: Table linkage
    tables = link_tables(tables, header_docs, parents)
    for table in tables:
        table["source"] = source
        table.setdefault("page", None)
        for heading_text, page in _heading_pages.items():
            if heading_text in table.get("section", ""):
                table["page"] = page
                break

    return {"children": children, "parents": parents, "tables": tables}
