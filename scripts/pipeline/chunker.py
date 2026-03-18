"""LangChain-based chunker: 7-stage pipeline (table extract, header/token split,
parent assembly, page interp, CCH, table linkage). FloTorch 512, Azure 20% overlap."""

from __future__ import annotations

import re
from collections import defaultdict

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
PAGE_MARKER_RE = re.compile(r"\n\s*[A-Z][A-Z0-9]{1,3}-\d+/\d+\s*\n")
IMAGE_PLACEHOLDER = "<!-- image -->"
ARTICLE_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*\.?)\s")

CHUNK_SIZE = 512  # FloTorch 2026 benchmark optimal
CHUNK_OVERLAP = 100  # 20% overlap, Microsoft Azure 2026 standard
PARENT_MAX_TOKENS = 2048  # EmbeddingGemma max seq + LLM mobile budget
TABLE_MIN_LINES = 3  # header + separator + at least 1 data row

_enc = tiktoken.get_encoding("cl100k_base")

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


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base."""
    return len(_enc.encode(text))


def extract_tables(markdown: str) -> tuple[str, list[dict]]:
    """Extract table blocks from markdown, replace with placeholders.

    Must be called BEFORE header splitting — MarkdownHeaderTextSplitter
    fragments tables across heading boundaries.

    Args:
        markdown: Raw markdown text.

    Returns:
        (clean_markdown, tables) where tables is list of {"raw_text": ...}.
    """
    markdown = PAGE_MARKER_RE.sub("\n", markdown)
    markdown = markdown.replace(IMAGE_PLACEHOLDER, "")

    lines = markdown.split("\n")
    tables: list[dict] = []
    clean_lines: list[str] = []
    i = 0
    while i < len(lines):
        if TABLE_LINE_RE.match(lines[i]):
            table_lines: list[str] = []
            while i < len(lines) and TABLE_LINE_RE.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            if len(table_lines) >= TABLE_MIN_LINES:
                tables.append({"raw_text": "\n".join(table_lines)})
                clean_lines.append(f"<!-- TABLE_{len(tables) - 1} -->")
            else:
                clean_lines.extend(table_lines)
        else:
            clean_lines.append(lines[i])
            i += 1
    return "\n".join(clean_lines), tables


def header_split(markdown: str) -> list[Document]:
    """Split markdown by headers, preserving hierarchy in metadata."""
    return _md_splitter.split_text(markdown)


def recursive_split(docs: list[Document]) -> list[Document]:
    """Split oversized sections by token count with overlap."""
    return _text_splitter.split_documents(docs)


def build_parents(children: list[Document], source: str) -> list[dict]:
    """Group children by (h1, h2), build parent dicts capped at 2048 tok."""
    groups: dict[tuple[str, str], list[Document]] = defaultdict(list)
    for child in children:
        key = (child.metadata.get("h1", ""), child.metadata.get("h2", ""))
        groups[key].append(child)

    parents: list[dict] = []
    counter = 0

    for (h1, h2), group in groups.items():
        section = " > ".join(p for p in (h1, h2) if p) or "root"
        full_text = "\n\n".join(c.page_content for c in group)
        tokens = count_tokens(full_text)

        if tokens <= PARENT_MAX_TOKENS:
            parents.append(
                {
                    "id": f"{source}-p{counter:03d}",
                    "text": full_text,
                    "source": source,
                    "section": section,
                    "tokens": tokens,
                }
            )
            counter += 1
        else:
            parts: list[str] = []
            part_tokens = 0
            for c in group:
                c_tok = count_tokens(c.page_content)
                if part_tokens + c_tok > PARENT_MAX_TOKENS and parts:
                    sub = "\n\n".join(parts)
                    parents.append(
                        {
                            "id": f"{source}-p{counter:03d}",
                            "text": sub,
                            "source": source,
                            "section": section,
                            "tokens": count_tokens(sub),
                        }
                    )
                    counter += 1
                    parts = []
                    part_tokens = 0
                parts.append(c.page_content)
                part_tokens += c_tok
            if parts:
                sub = "\n\n".join(parts)
                parents.append(
                    {
                        "id": f"{source}-p{counter:03d}",
                        "text": sub,
                        "source": source,
                        "section": section,
                        "tokens": count_tokens(sub),
                    }
                )
                counter += 1

    return parents


def interpolate_pages(
    children: list[dict],
    heading_pages: dict[str, int],
) -> list[dict]:
    """Assign page numbers from heading_pages mapping."""
    for child in children:
        if child.get("page") is not None:
            continue
        section = child.get("section", "")
        for heading_text, page in heading_pages.items():
            if heading_text in section:
                child["page"] = page
                break
    return children


def build_cch_title(metadata: dict) -> str:
    """Build CCH title from heading hierarchy metadata."""
    parts = [metadata.get(f"h{i}", "") for i in range(1, 5)]
    return " > ".join(p for p in parts if p)


def link_tables(
    tables: list[dict],
    header_docs: list[Document],
    parents: list[dict],
) -> list[dict]:
    """Link tables to parent sections via placeholder positions."""
    parent_by_section: dict[str, str] = {p["section"]: p["id"] for p in parents}

    for i, table in enumerate(tables):
        placeholder = f"<!-- TABLE_{i} -->"
        table.setdefault("section", "")
        table.setdefault("parent_id", "")
        for doc in header_docs:
            if placeholder in doc.page_content:
                table["section"] = build_cch_title(doc.metadata)
                key = " > ".join(
                    p
                    for p in (
                        doc.metadata.get("h1", ""),
                        doc.metadata.get("h2", ""),
                    )
                    if p
                )
                table["parent_id"] = parent_by_section.get(
                    key, parent_by_section.get(table["section"], "")
                )
                break
    return tables


def chunk_document(
    markdown: str,
    source: str,
    heading_pages: dict[str, int] | None = None,
) -> dict:
    """Chunk a markdown document into children, parents, and tables.

    Args:
        markdown: Markdown text with heading levels.
        source: Source PDF filename.
        heading_pages: Optional mapping heading text -> page number.

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

    # Stage 4: Parent assembly
    parents = build_parents(children_docs, source)
    pid_by_section: dict[str, str] = {p["section"]: p["id"] for p in parents}

    # Build children dicts
    children: list[dict] = []
    for i, doc in enumerate(children_docs):
        cch = build_cch_title(doc.metadata)
        section_key = (
            " > ".join(
                p for p in (doc.metadata.get("h1", ""), doc.metadata.get("h2", "")) if p
            )
            or "root"
        )
        pid = pid_by_section.get(section_key, "")

        art_match = ARTICLE_NUM_RE.match(doc.page_content)
        art_num = art_match.group(1).rstrip(".") if art_match else None

        # Strip table placeholders from child text (tables extracted in Stage 1)
        clean_text = re.sub(r"\s*<!-- TABLE_\d+ -->\s*", "\n", doc.page_content).strip()
        if not clean_text:
            continue  # Skip empty children (was only a placeholder)

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

    # Stage 5: Page interpolation
    children = interpolate_pages(children, _heading_pages)

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
