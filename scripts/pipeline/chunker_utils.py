"""Chunker utilities: table extraction, parent assembly, page interpolation,
CCH titles, table linkage, small chunk merge."""

from __future__ import annotations

import re
from collections import defaultdict

import tiktoken
from langchain_core.documents import Document

TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
PAGE_MARKER_RE = re.compile(r"\n\s*[A-Z][A-Z0-9]{1,3}-\d+/\d+\s*\n")
IMAGE_PLACEHOLDER = "<!-- image -->"
ARTICLE_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*\.?)\s")

CHUNK_SIZE = 450  # Firecrawl 2026 + jan benchmark (86.94% at 450t)
PARENT_MAX_TOKENS = 2048
TABLE_MIN_LINES = 3
MERGE_THRESHOLD = 200

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base."""
    return len(_enc.encode(text))


def extract_tables(markdown: str) -> tuple[str, list[dict]]:
    """Extract table blocks, replace with placeholders. BEFORE header split."""
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


def build_parents(
    children: list[Document],
    source: str,
) -> tuple[list[dict], dict[int, str]]:
    """Group children by (h1, h2), build parents capped at PARENT_MAX_TOKENS."""
    groups: dict[tuple[str, str], list[tuple[int, Document]]] = defaultdict(list)
    for idx, child in enumerate(children):
        key = (child.metadata.get("h1", ""), child.metadata.get("h2", ""))
        groups[key].append((idx, child))

    parents: list[dict] = []
    child_to_parent: dict[int, str] = {}
    counter = 0

    for (h1, h2), group in groups.items():
        section = " > ".join(p for p in (h1, h2) if p) or "root"
        full_text = "\n\n".join(c.page_content for _, c in group)
        tokens = count_tokens(full_text)

        if tokens <= PARENT_MAX_TOKENS:
            pid = f"{source}-p{counter:03d}"
            parents.append(_make_parent(pid, full_text, source, section, tokens))
            for idx, _ in group:
                child_to_parent[idx] = pid
            counter += 1
        else:
            counter = _split_oversized_group(
                group, source, section, counter, parents, child_to_parent
            )

    return parents, child_to_parent


def _make_parent(pid: str, text: str, source: str, section: str, tokens: int) -> dict:
    """Create a parent dict."""
    return {
        "id": pid,
        "text": text,
        "source": source,
        "section": section,
        "tokens": tokens,
        "page": None,
    }


def _split_oversized_group(
    group: list[tuple[int, Document]],
    source: str,
    section: str,
    counter: int,
    parents: list[dict],
    child_to_parent: dict[int, str],
) -> int:
    """Split a group exceeding PARENT_MAX_TOKENS into sub-parents."""
    batch_indices: list[int] = []
    parts: list[str] = []
    part_tokens = 0
    for idx, c in group:
        c_tok = count_tokens(c.page_content)
        if part_tokens + c_tok > PARENT_MAX_TOKENS and parts:
            pid = f"{source}-p{counter:03d}"
            sub = "\n\n".join(parts)
            parents.append(_make_parent(pid, sub, source, section, count_tokens(sub)))
            for bi in batch_indices:
                child_to_parent[bi] = pid
            counter += 1
            parts, batch_indices, part_tokens = [], [], 0
        parts.append(c.page_content)
        batch_indices.append(idx)
        part_tokens += c_tok
    if parts:
        pid = f"{source}-p{counter:03d}"
        sub = "\n\n".join(parts)
        parents.append(_make_parent(pid, sub, source, section, count_tokens(sub)))
        for bi in batch_indices:
            child_to_parent[bi] = pid
        counter += 1
    return counter


def _build_text_to_page(
    markdown: str,
    heading_pages: dict[str, int],
    text_pages: list[tuple[str, int]] | None = None,
) -> dict[str, int]:
    """Build line-text → page mapping by walking markdown with heading tracking.

    Uses two sources for page tracking:
    1. heading_pages (sparse): heading text → page from docling section_headers.
    2. text_pages (dense): ordered (text[:80], page_no) from ALL docling text
       items, enabling page boundary detection between headings.

    Without text_pages, pages between two headings inherit the previous heading's
    page (off-by-1 bug). With text_pages, each line's page is resolved from the
    nearest matching docling item.
    """
    fallback = min(heading_pages.values()) if heading_pages else 1
    current_page = fallback
    text_page: dict[str, int] = {}

    # Build dense lookup from text_pages: text[:80] → page.
    # For repeated text (page headers), build an ordered list of pages so we
    # can consume them sequentially as we walk the markdown.
    tp_index: dict[str, list[int]] = {}
    if text_pages:
        for text_snippet, page_no in text_pages:
            key = text_snippet.strip().lstrip("#").strip()[:80]
            if key:
                tp_index.setdefault(key, []).append(page_no)
    # Track consumption position for repeated keys
    tp_consumed: dict[str, int] = {}

    for line in markdown.split("\n"):
        stripped = line.strip().lstrip("#").strip()
        # Priority 1: heading_pages (most reliable, from docling provenance)
        if stripped in heading_pages:
            current_page = heading_pages[stripped]
        # Priority 2: text_pages dense tracking (fixes off-by-1 between headings)
        elif stripped[:80] in tp_index:
            key = stripped[:80]
            idx = tp_consumed.get(key, 0)
            pages_list = tp_index[key]
            if idx < len(pages_list):
                current_page = pages_list[idx]
                tp_consumed[key] = idx + 1
        if stripped:
            text_page[stripped[:80]] = current_page
    return text_page


def interpolate_pages(
    children: list[dict],
    heading_pages: dict[str, int],
    markdown: str = "",
    text_pages: list[tuple[str, int]] | None = None,
) -> list[dict]:
    """Assign pages via line-level tracking from markdown source.

    Args:
        children: Children dicts with page=None to fill.
        heading_pages: Heading text → page (sparse, from docling headers).
        markdown: Full markdown text for line-level tracking.
        text_pages: Ordered (text[:80], page_no) from ALL docling text items
            for dense page tracking. Fixes off-by-1 gaps between headings.
    """
    if not heading_pages:
        for child in children:
            if child.get("page") is None:
                child["page"] = 1
        return children
    text_page = (
        _build_text_to_page(markdown, heading_pages, text_pages) if markdown else {}
    )
    fallback = min(heading_pages.values())
    for child in children:
        if child.get("page") is None:
            child["page"] = _match_page(child, text_page, fallback)
    return children


def _match_page(child: dict, text_page: dict[str, int], fallback: int) -> int:
    """Find page for a child by matching its lines against text_page map."""
    for line in child.get("text", "").split("\n"):
        key = line.strip().lstrip("#").strip()[:80]
        if key and key in text_page:
            return text_page[key]
    return fallback


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


def merge_small_children(children: list[dict]) -> list[dict]:
    """Merge consecutive small children under same parent."""
    if not children:
        return children
    merged: list[dict] = [children[0]]
    for child in children[1:]:
        prev = merged[-1]
        if (
            prev["parent_id"] == child["parent_id"]
            and prev["tokens"] < MERGE_THRESHOLD
            and child["tokens"] < MERGE_THRESHOLD
            and prev["tokens"] + child["tokens"] <= CHUNK_SIZE
        ):
            prev["text"] = prev["text"] + "\n\n" + child["text"]
            prev["tokens"] = count_tokens(prev["text"])
            if child.get("section") and child["section"] != prev.get("section"):
                prev["section"] = prev["section"] + " + " + child["section"]
        else:
            merged.append(child)
    return merged
