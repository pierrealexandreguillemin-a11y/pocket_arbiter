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
            parents.append(
                {
                    "id": pid,
                    "text": full_text,
                    "source": source,
                    "section": section,
                    "tokens": tokens,
                }
            )
            for idx, _ in group:
                child_to_parent[idx] = pid
            counter += 1
        else:
            batch_indices: list[int] = []
            parts: list[str] = []
            part_tokens = 0
            for idx, c in group:
                c_tok = count_tokens(c.page_content)
                if part_tokens + c_tok > PARENT_MAX_TOKENS and parts:
                    pid = f"{source}-p{counter:03d}"
                    sub = "\n\n".join(parts)
                    parents.append(
                        {
                            "id": pid,
                            "text": sub,
                            "source": source,
                            "section": section,
                            "tokens": count_tokens(sub),
                        }
                    )
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
                parents.append(
                    {
                        "id": pid,
                        "text": sub,
                        "source": source,
                        "section": section,
                        "tokens": count_tokens(sub),
                    }
                )
                for bi in batch_indices:
                    child_to_parent[bi] = pid
                counter += 1

    return parents, child_to_parent


def _build_text_to_page(
    markdown: str,
    heading_pages: dict[str, int],
) -> dict[str, int]:
    """Build line-text → page mapping by walking markdown with heading tracking."""
    fallback = min(heading_pages.values()) if heading_pages else 1
    current_page = fallback
    text_page: dict[str, int] = {}
    for line in markdown.split("\n"):
        stripped = line.strip().lstrip("#").strip()
        if stripped in heading_pages:
            current_page = heading_pages[stripped]
        if stripped:
            text_page[stripped[:80]] = current_page
    return text_page


def interpolate_pages(
    children: list[dict],
    heading_pages: dict[str, int],
    markdown: str = "",
) -> list[dict]:
    """Assign pages via line-level tracking from markdown source."""
    if not heading_pages:
        for child in children:
            if child.get("page") is None:
                child["page"] = 1
        return children
    text_page = _build_text_to_page(markdown, heading_pages) if markdown else {}
    fallback = min(heading_pages.values())
    for child in children:
        if child.get("page") is not None:
            continue
        assigned = False
        for line in child.get("text", "").split("\n"):
            key = line.strip().lstrip("#").strip()[:80]
            if key and key in text_page:
                child["page"] = text_page[key]
                assigned = True
                break
        if not assigned:
            child["page"] = fallback
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
