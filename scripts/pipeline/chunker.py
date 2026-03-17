"""Structure-aware chunker for FFE regulation markdown.

Parses markdown with heading levels, builds parent-child hierarchy,
and produces chunks of 400-512 tokens respecting article boundaries.
"""

from __future__ import annotations

import re

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")

PAGE_MARKER_RE = re.compile(r"\n\s*[A-Z][A-Z0-9]{1,3}-\d+/\d+\s*\n")
HEADING_RE = re.compile(r"^(#{1,})\s+(.+)$", re.MULTILINE)
ARTICLE_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*\.?)\s")
IMAGE_PLACEHOLDER = "<!-- image -->"

MAX_TOKENS = 512
HARD_MAX_TOKENS = 2048  # EmbeddingGemma max sequence length — never exceed
MERGE_THRESHOLD = 250
SPLIT_TARGET = 450
SPLIT_OVERLAP = 50


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base."""
    return len(_enc.encode(text))


def parse_sections(markdown: str) -> list[dict]:
    """Parse markdown into sections based on headings.

    Returns list of dicts with: heading, level, body.
    """
    # Strip page markers and image placeholders
    markdown = PAGE_MARKER_RE.sub("\n", markdown)
    markdown = markdown.replace(IMAGE_PLACEHOLDER, "")

    sections: list[dict] = []
    matches = list(HEADING_RE.finditer(markdown))

    for i, match in enumerate(matches):
        hashes = match.group(1)
        heading = match.group(2).strip()
        level = min(len(hashes), 6)  # cap at h6; docling can produce h7+
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[body_start:body_end].strip()

        sections.append(
            {
                "heading": heading,
                "level": level,
                "body": body,
            }
        )

    return sections


def _infer_level_from_numbering(heading: str) -> int | None:
    """Infer heading level from article numbering (fallback for flat ##).

    "2" -> depth 1 (level 2), "2.1" -> depth 2 (level 3), "2.1.1" -> depth 3 (level 4).
    """
    m = ARTICLE_NUM_RE.match(heading)
    if not m:
        return None
    num = m.group(1).rstrip(".")
    depth = num.count(".") + 1
    return depth + 1  # offset: top articles = level 2


def build_hierarchy(sections: list[dict]) -> list[dict]:
    """Build parent-child hierarchy from sections.

    Uses heading levels if multi-level markdown available.
    Falls back to article numbering if all levels are the same.
    """
    if not sections:
        return []

    # Check if we have real heading levels
    levels = {s["level"] for s in sections}
    all_flat = len(levels) <= 1

    if all_flat:
        # Fallback: infer levels from article numbering
        for s in sections:
            inferred = _infer_level_from_numbering(s["heading"])
            if inferred is not None:
                s["level"] = inferred

    # Build tree using a stack
    root: list[dict] = []
    stack: list[dict] = []

    for s in sections:
        node = {
            "heading": s["heading"],
            "level": s["level"],
            "body": s["body"],
            "children": [],
        }

        # Pop stack until we find a parent with lower level
        while stack and stack[-1]["level"] >= s["level"]:
            stack.pop()

        if stack:
            stack[-1]["children"].append(node)
        else:
            root.append(node)

        stack.append(node)

    return root


def _split_long_text(text: str, heading: str) -> list[str]:
    """Split text > MAX_TOKENS into ~SPLIT_TARGET token chunks with overlap."""
    tokens = _enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + SPLIT_TARGET, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _enc.decode(chunk_tokens)
        # Prepend heading for context on continuation chunks
        if start > 0:
            chunk_text = f"{heading} (suite)\n\n{chunk_text}"
        chunks.append(chunk_text)
        start = end - SPLIT_OVERLAP if end < len(tokens) else end
    return chunks


def _is_table_section(body: str) -> bool:
    """Check if body is primarily a table (many | characters)."""
    lines = body.strip().split("\n")
    if not lines:
        return False
    pipe_lines = sum(1 for line in lines if "|" in line)
    return pipe_lines > len(lines) * 0.5


def _build_page_spans(
    heading_pages: dict[str, int],
) -> dict[str, tuple[int, int]]:
    """Compute page spans for each heading.

    For each heading, the span is (start_page, end_page) where end_page
    is the page before the next heading starts. Used to interpolate pages
    across split children.

    Args:
        heading_pages: Mapping heading text → page number.

    Returns:
        Dict mapping heading text → (start_page, end_page).
    """
    spans: dict[str, tuple[int, int]] = {}
    sorted_headings = sorted(heading_pages.items(), key=lambda x: x[1])
    for i, (heading_text, page) in enumerate(sorted_headings):
        next_page = sorted_headings[i + 1][1] if i + 1 < len(sorted_headings) else page
        spans[heading_text] = (page, max(page, next_page - 1))
    return spans


def chunk_document(  # noqa: C901
    markdown: str,
    source: str,
    heading_pages: dict[str, int] | None = None,
) -> dict:
    """Chunk a markdown document into children and parents.

    Args:
        markdown: Markdown text with heading levels.
        source: Source PDF filename.
        heading_pages: Optional mapping heading text → page number.

    Returns:
        Dict with "children" and "parents" lists.
    """
    sections = parse_sections(markdown)
    hierarchy = build_hierarchy(sections)
    _heading_pages = heading_pages or {}

    children: list[dict] = []
    parents: list[dict] = []
    child_counter = 0
    parent_counter = 0

    _page_spans = _build_page_spans(_heading_pages) if _heading_pages else {}

    def _make_parent_id() -> str:
        nonlocal parent_counter
        pid = f"{source}-p{parent_counter:03d}"
        parent_counter += 1
        return pid

    def _process_node(node: dict, parent_id: str) -> None:
        nonlocal child_counter

        has_children = len(node["children"]) > 0
        has_body = len(node["body"].strip()) > 10

        if has_children:
            # This node is a parent — build parent text
            parts = []
            if node["body"].strip():
                parts.append(f"{node['heading']}\n\n{node['body']}".strip())
            else:
                parts.append(node["heading"])
            for child_node in node["children"]:
                child_text = f"{child_node['heading']}\n\n{child_node['body']}".strip()
                parts.append(child_text)
                # Recurse into grandchildren text
                for gc in child_node.get("children", []):
                    parts.append(f"{gc['heading']}\n\n{gc['body']}".strip())
            parent_text = "\n\n".join(parts)

            my_parent_id = _make_parent_id()
            parent_page = _heading_pages.get(node["heading"])
            parents.append(
                {
                    "id": my_parent_id,
                    "text": parent_text,
                    "source": source,
                    "section": node["heading"],
                    "tokens": count_tokens(parent_text),
                    "page": parent_page,
                }
            )

            # If this parent also has its own body text, make it a child
            if has_body:
                _add_child(node["heading"], node["body"], source, my_parent_id)

            # Process children
            for child_node in node["children"]:
                _process_node(child_node, my_parent_id)

        elif has_body:
            # Leaf section with body → child
            _add_child(node["heading"], node["body"], source, parent_id)
        # else: empty heading with no children → skip

    def _add_child(heading: str, body: str, src: str, pid: str) -> None:
        nonlocal child_counter
        full_text = f"{heading}\n\n{body}".strip()
        tokens = count_tokens(full_text)

        is_table = _is_table_section(body)
        must_split = (
            (tokens > MAX_TOKENS and not is_table)
            or tokens > HARD_MAX_TOKENS  # enforce EmbeddingGemma limit
        )

        if must_split:
            chunks = _split_long_text(full_text, heading)
            # Interpolate pages across split children using page spans
            start_page, end_page = _page_spans.get(heading, (None, None))
            for i, chunk_text in enumerate(chunks):
                page_override = None
                if start_page is not None and end_page is not None:
                    span = end_page - start_page + 1
                    page_override = start_page + min(
                        int(i * span / len(chunks)),
                        end_page - start_page,
                    )
                _emit_child(chunk_text, src, pid, heading, page_override)
        else:
            # Single child (even if > 512 for tables — covered by summary)
            _emit_child(full_text, src, pid, heading)

    def _emit_child(
        text: str,
        src: str,
        pid: str,
        section: str,
        page_override: int | None = None,
    ) -> None:
        nonlocal child_counter
        art_match = ARTICLE_NUM_RE.match(section)
        art_num = art_match.group(1).rstrip(".") if art_match else None

        # Use override (from split interpolation) or heading_pages
        page = (
            page_override if page_override is not None else _heading_pages.get(section)
        )

        children.append(
            {
                "id": f"{src}-c{child_counter:04d}",
                "text": text,
                "parent_id": pid,
                "source": src,
                "article_num": art_num,
                "section": section,
                "tokens": count_tokens(text),
                "page": page,
            }
        )
        child_counter += 1

    # Root parent for orphan nodes
    root_pid = _make_parent_id()
    parents.append(
        {
            "id": root_pid,
            "text": "",
            "source": source,
            "section": "root",
            "tokens": 0,
        }
    )

    for node in hierarchy:
        _process_node(node, root_pid)

    # Merge consecutive small children under same parent
    children = _merge_small_children(children)

    return {"children": children, "parents": parents}


def _merge_small_children(children: list[dict]) -> list[dict]:
    """Merge consecutive children under same parent if both < MERGE_THRESHOLD."""
    if not children:
        return children

    merged: list[dict] = [children[0]]

    for child in children[1:]:
        prev = merged[-1]
        if (
            prev["parent_id"] == child["parent_id"]
            and prev["tokens"] < MERGE_THRESHOLD
            and child["tokens"] < MERGE_THRESHOLD
            and prev["tokens"] + child["tokens"] <= MAX_TOKENS
        ):
            prev["text"] = prev["text"] + "\n\n" + child["text"]
            prev["tokens"] = count_tokens(prev["text"])
            prev["section"] = prev["section"] + " + " + child["section"]
            if child.get("article_num") and prev.get("article_num"):
                prev["article_num"] = prev["article_num"] + "+" + child["article_num"]
        else:
            merged.append(child)

    return merged
