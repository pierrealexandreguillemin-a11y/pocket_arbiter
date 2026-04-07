"""Enrichment module: OPT 1-2-4 (context loader, abbreviations, chapter overrides).

Prepares chunk text for embedding by:
- OPT-1: Loading pre-generated contextual retrieval entries (Anthropic 2024)
- OPT-2: Expanding abbreviations in-place (Haystack pattern)
- OPT-4: Overriding CCH titles for specific LA page ranges (arXiv 2501.07391)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# === OPT-2: Abbreviation dictionary ===
# Verified against corpus: each key has >= 1 match in children table.
# AF1/AF2/AF3 removed (0 matches).

ABBREVIATIONS: dict[str, str] = {
    "AFC": "AFC (Arbitre Federal de Club)",
    "AFJ": "AFJ (Arbitre Federal Jeune)",
    "AI": "AI (Arbitre International)",
    "CDJE": "CDJE (Comite Departemental du Jeu d'Echecs)",
    "CM": "CM (Candidat Maitre)",
    "DNA": "DNA (Direction Nationale de l'Arbitrage)",
    "FFE": "FFE (Federation Francaise des Echecs)",
    "FIDE": "FIDE (Federation Internationale des Echecs)",
    "FM": "FM (Maitre FIDE)",
    "GMI": "GMI (Grand Maitre International)",
    "MI": "MI (Maitre International)",
    "UV": "UV (Unite de Valeur)",
}

# Pre-compiled patterns for performance (word boundary, case-sensitive)
_ABBR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"\b{k}\b(?!\s*\()"), v) for k, v in ABBREVIATIONS.items()
]

# === OPT-4: Chapter title overrides (LA-octobre2025.pdf only) ===
# Page ranges where heading hierarchy is insufficient.

CHAPTER_OVERRIDES: dict[tuple[int, int], str] = {
    (56, 57): "Annexe A - Cadence Rapide",
    (58, 66): "Annexe B - Cadence Blitz",
    (182, 186): "Classement Elo Standard FIDE",
    (187, 191): "Classement Rapide et Blitz FIDE",
    (192, 205): "Titres FIDE",
}

_OVERRIDE_SOURCE = "LA-octobre2025.pdf"

# === C.10: Targeted row-as-chunk (6 priority tables, ~45 rows) ===
TARGETED_TABLES: set[str] = {
    "LA-octobre2025-table2",
    "R01_2025_26_Regles_generales-table0",
    "LA-octobre2025-table73",
    "LA-octobre2025-table68",
    "R01_2025_26_Regles_generales-table1",
    "LA-octobre2025-table63",
}

UNIT_SUFFIXES: dict[str, str] = {
    "elo": "points",
    "classement": "points",
    "k": "coefficient",
    "cadence": "min",
    "temps": "min",
    "durée": "min",
    "duree": "min",
    "âge": "ans",
    "age": "ans",
    "dp": "points",
}

_IDEM_VARIANTS: set[str] = {"", "id.", "id", "idem"}

# === B.5: Column name normalization ===
# Maps abbreviated or non-canonical column headers to their canonical form.
# Lookup is case-insensitive (keys stored lowercase). Canonical values use
# proper French accented spelling so FTS5 and structured cell queries are
# consistent across tables.

COLUMN_NORMALIZATION: dict[str, str] = {
    "cat.": "Catégorie",
    "cat": "Catégorie",
    "categ.": "Catégorie",
    "niv.": "Niveau",
    "niv": "Niveau",
    "nb": "Nombre",
    "nb.": "Nombre",
    "tps": "Temps",
    "tps/ronde": "Temps par ronde",
    "dur.": "Durée",
    "age": "Âge",
    "min": "Minimum",
    "max": "Maximum",
    "dept": "Département",
    "dep": "Département",
    "pts": "Points",
    "class.": "Classement",
    "rk": "Rang",
}


def normalize_column_name(col_name: str) -> str:
    """Normalize column name abbreviations to canonical form (B.5).

    Looks up the stripped, lowercased column name in COLUMN_NORMALIZATION.
    Returns the canonical spelling if found, otherwise returns the original
    stripped value unchanged.

    Args:
        col_name: Raw column header string from a markdown table.

    Returns:
        Canonical column name, or the original if no mapping exists.
    """
    stripped = col_name.strip()
    if not stripped:
        return ""
    return COLUMN_NORMALIZATION.get(stripped.lower(), stripped)


def expand_abbreviations(text: str) -> str:
    """Expand abbreviations in chunk text (word boundary, skip if already expanded).

    Args:
        text: Raw chunk text.

    Returns:
        Text with abbreviations expanded (e.g. "DNA" -> "DNA (Direction ...)").
    """
    for pattern, replacement in _ABBR_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def apply_chapter_override(
    source: str,
    page: int | None,
    current_title: str,
) -> str:
    """Override CCH title for specific LA page ranges.

    Args:
        source: PDF filename.
        page: Page number (or None).
        current_title: Current CCH title from chunker.

    Returns:
        Overridden title if page matches, otherwise current_title unchanged.
    """
    if source != _OVERRIDE_SOURCE or page is None:
        return current_title
    for (start, end), title in CHAPTER_OVERRIDES.items():
        if start <= page <= end:
            return title
    return current_title


def load_contexts(path: Path) -> dict[str, str]:
    """Load chunk_contexts.json (OPT-1 contextual retrieval entries).

    Args:
        path: Path to chunk_contexts.json.

    Returns:
        Dict mapping chunk_id -> context string.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty.
    """
    with open(path, encoding="utf-8") as f:
        contexts: dict[str, str] = json.load(f)
    if not contexts:
        raise ValueError(f"chunk_contexts.json is empty: {path}")
    return contexts


def enrich_chunks(
    children: list[dict],
    contexts: dict[str, str],
) -> list[dict]:
    """Apply all enrichments to children chunks (OPT 1-2).

    Mutates children in-place:
    - OPT-1: Prepend contextual retrieval to text (Anthropic 2024 pattern)
    - OPT-2: Expand abbreviations in text

    OPT-4 (chapter overrides) is NOT applied here — it modifies the CCH title,
    which is handled separately in the indexer via apply_chapter_override().

    Args:
        children: List of chunk dicts with at least 'id' and 'text' keys.
        contexts: Dict mapping chunk_id -> context string.

    Returns:
        The same list (mutated), for chaining convenience.
    """
    for child in children:
        chunk_id = child["id"]
        # OPT-1: prepend context to text
        ctx = contexts.get(chunk_id, "")
        if ctx:
            child["text"] = f"{ctx}\n\n{child['text']}"
        # OPT-2: expand abbreviations
        child["text"] = expand_abbreviations(child["text"])
    return children


def enrich_table_summaries(summaries: list[dict]) -> list[dict]:
    """Apply abbreviation expansion to table summaries (OPT-2 only).

    Table summaries are already concise — no contextual retrieval (OPT-1).
    Mutates summaries in-place.

    Args:
        summaries: List of table summary dicts with 'summary_text' key.

    Returns:
        The same list (mutated), for chaining convenience.
    """
    for summary in summaries:
        summary["summary_text"] = expand_abbreviations(summary["summary_text"])
    return summaries


def _clean_table_line(line: str) -> str:
    """Remove dot-padding and excess whitespace from table cells."""
    # "| Préambule.........." → "| Préambule"
    line = re.sub(r"\.{3,}", "", line)
    # Collapse whitespace within cells
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip()


def forward_fill_rows(rows: list[dict]) -> list[dict]:
    """Fill empty / idem cells with the value from the previous row.

    Idem variants (case-insensitive): "", "id.", "id", "idem".

    Args:
        rows: List of row dicts (uniform keys). Mutated in-place.

    Returns:
        The same list, with empty/idem cells replaced.
    """
    if not rows:
        return rows
    prev: dict[str, str] = {}
    for row in rows:
        for key, val in row.items():
            if val.strip().lower() in _IDEM_VARIANTS and key in prev:
                row[key] = prev[key]
            else:
                prev[key] = val
    return rows


def _apply_unit_suffix(col_name: str, value: str) -> str:
    """Append a unit suffix to numeric values based on column name.

    Only applies if the value starts with a digit or minus sign.
    Lookup is case-insensitive on the column name.

    Args:
        col_name: Column header (e.g. "Elo", "dp").
        value: Cell value string.

    Returns:
        Value with unit suffix appended, or unchanged if not numeric.
    """
    stripped = value.strip()
    if not stripped:
        return value
    if not (stripped[0].isdigit() or stripped[0] == "-"):
        return value
    col_lower = col_name.strip().lower()
    suffix = UNIT_SUFFIXES.get(col_lower)
    if suffix and not stripped.endswith(suffix):
        return f"{stripped} {suffix}"
    return value


def _parse_rows_as_dicts(
    data_lines: list[str], col_names: list[str]
) -> list[dict[str, str]]:
    """Parse pipe-delimited data lines into a list of row dicts.

    Args:
        data_lines: Cleaned pipe-delimited lines (no header/separator).
        col_names: Column headers from the table.

    Returns:
        List of dicts mapping col_name -> cell value.
    """
    parsed: list[dict[str, str]] = []
    for row_line in data_lines:
        values = _parse_pipe_cells(_clean_table_line(row_line))
        row_dict: dict[str, str] = {}
        for j, val in enumerate(values):
            if j < len(col_names):
                row_dict[col_names[j]] = val.strip()
        for col in col_names:
            row_dict.setdefault(col, "")
        parsed.append(row_dict)
    return parsed


def _format_row_text(
    row_dict: dict[str, str], col_names: list[str], title: str
) -> str | None:
    """Format a single row dict into "[col: val]" text with title.

    Returns None if no non-empty cells.
    """
    pairs = []
    for col in col_names:
        val = row_dict.get(col, "").strip()
        if not val or not col.strip():
            continue
        val = _apply_unit_suffix(col, val)
        pairs.append(f"[{col.strip()}: {val}]")
    if not pairs:
        return None
    text = f"{title} | {' '.join(pairs)}"
    return expand_abbreviations(text)


def format_targeted_rows(summaries: list[dict]) -> list[dict]:
    """Build row-chunks for TARGETED_TABLES only (C.10).

    Format: "{table_title} | [col1: val1] [col2: val2] ..."

    Applies forward-fill, unit suffixes, and abbreviation expansion.

    Args:
        summaries: List of table summary dicts with raw_table_text.

    Returns:
        List of targeted row-chunk dicts with id, text, table_id,
        source, page, tokens.
    """
    row_chunks: list[dict] = []
    for summary in summaries:
        table_id = summary["id"]
        if table_id not in TARGETED_TABLES:
            continue

        raw = summary.get("raw_table_text", "")
        source = summary["source"]
        page = summary.get("page")
        title = _extract_table_title(summary.get("summary_text", ""))

        lines = [
            line.strip() for line in raw.split("\n") if line.strip().startswith("|")
        ]
        if len(lines) < 3:
            continue

        col_names = _parse_pipe_cells(_clean_table_line(lines[0]))
        data_lines = [line for line in lines[2:] if not _is_separator_line(line)]

        parsed_rows = _parse_rows_as_dicts(data_lines, col_names)
        forward_fill_rows(parsed_rows)

        for i, row_dict in enumerate(parsed_rows):
            text = _format_row_text(row_dict, col_names, title)
            if text is None:
                continue
            row_chunks.append(
                {
                    "id": f"{table_id}-tr{i:03d}",
                    "text": text,
                    "table_id": table_id,
                    "source": source,
                    "page": page,
                    "tokens": len(text.split()),
                }
            )

    return row_chunks


def parse_table_rows(summaries: list[dict]) -> list[dict]:
    """Parse raw table markdown into row-as-chunk entries.

    Each data row becomes a chunk: column headers + row values.
    Standard: Ragie table chunking, fix-pipeline-task3 level 2.

    Args:
        summaries: List of table summary dicts with 'raw_table_text' key.

    Returns:
        List of row-chunk dicts with id, text, table_id, source, page, tokens.
    """
    row_chunks: list[dict] = []
    for summary in summaries:
        raw = summary.get("raw_table_text", "")
        table_id = summary["id"]
        source = summary["source"]
        page = summary.get("page")

        lines = [
            line.strip() for line in raw.split("\n") if line.strip().startswith("|")
        ]
        if len(lines) < 3:  # header + separator + at least 1 row
            continue

        header = _clean_table_line(lines[0])
        data_lines = [line for line in lines[2:] if not _is_separator_line(line)]

        for i, row_line in enumerate(data_lines):
            clean_row = _clean_table_line(row_line)
            text = f"{header}\n{clean_row}"
            text = expand_abbreviations(text)
            row_chunks.append(
                {
                    "id": f"{table_id}-r{i:03d}",
                    "text": text,
                    "table_id": table_id,
                    "source": source,
                    "page": page,
                    "tokens": len(text.split()),  # approximate
                }
            )

    return row_chunks


def _extract_table_title(summary_text: str, max_len: int = 80) -> str:
    """Extract a concise title from the LLM-generated summary_text.

    Takes text before the first colon (if short enough), otherwise
    truncates at max_len on a word boundary.

    Args:
        summary_text: Full summary text from table_summaries.
        max_len: Maximum title length.

    Returns:
        Concise table title string.
    """
    if not summary_text:
        return "Tableau"
    colon_idx = summary_text.find(":")
    if 0 < colon_idx <= max_len:
        return summary_text[:colon_idx].strip()
    # Fallback: truncate at word boundary
    if len(summary_text) <= max_len:
        return summary_text.strip()
    truncated = summary_text[:max_len].rsplit(" ", 1)[0]
    return truncated.strip()


def _row_to_pairs(values: list[str], col_names: list[str]) -> list[str]:
    """Build 'col est val' pairs from a single row, skipping empty cells."""
    pairs = []
    for j, val in enumerate(values):
        if j >= len(col_names) or not val.strip():
            continue
        col = col_names[j].strip()
        if col:
            pairs.append(f"{col} est {val.strip()}")
    return pairs


def _narrate_one_table(summary: dict) -> list[dict]:
    """Narrate rows for a single table summary."""
    raw = summary.get("raw_table_text", "")
    table_id = summary["id"]
    source = summary["source"]
    page = summary.get("page")
    title = _extract_table_title(summary.get("summary_text", ""))

    lines = [line.strip() for line in raw.split("\n") if line.strip().startswith("|")]
    if len(lines) < 3:
        return []

    col_names = _parse_pipe_cells(_clean_table_line(lines[0]))
    col_names = [c[:40].strip() if len(c) > 40 else c for c in col_names]
    data_lines = [line for line in lines[2:] if not _is_separator_line(line)]

    chunks: list[dict] = []
    for i, row_line in enumerate(data_lines):
        values = _parse_pipe_cells(_clean_table_line(row_line))
        pairs = _row_to_pairs(values, col_names)
        if not pairs:
            continue
        narrative = f"{title} : {', '.join(pairs)}."
        narrative = expand_abbreviations(narrative)
        chunks.append(
            {
                "id": f"{table_id}-r{i:03d}",
                "text": narrative,
                "table_id": table_id,
                "source": source,
                "page": page,
                "tokens": len(narrative.split()),
            }
        )
    return chunks


def narrate_table_rows(summaries: list[dict]) -> list[dict]:
    """Convert table rows into narrative prose for better embedding quality.

    Each data row becomes a self-contained sentence:
    "[Title] : [col1] est [val1], [col2] est [val2], ..."

    Standard: Table-to-Text generation (template-based, deterministic).

    Args:
        summaries: List of table summary dicts with 'raw_table_text'
                   and 'summary_text' keys.

    Returns:
        List of narrative row dicts with id, text, table_id, source,
        page, tokens.
    """
    row_chunks: list[dict] = []
    for summary in summaries:
        row_chunks.extend(_narrate_one_table(summary))
    return row_chunks


def parse_structured_cells(summaries: list[dict]) -> list[dict]:
    """Parse raw table markdown into structured (col_name, cell_value) pairs.

    Standard: TableRAG NeurIPS 2024 cell retrieval pattern.
    Level 3 structured lookup (deterministic SQL, no embedding).

    Args:
        summaries: List of table summary dicts with 'raw_table_text' key.

    Returns:
        List of cell dicts with table_id, row_idx, col_name, cell_value, source, page.
    """
    cells: list[dict] = []
    for summary in summaries:
        cells.extend(_parse_one_table(summary))
    return cells


def _parse_row_cells(
    values: list[str],
    col_names: list[str],
    row_idx: int,
    meta: dict,
) -> list[dict]:
    """Extract non-empty cells from a single row."""
    cells = []
    for col_idx, value in enumerate(values):
        if col_idx >= len(col_names) or not value.strip():
            continue
        cells.append(
            {
                **meta,
                "row_idx": row_idx,
                "col_name": col_names[col_idx],
                "cell_value": value.strip(),
            }
        )
    return cells


def _parse_one_table(summary: dict) -> list[dict]:
    """Parse a single table into (col_name, cell_value) pairs."""
    raw = summary.get("raw_table_text", "")
    lines = [line.strip() for line in raw.split("\n") if line.strip().startswith("|")]
    if len(lines) < 3:
        return []

    col_names = [
        normalize_column_name(c) for c in _parse_pipe_cells(_clean_table_line(lines[0]))
    ]
    data_lines = [line for line in lines[2:] if not _is_separator_line(line)]
    meta = {
        "table_id": summary["id"],
        "source": summary["source"],
        "page": summary.get("page"),
    }

    cells: list[dict] = []
    for row_idx, row_line in enumerate(data_lines):
        values = _parse_pipe_cells(_clean_table_line(row_line))
        cells.extend(_parse_row_cells(values, col_names, row_idx, meta))
    return cells


def _is_separator_line(line: str) -> bool:
    """Check if a pipe-delimited line is a separator (|---|---|)."""
    stripped = line.strip()
    return bool(re.match(r"^\|[\s\-:|]+$", stripped))


def _parse_pipe_cells(line: str) -> list[str]:
    """Split a pipe-delimited table line into cell values."""
    parts = line.split("|")
    # Remove first and last empty parts (before first | and after last |)
    return [p.strip() for p in parts[1:-1]]
