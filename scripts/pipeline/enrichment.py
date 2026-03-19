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
