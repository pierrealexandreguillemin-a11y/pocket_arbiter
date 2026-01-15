"""
Chunker Article Detection - Pocket Arbiter

Detection des frontieres d'Articles/Sections pour chunking semantique.

ISO Reference:
    - ISO 82045 - Document metadata
"""

import re

# Article/Section detection patterns (French + English)
ARTICLE_PATTERNS = [
    # French patterns
    r"^(Article\s+\d+(?:\.\d+)*(?:\.\d+)?)",  # Article 4.1.2
    r"^(Chapitre\s+\d+(?:\.\d+)?)",  # Chapitre 2.1
    r"^(Section\s+\d+)",  # Section 1
    r"^(Annexe\s+[A-Z])",  # Annexe A
    r"^(Titre\s+[IVX]+)",  # Titre I
    r"^(Partie\s+\d+)",  # Partie 1
    r"^(TITRE\s+[IVX]+)",  # TITRE I
    r"^(STATUTS)",  # STATUTS
    r"^(REGLEMENT)",  # REGLEMENT
    # Numeric patterns
    r"^(\d+\.\d+\.\d+\.?\s)",  # 4.1.2
    r"^(\d+\.\d+\.?\s)",  # 4.1
    r"^(\d+\.\d+)\s",  # 5.5 Le toucher
    r"^(\d+\.\s+[A-Z])",  # 4. Le toucher
    # Lettered subsections
    r"^([a-z]\)\s)",  # a)
    r"^(\([a-z]\)\s)",  # (a)
    r"^([A-Z]\.\s)",  # A.
    # English patterns (FIDE)
    r"^(Article\s+\d+(?:\.\d+)*)",  # Article 4.1
    r"^(Chapter\s+\d+)",  # Chapter 2
    r"^(Appendix\s+[A-Z])",  # Appendix A
    r"^(Part\s+[IVX]+)",  # Part I
    r"^(Rule\s+\d+)",  # Rule 1
    r"^(Preface)",  # Preface
    r"^(Introduction)",  # Introduction
]


def detect_article_match(line: str) -> str | None:
    """Detect if a line starts an Article/Section."""
    line_stripped = line.strip()
    for pattern in ARTICLE_PATTERNS:
        match = re.match(pattern, line_stripped, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def detect_article_boundaries(text: str) -> list[dict]:
    """
    Detecte les frontieres d'Articles/Sections dans le texte.

    Args:
        text: Texte complet d'une page ou document.

    Returns:
        Liste de segments avec article et text.
    """
    if not text:
        return []

    segments: list[dict] = []
    lines = text.split("\n")
    current_article: str | None = None
    current_lines: list[str] = []

    for line in lines:
        article_match = detect_article_match(line)

        if article_match:
            if current_lines:
                segment_text = "\n".join(current_lines)
                if len(segment_text.strip()) > 50:
                    segments.append(
                        {"article": current_article, "text": segment_text.strip()}
                    )

            current_article = article_match
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        segment_text = "\n".join(current_lines)
        if len(segment_text.strip()) > 50:
            segments.append({"article": current_article, "text": segment_text.strip()})

    return segments
