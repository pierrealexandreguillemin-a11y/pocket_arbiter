"""
Re-extract full question text + answers from Docling JSON files.

This module re-extracts complete question/answer data directly from
the Docling-processed markdown of FFE exam annales, fixing truncation
and fusion issues found in the original extraction pipeline.

Supported formats (discovered via manual inspection):
    - dec2019-dec2021: "UV{X} - session de {mois} {année}" (no explicit Sujet)
    - jun2022-jun2024: same header, variable grille placement
    - jun2023 special: "UV{X} Métropole" + "UV{X} DOM-TOM"
    - dec2024-jun2025: "FFE DNA UV{X} session..." with explicit Sujet/Fin

Sessions NOT in GS: jun2018, dec2018, jun2019 (0 questions).
Sessions in GS: dec2019→jun2025 (386 questions across 10 sessions).

ISO Reference:
    - ISO/IEC 12207 - Root cause fix (re-extraction from source)
    - ISO/IEC 25010 - Data quality (full text, no truncation)
    - ISO/IEC 42001 - Traceability (session/uv/question_num mapping)
    - ISO/IEC 29119 - Testing (coverage >= 80%)
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Constants ---

CORPUS_BASE = Path("corpus/processed")

# Mapping: session -> (filename, subdirectory)
# Only sessions that have questions in the GS are included.
SESSION_TO_DOCLING: dict[str, tuple[str, str]] = {
    "dec2019": ("Annales-Session-decembre-2019-version-2.json", "annales_all"),
    "jun2021": ("Annales-Session-juin-2021.json", "annales_all"),
    "dec2021": ("Annales-Session-decembre-2021.json", "annales_all"),
    "jun2022": ("Annales-session-juin-2022-vers2.json", "annales_all"),
    "dec2022": ("Annales-session-decembre-2022.json", "annales_all"),
    "jun2023": ("Annales-session-Juin-2023.json", "annales_all"),
    "dec2023": ("Annales-decembre2023.json", "annales_all"),
    "jun2024": ("Annales-juin-2024.json", "annales_all"),
    "dec2024": ("Annales-Decembre-2024.json", "annales_dec_2024"),
    "jun2025": ("Annales-Juin-2025-VF2.json", "annales_dec_2024"),
}

UV_TO_CATEGORY: dict[str, str] = {
    "UVR": "rules",
    "UVC": "clubs",
    "UVO": "open",
    "UVT": "tournament",
}

# --- Text cleaning ---


def clean_text(text: str) -> str:
    """Clean extracted text: remove markdown artifacts, normalize whitespace."""
    # Remove image markers
    text = re.sub(r"<!--\s*image\s*-->", "", text)
    # Remove internal ## markdown headers (Docling artifact)
    text = re.sub(r"\s*##\s*", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Docling file access ---


def get_docling_path(session: str, corpus_base: Path | None = None) -> Path | None:
    """
    Get the filesystem path for a session's Docling JSON file.

    Args:
        session: Session identifier (e.g., 'dec2024').
        corpus_base: Base directory for corpus. Defaults to CORPUS_BASE.

    Returns:
        Path to JSON file, or None if session not mapped.
    """
    base = corpus_base or CORPUS_BASE
    mapping = SESSION_TO_DOCLING.get(session)
    if not mapping:
        return None
    filename, subdir = mapping
    return base / subdir / filename


def load_docling_markdown(session: str, corpus_base: Path | None = None) -> str | None:
    """
    Load the markdown content from a session's Docling JSON.

    Args:
        session: Session identifier.
        corpus_base: Base directory for corpus.

    Returns:
        Markdown string, or None if not available.
    """
    path = get_docling_path(session, corpus_base)
    if path is None or not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("markdown", "")


# --- Section detection ---
#
# Each UV in an annales PDF has up to 3 sections:
#   1. Sujet (questions without answers)
#   2. Grille des réponses (answer table: num, letter, article, taux)
#   3. Corrigé détaillé (questions + choices + article ref + explanation)
#
# The header format varies across sessions. We detect sections by
# looking for ## headers that mention the UV code + a keyword.


def _find_section(
    markdown: str,
    uv: str,
    keywords: list[str],
    *,
    skip_n: int = 0,
) -> tuple[int, int] | None:
    """
    Find a section in markdown by UV code + keyword in ## header.

    Searches for headers matching: ## ... {uv} ... {keyword} ...
    Returns the (start, end) of content AFTER the header, up to the
    next major section boundary.

    Args:
        markdown: Full markdown content.
        uv: UV code (UVR, UVC, UVO, UVT).
        keywords: List of keywords to match (any one).
        skip_n: Skip the first N matches (for disambiguation).

    Returns:
        (start, end) tuple or None.
    """
    kw_pattern = "|".join(re.escape(k) for k in keywords)
    pattern = re.compile(
        rf"^##[^\n]*{uv}[^\n]*(?:{kw_pattern})",
        re.MULTILINE | re.IGNORECASE,
    )

    matches = list(pattern.finditer(markdown))
    if len(matches) <= skip_n:
        return None

    match = matches[skip_n]
    start = match.end()

    # Find end: next ## section that indicates a boundary
    # Boundaries: another UV header, Fin, Grille, Corrigé, Commentaires, FFE DNA header
    end_pattern = re.compile(
        r"\n##\s*(?:"
        r"(?:Fin\b|FIN\b)"
        r"|UV[RCOT]\s*[-\u2013\u2014\s]"
        r"|[^\n]*UV[RCOT][^\n]*(?:[Gg]rille|[Cc]orrig|[Ss]ujet)"
        r"|F\s[^\n]*D\s*irection\s*N"
        r"|Commentaires?\s+du"
        r"|[*]+\s*FIN"
        r")",
        re.IGNORECASE,
    )
    end_match = end_pattern.search(markdown[start:])
    end = start + end_match.start() if end_match else len(markdown)

    return (start, end)


def find_uv_sujet(markdown: str, uv: str) -> tuple[int, int] | None:
    """Find the 'Sujet' (question) section for a UV."""
    # New format (dec2024+): explicit "Sujet"
    result = _find_section(markdown, uv, ["Sujet", "sujet"])
    if result:
        return result

    # Old format: first UV header = sujet (no keyword)
    # Match: ## ... UV{X} ... session ...  OR  ## ... UV{X} - session ...
    pattern = re.compile(
        rf"^##[^\n]*{uv}\s*[-\u2013\u2014\s]+(?:[Ss]ession|SESSION)",
        re.MULTILINE,
    )
    match = pattern.search(markdown)
    if not match:
        return None

    start = match.end()

    # End: next major section (Grille, Corrigé, FIN, or next UV)
    end_pattern = re.compile(
        r"\n##\s*(?:"
        r"Fin\b|FIN\b"
        r"|" + uv + r"[^\n]*(?:[Gg]rille|[Cc]orrig)"
        r"|UV[RCOT]\s*[-\u2013\u2014\s]"
        r"|F\s[^\n]*D\s*irection\s*N"
        r"|[*]+\s*FIN"
        r")",
        re.IGNORECASE,
    )
    end_match = end_pattern.search(markdown[start:])
    end = start + end_match.start() if end_match else len(markdown)

    return (start, end)


def find_uv_grille(markdown: str, uv: str) -> tuple[int, int] | None:
    """Find the 'Grille des réponses' section for a UV."""
    return _find_section(markdown, uv, ["grille", "Grille", "GRILLE"])


def find_uv_corrige(markdown: str, uv: str) -> tuple[int, int] | None:
    """Find the 'Corrigé détaillé' section for a UV."""
    return _find_section(markdown, uv, ["corrig", "Corrig", "CORRIG"])


# --- Question block extraction ---


def extract_question_block(section_text: str, question_num: int) -> str | None:
    """
    Extract the full text block for a question number from a section.

    Finds "Question {num}" and captures everything until the next
    "Question {N}" or end of section.

    Args:
        section_text: Text of a UV section.
        question_num: Question number to extract.

    Returns:
        Raw block text (not cleaned), or None if not found.
    """
    pattern = re.compile(
        r"(?:^|\n)(?:##\s*)?Question\s+(\d+)\s*:?\s*(.*?)(?=\n(?:##\s*)?Question\s+\d+\s*:?|$)",
        re.DOTALL | re.IGNORECASE,
    )

    for match in pattern.finditer(section_text):
        if int(match.group(1)) == question_num:
            return match.group(2).strip()
    return None


# --- Choice extraction ---

# Ordered by specificity (most specific first)
_CHOICE_PATTERNS = [
    # "- a) text"
    re.compile(
        r"(?:^|\n)\s*-\s*([a-fA-F])\)\s*(.+?)(?=\n\s*-\s*[a-fA-F]\)|$)",
        re.DOTALL | re.IGNORECASE,
    ),
    # "- A - text" (jun2021)
    re.compile(
        r"(?:^|\n)\s*-\s*([A-F])\s*[-\u2013\u2014]\s*(.+?)(?=\n\s*-\s*[A-F]\s*[-\u2013\u2014]|$)",
        re.DOTALL,
    ),
    # "A : text" or "- A : text" (dec2019)
    re.compile(
        r"(?:^|\n)\s*(?:-\s*)?([A-F])\s*:\s*(.+?)(?=\n\s*(?:-\s*)?[A-F]\s*:|$)",
        re.DOTALL,
    ),
    # "A - text"
    re.compile(
        r"(?:^|\n)(?:##\s*)?([A-F])\s*[-\u2013\u2014]\s*(.+?)(?=\n(?:##\s*)?[A-F]\s*[-\u2013\u2014]|$)",
        re.DOTALL,
    ),
    # "a. text"
    re.compile(
        r"(?:^|\n)\s*([a-fA-F])\.\s*(.+?)(?=\n\s*[a-fA-F]\.|$)",
        re.DOTALL,
    ),
    # Inline: "A - text. B - text." on same line (jun2021 corrigé)
    re.compile(
        r"(?:^|(?<=[.?!]\s))([A-F])\s*[-\u2013\u2014]\s*(.+?)(?=\.\s+[A-F]\s*[-\u2013\u2014]|\.\s*$|$)",
        re.MULTILINE,
    ),
]


def extract_choices(block: str) -> dict[str, str]:
    """
    Extract QCM choices from a question block.

    Tries multiple patterns to handle format variations across sessions.

    Args:
        block: Raw question block text.

    Returns:
        Dict mapping uppercase letter to cleaned choice text.
    """
    choices: dict[str, str] = {}
    for pattern in _CHOICE_PATTERNS:
        for match in pattern.finditer(block):
            letter = match.group(1).upper()
            text = clean_text(match.group(2))
            if letter not in choices and text:
                choices[letter] = text
    return choices


def extract_question_text(block: str) -> str:
    """
    Extract the question text (before choices) from a block.

    Args:
        block: Raw question block text.

    Returns:
        Cleaned question text.
    """
    # Find where choices start
    choice_start = re.search(
        r"\n\s*-\s*[a-fA-F][\)\-:\.\s]" r"|\n\s*[A-F]\s*[-\u2013\u2014:]\s",
        block,
    )
    text = block[: choice_start.start()] if choice_start else block
    return clean_text(text)


# --- Article reference + explanation extraction ---


def extract_article_reference(block: str) -> str | None:
    """
    Extract the article reference from a corrigé block.

    Args:
        block: Raw corrigé question block.

    Returns:
        Cleaned article reference, or None.
    """
    patterns = [
        r"((?:LA|Livre\s+de\s+l.Arbitre)\s*[-\u2013\u2014,]\s*[^\n]+)",
        r"(Article\s+\d[^\n]*)",
        r"(Chapitre\s+\d[^\n]*)",
        r"(Annexe\s+[A-Z][^\n]*)",
        r"(R\u00e8gle[^\n]*\d[^\n]*)",
        r"(Art\.\s*\d[^\n]*)",
        r"(R\d{2}\s+[^\n]+)",
        r"(LF[,\s][^\n]+)",
    ]
    for pat in patterns:
        match = re.search(pat, block, re.IGNORECASE)
        if match:
            ref = clean_text(match.group(1))
            if len(ref) > 8:
                return ref
    return None


def extract_explanation(block: str) -> str | None:
    """
    Extract the explanation text from a corrigé block.

    The explanation is the text AFTER the article reference line,
    excluding choices and question text.

    Args:
        block: Raw corrigé question block.

    Returns:
        Cleaned explanation text (>= 20 chars), or None.
    """
    lines = block.split("\n")

    # Find last choice line
    last_choice_idx = -1
    for i, line in enumerate(lines):
        if re.match(r"^\s*-?\s*[a-dA-F][\)\-:\.\s]", line.strip()):
            last_choice_idx = i

    # Find article reference line after choices
    article_idx = -1
    for i in range(max(0, last_choice_idx), len(lines)):
        line = lines[i].strip()
        if re.search(
            r"(?:Article|R\u00e8gle|Chapitre|LA\s*[-\u2013\u2014,]|Annexe|R\d{2}|Art\.|LF[,\s])",
            line,
            re.IGNORECASE,
        ):
            if len(line) > 10:
                article_idx = i
                break

    if article_idx == -1:
        return None

    # Collect lines after article reference
    explanation_lines = []
    for i in range(article_idx + 1, len(lines)):
        line = lines[i].strip()
        if re.match(r"^(?:##\s*)?[Qq]uestion\s+\d+", line):
            break
        if "<!-- image -->" in line:
            continue
        if line:
            explanation_lines.append(line)

    if not explanation_lines:
        return None

    text = clean_text(" ".join(explanation_lines))
    return text if len(text) >= 20 else None


# --- Grille table parsing ---


def _identify_columns(
    header_cells: list[str],
) -> dict[str, int]:
    """
    Map semantic column names to indices from a grille header row.

    Detects: question, answer, article, rate columns across
    varying formats (dec2019-jun2025).

    Args:
        header_cells: List of header cell texts (stripped, lowercase).

    Returns:
        Dict with keys 'question', 'answer', 'article', 'rate'
        mapped to column indices. Missing columns get index -1.
    """
    col_map: dict[str, int] = {
        "question": -1,
        "answer": -1,
        "article": -1,
        "rate": -1,
    }

    for i, cell in enumerate(header_cells):
        lower = cell.lower().strip()
        # Question/N° column
        if (
            any(kw in lower for kw in ["question", "n°", "n "])
            and col_map["question"] == -1
        ):
            col_map["question"] = i
        # Answer column
        elif (
            any(kw in lower for kw in ["réponse", "reponse", "rponse"])
            and col_map["answer"] == -1
        ):
            col_map["answer"] = i
        # Article column
        elif (
            any(kw in lower for kw in ["article", "référence", "reference"])
            and col_map["article"] == -1
        ):
            col_map["article"] = i
        # Rate column
        elif (
            any(kw in lower for kw in ["taux", "réussite", "reussite", "russite"])
            and col_map["rate"] == -1
        ):
            col_map["rate"] = i

    # Fallback: question column is usually first
    if col_map["question"] == -1 and header_cells:
        col_map["question"] = 0

    return col_map


def parse_grille_table(grille_text: str) -> dict[int, dict[str, Any]]:
    """
    Parse the 'Grille des réponses' section.

    Header-aware: detects column layout from the header row,
    then extracts answer letter, article reference, and success rate
    from data rows. Handles 3-6 column formats across all sessions.

    Args:
        grille_text: Text of the grille section.

    Returns:
        Dict mapping question_num to {answer, article, rate}.
    """
    results: dict[int, dict[str, Any]] = {}

    # Split into lines and find table rows (lines starting/containing |)
    lines = grille_text.split("\n")
    table_rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        # Skip separator rows (|---|---|)
        if re.match(r"^\|[\s\-|:]+\|$", stripped):
            continue
        cells = [c.strip() for c in stripped.split("|")]
        # Remove empty first/last from split
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells:
            table_rows.append(cells)

    if len(table_rows) < 2:
        return results

    # First row = header
    header = table_rows[0]
    col_map = _identify_columns(header)
    data_rows = table_rows[1:]

    for row in data_rows:
        # Extract question number
        q_num = _extract_q_num(row, col_map["question"])
        if q_num is None:
            continue

        # Extract answer letter (A-F)
        answer = _extract_answer(row, col_map["answer"])

        # Extract article reference
        article = _extract_article_cell(row, col_map["article"])

        # Extract success rate
        rate = _extract_rate(row, col_map["rate"])

        if answer:
            results[q_num] = {
                "answer": answer,
                "article": article,
                "rate": rate,
            }

    return results


def _extract_q_num(row: list[str], col_idx: int) -> int | None:
    """Extract question number from a row cell.

    Handles OCR-shifted cells by scanning all cells if the
    designated column doesn't have a number.
    """
    # Try designated column first
    if 0 <= col_idx < len(row):
        match = re.match(r"\s*(\d+)", row[col_idx])
        if match:
            return int(match.group(1))
    # Scan all cells for a leading number
    for cell in row:
        match = re.match(r"\s*(\d+)", cell.strip())
        if match:
            return int(match.group(1))
    return None


def _extract_answer(row: list[str], col_idx: int) -> str | None:
    """Extract answer letter (A-F) from a row cell.

    Handles multiple OCR formats:
    - Standalone letter in designated column
    - "num letter" merged cells (e.g., "1 C")
    - Standalone letter anywhere in row
    """
    # Try designated column first
    if 0 <= col_idx < len(row):
        cell = row[col_idx].strip()
        # Standalone letter
        if re.match(r"^[A-F]$", cell):
            return cell.upper()
        # Letter after number (merged "1 C")
        match = re.search(r"\b([A-F])\b", cell)
        if match:
            return match.group(1).upper()

    # Scan all cells: check for "num letter" merged pattern
    for cell in row:
        cell_stripped = cell.strip()
        # Standalone letter
        if re.match(r"^[A-F]$", cell_stripped):
            return cell_stripped.upper()
        # "num letter" pattern (e.g., "1 C" or "15 D")
        match = re.match(r"\d+\s+([A-F])\b", cell_stripped)
        if match:
            return match.group(1).upper()

    return None


def _extract_article_cell(row: list[str], col_idx: int) -> str | None:
    """Extract article reference from a row cell.

    Scans all cells if the designated column doesn't contain
    article-like content.
    """
    _article_re = re.compile(
        r"(?:Art\.|Article|LA\s*[-,\u2013\u2014]|Chapitre|R\d{2}|LF[,\s])",
        re.IGNORECASE,
    )
    # Try designated column first
    if 0 <= col_idx < len(row):
        if _article_re.search(row[col_idx]):
            text = clean_text(row[col_idx])
            return text if text else None
    # Scan all cells for article-like content
    for cell in row:
        if _article_re.search(cell):
            text = clean_text(cell)
            return text if text else None
    return None


def _extract_rate(row: list[str], col_idx: int) -> float | None:
    """Extract success rate from a row cell.

    Handles: "80%", "80 (%)", "80 %", standalone "80" in rate column.
    Scans from right to left as rate is typically the last column.
    """
    # Try designated column first
    if 0 <= col_idx < len(row):
        match = re.search(r"(\d{1,3})\s*(?:%|\(%\))?", row[col_idx])
        if match:
            val = int(match.group(1))
            if 0 <= val <= 100:
                return val / 100.0

    # Scan cells from right (rate is usually last)
    for cell in reversed(row):
        # Match "N%" or "N (%)" patterns
        match = re.search(r"(\d{1,3})\s*(?:%|\(%\))", cell)
        if match:
            val = int(match.group(1))
            if 0 <= val <= 100:
                return val / 100.0
    return None


# --- Extraction flags ---


_COMMENTARY_PATTERN = re.compile(
    r"(?:\d+\s*%\s*des\s*candidats"
    r"|Il\s+s['\u2019]agissait"
    r"|pas\s+[e\u00e9]t[e\u00e9]\s+comptabilis)",
    re.IGNORECASE,
)


def detect_extraction_flags(
    block: str | None,
    choices: dict[str, str],
    question_found: bool,
) -> list[str]:
    """
    Detect extraction quality flags for a question.

    Args:
        block: Raw question block (None if not found).
        choices: Extracted choices dict.
        question_found: Whether the question was located in markdown.

    Returns:
        List of flag strings.
    """
    flags: list[str] = []

    if not question_found or block is None:
        flags.append("no_question_found")
        return flags

    if not choices:
        flags.append("no_choices")

    if "<!-- image -->" in block:
        flags.append("image_dependent")

    if _COMMENTARY_PATTERN.search(block):
        flags.append("commentary")

    annulled_patterns = [
        r"question\s+annul[e\u00e9]e",
        r"pas\s+[e\u00e9]t[e\u00e9]\s+comptabilis",
        r"question\s+non\s+compt",
    ]
    for pat in annulled_patterns:
        if re.search(pat, block, re.IGNORECASE):
            flags.append("annulled")
            break

    # Internal "## Question N" = actual multi-block fusion (BAD).
    # Bare "## text" within a block is a Docling sub-heading artifact
    # (e.g., "## Quelle UV doit-il valider ?") — harmless.
    if re.search(r"\n##\s*Question\s+\d+", block, re.IGNORECASE):
        flags.append("multi_block")

    return flags


# --- Single question re-extraction ---


def reextract_question(
    session: str,
    uv: str,
    question_num: int,
    question_id: str,
    markdown: str,
) -> dict[str, Any]:
    """
    Re-extract a single question's full data from Docling markdown.

    Searches sujet, grille, and corrigé sections to assemble
    complete question + answer data.

    Args:
        session: Session identifier.
        uv: UV code (UVR, UVC, UVO, UVT).
        question_num: Question number within the UV.
        question_id: Original question ID from GS.
        markdown: Full Docling markdown content.

    Returns:
        Dict with id, question_full, choices, mcq_answer,
        answer_text_from_choice, answer_explanation,
        article_reference, success_rate, extraction_flags.
    """
    result: dict[str, Any] = {
        "id": question_id,
        "question_full": "",
        "choices": {},
        "mcq_answer": None,
        "answer_text_from_choice": None,
        "answer_explanation": None,
        "article_reference": None,
        "success_rate": None,
        "extraction_flags": [],
    }

    # 1. Try sujet section (primary for question text)
    sujet_block = None
    sujet_bounds = find_uv_sujet(markdown, uv)
    if sujet_bounds:
        s_start, s_end = sujet_bounds
        sujet_block = extract_question_block(markdown[s_start:s_end], question_num)

    # 2. Try corrigé section (has questions + article + explanation)
    corrige_block = None
    corrige_bounds = find_uv_corrige(markdown, uv)
    if corrige_bounds:
        c_start, c_end = corrige_bounds
        corrige_block = extract_question_block(markdown[c_start:c_end], question_num)

    # 3. Choose primary block
    primary_block = sujet_block or corrige_block
    question_found = primary_block is not None

    if primary_block:
        result["question_full"] = extract_question_text(primary_block)
        result["choices"] = extract_choices(primary_block)

    # 4. Enrich from corrigé (article + explanation)
    if corrige_block:
        result["article_reference"] = extract_article_reference(corrige_block)
        result["answer_explanation"] = extract_explanation(corrige_block)
        # Merge: use corrigé choices if richer than sujet choices
        corrige_choices = extract_choices(corrige_block)
        if len(corrige_choices) > len(result["choices"]):
            result["choices"] = corrige_choices

    # 5. Enrich from grille (answer letter + success rate)
    grille_bounds = find_uv_grille(markdown, uv)
    if grille_bounds:
        g_start, g_end = grille_bounds
        grille_data = parse_grille_table(markdown[g_start:g_end])
        if question_num in grille_data:
            gd = grille_data[question_num]
            result["mcq_answer"] = gd["answer"]
            result["success_rate"] = gd["rate"]
            if not result["article_reference"] and gd.get("article"):
                result["article_reference"] = gd["article"]

    # 6. Derive answer text from correct choice letter
    if result["mcq_answer"] and result["choices"]:
        letter = result["mcq_answer"]
        if letter in result["choices"]:
            result["answer_text_from_choice"] = result["choices"][letter]

    # 7. Flags
    result["extraction_flags"] = detect_extraction_flags(
        primary_block, result["choices"], question_found
    )

    return result


# --- Batch re-extraction ---


def reextract_all(
    gs_path: Path,
    corpus_base: Path | None = None,
    dry_run_limit: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Re-extract all annales questions from Docling JSONs.

    Args:
        gs_path: Path to gold standard JSON.
        corpus_base: Base directory for corpus.
        dry_run_limit: If > 0, limit to N questions per session+UV.

    Returns:
        Tuple of (results list, report dict).
    """
    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)

    questions = gs["questions"]
    results: list[dict[str, Any]] = []
    md_cache: dict[str, str | None] = {}

    # Per-session+uv counter for dry-run
    processed_counts: dict[str, int] = {}
    total_skipped = 0
    flag_counts: dict[str, int] = {}

    for q in questions:
        meta = q.get("metadata", {})
        src = meta.get("annales_source")
        if not src or not src.get("session"):
            continue  # skip human questions

        session = src["session"]
        uv = src.get("uv", "")
        question_num = src.get("question_num", 0)
        q_id = q["id"]

        # Dry-run limit
        key = f"{session}_{uv}"
        if dry_run_limit > 0:
            if processed_counts.get(key, 0) >= dry_run_limit:
                total_skipped += 1
                continue

        # Load markdown (cached)
        if session not in md_cache:
            md_cache[session] = load_docling_markdown(session, corpus_base)

        md = md_cache[session]
        if md is None:
            logger.warning(f"No docling file for session {session}")
            r: dict[str, Any] = {
                "id": q_id,
                "question_full": "",
                "choices": {},
                "mcq_answer": None,
                "answer_text_from_choice": None,
                "answer_explanation": None,
                "article_reference": None,
                "success_rate": None,
                "extraction_flags": ["no_question_found"],
            }
            results.append(r)
            processed_counts[key] = processed_counts.get(key, 0) + 1
            continue

        r = reextract_question(session, uv, question_num, q_id, md)
        results.append(r)
        processed_counts[key] = processed_counts.get(key, 0) + 1

        for flag in r["extraction_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    total_processed = len(results)
    logger.info(f"Re-extraction: {total_processed} processed, {total_skipped} skipped")

    # Build report
    report: dict[str, Any] = {
        "total_processed": total_processed,
        "total_skipped": total_skipped,
        "total_with_flags": sum(1 for r in results if r["extraction_flags"]),
        "flag_counts": flag_counts,
        "by_session": _aggregate_by_session(results, questions),
    }

    return results, report


def _aggregate_by_session(
    results: list[dict[str, Any]],
    gs_questions: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Aggregate extraction stats by session."""
    # Build id→session lookup
    id_to_session: dict[str, str] = {}
    for q in gs_questions:
        src = q.get("metadata", {}).get("annales_source", {})
        if src.get("session"):
            id_to_session[q["id"]] = src["session"]

    agg: dict[str, dict[str, int]] = {}
    for r in results:
        s = id_to_session.get(r["id"], "unknown")
        if s not in agg:
            agg[s] = {
                "total": 0,
                "with_full_text": 0,
                "with_choices": 0,
                "with_answer": 0,
                "with_explanation": 0,
                "flagged": 0,
            }
        agg[s]["total"] += 1
        if r.get("question_full"):
            agg[s]["with_full_text"] += 1
        if r.get("choices"):
            agg[s]["with_choices"] += 1
        if r.get("mcq_answer"):
            agg[s]["with_answer"] += 1
        if r.get("answer_explanation"):
            agg[s]["with_explanation"] += 1
        if r.get("extraction_flags"):
            agg[s]["flagged"] += 1

    return agg


# --- CLI ---


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-extract questions from Docling JSON files"
    )
    parser.add_argument(
        "--gs",
        type=Path,
        default=Path("tests/data/gold_standard_annales_fr_v7.json"),
    )
    parser.add_argument(
        "--corpus-base",
        type=Path,
        default=CORPUS_BASE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/data/checkpoints/reextraction_results.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("tests/data/checkpoints/reextraction_report.json"),
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=0,
        help="Limit to N questions per session+UV (0=all)",
    )
    args = parser.parse_args()

    results, report = reextract_all(
        gs_path=args.gs,
        corpus_base=args.corpus_base,
        dry_run_limit=args.dry_run,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Re-extraction Report ===")
    print(f"Total processed: {report['total_processed']}")
    print(f"Total with flags: {report['total_with_flags']}")
    for flag, count in sorted(report["flag_counts"].items()):
        print(f"  {flag}: {count}")


if __name__ == "__main__":
    main()
