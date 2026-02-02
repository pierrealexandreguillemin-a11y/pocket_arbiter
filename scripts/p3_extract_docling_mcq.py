#!/usr/bin/env python3
"""
Extract MCQ questions and correction grids from Docling JSON files.

Section-based approach: splits markdown into UV sections first, then
extracts questions and correction grids within each UV section.
This eliminates the unreliable heuristic UV matching of the previous version.

Handles correction table formats found across sessions (2019-2025):
- Clean 4-column: | Question | Réponse | Article | Taux |
- 5-column with CADENCE: | Q REGLE | CADENCE | REPONSE | ARTICLE | Taux |
- Merged Q+R: | 1 a | Article | Taux | |
- Text answers (dec2019 UVC): | N° | QUESTION | ARTICLE | REPONSE=text | Taux |
- Shifted columns (dec2022 UVC): question numbers at end of rows
- 5-column with Document: | Q | Document | Article | Réponse | Taux |

ISO Reference:
    - ISO/IEC 42001 A.7.3 - Data traceability
    - ISO/IEC 25010 - Functional suitability

Usage:
    python -m scripts.p3_extract_docling_mcq
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Mapping session -> Docling JSON path (relative to project root)
SESSION_FILES: dict[str, str] = {
    "dec2019": "corpus/processed/annales_all/Annales-Session-decembre-2019-version-2.json",
    "jun2021": "corpus/processed/annales_all/Annales-Session-juin-2021.json",
    "dec2021": "corpus/processed/annales_all/Annales-Session-decembre-2021.json",
    "jun2022": "corpus/processed/annales_all/Annales-session-juin-2022-vers2.json",
    "dec2022": "corpus/processed/annales_all/Annales-session-decembre-2022.json",
    "jun2023": "corpus/processed/annales_all/Annales-session-Juin-2023.json",
    "dec2023": "corpus/processed/annales_all/Annales-decembre2023.json",
    "jun2024": "corpus/processed/annales_all/Annales-juin-2024.json",
    "dec2024": "corpus/processed/annales_dec_2024/Annales-Decembre-2024.json",
    "jun2025": "corpus/processed/annales_juin_2025/Annales-Juin-2025-VF2.json",
}

UV_CANONICAL: dict[str, str] = {
    "UVR": "rules",
    "UVC": "clubs",
    "UVO": "open",
    "UVT": "tournament",
}

# Regex to detect UV type in a header line
UV_RE = re.compile(r"\bUV([RCOT])\b", re.IGNORECASE)

# Section type detection keywords
SECTION_SUJET_KW = ["sujet", "session de", "session d'"]
SECTION_GRILLE_KW = ["grille"]
SECTION_CORRIGE_KW = ["corrigé", "corrige", "corrig\u00e9"]

# Question number pattern (robust across all formats)
QUESTION_NUM_RE = re.compile(
    r"(?:^|\n)\s*(?:##\s*)?(?:QUESTION|Question)\s+(\d+)",
    re.IGNORECASE,
)

# Choice patterns (ordered by specificity)
CHOICE_PATTERNS: list[re.Pattern[str]] = [
    # "- a)" or "- A)" format
    re.compile(r"^\s*-\s*([a-fA-F])\)\s*(.+)", re.MULTILINE),
    # "- A :" format
    re.compile(r"^\s*-\s*([A-Fa-f])\s*:\s*(.+)", re.MULTILINE),
    # "- A -" or "- A –" format
    re.compile(r"^\s*-\s*([A-Fa-f])\s*[-\u2013\u2014]\s*(.+)", re.MULTILINE),
    # "A -" format (no leading dash)
    re.compile(r"^([A-Fa-f])\s*[-\u2013\u2014]\s*(.+)", re.MULTILINE),
    # "a." format
    re.compile(r"^\s*([a-fA-F])\.\s*(.+)", re.MULTILINE),
]


def _clean_text(text: str) -> str:
    """Normalize whitespace in extracted text."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------


def _detect_section_type(header: str) -> str:
    """Detect section type from header text: 'sujet', 'grille', 'corrige', or 'unknown'."""
    lower = header.lower()
    # Order matters: check grille and corrigé before sujet (sujet is default)
    for kw in SECTION_GRILLE_KW:
        if kw in lower:
            return "grille"
    for kw in SECTION_CORRIGE_KW:
        if kw in lower:
            return "corrige"
    # If it mentions the UV but not grille/corrige, it's sujet
    if UV_RE.search(header):
        return "sujet"
    return "unknown"


def _detect_uv_from_header(header: str) -> str | None:
    """Extract UV type (UVR, UVC, UVO, UVT) from a header line."""
    m = UV_RE.search(header.upper())
    if m:
        return "UV" + m.group(1).upper()
    return None


def split_into_uv_sections(
    markdown: str,
) -> dict[str, dict[str, str]]:
    """
    Split markdown into UV sections.

    Returns dict mapping UV code (UVR, UVC, UVO, UVT) to dict of
    section_type -> section_text.
    """
    lines = markdown.split("\n")
    sections: dict[str, dict[str, str]] = {}

    # Find all ## headers that mention a UV
    header_positions: list[
        tuple[int, str, str, str]
    ] = []  # (line_idx, uv, section_type, header_text)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        uv = _detect_uv_from_header(stripped)
        if not uv:
            continue
        section_type = _detect_section_type(stripped)
        header_positions.append((i, uv, section_type, stripped))

    # Extract text between consecutive UV headers
    for idx, (line_idx, uv, section_type, _header) in enumerate(header_positions):
        # End is the next UV header or end of document
        if idx + 1 < len(header_positions):
            end_idx = header_positions[idx + 1][0]
        else:
            end_idx = len(lines)

        section_text = "\n".join(lines[line_idx:end_idx])

        if uv not in sections:
            sections[uv] = {}

        # Append to existing section text (don't discard duplicates —
        # grille tables can appear in a secondary "sujet" section after "Fin de l'UV")
        if section_type in sections[uv]:
            sections[uv][section_type] += "\n" + section_text
        else:
            sections[uv][section_type] = section_text

    return sections


# ---------------------------------------------------------------------------
# Question extraction (from sujet sections)
# ---------------------------------------------------------------------------


def extract_questions(text: str) -> list[dict[str, Any]]:
    """
    Extract questions with choices from a sujet section.

    Returns list of dicts with keys: num, text, choices.
    """
    # Find all question start positions
    q_starts: list[tuple[int, int]] = []  # (char_pos, q_num)
    for m in QUESTION_NUM_RE.finditer(text):
        q_starts.append((m.start(), int(m.group(1))))

    if not q_starts:
        return []

    questions: list[dict[str, Any]] = []
    for i, (pos, q_num) in enumerate(q_starts):
        # Extract block until next question or end
        end_pos = q_starts[i + 1][0] if i + 1 < len(q_starts) else len(text)
        block = text[pos:end_pos]

        # Extract question text (first meaningful line after "Question N:")
        q_text = _extract_question_text(block)
        choices = _extract_choices(block)

        questions.append(
            {
                "num": q_num,
                "text": q_text,
                "choices": choices,
            }
        )

    return questions


def _extract_question_text(block: str) -> str:
    """Extract the question text from a question block."""
    # Remove the "## Question N :" header line
    lines_raw = block.split("\n")
    text_lines: list[str] = []
    skip_header = True

    for line in lines_raw:
        stripped = line.strip()
        # Skip the header line(s)
        if skip_header:
            if re.match(
                r"^(?:##\s*)?(?:QUESTION|Question)\s+\d+", stripped, re.IGNORECASE
            ):
                # Capture text after the colon on the same line
                after_colon = re.sub(
                    r"^(?:##\s*)?(?:QUESTION|Question)\s+\d+\s*:?\s*",
                    "",
                    stripped,
                    flags=re.IGNORECASE,
                )
                if after_colon:
                    text_lines.append(after_colon)
                skip_header = False
                continue
            skip_header = False

        # Stop at choices
        if re.match(r"^\s*-\s*[a-fA-F][\):\-]", stripped):
            break
        if re.match(r"^[A-Fa-f]\s*[-\u2013\u2014]\s*", stripped):
            break

        # Skip empty lines, image placeholders, other headers
        if not stripped or stripped.startswith("<!-- "):
            continue
        # Include continuation headers (## sub-text that's part of the question)
        if stripped.startswith("##"):
            stripped = stripped.lstrip("#").strip()

        text_lines.append(stripped)

    return _clean_text(" ".join(text_lines))


def _extract_choices(block: str) -> dict[str, str]:
    """Extract A/B/C/D/E/F choices from a question block."""
    choices: dict[str, str] = {}
    for pattern in CHOICE_PATTERNS:
        for m in pattern.finditer(block):
            letter = m.group(1).upper()
            choice_text = _clean_text(m.group(2))
            if letter not in choices and choice_text:
                choices[letter] = choice_text
    return choices


# ---------------------------------------------------------------------------
# Correction grid extraction (from grille sections)
# ---------------------------------------------------------------------------


def extract_correction_grid(
    text: str,
    session: str,
    uv: str,
) -> list[dict[str, Any]]:
    """
    Extract correction entries from a grille section.

    Returns list of dicts with keys: num, correct_answer, article_reference, success_rate.
    """
    # Find all pipe-delimited table blocks
    lines = text.split("\n")
    table_blocks = _find_table_blocks(lines)

    if not table_blocks:
        # Some sessions embed the grid in the corrigé section
        return []

    all_corrections: list[dict[str, Any]] = []
    for header_line, data_rows in table_blocks:
        # Skip non-grid tables (player rosters, tournament data, etc.)
        if not _is_correction_grid_table(header_line, data_rows):
            continue
        fmt = _detect_grid_format(header_line, data_rows, session, uv)
        corrections = _parse_grid_rows(data_rows, fmt, session, uv)
        if corrections:
            all_corrections.extend(corrections)

    # Deduplicate by question number — prefer entries with valid MCQ letter
    by_num: dict[int, dict[str, Any]] = {}
    for c in all_corrections:
        qn = c["num"]
        ans = c.get("correct_answer", "")
        is_letter = bool(ans and len(ans) <= 6 and re.match(r"^[A-Fa-f]+$", ans))
        existing = by_num.get(qn)
        if existing is None:
            by_num[qn] = c
        elif is_letter:
            # Prefer entry with MCQ letter over one without
            old_ans = existing.get("correct_answer", "")
            old_is_letter = bool(
                old_ans and len(old_ans) <= 6 and re.match(r"^[A-Fa-f]+$", old_ans)
            )
            if not old_is_letter:
                by_num[qn] = c
    unique = list(by_num.values())

    return unique


def _is_separator_row(line: str) -> bool:
    """Check if a line is a markdown table separator (e.g. |---|---|)."""
    return bool(re.match(r"^\s*\|[\s\-:|]+\|\s*$", line))


def _parse_table_block(
    table_lines: list[str],
) -> tuple[str, list[str]] | None:
    """Parse a consecutive block of pipe-delimited lines into (header, data_rows)."""
    header = ""
    data_start = 0

    for ti, tline in enumerate(table_lines):
        if _is_separator_row(tline):
            continue
        cells = _split_cells(tline)
        if _is_section_header_row(cells):
            continue
        header = tline
        data_start = ti + 1
        break

    if not header:
        return None

    data_rows: list[str] = []
    for row in table_lines[data_start:]:
        if _is_separator_row(row):
            continue
        cells = _split_cells(row)
        if cells and _is_section_header_row(cells):
            continue
        data_rows.append(row)

    if not data_rows:
        return None
    return header, data_rows


def _find_table_blocks(
    lines: list[str],
) -> list[tuple[str, list[str]]]:
    """Find all pipe-delimited table blocks in lines.

    Returns list of (header_line, data_rows) tuples.
    """
    blocks: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(lines):
        if "|" not in lines[i]:
            i += 1
            continue

        table_lines: list[str] = []
        j = i
        while j < len(lines) and "|" in lines[j]:
            table_lines.append(lines[j])
            j += 1

        if len(table_lines) >= 3:
            result = _parse_table_block(table_lines)
            if result:
                blocks.append(result)

        i = j

    return blocks


def _split_cells(line: str) -> list[str]:
    """Split a pipe-delimited row into cells."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _is_section_header_row(cells: list[str]) -> bool:
    """Check if a row is a section header (e.g. 'Questions diverses' repeated)."""
    if not cells:
        return False
    # If all non-empty cells have the same text, it's a header
    non_empty = [c for c in cells if c.strip()]
    if non_empty and len(set(non_empty)) == 1:
        return True
    # Category markers
    first = cells[0].strip().lower()
    if any(
        kw in first
        for kw in [
            "questions diverses",
            "compétitions par équipes",
            "coupe 2000",
            "composition",
            "a - questions",
            "b - match",
            "c - ",
        ]
    ):
        return True
    # Check if this is a grille/corrigé title row (e.g., "UVR - décembre 2022 - grille des réponses")
    joined = " ".join(c.strip().lower() for c in cells)
    return "grille des" in joined and "ponses" in joined


def _is_correction_grid_table(
    header: str,
    data_rows: list[str],
) -> bool:
    """Check if a table block looks like a correction grid (not a content table).

    A correction grid has:
    - Header mentioning 'question', 'réponse', 'article', or 'taux'
    - OR: majority of data rows have a number as first cell and a short answer as second cell
    """
    hl = header.lower()
    # Check header keywords typical of correction grids
    grid_keywords = ["réponse", "reponse", "taux", "grille"]
    if any(kw in hl for kw in grid_keywords):
        return True
    # Check if header has "question" + "article"
    if "question" in hl and ("article" in hl or "référence" in hl or "reference" in hl):
        return True
    # Heuristic: check data rows — if most have a number first cell + short second cell
    num_first_count = 0
    short_second_count = 0
    for row in data_rows[:10]:
        cells = _split_cells(row)
        if len(cells) >= 2:
            first = cells[0].strip()
            second = cells[1].strip()
            if re.match(r"^\d+$", first):
                num_first_count += 1
            if len(second) <= 6:
                short_second_count += 1
    # If most rows have numeric first cell AND short second cell, likely a grid
    check_count = min(len(data_rows), 10)
    return (
        check_count > 0
        and num_first_count >= check_count * 0.7
        and short_second_count >= check_count * 0.7
    )


def _has_merged_rep_taux(cells: list[str]) -> bool:
    """Check if a cell contains merged REPONSE+Taux (jun2022 format)."""
    return any(
        ("reponse" in c.lower() or "réponse" in c.lower()) and "taux" in c.lower()
        for c in cells
    )


def _has_merged_qr(cells: list[str], header_lower: str) -> bool:
    """Check if header has merged Question+Réponse (jun2023 format)."""
    for c in cells:
        cl = c.lower().strip()
        if "question" in cl and ("reponse" in cl or "réponse" in cl):
            return True
    return bool(re.search(r"question\s+r[ée]ponse", header_lower))


def _has_short_cell_keyword(cells: list[str], keyword: str) -> bool:
    """Check if any short cell (< 50 chars) contains keyword."""
    return any(keyword in c.lower() and len(c.strip()) < 50 for c in cells)


def _detect_grid_format(
    header: str,
    data_rows: list[str],
    session: str,
    uv: str,
) -> str:
    """Detect the correction grid format."""
    header_lower = header.lower()
    cells_h = _split_cells(header)

    # Check for 5-column with CADENCE (dec2019 UVR, jun2021 UVR)
    if "cadence" in header_lower or "regle" in header_lower:
        return "5col_cadence"

    # Check for merged REPONSE+Taux in one cell (jun2022)
    if _has_merged_rep_taux(cells_h):
        return "merged_rep_taux"

    # Check for merged Question+Réponse in header (jun2023)
    if _has_merged_qr(cells_h, header_lower):
        return "merged_qr"

    # Check for 5-column with Document (jun2021 UVO)
    has_document_cell = _has_short_cell_keyword(cells_h, "document")
    has_article_cell = _has_short_cell_keyword(cells_h, "article")
    if has_document_cell and has_article_cell:
        # Check if it also has Réponse column
        if "réponse" in header_lower or "reponse" in header_lower:
            return "5col_document"
        # No Réponse column = open-text format with no answer letters
        return "no_answer_col"

    # Check for text-answer format (dec2019 UVC)
    if session == "dec2019" and uv == "UVC":
        return "text_answers"

    # Check for no-answer format (Q + Article + Taux only)
    if (
        "question" in header_lower
        and (
            "article" in header_lower
            or "référence" in header_lower
            or "reference" in header_lower
        )
        and "réponse" not in header_lower
        and "reponse" not in header_lower
    ):
        return "no_answer_col"

    # Check for shifted columns by analyzing data rows
    if _detect_shifted_format(data_rows):
        return "shifted"

    # Default: clean 4-column
    return "clean_4col"


def _detect_shifted_format(data_rows: list[str]) -> bool:
    """Detect if columns are shifted (question numbers at end of rows)."""
    shifted_count = 0
    for row in data_rows[:6]:
        cells = _split_cells(row)
        if len(cells) < 3:
            continue
        cell0 = cells[0].strip()
        # If first cell is a single letter (answer), columns are shifted
        if re.match(r"^[A-Fa-f]$", cell0):
            shifted_count += 1
        # If last cell looks like a question number
        last = cells[-1].strip()
        if re.match(r"^\d{1,2}$", last):
            shifted_count += 1
    return shifted_count >= 2


def _parse_row_by_format(
    cells: list[str],
    fmt: str,
    session: str,
) -> dict[str, Any] | None:
    """Parse a single row according to the detected format."""
    parsers: dict[str, Any] = {
        "clean_4col": lambda c: _parse_clean_4col(c),
        "merged_qr": lambda c: _parse_merged_qr(c),
        "merged_rep_taux": lambda c: _parse_merged_rep_taux(c),
        "5col_document": lambda c: _parse_5col_document(c),
        "text_answers": lambda c: _parse_text_answers(c),
        "shifted": lambda c: _parse_shifted(c),
        "no_answer_col": lambda c: _parse_no_answer_col(c),
    }
    if fmt == "5col_cadence":
        return _parse_5col_cadence(cells, session)
    parser = parsers.get(fmt)
    return parser(cells) if parser else None


def _parse_grid_rows(
    data_rows: list[str],
    fmt: str,
    session: str,
    uv: str,
) -> list[dict[str, Any]]:
    """Parse correction grid rows according to detected format."""
    corrections: list[dict[str, Any]] = []
    for row in data_rows:
        cells = _split_cells(row)
        if not cells or len(cells) < 2:
            continue
        entry = _parse_row_by_format(cells, fmt, session)
        if entry and entry.get("num"):
            corrections.append(entry)
    return corrections


def _extract_qnum(text: str) -> int | None:
    """Extract question number from text."""
    text = text.strip()
    m = re.match(r"^(\d+)", text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 50:  # Reasonable question number range
            return val
    return None


def _extract_letter(text: str) -> str:
    """Extract answer letter(s) from text."""
    text = text.strip().upper()
    # Remove asterisks
    text = re.sub(r"\*", "", text)
    # Handle "A ou D" -> "A"
    text = re.sub(r"\s*OU\s+.*", "", text, flags=re.IGNORECASE)
    # Match single or multi-letter answers like "ADE", "ABDE"
    m = re.match(r"^([A-F]+)$", text)
    if m:
        return m.group(1)
    # Try leading single letter followed by article text (merged cell)
    # e.g. "A ARTICLE 3.6 DU C03..." or "B A02 -" → "A" / "B"
    # Negative lookahead ensures we don't split multi-letter answers like "ADE"
    m = re.match(r"^([A-F])\s+(?![A-F]+$).{3,}", text)
    if m:
        return m.group(1)
    # Try "Réponse c" format
    m = re.search(r"[Rr][ée]ponse\s+([A-Fa-f])", text)
    if m:
        return m.group(1).upper()
    return text


def _parse_taux(text: str) -> float | None:
    """Parse success rate from text."""
    text = text.strip()
    if not text or text.upper() in ("NE", "NEUTRALISÉ", "NEUTRALISE", "-"):
        return None
    # Remove (%) suffix and % sign
    text = re.sub(r"\s*\(%?\)\s*$", "", text)
    text = re.sub(r"\s*%\s*$", "", text)
    text = text.replace(",", ".")
    # Take first number (ignore "75 %+18 %*" patterns)
    text = re.split(r"[+*]", text)[0].strip()
    # Handle "2 erreurs" type
    if re.search(r"[a-zA-Z]", text):
        return None
    try:
        val = float(text)
        if val > 1:
            val = val / 100
        return round(val, 4)
    except ValueError:
        return None


def _parse_clean_4col(cells: list[str]) -> dict[str, Any] | None:
    """Parse: | Question | Réponse | Article | Taux |"""
    if len(cells) < 3:
        return None

    qnum = _extract_qnum(cells[0])
    if not qnum:
        return None

    answer = _extract_letter(cells[1]) if len(cells) > 1 else ""
    article = _clean_text(cells[2]) if len(cells) > 2 else ""
    taux = _parse_taux(cells[3]) if len(cells) > 3 else None

    # Handle case where taux is embedded in article cell
    if taux is None and len(cells) > 2:
        # Check if article cell ends with a percentage
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:\(%?\))?\s*$", cells[2])
        if m:
            taux = _parse_taux(m.group(1))
            article = _clean_text(cells[2][: m.start()])

    return {
        "num": qnum,
        "correct_answer": answer,
        "article_reference": article,
        "success_rate": taux,
    }


def _find_5col_answer(cells: list[str]) -> tuple[str, str, str]:
    """Find answer letter, article, and taux in a 5-column row with shifting."""
    # Try column positions in order: [2], [3], [1]
    for ans_idx in (2, 3, 1):
        if len(cells) <= ans_idx:
            continue
        candidate = _extract_letter(cells[ans_idx])
        if candidate and re.match(r"^[A-F]+$", candidate):
            # Determine article and taux positions based on answer position
            if ans_idx == 2:
                art = _clean_text(cells[3]) if len(cells) > 3 else ""
                taux = cells[4] if len(cells) > 4 else ""
            elif ans_idx == 3:
                art = _clean_text(cells[2]) if cells[2].strip() else ""
                taux = cells[4] if len(cells) > 4 else ""
            else:  # ans_idx == 1
                art = _clean_text(cells[2])
                taux = cells[3] if len(cells) > 3 else ""
            return candidate, art, taux
    return "", "", ""


def _parse_5col_cadence(
    cells: list[str],
    session: str,
) -> dict[str, Any] | None:
    """Parse: | Q REGLE | CADENCE | REPONSE | ARTICLE | Taux |

    Dec2019/Jun2021 UVR format. Question numbers sometimes missing from first
    cell, and sometimes rule text is merged with question number.
    """
    if len(cells) < 3:
        return None

    # Try to get question number from first cell
    cell0 = cells[0].strip()
    qnum = _extract_qnum(cell0)

    # Special: dec2019 has rows where qnum is missing but embedded elsewhere
    # e.g. "| Nulle | Lente | B | Art 9.1.2. | 8 81,88% |"
    if not qnum and len(cells) >= 5:
        # Try extracting from taux cell (sometimes merged)
        taux_cell = cells[4].strip() if len(cells) > 4 else cells[-1].strip()
        m = re.match(r"^(\d+)\s+", taux_cell)
        if m:
            qnum = int(m.group(1))
            if 1 <= qnum <= 50:
                cells[4] = taux_cell[m.end() :]  # Fix the taux cell
            else:
                qnum = None

    if not qnum:
        return None

    # Columns: [0]=Q+REGLE, [1]=CADENCE, [2]=REPONSE, [3]=ARTICLE, [4]=Taux
    # But shifting happens - find the answer letter
    answer, article, taux_str = _find_5col_answer(cells)

    # Handle question numbers with suffixes like "21C", "23J"
    m_suffix = re.match(r"^(\d+)\s*([CJ])\s", cell0)
    if m_suffix:
        qnum = int(m_suffix.group(1))

    return {
        "num": qnum,
        "correct_answer": answer,
        "article_reference": article,
        "success_rate": _parse_taux(taux_str),
    }


def _parse_merged_qr(cells: list[str]) -> dict[str, Any] | None:
    """Parse: | 1 a | Article | Taux | | (jun2023 format)."""
    if len(cells) < 2:
        return None

    cell0 = cells[0].strip()
    # Expected: "N LETTER" like "1 a" or "15 d"
    m = re.match(r"^(\d+)\s+([a-fA-F]+)$", cell0)
    if not m:
        # Try without letter (some rows might have different format)
        qnum = _extract_qnum(cell0)
        if not qnum:
            return None
        # Look for letter in next cell
        answer = _extract_letter(cells[1]) if len(cells) > 1 else ""
        article = _clean_text(cells[2]) if len(cells) > 2 else ""
        taux_str = cells[3] if len(cells) > 3 else ""
        return {
            "num": qnum,
            "correct_answer": answer,
            "article_reference": article,
            "success_rate": _parse_taux(taux_str),
        }

    qnum = int(m.group(1))
    answer = m.group(2).upper()
    article = _clean_text(cells[1]) if len(cells) > 1 else ""
    taux_str = cells[2] if len(cells) > 2 else ""

    return {
        "num": qnum,
        "correct_answer": answer,
        "article_reference": article,
        "success_rate": _parse_taux(taux_str),
    }


def _parse_merged_rep_taux(cells: list[str]) -> dict[str, Any] | None:
    """Parse: | QNUM | ARTICLE | LETTER TAUX | (jun2022 format).

    Answer letter and taux are merged: "D 80", "A RI 98.8", "C Article 93.8".
    """
    if len(cells) < 2:
        return None

    qnum = _extract_qnum(cells[0])
    if not qnum:
        return None

    # Article is in cell[1]
    article = _clean_text(cells[1]) if len(cells) > 1 else ""

    # Find the cell with merged LETTER+TAUX (usually last non-empty cell)
    answer = ""
    taux_str = ""

    for ci in range(len(cells) - 1, 0, -1):
        cell = cells[ci].strip()
        if not cell:
            continue
        # Pattern: "D 80" or "A 93.8" or "A RI 98.8" or "B Article 1.1 du 96.3"
        m = re.match(r"^([A-Fa-f])\s+(.*)$", cell)
        if m:
            answer = m.group(1).upper()
            # Extract taux from remainder
            remainder = m.group(2).strip()
            # Find last number in remainder (that's the taux)
            taux_match = re.search(r"(\d+(?:[.,]\d+)?)\s*$", remainder)
            if taux_match:
                taux_str = taux_match.group(1)
                # Any text before the taux is additional article info
                extra_article = remainder[: taux_match.start()].strip()
                if extra_article and len(extra_article) > 3:
                    article = _clean_text(f"{article} {extra_article}")
            else:
                # Maybe the whole remainder is a taux
                taux_str = remainder
            break

    return {
        "num": qnum,
        "correct_answer": answer,
        "article_reference": article,
        "success_rate": _parse_taux(taux_str),
    }


def _parse_5col_document(cells: list[str]) -> dict[str, Any] | None:
    """Parse: | Q | Document | Article | Réponse | Taux | (jun2021 UVO)."""
    if len(cells) < 4:
        return None

    qnum = _extract_qnum(cells[0])
    if not qnum:
        return None

    # Cells: [0]=Q, [1]=Document, [2]=Article, [3]=Réponse, [4]=Taux
    document = cells[1].strip() if len(cells) > 1 else ""
    article = cells[2].strip() if len(cells) > 2 else ""
    answer = _extract_letter(cells[3]) if len(cells) > 3 else ""
    taux_str = cells[4] if len(cells) > 4 else ""

    full_article = _clean_text(f"{document} {article}".strip())

    return {
        "num": qnum,
        "correct_answer": answer,
        "article_reference": full_article,
        "success_rate": _parse_taux(taux_str),
    }


def _parse_text_answers(cells: list[str]) -> dict[str, Any] | None:
    """Parse: | N° | QUESTION | ARTICLE | REPONSE=text | Taux | (dec2019 UVC).

    Answers are full text, not letters.
    """
    if len(cells) < 4:
        return None

    qnum = _extract_qnum(cells[0])
    if not qnum:
        return None

    article = _clean_text(cells[2]) if len(cells) > 2 else ""
    answer_text = _clean_text(cells[3]) if len(cells) > 3 else ""
    taux_str = cells[4] if len(cells) > 4 else ""

    # Try to extract a letter if possible (e.g., "Réponse c")
    letter = ""
    m = re.search(r"[Rr][ée]ponse\s+([A-Fa-f])\b", answer_text)
    if m:
        letter = m.group(1).upper()

    return {
        "num": qnum,
        "correct_answer": letter or answer_text,
        "article_reference": article,
        "success_rate": _parse_taux(taux_str),
    }


def _parse_shifted_last(cells: list[str], qnum: int) -> dict[str, Any]:
    """Parse row with qnum at the end: | LETTER | Article... | Taux | QNUM |."""
    answer = _extract_letter(cells[0])
    article_parts: list[str] = []
    taux_str = ""
    for part in cells[1:-1]:
        stripped = part.strip()
        if re.match(r"^\d+(?:[.,]\d+)?\s*(?:\(%?\))?$", stripped):
            taux_str = stripped
        else:
            article_parts.append(stripped)
    return {
        "num": qnum,
        "correct_answer": answer,
        "article_reference": _clean_text(" ".join(article_parts)),
        "success_rate": _parse_taux(taux_str),
    }


def _parse_shifted_scan(cells: list[str]) -> dict[str, Any] | None:
    """Parse shifted row by scanning for qnum in any cell."""
    candidate = _extract_letter(cells[0].strip())
    if not candidate or not re.match(r"^[A-F]+$", candidate) or len(cells) < 3:
        return None
    qnum = None
    for ci in range(1, len(cells)):
        cell_qnum = _extract_qnum(cells[ci])
        if cell_qnum:
            qnum = cell_qnum
            break
    if not qnum:
        return None
    remaining = [
        cells[ci].strip()
        for ci in range(1, len(cells))
        if _extract_qnum(cells[ci]) is None
    ]
    article = _clean_text(" ".join(remaining[:-1])) if len(remaining) > 1 else ""
    taux_str = remaining[-1] if remaining else ""
    return {
        "num": qnum,
        "correct_answer": candidate,
        "article_reference": article,
        "success_rate": _parse_taux(taux_str),
    }


def _parse_shifted(cells: list[str]) -> dict[str, Any] | None:
    """Parse shifted columns (dec2022 UVC) where qnum is at end of row."""
    if len(cells) < 3:
        return None

    first_qnum = _extract_qnum(cells[0].strip())
    last_qnum = _extract_qnum(cells[-1].strip())

    if first_qnum and len(cells) >= 4:
        return {
            "num": first_qnum,
            "correct_answer": _extract_letter(cells[1]),
            "article_reference": _clean_text(cells[2]),
            "success_rate": _parse_taux(cells[3] if len(cells) > 3 else ""),
        }
    if last_qnum and len(cells) >= 3:
        return _parse_shifted_last(cells, last_qnum)
    return _parse_shifted_scan(cells)


def _parse_no_answer_col(cells: list[str]) -> dict[str, Any] | None:
    """Parse grid with no answer column: | Q | Document/Article | Taux |.

    Used for open-text UVs where no MCQ letter is provided in the grid.
    Still extracts question numbers and article references.
    """
    if len(cells) < 2:
        return None

    # Handle sub-question numbering like "1a", "1b", "1c"
    cell0 = cells[0].strip()
    m = re.match(r"^(\d+)([a-z])?$", cell0)
    if not m:
        return None
    qnum = int(m.group(1))
    if not (1 <= qnum <= 50):
        return None

    # Collect article reference from middle cells
    article_parts = [cells[ci].strip() for ci in range(1, len(cells) - 1)]
    article = _clean_text(" ".join(article_parts))
    taux_str = cells[-1] if len(cells) > 2 else ""

    return {
        "num": qnum,
        "correct_answer": "",
        "article_reference": article,
        "success_rate": _parse_taux(taux_str),
    }


# ---------------------------------------------------------------------------
# Correction from corrigé sections (fallback for missing grilles)
# ---------------------------------------------------------------------------


def extract_corrections_from_corrige(
    text: str,
) -> list[dict[str, Any]]:
    """
    Extract correction info from corrigé détaillé sections.

    Used as fallback when no grille section is found.
    Looks for patterns like "Réponse : A" or "La bonne réponse est A".
    """
    corrections: list[dict[str, Any]] = []

    # Split by question headers
    q_starts: list[tuple[int, int]] = []
    for m in QUESTION_NUM_RE.finditer(text):
        q_starts.append((m.start(), int(m.group(1))))

    for i, (pos, q_num) in enumerate(q_starts):
        end_pos = q_starts[i + 1][0] if i + 1 < len(q_starts) else len(text)
        block = text[pos:end_pos]

        # Look for answer patterns in the block
        answer = ""
        article = ""

        # "Réponse : A" or "Bonne réponse : A"
        m_ans = re.search(
            r"(?:bonne\s+)?r[ée]ponse\s*:?\s*([A-Fa-f])\b",
            block,
            re.IGNORECASE,
        )
        if m_ans:
            answer = m_ans.group(1).upper()

        # "Article X.Y"
        m_art = re.search(
            r"(?:Article|Art\.?)\s+([A-Z0-9][\w.]+)",
            block,
            re.IGNORECASE,
        )
        if m_art:
            article = m_art.group(0)

        if answer:
            corrections.append(
                {
                    "num": q_num,
                    "correct_answer": answer,
                    "article_reference": _clean_text(article),
                    "success_rate": None,
                }
            )

    return corrections


# ---------------------------------------------------------------------------
# Session extraction
# ---------------------------------------------------------------------------


def _find_corrections(
    sections: dict[str, str],
    session: str,
    uv_code: str,
) -> tuple[list[dict[str, Any]], str]:
    """Find correction entries from available section sources (grille/sujet/corrigé)."""
    corrections: list[dict[str, Any]] = []
    source = ""

    if "grille" in sections:
        corrections = extract_correction_grid(sections["grille"], session, uv_code)
        source = "grille"

    if not corrections and "sujet" in sections:
        corrections = extract_correction_grid(sections["sujet"], session, uv_code)
        if corrections:
            source = "sujet-embedded"

    if not corrections and "corrige" in sections:
        corrections = extract_correction_grid(sections["corrige"], session, uv_code)
        if corrections:
            source = "corrigé-tables"

    if not corrections and "corrige" in sections:
        corrections = extract_corrections_from_corrige(sections["corrige"])
        if corrections:
            source = "corrigé-text"

    return corrections, source


def _build_entry(
    corr: dict[str, Any],
    q_lookup: dict[int, dict[str, Any]],
    session: str,
    uv_name: str,
) -> dict[str, Any]:
    """Build a reference entry from a correction + question data."""
    q_num = corr["num"]
    correct_answer = corr.get("correct_answer", "")

    if (
        correct_answer
        and len(correct_answer) <= 6
        and re.match(r"^[A-Fa-f]+$", correct_answer)
    ):
        correct_answer = correct_answer.upper()

    q_info = q_lookup.get(q_num, {})
    return {
        "session": session,
        "uv": uv_name,
        "question_num": q_num,
        "question_text": q_info.get("text", ""),
        "choices": q_info.get("choices", {}),
        "correct_letter": correct_answer if len(correct_answer) <= 6 else "",
        "correct_text": correct_answer if len(correct_answer) > 6 else "",
        "article_reference": corr.get("article_reference", ""),
        "success_rate": corr.get("success_rate"),
    }


def extract_session(
    session: str,
    json_path: Path,
) -> list[dict[str, Any]]:
    """
    Extract all MCQ reference data from a single session.

    Returns list of reference entries with session, uv, question_num, etc.
    """
    if not json_path.exists():
        logger.warning(f"File not found: {json_path}")
        return []

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    markdown = data.get("markdown", "")
    if not markdown:
        logger.warning(f"No markdown in {json_path}")
        return []

    logger.info(f"\nProcessing {session} ({json_path.name})")
    uv_sections = split_into_uv_sections(markdown)
    logger.info(f"  Found UV sections: {list(uv_sections.keys())}")

    entries: list[dict[str, Any]] = []

    for uv_code in ["UVR", "UVC", "UVO", "UVT"]:
        sections = uv_sections.get(uv_code)
        if not sections:
            continue

        uv_name = UV_CANONICAL.get(uv_code, uv_code.lower())
        corrections, corr_source = _find_corrections(sections, session, uv_code)
        logger.info(
            f"  {session} {uv_code}: {len(corrections)} corrections"
            + (f" from {corr_source}" if corr_source else " (none found)")
        )

        questions = _find_questions(sections, session, uv_code)
        q_lookup = {q["num"]: q for q in questions}

        for corr in corrections:
            entries.append(_build_entry(corr, q_lookup, session, uv_name))

    logger.info(f"  Total: {len(entries)} entries for {session}")
    return entries


def _find_questions(
    sections: dict[str, str],
    session: str,
    uv_code: str,
) -> list[dict[str, Any]]:
    """Extract questions from sujet or corrigé sections."""
    questions: list[dict[str, Any]] = []
    if "sujet" in sections:
        questions = extract_questions(sections["sujet"])
        logger.info(f"  {session} {uv_code}: {len(questions)} questions from sujet")
    if not questions and "corrige" in sections:
        questions = extract_questions(sections["corrige"])
        if questions:
            logger.info(
                f"  {session} {uv_code}: {len(questions)} questions from corrigé"
            )
    return questions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Extract MCQ reference data from all Docling sessions."""
    project_root = Path(__file__).resolve().parent.parent
    output_path = (
        project_root / "data" / "evaluation" / "annales" / "docling_mcq_reference.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_entries: list[dict[str, Any]] = []

    for session, rel_path in SESSION_FILES.items():
        json_path = project_root / rel_path
        entries = extract_session(session, json_path)
        all_entries.extend(entries)

    # Save reference
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    logger.info(f"\nSaved {len(all_entries)} reference entries to {output_path}")

    # Summary
    _print_summary(all_entries)

    # Alignment check
    _check_alignment(all_entries)


def _print_summary(entries: list[dict[str, Any]]) -> None:
    """Print extraction summary."""
    from collections import Counter

    session_counts = Counter(e["session"] for e in entries)
    uv_counts = Counter(e["uv"] for e in entries)
    with_letter = sum(1 for e in entries if e["correct_letter"])
    with_text = sum(1 for e in entries if e["question_text"])
    with_choices = sum(1 for e in entries if e["choices"])

    print("\n=== MCQ Reference Extraction Summary ===")
    print(f"Total entries: {len(entries)}")
    print(f"With correct_letter: {with_letter}")
    print(f"With question_text: {with_text}")
    print(f"With choices: {with_choices}")
    print("\nBy session:")
    for session in sorted(session_counts):
        count = session_counts[session]
        letters = sum(
            1 for e in entries if e["session"] == session and e["correct_letter"]
        )
        texts = sum(
            1 for e in entries if e["session"] == session and e["question_text"]
        )
        print(f"  {session}: {count} entries, {letters} letters, {texts} texts")
    print("\nBy UV:")
    for uv in sorted(uv_counts):
        print(f"  {uv}: {uv_counts[uv]}")


def _check_alignment(entries: list[dict[str, Any]]) -> None:
    """Check alignment quality of extracted data."""
    issues: list[str] = []

    for e in entries:
        sid = f"{e['session']}/{e['uv']}/Q{e['question_num']}"

        # Check: has correct_letter
        if not e["correct_letter"] and not e.get("correct_text"):
            issues.append(f"{sid}: no answer (letter or text)")

        # Check: letter is valid
        letter = e.get("correct_letter", "")
        if letter and not re.match(r"^[A-F]+$", letter):
            issues.append(f"{sid}: invalid letter '{letter}'")

        # Check: if has choices and letter, letter should be in choices
        choices = e.get("choices", {})
        if letter and choices and len(letter) == 1 and letter not in choices:
            issues.append(
                f"{sid}: letter {letter} not in choices {list(choices.keys())}"
            )

    print("\n=== Alignment Check ===")
    print(f"Total issues: {len(issues)}")
    if issues:
        for issue in issues[:30]:
            print(f"  {issue}")
        if len(issues) > 30:
            print(f"  ... and {len(issues) - 30} more")


if __name__ == "__main__":
    main()
