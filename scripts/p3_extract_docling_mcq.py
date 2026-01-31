#!/usr/bin/env python3
"""
Extract MCQ questions and correction grids from Docling JSON files.

Parses the markdown field of each Docling JSON to extract:
- Questions with their choices (A/B/C/D)
- Correction grids with correct answers, article references, success rates

Handles multiple correction table formats found across sessions:
- Clean 4-column: | N | LETTER | article | taux |
- Merged Q+R: | N LETTER | article | taux | |
- Shifted columns: | LETTER | article | taux | N |
- 5-column with CADENCE: | N RULE | CADENCE | REPONSE | ARTICLE | taux |
- Merged REPONSE+Taux: | N | article | LETTER taux |
- UVT without REPONSE: | Q | ARTICLE | Taux |

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

# UV name mapping (GS uses lowercase English names)
UV_MAP: dict[str, str] = {
    "UVR": "rules",
    "UVC": "clubs",
    "UVO": "open",
    "UVT": "tournament",
}

# Question block pattern
QUESTION_BLOCK_RE = re.compile(
    r"(?:^|\n)(?:##\s*)?Question\s+(\d+)\s*:\s*(.+?)"
    r"(?=(?:\n(?:##\s*)?Question\s+\d+\s*:)"
    r"|(?:\n##\s+UV)"
    r"|(?:\n##\s+Partie\s+[A-Z])"
    r"|(?:\n##\s*Corrig)"
    r"|(?:\n##\s*Fin)"
    r"|(?:\n##\s*Grille)"
    r"|(?:\n[0-9]+\s*%\s*de\s*bonnes)"
    r"|$)",
    re.DOTALL | re.IGNORECASE,
)

# Choice extraction patterns (ordered by specificity)
CHOICE_PATTERNS = [
    # "- a)" format
    re.compile(
        r"(?:^|\n)\s*-\s*([a-fA-F])\)\s*(.+?)(?=(?:\n\s*-\s*[a-fA-F]\))|$)",
        re.DOTALL | re.IGNORECASE,
    ),
    # "- A - " format
    re.compile(
        r"(?:^|\n)\s*-\s*([A-F])\s*[-\u2013\u2014]\s*(.+?)"
        r"(?=(?:\n\s*-\s*[A-F]\s*[-\u2013\u2014])|$)",
        re.DOTALL,
    ),
    # "A :" format
    re.compile(
        r"(?:^|\n)\s*(?:-\s*)?([A-F])\s*:\s*(.+?)" r"(?=(?:\n\s*(?:-\s*)?[A-F]\s*:)|$)",
        re.DOTALL,
    ),
    # "A - " format (no leading dash)
    re.compile(
        r"(?:^|\n)(?:##\s*)?([A-F])\s*[-\u2013\u2014]\s*(.+?)"
        r"(?=(?:\n(?:##\s*)?[A-F]\s*[-\u2013\u2014])|$)",
        re.DOTALL,
    ),
    # "a." format
    re.compile(
        r"(?:^|\n)\s*([a-fA-F])\.\s*(.+?)(?=(?:\n\s*[a-fA-F]\.)|$)",
        re.DOTALL | re.IGNORECASE,
    ),
    # Permissive fallback
    re.compile(
        r"(?:^|\n)\s*-?\s*([A-F])[-\u2013\u2014\s]*"
        r"([A-Za-z\u00e9\u00e8\u00ea\u00eb\u00e0\u00e2\u00e4\u00f9\u00fb"
        r"\u00fc\u00ee\u00ef\u00f4\u00f6\u00e7\u00c0\u00c2\u00c4\u00c9"
        r"\u00c8\u00ca\u00cb\u00ce\u00cf\u00d4\u00d6\u00d9\u00db\u00dc]"
        r"[^-\u2013\u2014\n]*?)"
        r"(?=(?:\n\s*-?\s*[A-F][-\u2013\u2014\s]*[A-Za-z])|(?:\n##)|$)",
        re.DOTALL,
    ),
]


def _clean_text(text: str) -> str:
    """Clean extracted text (normalize whitespace)."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_choices(block: str) -> dict[str, str]:
    """Extract A/B/C/D choices from a question block."""
    choices: dict[str, str] = {}
    for pattern in CHOICE_PATTERNS:
        for match in pattern.finditer(block):
            letter = match.group(1).upper()
            choice_text = _clean_text(match.group(2))
            if letter not in choices and choice_text:
                choices[letter] = choice_text
    return choices


def extract_questions_from_markdown(markdown: str) -> list[dict[str, Any]]:
    """Extract all questions with choices from markdown."""
    questions = []
    commentary_re = re.compile(
        r"^\s*(?:\d+\s*%\s*des\s*candidats"
        r"|Il\s+s['\u2019]agissait"
        r"|pas\s+[e\u00e9]t[e\u00e9]\s+comptabilis[e\u00e9]e)",
        re.IGNORECASE,
    )

    for match in QUESTION_BLOCK_RE.finditer(markdown):
        q_num = int(match.group(1))
        q_content = match.group(2)

        if commentary_re.search(q_content.strip()):
            continue

        # Extract question text (before first choice)
        q_text_match = re.search(
            r"^(.+?)(?=\n\s*-\s*[a-dA-D][\)\-])", q_content, re.DOTALL
        )
        q_text = (
            _clean_text(q_text_match.group(1))
            if q_text_match
            else _clean_text(q_content.split("\n")[0])
        )

        choices = _extract_choices(q_content)

        questions.append(
            {
                "num": q_num,
                "text": q_text,
                "choices": choices,
            }
        )

    return questions


def group_questions_by_sequence(
    questions: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group questions into UV sequences (each UV starts at Q1)."""
    if not questions:
        return []

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for q in questions:
        if q["num"] == 1 and current:
            groups.append(current)
            current = []
        current.append(q)

    if current:
        groups.append(current)

    return groups


def _parse_taux(text: str) -> float | None:
    """Parse success rate from various formats."""
    text = text.strip()
    if not text or text.upper() in ("NE", "NEUTRALIS\u00c9", "NEUTRALISE"):
        return None
    # Remove (%) suffix
    text = re.sub(r"\s*\(%\)\s*$", "", text)
    text = re.sub(r"\s*%\s*$", "", text)
    # Handle comma decimal: "85,9" -> "85.9"
    text = text.replace(",", ".")
    # Handle "75 %+18 %*" -> take first number
    text = re.split(r"[+*]", text)[0].strip()
    try:
        val = float(text)
        if val > 1:
            val = val / 100
        return round(val, 4)
    except ValueError:
        return None


def _extract_answer_letter(text: str) -> str | None:
    """Extract answer letter(s) from text."""
    text = text.strip().upper()
    # Handle "Aou D*" -> "A"
    text = re.sub(r"\s*OU\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*.*", "", text)
    # Match single or multi-letter answers
    m = re.match(r"^([A-F]+)$", text)
    if m:
        return m.group(1)
    # Handle "Réponse c" format
    m = re.search(r"[Rr]\u00e9ponse\s+([A-Fa-f])", text)
    if m:
        return m.group(1).upper()
    return None


def _detect_uv_from_context(lines: list[str], table_start: int) -> str | None:
    """Detect UV type from lines preceding a table."""
    # Use word boundary to avoid matching UVO inside "pouvoir" etc.
    uv_re = re.compile(r"\bUV[RCOT]\b")

    # Check table header and surrounding lines for UV mention
    for i in range(max(0, table_start - 15), min(len(lines), table_start + 3)):
        line = lines[i] if i < len(lines) else ""
        # Skip table-of-contents lines (have many dots)
        if "........" in line:
            continue
        m = uv_re.search(line.upper())
        if m:
            return m.group(0)

    # Search further back for section headers (## UV...)
    # Need large range because image placeholders can create big gaps
    for i in range(max(0, table_start - 100), max(0, table_start - 15)):
        line = lines[i] if i < len(lines) else ""
        if line.strip().startswith("##") or "grille" in line.lower():
            m = uv_re.search(line.upper())
            if m:
                return m.group(0)

    return None


def parse_correction_table_from_markdown(
    markdown: str,
    session: str,
) -> dict[str, list[dict[str, Any]]]:
    """
    Parse all correction tables from markdown.

    Returns dict mapping UV type to list of correction entries.
    Each entry has: num, correct_answer, article_reference, success_rate.
    """
    lines = markdown.split("\n")
    uv_corrections: dict[str, list[dict[str, Any]]] = {}

    # Find table sections by looking for pipe-delimited lines with keywords
    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip non-table lines
        if "|" not in line:
            i += 1
            continue

        # Check if this is a correction table header
        line_lower = line.lower()
        is_header = (
            (
                "question" in line_lower
                or "r\u00e9ponse" in line_lower
                or "reponse" in line_lower
            )
            and "grille" not in line_lower
            and "session" not in line_lower
            and "corrig" not in line_lower
            and "contact" not in line_lower
            and "........" not in line
        )

        if not is_header:
            i += 1
            continue

        # Detect UV from context
        uv = _detect_uv_from_context(lines, i)

        # Collect all rows of this table
        table_rows: list[str] = [line]
        j = i + 1
        while j < len(lines) and "|" in lines[j]:
            table_rows.append(lines[j])
            j += 1

        # Parse the table
        corrections = _parse_table_rows(table_rows, session)

        if corrections and uv:
            uv_key = uv
            if uv_key in uv_corrections:
                # Use whichever has more corrections
                if len(corrections) > len(uv_corrections[uv_key]):
                    uv_corrections[uv_key] = corrections
            else:
                uv_corrections[uv_key] = corrections
            logger.info(f"  {session} {uv}: {len(corrections)} corrections from table")

        i = j

    return uv_corrections


def _split_pipe_row(line: str) -> list[str]:
    """Split a pipe-delimited table row into cells."""
    # Remove leading/trailing pipe and split
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _parse_table_rows(
    rows: list[str],
    session: str,
) -> list[dict[str, Any]]:
    """Parse correction table rows with robust format detection."""
    if len(rows) < 2:
        return []

    header = rows[0]

    # Skip separator rows
    data_rows = []
    for row in rows[1:]:
        stripped = row.strip()
        if stripped.startswith("|---") or re.match(r"^\|[\s\-|]+\|$", stripped):
            continue
        # Skip section header duplicates (e.g., "| Composition des équipes...")
        cells = _split_pipe_row(stripped)
        if cells and any(
            kw in " ".join(cells).lower()
            for kw in [
                "composition",
                "arbitrage d'une",
                "calculs des prix",
                "partie 1",
                "partie 2",
                "coupe 2000",
                "questions diverses",
            ]
        ):
            continue
        data_rows.append(stripped)

    # Detect table format from header
    fmt = _detect_table_format(header, data_rows, session)
    logger.debug(f"  Format detected: {fmt}")

    corrections = []
    # Stateful: track last seen question number for shifted formats
    last_qnum = 0

    for row_str in data_rows:
        cells = _split_pipe_row(row_str)
        if not cells or len(cells) < 2:
            continue

        entry = _parse_row_by_format(cells, fmt, last_qnum, session)
        if entry and entry.get("num"):
            last_qnum = entry["num"]
            corrections.append(entry)

    return corrections


def _detect_table_format(
    header: str,
    data_rows: list[str],
    session: str,
) -> str:
    """Detect correction table format from header and first data rows."""
    fmt = _detect_format_from_header(header)
    if fmt:
        return fmt

    # Check data rows to detect shifted or mixed formats
    return _detect_format_from_data_rows(data_rows)


def _detect_format_from_header(header: str) -> str | None:
    """Detect correction table format from header keywords."""
    header_lower = header.lower()
    cells_h = _split_pipe_row(header)

    if "cadence" in header_lower or "regle" in header_lower:
        return "5col_cadence"

    if "reponse" in header_lower and "taux" in header_lower:
        for cell in cells_h:
            cl = cell.lower()
            if ("reponse" in cl or "réponse" in cl) and "taux" in cl:
                return "merged_rep_taux"

    if "question" in header_lower and "article" in header_lower:
        h_art = header_lower.find("article")
        h_rep = max(header_lower.find("reponse"), header_lower.find("réponse"))
        if h_art >= 0 and h_rep >= 0 and h_art < h_rep:
            return "nq_art_rep_taux"

    if _has_merged_qr_header(cells_h, header_lower):
        return "merged_qr"

    if len(cells_h) >= 5 and "document" in header_lower:
        return "q_doc_art_rep_taux"

    if _has_no_reponse_header(header_lower):
        return "q_art_taux_no_rep"

    return None


def _has_merged_qr_header(cells_h: list[str], header_lower: str) -> bool:
    """Check if header has merged Question+Réponse in one cell."""
    for c in cells_h:
        cl = c.lower()
        if "question" in cl and ("reponse" in cl or "réponse" in cl):
            return True
    return bool(re.search(r"question\s+reponse", header_lower))


def _has_no_reponse_header(header_lower: str) -> bool:
    """Check if header has QUESTION+ARTICLE but no REPONSE column."""
    return (
        "question" in header_lower
        and ("article" in header_lower or "reference" in header_lower)
        and "reponse" not in header_lower
        and "réponse" not in header_lower
    )


def _detect_format_from_data_rows(data_rows: list[str]) -> str:
    """Detect format by analyzing first data rows for shifted/merged patterns."""
    if not data_rows or len(data_rows) < 3:
        return "clean_4col"

    counts = _count_data_row_patterns(data_rows)

    if counts["shifted"] >= 2:
        return "mixed_shifted"
    if counts["merged_rep_taux"] >= 2:
        return "merged_rep_taux"
    if counts["answer_merged_article"] >= 3:
        return "clean_4col"

    return "clean_4col"


def _count_data_row_patterns(data_rows: list[str]) -> dict[str, int]:
    """Count occurrences of various format patterns in data rows."""
    shifted = 0
    merged_rep_taux = 0
    answer_merged_article = 0

    for row_str in data_rows[:6]:
        cells = _split_pipe_row(row_str)
        if len(cells) < 3:
            continue
        cell0 = cells[0].strip()
        cell1 = cells[1].strip() if len(cells) > 1 else ""

        if cell0 and re.match(r"^[A-Fa-f]$", cell0):
            shifted += 1
        if not cell0 and cell1 and re.match(r"^[A-Fa-f]\s+", cell1):
            shifted += 1

        for cell in cells:
            if re.match(r"^[A-Fa-f]\s+\d+(?:[.,]\d+)?$", cell.strip()):
                merged_rep_taux += 1

        if cell1 and re.match(r"^[A-Fa-f]\s+[A-Z0-9]", cell1):
            answer_merged_article += 1

    return {
        "shifted": shifted,
        "merged_rep_taux": merged_rep_taux,
        "answer_merged_article": answer_merged_article,
    }


def _parse_row_by_format(
    cells: list[str],
    fmt: str,
    last_qnum: int,
    session: str,
) -> dict[str, Any] | None:
    """Parse a single table row based on detected format."""
    if fmt == "clean_4col":
        return _parse_clean_4col(cells, last_qnum)
    elif fmt == "merged_qr":
        return _parse_merged_qr(cells)
    elif fmt == "mixed_shifted":
        return _parse_mixed_shifted(cells, last_qnum)
    elif fmt == "5col_cadence":
        return _parse_5col_cadence(cells, last_qnum)
    elif fmt == "merged_rep_taux":
        return _parse_merged_rep_taux(cells)
    elif fmt == "nq_art_rep_taux":
        return _parse_nq_art_rep_taux(cells)
    elif fmt == "q_doc_art_rep_taux":
        return _parse_q_doc_art_rep_taux(cells)
    elif fmt == "q_art_taux_no_rep":
        return _parse_q_art_taux_no_rep(cells)
    else:
        return _parse_clean_4col(cells, last_qnum)


def _extract_merged_answer_article(
    cell1: str,
    article: str,
) -> tuple[str | None, str]:
    """Extract answer letter from cell1 when merged with article text."""
    # Try: "A LA. Art 6.3.1..." -> answer=A, article=rest
    m = re.match(r"^([A-Fa-f])\s+(.+)", cell1)
    if m:
        return m.group(1).upper(), (m.group(2) + " " + article).strip()
    # Try: "B Annexe B - B2" where answer and article are merged
    if re.match(r"^([A-Fa-f])\s+[A-Z]", cell1):
        return cell1[0].upper(), (cell1[1:].strip() + " " + article).strip()
    return None, article


def _normalize_taux_str(taux_str: str, article: str) -> tuple[str, str]:
    """Extract taux from article when merged, and normalize taux_str format."""
    if not _parse_taux(taux_str) and article:
        m = re.search(r"\s+(\d+(?:[.,]\d+)?)\s*(?:\(%\))?\s*$", article)
        if m:
            taux_str = m.group(1)
            article = article[: m.start()].strip()
    if taux_str:
        m2 = re.search(r"(\d+(?:[.,]\d+)?)\s*\(%\)", taux_str)
        if m2:
            taux_str = m2.group(1)
    return taux_str, article


def _parse_clean_4col(
    cells: list[str],
    last_qnum: int,
) -> dict[str, Any] | None:
    """Parse: | QNUM | LETTER | article | taux |"""
    if len(cells) < 3:
        return None

    qnum = _extract_qnum(cells[0])
    cell1 = cells[1].strip()
    answer = _extract_answer_letter(cell1)
    article = cells[2].strip() if len(cells) > 2 else ""

    if not answer and cell1:
        answer, article = _extract_merged_answer_article(cell1, article)

    taux_str = cells[3].strip() if len(cells) > 3 else ""
    taux_str, article = _normalize_taux_str(taux_str, article)

    if not qnum and cell1:
        return _parse_clean_4col_shifted(cells, cell1, last_qnum)

    if not qnum:
        return None

    return {
        "num": qnum,
        "correct_answer": answer or "",
        "article_reference": _clean_text(article),
        "success_rate": _parse_taux(taux_str),
    }


def _parse_clean_4col_shifted(
    cells: list[str],
    cell1: str,
    last_qnum: int,
) -> dict[str, Any] | None:
    """Parse clean_4col row where cell0 is empty and data is shifted."""
    m_ans = re.match(r"^([A-Fa-f])\s+(.+)", cell1)
    if not m_ans:
        return None

    answer = m_ans.group(1).upper()
    article = m_ans.group(2).strip()
    qnum = None
    for ci in range(len(cells) - 1, 1, -1):
        c = cells[ci].strip()
        q = _extract_qnum(c)
        if q:
            qnum = q
            break
    if not qnum:
        qnum = last_qnum + 1
    taux_str = cells[2].strip() if len(cells) > 2 else ""
    return {
        "num": qnum,
        "correct_answer": answer,
        "article_reference": _clean_text(article),
        "success_rate": _parse_taux(taux_str),
    }


def _parse_merged_qr(cells: list[str]) -> dict[str, Any] | None:
    """Parse: | QNUM LETTER | article | taux | (empty) |"""
    if len(cells) < 2:
        return None

    cell0 = cells[0].strip()
    # Pattern: "1 a", "2 b", "10 c"
    m = re.match(r"^(\d+)\s+([A-Fa-f])$", cell0)
    if not m:
        return None

    qnum = int(m.group(1))
    answer = m.group(2).upper()
    article = cells[1].strip() if len(cells) > 1 else ""
    taux_str = cells[2].strip() if len(cells) > 2 else ""
    taux = _parse_taux(taux_str)

    return {
        "num": qnum,
        "correct_answer": answer,
        "article_reference": _clean_text(article),
        "success_rate": taux,
    }


def _extract_taux_from_article(
    article: str,
    taux_str: str,
) -> tuple[str, str]:
    """Extract taux value that may be embedded at end of article text."""
    m = re.search(r"\s+(\d+(?:[.,]\d+)?)\s*\(%\)\s*$", article)
    if m:
        if not _parse_taux(taux_str):
            taux_str = m.group(1)
        article = article[: m.start()].strip()
    if taux_str:
        m2 = re.search(r"(\d+(?:[.,]\d+)?)\s*\(%\)\s*$", taux_str)
        if m2:
            taux_str = m2.group(1)
    return article, taux_str


def _parse_mixed_shifted(
    cells: list[str],
    last_qnum: int,
) -> dict[str, Any] | None:
    """Parse tables where some rows are normal and some are shifted.

    Normal row: | QNUM | LETTER | article | taux |
    Shifted row: | LETTER | article taux | (empty) | QNUM |
    Merged first row: | QNUM LETTER | article taux | (empty) | |
    """
    if len(cells) < 3:
        return None

    cell0 = cells[0].strip()

    # Case 1: Normal row - cell0 is a number
    qnum = _extract_qnum(cell0)
    if qnum and not re.match(r"^\d+\s+[A-Fa-f]$", cell0):
        return _parse_mixed_normal_row(cells, qnum)

    # Case 2: Merged first cell "QNUM LETTER"
    m2 = re.match(r"^(\d+)\s+([A-Fa-f])$", cell0)
    if m2:
        return _parse_mixed_merged_row(cells, int(m2.group(1)), m2.group(2).upper())

    # Case 3: Shifted - cell0 is answer letter, qnum is in last non-empty cell
    return _parse_mixed_shifted_row(cells, cell0, last_qnum)


def _parse_mixed_normal_row(
    cells: list[str],
    qnum: int,
) -> dict[str, Any]:
    """Parse a normal row in mixed_shifted table: | QNUM | LETTER | article | taux |."""
    answer = _extract_answer_letter(cells[1].strip()) if len(cells) > 1 else None
    article = cells[2].strip() if len(cells) > 2 else ""
    taux_str = cells[3].strip() if len(cells) > 3 else ""
    article, taux_str = _extract_taux_from_article(article, taux_str)
    return {
        "num": qnum,
        "correct_answer": answer or "",
        "article_reference": _clean_text(article),
        "success_rate": _parse_taux(taux_str),
    }


def _parse_mixed_merged_row(
    cells: list[str],
    qnum: int,
    answer: str,
) -> dict[str, Any]:
    """Parse a merged QNUM+LETTER row in mixed_shifted table."""
    article = cells[1].strip() if len(cells) > 1 else ""
    taux_str = cells[2].strip() if len(cells) > 2 else ""
    article, taux_str = _extract_taux_from_article(article, taux_str)
    return {
        "num": qnum,
        "correct_answer": answer,
        "article_reference": _clean_text(article),
        "success_rate": _parse_taux(taux_str),
    }


def _parse_mixed_shifted_row(
    cells: list[str],
    cell0: str,
    last_qnum: int,
) -> dict[str, Any] | None:
    """Parse a shifted row: | LETTER | article taux | (empty) | QNUM |."""
    answer = _extract_answer_letter(cell0)
    if not answer:
        return None

    article = cells[1].strip() if len(cells) > 1 else ""
    taux_str = cells[2].strip() if len(cells) > 2 else ""

    qnum = None
    for ci in range(len(cells) - 1, 1, -1):
        q_candidate = _extract_qnum(cells[ci].strip())
        if q_candidate:
            qnum = q_candidate
            break

    article, taux_str = _extract_taux_from_article(article, taux_str)

    if not qnum:
        qnum = last_qnum + 1

    return {
        "num": qnum,
        "correct_answer": answer,
        "article_reference": _clean_text(article),
        "success_rate": _parse_taux(taux_str),
    }


def _parse_5col_cadence(
    cells: list[str],
    last_qnum: int,
) -> dict[str, Any] | None:
    """Parse 5-col: | QNUM RULE | CADENCE | REPONSE | ARTICLE | Taux |"""
    if len(cells) < 4:
        return None

    cell0 = cells[0].strip()
    # Extract question number from first cell ("1 Roque", "2 J'adoube", etc.)
    m = re.match(r"^(\d+)\s*", cell0)
    qnum = int(m.group(1)) if m else None

    # If no qnum in first cell, check for shifted rows
    # e.g. "Nulle | Lente | B | Art 9.1.2. | 8 81,88%"
    if not qnum:
        # Look for answer in cells[2]
        answer = _extract_answer_letter(cells[2].strip()) if len(cells) > 2 else None
        article = cells[3].strip() if len(cells) > 3 else ""
        taux_str = cells[4].strip() if len(cells) > 4 else ""

        # qnum might be embedded in taux: "8 81,88%"
        if taux_str:
            m2 = re.match(r"^(\d+)\s+(\d+.*)", taux_str)
            if m2:
                qnum = int(m2.group(1))
                taux_str = m2.group(2)

        if not qnum:
            qnum = last_qnum + 1

        return {
            "num": qnum,
            "correct_answer": answer or "",
            "article_reference": _clean_text(article),
            "success_rate": _parse_taux(taux_str),
        }

    # Normal 5-col: cell[1]=cadence, cell[2]=answer, cell[3]=article, cell[4]=taux
    # But sometimes cadence is empty and cells shift
    answer_str = cells[2].strip() if len(cells) > 2 else ""
    answer = _extract_answer_letter(answer_str)

    # If cell[2] is empty, answer might be in cell[1]
    if not answer and len(cells) > 1:
        answer = _extract_answer_letter(cells[1].strip())
        article = cells[2].strip() if len(cells) > 2 else ""
        taux_str = cells[3].strip() if len(cells) > 3 else ""
    else:
        article = cells[3].strip() if len(cells) > 3 else ""
        taux_str = cells[4].strip() if len(cells) > 4 else ""

    return {
        "num": qnum,
        "correct_answer": answer or "",
        "article_reference": _clean_text(article),
        "success_rate": _parse_taux(taux_str),
    }


def _parse_merged_rep_taux(cells: list[str]) -> dict[str, Any] | None:
    """Parse: | QNUM | article | LETTER taux | or | QNUM | article | LETTER taux | empty |"""
    if len(cells) < 2:
        return None

    qnum = _extract_qnum(cells[0].strip())
    if not qnum:
        return None

    # Find the cell with merged "LETTER taux" pattern
    article = ""
    answer = None
    taux_str = ""

    for ci in range(1, len(cells)):
        cell = cells[ci].strip()
        m = re.match(r"^([A-Fa-f])\s+(\d+(?:[.,]\d+)?)", cell)
        if m:
            answer = m.group(1).upper()
            taux_str = m.group(2)
            # Everything before this cell is article
            article = " ".join(cells[ci2].strip() for ci2 in range(1, ci)).strip()
            break
        # Also check for "A RI 98.8" pattern (letter + noise + taux)
        m2 = re.match(r"^([A-Fa-f])\s+\S+\s+(\d+(?:[.,]\d+)?)", cell)
        if m2:
            answer = m2.group(1).upper()
            taux_str = m2.group(2)
            article = " ".join(cells[ci2].strip() for ci2 in range(1, ci)).strip()
            break

    if not answer:
        # Fallback: try last cell for merged rep+taux
        last = cells[-1].strip()
        m3 = re.match(r"^([A-Fa-f])\s+(\d+(?:[.,]\d+)?)", last)
        if m3:
            answer = m3.group(1).upper()
            taux_str = m3.group(2)
            article = " ".join(
                cells[ci2].strip() for ci2 in range(1, len(cells) - 1)
            ).strip()

    if not answer:
        # Try second cell as article, third as merged
        article = cells[1].strip() if len(cells) > 1 else ""
        rep_taux = cells[2].strip() if len(cells) > 2 else ""
        m4 = re.match(r"^([A-Fa-f])\s+(\d+(?:[.,]\d+)?)", rep_taux)
        if m4:
            answer = m4.group(1).upper()
            taux_str = m4.group(2)

    return {
        "num": qnum,
        "correct_answer": answer or "",
        "article_reference": _clean_text(article),
        "success_rate": _parse_taux(taux_str),
    }


def _parse_nq_art_rep_taux(cells: list[str]) -> dict[str, Any] | None:
    """Parse: | N | QUESTION_TEXT | ARTICLE | REPONSE_TEXT | Taux |"""
    if len(cells) < 4:
        return None

    qnum = _extract_qnum(cells[0].strip())
    if not qnum:
        return None

    # For dec2019 UVC: N | question_desc | article_ref | answer_text | taux
    # The "answer" here is often full text, not a letter
    article = cells[2].strip()
    answer_text = cells[3].strip()
    taux_str = cells[4].strip() if len(cells) > 4 else ""

    # Try to extract letter from answer text
    answer = _extract_answer_letter(answer_text)

    return {
        "num": qnum,
        "correct_answer": answer or answer_text,
        "article_reference": _clean_text(article),
        "success_rate": _parse_taux(taux_str),
    }


def _parse_q_doc_art_rep_taux(cells: list[str]) -> dict[str, Any] | None:
    """Parse: | Q | Document | Article | Réponse | Taux |"""
    if len(cells) < 4:
        return None

    # Cell 0 might have "QNUM" or "QNUM letter" (merged)
    cell0 = cells[0].strip()
    qnum = _extract_qnum(cell0)

    # Check if qnum has letter merged: "4 LA Chap. 3.1..."
    answer = None
    if qnum and len(cells) >= 5:
        # Normal: | Q | Doc | Art | LETTER | Taux |
        answer = _extract_answer_letter(cells[3].strip())
        article = (cells[1].strip() + " " + cells[2].strip()).strip()
        taux_str = cells[4].strip() if len(cells) > 4 else ""
    elif qnum and len(cells) >= 4:
        answer = _extract_answer_letter(cells[3].strip())
        article = (cells[1].strip() + " " + cells[2].strip()).strip()
        taux_str = ""
    else:
        return None

    return {
        "num": qnum,
        "correct_answer": answer or "",
        "article_reference": _clean_text(article),
        "success_rate": _parse_taux(taux_str),
    }


def _parse_q_art_taux_no_rep(cells: list[str]) -> dict[str, Any] | None:
    """Parse: | QNUM | ARTICLE | Taux | (no REPONSE column)."""
    # This format has no answer - skip it, answers come from another table
    return None


def _extract_qnum(text: str) -> int | None:
    """Extract question number from text."""
    text = text.strip()
    m = re.match(r"^(\d+)", text)
    if m:
        return int(m.group(1))
    return None


def _assign_uvs_to_question_groups(
    groups: list[list[dict[str, Any]]],
    uv_corrections: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    """
    Match question groups to UVs using correction data.

    Returns list of (uv, questions, corrections) tuples.
    """
    uv_order = ["UVR", "UVC", "UVO", "UVT"]
    used_uvs: set[str] = set()
    result = []

    # Only process as many groups as we have corrections
    for group in groups[: len(uv_corrections)]:
        n_q = len(group)
        best_uv = None
        best_corr: list[dict[str, Any]] = []

        # Try to match by count (with tolerance)
        for uv in uv_order:
            if uv in used_uvs:
                continue
            corr = uv_corrections.get(uv, [])
            if corr and abs(len(corr) - n_q) <= 3:
                best_uv = uv
                best_corr = corr
                break

        # Fallback: assign by position
        if not best_uv:
            remaining = [uv for uv in uv_order if uv not in used_uvs]
            if remaining:
                best_uv = remaining[0]
                best_corr = uv_corrections.get(best_uv, [])

        if best_uv:
            used_uvs.add(best_uv)
            result.append((best_uv, group, best_corr))

    return result


def extract_session(
    session: str,
    json_path: Path,
) -> list[dict[str, Any]]:
    """
    Extract all MCQ reference data from a single session.

    Correction-centric approach: extract correction grids first (they have
    the authoritative answers), then optionally enrich with question text/choices.

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

    # Extract correction tables (primary source of truth for answers)
    uv_corrections = parse_correction_table_from_markdown(markdown, session)

    # Also extract questions for enrichment (text + choices)
    all_questions = extract_questions_from_markdown(markdown)
    groups = group_questions_by_sequence(all_questions)

    logger.info(
        f"  {len(all_questions)} questions in {len(groups)} groups, "
        f"{len(uv_corrections)} correction tables"
    )

    # Build question lookup: try to match questions to UVs
    # Use correction counts to guide assignment
    q_lookup = _build_question_lookup(groups, uv_corrections)

    # Build entries from correction grids
    entries: list[dict[str, Any]] = []
    for uv, corrections in uv_corrections.items():
        uv_name = UV_MAP.get(uv, uv.lower())

        for corr in corrections:
            q_num = corr["num"]
            correct_letter = corr.get("correct_answer", "")
            if correct_letter and len(correct_letter) == 1:
                correct_letter = correct_letter.upper()

            # Try to find matching question text/choices
            q_info = q_lookup.get((uv, q_num), {})

            entry = {
                "session": session,
                "uv": uv_name,
                "question_num": q_num,
                "question_text": q_info.get("text", ""),
                "choices": q_info.get("choices", {}),
                "correct_letter": correct_letter,
                "article_reference": corr.get("article_reference", ""),
                "success_rate": corr.get("success_rate"),
            }
            entries.append(entry)

    logger.info(f"  Extracted {len(entries)} reference entries")
    return entries


def _build_question_lookup(
    groups: list[list[dict[str, Any]]],
    uv_corrections: dict[str, list[dict[str, Any]]],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Build a lookup from (UV, question_num) to question info."""
    uv_order = ["UVR", "UVC", "UVO", "UVT"]
    used_uvs: set[str] = set()
    lookup: dict[tuple[str, int], dict[str, Any]] = {}

    for group in groups[: len(uv_corrections)]:
        n_q = len(group)
        best_uv = None

        for uv in uv_order:
            if uv in used_uvs:
                continue
            corr = uv_corrections.get(uv, [])
            if corr and abs(len(corr) - n_q) <= 3:
                best_uv = uv
                break

        if not best_uv:
            remaining = [uv for uv in uv_order if uv not in used_uvs]
            if remaining:
                best_uv = remaining[0]

        if best_uv:
            used_uvs.add(best_uv)
            for q in group:
                lookup[(best_uv, q["num"])] = q

    return lookup


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

    # Summary by session
    from collections import Counter

    session_counts = Counter(e["session"] for e in all_entries)
    uv_counts = Counter(e["uv"] for e in all_entries)
    with_answer = sum(1 for e in all_entries if e["correct_letter"])
    with_choices = sum(1 for e in all_entries if e["choices"])

    print("\n=== MCQ Reference Extraction Summary ===")
    print(f"Total entries: {len(all_entries)}")
    print(f"With correct_letter: {with_answer}")
    print(f"With choices: {with_choices}")
    print("\nBy session:")
    for session in sorted(session_counts):
        print(f"  {session}: {session_counts[session]}")
    print("\nBy UV:")
    for uv in sorted(uv_counts):
        print(f"  {uv}: {uv_counts[uv]}")


if __name__ == "__main__":
    main()
