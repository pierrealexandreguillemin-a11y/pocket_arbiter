"""
Shared session detection utilities for annales processing.

This module consolidates duplicated session detection logic from
parse_annales.py and extract_corrige_answers.py.

ISO Reference:
    - ISO/IEC 12207 - DRY principle (Don't Repeat Yourself)
    - ISO/IEC 25010 - Maintainability
"""

import re


# Month patterns for session detection (comprehensive)
MONTH_PATTERNS: dict[str, str] = {
    # December (most common for FFE exams)
    "decembre": "dec",
    "décembre": "dec",
    "december": "dec",
    "dec": "dec",
    # June (second most common)
    "juin": "jun",
    "june": "jun",
    "jun": "jun",
    # Other months (for future sessions)
    "janvier": "jan",
    "january": "jan",
    "jan": "jan",
    "fevrier": "feb",
    "février": "feb",
    "february": "feb",
    "feb": "feb",
    "mars": "mar",
    "march": "mar",
    "mar": "mar",
    "avril": "apr",
    "april": "apr",
    "apr": "apr",
    "mai": "may",
    "may": "may",
    "juillet": "jul",
    "july": "jul",
    "jul": "jul",
    "aout": "aug",
    "août": "aug",
    "august": "aug",
    "aug": "aug",
    "septembre": "sep",
    "september": "sep",
    "sep": "sep",
    "octobre": "oct",
    "october": "oct",
    "oct": "oct",
    "novembre": "nov",
    "november": "nov",
    "nov": "nov",
}

# Numeric month lookup
MONTH_NUMBERS: dict[int, str] = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dec",
}


def detect_session_from_filename(filename: str) -> str:
    """
    Detect session identifier from annales filename.

    Unified implementation replacing:
    - parse_annales.py:_detect_session_from_filename()
    - extract_corrige_answers.py:_session_from_filename()

    Args:
        filename: Annales filename (e.g., "Annales-Decembre-2024.json")

    Returns:
        Session identifier (e.g., "dec2024", "jun2025")

    Examples:
        >>> detect_session_from_filename("Annales-Decembre-2024.json")
        'dec2024'
        >>> detect_session_from_filename("Annales-Juin-2025.pdf")
        'jun2025'
        >>> detect_session_from_filename("201812_annales.pdf")
        'dec2018'
    """
    filename_lower = filename.lower()

    # Handle special format: YYYYMM (e.g., 201812)
    yearmonth_match = re.search(r"(20\d{2})(0[1-9]|1[0-2])", filename)
    if yearmonth_match:
        year = yearmonth_match.group(1)
        month_num = int(yearmonth_match.group(2))
        if month_num in MONTH_NUMBERS:
            return f"{MONTH_NUMBERS[month_num]}{year}"

    # Find month from text patterns (check longer patterns first)
    month = None
    for pattern in sorted(MONTH_PATTERNS.keys(), key=len, reverse=True):
        if pattern in filename_lower:
            month = MONTH_PATTERNS[pattern]
            break

    # Find year (4 digits starting with 20)
    year_match = re.search(r"20(\d{2})", filename)
    year = year_match.group(0) if year_match else "unknown"

    if month:
        return f"{month}{year}"
    return f"session_{year}"


def normalize_session_id(session: str) -> str:
    """
    Normalize session identifier to standard format.

    Args:
        session: Raw session string

    Returns:
        Normalized session (lowercase, consistent format)

    Examples:
        >>> normalize_session_id("Dec2024")
        'dec2024'
        >>> normalize_session_id("JUIN 2025")
        'jun2025'
    """
    session_lower = session.lower().replace(" ", "").replace("-", "")

    # Normalize month names
    for pattern, short in MONTH_PATTERNS.items():
        if pattern in session_lower:
            session_lower = session_lower.replace(pattern, short)
            break

    return session_lower
