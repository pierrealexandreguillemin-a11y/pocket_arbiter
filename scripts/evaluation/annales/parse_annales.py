"""
Parse annales JSON (Docling extraction) to structured Q/A format.

This module extracts questions, choices, and corrections from the
JSON files produced by Docling extraction of FFE exam annals.

ISO Reference:
    - ISO/IEC 42001 A.7.3 - Data traceability
    - ISO/IEC 25010 - Functional suitability

Usage:
    python -m scripts.evaluation.annales.parse_annales \
        --input corpus/processed/annales_dec_2024 \
        --output data/evaluation/annales

Example:
    >>> from scripts.evaluation.annales.parse_annales import parse_annales_file
    >>> result = parse_annales_file(Path("corpus/processed/annales_dec_2024/Annales-Decembre-2024.json"))
    >>> print(len(result["units"]))  # 4 UV units
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from scripts.evaluation.annales.session_utils import detect_session_from_filename
from scripts.pipeline.utils import get_timestamp, normalize_text, save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# UV identification patterns
UV_PATTERNS = {
    "UVR": r"UVR",
    "UVC": r"UVC",
    "UVO": r"UVO",
    "UVT": r"UVT",
}

# Taxonomy constants (industry standards)
QUESTION_TYPES = ["factual", "procedural", "scenario", "comparative"]
COGNITIVE_LEVELS = ["RECALL", "UNDERSTAND", "APPLY", "ANALYZE"]
REASONING_TYPES = ["single-hop", "multi-hop", "temporal"]
ANSWER_TYPES = ["extractive", "abstractive", "yes_no", "list", "multiple_choice"]

# Keywords for question type classification
SCENARIO_KEYWORDS = [
    "vous êtes",
    "vous etes",
    "un joueur",
    "une joueuse",
    "un arbitre",
    "lors d'une",
    "lors d'un",
    "pendant",
    "au cours",
    "en cours de",
    "votre club",
    "l'équipe",
    "le capitaine",
    "le directeur",
    "que faites-vous",
    "que décidez-vous",
    "que lui répondez-vous",
    "quelle décision",
    "comment réagissez-vous",
]

PROCEDURAL_KEYWORDS = [
    "comment",
    "quelle procédure",
    "quelles étapes",
    "de quelle manière",
    "que doit faire",
    "quelle démarche",
    "pour obtenir",
]

COMPARATIVE_KEYWORDS = [
    "quelle différence",
    "comparez",
    "par rapport à",
    "contrairement à",
    "à la différence de",
    "parmi les suivants",
]

# Full question block pattern (## Question N : text + choices until next question)
# Stop at: next Question N, UV headers, Corrigé/Fin/Grille sections, % stats
QUESTION_BLOCK_PATTERN = re.compile(
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

# Choice extraction patterns - multiple formats in annales
# Format 1: "- a)" or "-a)" (standard)
CHOICE_PATTERN_DASH_LETTER_PAREN = re.compile(
    r"(?:^|\n)\s*-\s*([a-fA-F])\)\s*(.+?)(?=(?:\n\s*-\s*[a-fA-F]\))|$)",
    re.DOTALL | re.IGNORECASE,
)
# Format 2: "A - " or "A-" (starts with letter)
CHOICE_PATTERN_LETTER_DASH = re.compile(
    r"(?:^|\n)(?:##\s*)?([A-F])\s*[-–—]\s*(.+?)(?=(?:\n(?:##\s*)?[A-F]\s*[-–—])|$)",
    re.DOTALL,
)
# Format 3: "- A - " (jun2021 style - dash letter dash)
CHOICE_PATTERN_DASH_LETTER_DASH = re.compile(
    r"(?:^|\n)\s*-\s*([A-F])\s*[-–—]\s*(.+?)(?=(?:\n\s*-\s*[A-F]\s*[-–—])|$)",
    re.DOTALL,
)
# Format 4: "- A :" or "A :" (dec2019 style - letter colon)
CHOICE_PATTERN_LETTER_COLON = re.compile(
    r"(?:^|\n)\s*(?:-\s*)?([A-F])\s*:\s*(.+?)(?=(?:\n\s*(?:-\s*)?[A-F]\s*:)|$)",
    re.DOTALL,
)
# Format 5: "a." or "A." (numbered list style)
CHOICE_PATTERN_LETTER_DOT = re.compile(
    r"(?:^|\n)\s*([a-fA-F])\.\s*(.+?)(?=(?:\n\s*[a-fA-F]\.)|$)",
    re.DOTALL,
)
# Format 6: Permissive pattern for inconsistent 2018 formats
# Matches: "- A - text", "- BText", "- CText", "AText", etc.
# Handles: dash optional, space optional after letter, separator optional
CHOICE_PATTERN_PERMISSIVE = re.compile(
    r"(?:^|\n)\s*-?\s*([A-F])[-–—\s]*([A-Za-zéèêëàâäùûüîïôöçÀÂÄÉÈÊËÎÏÔÖÙÛÜ][^-–—\n]*?)(?=(?:\n\s*-?\s*[A-F][-–—\s]*[A-Za-z])|(?:\n##)|$)",
    re.DOTALL,
)


def _classify_question_type(text_lower: str, uv: str, text_len: int) -> str:
    """
    Classify question type: factual, procedural, scenario, or comparative.

    Args:
        text_lower: Lowercase question text.
        uv: UV type (UVR, UVC, UVO, UVT).
        text_len: Length of original text.

    Returns:
        Question type string.
    """
    if any(kw in text_lower for kw in SCENARIO_KEYWORDS):
        return "scenario"
    if any(kw in text_lower for kw in PROCEDURAL_KEYWORDS):
        return "procedural"
    if any(kw in text_lower for kw in COMPARATIVE_KEYWORDS):
        return "comparative"

    # UVT questions are typically scenario-based (practical exam)
    if uv == "UVT" and text_len > 100:
        return "scenario"

    return "factual"


def _classify_cognitive_level(question_type: str) -> str:
    """
    Classify cognitive level based on Bloom's Taxonomy.

    Args:
        question_type: Type of question.

    Returns:
        Cognitive level: RECALL, UNDERSTAND, APPLY, or ANALYZE.
    """
    level_map = {
        "scenario": "APPLY",  # Application of rules to situation
        "procedural": "UNDERSTAND",  # Understanding process
        "comparative": "ANALYZE",  # Analysis and comparison
        "factual": "RECALL",  # Factual recall
    }
    return level_map.get(question_type, "RECALL")


def _classify_reasoning_type(
    text_lower: str,
    question_type: str,
    has_multiple_refs: bool,
) -> str:
    """
    Classify reasoning type: single-hop, multi-hop, or temporal.

    Args:
        text_lower: Lowercase question text.
        question_type: Type of question.
        has_multiple_refs: Whether question references multiple articles.

    Returns:
        Reasoning type string.
    """
    if has_multiple_refs or question_type == "scenario":
        return "multi-hop"  # Context → Rule → Answer
    if any(kw in text_lower for kw in ["quand", "depuis", "délai"]):
        return "temporal"
    return "single-hop"


def _classify_answer_type(
    text_lower: str,
    question_type: str,
    has_choices: bool,
) -> str:
    """
    Classify answer type: multiple_choice, yes_no, list, extractive, abstractive.

    Args:
        text_lower: Lowercase question text.
        question_type: Type of question.
        has_choices: Whether question has multiple choice options.

    Returns:
        Answer type string.
    """
    if has_choices:
        return "multiple_choice"
    if "vrai" in text_lower and "faux" in text_lower:
        return "yes_no"
    if any(kw in text_lower for kw in ["listez", "énumérez", "quels sont"]):
        return "list"
    if question_type == "scenario":
        return "abstractive"  # Scenario needs synthesis
    return "extractive"  # Direct fact from text


def classify_question_taxonomy(
    text: str,
    uv: str,
    has_multiple_refs: bool = False,
    has_choices: bool = True,
) -> dict[str, str]:
    """
    Classify question according to industry-standard taxonomy.

    Args:
        text: Question text.
        uv: UV type (UVR, UVC, UVO, UVT).
        has_multiple_refs: Whether question references multiple articles.
        has_choices: Whether question has multiple choice options.

    Returns:
        Dict with question_type, cognitive_level, reasoning_type, answer_type.

    Reference:
        - Bloom's Taxonomy for cognitive levels
        - RAGAS/BEIR standards for question types
    """
    text_lower = text.lower()

    question_type = _classify_question_type(text_lower, uv, len(text))
    cognitive_level = _classify_cognitive_level(question_type)
    reasoning_type = _classify_reasoning_type(
        text_lower, question_type, has_multiple_refs
    )
    answer_type = _classify_answer_type(text_lower, question_type, has_choices)

    return {
        "question_type": question_type,
        "cognitive_level": cognitive_level,
        "reasoning_type": reasoning_type,
        "answer_type": answer_type,
    }


def _clean_text(text: str) -> str:
    """Clean extracted text (normalize whitespace, fix encoding)."""
    # Fix common encoding issues
    text = text.replace("�", "é")
    text = text.replace("\u00e9", "é")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return normalize_text(text)


def _extract_choices_from_block(block: str) -> dict[str, str]:
    """Extract A/B/C/D/E/F choices from a question block using multiple patterns."""
    choices: dict[str, str] = {}

    # Try all choice patterns in order of specificity
    patterns = [
        CHOICE_PATTERN_DASH_LETTER_PAREN,  # "- a)" format
        CHOICE_PATTERN_DASH_LETTER_DASH,  # "- A - " format (jun2021)
        CHOICE_PATTERN_LETTER_COLON,  # "- A :" or "A :" format (dec2019)
        CHOICE_PATTERN_LETTER_DASH,  # "A - " format
        CHOICE_PATTERN_LETTER_DOT,  # "a." format
        CHOICE_PATTERN_PERMISSIVE,  # Permissive (2018 inconsistent formats)
    ]

    for pattern in patterns:
        for match in pattern.finditer(block):
            letter = match.group(1).upper()
            choice_text = _clean_text(match.group(2))
            # Only add if not already found (avoid duplicates)
            if letter not in choices and choice_text:
                choices[letter] = choice_text

    return choices


def _identify_uv_from_table(table: dict[str, Any]) -> str | None:
    """Identify which UV a correction table belongs to."""
    headers_str = " ".join(str(h) for h in table.get("headers", []))

    # Check for UV markers in headers
    for uv, pattern in UV_PATTERNS.items():
        if re.search(pattern, headers_str, re.IGNORECASE):
            return uv

    # Check first row for UV marker
    rows = table.get("rows", [])
    if rows:
        first_row_str = " ".join(str(c) for c in rows[0])
        for uv, pattern in UV_PATTERNS.items():
            if re.search(pattern, first_row_str, re.IGNORECASE):
                return uv

    return None


def _is_correction_table(table: dict[str, Any]) -> bool:
    """Check if a table contains correction data (has Question/Réponse/Taux)."""
    headers = table.get("headers", [])
    rows = table.get("rows", [])

    def _has_correction_keywords(text: str) -> bool:
        """Check if text contains correction table keywords."""
        text_lower = text.lower()
        # Must have "question" keyword
        if "question" not in text_lower:
            return False
        # Must have response keyword (various encodings)
        has_response = any(kw in text_lower for kw in ["ponse", "reponse", "réponse"])
        # Must have either taux or article keyword
        has_reference = any(kw in text_lower for kw in ["taux", "article"])
        return has_response and has_reference

    # Check headers
    headers_str = " ".join(str(h) for h in headers)
    if _has_correction_keywords(headers_str):
        return True

    # Some tables have UV info in header and actual headers in first row
    # Check first row as potential header
    if rows and len(rows) > 0:
        first_row_str = " ".join(str(c) for c in rows[0])
        if _has_correction_keywords(first_row_str):
            return True

    return False


def _parse_correction_table(table: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Parse a correction table to extract answers and article references.

    Expected columns: Question | Réponse | Articles de référence | Taux Réussite
    """
    corrections: list[dict[str, Any]] = []
    headers = table.get("headers", [])
    rows = table.get("rows", [])

    # Check if actual headers are in first row (UV info in table headers)
    if rows:
        first_row_str = " ".join(str(c).lower() for c in rows[0])
        has_response = any(kw in first_row_str for kw in ["ponse", "reponse"])
        if "question" in first_row_str and has_response:
            headers = rows[0]
            rows = rows[1:]  # Skip header row

    # Find column indices
    question_idx = None
    answer_idx = None
    article_idx = None
    rate_idx = None

    for i, h in enumerate(headers):
        h_lower = str(h).lower()
        if "question" in h_lower and question_idx is None:
            question_idx = i
        elif any(kw in h_lower for kw in ["ponse", "reponse"]) and answer_idx is None:
            answer_idx = i
        elif ("article" in h_lower or "document" in h_lower) and article_idx is None:
            article_idx = i
        elif "taux" in h_lower and rate_idx is None:
            rate_idx = i

    if question_idx is None or answer_idx is None:
        logger.warning(f"Cannot identify columns in table: {headers}")
        return corrections

    for row in rows:
        if len(row) <= max(
            filter(None, [question_idx, answer_idx, article_idx, rate_idx])
        ):
            continue

        # Extract question number (may be embedded in text like "1 Roque")
        q_str = str(row[question_idx]).strip()
        num_match = re.match(r"^(\d+)", q_str)
        if not num_match:
            continue
        q_num = int(num_match.group(1))

        correction = {
            "num": q_num,
            "correct_answer": str(row[answer_idx]).strip().upper(),
            "article_reference": _clean_text(str(row[article_idx]))
            if article_idx and article_idx < len(row)
            else "",
            "success_rate": None,
            "difficulty": None,
        }

        # Parse success rate (e.g., "84%" -> 0.84)
        if rate_idx and rate_idx < len(row):
            rate_str = str(row[rate_idx]).strip()
            rate_match = re.search(r"(\d+)", rate_str)
            if rate_match:
                rate = int(rate_match.group(1)) / 100
                correction["success_rate"] = rate
                correction["difficulty"] = round(1 - rate, 2)

        corrections.append(correction)

    return corrections


def _extract_all_questions_from_markdown(markdown: str) -> list[dict[str, Any]]:
    """
    Extract ALL questions from markdown regardless of UV section.

    Returns list of questions with text, choices, and question number.
    The questions will be associated with UV later via corrections.
    """
    questions = []

    # Patterns indicating commentary/annulled blocks (not real questions)
    commentary_patterns = re.compile(
        r"^\s*(?:\d+\s*%\s*des\s*candidats"
        r"|Il\s+s[''']agissait"
        r"|pas\s+[eé]t[eé]\s+comptabilis[eé]e)",
        re.IGNORECASE,
    )

    # Extract each question block
    for match in QUESTION_BLOCK_PATTERN.finditer(markdown):
        q_num = int(match.group(1))
        q_content = match.group(2)

        # Skip commentary/annulled blocks
        if commentary_patterns.search(q_content.strip()):
            logger.info(f"Skipping commentary/annulled block: Question {q_num}")
            continue

        # Extract question text (before first choice)
        q_text_match = re.search(r"^(.+?)(?=\n\s*-\s*[a-d]\))", q_content, re.DOTALL)
        q_text = (
            _clean_text(q_text_match.group(1))
            if q_text_match
            else _clean_text(q_content)
        )

        # Extract choices from content
        choices = _extract_choices_from_block(q_content)

        questions.append(
            {
                "num": q_num,
                "text": q_text,
                "choices": choices,
            }
        )

    logger.info(f"Extracted {len(questions)} questions from markdown")
    return questions


def _group_questions_by_sequence(
    questions: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """
    Group questions by contiguous sequences (Q1, Q2, ... Q30 = one UV).

    Each UV starts fresh with Question 1.
    """
    if not questions:
        return []

    groups: list[list[dict[str, Any]]] = []
    current_group: list[dict[str, Any]] = []

    for q in questions:
        q_num = q["num"]

        # New sequence starts when we see Q1 again
        if q_num == 1 and current_group:
            groups.append(current_group)
            current_group = []

        current_group.append(q)

    if current_group:
        groups.append(current_group)

    return groups


def _merge_questions_corrections(
    questions: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
    uv: str = "UVR",
) -> list[dict[str, Any]]:
    """Merge question text with correction data by question number."""
    # Index corrections by number
    corr_by_num = {c["num"]: c for c in corrections}

    merged = []
    for q in questions:
        q_num = q["num"]
        text = q.get("text", "")
        article_ref = ""

        choices = q.get("choices", {})
        has_choices = bool(choices)

        if q_num in corr_by_num:
            corr = corr_by_num[q_num]
            article_ref = corr.get("article_reference", "")

            # Check for multiple references (indicates multi-hop reasoning)
            has_multiple_refs = " et " in article_ref or "," in article_ref

            # Classify question using taxonomy
            taxonomy = classify_question_taxonomy(
                text, uv, has_multiple_refs, has_choices
            )

            merged.append(
                {
                    "num": q_num,
                    "text": text,
                    "choices": choices,
                    "correct_answer": corr["correct_answer"],
                    "article_reference": article_ref,
                    "success_rate": corr["success_rate"],
                    "difficulty": corr["difficulty"],
                    # Industry-standard taxonomy metadata
                    "question_type": taxonomy["question_type"],
                    "cognitive_level": taxonomy["cognitive_level"],
                    "reasoning_type": taxonomy["reasoning_type"],
                    "answer_type": taxonomy["answer_type"],
                }
            )
        else:
            logger.warning(f"No correction found for question {q_num}")
            # Still classify even without correction
            taxonomy = classify_question_taxonomy(text, uv, False, has_choices)
            merged.append(
                {
                    "num": q_num,
                    "text": text,
                    "choices": choices,
                    "correct_answer": None,
                    "article_reference": None,
                    "success_rate": None,
                    "difficulty": None,
                    "question_type": taxonomy["question_type"],
                    "cognitive_level": taxonomy["cognitive_level"],
                    "reasoning_type": taxonomy["reasoning_type"],
                    "answer_type": taxonomy["answer_type"],
                }
            )

    return merged


def _infer_uv_from_question_count(
    n_questions: int,
    existing_uvs: set[str],
) -> str | None:
    """
    Infer UV type from number of questions (heuristic).

    Args:
        n_questions: Number of questions in the correction table.
        existing_uvs: Set of UVs already assigned.

    Returns:
        Inferred UV type or None if cannot determine.
    """
    if n_questions >= 25 and n_questions <= 35:
        # Could be UVR, UVC, or UVT (typically ~30 questions each)
        for uv in ["UVR", "UVC", "UVT"]:
            if uv not in existing_uvs:
                return uv
    elif n_questions >= 15 and n_questions <= 25:
        # Could be UVO (~20 questions) or smaller UVR/UVC
        for uv in ["UVO", "UVR", "UVC"]:
            if uv not in existing_uvs:
                return uv
    elif n_questions >= 5 and n_questions <= 15:
        # Smaller correction set - assign to first available UV
        for uv in ["UVR", "UVC", "UVO", "UVT"]:
            if uv not in existing_uvs:
                return uv
    return None


def _extract_corrections_from_tables(
    tables: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Extract corrections from all tables and organize by UV.

    Args:
        tables: List of table dicts from Docling extraction.

    Returns:
        Dict mapping UV type to list of corrections.
    """
    uv_corrections: dict[str, list[dict[str, Any]]] = {}

    for table in tables:
        if not _is_correction_table(table):
            continue

        uv = _identify_uv_from_table(table)
        corrections = _parse_correction_table(table)

        if corrections:
            # Try to determine UV from context if not in table
            if uv is None:
                uv = _infer_uv_from_question_count(
                    len(corrections), set(uv_corrections.keys())
                )

            if uv:
                if uv in uv_corrections:
                    logger.warning(f"Multiple correction tables for {uv}, using first")
                else:
                    uv_corrections[uv] = corrections
                    logger.info(f"Found {len(corrections)} corrections for {uv}")

    return uv_corrections


def _match_question_group_to_uv(
    q_group: list[dict[str, Any]],
    uv_order: list[str],
    uv_corrections: dict[str, list[dict[str, Any]]],
    used_uvs: set[str],
) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Match a question group to the appropriate UV based on correction count.

    Args:
        q_group: List of questions to match.
        uv_order: Preferred order of UVs.
        uv_corrections: Dict of UV to corrections.
        used_uvs: Set of already-used UVs (modified in place).

    Returns:
        Tuple of (matched_uv, matched_corrections) or (None, []) if no match.
    """
    n_questions = len(q_group)

    # Try to match with corrections by count (with tolerance of ±2)
    for uv in uv_order:
        if uv in used_uvs:
            continue
        corr = uv_corrections.get(uv, [])
        # Match by similar count (tolerance for missing questions in extraction)
        if corr and abs(len(corr) - n_questions) <= 2:
            used_uvs.add(uv)
            return uv, corr

    # Fallback: assign by position to remaining UV
    remaining_uvs = [uv for uv in uv_order if uv not in used_uvs]
    if remaining_uvs:
        matched_uv = remaining_uvs[0]
        matched_corrections = uv_corrections.get(matched_uv, [])
        used_uvs.add(matched_uv)
        return matched_uv, matched_corrections

    return None, []


def _calculate_unit_statistics(questions: list[dict[str, Any]]) -> dict[str, int]:
    """
    Calculate statistics for a unit's questions.

    Args:
        questions: List of question dicts.

    Returns:
        Dict with total_questions, with_text, with_choices, with_corrections.
    """
    return {
        "total_questions": len(questions),
        "with_text": sum(
            1 for q in questions if q.get("text") and "[Question" not in q["text"]
        ),
        "with_choices": sum(1 for q in questions if q.get("choices")),
        "with_corrections": sum(1 for q in questions if q.get("correct_answer")),
    }


def parse_annales_file(json_path: Path) -> dict[str, Any]:
    """
    Parse a single annales JSON file (Docling extraction output).

    Args:
        json_path: Path to the JSON file from Docling extraction.

    Returns:
        Structured dict with session info and questions by UV.

    Raises:
        FileNotFoundError: If the JSON file doesn't exist.
        ValueError: If the JSON structure is invalid.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Annales JSON not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if "markdown" not in data or "tables" not in data:
        raise ValueError(f"Invalid Docling JSON structure in {json_path}")

    markdown = data["markdown"]
    tables = data["tables"]
    filename = data.get("filename", json_path.stem)
    session = detect_session_from_filename(filename)

    logger.info(f"Parsing {filename} (session: {session})")

    # Extract corrections organized by UV
    uv_corrections = _extract_corrections_from_tables(tables)

    # Extract and group questions from markdown
    all_questions = _extract_all_questions_from_markdown(markdown)
    question_groups = _group_questions_by_sequence(all_questions)
    logger.info(f"Found {len(question_groups)} question sequences")

    # Match question groups with correction groups
    uv_order = ["UVR", "UVC", "UVO", "UVT"]
    units = []
    total_questions = 0
    used_uvs: set[str] = set()

    # Only process sequences that have corrections
    question_groups_to_process = question_groups[: len(uv_corrections)]

    for q_group in question_groups_to_process:
        matched_uv, matched_corrections = _match_question_group_to_uv(
            q_group, uv_order, uv_corrections, used_uvs
        )

        if matched_uv is None:
            continue  # Skip extra sequences (corrigé duplicates)

        # Merge questions with corrections
        merged = _merge_questions_corrections(
            q_group, matched_corrections, uv=matched_uv
        )

        if merged:
            stats = _calculate_unit_statistics(merged)
            units.append(
                {
                    "uv": matched_uv,
                    "questions": merged,
                    "statistics": stats,
                }
            )
            total_questions += len(merged)
            logger.info(f"{matched_uv}: {len(merged)} questions ({stats})")

    return {
        "session": session,
        "source_file": filename,
        "extraction_date": get_timestamp(),
        "total_questions": total_questions,
        "units": units,
    }


def parse_annales_directory(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """
    Parse all annales JSON files in a directory.

    Args:
        input_dir: Directory containing Docling JSON files.
        output_dir: Directory for output files.

    Returns:
        Extraction report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files (exclude report files)
    json_files = [
        f
        for f in input_dir.glob("*.json")
        if "report" not in f.name.lower() and "Annales" in f.name
    ]

    if not json_files:
        logger.warning(f"No annales JSON files found in {input_dir}")
        return {"error": "No files found", "input_dir": str(input_dir)}

    logger.info(f"Found {len(json_files)} annales files to parse")

    all_results = []
    total_questions = 0
    errors = []

    for json_path in json_files:
        try:
            result = parse_annales_file(json_path)
            all_results.append(result)
            total_questions += result["total_questions"]

            # Save individual result
            output_file = output_dir / f"parsed_{json_path.stem}.json"
            save_json(result, output_file)
            logger.info(f"Saved: {output_file}")

        except Exception as e:
            error_msg = f"{json_path.name}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    # Generate combined report
    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files_processed": len(all_results),
        "total_questions": total_questions,
        "sessions": [r["session"] for r in all_results],
        "by_session": {
            r["session"]: {
                "total": r["total_questions"],
                "units": {
                    u["uv"]: u["statistics"]["total_questions"] for u in r["units"]
                },
            }
            for r in all_results
        },
        "errors": errors,
        "timestamp": get_timestamp(),
    }

    report_file = output_dir / "parsing_report.json"
    save_json(report, report_file)
    logger.info(f"Report saved: {report_file}")

    return report


def main() -> None:
    """CLI for annales parsing."""
    parser = argparse.ArgumentParser(
        description="Parse annales JSON (Docling) to structured Q/A format",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input directory with Docling JSON files",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/evaluation/annales/parsed"),
        help="Output directory for parsed results",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    report = parse_annales_directory(args.input, args.output)

    print("\n=== Parsing Report ===")
    print(f"Files processed: {report.get('files_processed', 0)}")
    print(f"Total questions: {report.get('total_questions', 0)}")
    if report.get("by_session"):
        print("\nBy session:")
        for session, data in report["by_session"].items():
            print(f"  {session}: {data['total']} questions")
            for uv, count in data.get("units", {}).items():
                print(f"    - {uv}: {count}")
    if report.get("errors"):
        print(f"\nErrors: {len(report['errors'])}")


if __name__ == "__main__":
    main()
