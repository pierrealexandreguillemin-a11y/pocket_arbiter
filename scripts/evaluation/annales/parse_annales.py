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
    "vous êtes", "vous etes", "un joueur", "une joueuse", "un arbitre",
    "lors d'une", "lors d'un", "pendant", "au cours", "en cours de",
    "votre club", "l'équipe", "le capitaine", "le directeur",
    "que faites-vous", "que décidez-vous", "que lui répondez-vous",
    "quelle décision", "comment réagissez-vous",
]

PROCEDURAL_KEYWORDS = [
    "comment", "quelle procédure", "quelles étapes", "de quelle manière",
    "que doit faire", "quelle démarche", "pour obtenir",
]

COMPARATIVE_KEYWORDS = [
    "quelle différence", "comparez", "par rapport à", "contrairement à",
    "à la différence de", "parmi les suivants",
]

# Full question block pattern (## Question N : text + choices until next question)
QUESTION_BLOCK_PATTERN = re.compile(
    r"(?:^|\n)(?:##\s*)?Question\s+(\d+)\s*:\s*(.+?)(?=(?:\n(?:##\s*)?Question\s+\d+\s*:)|(?:\n##\s+[A-Z])|$)",
    re.DOTALL | re.IGNORECASE,
)

# Choice extraction pattern (- a) or -a) format)
CHOICE_PATTERN = re.compile(
    r"(?:^|\n)\s*-\s*([a-d])\)\s*(.+?)(?=(?:\n\s*-\s*[a-d]\))|$)",
    re.DOTALL | re.IGNORECASE,
)


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

    # Determine question_type
    question_type = "factual"  # default

    if any(kw in text_lower for kw in SCENARIO_KEYWORDS):
        question_type = "scenario"
    elif any(kw in text_lower for kw in PROCEDURAL_KEYWORDS):
        question_type = "procedural"
    elif any(kw in text_lower for kw in COMPARATIVE_KEYWORDS):
        question_type = "comparative"

    # UVT questions are typically scenario-based (practical exam)
    if uv == "UVT" and question_type == "factual":
        # Check if it's really factual or just not detected as scenario
        if len(text) > 100:  # Longer questions tend to be scenarios
            question_type = "scenario"

    # Determine cognitive_level (Bloom's Taxonomy)
    if question_type == "scenario":
        cognitive_level = "APPLY"  # Application of rules to situation
    elif question_type == "procedural":
        cognitive_level = "UNDERSTAND"  # Understanding process
    elif question_type == "comparative":
        cognitive_level = "ANALYZE"  # Analysis and comparison
    else:
        cognitive_level = "RECALL"  # Factual recall

    # Determine reasoning_type
    if has_multiple_refs or question_type == "scenario":
        reasoning_type = "multi-hop"  # Context → Rule → Answer
    elif "quand" in text_lower or "depuis" in text_lower or "délai" in text_lower:
        reasoning_type = "temporal"
    else:
        reasoning_type = "single-hop"

    # Determine answer_type
    if has_choices:
        answer_type = "multiple_choice"  # QCM from annales
    elif "vrai" in text_lower and "faux" in text_lower:
        answer_type = "yes_no"
    elif "listez" in text_lower or "énumérez" in text_lower or "quels sont" in text_lower:
        answer_type = "list"
    elif question_type == "scenario":
        answer_type = "abstractive"  # Scenario needs synthesis
    else:
        answer_type = "extractive"  # Direct fact from text

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
    """Extract A/B/C/D choices from a question block."""
    choices: dict[str, str] = {}

    # Find all choice patterns
    for match in CHOICE_PATTERN.finditer(block):
        letter = match.group(1).upper()
        choice_text = _clean_text(match.group(2))
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
    corrections = []
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
        if len(row) <= max(filter(None, [question_idx, answer_idx, article_idx, rate_idx])):
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
            "article_reference": _clean_text(str(row[article_idx])) if article_idx and article_idx < len(row) else "",
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

    # Extract each question block
    for match in QUESTION_BLOCK_PATTERN.finditer(markdown):
        q_num = int(match.group(1))
        q_content = match.group(2)

        # Extract question text (before first choice)
        q_text_match = re.search(r"^(.+?)(?=\n\s*-\s*[a-d]\))", q_content, re.DOTALL)
        q_text = _clean_text(q_text_match.group(1)) if q_text_match else _clean_text(q_content[:200])

        # Extract choices from content
        choices = _extract_choices_from_block(q_content)

        questions.append({
            "num": q_num,
            "text": q_text,
            "choices": choices,
        })

    logger.info(f"Extracted {len(questions)} questions from markdown")
    return questions


def _group_questions_by_sequence(questions: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
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
            taxonomy = classify_question_taxonomy(text, uv, has_multiple_refs, has_choices)

            merged.append({
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
            })
        else:
            logger.warning(f"No correction found for question {q_num}")
            # Still classify even without correction
            taxonomy = classify_question_taxonomy(text, uv, False, has_choices)
            merged.append({
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
            })

    return merged


def _detect_session_from_filename(filename: str) -> str:
    """Detect session (e.g., 'dec2024', 'juin2025') from filename."""
    filename_lower = filename.lower()

    # Month patterns
    month_map = {
        "decembre": "dec",
        "décembre": "dec",
        "dec": "dec",
        "juin": "jun",
        "june": "jun",
        "jun": "jun",
    }

    # Find month
    month = None
    for pattern, short in month_map.items():
        if pattern in filename_lower:
            month = short
            break

    # Find year
    year_match = re.search(r"20(\d{2})", filename)
    year = year_match.group(0) if year_match else "unknown"

    if month:
        return f"{month}{year}"
    return f"session_{year}"


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
    session = _detect_session_from_filename(filename)

    logger.info(f"Parsing {filename} (session: {session})")

    # Find all correction tables and identify their UV
    uv_corrections: dict[str, list[dict[str, Any]]] = {}

    for table in tables:
        if not _is_correction_table(table):
            continue

        uv = _identify_uv_from_table(table)
        corrections = _parse_correction_table(table)

        if corrections:
            # Try to determine UV from context if not in table
            if uv is None:
                # Use number of questions as heuristic (expanded ranges)
                n_questions = len(corrections)
                if n_questions >= 25 and n_questions <= 35:
                    # Could be UVR, UVC, or UVT (typically ~30 questions each)
                    if "UVR" not in uv_corrections:
                        uv = "UVR"
                    elif "UVC" not in uv_corrections:
                        uv = "UVC"
                    elif "UVT" not in uv_corrections:
                        uv = "UVT"
                elif n_questions >= 15 and n_questions <= 25:
                    # Could be UVO (~20 questions) or smaller UVR/UVC
                    if "UVO" not in uv_corrections:
                        uv = "UVO"
                    elif "UVR" not in uv_corrections:
                        uv = "UVR"
                    elif "UVC" not in uv_corrections:
                        uv = "UVC"
                elif n_questions >= 5 and n_questions <= 15:
                    # Smaller correction set - assign to first available UV
                    for candidate in ["UVR", "UVC", "UVO", "UVT"]:
                        if candidate not in uv_corrections:
                            uv = candidate
                            break

            if uv:
                if uv in uv_corrections:
                    logger.warning(f"Multiple correction tables for {uv}, using first")
                else:
                    uv_corrections[uv] = corrections
                    logger.info(f"Found {len(corrections)} corrections for {uv}")

    # Extract ALL questions from markdown
    all_questions = _extract_all_questions_from_markdown(markdown)

    # Group questions by sequence (each Q1 starts a new UV)
    question_groups = _group_questions_by_sequence(all_questions)
    logger.info(f"Found {len(question_groups)} question sequences")

    # Match question groups with correction groups
    # Order is typically: UVR (30), UVC (30), UVO (20), UVT (varies)
    # The markdown contains both "Sujet" (exam) and "Corrigé" (answers) sections
    # We only want the first occurrence of each UV (the Sujet section)
    uv_order = ["UVR", "UVC", "UVO", "UVT"]

    units = []
    total_questions = 0
    used_uvs: set[str] = set()

    # Only process first 4 sequences (the exam questions, not the corrigés)
    max_sequences = len(uv_corrections)  # Number of UVs with corrections
    question_groups_to_process = question_groups[:max_sequences]

    for i, q_group in enumerate(question_groups_to_process):
        n_questions = len(q_group)

        # Try to match with corrections by count (with tolerance of ±2)
        matched_uv = None
        matched_corrections: list[dict[str, Any]] = []

        for uv in uv_order:
            if uv in used_uvs:
                continue
            corr = uv_corrections.get(uv, [])
            # Match by similar count (tolerance for missing questions in extraction)
            if corr and abs(len(corr) - n_questions) <= 2:
                matched_uv = uv
                matched_corrections = corr
                used_uvs.add(uv)
                break

        # Fallback: assign by position to remaining UV
        if matched_uv is None:
            remaining_uvs = [uv for uv in uv_order if uv not in used_uvs]
            if remaining_uvs:
                matched_uv = remaining_uvs[0]
                matched_corrections = uv_corrections.get(matched_uv, [])
                used_uvs.add(matched_uv)
            else:
                continue  # Skip extra sequences (corrigé duplicates)

        # Merge questions with corrections (pass UV for taxonomy classification)
        merged = _merge_questions_corrections(q_group, matched_corrections, uv=matched_uv)

        if merged:
            units.append({
                "uv": matched_uv,
                "questions": merged,
                "statistics": {
                    "total_questions": len(merged),
                    "with_text": sum(1 for q in merged if q.get("text") and "[Question" not in q["text"]),
                    "with_choices": sum(1 for q in merged if q.get("choices")),
                    "with_corrections": sum(1 for q in merged if q.get("correct_answer")),
                },
            })
            total_questions += len(merged)
            logger.info(f"{matched_uv}: {len(merged)} questions ({units[-1]['statistics']})")

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
        f for f in input_dir.glob("*.json")
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
                "units": {u["uv"]: u["statistics"]["total_questions"] for u in r["units"]},
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
