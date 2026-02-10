"""
Verify and populate gold standard metadata fields.

Validates every question against the required schema, checks enums,
ranges, type correctness, and cross-field coherence. Auto-populates
missing fields where derivable.

ISO Reference:
    - ISO/IEC 25010 - Schema completeness, data quality
    - ISO/IEC 42001 - Traceability (report, corrections log)
    - ISO/IEC 29119 - Testable verification criteria
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Schema definitions ---

VALID_CATEGORIES = {
    "competitions",
    "regles_jeu",
    "open",
    "tournoi",
    "interclubs",
    "regles_ffe",
    "classement",
    "jeunes",
    "administratif",
    "regional",
    "feminin",
    "handicap",
    "medical",
}
VALID_ANSWER_TYPES = {"multiple_choice", "extractive", "abstractive", "list", "yes_no"}
VALID_QUESTION_TYPES = {"factual", "scenario", "procedural", "comparative"}
VALID_REASONING_TYPES = {"multi-hop", "single-hop", "temporal"}
VALID_COGNITIVE_LEVELS = {"Remember", "Apply", "Understand", "Analyze"}
VALID_CHUNK_MATCH_METHODS = {
    "doc_page_semantic",
    "answer_in_chunk",
    "procrustes_qa_full",
    "qat_revalidation",
    "procrustes_realign",
    "page_keyword",
    "procrustes_qa_combined",
    "article_direct",
}
VALID_SESSIONS = {
    "dec2019",
    "jun2021",
    "dec2021",
    "jun2022",
    "dec2022",
    "jun2023",
    "dec2023",
    "jun2024",
    "dec2024",
    "jun2025",
}
VALID_UV_CODES = {"UVR", "UVC", "UVO", "UVT"}

UV_TO_CATEGORY: dict[str, str] = {
    "UVR": "rules",
    "UVC": "clubs",
    "UVO": "open",
    "UVT": "tournament",
}


# --- Auto-population helpers ---


def derive_category_from_id(question_id: str) -> str | None:
    """Derive category from question ID segment.

    ID format: ffe:(annales|human):CATEGORY:NUM:HASH
    """
    parts = question_id.split(":")
    if len(parts) >= 4:
        return parts[2]
    return None


def derive_answer_type(choices: dict[str, str] | None) -> str:
    """Derive answer_type from presence of choices."""
    if choices:
        return "mcq"
    return "open"


def derive_difficulty_from_rate(success_rate: float | None) -> float | None:
    """Use success rate directly as difficulty value.

    The GS stores difficulty as a float 0.0-1.0 (success rate).
    """
    if success_rate is None:
        return None
    return round(success_rate, 2)


def extract_keywords_from_question(
    question: str,
    article_ref: str | None = None,
) -> list[str]:
    """Extract basic keywords from question text and article reference.

    Extracts article references and key chess/arbitrage terms.
    """
    keywords: list[str] = []

    # Extract article references
    if article_ref:
        refs = re.findall(
            r"(?:Art\.?\s*[\d.]+|LA\s*[-–—,]\s*[\d.]+|Chapitre\s+\d+)",
            article_ref,
        )
        keywords.extend(refs)

    # Extract key terms from question
    chess_terms = [
        "cadence",
        "pendule",
        "roque",
        "mat",
        "pat",
        "nulle",
        "abandon",
        "arbitre",
        "joueur",
        "partie",
        "tournoi",
        "homologation",
        "classement",
        "elo",
        "licence",
        "club",
        "ligue",
        "comité",
        "compétition",
        "appariement",
        "grille",
        "résultat",
        "forfait",
        "promotion",
        "pion",
        "roi",
        "dame",
        "tour",
        "fou",
        "cavalier",
        "échecs",
        "fide",
        "ffe",
    ]
    q_lower = question.lower()
    for term in chess_terms:
        if term in q_lower:
            keywords.append(term)

    return keywords if keywords else ["arbitrage"]


def compute_triplet_ready(
    question: str,
    expected_answer: str,
    expected_chunk_id: str | None,
) -> bool:
    """Check if question is ready for triplet generation."""
    return bool(
        question
        and len(question) > 20
        and expected_answer
        and len(expected_answer) > 5
        and expected_chunk_id
    )


# --- Validation functions ---


def validate_question(
    q: dict[str, Any],
    index: int,
) -> tuple[list[str], list[str]]:
    """Validate a single question against the schema.

    Args:
        q: Question dict.
        index: Question index (for error messages).

    Returns:
        Tuple of (errors list, warnings list).
    """
    errors: list[str] = []
    warnings: list[str] = []
    q_id = q.get("id", f"<index {index}>")
    meta = q.get("metadata", {})

    # --- Root fields ---

    # id format
    if not q.get("id") or not isinstance(q["id"], str):
        errors.append(f"{q_id}: id missing or not string")

    # question text
    q_text = q.get("question", "")
    if not q_text or len(q_text) < 20:
        # Image-dependent questions may have only the image marker
        if "<!-- image -->" in q_text:
            warnings.append(f"{q_id}: image-dependent question ({len(q_text)} chars)")
        else:
            errors.append(f"{q_id}: question too short ({len(q_text)} chars)")
    if "##" in q_text:
        errors.append(f"{q_id}: question contains ## fusion artifact")

    # expected_answer
    answer = q.get("expected_answer", "")
    if not answer or not answer.strip():
        errors.append(f"{q_id}: expected_answer empty")

    # is_impossible
    if not isinstance(q.get("is_impossible"), bool):
        errors.append(f"{q_id}: is_impossible not bool")

    # expected_docs
    docs = q.get("expected_docs")
    if not docs or not isinstance(docs, list) or len(docs) == 0:
        warnings.append(f"{q_id}: expected_docs empty")

    # expected_pages
    pages = q.get("expected_pages")
    if not pages or not isinstance(pages, list) or len(pages) == 0:
        warnings.append(f"{q_id}: expected_pages empty")

    # category
    cat = q.get("category", "")
    if cat not in VALID_CATEGORIES:
        errors.append(f"{q_id}: category '{cat}' not in valid set")

    # keywords
    kw = q.get("keywords")
    if not kw or not isinstance(kw, list) or len(kw) == 0:
        warnings.append(f"{q_id}: keywords empty")

    # --- Metadata fields ---

    # answer_type
    at = meta.get("answer_type", "")
    if at not in VALID_ANSWER_TYPES:
        errors.append(f"{q_id}: answer_type '{at}' invalid")

    # question_type
    qt = meta.get("question_type", "")
    if qt not in VALID_QUESTION_TYPES:
        errors.append(f"{q_id}: question_type '{qt}' invalid")

    # reasoning_type
    rt = meta.get("reasoning_type", "")
    if rt not in VALID_REASONING_TYPES:
        errors.append(f"{q_id}: reasoning_type '{rt}' invalid")

    # cognitive_level
    cl = meta.get("cognitive_level", "")
    if cl not in VALID_COGNITIVE_LEVELS:
        errors.append(f"{q_id}: cognitive_level '{cl}' invalid")

    # reasoning_class
    rc = meta.get("reasoning_class")
    if not rc or not isinstance(rc, str):
        warnings.append(f"{q_id}: reasoning_class missing or not string")

    # difficulty (float 0.0-1.0, represents success rate)
    diff = meta.get("difficulty")
    if diff is not None:
        if not isinstance(diff, int | float) or diff < 0 or diff > 1:
            errors.append(f"{q_id}: difficulty {diff} out of range 0-1")
    else:
        warnings.append(f"{q_id}: difficulty is None")

    # quality_score
    qs = meta.get("quality_score")
    if qs is not None:
        if not isinstance(qs, int | float) or qs < 0 or qs > 100:
            errors.append(f"{q_id}: quality_score {qs} out of range 0-100")
    else:
        warnings.append(f"{q_id}: quality_score is None")

    # chunk_match_score (some legacy scores go up to 110)
    cms = meta.get("chunk_match_score")
    if cms is not None:
        if not isinstance(cms, int | float) or cms < 0 or cms > 110:
            errors.append(f"{q_id}: chunk_match_score {cms} out of range 0-110")

    # chunk_match_method
    cmm = meta.get("chunk_match_method", "")
    if cmm not in VALID_CHUNK_MATCH_METHODS:
        errors.append(f"{q_id}: chunk_match_method '{cmm}' invalid")

    # triplet_ready
    if not isinstance(meta.get("triplet_ready"), bool):
        errors.append(f"{q_id}: triplet_ready not bool")

    # --- MCQ coherence ---
    choices = meta.get("choices")
    mcq_answer = meta.get("mcq_answer")
    if choices and isinstance(choices, dict) and len(choices) > 0:
        if not mcq_answer:
            warnings.append(f"{q_id}: has choices but no mcq_answer")
        elif mcq_answer not in choices:
            warnings.append(
                f"{q_id}: mcq_answer '{mcq_answer}' not in choices keys {list(choices.keys())}"
            )

    if mcq_answer and (not choices or not isinstance(choices, dict)):
        warnings.append(f"{q_id}: has mcq_answer but no choices dict")

    # --- Chunk coherence ---
    chunk_id = q.get("expected_chunk_id")
    if chunk_id:
        if not docs or len(docs) == 0:
            warnings.append(f"{q_id}: has chunk_id but empty expected_docs")
        if not pages or len(pages) == 0:
            warnings.append(f"{q_id}: has chunk_id but empty expected_pages")

    # --- Annales coherence ---
    src = meta.get("annales_source")
    if src and isinstance(src, dict):
        session = src.get("session")
        uv = src.get("uv")
        qnum = src.get("question_num")

        if session and session not in VALID_SESSIONS:
            errors.append(f"{q_id}: annales session '{session}' invalid")
        if uv and uv not in VALID_UV_CODES:
            errors.append(f"{q_id}: annales uv '{uv}' invalid")
        if not session:
            errors.append(f"{q_id}: annales_source missing session")
        if not uv:
            errors.append(f"{q_id}: annales_source missing uv")
        if not qnum or not isinstance(qnum, int) or qnum < 1:
            errors.append(f"{q_id}: annales_source question_num invalid")

    return errors, warnings


# --- Auto-population ---


def auto_populate(
    gs: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Auto-populate missing metadata fields where derivable.

    Args:
        gs: Gold standard dict (modified in place).

    Returns:
        Tuple of (modified gs, corrections log).
    """
    corrections: list[dict[str, Any]] = []

    for q in gs["questions"]:
        q_id = q.get("id", "unknown")
        meta = q.setdefault("metadata", {})

        # 1. category from id
        if not q.get("category"):
            derived = derive_category_from_id(q_id)
            if derived and derived in VALID_CATEGORIES:
                q["category"] = derived
                corrections.append(
                    {
                        "id": q_id,
                        "field": "category",
                        "action": "derived_from_id",
                        "value": derived,
                    }
                )

        # 2. answer_type from choices
        if not meta.get("answer_type"):
            at = derive_answer_type(meta.get("choices"))
            meta["answer_type"] = at
            corrections.append(
                {
                    "id": q_id,
                    "field": "answer_type",
                    "action": "derived_from_choices",
                    "value": at,
                }
            )

        # 3. difficulty from success_rate
        if meta.get("difficulty") is None:
            src = meta.get("annales_source") or {}
            rate = src.get("success_rate")
            diff = derive_difficulty_from_rate(rate)
            if diff is not None:
                meta["difficulty"] = diff
                corrections.append(
                    {
                        "id": q_id,
                        "field": "difficulty",
                        "action": "derived_from_success_rate",
                        "value": diff,
                        "success_rate": rate,
                    }
                )

        # 4. keywords from question + article_reference
        kw = q.get("keywords")
        if not kw or not isinstance(kw, list) or len(kw) == 0:
            new_kw = extract_keywords_from_question(
                q.get("question", ""),
                meta.get("article_reference"),
            )
            q["keywords"] = new_kw
            corrections.append(
                {
                    "id": q_id,
                    "field": "keywords",
                    "action": "extracted_from_question",
                    "count": len(new_kw),
                }
            )

        # 5. is_impossible correction for annales with known answer
        if q.get("is_impossible") is True and meta.get("annales_source"):
            if q.get("expected_answer"):
                q["is_impossible"] = False
                corrections.append(
                    {
                        "id": q_id,
                        "field": "is_impossible",
                        "action": "corrected_to_false",
                    }
                )

        # 6. triplet_ready recalculation
        old_tr = meta.get("triplet_ready")
        new_tr = compute_triplet_ready(
            q.get("question", ""),
            q.get("expected_answer", ""),
            q.get("expected_chunk_id"),
        )
        if old_tr != new_tr:
            meta["triplet_ready"] = new_tr
            corrections.append(
                {
                    "id": q_id,
                    "field": "triplet_ready",
                    "action": "recalculated",
                    "old": old_tr,
                    "new": new_tr,
                }
            )

    return gs, corrections


# --- Main verification ---


def verify_gs_metadata(
    gs_path: Path,
    output_path: Path | None = None,
    report_path: Path | None = None,
    auto_fix: bool = True,
) -> dict[str, Any]:
    """Verify and optionally auto-populate GS metadata.

    Args:
        gs_path: Path to gold standard JSON.
        output_path: If set, write fixed GS here.
        report_path: If set, write verification report here.
        auto_fix: Whether to auto-populate missing fields.

    Returns:
        Verification report dict.
    """
    with open(gs_path, encoding="utf-8") as f:
        gs = json.load(f)

    corrections: list[dict[str, Any]] = []

    # Auto-populate first (before validation)
    if auto_fix:
        gs, corrections = auto_populate(gs)

    # Validate all questions
    all_errors: list[str] = []
    all_warnings: list[str] = []

    for i, q in enumerate(gs["questions"]):
        errs, warns = validate_question(q, i)
        all_errors.extend(errs)
        all_warnings.extend(warns)

    # Build report
    report: dict[str, Any] = {
        "total_questions": len(gs["questions"]),
        "total_errors": len(all_errors),
        "total_warnings": len(all_warnings),
        "total_corrections": len(corrections),
        "errors": all_errors,
        "warnings": all_warnings,
        "corrections": corrections,
        "gate5_validation": {
            "schema_100_pct": len(all_errors) == 0,
            "enums_valid": not any("invalid" in e for e in all_errors),
            "ranges_valid": not any("out of range" in e for e in all_errors),
            "mcq_coherence": not any("mcq_answer" in e for e in all_errors),
            "annales_coherence": not any("annales" in e for e in all_errors),
            "overall": "PASS" if len(all_errors) == 0 else "FAIL",
        },
    }

    # Save fixed GS
    if output_path and auto_fix:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(gs, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved fixed GS to {output_path}")

    # Save report
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved report to {report_path}")

    return report


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify and populate GS metadata")
    parser.add_argument(
        "--gs",
        type=Path,
        default=Path("tests/data/checkpoints/gold_standard_patched.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/data/checkpoints/gold_standard_patched.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("tests/data/checkpoints/metadata_verification_report.json"),
    )
    parser.add_argument(
        "--no-fix",
        action="store_true",
        help="Validate only, do not auto-populate.",
    )
    args = parser.parse_args()

    report = verify_gs_metadata(
        gs_path=args.gs,
        output_path=args.output,
        report_path=args.report,
        auto_fix=not args.no_fix,
    )

    print("\n=== Metadata Verification Report ===")
    print(f"Total questions: {report['total_questions']}")
    print(f"Errors: {report['total_errors']}")
    print(f"Warnings: {report['total_warnings']}")
    print(f"Auto-corrections: {report['total_corrections']}")

    g5 = report["gate5_validation"]
    print("\n=== GATE 5 ===")
    print(f"Schema 100%: {'PASS' if g5['schema_100_pct'] else 'FAIL'}")
    print(f"Enums valid: {'PASS' if g5['enums_valid'] else 'FAIL'}")
    print(f"Ranges valid: {'PASS' if g5['ranges_valid'] else 'FAIL'}")
    print(f"MCQ coherence: {'PASS' if g5['mcq_coherence'] else 'FAIL'}")
    print(f"Annales coherence: {'PASS' if g5['annales_coherence'] else 'FAIL'}")
    print(f"Overall: {g5['overall']}")

    if report["total_errors"] > 0:
        print("\nFirst 20 errors:")
        for e in report["errors"][:20]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
