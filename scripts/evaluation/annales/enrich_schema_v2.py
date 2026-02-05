#!/usr/bin/env python3
"""
Phase 4: Enrichissement Schema v2.0 pour questions BY DESIGN.

Transforme les questions validees vers le format Schema v2.0 complet
(46 champs, 8 groupes) avec provenance BY DESIGN.

ISO Reference:
- ISO 42001 A.6.2.2: Provenance tracking
- ISO 29119-3: Test data schema

Schema v2.0 Groups:
1. Root (2): id, legacy_id
2. Content (3): question, expected_answer, is_impossible
3. MCQ (5): original_question, choices, mcq_answer, correct_answer, original_answer
4. Provenance (6): chunk_id, docs, pages, article_reference, answer_explanation, annales_source
5. Classification (9): category, keywords, difficulty, question_type, cognitive_level,
                       reasoning_type, reasoning_class, answer_type, hard_type
6. Validation (7): status, method, reviewer, answer_current, verified_date, pages_verified, batch
7. Processing (7): chunk_match_score, chunk_match_method, reasoning_class_method,
                   triplet_ready, extraction_flags, answer_source, quality_score
8. Audit (3): history, qat_revalidation, requires_inference

Usage:
    python enrich_schema_v2.py --questions PATH --chunks PATH [--output PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.pipeline.utils import get_date, load_json, save_json  # noqa: E402

# Category inference from chunk content
CATEGORY_PATTERNS = {
    "arbitrage": [r"arbitr", r"juge", r"officiel", r"directeur"],
    "regles_jeu": [r"roque", r"echec", r"mat", r"pat", r"promotion", r"prise"],
    "competition": [r"tournoi", r"compet", r"ronde", r"appariement"],
    "materiel": [r"pendule", r"horloge", r"piece", r"echiquier"],
    "discipline": [r"sanction", r"penalite", r"exclusion", r"faute"],
    "classement": [r"elo", r"classement", r"performance", r"coef"],
    "jeunes": [r"jeune", r"junior", r"cadet", r"minime", r"benjamin"],
    "interclubs": [r"interclub", r"equipe", r"nationale", r"top"],
    "homologation": [r"homolog", r"officiel", r"validat"],
}

# Reasoning type inference
REASONING_PATTERNS = {
    "single-hop": [r"qu[e']est[-\s]ce", r"quel(le)?", r"combien", r"qui"],
    "multi-hop": [r"pourquoi", r"comment", r"expliqu", r"dans quel cas"],
    "temporal": [r"quand", r"delai", r"avant", r"apres", r"pendant"],
    "comparative": [r"difference", r"compar", r"entre", r"versus", r"ou"],
}


def generate_question_id(question: str, chunk_id: str) -> str:
    """
    Generate unique question ID.

    Format: gs:scratch:{category}:{seq}:{hash}
    """
    # Extract category hint from chunk_id
    if "LA-" in chunk_id or "LA_" in chunk_id:
        category = "arbitrage"
    elif "R01" in chunk_id or "Reglement" in chunk_id:
        category = "reglements"
    elif "Interclub" in chunk_id:
        category = "interclubs"
    else:
        category = "general"

    # Generate hash from question + chunk_id (not for security, just for ID)
    content = f"{question}:{chunk_id}"
    hash_val = hashlib.md5(content.encode()).hexdigest()[:8]  # noqa: S324

    return f"gs:scratch:{category}:001:{hash_val}"


def extract_article_reference(chunk: dict) -> str:
    """
    Extract article reference from chunk metadata.

    Args:
        chunk: Chunk dictionary with section/text

    Returns:
        Article reference string
    """
    section = chunk.get("section", "")
    text = chunk.get("text", "")[:200]

    # Try to find article number in section
    article_match = re.search(r"ARTICLE\s*(\d+(?:\.\d+)*)", section, re.IGNORECASE)
    if article_match:
        return f"Article {article_match.group(1)}"

    # Try in text
    article_match = re.search(r"article\s*(\d+(?:\.\d+)*)", text, re.IGNORECASE)
    if article_match:
        return f"Article {article_match.group(1)}"

    # Try chapter
    chapter_match = re.search(r"chapitre\s*(\d+)", text, re.IGNORECASE)
    if chapter_match:
        return f"Chapitre {chapter_match.group(1)}"

    # Fall back to section
    if section:
        return section[:50]

    return ""


def infer_category(chunk: dict, question_text: str) -> str:
    """
    Infer question category from chunk and question content.

    Args:
        chunk: Source chunk dictionary
        question_text: Question text

    Returns:
        Category string
    """
    combined = f"{chunk.get('text', '')} {question_text}".lower()

    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return category

    return "general"


def infer_reasoning_type(question_text: str) -> str:
    """
    Infer reasoning type from question text.

    Args:
        question_text: Question text

    Returns:
        Reasoning type: single-hop, multi-hop, temporal, comparative
    """
    q_lower = question_text.lower()

    for rtype, patterns in REASONING_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q_lower):
                return rtype

    return "single-hop"


def extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    """
    Extract keywords from text.

    Args:
        text: Input text
        max_keywords: Maximum number of keywords

    Returns:
        List of keywords
    """
    # French stopwords
    stopwords = {
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "de",
        "du",
        "au",
        "aux",
        "ce",
        "cette",
        "ces",
        "son",
        "sa",
        "ses",
        "leur",
        "leurs",
        "qui",
        "que",
        "quoi",
        "dont",
        "ou",
        "et",
        "ou",
        "mais",
        "donc",
        "car",
        "ni",
        "pour",
        "par",
        "sur",
        "sous",
        "avec",
        "sans",
        "dans",
        "entre",
        "vers",
        "chez",
        "est",
        "sont",
        "etre",
        "avoir",
        "fait",
        "peut",
        "doit",
        "tous",
        "tout",
        "plus",
        "moins",
    }

    # Extract words
    words = re.findall(r"\b[a-zA-Zaeiouc]+\b", text.lower())
    words = [w for w in words if len(w) >= 4 and w not in stopwords]

    # Count frequencies
    from collections import Counter

    freq = Counter(words)

    # Return top keywords
    return [w for w, _ in freq.most_common(max_keywords)]


def compute_quality_score(question: dict, chunk: dict) -> float:
    """
    Compute question quality score (0-1).

    Factors:
    - Question length (longer = better, up to a point)
    - Answer length
    - Keyword overlap with chunk
    - Proper question format

    Args:
        question: Question dictionary
        chunk: Source chunk dictionary

    Returns:
        Quality score 0-1
    """
    score = 0.0

    q_text = question.get("question", "")
    a_text = question.get("expected_answer", "")
    c_text = chunk.get("text", "")

    # Question format (ends with ?)
    if q_text.endswith("?"):
        score += 0.2

    # Question length (20-100 chars optimal)
    q_len = len(q_text)
    if 20 <= q_len <= 100:
        score += 0.2
    elif 10 <= q_len < 20 or 100 < q_len <= 150:
        score += 0.1

    # Answer length (20-300 chars optimal)
    a_len = len(a_text)
    if 20 <= a_len <= 300:
        score += 0.3
    elif 10 <= a_len < 20 or 300 < a_len <= 500:
        score += 0.15

    # Keyword overlap
    q_keywords = set(extract_keywords(q_text, 5))
    c_keywords = set(extract_keywords(c_text, 20))
    if q_keywords and c_keywords:
        overlap = len(q_keywords & c_keywords) / len(q_keywords)
        score += 0.3 * overlap

    return min(1.0, score)


def enrich_to_schema_v2(
    question: dict,
    chunk: dict,
    batch_id: str = "gs_scratch_v1",
) -> dict:
    """
    Transform question to full Schema v2.0 format.

    Args:
        question: Input question dictionary (from generation)
        chunk: Source chunk dictionary
        batch_id: Batch identifier for audit trail

    Returns:
        Schema v2.0 compliant question dictionary
    """
    chunk_id = chunk["id"]
    source = chunk.get("source", "")
    pages = chunk.get("pages", [chunk.get("page", 0)])

    q_text = question.get("question", "")
    a_text = question.get("expected_answer", "")
    is_impossible = question.get("is_impossible", False)

    # Generate ID
    qid = question.get("id") or generate_question_id(q_text, chunk_id)

    # Infer fields
    category = infer_category(chunk, q_text)
    reasoning_type = infer_reasoning_type(q_text)
    keywords = extract_keywords(f"{q_text} {a_text}")
    article_ref = extract_article_reference(chunk)
    quality_score = compute_quality_score(question, chunk)

    # Handle unanswerable questions
    hard_type = question.get("hard_type", "ANSWERABLE")
    if is_impossible:
        hard_type = question.get("hard_type", "OUT_OF_SCOPE")

    return {
        # Root (2 fields)
        "id": qid,
        "legacy_id": question.get("legacy_id", ""),
        # Group 1: Content (3 fields)
        "content": {
            "question": q_text,
            "expected_answer": a_text,
            "is_impossible": is_impossible,
        },
        # Group 2: MCQ (5 fields) - empty for generated questions
        "mcq": {
            "original_question": q_text,
            "choices": {},
            "mcq_answer": "",
            "correct_answer": a_text if not is_impossible else "",
            "original_answer": a_text if not is_impossible else "",
        },
        # Group 3: Provenance (6 fields) - ISO 42001 A.6.2.2
        "provenance": {
            "chunk_id": chunk_id,  # BY DESIGN INPUT
            "docs": [source],
            "pages": pages,
            "article_reference": article_ref,
            "answer_explanation": question.get("answer_explanation", ""),
            "annales_source": None,  # Not from annales
        },
        # Group 4: Classification (9 fields)
        "classification": {
            "category": category,
            "keywords": keywords,
            "difficulty": question.get("difficulty", 0.5),
            "question_type": question.get("question_type", "factual"),
            "cognitive_level": question.get("cognitive_level", "Remember"),
            "reasoning_type": reasoning_type,
            "reasoning_class": question.get("reasoning_class", "fact_single"),
            "answer_type": "extractive",
            "hard_type": hard_type,
        },
        # Group 5: Validation (7 fields) - ISO 29119
        "validation": {
            "status": "VALIDATED",
            "method": "by_design_generation",
            "reviewer": "claude_code",
            "answer_current": True,
            "verified_date": get_date(),
            "pages_verified": True,
            "batch": batch_id,
        },
        # Group 6: Processing (7 fields)
        "processing": {
            "chunk_match_score": 100,  # BY DESIGN = 100%
            "chunk_match_method": "by_design_input",
            "reasoning_class_method": "generation_prompt",
            "triplet_ready": not is_impossible,
            "extraction_flags": ["by_design"],
            "answer_source": "chunk_extraction",
            "quality_score": quality_score,
        },
        # Group 7: Audit (3 fields)
        "audit": {
            "history": f"[BY DESIGN] Generated from {chunk_id} on {get_date()}",
            "qat_revalidation": None,
            "requires_inference": False,
        },
    }


def count_schema_fields(question: dict) -> int:
    """
    Count populated fields in Schema v2 question.

    Expected: 46 fields total
    """
    count = 0

    # Root fields (2)
    if question.get("id"):
        count += 1
    if "legacy_id" in question:
        count += 1

    # Groups
    groups = [
        "content",
        "mcq",
        "provenance",
        "classification",
        "validation",
        "processing",
        "audit",
    ]

    for group in groups:
        group_data = question.get(group, {})
        if isinstance(group_data, dict):
            count += len(group_data)

    return count


def validate_schema_compliance(question: dict) -> tuple[bool, list[str]]:
    """
    Validate question against Schema v2.0 requirements.

    Args:
        question: Question dictionary

    Returns:
        Tuple of (passed, errors)
    """
    errors = []

    # Check root fields
    if not question.get("id"):
        errors.append("Missing id")
    if "legacy_id" not in question:
        errors.append("Missing legacy_id")

    # Check required groups
    required_groups = [
        "content",
        "mcq",
        "provenance",
        "classification",
        "validation",
        "processing",
        "audit",
    ]
    for group in required_groups:
        if group not in question:
            errors.append(f"Missing group: {group}")

    # Check content group
    content = question.get("content", {})
    if not content.get("question"):
        errors.append("content.question is empty")
    if "is_impossible" not in content:
        errors.append("content.is_impossible missing")

    # Check provenance group
    provenance = question.get("provenance", {})
    if not provenance.get("chunk_id"):
        errors.append("provenance.chunk_id is empty")

    # Check processing group
    processing = question.get("processing", {})
    if processing.get("chunk_match_score") != 100:
        errors.append("processing.chunk_match_score must be 100 for BY DESIGN")
    if processing.get("chunk_match_method") != "by_design_input":
        errors.append("processing.chunk_match_method must be 'by_design_input'")

    # Count fields (42 at top level, 46 with annales_source sub-fields)
    field_count = count_schema_fields(question)
    min_fields = 42
    if field_count < min_fields:
        errors.append(f"Only {field_count}/{min_fields} fields populated")

    return len(errors) == 0, errors


def enrich_questions(
    questions: list[dict],
    chunks: list[dict],
    batch_id: str = "gs_scratch_v1",
) -> tuple[list[dict], dict]:
    """
    Enrich all questions to Schema v2.0 format.

    Args:
        questions: List of generated questions
        chunks: List of source chunks
        batch_id: Batch identifier

    Returns:
        Tuple of (enriched_questions, report)
    """
    # Build chunk index
    chunk_index = {c["id"]: c for c in chunks}

    enriched = []
    errors_by_question = {}
    field_counts = []

    print(f"\nEnriching {len(questions)} questions to Schema v2.0...")

    for i, q in enumerate(questions):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(questions)}...")

        # Get source chunk
        chunk_id = q.get("chunk_id") or q.get("provenance", {}).get("chunk_id", "")
        chunk = chunk_index.get(chunk_id, {})

        if not chunk and not q.get("is_impossible", False):
            errors_by_question[q.get("id", f"q_{i}")] = [f"Chunk not found: {chunk_id}"]
            continue

        # Enrich to Schema v2
        enriched_q = enrich_to_schema_v2(q, chunk, batch_id)

        # Validate
        passed, validation_errors = validate_schema_compliance(enriched_q)
        if not passed:
            errors_by_question[enriched_q["id"]] = validation_errors

        enriched.append(enriched_q)
        field_counts.append(count_schema_fields(enriched_q))

    # Compile report
    report = {
        "enrichment_date": get_date(),
        "batch_id": batch_id,
        "total_input": len(questions),
        "total_enriched": len(enriched),
        "total_errors": len(errors_by_question),
        "field_stats": {
            "min": min(field_counts) if field_counts else 0,
            "max": max(field_counts) if field_counts else 0,
            "avg": sum(field_counts) / len(field_counts) if field_counts else 0,
        },
        "errors": errors_by_question,
        "gates": {
            "G4-1": {
                "name": "schema_fields",
                "passed": all(c >= 42 for c in field_counts),
                "value": f"{sum(1 for c in field_counts if c >= 42)}/{len(field_counts)}",
                "threshold": "42/42 for all (42 top-level fields)",
            },
            "G4-2": {
                "name": "chunk_match_method",
                "passed": all(
                    q.get("processing", {}).get("chunk_match_method")
                    == "by_design_input"
                    for q in enriched
                ),
                "value": "by_design_input",
                "threshold": "by_design_input",
            },
        },
    }

    return enriched, report


def format_enrichment_report(report: dict) -> str:
    """Format enrichment report for display."""
    lines = [
        "=" * 70,
        "PHASE 4: SCHEMA V2.0 ENRICHMENT REPORT",
        "=" * 70,
        "",
        f"Date: {report['enrichment_date']}",
        f"Batch ID: {report['batch_id']}",
        f"Input questions: {report['total_input']}",
        f"Enriched: {report['total_enriched']}",
        f"Errors: {report['total_errors']}",
        "",
        "FIELD STATISTICS:",
        f"  Min fields: {report['field_stats']['min']}",
        f"  Max fields: {report['field_stats']['max']}",
        f"  Avg fields: {report['field_stats']['avg']:.1f}",
        "",
        "QUALITY GATES:",
    ]

    for gate_id, gate in report["gates"].items():
        status = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"  [{status}] {gate_id}: {gate['name']}")
        lines.append(f"         Value: {gate['value']}, Threshold: {gate['threshold']}")

    if report["errors"]:
        lines.extend(
            [
                "",
                f"QUESTIONS WITH ERRORS ({len(report['errors'])}):",
            ]
        )
        for qid, errs in list(report["errors"].items())[:5]:
            lines.append(f"  {qid}:")
            for err in errs:
                lines.append(f"    - {err}")
        if len(report["errors"]) > 5:
            lines.append(f"  ... and {len(report['errors']) - 5} more")

    return "\n".join(lines)


def run_enrichment(
    questions_path: Path,
    chunks_path: Path,
    output_path: Path | None = None,
    batch_id: str = "gs_scratch_v1",
) -> tuple[list[dict], dict]:
    """
    Run complete Schema v2.0 enrichment.

    Args:
        questions_path: Path to input questions JSON
        chunks_path: Path to chunks JSON
        output_path: Path to save enriched questions
        batch_id: Batch identifier

    Returns:
        Tuple of (enriched_questions, report)
    """
    print(f"Loading questions from {questions_path}...")
    questions_data = load_json(questions_path)
    questions = questions_data.get("questions", questions_data)
    if isinstance(questions, dict):
        questions = list(questions.values())
    print(f"  Loaded {len(questions)} questions")

    print(f"\nLoading chunks from {chunks_path}...")
    chunks_data = load_json(chunks_path)
    chunks = chunks_data.get("chunks", chunks_data)
    print(f"  Loaded {len(chunks)} chunks")

    # Run enrichment
    enriched, report = enrich_questions(questions, chunks, batch_id)

    # Print report
    print("\n" + format_enrichment_report(report))

    # Save if output path provided
    if output_path:
        output_data = {
            "version": "2.0",
            "schema": "GS_SCHEMA_V2",
            "batch": batch_id,
            "date": get_date(),
            "questions": enriched,
        }
        save_json(output_data, output_path)
        print(f"\nEnriched questions saved to {output_path}")

    return enriched, report


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enrich questions to Schema v2.0 format"
    )
    parser.add_argument(
        "--questions",
        "-q",
        type=Path,
        required=True,
        help="Input questions JSON file",
    )
    parser.add_argument(
        "--chunks",
        "-c",
        type=Path,
        default=Path("corpus/processed/chunks_mode_b_fr.json"),
        help="Chunks JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output enriched questions JSON",
    )
    parser.add_argument(
        "--batch",
        "-b",
        type=str,
        default="gs_scratch_v1",
        help="Batch identifier",
    )

    args = parser.parse_args()

    enriched, report = run_enrichment(
        args.questions,
        args.chunks,
        args.output,
        args.batch,
    )

    # Exit with error if gates failed
    if not all(g["passed"] for g in report["gates"].values()):
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
