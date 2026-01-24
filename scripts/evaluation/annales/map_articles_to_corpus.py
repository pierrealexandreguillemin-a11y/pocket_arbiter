"""
Map article references from annales to corpus documents.

This module maps article references like "Article 1.3 des règles du jeu"
to their corresponding corpus documents (e.g., LA-octobre2025.pdf).

ISO Reference:
    - ISO/IEC 42001 A.7.3 - Data traceability

Usage:
    python -m scripts.evaluation.annales.map_articles_to_corpus \
        --input data/evaluation/annales/parsed \
        --corpus corpus/processed/docling_fr \
        --output data/evaluation/annales/mapped

Example:
    >>> from scripts.evaluation.annales.map_articles_to_corpus import map_article_to_document
    >>> result = map_article_to_document("Article 1.3 des règles du jeu")
    >>> print(result["document"])  # "LA-octobre2025.pdf"
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from scripts.pipeline.utils import get_timestamp, save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Document mapping rules based on reference patterns
DOCUMENT_MAPPING_RULES = [
    # LA - Livre de l'Arbitre (règles du jeu FIDE)
    {
        "patterns": [
            r"règles?\s+du\s+jeu",
            r"regles?\s+du\s+jeu",
            r"^Article\s+\d+\.\d+",
            r"^Art\.?\s+\d+\.\d+",  # Abbreviated "Art 3.8.2" or "Art. 3.8.2"
            r"^Commentaire\s+[Aa]rticle",
            r"^LA\s*[-:]\s*Chap(?:itre)?\.?\s*\d+",  # LA - Chapitre X or LA : Chap. X
            r"LA\s*:\s*Chap(?:itre)?\.?\s*\d+",  # "LA : Chapitre 5.1"
            r"^LA\s*-?\s*\d+\.\d+",
            r"Annexe\s+[A-G]",
            r"^\d+\.\d+(?:\.\d+)?(?:\s|$|\.)",  # Bare article numbers: "9.6.2", "1.3"
            r"^A\.\d+",  # Appendix A (rapid): A.4.1, A.4.2
            r"^B\.\d+",  # Appendix B (blitz): B.1, B.2
            r"^Pr[eé]ambule",  # Preamble
            r"RIDNA",  # Règlement Intérieur DNA (in LA)
            r"RI\s*DNA",  # RI DNA format
            r"LA\s*:\s*Commentaire",  # LA : Commentaire X
            r"de\s+la\s+DNA",  # "de la DNA, art."
        ],
        "document": "LA-octobre2025.pdf",
        "category": "regles_jeu",
    },
    # R01 - Règles générales FFE
    {
        "patterns": [
            r"^R01\s*-",
            r"^R\.01\s*:",  # R.01 : format
            r"R01.*R[eè]gles?\s+g[eé]n[eé]ral",
            r"R[eè]gles?\s+g[eé]n[eé]rales?.*art",  # Règles générales, art. X
        ],
        "document": "R01_2025_26_Regles_generales.pdf",
        "category": "regles_ffe",
    },
    # R02 - Annexes règles générales
    {
        "patterns": [
            r"^R02\s*-",
            r"R02.*Annexe",
        ],
        "document": "R02_2025_26_Regles_generales_Annexes.pdf",
        "category": "regles_ffe",
    },
    # R03 - Compétitions homologuées
    {
        "patterns": [
            r"^R03\s*-",
            r"^R\.03\s*:",  # R.03 : format
            r"R03.*[Cc]omp[eé]titions?\s+homologu[eé]",
            r"[Cc]omp[eé]titions?\s+homologu[eé]es?.*art",
        ],
        "document": "R03_2025_26_Competitions_homologuees.pdf",
        "category": "competitions",
    },
    # A01 - Championnat de France individuel
    {
        "patterns": [
            r"^A01\s*-",
            r"A01.*[Cc]hampionnat.*France",
        ],
        "document": "A01_2025_26_Championnat_de_France.pdf",
        "category": "championnats",
    },
    # A02 - Championnat de France des clubs
    {
        "patterns": [
            r"^A02\s*-",
            r"A02.*[Cc]lubs?",
            r"[Ii]nterclubs?.*[Aa]rticle",
        ],
        "document": "A02_2025_26_Championnat_de_France_des_Clubs.pdf",
        "category": "interclubs",
    },
    # A03 - Championnat de France clubs rapides
    {
        "patterns": [
            r"^A03\s*-",
        ],
        "document": "A03_2025_26_Championnat_de_France_des_Clubs_rapides.pdf",
        "category": "interclubs",
    },
    # C01 - Coupe de France
    {
        "patterns": [
            r"^C01\s*-",
            r"^C\.01\s*:",  # C.01 : format
            r"C01.*[Cc]oupe",
            r"[Cc]oupe\s+de\s+[Ff]rance.*art",
        ],
        "document": "C01_2025_26_Coupe_de_France.pdf",
        "category": "coupes",
    },
    # C03 - Coupe Jean-Claude Loubatière
    {
        "patterns": [
            r"^C03\s*-",
            r"^C\.03\s*:",  # C.03 : format
            r"[Ll]oubati[eè]re",
        ],
        "document": "C03_2025_26_Coupe_Jean_Claude_Loubatiere.pdf",
        "category": "coupes",
    },
    # C04 - Coupe de la parité
    {
        "patterns": [
            r"^C04\s*-",
            r"^C04\s+",
            r"[Cc]oupe\s+de\s+la\s+parit[eé]",
            r"[Pp]arit[eé]",
        ],
        "document": "C04_2025_26_Coupe_de_la_parité.pdf",
        "category": "coupes",
    },
    # J01 - Championnat de France Jeunes
    {
        "patterns": [
            r"^J01\s*-",
            r"[Ii]nterclubs?\s+[Jj]eunes",
        ],
        "document": "J01_2025_26_Championnat_de_France_Jeunes.pdf",
        "category": "jeunes",
    },
    # J02 - Interclubs Jeunes
    {
        "patterns": [
            r"^J02\s*-",
        ],
        "document": "J02_2025_26_Championnat_de_France_Interclubs_Jeunes.pdf",
        "category": "jeunes",
    },
]


def _extract_article_number(reference: str) -> str | None:
    """Extract article number from reference (e.g., '1.3' from 'Article 1.3')."""
    # Pattern for article numbers like "1.3", "4.2.1", "6.11.4"
    match = re.search(r"(?:Article|Art\.?)\s*(\d+(?:\.\d+)+)", reference, re.IGNORECASE)
    if match:
        return match.group(1)

    # Pattern for chapter references like "Chapitre 2.1"
    match = re.search(r"Chapitre\s*(\d+(?:\.\d+)*)", reference, re.IGNORECASE)
    if match:
        return f"Ch.{match.group(1)}"

    # Pattern for bare article numbers at start: "9.6.2", "1.3"
    match = re.match(r"^(\d+(?:\.\d+)+)", reference.strip())
    if match:
        return match.group(1)

    return None


def _extract_section_info(reference: str) -> dict[str, Any]:
    """Extract structured section information from reference."""
    info: dict[str, Any] = {
        "article_num": _extract_article_number(reference),
        "is_commentary": "commentaire" in reference.lower(),
        "has_multiple_refs": " et " in reference or "," in reference,
    }

    # Extract chapter if present
    chapter_match = re.search(r"Chapitre\s*(\d+(?:\.\d+)*)", reference, re.IGNORECASE)
    if chapter_match:
        info["chapter"] = chapter_match.group(1)

    return info


def map_article_to_document(reference: str) -> dict[str, Any]:
    """
    Map an article reference to its corpus document.

    Args:
        reference: Article reference string from annales.

    Returns:
        Dict with document, category, article_num, confidence.
    """
    if not reference:
        return {
            "document": None,
            "category": "unknown",
            "article_num": None,
            "confidence": 0.0,
            "error": "Empty reference",
        }

    # Try each mapping rule
    for rule in DOCUMENT_MAPPING_RULES:
        for pattern in rule["patterns"]:
            if re.search(pattern, reference, re.IGNORECASE):
                section_info = _extract_section_info(reference)
                return {
                    "document": rule["document"],
                    "category": rule["category"],
                    "article_num": section_info["article_num"],
                    "is_commentary": section_info["is_commentary"],
                    "confidence": 0.9 if section_info["article_num"] else 0.7,
                    "matched_pattern": pattern,
                }

    # Default fallback - try to detect document code
    doc_code_match = re.match(r"^([A-Z]\d{2})\s*-", reference)
    if doc_code_match:
        code = doc_code_match.group(1)
        return {
            "document": f"{code}_unknown.pdf",
            "category": "unknown",
            "article_num": _extract_article_number(reference),
            "confidence": 0.5,
            "note": f"Unknown document code: {code}",
        }

    # Last resort - assume LA if mentions article
    if re.search(r"article", reference, re.IGNORECASE):
        return {
            "document": "LA-octobre2025.pdf",
            "category": "regles_jeu",
            "article_num": _extract_article_number(reference),
            "confidence": 0.4,
            "note": "Fallback to LA based on 'article' keyword",
        }

    return {
        "document": None,
        "category": "unknown",
        "article_num": None,
        "confidence": 0.0,
        "error": f"No matching pattern for: {reference[:50]}",
    }


def map_parsed_annales(parsed_file: Path, corpus_dir: Path) -> dict[str, Any]:
    """
    Map all article references in a parsed annales file.

    Args:
        parsed_file: Path to parsed annales JSON.
        corpus_dir: Path to corpus documents (for validation).

    Returns:
        Mapped data with document references added.
    """
    with open(parsed_file, encoding="utf-8") as f:
        data = json.load(f)

    # Get available corpus documents for validation
    available_docs = set()
    if corpus_dir.exists():
        available_docs = {f.stem + ".pdf" for f in corpus_dir.glob("*.json")}

    stats_total = 0
    stats_mapped = 0
    stats_unmapped = 0
    stats_high_confidence = 0
    documents_used: set[str] = set()

    for unit in data.get("units", []):
        for question in unit.get("questions", []):
            stats_total += 1
            ref = question.get("article_reference", "")

            mapping = map_article_to_document(ref)
            question["document_mapping"] = mapping

            if mapping.get("document"):
                stats_mapped += 1
                documents_used.add(mapping["document"])

                if mapping.get("confidence", 0) >= 0.7:
                    stats_high_confidence += 1

                # Validate document exists
                doc_stem = mapping["document"].replace(".pdf", "")
                if available_docs and doc_stem + ".pdf" not in available_docs:
                    mapping["validation"] = "document_not_found"
                else:
                    mapping["validation"] = "ok"
            else:
                stats_unmapped += 1

    stats: dict[str, Any] = {
        "total_questions": stats_total,
        "mapped": stats_mapped,
        "unmapped": stats_unmapped,
        "high_confidence": stats_high_confidence,
        "documents_used": list(documents_used),
    }

    return {
        "source_file": parsed_file.name,
        "mapping_date": get_timestamp(),
        "statistics": stats,
        "data": data,
    }


def map_all_parsed_annales(
    input_dir: Path,
    corpus_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Map all parsed annales files in a directory.

    Args:
        input_dir: Directory with parsed annales JSON files.
        corpus_dir: Directory with corpus documents.
        output_dir: Output directory for mapped files.

    Returns:
        Mapping report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_files = list(input_dir.glob("parsed_*.json"))
    if not parsed_files:
        logger.warning(f"No parsed files found in {input_dir}")
        return {"error": "No files found"}

    logger.info(f"Found {len(parsed_files)} parsed files to map")

    all_total = 0
    all_mapped = 0
    all_unmapped = 0
    all_documents: set[str] = set()

    results = []

    for parsed_file in parsed_files:
        logger.info(f"Mapping: {parsed_file.name}")

        result = map_parsed_annales(parsed_file, corpus_dir)
        results.append(result)

        # Aggregate stats
        result_stats = result["statistics"]
        all_total += result_stats["total_questions"]
        all_mapped += result_stats["mapped"]
        all_unmapped += result_stats["unmapped"]
        all_documents.update(result_stats["documents_used"])

        # Save mapped file
        output_file = output_dir / f"mapped_{parsed_file.stem.replace('parsed_', '')}.json"
        save_json(result["data"], output_file)
        logger.info(f"Saved: {output_file}")

    all_stats: dict[str, Any] = {
        "total_questions": all_total,
        "total_mapped": all_mapped,
        "total_unmapped": all_unmapped,
        "all_documents": sorted(all_documents),
    }

    report: dict[str, Any] = {
        "input_dir": str(input_dir),
        "corpus_dir": str(corpus_dir),
        "output_dir": str(output_dir),
        "files_processed": len(results),
        "statistics": all_stats,
        "mapping_rate": all_mapped / all_total if all_total > 0 else 0,
        "timestamp": get_timestamp(),
    }

    report_file = output_dir / "mapping_report.json"
    save_json(report, report_file)
    logger.info(f"Report saved: {report_file}")

    return report


def main() -> None:
    """CLI for article mapping."""
    parser = argparse.ArgumentParser(
        description="Map article references to corpus documents",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/evaluation/annales/parsed"),
        help="Input directory with parsed annales",
    )

    parser.add_argument(
        "--corpus",
        "-c",
        type=Path,
        default=Path("corpus/processed/docling_fr"),
        help="Corpus directory for validation",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/evaluation/annales/mapped"),
        help="Output directory",
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

    report = map_all_parsed_annales(args.input, args.corpus, args.output)

    print("\n=== Mapping Report ===")
    print(f"Files processed: {report.get('files_processed', 0)}")
    stats = report.get("statistics", {})
    print(f"Total questions: {stats.get('total_questions', 0)}")
    print(f"Mapped: {stats.get('total_mapped', 0)}")
    print(f"Unmapped: {stats.get('total_unmapped', 0)}")
    print(f"Mapping rate: {report.get('mapping_rate', 0):.1%}")
    print(f"Documents used: {len(stats.get('all_documents', []))}")


if __name__ == "__main__":
    main()
