"""
Extraction de texte depuis PDF - Pocket Arbiter

Ce module extrait le contenu textuel des fichiers PDF du corpus
avec preservation des metadonnees (page, section).

ISO Reference:
    - ISO/IEC 12207 §7.3.3 - Implementation
    - ISO/IEC 25010 §4.5 - Reliability

Dependencies:
    - PyMuPDF (fitz) >= 1.23.0

Usage:
    python extract_pdf.py --input ../corpus/fr --output ../corpus/processed/raw_fr
    python extract_pdf.py --input ../corpus/intl --output ../corpus/processed/raw_intl

Example:
    >>> from scripts.pipeline.extract_pdf import extract_pdf
    >>> result = extract_pdf(Path("corpus/fr/LA-octobre2025.pdf"))
    >>> print(result["total_pages"])
    227
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --- Constants ---

SECTION_PATTERNS = [
    r"^Article\s+\d+",  # Article 4.1
    r"^Chapitre\s+\d+",  # Chapitre 3
    r"^Section\s+\d+",  # Section 2
    r"^\d+\.\d+",  # 4.1.2
]

MIN_PAGE_CHARS = 50  # Pages avec moins de caracteres sont ignorees


# --- Main Functions ---


def extract_pdf(pdf_path: Path) -> dict:
    """
    Extrait le texte d'un PDF avec metadonnees.

    Cette fonction lit un fichier PDF et extrait le contenu textuel
    de chaque page, en detectant les titres de section et en
    preservant les numeros de page.

    Args:
        pdf_path: Chemin vers le fichier PDF a extraire.

    Returns:
        dict contenant:
            - filename (str): Nom du fichier PDF
            - pages (list[dict]): Liste des pages avec:
                - page_num (int): Numero de page (1-indexed)
                - text (str): Contenu textuel
                - section (str|None): Titre de section detecte
            - total_pages (int): Nombre total de pages
            - extraction_date (str): Timestamp ISO 8601

    Raises:
        FileNotFoundError: Si le fichier PDF n'existe pas.
        ValueError: Si le fichier n'est pas un PDF valide.
        RuntimeError: Si l'extraction echoue (PDF corrompu).

    Example:
        >>> result = extract_pdf(Path("corpus/fr/LA-octobre2025.pdf"))
        >>> len(result["pages"])
        227
        >>> result["pages"][0]["page_num"]
        1
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not validate_pdf(pdf_path):
        raise ValueError(f"Not a valid PDF file: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}") from e

    pages = []
    total_pages = len(doc)

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text()

        # Normaliser l'encodage UTF-8 (ISO conforme)
        # PyMuPDF peut retourner du texte avec encodage mixte
        from scripts.pipeline.utils import normalize_text

        text = normalize_text(text)

        # Skip pages with minimal content
        if len(text.strip()) < MIN_PAGE_CHARS:
            continue

        section = detect_section(text)
        pages.append(
            {
                "page_num": page_num + 1,  # 1-indexed
                "text": text,
                "section": section,
            }
        )

    doc.close()

    from scripts.pipeline.utils import get_timestamp

    return {
        "filename": pdf_path.name,
        "pages": pages,
        "total_pages": total_pages,
        "extraction_date": get_timestamp(),
    }


def detect_section(text: str) -> Optional[str]:
    """
    Detecte le titre de section dans un bloc de texte.

    Recherche des patterns comme "Article X", "Chapitre Y", etc.
    au debut du texte.

    Args:
        text: Texte a analyser (premiere ligne ou paragraphe).

    Returns:
        Titre de section detecte ou None si aucun pattern match.

    Example:
        >>> detect_section("Article 4.1 - Le toucher-jouer")
        "Article 4.1 - Le toucher-jouer"
        >>> detect_section("Texte normal sans section")
        None
    """
    if not text:
        return None

    first_line = text.split("\n")[0].strip()

    for pattern in SECTION_PATTERNS:
        if re.match(pattern, first_line):
            return first_line

    return None


def extract_corpus(
    input_dir: Path, output_dir: Path, corpus_name: str = "unknown"
) -> dict:
    """
    Extrait tous les PDF d'un dossier corpus.

    Parcourt tous les fichiers PDF dans input_dir, les extrait
    et sauvegarde les resultats dans output_dir.

    Args:
        input_dir: Dossier contenant les PDF sources.
        output_dir: Dossier de destination pour les extractions.
        corpus_name: Nom du corpus (fr ou intl).

    Returns:
        dict rapport d'extraction avec:
            - corpus (str): Nom du corpus
            - files_processed (int): Nombre de fichiers traites
            - total_pages (int): Nombre total de pages
            - errors (list): Liste des erreurs rencontrees
            - timestamp (str): Date d'execution

    Example:
        >>> report = extract_corpus(
        ...     Path("corpus/fr"),
        ...     Path("corpus/processed/raw_fr"),
        ...     corpus_name="fr"
        ... )
        >>> print(report["files_processed"])
        29
    """
    from scripts.pipeline.utils import get_timestamp, list_pdf_files, save_json

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list_pdf_files(input_dir)
    logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")

    errors: list[str] = []
    files_processed = 0
    total_pages = 0

    for pdf_path in pdf_files:
        try:
            logger.info(f"Extracting: {pdf_path.name}")
            result = extract_pdf(pdf_path)

            output_file = output_dir / f"{pdf_path.stem}.json"
            save_json(result, output_file)

            files_processed += 1
            total_pages += result["total_pages"]

        except Exception as e:
            error_msg = f"{pdf_path.name}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    report = {
        "corpus": corpus_name,
        "files_processed": files_processed,
        "total_pages": total_pages,
        "errors": errors,
        "timestamp": get_timestamp(),
    }

    report_file = output_dir / "extraction_report.json"
    save_json(report, report_file)

    return report


def validate_pdf(pdf_path: Path) -> bool:
    """
    Verifie qu'un fichier est un PDF valide.

    Checks:
        1. File exists
        2. Has .pdf extension
        3. Starts with PDF magic number (%PDF-)

    Args:
        pdf_path: Chemin vers le fichier a valider.

    Returns:
        True si le fichier est un PDF valide, False sinon.
    """
    if not pdf_path.exists():
        return False
    if pdf_path.suffix.lower() != ".pdf":
        return False

    # Check PDF magic number (ISO 32000-1:2008)
    try:
        with open(pdf_path, "rb") as f:
            magic = f.read(5)
            return magic == b"%PDF-"
    except (OSError, IOError):
        return False


# --- CLI ---


def main() -> None:
    """Point d'entree CLI pour l'extraction PDF."""
    parser = argparse.ArgumentParser(
        description="Extraction de texte depuis PDF pour Pocket Arbiter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python extract_pdf.py --input corpus/fr --output corpus/processed/raw_fr
    python extract_pdf.py --input corpus/intl --output corpus/processed/raw_intl --verbose
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Dossier source contenant les PDF",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Dossier de destination pour les extractions",
    )

    parser.add_argument(
        "--corpus",
        "-c",
        type=str,
        choices=["fr", "intl"],
        default="fr",
        help="Nom du corpus (default: fr)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Afficher les logs detailles",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Extraction corpus {args.corpus}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    report = extract_corpus(args.input, args.output, corpus_name=args.corpus)

    logger.info(f"Processed: {report['files_processed']} files")
    logger.info(f"Total pages: {report['total_pages']}")
    if report["errors"]:
        logger.warning(f"Errors: {len(report['errors'])}")


if __name__ == "__main__":
    main()
