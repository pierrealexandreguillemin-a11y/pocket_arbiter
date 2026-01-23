"""
Extraction PDF avec Docling - Pocket Arbiter

Ce module extrait le contenu des fichiers PDF en utilisant Docling,
qui offre une meilleure extraction de structure (tables, sections).

ISO Reference:
    - ISO/IEC 12207 §7.3.3 - Implementation
    - ISO/IEC 25010 §4.5 - Reliability
    - ISO/IEC 42001 - AI traceability

Dependencies:
    - docling >= 2.0.0

Usage:
    python -m scripts.pipeline.extract_docling \
        --input corpus/fr \
        --output corpus/processed/docling_fr

Example:
    >>> from scripts.pipeline.extract_docling import extract_pdf_docling
    >>> result = extract_pdf_docling(Path("corpus/fr/LA-octobre2025.pdf"))
    >>> print(result["total_pages"])
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

# Workaround for Windows symlink permission issue with HuggingFace
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from scripts.pipeline.utils import get_timestamp, list_pdf_files, normalize_text, save_json

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _extract_row_cells(row: list[Any]) -> list[str]:
    """Extract text from a row of cells (ISO 25010 - reduced complexity)."""
    return [normalize_text(getattr(cell, "text", "") or "") for cell in row]


def _build_table_text(headers: list[str], rows: list[list[str]]) -> str:
    """Build text representation for embedding/search (ISO 25010)."""
    lines = []
    if headers:
        lines.append(" | ".join(headers))
        lines.append("-" * 40)
    lines.extend(" | ".join(row) for row in rows)
    return "\n".join(lines)


def _extract_table_content(table: Any, doc: Any) -> dict[str, Any]:
    """
    Extract structured content from a docling TableItem.

    Uses table.data.grid for cell-by-cell extraction (ISO 42001 traceability).

    Args:
        table: Docling TableItem object.
        doc: Parent DoclingDocument for context.

    Returns:
        dict with num_rows, num_cols, headers, rows, text.
    """
    result: dict[str, Any] = {
        "num_rows": 0,
        "num_cols": 0,
        "headers": [],
        "rows": [],
        "text": "",
    }

    # Get table data structure
    if not hasattr(table, "data") or table.data is None:
        return result

    data = table.data
    result["num_rows"] = getattr(data, "num_rows", 0)
    result["num_cols"] = getattr(data, "num_cols", 0)

    grid = getattr(data, "grid", None)
    if not grid:
        return result

    # Extract all rows
    rows_text = [_extract_row_cells(row) for row in grid]

    # First row is header (by convention or if marked)
    if rows_text:
        result["headers"] = rows_text[0]
        result["rows"] = rows_text[1:] if len(rows_text) > 1 else []

    result["text"] = _build_table_text(result["headers"], result["rows"])
    return result


def extract_pdf_docling(
    pdf_path: Path,
    do_ocr: bool = False,
    do_table_structure: bool = True,  # Required for table extraction (ISO 42001)
) -> dict[str, Any]:
    """
    Extrait le contenu d'un PDF avec Docling.

    Docling offre une extraction structuree avec detection
    automatique des sections, tables et elements.

    Args:
        pdf_path: Chemin vers le fichier PDF.
        do_ocr: Activer l'OCR pour les pages scannees.
        do_table_structure: Activer la structure des tables (plus lent).

    Returns:
        dict contenant:
            - filename (str): Nom du fichier
            - markdown (str): Contenu en Markdown
            - tables (list): Tables extraites
            - extraction_date (str): Timestamp
            - extractor (str): "docling"

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        RuntimeError: Si l'extraction echoue.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    pipeline_options.do_table_structure = do_table_structure

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    try:
        result = converter.convert(pdf_path)
        doc = result.document
    except Exception as e:
        raise RuntimeError(f"Docling extraction failed: {e}") from e

    # Export to markdown
    markdown = doc.export_to_markdown()
    markdown = normalize_text(markdown)

    # Extract tables with proper structure (ISO 42001 traceability)
    tables = []
    if doc.tables:
        for i, table in enumerate(doc.tables):
            table_content = _extract_table_content(table, doc)
            table_data = {
                "id": f"{pdf_path.stem}-table{i}",
                "source": pdf_path.name,
                "num_rows": table_content["num_rows"],
                "num_cols": table_content["num_cols"],
                "headers": table_content["headers"],
                "rows": table_content["rows"],
                "text": table_content["text"],
            }
            tables.append(table_data)

    # Export full document structure for chunking with page provenance (ISO 42001)
    # See: https://github.com/docling-project/docling/discussions/1012
    doc_dict = doc.export_to_dict()

    return {
        "filename": pdf_path.name,
        "markdown": markdown,
        "tables": tables,
        "total_tables": len(tables),
        "extraction_date": get_timestamp(),
        "extractor": "docling",
        "docling_document": doc_dict,  # Full DoclingDocument for page provenance (ISO 42001)
    }


def extract_corpus_docling(
    input_dir: Path,
    output_dir: Path,
    corpus_name: str = "fr",
    do_ocr: bool = False,
) -> dict[str, Any]:
    """
    Extrait tous les PDF d'un corpus avec Docling.

    Args:
        input_dir: Dossier source contenant les PDF.
        output_dir: Dossier de destination.
        corpus_name: Nom du corpus (fr, intl).
        do_ocr: Activer l'OCR.

    Returns:
        Rapport d'extraction.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list_pdf_files(input_dir)
    logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")

    errors: list[str] = []
    files_processed = 0
    total_tables = 0

    for pdf_path in pdf_files:
        try:
            logger.info(f"Extracting: {pdf_path.name}")
            result = extract_pdf_docling(pdf_path, do_ocr=do_ocr)
            result["corpus"] = corpus_name

            # Save individual result
            output_file = output_dir / f"{pdf_path.stem}.json"
            save_json(result, output_file)

            files_processed += 1
            total_tables += result["total_tables"]

        except Exception as e:
            error_msg = f"{pdf_path.name}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    report = {
        "corpus": corpus_name,
        "extractor": "docling",
        "files_processed": files_processed,
        "total_tables": total_tables,
        "errors": errors,
        "timestamp": get_timestamp(),
    }

    report_file = output_dir / "extraction_report.json"
    save_json(report, report_file)
    logger.info(f"Report saved: {report_file}")

    return report


def main() -> None:
    """CLI pour l'extraction Docling."""
    parser = argparse.ArgumentParser(
        description="Extraction PDF avec Docling - Pocket Arbiter",
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
        help="Dossier de destination",
    )

    parser.add_argument(
        "--corpus",
        "-c",
        choices=["fr", "intl"],
        default="fr",
        help="Nom du corpus (default: fr)",
    )

    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Activer l'OCR pour les pages scannees",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Logs detailles",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Extraction corpus {args.corpus} avec Docling")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    report = extract_corpus_docling(
        input_dir=args.input,
        output_dir=args.output,
        corpus_name=args.corpus,
        do_ocr=args.ocr,
    )

    print("\n=== Extraction Report ===")
    for k, v in report.items():
        if k != "errors":
            print(f"{k}: {v}")
    if report["errors"]:
        print(f"errors: {len(report['errors'])} failures")


if __name__ == "__main__":
    main()
