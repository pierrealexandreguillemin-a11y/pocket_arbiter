"""
Table Extractor - Pocket Arbiter

Extract tables from PDF documents using Camelot.
Tables are converted to JSON with metadata for RAG retrieval.

Types of tables in chess regulations:
- Time control tables (cadence)
- Penalty grids
- Elo rating tables
- Tiebreak matrices

ISO Reference:
    - ISO/IEC 25010 S4.2 - Performance efficiency
    - ISO/IEC 42001 - AI traceability

Changelog:
    - 2026-01-18: Initial implementation (Step 4 chunking strategy)

Usage:
    python -m scripts.pipeline.table_extractor \
        --input corpus/fr \
        --output corpus/processed/tables_fr.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import camelot

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_tables_from_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """
    Extract all tables from a PDF file.

    Args:
        pdf_path: Path to PDF file.

    Returns:
        List of table dictionaries with metadata.
    """
    tables_data = []

    try:
        # Try lattice method first (for tables with clear lines)
        tables = camelot.read_pdf(
            str(pdf_path),
            pages="all",
            flavor="lattice",
            line_scale=40,
        )
        logger.info(f"  Lattice: Found {len(tables)} tables")

        if len(tables) == 0:
            # Fallback to stream method (for tables without lines)
            tables = camelot.read_pdf(
                str(pdf_path),
                pages="all",
                flavor="stream",
                edge_tol=50,
            )
            logger.info(f"  Stream: Found {len(tables)} tables")

    except Exception as e:
        logger.warning(f"  Camelot failed for {pdf_path.name}: {e}")
        return []

    for i, table in enumerate(tables):
        # Get table as DataFrame
        df = table.df

        # Skip empty or tiny tables
        if df.empty or df.shape[0] < 2 or df.shape[1] < 2:
            continue

        # Convert to list of lists with UTF-8 normalization (ISO 42001)
        from scripts.pipeline.utils import normalize_text

        rows = df.values.tolist()
        # Normalize text to fix Camelot encoding issues (Pièce → Piece not Pi�ce)
        headers = [normalize_text(str(h)) if h else "" for h in (rows[0] if rows else [])]
        data_rows = [
            [normalize_text(str(cell)) if cell else "" for cell in row]
            for row in (rows[1:] if len(rows) > 1 else [])
        ]

        # Detect table type based on content
        table_type = detect_table_type(headers, data_rows)

        # Convert to text representation for embedding
        text_repr = table_to_text(headers, data_rows, table_type)

        table_dict = {
            "id": f"{pdf_path.stem}-table{i}",
            "source": pdf_path.name,
            "page": table.page,
            "table_index": i,
            "table_type": table_type,
            "headers": headers,
            "rows": data_rows,
            "text": text_repr,
            "accuracy": table.accuracy,
            "whitespace": table.whitespace,
        }
        tables_data.append(table_dict)

    return tables_data


def detect_table_type(headers: list, rows: list) -> str:
    """
    Detect the type of table based on headers and content.

    Types:
    - cadence: Time control tables
    - penalty: Penalty grids
    - elo: Elo rating tables
    - tiebreak: Tiebreak matrices
    - other: Unknown type
    """
    header_text = " ".join(str(h).lower() for h in headers)
    all_text = header_text + " " + " ".join(
        str(cell).lower() for row in rows for cell in row
    )

    cadence_kw = ["cadence", "temps", "minutes", "secondes", "increment"]
    penalty_kw = ["penalite", "sanction", "amende", "suspension"]
    elo_kw = ["elo", "classement", "points", "coefficient"]
    tiebreak_kw = ["departage", "buchholz", "sonneborn", "cumulatif"]

    if any(kw in all_text for kw in cadence_kw):
        return "cadence"
    if any(kw in all_text for kw in penalty_kw):
        return "penalty"
    if any(kw in all_text for kw in elo_kw):
        return "elo"
    if any(kw in all_text for kw in tiebreak_kw):
        return "tiebreak"

    return "other"


def table_to_text(headers: list, rows: list, table_type: str) -> str:
    """
    Convert table to text representation for embedding.

    Args:
        headers: Table headers.
        rows: Table data rows.
        table_type: Detected table type.

    Returns:
        Text representation of the table.
    """
    lines = []

    # Add type context
    type_context = {
        "cadence": "Table de cadence (temps de reflexion)",
        "penalty": "Grille de penalites",
        "elo": "Table de classement Elo",
        "tiebreak": "Systeme de departage",
        "other": "Table",
    }
    lines.append(type_context.get(table_type, "Table"))
    lines.append("")

    # Add headers
    if headers:
        lines.append(" | ".join(str(h) for h in headers))
        lines.append("-" * 40)

    # Add rows
    for row in rows:
        lines.append(" | ".join(str(cell) for cell in row))

    return "\n".join(lines)


def process_corpus_tables(
    input_dir: Path,
    output_file: Path,
    corpus: str = "fr",
) -> dict[str, Any]:
    """
    Extract tables from all PDFs in a corpus directory.

    Args:
        input_dir: Directory containing PDF files.
        output_file: Output JSON file.
        corpus: Corpus code (fr, intl).

    Returns:
        Processing report.
    """
    pdf_files = sorted(input_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")

    all_tables = []
    files_with_tables = 0

    for pdf_path in pdf_files:
        logger.info(f"Processing: {pdf_path.name}")
        tables = extract_tables_from_pdf(pdf_path)

        if tables:
            files_with_tables += 1
            for table in tables:
                table["corpus"] = corpus
            all_tables.extend(tables)

    # Save output
    output_data = {
        "corpus": corpus,
        "config": {
            "extractor": "camelot",
            "methods": ["lattice", "stream"],
        },
        "total_tables": len(all_tables),
        "files_with_tables": files_with_tables,
        "total_files": len(pdf_files),
        "tables": all_tables,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(all_tables)} tables to {output_file}")

    # Stats by type
    type_counts: dict[str, int] = {}
    for table in all_tables:
        t = table["table_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    report = {
        "corpus": corpus,
        "extractor": "camelot",
        "total_tables": len(all_tables),
        "files_with_tables": files_with_tables,
        "total_files": len(pdf_files),
        "by_type": type_counts,
    }

    return report


def main() -> None:
    """CLI for table extraction."""
    parser = argparse.ArgumentParser(
        description="Table Extractor - Pocket Arbiter (Camelot)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input directory with PDF files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output tables JSON file",
    )
    parser.add_argument(
        "--corpus",
        "-c",
        choices=["fr", "intl"],
        default="fr",
        help="Corpus code (default: fr)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    report = process_corpus_tables(
        input_dir=args.input,
        output_file=args.output,
        corpus=args.corpus,
    )

    print("\n=== Table Extraction Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
