"""
Migrate table summaries to add page numbers from DoclingDocument provenance.

This script adds page numbers to existing table summaries by:
1. Loading table summaries from tables_multivector_*.json
2. Finding corresponding tables in DoclingDocument extractions
3. Extracting page_no from table provenance
4. Updating table summaries with page numbers

ISO Reference:
    - ISO/IEC 42001 A.6.2.2 - AI traceability (100% page provenance)
    - ISO/IEC 12207 S7.3.3 - Implementation

Usage:
    python -m scripts.pipeline.migrate_table_summaries \
        --tables corpus/processed/tables_multivector_fr.json \
        --docling corpus/processed/docling_fr
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _extract_page_from_table_prov(table_data: dict) -> int:
    """Extract page_no from table provenance (ISO 25010 - reduced complexity)."""
    prov = table_data.get("prov", [])
    if prov and isinstance(prov[0], dict):
        return prov[0].get("page_no", 0)
    return 0


def _process_tables_dict(
    doc_tables: dict, source_stem: str, table_to_page: dict[str, int]
) -> None:
    """Process tables in dict format (ISO 25010 - reduced complexity)."""
    for ref, table_data in doc_tables.items():
        idx = ref.split("/")[-1] if "/" in ref else ref
        table_id = f"{source_stem}-table{idx}"
        page_no = _extract_page_from_table_prov(table_data)
        if page_no > 0:
            table_to_page[table_id] = page_no


def _process_tables_list(
    doc_tables: list, source_stem: str, table_to_page: dict[str, int]
) -> None:
    """Process tables in list format (ISO 25010 - reduced complexity)."""
    for i, table_data in enumerate(doc_tables):
        table_id = f"{source_stem}-table{i}"
        page_no = _extract_page_from_table_prov(table_data)
        if page_no > 0:
            table_to_page[table_id] = page_no


def build_table_to_page_mapping(docling_dir: Path) -> dict[str, int]:
    """
    Build mapping from table_id to page_no from all DoclingDocument files.

    Args:
        docling_dir: Directory containing docling extraction JSON files.

    Returns:
        Dict mapping table_id (e.g., "LA-octobre2025-table0") to page number.
    """
    table_to_page: dict[str, int] = {}

    json_files = list(docling_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != "extraction_report.json"]

    logger.info(f"Scanning {len(json_files)} docling files for table provenance")

    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            source = data.get("filename", json_file.stem + ".pdf")
            source_stem = source.replace(".pdf", "")
            docling_doc = data.get("docling_document", {})

            if not docling_doc:
                continue

            doc_tables = docling_doc.get("tables", {})
            if isinstance(doc_tables, dict):
                _process_tables_dict(doc_tables, source_stem, table_to_page)
            elif isinstance(doc_tables, list):
                _process_tables_list(doc_tables, source_stem, table_to_page)

        except Exception as e:
            logger.warning(f"Error processing {json_file.name}: {e}")

    logger.info(f"Built mapping for {len(table_to_page)} tables")
    return table_to_page


def migrate_table_summaries(
    tables_file: Path,
    docling_dir: Path,
    output_file: Path | None = None,
) -> dict[str, Any]:
    """
    Add page numbers to existing table summaries.

    Args:
        tables_file: Path to tables_multivector_*.json
        docling_dir: Path to docling extractions directory
        output_file: Output file (defaults to overwriting tables_file)

    Returns:
        Migration report dict.
    """
    # Build table -> page mapping
    table_to_page = build_table_to_page_mapping(docling_dir)

    # Load existing table summaries
    with open(tables_file, encoding="utf-8") as f:
        data = json.load(f)

    children = data.get("children", [])
    logger.info(f"Processing {len(children)} table summaries from {tables_file.name}")

    stats: dict[str, Any] = {
        "total": len(children),
        "migrated": 0,
        "already_has_page": 0,
        "no_mapping": 0,
        "errors": [],
    }

    for child in children:
        # doc_id format: "LA-octobre2025-table0"
        doc_id = child.get("doc_id", "")

        if child.get("page") and child["page"] > 0:
            stats["already_has_page"] += 1
            continue

        if doc_id in table_to_page:
            child["page"] = table_to_page[doc_id]
            stats["migrated"] += 1
        else:
            # Try alternative ID formats
            # Some summaries might have "-summary" suffix in doc_id
            base_id = doc_id.replace("-summary", "")
            if base_id in table_to_page:
                child["page"] = table_to_page[base_id]
                stats["migrated"] += 1
            else:
                stats["no_mapping"] += 1
                stats["errors"].append(f"No page mapping for {doc_id}")

    # Update data
    data["children"] = children

    # Save result
    output_path = output_file or tables_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Report
    logger.info(
        f"Migration complete: {stats['migrated']} migrated, {stats['already_has_page']} already had page"
    )
    if stats["no_mapping"] > 0:
        logger.warning(f"  {stats['no_mapping']} without mapping (need re-extraction)")

    return stats


def validate_table_summaries(tables_file: Path) -> dict[str, Any]:
    """
    Validate that all table summaries have page >= 1.

    Args:
        tables_file: Path to tables_multivector_*.json

    Returns:
        Validation report dict.
    """
    with open(tables_file, encoding="utf-8") as f:
        data = json.load(f)

    children = data.get("children", [])

    with_page = sum(1 for c in children if c.get("page") and c["page"] > 0)
    null_page = sum(1 for c in children if c.get("page") is None)
    zero_page = sum(1 for c in children if c.get("page") == 0)

    pct = 100 * with_page / max(1, len(children))

    report = {
        "file": str(tables_file),
        "total": len(children),
        "with_page": with_page,
        "null_page": null_page,
        "zero_page": zero_page,
        "coverage_pct": pct,
        "valid": with_page == len(children),
    }

    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate table summaries to add page numbers (ISO 42001)"
    )
    parser.add_argument(
        "--tables",
        "-t",
        type=Path,
        required=True,
        help="Path to tables_multivector_*.json",
    )
    parser.add_argument(
        "--docling",
        "-d",
        type=Path,
        required=True,
        help="Path to docling extractions directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file (default: overwrite input)",
    )
    parser.add_argument(
        "--validate", "-v", action="store_true", help="Only validate, don't migrate"
    )

    args = parser.parse_args()

    if args.validate:
        report = validate_table_summaries(args.tables)
        print(f"\nValidation: {report['file']}")
        print(f"  Total: {report['total']}")
        print(f"  With page: {report['with_page']} ({report['coverage_pct']:.1f}%)")
        print(f"  Null page: {report['null_page']}")
        print(f"  Zero page: {report['zero_page']}")
        print(f"  Valid: {'PASS' if report['valid'] else 'FAIL'}")
    else:
        stats = migrate_table_summaries(args.tables, args.docling, args.output)
        print(f"\nMigration: {args.tables.name}")
        print(f"  Total: {stats['total']}")
        print(f"  Migrated: {stats['migrated']}")
        print(f"  Already had page: {stats['already_has_page']}")
        print(f"  No mapping: {stats['no_mapping']}")


if __name__ == "__main__":
    main()
